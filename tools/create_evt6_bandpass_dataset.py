#!/usr/bin/env python3
"""Create a filtered EVT6 dataset without changing its split or metadata.

The input directory must already be a validated EVT6 dataset.  Every waveform
is transformed independently using exactly the same processing chain:

1. per-component linear detrending;
2. a Tukey taper with 5 percent of the record at each endpoint; and
3. a fourth-order, zero-phase Butterworth band-pass filter.

The CSV manifests are copied byte-for-byte.  Thus the source and output differ
only in waveform preprocessing, allowing validation-set comparisons without
changing the cohort, labels, event split, or input window.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
from scipy import signal


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sampling-rate-hz", type=float, default=50.0)
    parser.add_argument("--low-hz", type=float, default=1.0)
    parser.add_argument("--high-hz", type=float, default=20.0)
    parser.add_argument("--order", type=int, default=4)
    parser.add_argument(
        "--taper-edge-fraction",
        type=float,
        default=0.05,
        help="Fraction tapered at each endpoint; default: 0.05 (5 percent).",
    )
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--progress-every", type=int, default=500)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def manifest_wave_paths(source_dir: Path) -> List[Path]:
    required = [source_dir / f"meta_evt6_{split}.csv" for split in ("train", "val", "test")]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing split manifest(s): " + ", ".join(missing))

    relative_paths: set[Path] = set()
    for manifest in required:
        with manifest.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                raw_path = row.get("_npy_path", "")
                if not raw_path:
                    raise ValueError(f"Missing _npy_path in {manifest}")
                relative = Path(raw_path)
                if relative.is_absolute() or ".." in relative.parts:
                    raise ValueError(f"Unsafe waveform path in {manifest}: {raw_path}")
                relative_paths.add(relative)

    missing_waves = [str(path) for path in relative_paths if not (source_dir / path).is_file()]
    if missing_waves:
        raise FileNotFoundError("Missing source waveform(s), e.g. " + missing_waves[0])
    return sorted(relative_paths, key=str)


def atomic_save(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, array)
    os.replace(temporary, path)


def process_one(task: Tuple[str, str, float, float, float, int, float]) -> str:
    source_name, output_name, fs, low_hz, high_hz, order, edge_fraction = task
    source = Path(source_name)
    output = Path(output_name)
    waveform = np.load(source, allow_pickle=False)
    if waveform.ndim != 2 or waveform.shape[0] != 3:
        raise ValueError(f"Expected [3, samples], got {waveform.shape}: {source}")
    if not np.isfinite(waveform).all():
        raise ValueError(f"Non-finite waveform: {source}")

    samples = waveform.shape[1]
    if samples < 64:
        raise ValueError(f"Waveform too short for zero-phase filtering ({samples}): {source}")

    detrended = signal.detrend(waveform.astype(np.float64, copy=False), axis=1, type="linear")
    # scipy's alpha is the total tapered fraction, so alpha=2*edge_fraction
    # gives exactly the requested taper length at each record endpoint.
    taper = signal.windows.tukey(samples, alpha=2.0 * edge_fraction)
    sos = signal.butter(order, [low_hz, high_hz], btype="bandpass", fs=fs, output="sos")
    filtered = signal.sosfiltfilt(sos, detrended * taper[None, :], axis=1)
    if not np.isfinite(filtered).all():
        raise ValueError(f"Filtering created non-finite values: {source}")

    atomic_save(output, filtered.astype(np.float32, copy=False))
    return str(source)


def copy_manifests(source_dir: Path, output_dir: Path) -> Dict[str, str]:
    copied: Dict[str, str] = {}
    for source in sorted(source_dir.glob("meta_evt6*.csv")):
        target = output_dir / source.name
        shutil.copy2(source, target)
        copied[source.name] = sha256(target)
    return copied


def main() -> None:
    args = parse_args()
    source_dir = args.source_dir.resolve()
    output_dir = args.output_dir.resolve()
    if not source_dir.is_dir():
        raise NotADirectoryError(source_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to write into non-empty directory: {output_dir}")
    if args.workers <= 0:
        raise ValueError("--workers must be positive")
    if args.order <= 0:
        raise ValueError("--order must be positive")
    if not (0 < args.low_hz < args.high_hz < args.sampling_rate_hz / 2.0):
        raise ValueError("Require 0 < low_hz < high_hz < Nyquist frequency")
    if not (0 < args.taper_edge_fraction < 0.5):
        raise ValueError("--taper-edge-fraction must be in (0, 0.5)")

    paths = manifest_wave_paths(source_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    manifest_hashes = copy_manifests(source_dir, output_dir)
    source_protocol = source_dir / "protocol_report.json"
    if source_protocol.is_file():
        shutil.copy2(source_protocol, output_dir / "source_protocol_report.json")

    tasks = [
        (
            str(source_dir / relative),
            str(output_dir / relative),
            float(args.sampling_rate_hz),
            float(args.low_hz),
            float(args.high_hz),
            int(args.order),
            float(args.taper_edge_fraction),
        )
        for relative in paths
    ]

    completed = 0
    try:
        if args.workers == 1:
            iterator: Iterable[str] = map(process_one, tasks)
            for _ in iterator:
                completed += 1
                if completed % args.progress_every == 0 or completed == len(tasks):
                    print(f"[FILTER] {completed:,}/{len(tasks):,}", flush=True)
        else:
            with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
                for _ in executor.map(process_one, tasks, chunksize=16):
                    completed += 1
                    if completed % args.progress_every == 0 or completed == len(tasks):
                        print(f"[FILTER] {completed:,}/{len(tasks):,}", flush=True)
    except Exception:
        print(f"[FAILED] Filtered {completed:,}/{len(tasks):,} waveforms; partial output kept at {output_dir}")
        raise

    report: Dict[str, Any] = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_dataset": str(source_dir),
        "source_protocol_report": str(source_protocol) if source_protocol.is_file() else None,
        "output_dataset": str(output_dir),
        "num_waveforms": len(tasks),
        "processing": {
            "sampling_rate_hz": args.sampling_rate_hz,
            "detrend": "per-component linear detrend",
            "taper": {
                "type": "Tukey",
                "edge_fraction_each_side": args.taper_edge_fraction,
            },
            "filter": {
                "type": "Butterworth band-pass, zero-phase sosfiltfilt",
                "order": args.order,
                "low_hz": args.low_hz,
                "high_hz": args.high_hz,
            },
            "saved_dtype": "float32",
            "runtime_normalization": "per-component demean and standard deviation (norm-mode std)",
        },
        "manifest_sha256": manifest_hashes,
    }
    with (output_dir / "bandpass_preprocessing_report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
