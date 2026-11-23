#!/usr/bin/env python3
"""Create a directly-readable, non-degenerate DiTing 2.0 natural-event JSON.

The natural-event HDF5 on the project server has a damaged root symbol table:
enumerating ``h5.keys()`` fails, whereas many known dataset identifiers can
still be read.  This utility deliberately iterates the authoritative source
JSON and accesses each waveform as ``h5[record_id]``.  It never enumerates the
HDF5 root.

The output JSON preserves the metadata of records that pass all checks:

* the HDF5 object can be opened and read directly;
* it is a three-component waveform (optionally with the expected length);
* all values are finite; and
* it is not an all-zero waveform (unless ``--keep-all-zero`` is specified).

Records rejected by a check are written to a CSV sidecar, and a JSON report
records every decision count.  This is an upstream data-quality audit only;
P/S validity, event capping, and event-level splitting remain later steps.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Tuple

import h5py
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-json", type=Path, required=True)
    parser.add_argument("--hdf5", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument(
        "--rejected-csv",
        type=Path,
        default=None,
        help="Default: <output-json stem>_rejected.csv",
    )
    parser.add_argument(
        "--report-json",
        type=Path,
        default=None,
        help="Default: <output-json stem>_report.json",
    )
    parser.add_argument("--expected-samples", type=int, default=10000)
    parser.add_argument("--expected-channels", type=int, default=3)
    parser.add_argument(
        "--allow-variable-length",
        action="store_true",
        help=(
            "Keep any readable 2-D three-component waveform regardless of "
            "its sample length.  Use this for the upstream readable-source "
            "JSON; length eligibility belongs to the later P-aligned crop."
        ),
    )
    parser.add_argument(
        "--keep-all-zero",
        action="store_true",
        help="Keep readable all-zero arrays; disabled by default.",
    )
    parser.add_argument("--progress-every", type=int, default=10000)
    parser.add_argument(
        "--stop-after",
        type=int,
        default=None,
        help="Audit only the first N JSON rows (for a smoke test).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacement of an existing output JSON and sidecars.",
    )
    return parser.parse_args()


def default_sidecar(path: Path, suffix: str) -> Path:
    return path.with_name(f"{path.stem}_{suffix}")


def normalise_shape(
    raw: np.ndarray,
    expected_samples: int,
    expected_channels: int,
    allow_variable_length: bool,
) -> Tuple[int, int]:
    if raw.ndim != 2:
        raise ValueError(f"shape_mismatch: expected 2-D waveform, got {tuple(raw.shape)}")
    if raw.shape[1] == expected_channels:
        num_samples = int(raw.shape[0])
    elif raw.shape[0] == expected_channels:
        num_samples = int(raw.shape[1])
    else:
        raise ValueError(
            "shape_mismatch: expected one dimension of size "
            f"{expected_channels}, got {tuple(raw.shape)}"
        )
    if allow_variable_length or num_samples == expected_samples:
        return num_samples, expected_channels
    raise ValueError(
        "shape_mismatch: expected "
        f"({expected_samples}, {expected_channels}) or "
        f"({expected_channels}, {expected_samples}), got {tuple(raw.shape)}"
    )


def classify_exception(exc: Exception) -> str:
    text = str(exc)
    if text.startswith("shape_mismatch:"):
        return "shape_mismatch"
    return "read_error"


def atomic_writer(output_path: Path) -> Tuple[Path, Path]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    partial_path = output_path.with_name(f".{output_path.name}.partial")
    if partial_path.exists():
        partial_path.unlink()
    return partial_path, output_path


def main() -> None:
    args = parse_args()
    source_json = args.source_json.resolve()
    hdf5_path = args.hdf5.resolve()
    output_json = args.output_json.resolve()
    rejected_csv = (args.rejected_csv or default_sidecar(output_json, "rejected.csv")).resolve()
    report_json = (args.report_json or default_sidecar(output_json, "report.json")).resolve()

    for path in (source_json, hdf5_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.expected_samples <= 0 or args.expected_channels <= 0:
        raise ValueError("--expected-samples and --expected-channels must be positive")

    output_paths = (output_json, rejected_csv, report_json)
    existing = [str(path) for path in output_paths if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError("Output already exists; use --overwrite: " + ", ".join(existing))

    with source_json.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    if not isinstance(metadata, dict):
        raise TypeError("Source JSON must contain a top-level object keyed by record_id")

    partial_json, final_json = atomic_writer(output_json)
    rejected_csv.parent.mkdir(parents=True, exist_ok=True)
    report_json.parent.mkdir(parents=True, exist_ok=True)

    counts: Counter[str] = Counter()
    kept_by_class: Counter[str] = Counter()
    rejected_by_class: Counter[str] = Counter()
    kept_shape_counts: Counter[str] = Counter()
    started_at = datetime.now(timezone.utc).isoformat()
    first_json_item = True

    try:
        with h5py.File(hdf5_path, "r") as h5, \
             partial_json.open("w", encoding="utf-8") as output_handle, \
             rejected_csv.open("w", newline="", encoding="utf-8") as rejected_handle:
            writer = csv.DictWriter(
                rejected_handle,
                fieldnames=["record_id", "evtype", "reason", "detail"],
            )
            writer.writeheader()
            output_handle.write("{\n")

            for index, (record_id_raw, record_metadata) in enumerate(metadata.items(), start=1):
                if args.stop_after is not None and index > args.stop_after:
                    break
                record_id = str(record_id_raw)
                evtype = (
                    str(record_metadata.get("evtype", ""))
                    if isinstance(record_metadata, dict)
                    else ""
                )
                counts["attempted"] += 1

                try:
                    node = h5[record_id]
                    if not isinstance(node, h5py.Dataset):
                        raise TypeError(f"HDF5 object is {type(node).__name__}, not Dataset")
                    raw = node[()]
                    num_samples, num_channels = normalise_shape(
                        raw,
                        args.expected_samples,
                        args.expected_channels,
                        args.allow_variable_length,
                    )
                    if not bool(np.isfinite(raw).all()):
                        counts["non_finite"] += 1
                        rejected_by_class[evtype] += 1
                        writer.writerow({
                            "record_id": record_id,
                            "evtype": evtype,
                            "reason": "non_finite",
                            "detail": "waveform contains NaN or Inf",
                        })
                        continue
                    if not args.keep_all_zero and not bool(np.any(raw != 0)):
                        counts["all_zero"] += 1
                        rejected_by_class[evtype] += 1
                        writer.writerow({
                            "record_id": record_id,
                            "evtype": evtype,
                            "reason": "all_zero",
                            "detail": "all waveform samples equal zero",
                        })
                        continue
                except Exception as exc:  # HDF5 must be isolated per record.
                    reason = classify_exception(exc)
                    counts[reason] += 1
                    rejected_by_class[evtype] += 1
                    writer.writerow({
                        "record_id": record_id,
                        "evtype": evtype,
                        "reason": reason,
                        "detail": f"{type(exc).__name__}: {exc}",
                    })
                    continue

                if not first_json_item:
                    output_handle.write(",\n")
                json.dump(record_id, output_handle, ensure_ascii=False)
                output_handle.write(": ")
                json.dump(record_metadata, output_handle, ensure_ascii=False, separators=(",", ":"))
                first_json_item = False
                counts["kept"] += 1
                kept_by_class[evtype] += 1
                kept_shape_counts[f"({num_samples}, {num_channels})"] += 1

                if args.progress_every and index % args.progress_every == 0:
                    print(
                        f"processed={index:,} kept={counts['kept']:,} "
                        f"rejected={index - counts['kept']:,}",
                        flush=True,
                    )

            output_handle.write("\n}\n")

        os.replace(partial_json, final_json)
    except Exception:
        if partial_json.exists():
            partial_json.unlink()
        raise

    processed = int(counts["attempted"])
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_json": str(source_json),
        "hdf5": str(hdf5_path),
        "output_json": str(final_json),
        "rejected_csv": str(rejected_csv),
        "selection": {
            "expected_samples": args.expected_samples,
            "expected_channels": args.expected_channels,
            "allow_variable_length": args.allow_variable_length,
            "drop_all_zero": not args.keep_all_zero,
            "stop_after": args.stop_after,
        },
        "started_at_utc": started_at,
        "source_records": len(metadata),
        "processed_records": processed,
        "counts": {
            "kept": int(counts["kept"]),
            "rejected": processed - int(counts["kept"]),
            "read_error": int(counts["read_error"]),
            "shape_mismatch": int(counts["shape_mismatch"]),
            "non_finite": int(counts["non_finite"]),
            "all_zero": int(counts["all_zero"]),
        },
        "kept_by_evtype": dict(sorted(kept_by_class.items())),
        "rejected_by_evtype": dict(sorted(rejected_by_class.items())),
        "kept_shape_counts": dict(sorted(kept_shape_counts.items())),
    }
    with report_json.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
