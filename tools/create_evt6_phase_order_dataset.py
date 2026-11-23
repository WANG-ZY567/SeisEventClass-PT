#!/usr/bin/env python3
"""Copy an EVT6 dataset while excluding one verified invalid P/S record.

This is a derived dataset, not a reconstruction from the original HDF5 files.
The existing event assignments and waveform processing are preserved.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
from collections import Counter
from pathlib import Path


SPLITS = ("train", "val", "test")
LABELS = ("eq", "ep", "ss", "sp", "se", "ot")
REQUIRED = {"key", "part", "trace_uid", "event_id_raw", "_evt6", "_npy_path", "p_pick", "s_pick"}
EXPECTED_COUNTS = {"train": 16479, "val": 1911, "test": 1948}
EXPECTED_HASHES = {
    "train": "b05fef5e37de5b6025ac88352b73b87a0cdfe331d12c396ec75bb84c008c228d",
    "val": "d47ae5a8ae63b17b2caecd14bb0bec496f9bb11034fff745cb72b7514de73cc9",
    "test": "873f043007813e8f3ee58858f236ca53162e85bcec0cceaeedde5d3fd5dda15c",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        missing = REQUIRED - set(fields)
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        return fields, list(reader)


def identity(row: dict[str, str]) -> tuple[str, str]:
    key = (row["key"].strip(), row["part"].strip())
    if not all(key):
        raise ValueError(f"Empty record identity: {key}")
    return key


def wave_path(source: Path, row: dict[str, str]) -> tuple[Path, Path]:
    raw = row["_npy_path"].strip()
    if not raw:
        raise ValueError(f"Empty waveform path for {row['trace_uid']}")
    path = Path(raw)
    resolved = (path if path.is_absolute() else source / path).resolve()
    try:
        relative = resolved.relative_to(source)
    except ValueError as exc:
        raise ValueError(f"Waveform is outside source dataset: {resolved}") from exc
    if not resolved.is_file():
        raise FileNotFoundError(f"Missing waveform: {resolved}")
    return resolved, relative


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--exclude-trace", default="non_131917_42")
    args = parser.parse_args()

    source = args.source_dir.expanduser().resolve(strict=True)
    output = args.output_dir.expanduser().absolute()
    if not source.is_dir():
        raise ValueError(f"Not a source directory: {source}")
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output = output.parent.resolve() / output.name
    if source == output or source in output.parents or output in source.parents:
        raise ValueError("Source and output datasets must be separate directories")

    fields_by_name: dict[str, list[str]] = {}
    rows_by_name: dict[str, list[dict[str, str]]] = {}
    for split, expected in EXPECTED_HASHES.items():
        path = source / f"meta_evt6_{split}.csv"
        actual = sha256(path)
        if actual != expected:
            raise ValueError(f"Source {split} manifest hash differs: {actual}")
    for name in ("all", *SPLITS):
        file = "meta_evt6.csv" if name == "all" else f"meta_evt6_{name}.csv"
        fields_by_name[name], rows_by_name[name] = read_csv(source / file)

    source_counts = {name: len(rows_by_name[name]) for name in SPLITS}
    if source_counts != EXPECTED_COUNTS:
        raise ValueError(f"Source split counts differ from the audited dataset: {source_counts}")

    combined = Counter(identity(row) for row in rows_by_name["all"])
    split_records = Counter(identity(row) for split in SPLITS for row in rows_by_name[split])
    if combined != split_records or any(n != 1 for n in combined.values()):
        raise ValueError("Combined manifest and split manifests do not contain the same unique records")

    event_sets: dict[str, set[str]] = {}
    removed = []
    files: dict[Path, Path] = {}
    summaries = {}
    for split in SPLITS:
        kept = []
        event_sets[split] = set()
        for row in rows_by_name[split]:
            trace = row["trace_uid"].strip()
            try:
                p, s = float(row["p_pick"]), float(row["s_pick"])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid P/S pick in {trace}") from exc
            if not (math.isfinite(p) and math.isfinite(s) and 0 <= p < 8192 and 0 <= s < 8192):
                raise ValueError(f"Non-finite or out-of-window P/S pick in {trace}: P={p}, S={s}")
            if not p < s:
                if split != "train" or trace != args.exclude_trace:
                    raise ValueError(f"Unexpected P/S order: split={split}, trace={trace}, P={p}, S={s}")
                removed.append({"split": split, "trace_uid": trace, "class_id": row["_evt6"], "p_pick": p, "s_pick": s})
                continue
            if trace == args.exclude_trace:
                raise ValueError(f"Requested exclusion has valid P/S order: {trace}")
            event = row["event_id_raw"].strip()
            if not event or event.lower() == "nan":
                raise ValueError(f"Missing raw event ID: {trace}")
            event_sets[split].add(event)
            src, relative = wave_path(source, row)
            files[src] = relative
            row["_npy_path"] = relative.as_posix()
            kept.append(row)
        rows_by_name[split] = kept
        counts = Counter(int(float(row["_evt6"])) for row in kept)
        if not set(counts).issubset(range(6)):
            raise ValueError(f"Invalid class IDs in {split}: {counts}")
        summaries[split] = {
            "waveforms": len(kept),
            "raw_event_ids": len(event_sets[split]),
            "class_counts": {LABELS[i]: counts[i] for i in range(6)},
        }
    if len(removed) != 1:
        raise ValueError(f"Expected exactly one excluded record, found {len(removed)}")
    for i, left in enumerate(SPLITS):
        for right in SPLITS[i + 1 :]:
            if event_sets[left] & event_sets[right]:
                raise ValueError(f"Raw event identifiers overlap between {left} and {right}")

    combined_removed = [row for row in rows_by_name["all"] if row["trace_uid"].strip() == args.exclude_trace]
    if len(combined_removed) != 1:
        raise ValueError(f"Combined manifest has {len(combined_removed)} matching excluded records")
    removed_key = identity(combined_removed[0])
    rows_by_name["all"] = [row for row in rows_by_name["all"] if identity(row) != removed_key]
    for row in rows_by_name["all"]:
        _, relative = wave_path(source, row)
        row["_npy_path"] = relative.as_posix()

    total_bytes = sum(path.stat().st_size for path in files)
    free = shutil.disk_usage(output.parent).free
    if free < total_bytes + 256 * 1024 * 1024:
        raise RuntimeError(f"Insufficient disk space: need at least {total_bytes + 256 * 1024 * 1024} bytes, free {free}")

    output.mkdir(exist_ok=False)
    print(f"[COPY] {len(files):,} waveform files; source remains unchanged", flush=True)
    for index, (src, relative) in enumerate(sorted(files.items(), key=lambda item: str(item[1])), 1):
        destination = output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, destination)
        if index % 1000 == 0 or index == len(files):
            print(f"[COPY] {index:,}/{len(files):,}", flush=True)

    for name in ("all", *SPLITS):
        filename = "meta_evt6.csv" if name == "all" else f"meta_evt6_{name}.csv"
        with (output / filename).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields_by_name[name])
            writer.writeheader()
            writer.writerows(rows_by_name[name])

    provenance = output / "provenance"
    provenance.mkdir()
    for name in ("source_protocol_report.json", "bandpass_preprocessing_report.json"):
        if (source / name).is_file():
            shutil.copy2(source / name, provenance / name)

    names = ("meta_evt6.csv", *(f"meta_evt6_{split}.csv" for split in SPLITS))
    report = {
        "source_dataset": str(source),
        "output_dataset": str(output),
        "description": "Derived copy; same waveforms, preprocessing and event assignments, excluding one invalid P/S record.",
        "phase_order_requirement": "0 <= P < S < 8192",
        "excluded_records": removed,
        "split_summary": summaries,
        "source_manifest_sha256": {name: sha256(source / name) for name in names},
        "output_manifest_sha256": {name: sha256(output / name) for name in names},
        "source_reports": "provenance/ contains source reports, which describe the pre-exclusion dataset.",
    }
    (output / "phase_order_derivation_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
