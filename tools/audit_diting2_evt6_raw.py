#!/usr/bin/env python3
"""Audit raw DiTing 2.0 metadata and waveform files for EVT6 rebuilding.

The script deliberately uses JSON record identifiers as the HDF5 index.  It
does not enumerate the HDF5 root, so it can still diagnose files whose root
symbol table is damaged while many individual datasets remain readable.

The default waveform audit is a small, class-stratified sample.  It never
exports waveforms and therefore requires little additional disk space.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Tuple

import h5py
import numpy as np


EVT6_ALIASES = {
    "eq": "eq",
    "ep": "ep",
    "ss": "ss",
    "co": "ss",
    "sp": "sp",
    "se": "se",
    "ot": "ot",
}

NUMERIC_FIELDS = (
    "Pg",
    "Sg",
    "Pg_dist",
    "Sg_dist",
    "evmag",
    "mag",
    "sampling_rate",
    "sample_rate",
    "sampling_rate_hz",
)

PRESENCE_FIELDS = (
    "evtype",
    "Pg",
    "Sg",
    "Pg_dist",
    "Sg_dist",
    "evmag",
    "mag",
    "Pg_polarity",
)


def _normalise_evtype(value: Any) -> str:
    if value is None:
        return "unknown"
    label = str(value).strip().lower()
    if not label:
        return "unknown"
    return EVT6_ALIASES.get(label, f"unknown:{label}")


def _split_record_id(record_id: str) -> Tuple[str, str]:
    value = str(record_id)
    if "_" not in value:
        return value, ""
    return tuple(value.split("_", 1))  # type: ignore[return-value]


def _to_float(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


@dataclass
class RunningNumber:
    count: int = 0
    minimum: float = math.inf
    maximum: float = -math.inf
    total: float = 0.0

    def add(self, value: Any) -> None:
        number = _to_float(value)
        if number is None:
            return
        self.count += 1
        self.minimum = min(self.minimum, number)
        self.maximum = max(self.maximum, number)
        self.total += number

    def as_dict(self) -> Dict[str, Any]:
        if not self.count:
            return {"count": 0, "min": None, "max": None, "mean": None}
        return {
            "count": self.count,
            "min": self.minimum,
            "max": self.maximum,
            "mean": self.total / self.count,
        }


def _file_info(path: Optional[Path]) -> Optional[Dict[str, Any]]:
    if path is None:
        return None
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size_bytes": int(stat.st_size),
        "modified_time": datetime.fromtimestamp(
            stat.st_mtime, tz=timezone.utc
        ).isoformat(),
    }


def _load_metadata(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object at the top level: {path}")
    return value


def _metadata_report(metadata: Mapping[str, Any]) -> Dict[str, Any]:
    events: set[str] = set()
    stations: set[str] = set()
    labels: Counter[str] = Counter()
    records_by_event: Counter[str] = Counter()
    presence: Counter[str] = Counter()
    numeric = {name: RunningNumber() for name in NUMERIC_FIELDS}
    non_object_rows = 0

    for record_id, raw_meta in metadata.items():
        event_id, station_id = _split_record_id(str(record_id))
        events.add(event_id)
        if station_id:
            stations.add(station_id)
        records_by_event[event_id] += 1

        if not isinstance(raw_meta, dict):
            non_object_rows += 1
            labels["unknown"] += 1
            continue

        labels[_normalise_evtype(raw_meta.get("evtype"))] += 1
        for field in PRESENCE_FIELDS:
            if raw_meta.get(field) not in (None, ""):
                presence[field] += 1
        for field, accumulator in numeric.items():
            accumulator.add(raw_meta.get(field))

    event_sizes = np.asarray(list(records_by_event.values()), dtype=np.int64)
    event_size_summary: Dict[str, Any]
    if event_sizes.size:
        event_size_summary = {
            "min": int(event_sizes.min()),
            "max": int(event_sizes.max()),
            "mean": float(event_sizes.mean()),
            "median": float(np.median(event_sizes)),
            "p95": float(np.percentile(event_sizes, 95)),
        }
    else:
        event_size_summary = {
            "min": None,
            "max": None,
            "mean": None,
            "median": None,
            "p95": None,
        }

    num_records = len(metadata)
    return {
        "num_records": num_records,
        "num_events_parsed_from_key": len(events),
        "num_stations_parsed_from_key": len(stations),
        "non_object_metadata_rows": non_object_rows,
        "evt6_record_counts": dict(sorted(labels.items())),
        "field_presence": {
            field: {
                "count": int(presence[field]),
                "fraction": (float(presence[field]) / num_records if num_records else 0.0),
            }
            for field in PRESENCE_FIELDS
        },
        "numeric_fields": {name: value.as_dict() for name, value in numeric.items()},
        "records_per_event": event_size_summary,
    }


def _stratified_reservoir(
    metadata: Mapping[str, Any], per_class: int, seed: int
) -> List[Tuple[str, Any]]:
    """Select up to ``per_class`` records per normalized EVT6 label."""

    rng = random.Random(seed)
    seen: Counter[str] = Counter()
    reservoirs: Dict[str, List[Tuple[str, Any]]] = defaultdict(list)

    for record_id, raw_meta in metadata.items():
        label = _normalise_evtype(
            raw_meta.get("evtype") if isinstance(raw_meta, dict) else None
        )
        seen[label] += 1
        bucket = reservoirs[label]
        item = (str(record_id), raw_meta)
        if len(bucket) < per_class:
            bucket.append(item)
        else:
            replacement = rng.randrange(seen[label])
            if replacement < per_class:
                bucket[replacement] = item

    selected: List[Tuple[str, Any]] = []
    for label in sorted(reservoirs):
        selected.extend(reservoirs[label])
    rng.shuffle(selected)
    return selected


def _records_to_scan(
    metadata: Mapping[str, Any], scan_mode: str, sample_per_class: int, seed: int
) -> Iterable[Tuple[str, Any]]:
    if scan_mode == "none":
        return ()
    if scan_mode == "all":
        return ((str(key), value) for key, value in metadata.items())
    return _stratified_reservoir(metadata, sample_per_class, seed)


def _read_first_dataset(node: Any) -> np.ndarray:
    if isinstance(node, h5py.Dataset):
        return np.asarray(node[()])
    if isinstance(node, h5py.Group):
        for child_name in node:
            child = node[child_name]
            if isinstance(child, h5py.Dataset):
                return np.asarray(child[()])
    raise TypeError(f"HDF5 object is not a readable dataset: {type(node).__name__}")


def _as_three_component(waveform: np.ndarray) -> Tuple[np.ndarray, int]:
    if waveform.ndim != 2:
        raise ValueError(f"expected 2 dimensions, got shape={waveform.shape}")
    if waveform.shape[0] == 3:
        return waveform, int(waveform.shape[1])
    if waveform.shape[1] == 3:
        return waveform.T, int(waveform.shape[0])
    raise ValueError(f"expected one dimension of size 3, got shape={waveform.shape}")


def _estimate_snr_db(waveform: np.ndarray, pg: Any, window: int) -> Optional[float]:
    pg_value = _to_float(pg)
    if pg_value is None:
        return None
    pick = int(round(pg_value))
    length = waveform.shape[1]
    use_window = min(int(window), pick, length - pick)
    if use_window <= 0:
        return None
    noise = waveform[:, pick - use_window : pick].astype(np.float64, copy=False)
    signal = waveform[:, pick : pick + use_window].astype(np.float64, copy=False)
    noise_power = float(np.mean(np.square(noise)))
    signal_power = float(np.mean(np.square(signal)))
    return float(10.0 * np.log10((signal_power + 1e-12) / (noise_power + 1e-12)))


def _audit_waveforms(
    source: str,
    h5_path: Path,
    records: Iterable[Tuple[str, Any]],
    snr_window: int,
    detail_writer: csv.DictWriter,
) -> Dict[str, Any]:
    status_counts: Counter[str] = Counter()
    shape_counts: Counter[str] = Counter()
    dtype_counts: Counter[str] = Counter()
    length_stats = RunningNumber()
    snr_values: List[float] = []
    attempted = 0

    with h5py.File(h5_path, "r") as handle:
        for record_id, raw_meta in records:
            attempted += 1
            metadata = raw_meta if isinstance(raw_meta, dict) else {}
            event_id, station_id = _split_record_id(record_id)
            label = _normalise_evtype(metadata.get("evtype"))
            row: Dict[str, Any] = {
                "source": source,
                "record_id": record_id,
                "event_id": event_id,
                "station_id": station_id,
                "evt6": label,
                "status": "",
                "error": "",
                "shape": "",
                "dtype": "",
                "num_samples": "",
                "all_finite": "",
                "all_zero": "",
                "constant_components": "",
                "Pg": metadata.get("Pg", ""),
                "Sg": metadata.get("Sg", ""),
                "snr_db": "",
            }
            try:
                raw = _read_first_dataset(handle[record_id])
                waveform, num_samples = _as_three_component(raw)
                finite = bool(np.isfinite(waveform).all())
                all_zero = bool(np.all(waveform == 0))
                constant_components = int(
                    np.sum(np.ptp(waveform.astype(np.float64, copy=False), axis=1) == 0)
                )
                snr_db = _estimate_snr_db(waveform, metadata.get("Pg"), snr_window)

                row.update(
                    {
                        "status": "ok" if finite else "non_finite",
                        "shape": "x".join(str(v) for v in raw.shape),
                        "dtype": str(raw.dtype),
                        "num_samples": num_samples,
                        "all_finite": int(finite),
                        "all_zero": int(all_zero),
                        "constant_components": constant_components,
                        "snr_db": "" if snr_db is None else snr_db,
                    }
                )
                status_counts[row["status"]] += 1
                shape_counts[row["shape"]] += 1
                dtype_counts[row["dtype"]] += 1
                length_stats.add(num_samples)
                if snr_db is not None and math.isfinite(snr_db):
                    snr_values.append(snr_db)
            except Exception as exc:  # record every unreadable JSON-indexed key
                row["status"] = "failed"
                row["error"] = f"{type(exc).__name__}: {exc}"
                status_counts["failed"] += 1
            detail_writer.writerow(row)

    snr_array = np.asarray(snr_values, dtype=np.float64)
    snr_summary = {
        "count": int(snr_array.size),
        "min": float(snr_array.min()) if snr_array.size else None,
        "max": float(snr_array.max()) if snr_array.size else None,
        "mean": float(snr_array.mean()) if snr_array.size else None,
        "median": float(np.median(snr_array)) if snr_array.size else None,
    }
    return {
        "attempted": attempted,
        "status_counts": dict(sorted(status_counts.items())),
        "shape_counts": dict(sorted(shape_counts.items())),
        "dtype_counts": dict(sorted(dtype_counts.items())),
        "num_samples": length_stats.as_dict(),
        "snr_db_around_Pg": snr_summary,
    }


def _parse_source(
    source: str,
    json_path: Path,
    h5_path: Optional[Path],
    args: argparse.Namespace,
    detail_writer: csv.DictWriter,
) -> Dict[str, Any]:
    print(f"[{source}] Loading metadata: {json_path}", flush=True)
    metadata = _load_metadata(json_path)
    result: Dict[str, Any] = {
        "json_file": _file_info(json_path),
        "hdf5_file": _file_info(h5_path),
        "metadata": _metadata_report(metadata),
        "waveforms": None,
    }
    print(
        f"[{source}] records={len(metadata):,}, "
        f"events={result['metadata']['num_events_parsed_from_key']:,}, "
        f"labels={result['metadata']['evt6_record_counts']}",
        flush=True,
    )

    if args.waveform_scan != "none":
        if h5_path is None:
            raise ValueError(f"--{source}_h5 is required when waveform scanning is enabled")
        records = _records_to_scan(
            metadata,
            scan_mode=args.waveform_scan,
            sample_per_class=args.sample_per_class,
            seed=args.seed + (0 if source == "natural" else 1),
        )
        print(
            f"[{source}] Auditing waveforms (mode={args.waveform_scan}) from: {h5_path}",
            flush=True,
        )
        result["waveforms"] = _audit_waveforms(
            source=source,
            h5_path=h5_path,
            records=records,
            snr_window=args.snr_window,
            detail_writer=detail_writer,
        )
        print(f"[{source}] waveform audit: {result['waveforms']}", flush=True)
    return result


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit raw DiTing 2.0 JSON/HDF5 files before rebuilding EVT6"
    )
    parser.add_argument("--natural_json", type=Path, default=None)
    parser.add_argument("--natural_h5", type=Path, default=None)
    parser.add_argument("--non_json", type=Path, default=None)
    parser.add_argument("--non_h5", type=Path, default=None)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument(
        "--waveform_scan",
        choices=("none", "sample", "all"),
        default="sample",
        help="none=metadata only, sample=class-stratified sample, all=read every JSON key",
    )
    parser.add_argument(
        "--sample_per_class",
        type=int,
        default=100,
        help="records per normalized class for sample mode (default: 100)",
    )
    parser.add_argument("--seed", type=int, default=20261105)
    parser.add_argument(
        "--snr_window",
        type=int,
        default=500,
        help="samples before/after Pg used for a diagnostic SNR estimate",
    )
    return parser


def _write_markdown_summary(report: Mapping[str, Any], path: Path) -> None:
    lines = [
        "# DiTing 2.0 EVT6 raw data audit\n\n",
        f"Generated at UTC: `{report['generated_at_utc']}`\n\n",
        "## Audit configuration\n\n",
        f"- Waveform scan: `{report['configuration']['waveform_scan']}`\n",
        f"- Sample records per class: `{report['configuration']['sample_per_class']}`\n",
        f"- Random seed: `{report['configuration']['seed']}`\n",
        f"- Diagnostic SNR window: `{report['configuration']['snr_window']}` samples\n\n",
    ]

    for source_name, source in report["sources"].items():
        metadata = source["metadata"]
        lines.extend(
            [
                f"## {source_name.replace('_', ' ').title()}\n\n",
                f"- Metadata records: **{metadata['num_records']:,}**\n",
                f"- Parsed events: **{metadata['num_events_parsed_from_key']:,}**\n",
                f"- Parsed stations: **{metadata['num_stations_parsed_from_key']:,}**\n",
                f"- EVT6 record counts: `{metadata['evt6_record_counts']}`\n",
                f"- Pg availability: **{metadata['field_presence']['Pg']['fraction']:.2%}**\n",
                f"- Sg availability: **{metadata['field_presence']['Sg']['fraction']:.2%}**\n\n",
            ]
        )
        waveforms = source.get("waveforms")
        if waveforms is not None:
            statuses = waveforms["status_counts"]
            attempted = int(waveforms["attempted"])
            failed = int(statuses.get("failed", 0))
            lines.extend(
                [
                    "### Waveform audit\n\n",
                    f"- Attempted: **{attempted:,}**\n",
                    f"- Status counts: `{statuses}`\n",
                    f"- Read failure fraction: **{(failed / attempted if attempted else 0.0):.2%}**\n",
                    f"- Shapes: `{waveforms['shape_counts']}`\n",
                    f"- Data types: `{waveforms['dtype_counts']}`\n",
                    f"- Sample lengths: `{waveforms['num_samples']}`\n",
                    f"- Diagnostic SNR: `{waveforms['snr_db_around_Pg']}`\n\n",
                ]
            )

    lines.extend(
        [
            "## Interpretation boundary\n\n",
            "This audit describes file integrity and metadata availability only. "
            "It does not define the final training cohort, apply quality filters, "
            "or assign unknown labels to the EVT6 `ot` class.\n",
        ]
    )
    path.write_text("".join(lines), encoding="utf-8")


def main() -> None:
    args = _build_parser().parse_args()
    if args.natural_json is None and args.non_json is None:
        raise ValueError("Provide at least one of --natural_json or --non_json")
    if args.sample_per_class <= 0:
        raise ValueError("--sample_per_class must be positive")
    if args.snr_window <= 0:
        raise ValueError("--snr_window must be positive")

    for name in ("natural_json", "natural_h5", "non_json", "non_h5"):
        path = getattr(args, name)
        if path is not None and not path.is_file():
            raise FileNotFoundError(f"{name}: {path}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    details_path = args.out_dir / "waveform_audit_details.csv"
    report_path = args.out_dir / "raw_data_audit_report.json"
    summary_path = args.out_dir / "raw_data_audit_summary.md"
    detail_fields = [
        "source",
        "record_id",
        "event_id",
        "station_id",
        "evt6",
        "status",
        "error",
        "shape",
        "dtype",
        "num_samples",
        "all_finite",
        "all_zero",
        "constant_components",
        "Pg",
        "Sg",
        "snr_db",
    ]

    report: Dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "configuration": {
            "waveform_scan": args.waveform_scan,
            "sample_per_class": args.sample_per_class,
            "seed": args.seed,
            "snr_window": args.snr_window,
            "notes": [
                "Event and station identifiers are parsed from <event>_<station> record IDs.",
                "Unknown evtype values remain unknown; they are not silently mapped to ot.",
                "Diagnostic SNR is not a dataset label and is not used to filter records.",
            ],
        },
        "sources": {},
    }

    with details_path.open("w", newline="", encoding="utf-8-sig") as detail_file:
        writer = csv.DictWriter(detail_file, fieldnames=detail_fields)
        writer.writeheader()
        if args.natural_json is not None:
            report["sources"]["natural"] = _parse_source(
                "natural", args.natural_json, args.natural_h5, args, writer
            )
        if args.non_json is not None:
            report["sources"]["non_natural"] = _parse_source(
                "non", args.non_json, args.non_h5, args, writer
            )

    with report_path.open("w", encoding="utf-8") as report_file:
        json.dump(report, report_file, ensure_ascii=False, indent=2)
        report_file.write("\n")
    _write_markdown_summary(report, summary_path)

    print(f"[OK] JSON report: {report_path}")
    print(f"[OK] Markdown summary: {summary_path}")
    print(f"[OK] Waveform details: {details_path}")


if __name__ == "__main__":
    main()
