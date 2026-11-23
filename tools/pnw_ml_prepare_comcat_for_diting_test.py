#!/usr/bin/env python3
"""
data ComCat metadata `dataset-name diting2_evt6` data `meta_evt6_test.csv`. 

data(data DiTing2Evt6 data): 
  - key, part, _npy_path, _evt6

data(data DiTing EVT6 data head data): 
  - ComCat earthquake -> _evt6=0  (data DiTing evt6: eq)
  - ComCat explosion  -> _evt6=1  (data DiTing evt6: ep)

_npy_path is stored relative to data_dir, for example waves/<safe_key>.npy
data pnw_ml_export_hdf5_to_npy.py data npy. 

data(data, data): 
  comcat_event_id, station_code, trace_name
"""
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


def safe_name(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", s)[:200]


def first_present(row: dict[str, str], names: tuple[str, ...]) -> str:
    """Return the first non-empty metadata field among documented aliases."""
    for name in names:
        value = (row.get(name) or "").strip()
        if value:
            return value
    return ""


def comcat_label(row: dict[str, str]) -> tuple[int, str] | None:
    """Use PNSN's event-type label when available, otherwise source_type.

    PNW-ML documents a small number of inconsistent legacy ``source_type``
    entries and asks users to prioritize the PNSN label in those cases.
    """
    pnsn = first_present(row, ("source_type_pnsn_label", "event_type_pnsn_label")).lower()
    source = first_present(row, ("source_type", "event_type")).lower()
    value = pnsn if pnsn in {"eq", "ep"} else source
    if value in {"eq", "earthquake"}:
        return 0, "earthquake"
    if value in {"ep", "explosion"}:
        return 1, "explosion"
    return None


def positive_float(value: str) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number >= 0 else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--comcat_csv", type=Path, default=Path("/path/to/comcat_metadata.csv"))
    ap.add_argument("--out_csv", type=Path, required=True)
    ap.add_argument(
        "--waves_subdir",
        default="waves",
        help="data external data_dir data",
    )
    args = ap.parse_args()

    rows_out = []
    skipped_missing_phase_metadata = 0
    with args.comcat_csv.open(newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            label = comcat_label(row)
            if label is None:
                continue
            cls_id, label_name = label
            tid = (row.get("trace_name") or "").strip()
            eid = (row.get("event_id") or "").strip()
            st_code = (row.get("station_code") or "").strip()
            if not tid:
                continue
            p_arrival = first_present(
                row,
                ("trace_P_arrival_sample", "trace_p_arrival_sample", "p_arrival_sample"),
            )
            sample_rate = first_present(row, ("trace_sampling_rate_hz", "sampling_rate_hz"))
            if positive_float(p_arrival) is None or positive_float(sample_rate) in (None, 0.0):
                skipped_missing_phase_metadata += 1
                continue
            key = f"{eid}|{st_code}|{tid}"
            npy_rel = f"{args.waves_subdir}/{safe_name(key)}.npy"
            rows_out.append(
                {
                    "key": key,
                    "part": "test",
                    "_npy_path": npy_rel,
                    "_evt6": cls_id,
                    "comcat_event_id": eid,
                    "station_code": st_code,
                    "trace_name": tid,
                    # PNW-ML metadata: retain these fields so the waveform
                    # exporter can build a phase-aligned external window.
                    "pnw_p_arrival_sample": p_arrival,
                    "pnw_sampling_rate_hz": sample_rate,
                    "pnw_label_source": "pnsn" if (row.get("source_type_pnsn_label") or "").strip().lower() in {"eq", "ep"} else "source_type",
                }
            )

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "key",
        "part",
        "_npy_path",
        "_evt6",
        "comcat_event_id",
        "station_code",
        "trace_name",
        "pnw_p_arrival_sample",
        "pnw_sampling_rate_hz",
        "pnw_label_source",
    ]
    with args.out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows_out)
    print(
        "wrote", len(rows_out), "phase-valid eq/ep rows ->", args.out_csv,
        "; skipped missing/invalid P or sampling-rate metadata:", skipped_missing_phase_metadata,
    )


if __name__ == "__main__":
    main()

