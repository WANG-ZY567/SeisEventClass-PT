#!/usr/bin/env python3
"""
ComCat: data region-disjoint data, data event data trace data manifest CSV. 

value: earthquake=0, explosion=1; data source_type. 

data(data): 
  - data source_longitude_deg data, data west / east data. 
  - source_region=west: lon < lon_threshold
  - source_region=east: lon >= lon_threshold

value: 
  - source value: train + val(event data)
  - target value: test(zero-shot data); few-shot data target train pool data

value: reports/pnw_ml/splits/comcat_binary_lon-121.5/
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path


def load_unique_events_comcat(path: Path) -> dict[str, dict]:
    """Open-source note: implementation detail."""
    out: dict[str, dict] = {}
    with path.open(newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            eid = (row.get("event_id") or "").strip()
            if not eid or eid in out:
                continue
            try:
                lat = float(row["source_latitude_deg"])
                lon = float(row["source_longitude_deg"])
            except (KeyError, ValueError, TypeError):
                continue
            st = (row.get("source_type") or "").strip().lower()
            out[eid] = {
                "source_type": st,
                "source_latitude_deg": lat,
                "source_longitude_deg": lon,
                "source_type_pnsn_label": (row.get("source_type_pnsn_label") or "").strip(),
            }
    return out


def assign_region(lon: float, lat: float, lon_th: float, mode: str) -> str:
    if mode == "lon_split":
        return "west" if lon < lon_th else "east"
    if mode == "nw_quadrant":
        # N: lat>=47, W: lon < -121
        ns = "N" if lat >= 47.0 else "S"
        we = "W" if lon < -121.0 else "E"
        return ns + we
    raise ValueError(mode)


def split_events(
    event_ids: list[str],
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> tuple[set[str], set[str], set[str]]:
    rng = random.Random(seed)
    ids = list(event_ids)
    rng.shuffle(ids)
    n = len(ids)
    n_test = int(round(n * test_ratio))
    n_val = int(round(n * val_ratio))
    n_train = n - n_test - n_val
    if n_train <= 0:
        raise ValueError("train data, data val/test data")
    test_set = set(ids[:n_test])
    val_set = set(ids[n_test : n_test + n_val])
    train_set = set(ids[n_test + n_val :])
    return train_set, val_set, test_set


def write_event_csv(
    path: Path,
    rows: list[dict],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def build_trace_manifest(
    comcat_csv: Path,
    event_split: dict[str, str],
    allowed_types: set[str],
    out_csv: Path,
) -> int:
    """event_split: event_id -> train|val|test|unused"""
    n = 0
    rows = []
    with comcat_csv.open(newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            eid = (row.get("event_id") or "").strip()
            st = (row.get("source_type") or "").strip().lower()
            if st not in allowed_types:
                continue
            sp = event_split.get(eid)
            if sp is None or sp == "unused":
                continue
            label = 0 if st == "earthquake" else 1
            rows.append(
                {
                    "event_id": eid,
                    "split": sp,
                    "label": label,
                    "source_type": st,
                    "trace_name": row.get("trace_name", ""),
                    "station_code": row.get("station_code", ""),
                    "trace_sampling_rate_hz": row.get("trace_sampling_rate_hz", ""),
                }
            )
            n += 1
    # Open-source note: implementation detail.
    write_event_csv(out_csv, rows)
    return n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--comcat_csv",
        type=Path,
        default=Path("/path/to/comcat_metadata.csv"),
    )
    ap.add_argument("--lon_threshold", type=float, default=-121.5)
    ap.add_argument(
        "--region_mode",
        choices=["lon_split", "nw_quadrant"],
        default="lon_split",
        help="lon_split: west/east data; nw_quadrant: NW/NE/SW/SE data(data source/target data)",
    )
    ap.add_argument(
        "--source_region",
        default="west",
        help="lon_split data west data east; nw_quadrant data NW/NE/SW/SE",
    )
    ap.add_argument(
        "--target_region",
        default="east",
        help="lon_split data east data west",
    )
    ap.add_argument("--val_ratio", type=float, default=0.1)
    ap.add_argument("--test_ratio", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--out_dir",
        type=Path,
        default=None,
        help="data reports/pnw_ml/splits/comcat_binary_<mode>_lon<threshold>/",
    )
    args = ap.parse_args()

    events = load_unique_events_comcat(args.comcat_csv)
    binary = {
        e: v
        for e, v in events.items()
        if v["source_type"] in ("earthquake", "explosion")
    }

    region_of: dict[str, str] = {}
    for eid, v in binary.items():
        region_of[eid] = assign_region(
            v["source_longitude_deg"],
            v["source_latitude_deg"],
            args.lon_threshold,
            args.region_mode,
        )

    if args.region_mode == "lon_split":
        src_name, tgt_name = args.source_region, args.target_region
        if {src_name, tgt_name} != {"west", "east"}:
            raise SystemExit("lon_split data source_region/target_region data west data east data")
        source_events = [e for e, r in region_of.items() if r == src_name]
        target_events = [e for e, r in region_of.items() if r == tgt_name]
    else:
        # Open-source note: implementation detail.
        src_name, tgt_name = args.source_region, args.target_region
        source_events = [e for e, r in region_of.items() if r == src_name]
        target_events = [e for e, r in region_of.items() if r == tgt_name]

    out_dir = args.out_dir
    if out_dir is None:
        tag = f"comcat_binary_{args.region_mode}_lon{args.lon_threshold}"
        out_dir = Path("reports/pnw_ml/splits") / tag
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    tr_train, tr_val, tr_test = split_events(
        source_events, args.val_ratio, args.test_ratio, args.seed
    )
    # Open-source note: implementation detail.
    ta_train, ta_val, ta_test = split_events(
        target_events, args.val_ratio, args.test_ratio, args.seed + 1
    )

    event_split: dict[str, str] = {}
    for e in binary:
        event_split[e] = "unused"

    for e in tr_train:
        event_split[e] = "source_train"
    for e in tr_val:
        event_split[e] = "source_val"
    for e in tr_test:
        event_split[e] = "source_test"

    for e in ta_train:
        event_split[e] = "target_train_pool"
    for e in ta_val:
        event_split[e] = "target_val"
    for e in ta_test:
        event_split[e] = "target_test"

    summary = {
        "comcat_csv": str(args.comcat_csv),
        "lon_threshold": args.lon_threshold,
        "region_mode": args.region_mode,
        "source_region": src_name,
        "target_region": tgt_name,
        "counts_events": {
            "source_total": len(source_events),
            "target_total": len(target_events),
            "source_train": len(tr_train),
            "source_val": len(tr_val),
            "source_test": len(tr_test),
            "target_train_pool": len(ta_train),
            "target_val": len(ta_val),
            "target_test": len(ta_test),
        },
        "seed": args.seed,
    }
    (out_dir / "split_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n"
    )

    # Open-source note: implementation detail.
    def rows_for(split_name: str, eids: set[str]) -> list[dict]:
        out = []
        for eid in sorted(eids):
            v = binary[eid]
            out.append(
                {
                    "event_id": eid,
                    "split": split_name,
                    "label": 0 if v["source_type"] == "earthquake" else 1,
                    "source_type": v["source_type"],
                    "source_latitude_deg": v["source_latitude_deg"],
                    "source_longitude_deg": v["source_longitude_deg"],
                    "region": region_of[eid],
                }
            )
        return out

    write_event_csv(
        out_dir / "events_source_train.csv",
        rows_for("source_train", tr_train),
    )
    write_event_csv(
        out_dir / "events_source_val.csv",
        rows_for("source_val", tr_val),
    )
    write_event_csv(
        out_dir / "events_source_test.csv",
        rows_for("source_test", tr_test),
    )
    write_event_csv(
        out_dir / "events_target_train_pool.csv",
        rows_for("target_train_pool", ta_train),
    )
    write_event_csv(
        out_dir / "events_target_val.csv",
        rows_for("target_val", ta_val),
    )
    write_event_csv(
        out_dir / "events_target_test.csv",
        rows_for("target_test", ta_test),
    )

    # Open-source note: implementation detail.
    # Open-source note: implementation detail.

    # Open-source note: implementation detail.
    all_rows = []
    for eid, v in sorted(binary.items()):
        all_rows.append(
            {
                "event_id": eid,
                "split_role": event_split.get(eid, "unused"),
                "label": 0 if v["source_type"] == "earthquake" else 1,
                "source_type": v["source_type"],
                "source_latitude_deg": v["source_latitude_deg"],
                "source_longitude_deg": v["source_longitude_deg"],
                "region": region_of[eid],
            }
        )
    write_event_csv(out_dir / "events_all_labeled.csv", all_rows)

    trace_out = out_dir / "traces_manifest_all_splits.csv"
    n_tr = build_trace_manifest(
        args.comcat_csv,
        {e: event_split[e] for e in binary if event_split[e] != "unused"},
        {"earthquake", "explosion"},
        trace_out,
    )

    print("value:", out_dir)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("trace data(data):", n_tr)
    print("value:", trace_out)


if __name__ == "__main__":
    main()
