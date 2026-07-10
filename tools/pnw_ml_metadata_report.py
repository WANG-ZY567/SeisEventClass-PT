#!/usr/bin/env python3
"""
PNW-ML：读取 ComCat / Exotic metadata CSV，打印字段可用性与类别/事件统计。
不访问 HDF5；不训练模型。用法见 reports/pnw_ml/PNW_ML_EXPERIMENT_PROTOCOL.md
"""
from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path


REQUIRED_COMCAT = [
    "event_id",
    "source_type",
    "source_type_pnsn_label",
    "source_latitude_deg",
    "source_longitude_deg",
    "station_code",
    "trace_sampling_rate_hz",
    "trace_P_arrival_sample",
    "trace_S_arrival_sample",
]


def summarize_comcat(path: Path) -> None:
    print(f"\n=== ComCat: {path} ===")
    na = {k: 0 for k in REQUIRED_COMCAT}
    st_tr = Counter()
    st_ev: dict[str, set[str]] = defaultdict(set)
    all_ev: set[str] = set()
    n = 0
    with path.open(newline="") as f:
        r = csv.DictReader(f)
        fields = r.fieldnames or []
        print("列数:", len(fields))
        for c in REQUIRED_COMCAT:
            print(f"  [{c}]", "存在" if c in fields else "缺失")
        for row in r:
            n += 1
            eid = (row.get("event_id") or "").strip()
            if eid:
                all_ev.add(eid)
            st = (row.get("source_type") or "").strip()
            st_tr[st] += 1
            if eid and st:
                st_ev[st.lower()].add(eid)
            for c in REQUIRED_COMCAT:
                v = row.get(c)
                if v is None or str(v).strip() == "" or str(v).lower() == "nan":
                    na[c] += 1
    print("行数(样本/波形级):", n)
    print("唯一 event_id 数:", len(all_ev))
    for c in REQUIRED_COMCAT:
        print(f"  {c} 非空约: {n - na[c]}")
    print("source_type 分布 (trace 数):")
    for k, v in st_tr.most_common():
        ne = len(st_ev.get(k.lower(), set()))
        print(f"  {k!r}: traces={v}, unique_events≈{ne}")
    eq = st_tr.get("earthquake", 0)
    ex = st_tr.get("explosion", 0)
    print(
        "二分类 trace: earthquake=", eq, " explosion=", ex,
        " 合计=", eq + ex,
    )
    print(
        "二分类 event: earthquake=",
        len(st_ev.get("earthquake", set())),
        " explosion=",
        len(st_ev.get("explosion", set())),
    )


def summarize_exotic(path: Path) -> None:
    print(f"\n=== Exotic: {path} ===")
    st_tr = Counter()
    st_ev: dict[str, set[str]] = defaultdict(set)
    all_ev: set[str] = set()
    n = 0
    has_station = False
    with path.open(newline="") as f:
        r = csv.DictReader(f)
        fields = list(r.fieldnames or [])
        print("列:", fields)
        print("source_type_pnsn_label:", "存在" if "source_type_pnsn_label" in fields else "不存在(本文件通常无此列)")
        has_station = "station_latitude_deg" in fields and "station_longitude_deg" in fields
        print("可用于区域/地理的坐标: 仅有 station_latitude_deg / station_longitude_deg (无震源经纬度)")
        for row in r:
            n += 1
            eid = (row.get("event_id") or "").strip()
            if eid:
                all_ev.add(eid)
            st = (row.get("source_type") or "").strip()
            st_tr[st] += 1
            if eid and st:
                st_ev[st].add(eid)
    print("行数(样本/波形级):", n)
    print("唯一 event_id 数:", len(all_ev))
    print("source_type 分布:")
    for k, v in st_tr.most_common():
        ne = len(st_ev[k])
        print(f"  {k!r}: traces={v}, events={ne}")
    print("说明: plane crash 若 events 极少，不宜作为独立主类，建议并入 other_exotic 或仅做 case study。")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--comcat_csv",
        type=Path,
        default=Path("/path/to/comcat_metadata.csv"),
    )
    ap.add_argument(
        "--exotic_csv",
        type=Path,
        default=Path("/path/to/exotic_metadata.csv"),
    )
    args = ap.parse_args()
    summarize_comcat(args.comcat_csv)
    summarize_exotic(args.exotic_csv)


if __name__ == "__main__":
    main()
