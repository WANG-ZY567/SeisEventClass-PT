#!/usr/bin/env python3
"""
ComCat：按震源经纬度做 region-disjoint 划分，生成 event 级与 trace 级 manifest CSV。

默认二分类：earthquake=0, explosion=1；仅保留这两种 source_type。

区域规则（可调）：
  - 用 source_longitude_deg 与阈值比较，将事件分为 west / east 两区。
  - source_region=west: lon < lon_threshold
  - source_region=east: lon >= lon_threshold

划分：
  - source 区：train + val（event 不重叠）
  - target 区：test（zero-shot 评测）；few-shot 时从 target train pool 再抽比例

输出目录示例：reports/pnw_ml/splits/comcat_binary_lon-121.5/
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path


def load_unique_events_comcat(path: Path) -> dict[str, dict]:
    """每个 event_id 保留第一次出现行的震源信息与类型。"""
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
        raise ValueError("train 为空，请减小 val/test 比例或检查数据量")
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
    # 保留原 CSV 中其它列可选扩展：此处最小集
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
        help="lon_split: west/east 由经度阈值划分；nw_quadrant: NW/NE/SW/SE 四象限（需另选 source/target 象限）",
    )
    ap.add_argument(
        "--source_region",
        default="west",
        help="lon_split 下为 west 或 east；nw_quadrant 下为 NW/NE/SW/SE",
    )
    ap.add_argument(
        "--target_region",
        default="east",
        help="lon_split 下为 east 或 west",
    )
    ap.add_argument("--val_ratio", type=float, default=0.1)
    ap.add_argument("--test_ratio", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--out_dir",
        type=Path,
        default=None,
        help="默认 reports/pnw_ml/splits/comcat_binary_<mode>_lon<threshold>/",
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
            raise SystemExit("lon_split 下 source_region/target_region 应为 west 与 east 的组合")
        source_events = [e for e, r in region_of.items() if r == src_name]
        target_events = [e for e, r in region_of.items() if r == tgt_name]
    else:
        # 四象限：source 与 target 为两个不同象限
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
    # target：整体作为 target pool；test 用于 zero-shot，few-shot 从 target 再划分
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

    # 保存 event 列表
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

    # trace 级 manifest（含 split 标签，供 DataLoader 使用）
    # zero-shot：仅用 source_train+val 训练，target_test 测；target_train_pool 可供 few-shot 抽样

    # 写 event_split 全表
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

    print("输出目录:", out_dir)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("trace 行数(过滤二分类后):", n_tr)
    print("已写:", trace_out)


if __name__ == "__main__":
    main()
