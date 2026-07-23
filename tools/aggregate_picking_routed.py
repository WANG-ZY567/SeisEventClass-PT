#!/usr/bin/env python3
"""
Predicted-conditioned picking: data evt6 data, data trace data class-specific picker data, 
data P/S picking data(Precision/Recall/F1 + MAE data, data utils/metrics.py data). 

value: 
1) route_csv: data(waveform-level)test_results_*.csv, value: 
   - trace_uid(data key+part data)
   - pred_evt6(data prob_evt6_*)
2) per-class picker data test_results_*.csv(dpk data), value: 
   - trace_uid(data key+part)
   - tgt_ppk, pred_ppk
   - tgt_spk, pred_spk

data `trace_uid` data join(xapp meta data). 
"""

from __future__ import annotations

import argparse
import os
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd

ID2NAME = {0: "eq", 1: "ep", 2: "co", 3: "sp", 4: "se", 5: "ot"}
NUM_CLASSES = 6

def _is_nan(x: object) -> bool:
    # pandas/py float NaN handling (avoid isinstance checks everywhere)
    try:
        return bool(pd.isna(x))
    except Exception:
        return False


def _to_int_or_default(x: object, default: int = -1) -> int:
    if _is_nan(x):
        return int(default)
    try:
        return int(x)
    except Exception:
        return int(default)


def _safe_div(a: float, b: float) -> float:
    return float(a / b) if b != 0 else 0.0


def _compute_pick_metrics(
    tgt: np.ndarray, pred: np.ndarray, num_samples: int, sampling_rate: int, time_threshold_s: float
) -> Dict[str, float]:
    """
    data utils/metrics.py data ppk/spk value: 
    - pred/target data [0,num_samples) data"data"
    - tp: pred & target data, data |err| <= t_thres
    """
    t_thres = int(float(time_threshold_s) * float(sampling_rate))

    tgt = tgt.astype(np.int64)
    pred = pred.astype(np.int64)

    preds_bin = (pred >= 0) & (pred < int(num_samples))
    tgts_bin = (tgt >= 0) & (tgt < int(num_samples))
    ae = np.abs(tgt - pred)
    tp_bin = preds_bin & tgts_bin & (ae <= int(t_thres))

    tp = float(tp_bin.sum())
    predp = float(preds_bin.sum())
    possp = float(tgts_bin.sum())

    precision = _safe_div(tp, predp)
    recall = _safe_div(tp, possp)
    f1 = _safe_div(2 * precision * recall, precision + recall) if (precision + recall) > 0 else 0.0

    # Open-source note: implementation detail.
    mae = float(ae[tp_bin].mean()) if tp_bin.any() else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "mae": mae, "tp": tp, "predp": predp, "possp": possp}


def _load_results_csv(path: str) -> pd.DataFrame:
    p = os.path.abspath(path)
    if not os.path.exists(p):
        raise FileNotFoundError(p)
    df = pd.read_csv(p, low_memory=False)
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--route_csv", required=True, help="data test_results_*.csv(data trace_uid + pred_evt6)")
    ap.add_argument(
        "--picker_csv_by_class",
        required=True,
        help="data picker data test_results csv data, value: 0=/path/a.csv,1=/path/b.csv,...",
    )
    ap.add_argument("--out_md", required=True)
    ap.add_argument("--num_samples", type=int, default=8192)
    ap.add_argument("--sampling_rate", type=int, default=50)
    ap.add_argument("--time_threshold", type=float, default=0.1, help="seconds")
    args = ap.parse_args()

    route_df = _load_results_csv(args.route_csv)
    if "trace_uid" not in route_df.columns:
        raise KeyError("route_csv value: trace_uid")
    if "pred_evt6" not in route_df.columns:
        raise KeyError("route_csv value: pred_evt6")
    keep_cols = ["trace_uid", "pred_evt6"]
    # If available, use the true P/S targets from route_csv directly. This is important for predicted-routing:
    # even if a trace is misrouted, the global targets should still come from ground truth.
    if "p_pick" in route_df.columns:
        keep_cols.append("p_pick")
    if "s_pick" in route_df.columns:
        keep_cols.append("s_pick")
    route_df = route_df[keep_cols].copy()
    route_df["pred_evt6"] = route_df["pred_evt6"].astype(int)

    # parse mapping
    mapping: Dict[int, str] = {}
    for item in str(args.picker_csv_by_class).split(","):
        item = item.strip()
        if not item:
            continue
        k, v = item.split("=", 1)
        mapping[int(k.strip())] = os.path.abspath(v.strip())
    for cid in range(NUM_CLASSES):
        if cid not in mapping:
            raise ValueError(f"picker_csv_by_class data {cid} data csv data")

    # load picker outputs and build a union table of per-trace predictions per class
    picker_df_by_class: Dict[int, pd.DataFrame] = {}
    for cid, p in mapping.items():
        df = _load_results_csv(p)
        if "trace_uid" not in df.columns:
            raise KeyError(f"class {cid} picker csv value: trace_uid: {p}")
        need = ["tgt_ppk", "pred_ppk", "tgt_spk", "pred_spk"]
        miss = [c for c in need if c not in df.columns]
        if miss:
            raise KeyError(f"class {cid} picker csv value: {miss}: {p}")
        df = df[["trace_uid"] + need].copy()
        # ensure int
        for c in need:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(-1).astype(int)
        picker_df_by_class[cid] = df

    # join route with each class picker and then select routed rows
    merged = route_df.copy()
    # also keep one tgt from any picker to validate consistency
    for cid in range(NUM_CLASSES):
        dfc = picker_df_by_class[cid]
        merged = merged.merge(dfc, on="trace_uid", how="left", suffixes=("", f"_c{cid}"))
        merged = merged.rename(
            columns={
                "tgt_ppk": f"tgt_ppk_c{cid}",
                "pred_ppk": f"pred_ppk_c{cid}",
                "tgt_spk": f"tgt_spk_c{cid}",
                "pred_spk": f"pred_spk_c{cid}",
            }
        )

    # build routed predictions
    out_rows = []
    for _, row in merged.iterrows():
        cid = int(row["pred_evt6"])
        if cid < 0 or cid >= NUM_CLASSES:
            continue
        # Targets: prefer route_csv's p_pick/s_pick when present.
        # (These are already crop-aligned sample indices in xapp picking meta.)
        if "p_pick" in route_df.columns:
            tgt_ppk = _to_int_or_default(row.get("p_pick", -1), default=-1)
        else:
            tgt_ppk = _to_int_or_default(row.get(f"tgt_ppk_c{cid}", -1), default=-1)

        if "s_pick" in route_df.columns:
            tgt_spk = _to_int_or_default(row.get("s_pick", -1), default=-1)
        else:
            tgt_spk = _to_int_or_default(row.get(f"tgt_spk_c{cid}", -1), default=-1)

        # Predictions: from routed class picker output (if trace exists, otherwise NaN -> -1).
        pred_ppk = _to_int_or_default(row.get(f"pred_ppk_c{cid}", -1), default=-1)
        pred_spk = _to_int_or_default(row.get(f"pred_spk_c{cid}", -1), default=-1)

        out_rows.append(
            {
                "trace_uid": row["trace_uid"],
                "route_evt6": cid,
                "tgt_ppk": tgt_ppk,
                "pred_ppk": pred_ppk,
                "tgt_spk": tgt_spk,
                "pred_spk": pred_spk,
            }
        )
    routed_df = pd.DataFrame(out_rows)

    # metrics
    p_metrics = _compute_pick_metrics(
        tgt=routed_df["tgt_ppk"].to_numpy(),
        pred=routed_df["pred_ppk"].to_numpy(),
        num_samples=int(args.num_samples),
        sampling_rate=int(args.sampling_rate),
        time_threshold_s=float(args.time_threshold),
    )
    s_metrics = _compute_pick_metrics(
        tgt=routed_df["tgt_spk"].to_numpy(),
        pred=routed_df["pred_spk"].to_numpy(),
        num_samples=int(args.num_samples),
        sampling_rate=int(args.sampling_rate),
        time_threshold_s=float(args.time_threshold),
    )

    out_md = os.path.abspath(args.out_md)
    os.makedirs(os.path.dirname(out_md), exist_ok=True)
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("# Picking Routed Evaluation Report\n\n")
        f.write(f"- route_csv: `{os.path.abspath(args.route_csv)}`\n")
        f.write(f"- num_samples: {int(args.num_samples)}\n")
        f.write(f"- sampling_rate: {int(args.sampling_rate)}\n")
        f.write(f"- time_threshold(s): {float(args.time_threshold)}\n")
        f.write("\n---\n\n")
        f.write("## P-phase (ppk)\n\n")
        f.write(f"- precision: {p_metrics['precision']:.4f}\n")
        f.write(f"- recall: {p_metrics['recall']:.4f}\n")
        f.write(f"- f1: {p_metrics['f1']:.4f}\n")
        f.write(f"- mae(samples): {p_metrics['mae']:.4f}\n")
        f.write("\n## S-phase (spk)\n\n")
        f.write(f"- precision: {s_metrics['precision']:.4f}\n")
        f.write(f"- recall: {s_metrics['recall']:.4f}\n")
        f.write(f"- f1: {s_metrics['f1']:.4f}\n")
        f.write(f"- mae(samples): {s_metrics['mae']:.4f}\n")

    print("[OK] wrote:", out_md)


if __name__ == "__main__":
    main()

