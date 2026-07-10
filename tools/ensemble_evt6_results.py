import argparse
import os
import subprocess
from typing import List

import numpy as np
import pandas as pd


def _parse_args():
    p = argparse.ArgumentParser(description="EVT6 probability ensemble (average probs across runs)")
    p.add_argument(
        "--results_csvs",
        type=str,
        nargs="+",
        required=True,
        help="List of test_results_*.csv paths. Each must contain prob_evt6_0..5 columns.",
    )
    p.add_argument(
        "--out_csv",
        type=str,
        required=True,
        help="Output ensembled results csv path.",
    )
    p.add_argument(
        "--out_md",
        type=str,
        default="",
        help="Optional: output report markdown path (will call tools/report_evt6_results.py).",
    )
    p.add_argument(
        "--id_cols",
        type=str,
        nargs="+",
        default=["key", "part"],
        help="Columns used to align rows across csvs (default: key part).",
    )
    p.add_argument(
        "--weights",
        type=float,
        nargs="*",
        default=[],
        help="Optional weights for each csv (same length as results_csvs). Default: uniform.",
    )
    return p.parse_args()


def main():
    args = _parse_args()
    paths: List[str] = list(args.results_csvs)
    if len(paths) < 2:
        raise ValueError("Need at least 2 csvs to ensemble")

    weights = list(args.weights) if args.weights else [1.0] * len(paths)
    if len(weights) != len(paths):
        raise ValueError(f"--weights length must match --results_csvs: {len(weights)} vs {len(paths)}")
    w = np.asarray(weights, dtype=np.float64)
    if np.all(w == 0):
        raise ValueError("All weights are zero")
    w = w / w.sum()

    dfs = []
    for pth in paths:
        df = pd.read_csv(pth)
        # find prob columns
        prob_cols = [f"prob_evt6_{i}" for i in range(6)]
        missing = [c for c in prob_cols if c not in df.columns]
        if missing:
            raise ValueError(
                f"Missing prob columns in {pth}: {missing}. "
                f"Re-run test with --save-test-probs true"
            )
        for c in args.id_cols:
            if c not in df.columns:
                raise ValueError(f"Missing id col {c} in {pth}")
        dfs.append(df)

    # Align by id cols
    base = dfs[0].copy()
    base_key = base[args.id_cols].astype(str).agg("||".join, axis=1)
    base["_ens_key"] = base_key
    base = base.set_index("_ens_key", drop=True)

    probs_acc = None
    for wi, df in zip(w, dfs):
        k = df[args.id_cols].astype(str).agg("||".join, axis=1)
        df = df.copy()
        df["_ens_key"] = k
        df = df.set_index("_ens_key", drop=True)
        df = df.loc[base.index]
        probs = df[[f"prob_evt6_{i}" for i in range(6)]].to_numpy(dtype=np.float64)
        if probs_acc is None:
            probs_acc = wi * probs
        else:
            probs_acc += wi * probs

    probs_acc = probs_acc.astype(np.float32)
    pred = probs_acc.argmax(axis=1).astype(np.int64)

    # write output
    out = base.reset_index(drop=True)
    out["pred_evt6"] = pred
    # keep tgt_evt6 from base if exists
    if "tgt_evt6" in out.columns:
        pass
    # overwrite prob columns with ensembled probs
    for i in range(6):
        out[f"prob_evt6_{i}"] = probs_acc[:, i]

    out_csv = args.out_csv
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    out.to_csv(out_csv, index=False)
    print(f"[OK] wrote ensembled csv: {out_csv}")

    if args.out_md:
        out_md = args.out_md
        os.makedirs(os.path.dirname(out_md), exist_ok=True)
        # call existing report script
        cmd = [
            "python",
            "tools/report_evt6_results.py",
            "--results_csv",
            out_csv,
            "--out_md",
            out_md,
        ]
        print("[RUN]", " ".join(cmd))
        subprocess.run(cmd, check=True)
        print(f"[OK] wrote report: {out_md}")


if __name__ == "__main__":
    main()


