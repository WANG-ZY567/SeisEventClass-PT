#!/usr/bin/env python3
"""
从 waveform/station-level 的 EVT6 `test_results_*.csv` 聚合到 event-level。

输入 CSV（xapp 跑分类时，ResultSaver 在 `--save-test-probs true` 下会写出）至少需要：
  - event_uid
  - tgt_evt6, pred_evt6
  - （可选）prob_evt6_0..prob_evt6_5，用于 probability averaging / confidence weighted

输出：
  - 事件级混淆矩阵 CSV（各方法一份）
  - 事件级 report markdown（各方法一份）
  - event-level 明细 CSV（各方法一份）
"""

import argparse
import os
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd


ID2NAME: Dict[int, str] = {0: "eq", 1: "ep", 2: "co", 3: "sp", 4: "se", 5: "ot"}
NUM_CLASSES = 6


def _safe_div(a: float, b: float) -> float:
    return float(a / b) if b != 0 else 0.0


def _compute_acc_cm(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, np.ndarray]:
    cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    for t, p in zip(y_true.tolist(), y_pred.tolist()):
        if 0 <= t < NUM_CLASSES and 0 <= p < NUM_CLASSES:
            cm[int(t), int(p)] += 1
    total = int(cm.sum())
    correct = int(np.trace(cm))
    acc = _safe_div(correct, total)
    return acc, cm


def _per_class_f1(cm: np.ndarray) -> Tuple[List[dict], float]:
    per_class: List[dict] = []
    f1s: List[float] = []
    for i in range(NUM_CLASSES):
        tp = int(cm[i, i])
        fp = int(cm[:, i].sum() - tp)
        fn = int(cm[i, :].sum() - tp)
        prec = _safe_div(tp, tp + fp)
        rec = _safe_div(tp, tp + fn)
        f1 = _safe_div(2 * prec * rec, prec + rec) if (prec + rec) > 0 else 0.0
        support = int(cm[i, :].sum())
        per_class.append(
            {
                "class_id": i,
                "class_name": ID2NAME.get(i, str(i)),
                "precision": prec,
                "recall": rec,
                "f1": f1,
                "support": support,
            }
        )
        f1s.append(f1)
    macro_f1 = float(np.mean(f1s)) if f1s else 0.0
    return per_class, macro_f1


def _bucket_by_event_size(n: int) -> str:
    if n == 1:
        return "1"
    if 2 <= n <= 3:
        return "2-3"
    if 4 <= n <= 5:
        return "4-5"
    return "6+"


def _method_name(method: str) -> str:
    if method == "hard":
        return "Hard voting (mode pred_evt6)"
    if method == "prob_avg":
        return "Probability averaging (mean prob)"
    if method == "conf_weighted":
        return "Confidence-weighted (weight=max prob)"
    return method


def _aggregate_one_event(
    df_evt: pd.DataFrame,
    method: str,
    prob_cols: Optional[List[str]],
) -> Tuple[int, int, int]:
    """
    Returns:
      y_true_event, y_pred_event, num_samples
    """
    tgt_vals = df_evt["tgt_evt6"].astype(int).to_numpy()
    pred_vals = df_evt["pred_evt6"].astype(int).to_numpy()
    num_samples = int(df_evt.shape[0])

    # Ground truth: ideally each event is consistent. If not, take mode and report at higher level.
    uniq_tgt, counts = np.unique(tgt_vals, return_counts=True)
    y_true_event = int(uniq_tgt[int(np.argmax(counts))]) if len(uniq_tgt) else -1

    if method == "hard":
        uniq_pred, pred_counts = np.unique(pred_vals, return_counts=True)
        y_pred_event = int(uniq_pred[int(np.argmax(pred_counts))]) if len(uniq_pred) else -1
        return y_true_event, y_pred_event, num_samples

    if method in ("prob_avg", "conf_weighted"):
        if not prob_cols:
            raise ValueError(f"method={method} requires prob columns prob_evt6_0..5, but they are missing")
        prob_mat = df_evt[prob_cols].to_numpy(dtype=np.float64)  # [K, C]
        # numerical safety
        prob_mat = np.nan_to_num(prob_mat, nan=0.0, posinf=0.0, neginf=0.0)
        if method == "prob_avg":
            prob_event = prob_mat.mean(axis=0)
        else:
            weights = prob_mat.max(axis=1)  # [K]
            wsum = float(weights.sum())
            if wsum <= 0:
                prob_event = prob_mat.mean(axis=0)
            else:
                prob_event = (prob_mat * weights[:, None]).sum(axis=0) / wsum
        y_pred_event = int(np.argmax(prob_event))
        return y_true_event, y_pred_event, num_samples

    raise ValueError(f"Unknown aggregation method: {method}")


def _cm_to_markdown(cm: np.ndarray, names: List[str]) -> str:
    header = "| true\\pred | " + " | ".join(names) + " |\n"
    sep = "|---" + "|---" * (len(names) + 1) + "|\n"
    rows = []
    for i, n in enumerate(names):
        rows.append("| " + n + " | " + " | ".join(str(int(x)) for x in cm[i, :].tolist()) + " |")
    return header + sep + "\n".join(rows) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_csv", type=str, required=True, help="test_results_*.csv path")
    ap.add_argument("--out_dir", type=str, default="", help="输出目录（默认基于 results_csv 所在目录）")
    ap.add_argument(
        "--methods",
        type=str,
        default="hard,prob_avg,conf_weighted",
        help="聚合方法列表：hard,prob_avg,conf_weighted（逗号分隔）",
    )
    args = ap.parse_args()

    csv_path = os.path.abspath(args.results_csv)
    if not os.path.exists(csv_path):
        raise FileNotFoundError(csv_path)

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    if not methods:
        raise ValueError("methods 不能为空")

    out_dir = os.path.abspath(args.out_dir) if args.out_dir else os.path.dirname(csv_path)
    os.makedirs(out_dir, exist_ok=True)

    df = pd.read_csv(csv_path, low_memory=False)
    required_cols = ["event_uid", "tgt_evt6", "pred_evt6"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise KeyError(f"CSV 缺少必要列：{missing}，实际列：{list(df.columns)}")

    prob_cols = [f"prob_evt6_{i}" for i in range(NUM_CLASSES)]
    has_prob = all(c in df.columns for c in prob_cols)

    y_true_evt_by_method: Dict[str, np.ndarray] = {}
    y_pred_evt_by_method: Dict[str, np.ndarray] = {}
    event_rows_by_method: Dict[str, pd.DataFrame] = {}

    event_uids = df["event_uid"].astype(str)
    # 预先计算 event 内 tgt 是否一致，用于报告
    tgt_consistent_flags = {}
    for evt_uid, g in df.groupby(event_uids):
        tgt_vals = g["tgt_evt6"].astype(int).to_numpy()
        tgt_consistent_flags[evt_uid] = int(len(np.unique(tgt_vals)) == 1)
    num_inconsistent = int(sum(1 for v in tgt_consistent_flags.values() if v == 0))

    # Event-level 聚合
    for method in methods:
        method_prob_cols = prob_cols if (method != "hard" and has_prob) else None
        rows = []
        for evt_uid, g in df.groupby(event_uids):
            y_true_evt, y_pred_evt, num_samples = _aggregate_one_event(
                df_evt=g,
                method=method,
                prob_cols=method_prob_cols,
            )
            rows.append(
                {
                    "event_uid": evt_uid,
                    "tgt_evt6": y_true_evt,
                    "pred_evt6": y_pred_evt,
                    "num_samples": int(num_samples),
                    "bucket": _bucket_by_event_size(int(num_samples)),
                }
            )

        event_df = pd.DataFrame(rows)
        # 保存明细
        method_tag = method
        detail_csv = os.path.join(out_dir, f"event_level_details_evt6_{method_tag}.csv")
        event_df.to_csv(detail_csv, index=False)

        # 计算指标
        y_true_evt = event_df["tgt_evt6"].astype(int).to_numpy()
        y_pred_evt = event_df["pred_evt6"].astype(int).to_numpy()
        acc, cm = _compute_acc_cm(y_true_evt, y_pred_evt)
        per_class, macro_f1 = _per_class_f1(cm)

        y_true_evt_by_method[method] = y_true_evt
        y_pred_evt_by_method[method] = y_pred_evt
        event_rows_by_method[method] = event_df

        cm_csv = os.path.join(out_dir, f"confusion_matrix_evt6_event_level_{method_tag}.csv")
        pd.DataFrame(
            cm,
            index=[f"true_{ID2NAME[i]}" for i in range(NUM_CLASSES)],
            columns=[f"pred_{ID2NAME[i]}" for i in range(NUM_CLASSES)],
        ).to_csv(cm_csv)

        # 分桶分析
        bucket_metrics = []
        for bucket_name in ["1", "2-3", "4-5", "6+"]:
            sub = event_df.loc[event_df["bucket"] == bucket_name]
            if sub.empty:
                continue
            y_t = sub["tgt_evt6"].astype(int).to_numpy()
            y_p = sub["pred_evt6"].astype(int).to_numpy()
            acc_b, cm_b = _compute_acc_cm(y_t, y_p)
            _, macro_f1_b = _per_class_f1(cm_b)
            bucket_metrics.append((bucket_name, int(sub.shape[0]), acc_b, macro_f1_b))

        # 输出 report
        names = [ID2NAME[i] for i in range(NUM_CLASSES)]
        md_path = os.path.join(out_dir, f"EVT6_EVENT_LEVEL_REPORT_{method_tag}.md")
        with open(md_path, "w", encoding="utf-8") as f:
            f.write("# EVT6 事件级聚合评估报告 (Event-level)\n\n")
            f.write(f"- 输入 CSV：`{csv_path}`\n")
            f.write(f"- 聚合方法：`{method}` ({_method_name(method)})\n")
            f.write(f"- 事件数：{int(event_df.shape[0])}\n")
            f.write(f"- event 内 tgt_evt6 不一致事件数（取 mode）：{num_inconsistent}\n")
            f.write(f"- 事件级 Accuracy：{acc:.4f}\n")
            f.write(f"- 事件级 Macro-F1：{macro_f1:.4f}\n")
            f.write(f"- 混淆矩阵 CSV：`{cm_csv}`\n")
            f.write("\n---\n\n")

            f.write("## 1) 各类别指标（按事件统计）\n\n")
            f.write("| class | support | precision | recall | f1 |\n")
            f.write("|---|---:|---:|---:|---:|\n")
            for r in per_class:
                f.write(
                    f"| {r['class_name']} | {int(r['support'])} | {r['precision']:.4f} | {r['recall']:.4f} | {r['f1']:.4f} |\n"
                )

            f.write("\n---\n\n")
            f.write("## 2) Confusion Matrix（true x pred）\n\n")
            f.write(_cm_to_markdown(cm, names))

            f.write("\n---\n\n")
            f.write("## 3) 按 event 内样本数分桶\n\n")
            f.write("| bucket | #events | accuracy | macro-f1 |\n")
            f.write("|---|---:|---:|---:|\n")
            for bucket_name, n_evt, acc_b, macro_f1_b in bucket_metrics:
                f.write(f"| {bucket_name} | {n_evt} | {acc_b:.4f} | {macro_f1_b:.4f} |\n")

    print(f"[OK] Aggregated event-level results are saved under: {out_dir}")


if __name__ == "__main__":
    main()

