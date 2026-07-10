#!/usr/bin/env python3
"""
从 test_results_*.csv 生成 evt6 分类结果报告（论文口径）：
- Overall Accuracy（整体准确率）
- 每类别 Precision / Recall / F1
- Macro-F1
- 混淆矩阵（csv + Markdown 表格）

输入文件通常由 `training/validate.py` 在 `--save_test_results` 时生成：
  reports/<run>/test_results_diting2_evt6_test.csv
其中至少包含列：pred_evt6, tgt_evt6（整数类别 id）
"""

import argparse
import os
from datetime import datetime
from glob import glob
from typing import Dict, List, Tuple, Optional, Set

import numpy as np
import pandas as pd


# 与 evt6 定义保持一致：0 eq,1 ep,2 ss(论文里常写 co),3 sp,4 se,5 ot
ID2NAME: Dict[int, str] = {0: "eq", 1: "ep", 2: "co", 3: "sp", 4: "se", 5: "ot"}
NAME2ID: Dict[str, int] = {v: k for k, v in ID2NAME.items()}


def _safe_div(a: float, b: float) -> float:
    return float(a / b) if b != 0 else 0.0


def _find_default_results_csv(run_dir: str) -> str:
    cands = sorted(
        glob(os.path.join(run_dir, "test_results_*_test.csv")) + glob(os.path.join(run_dir, "test_results_*.csv")),
        key=lambda p: os.path.getmtime(p),
        reverse=True,
    )
    if not cands:
        raise FileNotFoundError(f"在 run_dir 未找到 test_results_*.csv：{run_dir}")
    return cands[0]


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> Tuple[float, np.ndarray, List[dict], float]:
    """
    Returns:
      acc, cm, per_class_rows, macro_f1
    """
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true.tolist(), y_pred.tolist()):
        if 0 <= t < num_classes and 0 <= p < num_classes:
            cm[t, p] += 1
    total = int(cm.sum())
    correct = int(np.trace(cm))
    acc = _safe_div(correct, total)

    per_class = []
    f1s = []
    for i in range(num_classes):
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
    macro_f1 = float(np.mean(f1s)) if len(f1s) else 0.0
    return acc, cm, per_class, macro_f1


def _cm_to_markdown(cm: np.ndarray, names: List[str]) -> str:
    header = "| true\\pred | " + " | ".join(names) + " |\n"
    sep = "|---" + "|---" * (len(names) + 1) + "|\n"
    rows = []
    for i, n in enumerate(names):
        rows.append("| " + n + " | " + " | ".join(str(int(x)) for x in cm[i].tolist()) + " |")
    return header + sep + "\n".join(rows) + "\n"


def _cm_to_markdown_rect(cm: np.ndarray, true_names: List[str], pred_names: List[str]) -> str:
    """
    支持非方阵 confusion matrix：
      - 行：true_names（长度应等于 cm.shape[0]）
      - 列：pred_names（长度应等于 cm.shape[1]）
    """
    if cm.shape[0] != len(true_names):
        raise ValueError(f"cm 行数与 true_names 不一致：cm={cm.shape}, true_names={len(true_names)}")
    if cm.shape[1] != len(pred_names):
        raise ValueError(f"cm 列数与 pred_names 不一致：cm={cm.shape}, pred_names={len(pred_names)}")
    header = "| true\\pred | " + " | ".join(pred_names) + " |\n"
    sep = "|---" + "|---" * (len(pred_names) + 1) + "|\n"
    rows = []
    for i, n in enumerate(true_names):
        rows.append("| " + n + " | " + " | ".join(str(int(x)) for x in cm[i].tolist()) + " |")
    return header + sep + "\n".join(rows) + "\n"


def _parse_class_list(s: str) -> List[int]:
    """
    支持输入：
      - 逗号分隔的 class name：eq,ep,co,sp,se,ot
      - 或数字 id：0,1,2
    """
    s = (s or "").strip()
    if not s:
        return []
    parts = [p.strip() for p in s.split(",") if p.strip()]
    out: List[int] = []
    for p in parts:
        if p.isdigit():
            out.append(int(p))
        else:
            if p not in NAME2ID:
                raise ValueError(f"未知类别名：{p}（支持：{sorted(NAME2ID.keys())} 或 0-5）")
            out.append(NAME2ID[p])
    # 去重并保持稳定顺序
    seen: Set[int] = set()
    uniq: List[int] = []
    for x in out:
        if x not in seen:
            seen.add(x)
            uniq.append(x)
    return uniq


def _compute_metrics_excluding_true(
    y_true: np.ndarray, y_pred: np.ndarray, num_classes: int, exclude_true_ids: List[int]
) -> Tuple[int, float, np.ndarray, List[dict], float, List[int]]:
    """
    在“剔除某些 true label”的子集上计算指标：
      - 过滤掉 y_true 属于 exclude_true_ids 的样本
      - confusion matrix 仍保留 pred 侧的全部 num_classes 列（方便看“被错分到 se”的情况）
      - per-class / macro-f1 只对保留的 true 类做平均（即 5-way macro-f1）

    Returns:
      kept_total, acc, cm_kept_rows (K x num_classes), per_class_rows(K), macro_f1, kept_true_ids
    """
    exclude_set = set(int(x) for x in exclude_true_ids)
    keep_mask = np.array([int(t) not in exclude_set for t in y_true.tolist()], dtype=bool)
    yt = y_true[keep_mask]
    yp = y_pred[keep_mask]
    kept_total = int(yt.shape[0])
    kept_true_ids = [i for i in range(num_classes) if i not in exclude_set]

    # accuracy 直接在过滤后的样本上算
    acc = float(np.mean((yt == yp).astype(np.float64))) if kept_total > 0 else 0.0

    # full cm on filtered, then keep only selected true rows
    cm_full = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(yt.tolist(), yp.tolist()):
        if 0 <= t < num_classes and 0 <= p < num_classes:
            cm_full[t, p] += 1
    cm_kept = cm_full[kept_true_ids, :]

    per_class: List[dict] = []
    f1s: List[float] = []
    for i in kept_true_ids:
        tp = int(cm_full[i, i])
        fp = int(cm_full[:, i].sum() - tp)  # 过滤后的子集上，其他 true 类预测成 i 的数量
        fn = int(cm_full[i, :].sum() - tp)
        prec = _safe_div(tp, tp + fp)
        rec = _safe_div(tp, tp + fn)
        f1 = _safe_div(2 * prec * rec, prec + rec) if (prec + rec) > 0 else 0.0
        support = int(cm_full[i, :].sum())
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
    macro_f1 = float(np.mean(f1s)) if len(f1s) else 0.0
    return kept_total, acc, cm_kept, per_class, macro_f1, kept_true_ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", default="", help="reports/<timestamp>_<model>_<dataset> 目录；留空则必须提供 --results_csv")
    ap.add_argument("--results_csv", default="", help="test_results_*.csv 路径（优先于 --run_dir 自动探测）")
    ap.add_argument("--out_md", default="", help="输出 Markdown 路径；默认写入 run_dir/EVT6_TEST_REPORT.md")
    ap.add_argument(
        "--exclude_true_classes",
        default="",
        help="剔除某些 true label 后再额外计算一套参考指标（逗号分隔，支持名称 eq,ep,co,sp,se,ot 或 id 0-5），例如：se",
    )
    args = ap.parse_args()

    run_dir = os.path.abspath(args.run_dir) if args.run_dir else ""
    results_csv = os.path.abspath(args.results_csv) if args.results_csv else ""

    if results_csv:
        csv_path = results_csv
        if not os.path.exists(csv_path):
            raise FileNotFoundError(csv_path)
        if not run_dir:
            run_dir = os.path.dirname(csv_path)
    else:
        if not run_dir:
            raise ValueError("请提供 --results_csv 或 --run_dir")
        csv_path = _find_default_results_csv(run_dir)

    df = pd.read_csv(csv_path, low_memory=False)
    if "tgt_evt6" not in df.columns or "pred_evt6" not in df.columns:
        raise KeyError(f"CSV 缺少列：需要 tgt_evt6/pred_evt6，实际列={list(df.columns)}")

    y_true = df["tgt_evt6"].astype(int).to_numpy()
    y_pred = df["pred_evt6"].astype(int).to_numpy()

    num_classes = 6
    acc, cm, per_class, macro_f1 = _compute_metrics(y_true, y_pred, num_classes=num_classes)
    names = [ID2NAME[i] for i in range(num_classes)]
    exclude_true_ids = _parse_class_list(args.exclude_true_classes)

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    out_md = os.path.abspath(args.out_md) if args.out_md else os.path.join(run_dir, "EVT6_TEST_REPORT.md")
    os.makedirs(os.path.dirname(out_md), exist_ok=True)

    # 同时输出 confusion matrix csv，方便你后续画图/比对论文
    cm_csv = os.path.join(os.path.dirname(out_md), "confusion_matrix_evt6.csv")
    pd.DataFrame(cm, index=[f"true_{n}" for n in names], columns=[f"pred_{n}" for n in names]).to_csv(cm_csv)

    lines = []
    lines.append("# EVT6 分类测试报告（Accuracy / Macro-F1 / Confusion Matrix）\n\n")
    lines.append(f"- 生成时间：{now}\n")
    lines.append(f"- 输入结果：`{csv_path}`\n")
    lines.append(f"- 输出报告：`{out_md}`\n")
    lines.append(f"- 混淆矩阵 CSV：`{cm_csv}`\n")
    lines.append("\n---\n\n")

    lines.append("## 1. Overall 指标\n\n")
    lines.append(f"- **Accuracy**：{acc:.4f}\n")
    lines.append(f"- **Macro-F1**：{macro_f1:.4f}\n")
    lines.append("\n---\n\n")

    lines.append("## 2. 各类别指标（论文常见表格口径）\n\n")
    lines.append("| class | support | precision | recall | f1 |\n")
    lines.append("|---|---:|---:|---:|---:|\n")
    for r in per_class:
        lines.append(
            f"| {r['class_name']} | {int(r['support'])} | {r['precision']:.4f} | {r['recall']:.4f} | {r['f1']:.4f} |\n"
        )
    lines.append("\n---\n\n")

    lines.append("## 3. Confusion Matrix（true × pred）\n\n")
    lines.append(_cm_to_markdown(cm, names))

    if exclude_true_ids:
        kept_total, acc_ex, cm_kept, per_class_ex, macro_f1_ex, kept_true_ids = _compute_metrics_excluding_true(
            y_true, y_pred, num_classes=num_classes, exclude_true_ids=exclude_true_ids
        )
        kept_true_names = [ID2NAME[i] for i in kept_true_ids]
        exclude_true_names = [ID2NAME.get(i, str(i)) for i in exclude_true_ids]

        cm_csv_ex = os.path.join(os.path.dirname(out_md), f"confusion_matrix_evt6_excl-{'-'.join(exclude_true_names)}.csv")
        pd.DataFrame(
            cm_kept, index=[f"true_{n}" for n in kept_true_names], columns=[f"pred_{n}" for n in names]
        ).to_csv(cm_csv_ex)

        lines.append("\n---\n\n")
        lines.append("## 4. 参考：剔除部分 true 类后的指标（仅用于口径对比）\n\n")
        lines.append(f"- 剔除 true 类：`{', '.join(exclude_true_names)}`\n")
        lines.append(f"- 保留样本数：{kept_total}\n")
        lines.append(f"- **Accuracy（exclude true={','.join(exclude_true_names)}）**：{acc_ex:.4f}\n")
        lines.append(f"- **Macro-F1（exclude true={','.join(exclude_true_names)}）**：{macro_f1_ex:.4f}\n")
        lines.append(f"- 混淆矩阵 CSV（仅保留 true 行，pred 列仍为 6 类）：`{cm_csv_ex}`\n")
        lines.append("\n\n")

        lines.append("### 4.1 各类别指标（仅对保留的 true 类统计）\n\n")
        lines.append("| class | support | precision | recall | f1 |\n")
        lines.append("|---|---:|---:|---:|---:|\n")
        for r in per_class_ex:
            lines.append(
                f"| {r['class_name']} | {int(r['support'])} | {r['precision']:.4f} | {r['recall']:.4f} | {r['f1']:.4f} |\n"
            )
        lines.append("\n\n")

        lines.append("### 4.2 Confusion Matrix（true×pred；true 为保留类，pred 仍为 6 类）\n\n")
        lines.append(_cm_to_markdown_rect(cm_kept, kept_true_names, names))

    with open(out_md, "w", encoding="utf-8") as f:
        f.write("".join(lines))

    print(f"[OK] report: {out_md}")
    print(f"[OK] confusion matrix csv: {cm_csv}")


if __name__ == "__main__":
    main()



