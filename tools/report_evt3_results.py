"""
从 test_results_*.csv 生成 evt3（三分类）结果报告（论文口径）：
- Overall Accuracy / Macro-F1
- per-class Precision/Recall/F1
- confusion matrix

期望输入：
  reports/<run>/test_results_*_test.csv
其中至少包含列：pred_evt3, tgt_evt3（整数类别 id）
"""

import argparse
import os
import numpy as np
import pandas as pd

ID2NAME = {0: "eq", 1: "ep", 2: "co"}


def _safe_div(a: float, b: float) -> float:
    return float(a) / float(b) if b else 0.0


def _per_class_prf(cm: np.ndarray):
    # cm[true, pred]
    num = cm.shape[0]
    out = []
    for i in range(num):
        tp = cm[i, i]
        pred_pos = cm[:, i].sum()
        true_pos = cm[i, :].sum()
        p = _safe_div(tp, pred_pos)
        r = _safe_div(tp, true_pos)
        f1 = _safe_div(2 * p * r, (p + r))
        out.append((p, r, f1, int(true_pos)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True, help="训练 run 目录（包含 test_results_*.csv）")
    args = ap.parse_args()

    run_dir = args.run_dir
    if os.path.isdir(run_dir) and not os.path.isabs(run_dir):
        run_dir = os.path.abspath(run_dir)

    # find test_results csv
    cands = [f for f in os.listdir(run_dir) if f.startswith("test_results_") and f.endswith(".csv")]
    if not cands:
        raise FileNotFoundError(f"未在 {run_dir} 找到 test_results_*.csv")
    cands.sort()
    csv_path = os.path.join(run_dir, cands[-1])

    df = pd.read_csv(csv_path)
    if "tgt_evt3" not in df.columns or "pred_evt3" not in df.columns:
        raise KeyError(f"CSV 缺少列：需要 tgt_evt3/pred_evt3，实际列={list(df.columns)}")

    y_true = df["tgt_evt3"].astype(int).to_numpy()
    y_pred = df["pred_evt3"].astype(int).to_numpy()

    acc = float((y_true == y_pred).mean()) if len(y_true) else 0.0

    cm = np.zeros((3, 3), dtype=int)
    for t, p in zip(y_true, y_pred):
        if 0 <= t < 3 and 0 <= p < 3:
            cm[t, p] += 1

    per = _per_class_prf(cm)
    macro_f1 = float(np.mean([x[2] for x in per])) if per else 0.0

    out_md = os.path.join(run_dir, "EVT3_TEST_REPORT.md")
    cm_csv = os.path.join(run_dir, "confusion_matrix_evt3.csv")

    # write confusion matrix csv
    cm_df = pd.DataFrame(cm, index=[ID2NAME[i] for i in range(3)], columns=[ID2NAME[i] for i in range(3)])
    cm_df.to_csv(cm_csv)

    lines = []
    lines.append("# EVT3 分类测试报告（Accuracy / Macro-F1 / Confusion Matrix）\n")
    lines.append(f"- 输入结果：`{csv_path}`\n")
    lines.append(f"- 输出报告：`{out_md}`\n")
    lines.append(f"- 混淆矩阵 CSV：`{cm_csv}`\n\n")
    lines.append("---\n\n")
    lines.append("## 1. Overall 指标\n\n")
    lines.append(f"- **Accuracy**：{acc:.4f}\n")
    lines.append(f"- **Macro-F1**：{macro_f1:.4f}\n\n")
    lines.append("---\n\n")
    lines.append("## 2. 各类别指标（论文常见表格口径）\n\n")
    lines.append("| class | support | precision | recall | f1 |\n")
    lines.append("|---|---:|---:|---:|---:|\n")
    for i in range(3):
        name = ID2NAME[i]
        p, r, f1, sup = per[i]
        lines.append(f"| {name} | {sup} | {p:.4f} | {r:.4f} | {f1:.4f} |\n")
    lines.append("\n---\n\n")
    lines.append("## 3. Confusion Matrix（true × pred）\n\n")
    lines.append("| true\\pred | eq | ep | co |\n")
    lines.append("|---|---|---|---|\n")
    for i in range(3):
        lines.append(f"| {ID2NAME[i]} | {cm[i,0]} | {cm[i,1]} | {cm[i,2]} |\n")

    with open(out_md, "w", encoding="utf-8") as f:
        f.writelines(lines)

    print(f"[OK] 写出：{out_md}")
    print(f"[OK] 写出：{cm_csv}")


if __name__ == "__main__":
    main()


