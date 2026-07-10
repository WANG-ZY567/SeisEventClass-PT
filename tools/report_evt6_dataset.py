#!/usr/bin/env python3
"""
生成 evt6 六分类数据集报告（Markdown）。

默认读取：
  OUT_DIR/meta_evt6.csv
并自动探测是否存在严格划分文件：
  meta_evt6_train.csv / meta_evt6_val.csv / meta_evt6_test.csv
"""

import argparse
import os
from datetime import datetime

import pandas as pd


# 注意：一些论文把“坍塌”记为 CO；在 DiTing2.0 的 evtype 中常见写法是 ss，
# 但我们在 evt6 报告里把 id=2 显示为 co（便于与论文对齐）。
ID2NAME = {0: "eq", 1: "ep", 2: "co", 3: "sp", 4: "se", 5: "ot"}


def _value_counts(df: pd.DataFrame, col: str):
    vc = df[col].value_counts(dropna=False)
    if col == "_evt6":
        vc = vc.sort_index()
    return vc


def _read_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="/path/to/diting2_evt6")
    ap.add_argument("--report_path", default="")
    args = ap.parse_args()

    out_dir = os.path.abspath(args.out_dir)
    meta = os.path.join(out_dir, "meta_evt6.csv")
    if not os.path.exists(meta):
        raise FileNotFoundError(meta)

    strict = {
        split: os.path.join(out_dir, f"meta_evt6_{split}.csv")
        for split in ("train", "val", "test")
    }
    strict_exist = {k: os.path.exists(v) for k, v in strict.items()}
    record_path = os.path.join(out_dir, "selection_record_evt6.json")
    record_exist = os.path.exists(record_path)
    record = None
    if record_exist:
        try:
            import json as _json
            record = _json.load(open(record_path, "r", encoding="utf-8"))
        except Exception:
            record = None

    df = _read_csv(meta)
    df["_evt6_name"] = df["_evt6"].map(ID2NAME)
    isabs = df["_npy_path"].astype(str).apply(os.path.isabs)
    exists = df["_npy_path"].astype(str).apply(os.path.exists)

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    report_path = (
        os.path.join(out_dir, "EVT6_DATASET_REPORT.md")
        if not args.report_path
        else os.path.abspath(args.report_path)
    )
    os.makedirs(os.path.dirname(report_path), exist_ok=True)

    lines = []
    lines.append(f"# EVT6 六分类数据集报告\n")
    lines.append(f"- 生成时间：{now}\n")
    lines.append(f"- 数据目录：`{out_dir}`\n")
    lines.append(f"- 主 meta：`{meta}`\n")
    lines.append("\n---\n")
    lines.append("## 0. 生成记录（可复现性）\n")
    if record_exist:
        lines.append(f"- 记录文件：`{record_path}`\n")
        ts = ((record or {}).get("strict_split") or {}).get("test_from_val", False)
        if ts:
            lines.append("- 评测口径：**test 与 val 相同（test_from_val=true）**\n")
    else:
        lines.append("- 记录文件：未找到 `selection_record_evt6.json`（可通过重新运行 prepare 脚本生成）\n")
    lines.append("\n---\n")

    lines.append("## 1. 任务定义（evt6）\n")
    lines.append("`evtype` -> `evt6 id`：\n")
    for k, v in ID2NAME.items():
        lines.append(f"- `{v}` -> `{k}`\n")

    lines.append("\n---\n")
    lines.append("## 2. 数据来源与样本选择（来自代码）\n")
    lines.append("该数据集由 `tools/prepare_diting2_evt6.py` 生成：\n")
    lines.append("- **自然事件（nat）**：复用 `diting2_seismollm_full5` 已生成的波形 `.npy`（避免重新裁窗全量 natural JSON）。\n")
    lines.append("  - 波形来自：`{natural_full5_dir}/waves/*.npy`（通过 `meta_full5.csv` 的 `_npy_path` 连接）。\n")
    lines.append("  - `evtype` 来自：`CENC_DiTingv2_natural_earthquake.json`，按 key 关联。\n")
    lines.append("- **非自然事件（non）**：从 `CENC_DiTingv2_non_natural_earthquake.hdf5` 裁窗生成 `waves_non/*.npy`。\n")
    lines.append("  - 过滤条件：必须存在 `Pg`（没有 Pg 的样本会被跳过）。\n")
    lines.append("  - 裁窗：以 `Pg - pre` 为左端（默认 `pre=2000`），输出长度 `in_samples=8192`。\n")
    lines.append("\n> 字段 `_src` 标记样本来源：`nat` 或 `non`。\n")

    lines.append("\n---\n")
    lines.append("## 3. 类别均衡（当前数据）\n")
    lines.append(f"- 总样本数：**{len(df)}**\n")
    if "_src" in df.columns:
        lines.append(f"- 来源分布（_src）：\n\n```\n{_value_counts(df,'_src').to_string()}\n```\n")
    lines.append("### 3.1 6 类分布（_evt6）\n")
    lines.append(f"```\n{_value_counts(df,'_evt6').to_string()}\n```\n")
    lines.append("### 3.2 6 类分布（名称）\n")
    lines.append(f"```\n{df['_evt6_name'].value_counts().reindex(list(ID2NAME.values())).to_string()}\n```\n")

    lines.append("\n---\n")
    lines.append("## 4. 路径与文件存在性（当前数据）\n")
    lines.append(f"- `_npy_path` 绝对路径：true={int(isabs.sum())} / false={int((~isabs).sum())}\n")
    lines.append(f"- 文件存在：missing={int((~exists).sum())}\n")

    lines.append("\n---\n")
    lines.append("## 5. 数据划分方式\n")
    if all(strict_exist.values()):
        lines.append("检测到**严格计数划分文件**（推荐用于与论文严格对齐）：\n")
        for k, p in strict.items():
            lines.append(f"- `{k}`：`{p}`\n")
        lines.append("\n各 split 的真实分布：\n")
        for split in ("train", "val", "test"):
            sdf = _read_csv(strict[split])
            vc = sdf["_evt6"].value_counts().sort_index()
            lines.append(f"\n### 5.{['train','val','test'].index(split)+1} {split}\n")
            lines.append(f"- 行数：{len(sdf)}\n")
            lines.append(f"```\n{vc.to_string()}\n```\n")
    else:
        lines.append("当前未检测到 `meta_evt6_train/val/test.csv`，因此使用 **ratio split**：\n")
        lines.append("- 在 `datasets/diting2_evt6.py` 中：先 shuffle（seed 固定），再按 `train_size=0.8, val_size=0.1, test=0.1` 切片。\n")
        N = len(df)
        tr = int(0.8 * N)
        va = tr + int(0.1 * N)
        lines.append(f"- 以当前 N={N} 计算：train={tr}, val={va-tr}, test={N-va}\n")

    lines.append("\n---\n")
    lines.append("## 6. 如何生成“严格计数划分”（建议方案）\n")
    lines.append("为了参考你贴的 CNN 论文那种“固定数量划分”的口径，建议使用：\n")
    lines.append("\n```bash\n")
    lines.append("python tools/prepare_diting2_evt6.py \\\n")
    lines.append(f"  --out_dir \"{out_dir}\" \\\n")
    # If selection record exists, prefer printing a command consistent with it.
    quota_mode = str((record or {}).get("quota_mode") or "none").lower()
    strict_split = (record or {}).get("strict_split") or {}
    seed = strict_split.get("seed", 100)
    val_pc = strict_split.get("val_per_class", 300)
    test_pc = strict_split.get("test_per_class", 300)
    quota_seed = (record or {}).get("quota_seed", 100)
    test_from_val = bool(strict_split.get("test_from_val", False))

    if quota_mode != "none":
        # paper-aligned quota mode (no upsampling)
        lines.append(f"  --quota_mode {quota_mode} \\\n")
        lines.append("  --balance none \\\n")
        lines.append("  --split_mode strict \\\n")
        lines.append(f"  --val_per_class {val_pc} --test_per_class {test_pc} \\\n")
        if test_from_val:
            lines.append("  --test_from_val \\\n")
        lines.append(f"  --seed {seed} --quota_seed {quota_seed}\n")
    else:
        # default balanced recipe (kept for the fully balanced setting)
        lines.append("  --balance hybrid --target_per_class 4500 \\\n")
        lines.append("  --split_mode strict \\\n")
        lines.append("  --val_per_class 300 --test_per_class 300 \\\n")
        lines.append("  --seed 100\n")
    lines.append("```\n")
    lines.append("\n说明：\n")
    lines.append("- 上面会输出 `meta_evt6_train.csv / meta_evt6_val.csv / meta_evt6_test.csv`，并确保每类数量严格一致。\n")
    lines.append("- 训练时 `datasets/diting2_evt6.py` 会自动优先读取对应 split 的 meta 文件。\n")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("".join(lines))

    print(f"[OK] report: {report_path}")


if __name__ == "__main__":
    main()


