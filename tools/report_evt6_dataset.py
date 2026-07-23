#!/usr/bin/env python3
"""
data evt6 data(Markdown). 

value: 
  OUT_DIR/meta_evt6.csv
value: 
  meta_evt6_train.csv / meta_evt6_val.csv / meta_evt6_test.csv
"""

import argparse
import os
from datetime import datetime

import pandas as pd


# Open-source note: implementation detail.
# Open-source note: implementation detail.
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
    lines.append(f"# EVT6 Dataset Report\n\n")
    lines.append(f"- value: {now}\n")
    lines.append(f"- value: `{out_dir}`\n")
    lines.append(f"- data meta: `{meta}`\n")
    lines.append("\n---\n")
    lines.append("## Selection record\n\n")
    if record_exist:
        lines.append(f"- value: `{record_path}`\n")
        ts = ((record or {}).get("strict_split") or {}).get("test_from_val", False)
        if ts:
            lines.append("- value: **test data val data(test_from_val=true)**\n")
    else:
        lines.append("- value: data `selection_record_evt6.json`(data prepare data)\n")
    lines.append("\n---\n")

    lines.append("## Split summary\n\n")
    lines.append("`evtype` -> `evt6 id`: \n")
    for k, v in ID2NAME.items():
        lines.append(f"- `{v}` -> `{k}`\n")

    lines.append("\n---\n")
    lines.append("## Event-level summary\n\n")
    lines.append("data `tools/prepare_diting2_evt6.py` value: \n")
    lines.append("- **data(nat)**: data `diting2_seismollm_full5` data `.npy`(data natural JSON). \n")
    lines.append("  - value: `{natural_full5_dir}/waves/*.npy`(data `meta_full5.csv` data `_npy_path` data). \n")
    lines.append("  - `evtype` value: `CENC_DiTingv2_natural_earthquake.json`, data key data. \n")
    lines.append("- **data(non)**: data `CENC_DiTingv2_non_natural_earthquake.hdf5` data `waves_non/*.npy`. \n")
    lines.append("  - value: data `Pg`(data Pg data). \n")
    lines.append("  - value: data `Pg - pre` data(data `pre=2000`), data `in_samples=8192`. \n")
    lines.append("\n> data `_src` value: `nat` data `non`. \n")

    lines.append("\n---\n")
    lines.append("## Class distribution\n\n")
    lines.append(f"- value: **{len(df)}**\n")
    if "_src" in df.columns:
        lines.append(f"- data(_src): \n\n```\n{_value_counts(df,'_src').to_string()}\n```\n")
    lines.append("## Examples per class\n\n")
    lines.append(f"```\n{_value_counts(df,'_evt6').to_string()}\n```\n")
    lines.append("## Metadata preview\n\n")
    lines.append(f"```\n{df['_evt6_name'].value_counts().reindex(list(ID2NAME.values())).to_string()}\n```\n")

    lines.append("\n---\n")
    lines.append("## Files\n\n")
    lines.append(f"- `_npy_path` value: true={int(isabs.sum())} / false={int((~isabs).sum())}\n")
    lines.append(f"- value: missing={int((~exists).sum())}\n")

    lines.append("\n---\n")
    lines.append("## Notes\n\n")
    if all(strict_exist.values()):
        lines.append("data**data**(data): \n")
        for k, p in strict.items():
            lines.append(f"- `{k}`: `{p}`\n")
        lines.append("\ntext split value: \n")
        for split in ("train", "val", "test"):
            sdf = _read_csv(strict[split])
            vc = sdf["_evt6"].value_counts().sort_index()
            lines.append(f"\n### 5.{['train','val','test'].index(split)+1} {split}\n")
            lines.append(f"- value: {len(sdf)}\n")
            lines.append(f"```\n{vc.to_string()}\n```\n")
    else:
        lines.append("data `meta_evt6_train/val/test.csv`, data **ratio split**: \n")
        lines.append("- data `datasets/diting2_evt6.py` value: data shuffle(seed data), data `train_size=0.8, val_size=0.1, test=0.1` data. \n")
        N = len(df)
        tr = int(0.8 * N)
        va = tr + int(0.1 * N)
        lines.append(f"- Fallback split for N={N}: train={tr}, val={va-tr}, test={N-va}\n")

    lines.append("\n---\n")
    lines.append("## Label mapping\n\n")
    lines.append("Example command for preparing an EVT6 dataset directory:\n")
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
    lines.append("\ntext: \n")
    lines.append("- data `meta_evt6_train.csv / meta_evt6_val.csv / meta_evt6_test.csv`, data. \n")
    lines.append("- data `datasets/diting2_evt6.py` data split data meta data. \n")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("".join(lines))

    print(f"[OK] report: {report_path}")


if __name__ == "__main__":
    main()


