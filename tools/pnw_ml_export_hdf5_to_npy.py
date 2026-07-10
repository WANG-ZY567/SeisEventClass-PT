#!/usr/bin/env python3
"""
从 PNW HDF5 导出与 DiTing 训练一致的 .npy 波形（需 h5py）。

⚠️ HDF5 内路径与 trace_name 的对应关系因数据集版本而异。
请先运行：python tools/pnw_ml_inspect_hdf5.py <file.h5>

trace_name 支持：
  - **PNW/ComCat**：`bucket4$0,:3,:15001` → 读取 `f['data/bucket4'][0, :3, :15001]`（bucket 名可带或不带 `data/` 前缀）
  - **其它**：整段作为 dataset 路径；或 `waveforms/` 下；或将 `$ , :` 替换为 `_` 后再试（旧逻辑）

导出形状：尽量变为 (3, L)；若 L != in_samples，则 **中心裁剪或填充** 到 in_samples（默认 8192）。

**不运行实验时**你可只检查本文件参数；首次导出建议先用 --max_rows 10 试跑。
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path


def sanitize_key(s: str) -> str:
    return re.sub(r"[$,:]", "_", s)


def parse_pnw_trace_index(idx_part: str):
    """
    解析 metadata 中形如 `0,:3,:15001` 的索引串（逗号分三段：轴0 整数、轴1/2 可为 :k 或 :）。
    返回 (int, slice|int, slice|int)，供 h5 dataset [...] 使用。
    """
    parts = [p.strip() for p in idx_part.split(",")]
    if len(parts) != 3:
        raise ValueError(f"expected 3 comma-separated index parts, got {idx_part!r}")

    def parse_axis(p: str):
        p = p.strip()
        if p == ":" or p == "":
            return slice(None)
        if p.startswith(":"):
            return slice(0, int(p[1:]))
        return int(p)

    row = int(parts[0])
    a1 = parse_axis(parts[1])
    a2 = parse_axis(parts[2])
    return row, a1, a2


def read_waveform_from_h5(h5, trace_name: str):
    """
    支持两种 trace_name：
    1) PNW/ComCat：`bucket4$0,:3,:15001` → 数据集路径 `data/bucket4`，再按索引切片。
    2) 其它：整段作为 dataset 路径（或 waveforms/ 下、sanitize 后），与旧逻辑一致。
    """
    tn = trace_name.strip()
    if "$" in tn:
        bucket_part, idx_part = tn.split("$", 1)
        path = bucket_part if bucket_part.startswith("data/") else f"data/{bucket_part}"
        row, a1, a2 = parse_pnw_trace_index(idx_part)
        if path not in h5:
            raise KeyError(path)
        ds = h5[path]
        return ds[row, a1, a2]

    if tn in h5:
        return h5[tn][()]
    if "waveforms" in h5 and tn in h5["waveforms"]:
        return h5["waveforms"][tn][()]
    sn = sanitize_key(tn)
    if sn in h5:
        return h5[sn][()]
    if "waveforms" in h5 and sn in h5["waveforms"]:
        return h5["waveforms"][sn][()]
    raise KeyError(tn)


def to_3c_L(arr, target_L: int):
    import numpy as np

    x = np.asarray(arr, dtype=np.float32)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    if x.shape[0] == 3:
        pass
    elif x.shape[-1] == 3:
        x = x.T
    else:
        raise ValueError(f"unexpected shape {x.shape}, expect 3xL or Lx3")
    L = x.shape[1]
    if L == target_L:
        return x
    if L > target_L:
        s = (L - target_L) // 2
        return x[:, s : s + target_L]
    pad = target_L - L
    return np.pad(x, ((0, 0), (0, pad)), mode="constant")


def main() -> None:
    try:
        import h5py  # type: ignore
        import numpy as np
    except ImportError:
        print("需要: pip install h5py numpy")
        raise SystemExit(1)

    ap = argparse.ArgumentParser()
    ap.add_argument("--hdf5", type=Path, required=True)
    ap.add_argument("--meta_csv", type=Path, required=True, help="含 trace_name 与 _npy_path")
    ap.add_argument("--data_dir", type=Path, required=True, help="meta 中 _npy_path 的根目录")
    ap.add_argument("--in_samples", type=int, default=8192)
    ap.add_argument("--max_rows", type=int, default=0, help="0 表示全部")
    args = ap.parse_args()

    import csv

    n_ok = 0
    n_fail = 0
    rows = []
    with args.meta_csv.open(newline="") as f:
        r = csv.DictReader(f)
        rows = list(r)
    if args.max_rows > 0:
        rows = rows[: args.max_rows]

    with h5py.File(args.hdf5, "r") as h5:
        for row in rows:
            tn = (row.get("trace_name") or "").strip()
            rel = (row.get("_npy_path") or "").strip()
            if not tn or not rel:
                n_fail += 1
                continue
            out_path = args.data_dir / rel
            try:
                raw = read_waveform_from_h5(h5, tn)
                arr = to_3c_L(raw, args.in_samples)
                out_path.parent.mkdir(parents=True, exist_ok=True)
                np.save(out_path, arr)
                n_ok += 1
            except Exception as e:
                n_fail += 1
                if n_fail <= 5:
                    print("FAIL", tn, e)

    print("ok", n_ok, "fail", n_fail)


if __name__ == "__main__":
    main()
