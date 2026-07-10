#!/usr/bin/env python3
"""
列出 HDF5 顶层结构与若干 dataset 形状（需安装 h5py）。
不读取完整波形到内存。用于确认 trace_name 与文件内路径的对应关系。
"""
from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    try:
        import h5py  # type: ignore
    except ImportError:
        print("请先安装: pip install h5py")
        raise SystemExit(1)

    ap = argparse.ArgumentParser()
    ap.add_argument("hdf5_path", type=Path)
    ap.add_argument("--max_datasets", type=int, default=30, help="最多打印多少个 dataset")
    args = ap.parse_args()

    path = args.hdf5_path
    if not path.is_file():
        print("文件不存在:", path)
        raise SystemExit(1)

    n = 0
    with h5py.File(path, "r") as f:
        print("文件:", path)
        print("顶层 keys (前 50):", list(f.keys())[:50])

        def visitor(name, obj):
            nonlocal n
            if isinstance(obj, h5py.Dataset) and n < args.max_datasets:
                print(f"  [D] {name} shape={obj.shape} dtype={obj.dtype}")
                n += 1

        f.visititems(visitor)
        if n >= args.max_datasets:
            print(f"... 仅展示前 {args.max_datasets} 个 dataset，可增加 --max_datasets")


if __name__ == "__main__":
    main()
