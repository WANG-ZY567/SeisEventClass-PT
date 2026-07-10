#!/usr/bin/env python3
"""
DiTing2.0 数据预处理脚本

将 DiTing2.0 的 HDF5 + JSON 格式转换为 SeisMoLLM 可直接使用的 npy + meta.csv 格式。

用法示例：
    python tools/prepare_diting2.py \
        --h5 /path/to/CENC_DiTingv2_natural_earthquake.hdf5 \
        --json /path/to/CENC_DiTingv2_natural_earthquake.json \
        --out_dir /path/to/diting2_seismollm_full5 \
        --meta_csv meta_full5.csv \
        --in_samples 8192 \
        --pre 2000 \
        --azi_offset 0 \
        --require_full5
"""

import argparse
import json
import os
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm


def crop_with_anchor(x_c_l, x, L_out=8192):
    """
    以指定位置为锚点裁剪波形，不足部分用 0 填充。
    
    Args:
        x_c_l: 裁剪起始位置（相对于原始波形）
        x: 输入波形，shape (3, L)
        L_out: 输出长度
    
    Returns:
        out: 裁剪后的波形，shape (3, L_out)
        c_l: 实际的裁剪起始位置
    """
    c_l = int(x_c_l)
    c_r = c_l + L_out
    L = x.shape[1]

    if c_l >= 0 and c_r <= L:
        return x[:, c_l:c_r], c_l

    out = np.zeros((x.shape[0], L_out), dtype=x.dtype)
    src_l = max(c_l, 0)
    src_r = min(c_r, L)
    dst_l = src_l - c_l
    dst_r = dst_l + (src_r - src_l)
    if src_r > src_l:
        out[:, dst_l:dst_r] = x[:, src_l:src_r]
    return out, c_l


def mag_to_ml(mag, magtype):
    """
    将不同类型的震级转换为 ML，匹配 SeisMoLLM 代码逻辑。
    
    Args:
        mag: 震级值
        magtype: 震级类型（Ms/Mb/ML）
    
    Returns:
        转换后的 ML 震级，clip 到 [0, 8]
    """
    if mag is None or mag == '':
        return None
    try:
        evmag = float(mag)
    except Exception:
        return None

    mt = (magtype or "").strip().lower()
    if mt == "ms":
        evmag = (evmag + 1.08) / 1.13
    elif mt == "mb":
        evmag = (1.17 * evmag + 0.67) / 1.13
    # else assume already ML or compatible
    evmag = float(np.clip(evmag, 0.0, 8.0))
    return evmag


def pol_to_bin(motion):
    """
    将极性标签转换为二分类，匹配 SeisMoLLM 代码逻辑：
    {"u": 0, "c": 0, "r": 1, "d": 1}
    
    Args:
        motion: 极性标签字符串
    
    Returns:
        0 或 1，如果不可用则返回 None
    """
    if motion is None or motion == '':
        return None
    m = str(motion).strip().upper()  # DiTing2.0 是大写
    if m in ["", "N"]:
        return None
    mp = {"U": 0, "C": 0, "R": 1, "D": 1}
    return mp.get(m, None)


def main():
    ap = argparse.ArgumentParser(description="DiTing2.0 数据预处理")
    ap.add_argument("--h5", required=True, help="DiTing2.0 HDF5 文件路径")
    ap.add_argument("--json", required=True, help="DiTing2.0 JSON 元数据文件路径")
    ap.add_argument("--out_dir", required=True, help="输出目录")
    ap.add_argument("--meta_csv", default="meta_diting2_full.csv", help="输出的 CSV 文件名")
    ap.add_argument("--in_samples", type=int, default=8192, help="输出波形长度")

    # 裁窗参数
    ap.add_argument("--pre", type=int, default=2000, help="Pg 前的样本数")
    ap.add_argument("--azi_offset", type=float, default=0.0, 
                    help="方位角偏移（如果 Pg_azi 是 azimuth 而非 back-azimuth，设为 180）")
    ap.add_argument("--require_full5", action="store_true",
                    help="只保留 5 任务标签齐全的样本（Pg,Sg,mag,Pg_azi,Pg_dist,polarity∈{U,C,R,D}）")
    ap.add_argument("--require_ps", action="store_true",
                    help="dpk 数据要求 Pg 和 Sg 都存在且都在窗内（避免 S 通道被 -1 污染）")

    # 距离上限（匹配 Sigmoid*500 head）
    ap.add_argument("--dist_cap_km", type=float, default=500.0, help="距离上限（km）")
    
    # 小跑验证
    ap.add_argument("--max_records", type=int, default=0, 
                    help="最大处理记录数（0=全部），用于小跑验证")

    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    wave_dir = out_dir / "waves"
    wave_dir.mkdir(parents=True, exist_ok=True)

    # 读取 JSON 元数据
    print(f"正在读取 JSON 元数据：{args.json}")
    with open(args.json, 'r', encoding='utf-8') as f:
        meta_dict = json.load(f)
    print(f"  加载了 {len(meta_dict)} 条记录的元数据")

    meta_rows = []
    kept = 0
    skipped_no_pg = 0
    skipped_full5 = 0
    skipped_window = 0

    # 打开 HDF5 文件（从 JSON 获取 keys，不列举所有 keys）
    print(f"正在处理 HDF5 波形数据：{args.h5}")
    print(f"  将从 JSON 元数据中获取 key 列表（不列举 HDF5 中所有 keys）")
    
    keys = list(meta_dict.keys())
    if args.max_records > 0:
        keys = keys[:args.max_records]
        print(f"  ⚠️ 小跑模式：只处理前 {args.max_records} 条记录")
    print(f"  JSON 包含 {len(keys)} 条记录的元数据")
    
    with h5py.File(args.h5, "r") as f:
        for rid in tqdm(keys, desc="处理进度"):
            rid = str(rid)
            meta = meta_dict[rid]

            # 读取波形（直接访问，不预先列举）
            try:
                if rid not in f:
                    continue
                x = f[rid][()]  # shape 可能是 (10000,3) 或 (3,10000)
                x = np.asarray(x)
                if x.ndim != 2:
                    continue
                if x.shape[0] == 3:
                    x = x
                elif x.shape[1] == 3:
                    x = x.T
                else:
                    continue
                x = x.astype(np.float32)
                L = x.shape[1]
            except Exception:
                continue

            # ----- 提取标签 -----
            Pg = meta.get("Pg", None)
            Sg = meta.get("Sg", None)
            Pg_azi = meta.get("Pg_azi", None)
            Pg_dist = meta.get("Pg_dist", None)
            mag = meta.get("mag", None)
            magtype = meta.get("magtype", None) or meta.get("mag_type", None)
            pol_raw = meta.get("Pg_polarity", None)

            # 解析数值标签
            try:
                Pg = int(Pg) if Pg not in [None, ''] else None
            except Exception:
                Pg = None
            try:
                Sg = int(Sg) if Sg not in [None, ''] else None
            except Exception:
                Sg = None

            try:
                baz = float(Pg_azi) if Pg_azi not in [None, ''] else None
            except Exception:
                baz = None
            if baz is not None:
                baz = (baz + args.azi_offset) % 360.0

            try:
                dis = float(Pg_dist) if Pg_dist not in [None, ''] else None
            except Exception:
                dis = None
            if dis is not None:
                dis = float(np.clip(dis, 0.0, args.dist_cap_km))

            emg = mag_to_ml(mag, magtype)
            pmp = pol_to_bin(pol_raw)

            # 全标签筛选
            if args.require_full5:
                if (Pg is None) or (Sg is None) or (baz is None) or \
                   (dis is None) or (emg is None) or (pmp is None):
                    skipped_full5 += 1
                    continue

            # 必须有 Pg 才能裁窗
            if Pg is None:
                skipped_no_pg += 1
                continue

            # ----- 以 Pg 为锚点裁剪到 8192 -----
            c_l = Pg - args.pre
            x_crop, c_l_actual = crop_with_anchor(c_l, x, L_out=args.in_samples)

            # 调整 picks
            Pg_new = Pg - c_l_actual
            Sg_new = Sg - c_l_actual if Sg is not None else None

            # 验证 picks 在窗口内
            if args.require_full5 or args.require_ps:
                if not (0 <= Pg_new < args.in_samples):
                    skipped_window += 1
                    continue
                if Sg_new is None or not (0 <= Sg_new < args.in_samples):
                    skipped_window += 1
                    continue

            # 保存波形 npy（文件名格式：{rid}_{part}.npy，兼容原 loader）
            part = 0  # DiTing2.0 统一为 part=0
            npy_name = f"{rid}_{part}.npy"
            npy_path = wave_dir / npy_name
            np.save(npy_path, x_crop)

            # 构造 meta 行（匹配 SeisMoLLM datasets/diting.py 的列名）
            # ⚠️ 数值列用 np.nan 而不是 ''，避免整列变 object
            meta_rows.append({
                "key": rid,
                "part": part,
                "ev_id": meta.get("event_id", 0),
                "p_pick": Pg_new if Pg_new is not None and Pg_new >= 0 else -1,
                "s_pick": Sg_new if Sg_new is not None and Sg_new >= 0 else -1,
                "baz": baz if baz is not None else np.nan,
                "dis": dis if dis is not None else np.nan,
                "evmag": emg if emg is not None else np.nan,
                "mag_type": (magtype or "ML").upper(),
                "st_mag": np.nan,  # DiTing2.0 没有 st_mag
                "p_motion": str(pol_raw).upper() if pol_raw not in [None, ''] else '',
                "p_clarity": '',  # DiTing2.0 没有 p_clarity
                "Z_P_power_snr": 10.0,  # 默认值
                "N_S_power_snr": 10.0,
                "E_S_power_snr": 10.0,
                "P_residual": '',
                "S_residual": '',
                "net": '',
                "sta_id": 0,
                # 额外字段用于调试
                "_npy_path": str(npy_path.relative_to(args.out_dir)),
                "_pmp_bin": pmp if pmp is not None else -1,
            })
            kept += 1

    # 保存 CSV
    df = pd.DataFrame(meta_rows)
    out_csv = out_dir / args.meta_csv
    df.to_csv(out_csv, index=False)

    # 统计信息
    print(f"\n{'='*60}")
    print(f"预处理完成！")
    print(f"{'='*60}")
    print(f"保留样本数：{kept:,}")
    print(f"跳过样本数：")
    print(f"  - 无 Pg：{skipped_no_pg:,}")
    if args.require_full5:
        print(f"  - 5 任务标签不全：{skipped_full5:,}")
        print(f"  - P/S 不在窗口内：{skipped_window:,}")
    print(f"\n输出文件：")
    print(f"  - 波形目录：{wave_dir}")
    print(f"  - 元数据 CSV：{out_csv}")
    print(f"\n标签统计（基于保留样本）：")
    
    # 极性统计
    pmp_counts = df['_pmp_bin'].value_counts()
    if -1 in pmp_counts.index:
        pmp_valid = len(df) - pmp_counts[-1]
    else:
        pmp_valid = len(df)
    print(f"  - 极性 (pmp) 有效样本：{pmp_valid:,} / {len(df):,} ({pmp_valid/len(df)*100:.1f}%)")
    if 0 in pmp_counts.index and 1 in pmp_counts.index:
        print(f"    - Class 0 (U/C): {pmp_counts[0]:,}")
        print(f"    - Class 1 (R/D): {pmp_counts[1]:,}")
        print(f"    - 比例: {pmp_counts[0]/pmp_counts[1]:.2f}:1")
    
    # baz/dis/emg 统计（注意：emg=0 也是有效值）
    baz_valid = df['baz'].notna().sum()
    dis_valid = df['dis'].notna().sum()
    emg_valid = df['evmag'].notna().sum()
    print(f"  - 方位角 (baz)：{baz_valid:,} / {len(df):,} ({baz_valid/len(df)*100:.1f}%)")
    print(f"  - 距离 (dis)：{dis_valid:,} / {len(df):,} ({dis_valid/len(df)*100:.1f}%)")
    print(f"  - 震级 (emg)：{emg_valid:,} / {len(df):,} ({emg_valid/len(df)*100:.1f}%)")
    
    print(f"\n下一步：")
    print(f"1. 修改 datasets/diting.py，添加 DiTing2 类")
    print(f"2. 或使用提供的 datasets/diting2.py")
    print(f"3. 运行训练：python main.py --dataset-name diting2 --data {args.out_dir} ...")


if __name__ == "__main__":
    main()

