#!/usr/bin/env python3
"""
DiTing2.0 data

data DiTing2.0 data HDF5 + JSON data SeisMoLLM data npy + meta.csv data. 

value: 
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
    data, data 0 data. 
    
    Args:
        x_c_l: data(data)
        x: data, shape (3, L)
        L_out: data
    
    Returns:
        out: data, shape (3, L_out)
        c_l: data
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
    data ML, data SeisMoLLM data. 
    
    Args:
        mag: data
        magtype: data(Ms/Mb/ML)
    
    Returns:
        data ML data, clip data [0, 8]
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
    data, data SeisMoLLM value: 
    {"u": 0, "c": 0, "r": 1, "d": 1}
    
    Args:
        motion: data
    
    Returns:
        0 data 1, data None
    """
    if motion is None or motion == '':
        return None
    m = str(motion).strip().upper()  # Open-source note: implementation detail.
    if m in ["", "N"]:
        return None
    mp = {"U": 0, "C": 0, "R": 1, "D": 1}
    return mp.get(m, None)


def main():
    ap = argparse.ArgumentParser(description="DiTing2.0 data")
    ap.add_argument("--h5", required=True, help="DiTing2.0 HDF5 data")
    ap.add_argument("--json", required=True, help="DiTing2.0 JSON data")
    ap.add_argument("--out_dir", required=True, help="data")
    ap.add_argument("--meta_csv", default="meta_diting2_full.csv", help="metadata CSV data")
    ap.add_argument("--in_samples", type=int, default=8192, help="data")

    # Open-source note: implementation detail.
    ap.add_argument("--pre", type=int, default=2000, help="Pg data")
    ap.add_argument("--azi_offset", type=float, default=0.0, 
                    help="data(data Pg_azi data azimuth data back-azimuth, data 180)")
    ap.add_argument("--require_full5", action="store_true",
                    help="data 5 data(Pg,Sg,mag,Pg_azi,Pg_dist,polarityin{U,C,R,D})")
    ap.add_argument("--require_ps", action="store_true",
                    help="dpk data Pg data Sg data(data S data -1 data)")

    # Open-source note: implementation detail.
    ap.add_argument("--dist_cap_km", type=float, default=500.0, help="data(km)")
    
    # Open-source note: implementation detail.
    ap.add_argument("--max_records", type=int, default=0, 
                    help="data(0=data), data")

    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    wave_dir = out_dir / "waves"
    wave_dir.mkdir(parents=True, exist_ok=True)

    # Open-source note: implementation detail.
    print(f"data JSON value: {args.json}")
    with open(args.json, 'r', encoding='utf-8') as f:
        meta_dict = json.load(f)
    print(f"  data {len(meta_dict)} data")

    meta_rows = []
    kept = 0
    skipped_no_pg = 0
    skipped_full5 = 0
    skipped_window = 0

    # Open-source note: implementation detail.
    print(f"data HDF5 value: {args.h5}")
    print(f"  data JSON data key data(data HDF5 data keys)")
    
    keys = list(meta_dict.keys())
    if args.max_records > 0:
        keys = keys[:args.max_records]
        print(f"  [WARN] value: data {args.max_records} data")
    print(f"  JSON data {len(keys)} data")
    
    with h5py.File(args.h5, "r") as f:
        for rid in tqdm(keys, desc="data"):
            rid = str(rid)
            meta = meta_dict[rid]

            # Open-source note: implementation detail.
            try:
                if rid not in f:
                    continue
                x = f[rid][()]  # Open-source note: implementation detail.
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

            # Open-source note: implementation detail.
            Pg = meta.get("Pg", None)
            Sg = meta.get("Sg", None)
            Pg_azi = meta.get("Pg_azi", None)
            Pg_dist = meta.get("Pg_dist", None)
            mag = meta.get("mag", None)
            magtype = meta.get("magtype", None) or meta.get("mag_type", None)
            pol_raw = meta.get("Pg_polarity", None)

            # Open-source note: implementation detail.
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

            # Open-source note: implementation detail.
            if args.require_full5:
                if (Pg is None) or (Sg is None) or (baz is None) or \
                   (dis is None) or (emg is None) or (pmp is None):
                    skipped_full5 += 1
                    continue

            # Open-source note: implementation detail.
            if Pg is None:
                skipped_no_pg += 1
                continue

            # Open-source note: implementation detail.
            c_l = Pg - args.pre
            x_crop, c_l_actual = crop_with_anchor(c_l, x, L_out=args.in_samples)

            # Open-source note: implementation detail.
            Pg_new = Pg - c_l_actual
            Sg_new = Sg - c_l_actual if Sg is not None else None

            # Open-source note: implementation detail.
            if args.require_full5 or args.require_ps:
                if not (0 <= Pg_new < args.in_samples):
                    skipped_window += 1
                    continue
                if Sg_new is None or not (0 <= Sg_new < args.in_samples):
                    skipped_window += 1
                    continue

            # Open-source note: implementation detail.
            part = 0  # Open-source note: implementation detail.
            npy_name = f"{rid}_{part}.npy"
            npy_path = wave_dir / npy_name
            np.save(npy_path, x_crop)

            # Open-source note: implementation detail.
            # Open-source note: implementation detail.
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
                "st_mag": np.nan,  # Open-source note: implementation detail.
                "p_motion": str(pol_raw).upper() if pol_raw not in [None, ''] else '',
                "p_clarity": '',  # Open-source note: implementation detail.
                "Z_P_power_snr": 10.0,  # Open-source note: implementation detail.
                "N_S_power_snr": 10.0,
                "E_S_power_snr": 10.0,
                "P_residual": '',
                "S_residual": '',
                "net": '',
                "sta_id": 0,
                # Open-source note: implementation detail.
                "_npy_path": str(npy_path.relative_to(args.out_dir)),
                "_pmp_bin": pmp if pmp is not None else -1,
            })
            kept += 1

    # Open-source note: implementation detail.
    df = pd.DataFrame(meta_rows)
    out_csv = out_dir / args.meta_csv
    df.to_csv(out_csv, index=False)

    # Open-source note: implementation detail.
    print(f"\n{'='*60}")
    print(f"data!")
    print(f"{'='*60}")
    print(f"value: {kept:,}")
    print(f"value: ")
    print(f"  - data Pg: {skipped_no_pg:,}")
    if args.require_full5:
        print(f"  - 5 value: {skipped_full5:,}")
        print(f"  - P/S value: {skipped_window:,}")
    print(f"\ntext: ")
    print(f"  - value: {wave_dir}")
    print(f"  - metadata CSV: {out_csv}")
    print(f"\ntext(data): ")
    
    # Open-source note: implementation detail.
    pmp_counts = df['_pmp_bin'].value_counts()
    if -1 in pmp_counts.index:
        pmp_valid = len(df) - pmp_counts[-1]
    else:
        pmp_valid = len(df)
    print(f"  - data (pmp) value: {pmp_valid:,} / {len(df):,} ({pmp_valid/len(df)*100:.1f}%)")
    if 0 in pmp_counts.index and 1 in pmp_counts.index:
        print(f"    - Class 0 (U/C): {pmp_counts[0]:,}")
        print(f"    - Class 1 (R/D): {pmp_counts[1]:,}")
        print(f"    - value: {pmp_counts[0]/pmp_counts[1]:.2f}:1")
    
    # Open-source note: implementation detail.
    baz_valid = df['baz'].notna().sum()
    dis_valid = df['dis'].notna().sum()
    emg_valid = df['evmag'].notna().sum()
    print(f"  - data (baz): {baz_valid:,} / {len(df):,} ({baz_valid/len(df)*100:.1f}%)")
    print(f"  - data (dis): {dis_valid:,} / {len(df):,} ({dis_valid/len(df)*100:.1f}%)")
    print(f"  - data (emg): {emg_valid:,} / {len(df):,} ({emg_valid/len(df)*100:.1f}%)")
    
    print(f"\ntext: ")
    print(f"1. datasets/diting.py, data DiTing2 data")
    print(f"2. datasets/diting2.py")
    print(f"3. value: python main.py --dataset-name diting2 --data {args.out_dir} ...")


if __name__ == "__main__":
    main()

