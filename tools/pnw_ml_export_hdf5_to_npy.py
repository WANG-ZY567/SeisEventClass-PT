#!/usr/bin/env python3
"""
data PNW HDF5 data DiTing data .npy data(data h5py). 

[WARN] HDF5 data trace_name data. 
value: python tools/pnw_ml_inspect_hdf5.py <file.h5>

trace_name value: 
  - **PNW/ComCat**: `bucket4$0,:3,:15001` -> data `f['data/bucket4'][0, :3, :15001]`(bucket data `data/` data)
  - **data**: dataset data; data `waveforms/` data; data `$ , :` data `_` data(data)

value: data (3, L); data L != in_samples, data **data** data in_samples(data 8192). 

**data**data; data --max_rows 10 data. 
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path


def sanitize_key(s: str) -> str:
    return re.sub(r"[$,:]", "_", s)


def parse_pnw_trace_index(idx_part: str):
    """
    data metadata `0,:3,:15001` data(value: text0 data, text1/2 data :k data :). 
    data (int, slice|int, slice|int), data h5 dataset [...] data. 
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
    data trace_name: 
    1) PNW/ComCat: `bucket4$0,:3,:15001` -> data `data/bucket4`, data. 
    2) value: dataset data(data waveforms/ data, sanitize data), data. 
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


def p_aligned_window(
    arr,
    p_arrival_sample: float,
    source_rate_hz: float,
    target_rate_hz: float,
    pre_p_seconds: float,
    post_p_seconds: float,
):
    """Crop a fixed physical P-centred window and resample it.

    The PNW ComCat files are 100 Hz whereas the revised DiTing protocol is
    50 Hz.  Cropping in physical time before downsampling prevents an
    accidental centre-crop/window-offset confound in the external test.
    """
    import numpy as np
    from scipy.signal import resample_poly

    x = np.asarray(arr, dtype=np.float32)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    if x.shape[0] == 3:
        pass
    elif x.shape[-1] == 3:
        x = x.T
    else:
        raise ValueError(f"unexpected shape {x.shape}, expect 3xL or Lx3")
    if source_rate_hz <= 0 or target_rate_hz <= 0:
        raise ValueError("sampling rates must be positive")

    pre_source = int(round(pre_p_seconds * source_rate_hz))
    total_source = int(round((pre_p_seconds + post_p_seconds) * source_rate_hz))
    start = int(round(p_arrival_sample)) - pre_source
    stop = start + total_source
    padded = np.zeros((3, total_source), dtype=np.float32)
    src_start, src_stop = max(start, 0), min(stop, x.shape[1])
    if src_stop > src_start:
        dst_start = src_start - start
        padded[:, dst_start : dst_start + (src_stop - src_start)] = x[:, src_start:src_stop]

    # PNW is 100 Hz and DiTing is 50 Hz; use rational polyphase resampling
    # rather than decimation so this remains valid if a documented rate changes.
    from fractions import Fraction

    ratio = Fraction(float(target_rate_hz) / float(source_rate_hz)).limit_denominator(1000)
    y = resample_poly(padded, ratio.numerator, ratio.denominator, axis=1).astype(np.float32, copy=False)
    expected = int(round((pre_p_seconds + post_p_seconds) * target_rate_hz))
    return to_3c_L(y, expected)


def main() -> None:
    try:
        import h5py  # type: ignore
        import numpy as np
    except ImportError:
        print("value: pip install h5py numpy")
        raise SystemExit(1)

    ap = argparse.ArgumentParser()
    ap.add_argument("--hdf5", type=Path, required=True)
    ap.add_argument("--meta_csv", type=Path, required=True, help="data trace_name data _npy_path")
    ap.add_argument("--data_dir", type=Path, required=True, help="meta data _npy_path data")
    ap.add_argument("--in_samples", type=int, default=8192)
    ap.add_argument(
        "--window-policy",
        choices=("center", "p_aligned"),
        default="center",
        help="center preserves the legacy behavior; p_aligned uses metadata P arrivals.",
    )
    ap.add_argument("--p-column", default="pnw_p_arrival_sample")
    ap.add_argument("--source-rate-column", default="pnw_sampling_rate_hz")
    ap.add_argument("--target-sampling-rate", type=float, default=50.0)
    ap.add_argument("--pre-p-seconds", type=float, default=10.0)
    ap.add_argument("--post-p-seconds", type=float, default=50.0)
    ap.add_argument("--max_rows", type=int, default=0, help="0 data")
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
                if args.window_policy == "p_aligned":
                    p_sample = float(row[args.p_column])
                    source_rate = float(row[args.source_rate_column])
                    arr = p_aligned_window(
                        raw,
                        p_arrival_sample=p_sample,
                        source_rate_hz=source_rate,
                        target_rate_hz=args.target_sampling_rate,
                        pre_p_seconds=args.pre_p_seconds,
                        post_p_seconds=args.post_p_seconds,
                    )
                    if arr.shape[1] != args.in_samples:
                        raise ValueError(
                            f"p_aligned window has {arr.shape[1]} samples, expected --in_samples {args.in_samples}; "
                            "make --in-samples equal to (pre_p_seconds + post_p_seconds) * target_sampling_rate"
                        )
                else:
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
