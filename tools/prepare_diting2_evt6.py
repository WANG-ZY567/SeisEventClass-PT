#!/usr/bin/env python3
"""
DiTing2.0 事件类型（evtype）6类分类数据准备脚本

目标：
- 生成一个 meta_evt6.csv，用于 6 类分类任务（evt6: eq/ep/ss/sp/se/ot -> 0..5）
- 自然地震部分：复用你已经生成的 Full5 波形（diting2_seismollm_full5/waves/*.npy），避免重新预处理百万级数据
- 非自然地震部分：从 non_natural 的 HDF5+JSON 裁窗生成 waves，并写入 meta

说明：
- 该脚本不会去处理 natural JSON 全量（1,089,920），否则磁盘会爆炸。
- 默认：自然地震取 Full5 的 209,252 条；非自然取全部 15,375 条（或用 --max_non 限制）。

输出：
- out_dir/meta_evt6.csv
- out_dir/waves_non/*.npy（非自然地震的波形）
"""

import argparse
import json
import os
from pathlib import Path
from collections import Counter

import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm


# 说明：
# - 你提到的“CO(坍塌)”在一些论文里用 `co` 表示，但在 DiTing2.0 的 evtype 编码里并没有 `co`，
#   而是用 `ss`（其样本量在 non-natural JSON 中恰好为 5311，与很多论文的 CO=5311 对齐）。
# - 因此这里把 `co` 作为 `ss` 的别名，统一映射到同一个类别 id=2。
EVT6_MAP = {
    "eq": 0,
    "ep": 1,
    "co": 2,  # alias of collapse
    "ss": 2,  # collapse (paper's CO)
    "sp": 3,
    "se": 4,
    "ot": 5,
}


def norm_evtype(x) -> str:
    if x is None:
        return ""
    s = str(x).strip().lower()
    # 去掉可能的前导空格/奇怪字符
    s = s.replace("\u3000", "").strip()
    return s


def crop_with_anchor(x_c_l, x, L_out=8192):
    """
    以指定位置为锚点裁剪波形，不足部分用 0 填充。
    x: (3, L)
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


def main():
    ap = argparse.ArgumentParser(description="Prepare DiTing2.0 evt6 classification dataset")
    ap.add_argument("--out_dir", required=True, help="输出目录（写 meta_evt6.csv 与 waves_non/）")
    ap.add_argument(
        "--add_xapp_meta",
        action="store_true",
        help=(
            "在 meta_evt6*.csv 中额外写入跨应用实验所需字段："
            "event_id_raw/station_id_raw/src_domain/event_uid/station_uid/trace_uid，"
            "以及 picking/震源相关字段（尽可能从 full5 meta 或 DiTing JSON 补齐）。"
        ),
    )

    # natural: reuse existing full5
    ap.add_argument(
        "--natural_full5_dir",
        default="/path/to/diting2_preprocessed",
        help="已生成的 Full5 目录（包含 meta_full5.csv 与 waves/）",
    )
    ap.add_argument(
        "--natural_json",
        default="/path/to/CENC_DiTingv2_natural_earthquake.json",
        help="natural JSON（用于提供 evtype）",
    )
    ap.add_argument("--max_nat", type=int, default=0, help="最多使用多少条 natural（0=全用 full5）")
    ap.add_argument("--max_eq", type=int, default=0, help="natural 中最多保留多少条 eq（0=不限制，用于缓解类别不均衡）")

    # non-natural: preprocess
    ap.add_argument(
        "--non_h5",
        default="/path/to/CENC_DiTingv2_non_natural_earthquake.hdf5",
        help="non-natural HDF5",
    )
    ap.add_argument(
        "--non_json",
        default="/path/to/CENC_DiTingv2_non_natural_earthquake.json",
        help="non-natural JSON（包含 evtype）",
    )
    ap.add_argument("--max_non", type=int, default=0, help="最多处理多少条 non-natural（0=全部）")

    # crop settings
    ap.add_argument("--in_samples", type=int, default=8192)
    ap.add_argument("--pre", type=int, default=2000)

    ap.add_argument("--meta_csv", default="meta_evt6.csv")

    # balance settings (operate on meta only; will NOT duplicate waveform files)
    ap.add_argument(
        "--balance",
        default="none",
        choices=["none", "down", "up", "hybrid"],
        help=(
            "类别均衡策略："
            "none=不处理；down=对多类下采样到 target；"
            "up=对少类过采样到 target；"
            "hybrid=多类下采样+少类过采样到同一 target。"
        ),
    )
    ap.add_argument(
        "--target_per_class",
        type=int,
        default=0,
        help=(
            "均衡目标：每个类别最终样本数。0=自动（取 6 类计数的中位数）。"
            "注意：se 类通常最少，若 target 很大将导致大量重复采样。"
        ),
    )
    ap.add_argument("--seed", type=int, default=0, help="抽样随机种子")

    # strict split settings (for fair comparison with other papers that use fixed per-class counts)
    ap.add_argument(
        "--split_mode",
        default="ratio",
        choices=["ratio", "strict"],
        help=(
            "数据划分方式："
            "ratio=保持现状（由 Dataset 内部 shuffle 后按 0.8/0.1/0.1 切分）；"
            "strict=严格按每个类别固定数量划分，并输出 meta_evt6_{train,val,test}.csv。"
        ),
    )
    ap.add_argument(
        "--train_per_class",
        type=int,
        default=0,
        help="strict 模式：每类训练样本数（0=自动=target_per_class - val_per_class - test_per_class）",
    )
    ap.add_argument(
        "--val_per_class",
        type=int,
        default=300,
        help="strict 模式：每类验证样本数（默认 300，便于对齐常见论文的固定验证规模）",
    )
    ap.add_argument(
        "--test_per_class",
        type=int,
        default=300,
        help="strict 模式：每类测试样本数（默认 300；本项目训练/测试流程需要单独 test 集）",
    )
    ap.add_argument(
        "--test_from_val",
        action="store_true",
        help=(
            "对齐某些论文口径：没有独立测试集，把验证集当测试集。"
            "启用后会写出 meta_evt6_test.csv == meta_evt6_val.csv，并在记录文件中注明。"
        ),
    )

    # paper-aligned quota mode (no upsampling for minority classes)
    ap.add_argument(
        "--quota_mode",
        default="none",
        choices=["none", "paper_eq_ep_co"],
        help=(
            "配额模式："
            "none=不做配额；"
            "paper_eq_ep_co=对齐论文总量：EQ=7078, EP=5613, CO=5311(对应 ss)，"
            "其它类(sp/se/ot)不够则全用，不进行重复过采样。"
        ),
    )
    ap.add_argument("--quota_seed", type=int, default=100, help="配额抽样随机种子（与 --seed 分开，便于记录）")

    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    waves_non = out_dir / "waves_non"
    waves_non.mkdir(parents=True, exist_ok=True)

    # --------- load natural full5 meta & natural json (for evtype)
    nat_dir = Path(args.natural_full5_dir)
    nat_meta_csv = nat_dir / "meta_full5.csv"
    if not nat_meta_csv.exists():
        raise FileNotFoundError(f"natural meta not found: {nat_meta_csv}")
    nat_waves_dir = nat_dir / "waves"
    if not nat_waves_dir.exists():
        raise FileNotFoundError(f"natural waves not found: {nat_waves_dir}")

    print(f"[NAT] reading full5 meta: {nat_meta_csv}")
    nat_usecols = ["key", "part", "_npy_path"]
    if bool(args.add_xapp_meta):
        # Prefer full5-aligned picking/meta fields for natural events (they match cropped npy window).
        nat_usecols += [
            "net",
            "sta_id",
            "p_pick",
            "s_pick",
            "baz",
            "dis",
            "evmag",
            "mag_type",
            "P_residual",
            "S_residual",
        ]
    # Some columns may be absent in certain full5 variants; read defensively.
    nat_df = pd.read_csv(nat_meta_csv, low_memory=False)
    nat_df = nat_df[[c for c in nat_usecols if c in nat_df.columns]].copy()

    if args.max_nat and args.max_nat > 0:
        nat_df = nat_df.iloc[: args.max_nat].copy()

    print(f"[NAT] reading json for evtype: {args.natural_json}")
    with open(args.natural_json, "r", encoding="utf-8") as f:
        nat_meta = json.load(f)

    # join evtype
    def get_evtype_from_json(k: str) -> str:
        v = nat_meta.get(str(k))
        if not isinstance(v, dict):
            return ""
        return norm_evtype(v.get("evtype"))

    # 先提取 raw evtype，再规范化成最终 `_evtype`
    nat_df["_evtype_raw"] = nat_df["key"].astype(str).apply(get_evtype_from_json)
    nat_df["_evtype"] = nat_df["_evtype_raw"].apply(norm_evtype)
    # map to evt6 (unknown -> ot)
    nat_df["_evt6"] = nat_df["_evtype"].apply(lambda x: EVT6_MAP.get(x, EVT6_MAP["ot"]))
    nat_df["_src"] = "nat"

    # limit eq if asked
    if args.max_eq and args.max_eq > 0:
        eq_mask = nat_df["_evt6"] == EVT6_MAP["eq"]
        keep_eq = nat_df[eq_mask].iloc[: args.max_eq]
        keep_other = nat_df[~eq_mask]
        nat_df = pd.concat([keep_eq, keep_other], axis=0).reset_index(drop=True)

    # make abs npy path for natural (avoid copying waves)
    def nat_abs(p):
        p = str(p).strip().replace("\\", "/")
        if os.path.isabs(p):
            return p
        return str(nat_dir / p)

    nat_df["_npy_path"] = nat_df["_npy_path"].apply(nat_abs)

    if bool(args.add_xapp_meta):
        # Parse event/station from raw full5 key: "<event>_<station>"
        # Note: downstream should use event_uid to avoid nat/non collisions.
        def _split_key_event_station(k: str):
            s = str(k)
            if "_" not in s:
                return "", ""
            a, b = s.split("_", 1)
            return a, b

        ev_st = nat_df["key"].astype(str).apply(_split_key_event_station)
        nat_df["event_id_raw"] = ev_st.apply(lambda t: t[0])
        nat_df["station_id_raw"] = ev_st.apply(lambda t: t[1])
        nat_df["src_domain"] = "nat"
        nat_df["event_uid"] = nat_df["src_domain"] + "_" + nat_df["event_id_raw"].astype(str)
        nat_df["station_uid"] = nat_df["src_domain"] + "_" + nat_df["station_id_raw"].astype(str)
        nat_df["trace_uid"] = (
            nat_df["src_domain"]
            + "_"
            + nat_df["event_id_raw"].astype(str)
            + "_"
            + nat_df["station_id_raw"].astype(str)
            + "_"
            + nat_df["part"].astype(str)
        )
        # Full5-aligned picks (in cropped window): p_pick/s_pick
        if "p_pick" in nat_df.columns:
            nat_df["Pg"] = nat_df["p_pick"]
        if "s_pick" in nat_df.columns:
            nat_df["Sg"] = nat_df["s_pick"]
        if "P_residual" in nat_df.columns:
            nat_df["Pg_res"] = nat_df["P_residual"]
        if "S_residual" in nat_df.columns:
            nat_df["Sg_res"] = nat_df["S_residual"]
        if "baz" in nat_df.columns:
            nat_df["Pg_azi"] = nat_df["baz"]
        if "dis" in nat_df.columns:
            nat_df["Pg_dist"] = nat_df["dis"]
        if "evmag" in nat_df.columns:
            nat_df["mag"] = nat_df["evmag"]
        if "mag_type" in nat_df.columns:
            nat_df["magtype"] = nat_df["mag_type"]

    # --------- non-natural preprocess
    print(f"[NON] reading json: {args.non_json}")
    with open(args.non_json, "r", encoding="utf-8") as f:
        non_meta = json.load(f)
    non_keys = list(non_meta.keys())
    if args.max_non and args.max_non > 0:
        non_keys = non_keys[: args.max_non]
        print(f"[NON] max_non={args.max_non}, using {len(non_keys)}")

    non_rows = []
    evt_cnt = Counter()
    kept = 0
    skipped_no_pg = 0

    print(f"[NON] processing h5: {args.non_h5}")
    with h5py.File(args.non_h5, "r") as f:
        for rid in tqdm(non_keys, desc="non_natural"):
            rid = str(rid)
            meta = non_meta.get(rid, {})
            ev = norm_evtype(meta.get("evtype"))
            cls = EVT6_MAP.get(ev, EVT6_MAP["ot"])

            Pg = meta.get("Pg", None)
            try:
                Pg = int(float(Pg)) if Pg not in [None, ""] else None
            except Exception:
                Pg = None

            if Pg is None:
                skipped_no_pg += 1
                continue

            if rid not in f:
                continue
            x = np.asarray(f[rid][()])
            if x.ndim != 2:
                continue
            if x.shape[0] == 3:
                x = x
            elif x.shape[1] == 3:
                x = x.T
            else:
                continue
            x = x.astype(np.float32)

            c_l = Pg - args.pre
            x_crop, c_l_actual = crop_with_anchor(c_l, x, L_out=args.in_samples)

            # key needs to be unique across src
            key_out = f"non_{rid}"
            part = 0
            npy_name = f"{key_out}_{part}.npy"
            npy_path = waves_non / npy_name
            np.save(npy_path, x_crop)

            row = {
                "key": key_out,
                "part": part,
                "_src": "non",
                "_rid": rid,
                "_evtype": ev,
                "_evt6": cls,
                "_npy_path": str(npy_path),
            }

            if bool(args.add_xapp_meta):
                # Keep raw JSON fields (as-is) and compute full5-aligned picks in cropped window.
                def _to_float(x):
                    try:
                        return float(x)
                    except Exception:
                        return np.nan

                Sg = meta.get("Sg", None)
                row.update(
                    {
                        "src_domain": "non",
                        "event_id_raw": str(rid).split("_", 1)[0] if "_" in str(rid) else str(rid),
                        "station_id_raw": str(rid).split("_", 1)[1] if "_" in str(rid) else "",
                        "event_uid": "non_" + (str(rid).split("_", 1)[0] if "_" in str(rid) else str(rid)),
                        "station_uid": "non_" + (str(rid).split("_", 1)[1] if "_" in str(rid) else ""),
                        "trace_uid": f"non_{rid}_{part}",
                        # JSON raw values
                        "Pg": _to_float(meta.get("Pg", None)),
                        "Sg": _to_float(Sg),
                        "Pg_res": _to_float(meta.get("Pg_res", None)),
                        "Sg_res": _to_float(meta.get("Sg_res", None)),
                        "Pg_azi": _to_float(meta.get("Pg_azi", None)),
                        "Sg_azi": _to_float(meta.get("Sg_azi", None)),
                        "Pg_dist": _to_float(meta.get("Pg_dist", None)),
                        "Sg_dist": _to_float(meta.get("Sg_dist", None)),
                        "se_time": _to_float(meta.get("se_time", None)),
                        "sn_time": _to_float(meta.get("sn_time", None)),
                        "se_mag": _to_float(meta.get("se_mag", None)),
                        "sn_mag": _to_float(meta.get("sn_mag", None)),
                        "mag": _to_float(meta.get("mag", None)),
                        "magtype": str(meta.get("magtype", "")) if meta.get("magtype", "") is not None else "",
                    }
                )
                # cropped-window aligned picks:
                # Pg is the crop anchor => p_pick is always `pre` in the cropped window
                row["p_pick"] = float(args.pre)
                # s_pick in cropped window if Sg exists:
                try:
                    sg_f = float(Sg) if Sg not in [None, ""] else np.nan
                except Exception:
                    sg_f = np.nan
                if np.isfinite(sg_f) and Pg is not None:
                    row["s_pick"] = float(sg_f - float(Pg) + float(args.pre))
                else:
                    row["s_pick"] = np.nan

            non_rows.append(row)
            evt_cnt[ev] += 1
            kept += 1

    print(f"[NON] kept={kept}, skipped_no_pg={skipped_no_pg}")

    non_df = pd.DataFrame(non_rows)

    # --------- merge
    # natural keys may collide with non_ prefix; keep natural as nat_{key}
    nat_df = nat_df.copy()
    nat_df["key"] = nat_df["key"].astype(str).apply(lambda k: f"nat_{k}")
    nat_df["_rid"] = nat_df["key"].astype(str).str.replace("^nat_", "", regex=True)
    # 删除中间字段，避免列名冲突
    if "_evtype_raw" in nat_df.columns:
        # 保留 `_evtype`（规范化后的）
        pass

    base_cols = ["key", "part", "_src", "_rid", "_evtype", "_evt6", "_npy_path"]
    extra_cols = []
    if bool(args.add_xapp_meta):
        extra_cols = [
            "src_domain",
            "event_id_raw",
            "station_id_raw",
            "event_uid",
            "station_uid",
            "trace_uid",
            # picking/geo/mag/time (best-effort; may be missing in nat_df depending on full5 meta)
            "p_pick",
            "s_pick",
            "Pg",
            "Sg",
            "Pg_res",
            "Sg_res",
            "Pg_azi",
            "Sg_azi",
            "Pg_dist",
            "Sg_dist",
            "se_time",
            "sn_time",
            "se_mag",
            "sn_mag",
            "mag",
            "magtype",
            # station/network if available
            "net",
            "sta_id",
        ]
    nat_cols = [c for c in base_cols + extra_cols if c in nat_df.columns]
    non_cols = [c for c in base_cols + extra_cols if c in non_df.columns]
    merged = pd.concat(
        [
            nat_df[nat_cols],
            non_df[non_cols],
        ],
        axis=0,
    ).reset_index(drop=True)

    # --------- quota selection (optional, no upsampling)
    quota_mode = str(args.quota_mode).lower()
    quota_record = {
        "quota_mode": quota_mode,
        "quota_seed": int(args.quota_seed),
        "requested": {},
        "available_before": merged["_evt6"].value_counts().sort_index().to_dict(),
        "selected_total": {},
        "notes": [],
    }
    if quota_mode != "none":
        rngq = np.random.default_rng(int(args.quota_seed))
        # paper: CO(坍塌)=ss=class 2
        # request totals; other classes use all available (no downsample unless you set it later)
        requested = {0: 7078, 1: 5613, 2: 5311}
        quota_record["requested"] = requested

        selected_parts = []
        for cls in range(6):
            sub = merged[merged["_evt6"] == cls]
            n = len(sub)
            if cls in requested:
                need = int(requested[cls])
                if n < need:
                    quota_record["notes"].append(f"class {cls} need {need} but only {n} available; will use all (cannot strictly match paper)")
                    take = n
                else:
                    take = need
            else:
                # other classes: use all available (no upsampling, no forced downsampling)
                take = n

            if take <= 0:
                continue
            if take == n:
                selected_parts.append(sub)
            else:
                idx = sub.index.to_numpy()
                pick = rngq.choice(idx, size=take, replace=False)
                selected_parts.append(merged.loc[pick])

            quota_record["selected_total"][cls] = int(take)

        merged = pd.concat(selected_parts, axis=0).reset_index(drop=True)
        # shuffle after quota selection (membership fixed)
        merged = merged.sample(frac=1.0, random_state=int(args.quota_seed)).reset_index(drop=True)

    # --------- balance classes (optional)
    balance = str(args.balance).lower()
    if balance != "none":
        rng = np.random.default_rng(int(args.seed))
        counts = merged["_evt6"].value_counts().sort_index()
        cls_list = sorted(counts.index.tolist())
        # ensure 0..5 all appear in counts; if not, they can't be balanced without new data
        missing_cls = [c for c in range(6) if c not in set(cls_list)]
        if missing_cls:
            print(f"[WARN] 缺少类别 {missing_cls}，无法真正均衡；请检查数据源是否包含这些类。")

        if int(args.target_per_class) > 0:
            target = int(args.target_per_class)
        else:
            # auto target: median of existing class counts (robust to eq extreme)
            target = int(np.median(counts.values))
            target = max(target, 1)

        parts = []
        for cls in sorted(set(cls_list) | set(missing_cls)):
            sub = merged[merged["_evt6"] == cls]
            n = len(sub)
            if n == 0:
                continue

            if balance == "down":
                take = min(n, target)
                # sample without replacement
                idx = rng.choice(sub.index.to_numpy(), size=take, replace=False)
                parts.append(merged.loc[idx])
            elif balance == "up":
                take = target
                replace = n < take
                idx = rng.choice(sub.index.to_numpy(), size=take, replace=replace)
                parts.append(merged.loc[idx])
            elif balance == "hybrid":
                take = target
                replace = n < take
                idx = rng.choice(sub.index.to_numpy(), size=take, replace=replace)
                parts.append(merged.loc[idx])
            else:
                raise ValueError(f"Unknown balance mode: {balance}")

        if parts:
            merged = pd.concat(parts, axis=0).reset_index(drop=True)
            # shuffle after balancing
            merged = merged.sample(frac=1.0, random_state=int(args.seed)).reset_index(drop=True)

    out_csv = out_dir / args.meta_csv
    merged.to_csv(out_csv, index=False)

    # --------- strict split (optional)
    if str(args.split_mode).lower() == "strict":
        rng = np.random.default_rng(int(args.seed))

        val_n = int(args.val_per_class)
        test_n = int(args.test_per_class)
        if val_n < 0 or test_n < 0:
            raise ValueError("val_per_class/test_per_class must be >= 0")

        # train_per_class is optional:
        # - if provided (>0): try to use that many train samples per class when possible
        # - else: use all remaining samples after allocating val/test (per-class), i.e. "all-in" without dropping
        train_n = int(args.train_per_class) if int(args.train_per_class) > 0 else None

        parts = {"train": [], "val": [], "test": []}
        actual_split = {s: {} for s in ("train", "val", "test")}
        for cls in range(6):
            sub = merged[merged["_evt6"] == cls]
            n = len(sub)
            if n == 0:
                continue

            # 若该类样本足够：优先使用固定 val/test；train 用“剩余”或用户指定的 train_n（取其可行上限）
            if n >= (val_n + test_n + 1):
                v = val_n
                t = test_n
                remain = n - v - t
                tr = remain if train_n is None else min(train_n, remain)
            else:
                # 小类：按比例 0.1/0.1 切分（全用不丢，且尽量保留足够 train）
                v = int(round(0.1 * n))
                t = int(round(0.1 * n))
                # 至少留 1 个训练样本
                if v + t >= n:
                    v = max(0, n - 1)
                    t = 0
                tr = n - v - t
                # 若用户强制 train_n，则最多也只能用 tr（不丢样本时，train_n 只是上限）
                if train_n is not None:
                    tr = min(tr, train_n)
                    # 把多出来的“训练上限”样本留在 train 外会丢数据，违背“全用不丢”
                    # 因此这里对小类忽略 train_n，下方会用 tr=全剩余
                    tr = n - v - t

            idx = sub.index.to_numpy()
            rng.shuffle(idx)
            val_idx = idx[:v]
            test_idx = idx[v : v + t]
            train_idx = idx[v + t : v + t + tr]

            parts["train"].append(merged.loc[train_idx])
            parts["val"].append(merged.loc[val_idx])
            parts["test"].append(merged.loc[test_idx])
            actual_split["train"][cls] = int(len(train_idx))
            actual_split["val"][cls] = int(len(val_idx))
            actual_split["test"][cls] = int(len(test_idx))

        for split in ("train", "val", "test"):
            df = pd.concat(parts[split], axis=0).reset_index(drop=True)
            # shuffle within split for better training randomness (membership is fixed)
            df = df.sample(frac=1.0, random_state=int(args.seed)).reset_index(drop=True)
            df.to_csv(out_dir / f"meta_evt6_{split}.csv", index=False)

        if bool(args.test_from_val):
            # overwrite test with val for paper-aligned evaluation setting
            val_path = out_dir / "meta_evt6_val.csv"
            test_path = out_dir / "meta_evt6_test.csv"
            try:
                vdf = pd.read_csv(val_path, low_memory=False)
                vdf.to_csv(test_path, index=False)
            except Exception as e:
                print(f"[WARN] failed to set test_from_val: {e}")
        quota_record["strict_split"] = {
            "seed": int(args.seed),
            "val_per_class": int(args.val_per_class),
            "test_per_class": int(args.test_per_class),
            "train_per_class": int(args.train_per_class) if int(args.train_per_class) > 0 else None,
            "actual_per_class": actual_split,
            "test_from_val": bool(args.test_from_val),
        }

    # write record for reproducibility
    try:
        import json as _json
        rec_path = out_dir / "selection_record_evt6.json"
        # include code inputs for traceability
        quota_record["paths"] = {
            "natural_full5_dir": str(args.natural_full5_dir),
            "natural_json": str(args.natural_json),
            "non_h5": str(args.non_h5),
            "non_json": str(args.non_json),
        }
        quota_record["final_total"] = int(len(merged))
        quota_record["final_counts"] = merged["_evt6"].value_counts().sort_index().to_dict()
        with open(rec_path, "w", encoding="utf-8") as f:
            _json.dump(quota_record, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[WARN] failed to write selection record: {e}")

    print("=" * 60)
    print("evt6 数据集准备完成")
    print(f"meta: {out_csv}")
    print(f"rows: {len(merged)} (nat={len(nat_df)}, non={len(non_df)})")
    print("evt6 mapping:", EVT6_MAP)
    print("label distribution (evt6 id):")
    print(merged['_evt6'].value_counts().sort_index().to_string())
    if balance != "none":
        print(f"balance: {balance}, target_per_class={args.target_per_class or 'auto(median)'} , seed={args.seed}")
    if str(args.split_mode).lower() == "strict":
        print(f"split_mode: strict, per_class: train={train_n}, val={val_n}, test={test_n}")
    print("=" * 60)


if __name__ == "__main__":
    main()


