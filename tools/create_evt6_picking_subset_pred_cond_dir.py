#!/usr/bin/env python3
"""
根据 evt6 路由分类器，对 picking subset 的 meta 写入伪标签：
  - 生成 pred column（如 pred_evt6_hier_sp / pred_evt6_multihead_w01）
  - 保留 waves_non/ 目录结构
  - 输出到一个新的 picking subset 目录（避免覆盖原数据）

该脚本用于“Predicted-conditioned phase picking”实验：
训练 dpk_cond 时，evt6_cond 来自该 pred column。
"""

from __future__ import annotations

import sys
import argparse
import json
import os
import shutil
from typing import Dict, List, Tuple

 # When executed as `python tools/<script>.py`, sys.path[0] becomes `tools/`,
 # so sibling imports like `import config` fail unless we add project root.
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import pandas as pd
import torch
from torch.utils.data import DataLoader

from config import Config
from models import create_model, load_checkpoint
from training.preprocess import SeismicDataset
from utils import logger as global_logger


def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def _get_evt6_output(outputs, model_labels: List[str]) -> torch.Tensor:
    """
    outputs: tensor or tuple/list of tensors.
    model_labels: config.labels list that aligns with outputs order.
    """
    if isinstance(outputs, (tuple, list)):
        if "evt6" not in model_labels:
            raise KeyError(f"router model labels缺少 evt6: {model_labels}")
        idx = model_labels.index("evt6")
        return outputs[idx]
    # Some models might output only evt6 prob.
    return outputs


def _extract_trace_uid(meta_jsons: List[str]) -> List[str]:
    trace_uids = []
    for mj in meta_jsons:
        if isinstance(mj, str):
            d = json.loads(mj)
        else:
            # In case of unexpected type, try to parse as string.
            d = json.loads(str(mj))
        trace_uids.append(str(d.get("trace_uid")))
    return trace_uids


def infer_split_to_pred_map(
    *,
    model,
    device: torch.device,
    args_for_ds: argparse.Namespace,
    dataset_name: str,
    data_dir: str,
    split: str,
    router_model_name: str,
    batch_size: int,
    workers: int,
) -> Dict[str, int]:
    """
    Run router classifier inference on one split (train/val/test)
    and return {trace_uid -> pred_evt6}.
    """
    # Configure dataset wrapper inputs/labels/eval for the router model.
    model_inputs, model_labels, model_tasks = Config.get_model_config_(
        router_model_name, "inputs", "labels", "eval"
    )

    # Build SeismicDataset for the desired split.
    ds = SeismicDataset(
        args=args_for_ds,
        input_names=model_inputs,
        label_names=model_labels,
        task_names=model_tasks,
        mode=split,
    )

    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=getattr(args_for_ds, "pin_memory", True),
        num_workers=workers,
    )

    # Which output index corresponds to evt6?
    labels_list = list(model_labels) if isinstance(model_labels, list) else [str(model_labels)]

    pred_map: Dict[str, int] = {}
    model.eval()
    with torch.no_grad():
        for x, _loss_targets, _metrics_targets, meta_data_jsons in loader:
            if isinstance(x, (list, tuple)):
                x = [xi.to(device) for xi in x]
                # router models here don't expect conditional inputs, so x should be a waveform tensor.
                x_wave = x[0]
            else:
                x_wave = x.to(device)

            outputs = model(x_wave)
            y_evt6 = _get_evt6_output(outputs, labels_list)  # [B,6] probability
            pred_evt6 = torch.argmax(y_evt6, dim=-1).detach().cpu().tolist()

            # meta_data_jsons: list[str] with batch length == B
            trace_uids = _extract_trace_uid(meta_data_jsons)
            if len(trace_uids) != len(pred_evt6):
                raise RuntimeError(
                    f"trace_uid/pred length mismatch split={split}: {len(trace_uids)} vs {len(pred_evt6)}"
                )
            for tu, p in zip(trace_uids, pred_evt6):
                pred_map[tu] = int(p)

    return pred_map


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src_dir", required=True, help="picking subset dir (must contain meta_evt6_{train,val,test}.csv)")
    ap.add_argument("--dst_dir", required=True, help="output dir (new picking subset with pred column)")
    ap.add_argument("--router-model-name", required=True, help="SeisMoLLM_evt6_hier_sp or SeisMoLLM_evt6_multihead_w01")
    ap.add_argument("--router-checkpoint", required=True, help="checkpoint path for router model")
    ap.add_argument("--pred-col", required=True, help="new column name, e.g. pred_evt6_hier_sp")
    ap.add_argument("--device", default="cuda:0", type=str)
    ap.add_argument("--batch-size", default=128, type=int)
    ap.add_argument("--workers", default=8, type=int)
    ap.add_argument("--in-samples", default=8192, type=int)
    ap.add_argument("--norm-mode", default="max", type=str)
    ap.add_argument("--pin-memory", default=True, type=bool)
    ap.add_argument("--seed", default=0, type=int)
    args = ap.parse_args()

    src_dir = os.path.abspath(args.src_dir)
    dst_dir = os.path.abspath(args.dst_dir)

    _ensure_dir(dst_dir)

    # Initialize logger for dataset adapters (they call utils.logger.info).
    # utils.logger requires set_logdir before create/set logger.
    log_dir = os.path.join(dst_dir, "_pred_cond_logs")
    try:
        global_logger.set_logdir(log_dir)
    except Exception:
        # If already initialized in this process, ignore.
        pass
    try:
        global_logger.set_logger("global")
    except Exception:
        # Best-effort: some callers may already have an active logger.
        pass

    # Copy/link waves_non (required by evt6 dataset meta _npy_path paths).
    waves_non_src = os.path.join(src_dir, "waves_non")
    waves_non_dst = os.path.join(dst_dir, "waves_non")
    if os.path.exists(waves_non_dst):
        pass
    elif os.path.exists(waves_non_src):
        os.symlink(waves_non_src, waves_non_dst)

    # Prepare dataset args for SeismicDataset wrapper.
    # Disable augmentation for deterministic pseudo labels.
    args_for_ds = argparse.Namespace(
        seed=int(args.seed),
        augmentation=False,
        dataset_name="diting2_evt6",
        data_split=True,
        shuffle=False,
        train_size=0.8,
        val_size=0.1,
        data=src_dir,
        pin_memory=bool(args.pin_memory),
        in_samples=int(args.in_samples),
        min_snr=-float("inf"),
        coda_ratio=2.0,
        p_position_ratio=-1,
        norm_mode=args.norm_mode,
        add_event_rate=0.0,
        add_noise_rate=0.0,
        add_gap_rate=0.0,
        drop_channel_rate=0.0,
        scale_amplitude_rate=0.0,
        pre_emphasis_rate=0.0,
        pre_emphasis_ratio=0.97,
        max_event_num=1,
        generate_noise_rate=0.0,
        shift_event_rate=0.0,
        mask_percent=0,
        noise_percent=0,
        min_event_gap=0.5,
        label_shape="gaussian",
        label_width=0.0,
    )

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Load router model checkpoint.
    in_channels = Config.get_num_inchannels(model_name=args.router_model_name)
    model = create_model(model_name=args.router_model_name, in_channels=in_channels, in_samples=args.in_samples)
    ckpt = load_checkpoint(
        os.path.abspath(args.router_checkpoint),
        device=device,
        dist_mode=False,
        compile_mode=False,
        resume=False,
        strict=True,
    )
    if ckpt is not None and "model_dict" in ckpt:
        model.load_state_dict(ckpt["model_dict"], strict=True)
    model = model.to(device)

    # Predict & merge for each split.
    splits = ["train", "val", "test"]
    for split in splits:
        pred_map = infer_split_to_pred_map(
            model=model,
            device=device,
            args_for_ds=args_for_ds,
            dataset_name="diting2_evt6",
            data_dir=src_dir,
            split=split,
            router_model_name=args.router_model_name,
            batch_size=int(args.batch_size),
            workers=int(args.workers),
        )

        meta_path = os.path.join(src_dir, f"meta_evt6_{split}.csv")
        if not os.path.exists(meta_path):
            raise FileNotFoundError(meta_path)
        df = pd.read_csv(meta_path, low_memory=False)
        if "trace_uid" not in df.columns:
            raise KeyError(f"{meta_path} 缺少 trace_uid 列，无法写入 {args.pred_col}")

        df[args.pred_col] = df["trace_uid"].map(pred_map).astype("Int64")
        missing = int(df[args.pred_col].isna().sum())
        if missing:
            raise RuntimeError(f"pred_map 未覆盖 {missing} 条样本（split={split}）。请检查 join key=trace_uid 是否一致。")

        out_meta_path = os.path.join(dst_dir, f"meta_evt6_{split}.csv")
        df.to_csv(out_meta_path, index=False)

    # Copy selection record if exists (for traceability).
    sel = os.path.join(src_dir, "selection_record_evt6_picking_subset.json")
    if os.path.exists(sel):
        shutil.copy2(sel, os.path.join(dst_dir, os.path.basename(sel)))

    print(f"[OK] wrote pred-conditioned picking subset dir: {dst_dir}")


if __name__ == "__main__":
    main()

