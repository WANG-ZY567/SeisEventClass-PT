#!/usr/bin/env python3
"""
Export learned global channel weights (Z/N/E) from a trained EVT6 channel-weighted model.

This script is read-only to your experiments: it does NOT run training/testing.
It simply loads a checkpoint and prints/saves the weights for paper reporting.

Usage example:
python tools/export_evt6_channel_weights.py \
  --checkpoint logs/<run>/checkpoints/model-xxx.pth \
  --model-name SeisMoLLM_evt6_cw \
  --out_csv  logs/<run>/channel_weights_zne.csv
"""

import argparse
import csv
import os

import sys
from pathlib import Path

import torch

# Ensure repo root is importable when running from tools/.
_ROOT = str(Path(__file__).resolve().parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from models import create_model
from config import Config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, help="Path to .pth checkpoint")
    ap.add_argument("--model-name", default="SeisMoLLM_evt6_cw", help="Model name (default: SeisMoLLM_evt6_cw)")
    ap.add_argument("--out_csv", default="", help="Optional: write weights to csv")
    args = ap.parse_args()

    ckpt_path = os.path.abspath(args.checkpoint)
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(ckpt_path)

    in_channels = Config.get_num_inchannels(model_name=args.model_name)
    # Use the repo's model factory (public API in models/__init__.py)
    model = create_model(model_name=args.model_name, in_channels=in_channels, in_samples=8192)

    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state, strict=False)
    model.eval()

    if not hasattr(model, "get_channel_weights"):
        raise RuntimeError(f"Model {args.model_name} does not expose get_channel_weights().")

    w = model.get_channel_weights()
    if not w:
        raise RuntimeError(f"Checkpoint loaded but no channel weights found. Is this a *_cw model?")

    print("[channel_weights]", w, flush=True)

    if args.out_csv:
        out_csv = os.path.abspath(args.out_csv)
        os.makedirs(os.path.dirname(out_csv), exist_ok=True)
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            wr = csv.DictWriter(f, fieldnames=list(w.keys()))
            wr.writeheader()
            wr.writerow(w)
        print(f"[OK] wrote: {out_csv}", flush=True)


if __name__ == "__main__":
    main()

