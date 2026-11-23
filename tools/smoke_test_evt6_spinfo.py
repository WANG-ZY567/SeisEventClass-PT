"""Smoke-test the EVT6 classifier with explicit S-P information."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from models.SeisMoLLM import SeisMoLLM_evt6_spinfo  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summaries = {}
    for split in ("train", "val", "test"):
        path = args.data / f"meta_evt6_{split}.csv"
        frame = pd.read_csv(path, low_memory=False)
        p = pd.to_numeric(frame.get("p_pick"), errors="coerce")
        s = pd.to_numeric(frame.get("s_pick"), errors="coerce")
        valid = p.notna() & s.notna() & (p >= 0) & (s > p) & (s < 8192)
        interval = (s[valid] - p[valid]) / 50.0
        summaries[split] = {
            "records": int(len(frame)),
            "valid_s": int(valid.sum()),
            "missing_or_invalid_s": int((~valid).sum()),
            "sp_seconds_min": float(interval.min()) if len(interval) else None,
            "sp_seconds_median": float(interval.median()) if len(interval) else None,
            "sp_seconds_max": float(interval.max()) if len(interval) else None,
        }

    device = torch.device(args.device)
    model = SeisMoLLM_evt6_spinfo(
        in_channels=3,
        in_samples=8192,
        llm_start_layer=0,
        llm_layers=3,
        d_model=768,
        llm_pretrain=True,
        llm_freeze=True,
    ).to(device)
    model.eval()
    waveform = torch.randn(2, 3, 8192, device=device)
    sp_info = torch.tensor([[0.08, 1.0], [0.0, 0.0]], device=device)
    with torch.no_grad():
        output = model((waveform, sp_info))
    if output.shape != (2, 6):
        raise RuntimeError(f"unexpected output shape: {tuple(output.shape)}")
    if not torch.isfinite(output).all():
        raise RuntimeError("model output contains NaN/Inf")

    print("S-P metadata audit:")
    for split, values in summaries.items():
        print(split, values)
    print(
        f"Smoke test passed on {device}: output={tuple(output.shape)}, "
        f"probability_sums={output.sum(dim=1).cpu().tolist()}"
    )


if __name__ == "__main__":
    main()
