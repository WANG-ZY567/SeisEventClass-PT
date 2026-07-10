import argparse
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import pandas as pd


@dataclass(frozen=True)
class TTAConfig:
    name: str
    tta_times: int
    shift: int
    noise: float
    scale: float
    drop_ch_p: float = 0.0
    preemph_p: float = 0.0
    preemph_alpha: float = 0.97


def _parse_args():
    p = argparse.ArgumentParser(description="Quick EVT6 TTA sweep (no training).")
    p.add_argument("--ckpt", type=str, required=True, help="checkpoint .pth path")
    p.add_argument(
        "--out_dir",
        type=str,
        required=True,
        help="directory to store sweep logs/results (best report will be here)",
    )
    p.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="device string (default: cuda:0). Note: under DDP/srun this may be ignored.",
    )
    p.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="test batch size (default: 32)",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=0,
        help="dataloader workers (default: 0)",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=2,
        help="random seed for TTA randomness (default: 2)",
    )
    p.add_argument(
        "--dataset_dir",
        type=str,
        default="/path/to/diting2_evt6",
        help="dataset path (default: diting2_evt6_paperalign)",
    )
    p.add_argument(
        "--dataset_name",
        type=str,
        default="diting2_evt6",
        help="dataset name (default: diting2_evt6)",
    )
    p.add_argument(
        "--model_name",
        type=str,
        default="SeisMoLLM_evt6",
        help="model name (default: SeisMoLLM_evt6)",
    )
    p.add_argument(
        "--norm_mode",
        type=str,
        default="max",
        help="norm_mode used in pipeline (default: max)",
    )
    p.add_argument(
        "--augmentation",
        type=str,
        default="true",
        help="augmentation flag passed to main.py (default: true)",
    )
    p.add_argument(
        "--max_trials",
        type=int,
        default=8,
        help="max number of TTA configs to try from preset list (default: 8)",
    )
    return p.parse_args()


def _bool_str(x: str) -> str:
    return "true" if str(x).strip().lower() in ("1", "true", "t", "yes", "y") else "false"


def _latest_new_csv(ckpt_dir: Path, since_ts: float) -> Optional[Path]:
    candidates = sorted(ckpt_dir.glob("test_results_diting2_evt6_test_new*.csv"), key=lambda p: p.stat().st_mtime)
    candidates = [p for p in candidates if p.stat().st_mtime >= since_ts - 1.0]
    return candidates[-1] if candidates else None


def _accuracy(csv_path: Path) -> float:
    df = pd.read_csv(csv_path)
    return float((df["pred_evt6"].astype(int) == df["tgt_evt6"].astype(int)).mean())


def main():
    args = _parse_args()
    ckpt = Path(args.ckpt)
    if not ckpt.exists():
        raise FileNotFoundError(str(ckpt))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt_dir = ckpt.parent.parent if ckpt.parent.name == "checkpoints" else ckpt.parent

    # Preset small grid around the currently-best-ish setting.
    presets: List[TTAConfig] = [
        TTAConfig("base", 8, 64, 0.010, 0.050),
        TTAConfig("shift128", 8, 128, 0.010, 0.050),
        TTAConfig("shift64_noise005", 8, 64, 0.005, 0.050),
        TTAConfig("shift64_scale03", 8, 64, 0.010, 0.030),
        TTAConfig("shift96_noise008_scale04", 8, 96, 0.008, 0.040),
        TTAConfig("shift96_noise005_scale03", 8, 96, 0.005, 0.030),
        # a bit riskier, sometimes helps:
        TTAConfig("dropch005", 8, 64, 0.010, 0.050, drop_ch_p=0.05),
        TTAConfig("preemph05", 8, 64, 0.010, 0.050, preemph_p=0.5, preemph_alpha=0.97),
        # stronger averaging (slower but can nudge over the line):
        TTAConfig("times16", 16, 64, 0.010, 0.050),
        TTAConfig("times16_noise005", 16, 64, 0.005, 0.050),
        TTAConfig("times16_scale03", 16, 64, 0.010, 0.030),
        TTAConfig("times16_shift96", 16, 96, 0.010, 0.050),
        # slightly more aggressive noise/scale:
        TTAConfig("times16_noise015_scale07", 16, 64, 0.015, 0.070),
        # very strong (slow):
        TTAConfig("times32", 32, 64, 0.010, 0.050),
    ][: int(args.max_trials)]

    rows = []
    best = None

    for cfg in presets:
        ts = time.time()
        log_path = out_dir / f"tta_{cfg.name}.log"
        cmd = [
            "python",
            "main.py",
            "--mode",
            "test",
            "--model-name",
            args.model_name,
            "--device",
            args.device,
            "--seed",
            str(int(args.seed)),
            "--checkpoint",
            str(ckpt),
            "--checkpoint-strict",
            "true",
            "--data",
            args.dataset_dir,
            "--dataset-name",
            args.dataset_name,
            "--shuffle",
            "false",
            "--workers",
            str(int(args.workers)),
            "--in-samples",
            "8192",
            "--batch-size",
            str(int(args.batch_size)),
            "--augmentation",
            _bool_str(args.augmentation),
            "--norm-mode",
            str(args.norm_mode),
            "--label-width",
            "0",
            "--label-shape",
            "2",
            "--save-test-results",
            "true",
            "--save-test-probs",
            "true",
            "--tta-times",
            str(int(cfg.tta_times)),
            "--tta-shift-samples",
            str(int(cfg.shift)),
            "--tta-noise-std",
            str(float(cfg.noise)),
            "--tta-scale",
            str(float(cfg.scale)),
            "--tta-drop-channel-p",
            str(float(cfg.drop_ch_p)),
            "--tta-preemph-p",
            str(float(cfg.preemph_p)),
            "--tta-preemph-alpha",
            str(float(cfg.preemph_alpha)),
            "--log-base",
            str(out_dir),
        ]

        with open(log_path, "w", encoding="utf-8") as f:
            f.write("[CMD] " + " ".join(cmd) + "\n")
            f.flush()
            subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, check=True)

        csv_path = _latest_new_csv(ckpt_dir=ckpt_dir, since_ts=ts)
        if csv_path is None:
            rows.append(
                {
                    "name": cfg.name,
                    "acc": None,
                    "csv": "",
                    "note": "no new csv found (logdir may differ)",
                }
            )
            continue

        acc = _accuracy(csv_path)
        rows.append(
            {
                "name": cfg.name,
                "acc": acc,
                "csv": str(csv_path),
                "tta_times": cfg.tta_times,
                "shift": cfg.shift,
                "noise": cfg.noise,
                "scale": cfg.scale,
                "drop_ch_p": cfg.drop_ch_p,
                "preemph_p": cfg.preemph_p,
                "preemph_alpha": cfg.preemph_alpha,
            }
        )

        if best is None or (acc is not None and acc > best["acc"]):
            best = rows[-1]

        print(f"[{cfg.name}] acc={acc:.4f} csv={csv_path}")

    df = pd.DataFrame(rows).sort_values(by=["acc"], ascending=False, na_position="last")
    out_csv = out_dir / "tta_sweep_summary.csv"
    df.to_csv(out_csv, index=False)
    print(f"[OK] wrote sweep summary: {out_csv}")

    if best and best.get("csv"):
        out_md = out_dir / "EVT6_TEST_REPORT_TTA_SWEEP_BEST.md"
        subprocess.run(
            [
                "python",
                "tools/report_evt6_results.py",
                "--results_csv",
                best["csv"],
                "--out_md",
                str(out_md),
            ],
            check=True,
        )
        print(f"[OK] best={best['name']} acc={best['acc']:.4f}")
        print(f"[OK] best report: {out_md}")


if __name__ == "__main__":
    main()


