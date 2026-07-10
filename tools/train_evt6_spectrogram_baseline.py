#!/usr/bin/env python3
"""
EVT6 waveform vs spectrogram baseline (Z-only) for seismic cross-application analysis.

This is a minimal, paper-oriented baseline:
- Input: Z-only waveform (1D, length 8192) -> STFT magnitude (or log-magnitude) spectrogram
- Model: lightweight 2D CNN classifier
- Output: `test_results_diting2_evt6_test.csv` with pred/tgt (+ optional prob columns)

It is intentionally implemented as an independent script to avoid impacting mainline `main.py`.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


EVT6_ID2NAME: Dict[int, str] = {0: "eq", 1: "ep", 2: "co", 3: "sp", 4: "se", 5: "ot"}


def _seed_everything(seed: int):
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _is_missing(v: str) -> bool:
    if v is None:
        return True
    s = str(v).strip()
    return s == "" or s.lower() == "nan"


def _safe_float(v: str) -> float | None:
    if _is_missing(v):
        return None
    try:
        f = float(v)
        if not math.isfinite(f):
            return None
        return f
    except Exception:
        return None


class Evt6SpectrogramDataset(Dataset):
    """
    Reads meta csv rows with columns:
      - _npy_path: npy path to waveform (shape: (3, L))
      - _evt6: int label

    Returns:
      - spec: [1, F, T] float tensor
      - y: int64
      - meta: dict
    """

    def __init__(
        self,
        meta_csv: str,
        norm_mode: str = "max",
        in_samples: int = 8192,
        stft_n_fft: int = 256,
        stft_hop_length: int = 64,
        stft_win_length: int = 256,
        log_spec: bool = True,
        eps: float = 1e-6,
    ):
        self.meta_csv = os.path.abspath(meta_csv)
        if not os.path.exists(self.meta_csv):
            raise FileNotFoundError(self.meta_csv)

        self.norm_mode = str(norm_mode).strip().lower()
        if self.norm_mode not in ("max", "std", "none", ""):
            raise ValueError(f"unknown norm_mode={norm_mode}")
        if self.norm_mode in ("", "none"):
            self.norm_mode = "none"

        self.in_samples = int(in_samples)
        self.n_fft = int(stft_n_fft)
        self.hop_length = int(stft_hop_length)
        self.win_length = int(stft_win_length)
        self.log_spec = bool(log_spec)
        self.eps = float(eps)

        # load csv (streaming) into memory as list of dict to avoid pandas dependency
        with open(self.meta_csv, "r", newline="", encoding="utf-8") as f:
            r = csv.DictReader(f)
            self.rows = [row for row in r]
            self.cols = r.fieldnames or []

        need = {"_npy_path", "_evt6"}
        miss = need - set(self.cols)
        if miss:
            raise KeyError(f"meta missing columns: {sorted(miss)} in {self.meta_csv}")

        # filter missing paths
        keep = []
        for row in self.rows:
            p = str(row.get("_npy_path", "")).strip().replace("\\", "/")
            if p and os.path.exists(p):
                row["_npy_path"] = p
                keep.append(row)
        self.rows = keep

        # pre-build window
        self.window = torch.hann_window(self.win_length, periodic=True)

    def __len__(self):
        return int(len(self.rows))

    def _load_wave_z(self, path: str) -> np.ndarray:
        sleeps = [0.05, 0.1, 0.2, 0.5, 1.0, 2.0]
        last_e: Exception | None = None
        for s in sleeps:
            try:
                a = np.load(path).astype(np.float32, copy=False)
                if a.ndim != 2 or a.shape[0] < 1:
                    raise ValueError(f"bad npy shape {a.shape}")
                z = a[0]
                if z.shape[0] != self.in_samples:
                    # pad/crop defensively
                    if z.shape[0] < self.in_samples:
                        z = np.pad(z, (0, self.in_samples - z.shape[0]), mode="constant")
                    else:
                        z = z[: self.in_samples]
                z = np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
                # normalize per-sample
                if self.norm_mode == "max":
                    denom = float(np.max(np.abs(z)))
                    if denom < 1e-6:
                        denom = 1.0
                    z = z / denom
                elif self.norm_mode == "std":
                    mu = float(np.mean(z))
                    sd = float(np.std(z))
                    if sd < 1e-6:
                        sd = 1.0
                    z = (z - mu) / sd
                return z.astype(np.float32, copy=False)
            except (FileNotFoundError, OSError, ValueError) as e:
                last_e = e
                time.sleep(s)
        raise RuntimeError(f"failed to load npy after retries: {path} ({type(last_e).__name__}: {last_e})")

    def __getitem__(self, idx: int):
        row = self.rows[idx]
        path = str(row["_npy_path"])
        z = self._load_wave_z(path)
        y = int(float(row["_evt6"]))

        x = torch.from_numpy(z)  # [L]
        # STFT -> complex: [F, T]
        spec = torch.stft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            center=True,
            return_complex=True,
        )
        mag = spec.abs()  # [F, T]
        if self.log_spec:
            mag = torch.log(mag + self.eps)
        # standardize spec per-sample (helps stability)
        mag = (mag - mag.mean()) / (mag.std() + 1e-6)
        mag = mag.unsqueeze(0)  # [1, F, T]
        return mag, torch.tensor(y, dtype=torch.long), dict(row)


class TinyCNN2D(nn.Module):
    def __init__(self, in_ch: int = 1, num_classes: int = 6, drop: float = 0.1):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_ch, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=float(drop)),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.head(x)
        return x


@dataclass
class Metrics:
    loss: float
    acc: float


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, criterion: nn.Module) -> Metrics:
    model.eval()
    total = 0
    correct = 0
    total_loss = 0.0
    for x, y, _ in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        loss = criterion(logits, y)
        total_loss += float(loss.item()) * int(y.size(0))
        pred = logits.argmax(dim=1)
        correct += int((pred == y).sum().item())
        total += int(y.size(0))
    return Metrics(loss=total_loss / max(1, total), acc=correct / max(1, total))


@torch.no_grad()
def predict_to_csv(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    out_csv: str,
    save_probs: bool = True,
):
    model.eval()
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    rows_out: List[Dict[str, object]] = []
    for x, y, meta in loader:
        x = x.to(device, non_blocking=True)
        logits = model(x)
        prob = F.softmax(logits, dim=-1).detach().cpu().numpy()
        pred = prob.argmax(axis=1).astype(int)
        y_np = y.detach().cpu().numpy().astype(int)

        # meta collate: dict-of-lists
        if isinstance(meta, dict):
            bs = len(pred)
            keys = list(meta.keys())
            for i in range(bs):
                r = {k: (meta[k][i] if isinstance(meta[k], (list, tuple)) else meta[k]) for k in keys}
                r["pred_evt6"] = int(pred[i])
                r["tgt_evt6"] = int(y_np[i])
                if save_probs:
                    for k in range(6):
                        r[f"prob_evt6_{k}"] = float(prob[i, k])
                rows_out.append(r)
        else:
            for i, m in enumerate(meta):
                r = dict(m) if isinstance(m, dict) else {}
                r["pred_evt6"] = int(pred[i])
                r["tgt_evt6"] = int(y_np[i])
                if save_probs:
                    for k in range(6):
                        r[f"prob_evt6_{k}"] = float(prob[i, k])
                rows_out.append(r)

    # write csv
    fieldnames = list(rows_out[0].keys()) if rows_out else ["pred_evt6", "tgt_evt6"]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows_out:
            w.writerow(r)


def main():
    ap = argparse.ArgumentParser(description="EVT6 spectrogram (Z-only) baseline")
    ap.add_argument("--mode", type=str, default="train_test", choices=["train_test", "test_only"])
    ap.add_argument("--data_dir", required=True, help="EVT6 protocol dir containing meta_evt6_{train,val,test}.csv")
    ap.add_argument("--out_dir", required=True, help="Output dir (ckpt + test_results)")
    ap.add_argument("--checkpoint", type=str, default="", help="For test_only: path to checkpoint; default out_dir/best.pt")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--patience", type=int, default=0, help="early stop patience by val_acc; 0 disables")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--max_lr", type=float, default=3e-3, help="OneCycleLR max_lr (if enabled)")
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=100)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--pin_memory", type=int, default=1)
    ap.add_argument("--amp", type=int, default=1)
    ap.add_argument("--norm_mode", type=str, default="max")
    ap.add_argument("--in_samples", type=int, default=8192)
    ap.add_argument("--stft_n_fft", type=int, default=256)
    ap.add_argument("--stft_hop_length", type=int, default=64)
    ap.add_argument("--stft_win_length", type=int, default=256)
    ap.add_argument("--log_spec", type=int, default=1)
    ap.add_argument("--label_smoothing", type=float, default=0.0)
    ap.add_argument("--save_test_probs", type=int, default=1)
    ap.add_argument("--use_scheduler", type=int, default=1, help="enable OneCycleLR (1/0)")
    args = ap.parse_args()

    _seed_everything(int(args.seed))
    data_dir = os.path.abspath(args.data_dir)
    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    train_csv = os.path.join(data_dir, "meta_evt6_train.csv")
    val_csv = os.path.join(data_dir, "meta_evt6_val.csv")
    test_csv = os.path.join(data_dir, "meta_evt6_test.csv")
    for p in (train_csv, val_csv, test_csv):
        if not os.path.exists(p):
            raise FileNotFoundError(p)

    train_ds = Evt6SpectrogramDataset(
        train_csv,
        norm_mode=args.norm_mode,
        in_samples=int(args.in_samples),
        stft_n_fft=int(args.stft_n_fft),
        stft_hop_length=int(args.stft_hop_length),
        stft_win_length=int(args.stft_win_length),
        log_spec=bool(int(args.log_spec)),
    )
    val_ds = Evt6SpectrogramDataset(
        val_csv,
        norm_mode=args.norm_mode,
        in_samples=int(args.in_samples),
        stft_n_fft=int(args.stft_n_fft),
        stft_hop_length=int(args.stft_hop_length),
        stft_win_length=int(args.stft_win_length),
        log_spec=bool(int(args.log_spec)),
    )
    test_ds = Evt6SpectrogramDataset(
        test_csv,
        norm_mode=args.norm_mode,
        in_samples=int(args.in_samples),
        stft_n_fft=int(args.stft_n_fft),
        stft_hop_length=int(args.stft_hop_length),
        stft_win_length=int(args.stft_win_length),
        log_spec=bool(int(args.log_spec)),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = DataLoader(
        train_ds,
        batch_size=int(args.batch_size),
        shuffle=True,
        num_workers=int(args.num_workers),
        pin_memory=bool(int(args.pin_memory)),
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=bool(int(args.pin_memory)),
        drop_last=False,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=bool(int(args.pin_memory)),
        drop_last=False,
    )

    model = TinyCNN2D(in_ch=1, num_classes=6, drop=float(args.dropout)).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=float(args.label_smoothing))
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))

    scheduler = None
    if int(args.use_scheduler) == 1:
        steps_per_epoch = max(1, len(train_loader))
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=float(args.max_lr),
            epochs=int(args.epochs),
            steps_per_epoch=steps_per_epoch,
            pct_start=0.1,
            div_factor=max(1.0, float(args.max_lr) / max(1e-12, float(args.lr))),
            final_div_factor=100.0,
        )

    use_amp = bool(int(args.amp)) and device.type == "cuda"
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
        autocast = lambda: torch.amp.autocast("cuda", enabled=scaler.is_enabled())
    else:
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
        autocast = lambda: torch.cuda.amp.autocast(enabled=scaler.is_enabled())

    best_acc = -1.0
    best_path = os.path.join(out_dir, "best.pt")
    patience = int(args.patience)
    bad = 0

    mode = str(args.mode).strip().lower()
    if mode == "train_test":
        for epoch in range(1, int(args.epochs) + 1):
            model.train()
            total = 0
            total_loss = 0.0
            for x, y, _ in train_loader:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with autocast():
                    logits = model(x)
                    loss = criterion(logits, y)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                if scheduler is not None:
                    scheduler.step()
                total_loss += float(loss.item()) * int(y.size(0))
                total += int(y.size(0))
            train_loss = total_loss / max(1, total)
            val_m = evaluate(model, val_loader, device, criterion)
            print(
                f"Epoch {epoch:03d} | train_loss={train_loss:.4f} | val_loss={val_m.loss:.4f} | val_acc={val_m.acc:.4f}",
                flush=True,
            )
            if val_m.acc > best_acc:
                best_acc = float(val_m.acc)
                torch.save({"model": model.state_dict(), "epoch": epoch, "val_acc": best_acc, "args": vars(args)}, best_path)
                bad = 0
            else:
                if patience > 0:
                    bad += 1
                    if bad >= patience:
                        print(f"[early_stop] patience={patience} reached at epoch={epoch}, best_val_acc={best_acc:.4f}", flush=True)
                        break
        ckpt_path = best_path
        print(f"[best] val_acc={best_acc:.4f} saved: {best_path}", flush=True)
    else:
        ckpt_path = os.path.abspath(args.checkpoint) if args.checkpoint else best_path
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(ckpt_path)

    # test
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state, strict=True)
    model.to(device)
    test_m = evaluate(model, test_loader, device, criterion)
    print(f"[test] loss={test_m.loss:.4f} acc={test_m.acc:.4f}", flush=True)

    out_csv = os.path.join(out_dir, "test_results_diting2_evt6_test.csv")
    predict_to_csv(model, test_loader, device, out_csv, save_probs=bool(int(args.save_test_probs)))
    print(f"[OK] wrote: {out_csv}", flush=True)


if __name__ == "__main__":
    main()

