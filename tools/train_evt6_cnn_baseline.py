#!/usr/bin/env python3
"""
EVT6 六分类 CNN baseline（用于与 SeisMoLLM_evt6 对比）。

设计目标：
- **同口径数据划分**：直接读取 `diting2_evt6_paperalign/meta_evt6_{train,val,test}.csv`
- **同口径评测**：输出 `test_results_diting2_evt6_test.csv`（至少包含 pred_evt6/tgt_evt6），可直接喂给
  `tools/report_evt6_results.py` 生成报告
- **可复现**：seed 固定；保存 best checkpoint（按 val_acc）
- **尽量不改现有框架**：独立脚本，便于 sbatch / nohup

用法示例：
python tools/train_evt6_cnn_baseline.py \
  --data_dir /path/to/diting2_evt6 \
  --out_dir  `$REPO_ROOT/reports/cnn_evt6_baseline_xxx \
  --epochs 30 --batch_size 64 --lr 1e-3 --weight_decay 1e-4
"""

import argparse
import json
import os
import time
from dataclasses import dataclass
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


ID2NAME: Dict[int, str] = {0: "eq", 1: "ep", 2: "co", 3: "sp", 4: "se", 5: "ot"}
_WARN_ONCE = set()


def _warn_once(msg: str, key: str):
    if key in _WARN_ONCE:
        return
    _WARN_ONCE.add(key)
    print(msg, flush=True)


def _seed_everything(seed: int):
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class Evt6NpyDataset(Dataset):
    """
    Reads meta csv rows with columns:
      - _npy_path: absolute path to wave npy (shape: (3, 8192))
      - _evt6: int label in [0..5]
    """

    def __init__(self, meta_csv: str, norm_mode: str = "max"):
        self.meta_csv = os.path.abspath(meta_csv)
        if not os.path.exists(self.meta_csv):
            raise FileNotFoundError(self.meta_csv)
        norm_mode = str(norm_mode).strip().lower()
        if norm_mode not in ("max", "std", "none", ""):
            raise ValueError(f"未知 norm_mode={norm_mode}，应为 max/std/none")
        self.norm_mode = "none" if norm_mode in ("none", "") else norm_mode
        self.df = pd.read_csv(self.meta_csv, low_memory=False)
        need_cols = {"_npy_path", "_evt6"}
        miss = need_cols - set(self.df.columns)
        if miss:
            raise KeyError(f"meta 缺少列：{sorted(miss)} in {self.meta_csv}")

        # normalize path
        self.df["_npy_path"] = (
            self.df["_npy_path"].astype(str).str.strip().str.replace("\\\\", "/", regex=False)
        )

        # filter missing (avoid crash on shared fs)
        m = self.df["_npy_path"].apply(os.path.exists)
        missing = int((~m).sum())
        if missing:
            self.df = self.df.loc[m].copy()
            self.df.reset_index(drop=True, inplace=True)

        self.y = self.df["_evt6"].astype(int).to_numpy()
        # 用于 shared fs 抖动/文件缺失时返回全零样本，避免训练崩掉
        self.fallback_shape = (3, 8192)
        try:
            if len(self.df):
                p0 = str(self.df.iloc[0]["_npy_path"])
                a0 = np.load(p0, mmap_mode="r")
                if a0.ndim == 2 and a0.shape[0] == 3:
                    self.fallback_shape = (int(a0.shape[0]), int(a0.shape[1]))
        except Exception:
            pass

    def __len__(self):
        return int(len(self.df))

    def _safe_load_npy(self, path: str) -> np.ndarray | None:
        """共享盘偶发 ENOENT/IO 抖动：做重试；最终失败返回 None。"""
        sleeps = [0.05, 0.1, 0.2, 0.5, 1.0, 2.0]
        last_e: Exception | None = None
        for i, s in enumerate(sleeps, start=1):
            try:
                return np.load(path).astype(np.float32, copy=False)
            except (FileNotFoundError, OSError) as e:
                last_e = e
                if i == 1:
                    _warn_once(
                        f"[warn] npy open failed, will retry: {path} ({type(e).__name__}: {e})",
                        key=f"npy_retry|{path}",
                    )
                time.sleep(s)
        _warn_once(
            f"[warn] npy open failed after retries, will use zeros: {path} ({type(last_e).__name__}: {last_e})",
            key=f"npy_zeros|{path}",
        )
        return None

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        p = str(row["_npy_path"])
        x = self._safe_load_npy(p)
        if x is None:
            x = np.zeros(self.fallback_shape, dtype=np.float32)
        if x.ndim != 2 or x.shape[0] != 3:
            raise ValueError(f"期望 (3,L) npy，但得到 {x.shape}: {p}")
        # 清理 NaN/Inf，避免 loss=nan
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        # 归一化（避免 AMP/大幅值输入导致溢出）
        if self.norm_mode == "max":
            denom = np.max(np.abs(x), axis=1, keepdims=True)
            denom = np.where(denom < 1e-6, 1.0, denom)
            x = x / denom
        elif self.norm_mode == "std":
            mean = x.mean(axis=1, keepdims=True)
            std = x.std(axis=1, keepdims=True)
            std = np.where(std < 1e-6, 1.0, std)
            x = (x - mean) / std
        y = int(row["_evt6"])
        return torch.from_numpy(x), torch.tensor(y, dtype=torch.long), row.to_dict()


def build_cnn(num_classes: int = 6) -> nn.Module:
    # Lightweight 1D CNN on 3 channels
    return nn.Sequential(
        nn.Conv1d(3, 32, kernel_size=9, stride=2, padding=4, bias=False),
        nn.BatchNorm1d(32),
        nn.ReLU(inplace=True),
        nn.MaxPool1d(kernel_size=4, stride=4),
        nn.Conv1d(32, 64, kernel_size=9, stride=2, padding=4, bias=False),
        nn.BatchNorm1d(64),
        nn.ReLU(inplace=True),
        nn.MaxPool1d(kernel_size=4, stride=4),
        nn.Conv1d(64, 128, kernel_size=9, stride=2, padding=4, bias=False),
        nn.BatchNorm1d(128),
        nn.ReLU(inplace=True),
        nn.AdaptiveAvgPool1d(1),
        nn.Flatten(),
        nn.Linear(128, num_classes),
    )


class _BasicBlock1D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1, drop: float = 0.0):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size=7, stride=stride, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.act = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size=7, stride=1, padding=3, bias=False)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.drop = nn.Dropout(p=drop) if drop > 0 else nn.Identity()

        self.down = None
        if stride != 1 or in_ch != out_ch:
            self.down = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_ch),
            )

    def forward(self, x):
        identity = x
        out = self.act(self.bn1(self.conv1(x)))
        out = self.drop(out)
        out = self.bn2(self.conv2(out))
        if self.down is not None:
            identity = self.down(identity)
        out = self.act(out + identity)
        return out


class ResNet1D(nn.Module):
    """更强的 1D ResNet baseline（比 cnn_small 更容易把 val acc 拉上去）。"""

    def __init__(self, num_classes: int = 6, drop: float = 0.1):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(3, 64, kernel_size=11, stride=2, padding=5, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=4, stride=4),
        )
        self.layer1 = nn.Sequential(_BasicBlock1D(64, 64, stride=1, drop=drop), _BasicBlock1D(64, 64, stride=1, drop=drop))
        self.layer2 = nn.Sequential(_BasicBlock1D(64, 128, stride=2, drop=drop), _BasicBlock1D(128, 128, stride=1, drop=drop))
        self.layer3 = nn.Sequential(_BasicBlock1D(128, 256, stride=2, drop=drop), _BasicBlock1D(256, 256, stride=1, drop=drop))
        self.layer4 = nn.Sequential(_BasicBlock1D(256, 256, stride=2, drop=drop), _BasicBlock1D(256, 256, stride=1, drop=drop))
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(p=drop),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.head(x)
        return x


def build_model(name: str, num_classes: int = 6, drop: float = 0.1) -> nn.Module:
    name = str(name).strip().lower()
    if name in ("cnn_small", "cnn"):
        return build_cnn(num_classes=num_classes)
    if name in ("resnet", "resnet1d", "resnet18_1d"):
        return ResNet1D(num_classes=num_classes, drop=drop)
    raise ValueError(f"未知 model={name}，可选：cnn_small / resnet1d")


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
    model: nn.Module, loader: DataLoader, device: torch.device, out_csv: str, task_name: str = "evt6"
):
    model.eval()
    rows = []
    for x, y, meta in loader:
        x = x.to(device, non_blocking=True)
        logits = model(x)
        pred = logits.argmax(dim=1).detach().cpu().numpy().astype(int)
        y = y.detach().cpu().numpy().astype(int)
        # Default collate for dict returns a dict-of-lists (not list-of-dicts).
        # Support both shapes to be robust.
        if isinstance(meta, dict):
            bs = len(pred)
            keys = list(meta.keys())
            for i in range(bs):
                r = {k: (meta[k][i] if isinstance(meta[k], (list, tuple)) else meta[k]) for k in keys}
                r[f"pred_{task_name}"] = int(pred[i])
                r[f"tgt_{task_name}"] = int(y[i])
                rows.append(r)
        else:
            # list/tuple of dicts
            for p, t, m in zip(pred.tolist(), y.tolist(), meta):
                r = dict(m) if isinstance(m, dict) else {}
                r[f"pred_{task_name}"] = int(p)
                r[f"tgt_{task_name}"] = int(t)
                rows.append(r)

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    pd.DataFrame(rows).to_csv(out_csv, index=False)


def _counts(y: np.ndarray) -> Dict[str, int]:
    d = {ID2NAME[i]: int(np.sum(y == i)) for i in range(6)}
    d["total"] = int(len(y))
    return d


def main():
    ap = argparse.ArgumentParser(description="EVT6 CNN baseline training")
    ap.add_argument("--mode", type=str, default="train_test", help="train_test 或 test_only（默认 train_test）")
    ap.add_argument("--checkpoint", type=str, default="", help="test_only 时使用的 checkpoint；默认 out_dir/best.pt")
    ap.add_argument("--data_dir", required=True, help="diting2_evt6_paperalign 目录（含 meta_evt6_*.csv）")
    ap.add_argument("--out_dir", required=True, help="输出目录（保存 best.pt 与 test_results*.csv）")
    ap.add_argument("--model", type=str, default="resnet1d", help="cnn_small 或 resnet1d（默认 resnet1d）")
    ap.add_argument("--dropout", type=float, default=0.1, help="dropout 概率（默认 0.1）")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--max_lr", type=float, default=3e-3, help="OneCycleLR 的 max_lr（默认 3e-3）")
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--class_weight", type=str, default="balanced", help="none 或 balanced（默认 balanced）")
    ap.add_argument("--label_smoothing", type=float, default=0.0, help="CrossEntropy label_smoothing（默认 0）")
    ap.add_argument("--use_scheduler", type=int, default=1, help="是否启用 OneCycleLR（1/0）")
    ap.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值（<=0 表示关闭）")
    ap.add_argument("--seed", type=int, default=100)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--pin_memory", type=int, default=1)
    ap.add_argument("--norm_mode", type=str, default="max", help="输入归一化：max/std/none（默认 max）")
    ap.add_argument("--amp", type=int, default=1, help="是否使用 AMP (1/0)")
    args = ap.parse_args()

    data_dir = os.path.abspath(args.data_dir)
    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    _seed_everything(args.seed)

    train_csv = os.path.join(data_dir, "meta_evt6_train.csv")
    val_csv = os.path.join(data_dir, "meta_evt6_val.csv")
    test_csv = os.path.join(data_dir, "meta_evt6_test.csv")
    for p in [train_csv, val_csv, test_csv]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"缺少 split 文件：{p}")

    train_ds = Evt6NpyDataset(train_csv, norm_mode=args.norm_mode)
    val_ds = Evt6NpyDataset(val_csv, norm_mode=args.norm_mode)
    test_ds = Evt6NpyDataset(test_csv, norm_mode=args.norm_mode)

    print("[data] train:", _counts(train_ds.y))
    print("[data] val  :", _counts(val_ds.y))
    print("[data] test :", _counts(test_ds.y))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[device]", device)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=bool(args.pin_memory),
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=bool(args.pin_memory),
        drop_last=False,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=bool(args.pin_memory),
        drop_last=False,
    )

    model = build_model(args.model, num_classes=6, drop=float(args.dropout)).to(device)

    class_weight = None
    cw_mode = str(args.class_weight).strip().lower()
    if cw_mode not in ("none", "", "balanced"):
        raise ValueError(f"未知 class_weight={args.class_weight}，应为 none/balanced")
    if cw_mode == "balanced":
        # w_i = total / (C * count_i)
        counts = np.array([np.sum(train_ds.y == i) for i in range(6)], dtype=np.float32)
        counts = np.where(counts < 1, 1.0, counts)
        total = float(counts.sum())
        w = total / (6.0 * counts)
        # normalize to mean=1 for stability
        w = w / float(w.mean())
        class_weight = torch.tensor(w, dtype=torch.float32, device=device)
        print("[loss] class_weight:", {ID2NAME[i]: float(w[i]) for i in range(6)}, flush=True)

    # label_smoothing 需要 torch>=1.10
    criterion = nn.CrossEntropyLoss(weight=class_weight, label_smoothing=float(args.label_smoothing))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    use_amp = bool(args.amp) and device.type == "cuda"
    # torch>=2.0 推荐 torch.amp；旧版本回退 torch.cuda.amp
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
        autocast = lambda: torch.amp.autocast("cuda", enabled=scaler.is_enabled())
    else:
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
        autocast = lambda: torch.cuda.amp.autocast(enabled=scaler.is_enabled())

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

    best_acc = -1.0
    best_path = os.path.join(out_dir, "best.pt")

    record = {
        "seed": int(args.seed),
        "args": vars(args),
        "data_dir": data_dir,
        "splits": {"train": train_csv, "val": val_csv, "test": test_csv},
        "counts": {"train": _counts(train_ds.y), "val": _counts(val_ds.y), "test": _counts(test_ds.y)},
    }
    with open(os.path.join(out_dir, "cnn_evt6_record.json"), "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)

    mode = str(args.mode).strip().lower()
    if mode not in ("train_test", "test_only"):
        raise ValueError(f"未知 mode={args.mode}，应为 train_test/test_only")

    if mode == "train_test":
        for epoch in range(1, args.epochs + 1):
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
                if float(args.grad_clip) > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(args.grad_clip))
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
                torch.save(
                    {"model": model.state_dict(), "epoch": epoch, "val_acc": best_acc, "args": vars(args)},
                    best_path,
                )

        print(f"[best] val_acc={best_acc:.4f} saved to: {best_path}", flush=True)
        ckpt_path = best_path
    else:
        ckpt_path = os.path.abspath(args.checkpoint) if args.checkpoint else best_path
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"test_only 找不到 checkpoint: {ckpt_path}")

    # Test with checkpoint
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    model.to(device)

    test_m = evaluate(model, test_loader, device, criterion)
    print(f"[test] loss={test_m.loss:.4f} acc={test_m.acc:.4f}")

    out_csv = os.path.join(out_dir, "test_results_diting2_evt6_test.csv")
    predict_to_csv(model, test_loader, device, out_csv, task_name="evt6")
    print(f"[OK] wrote: {out_csv}")


if __name__ == "__main__":
    main()


