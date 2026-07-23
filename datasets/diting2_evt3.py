"""
DiTing2.0 data(data)value: EQ / EP / CO

value: 
- data `/text2/data` data numpy: 
  - natural_datas.npy                 -> EQ (0)
  - non_natural_datas_blasting.npy    -> EP (1)
  - non_natural_datas_collapse.npy    -> CO (2)
- data meta.csv; split dataset data seed data. 
- data/data(EQ: 6778/300, EP: 5300/313, CO: 5000/311), 
  data(data EP=5612)data, data"data", data. 
- value: test data val data(test_from_val=True). 
"""

from __future__ import annotations

import os
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from utils import logger

from .base import DatasetBase
from ._factory import register_dataset

__all__ = ["DiTing2Evt3"]


@register_dataset
class DiTing2Evt3(DatasetBase):
    _name = "diting2_evt3"
    _channels = ["z", "n", "e"]
    _sampling_rate = 50

    def __init__(
        self,
        seed: int,
        mode: str,
        data_dir: str,
        shuffle: bool = True,
        data_split: bool = True,
        train_size: float = 0.8,
        val_size: float = 0.1,
        # paper-aligned defaults
        eq_total: int = 7078,
        ep_total: int = 5613,
        co_total: int = 5311,
        eq_val: int = 300,
        ep_val: int = 313,
        co_val: int = 311,
        test_from_val: bool = True,
        **kwargs,
    ):
        self._eq_total = int(eq_total)
        self._ep_total = int(ep_total)
        self._co_total = int(co_total)
        self._eq_val = int(eq_val)
        self._ep_val = int(ep_val)
        self._co_val = int(co_val)
        self._test_from_val = bool(test_from_val)

        # memmap arrays will be loaded in _load_meta_data()
        self._x_eq = None
        self._x_ep = None
        self._x_co = None

        super().__init__(
            seed=seed,
            mode=mode,
            data_dir=data_dir,
            shuffle=shuffle,
            data_split=data_split,
            train_size=train_size,
            val_size=val_size,
        )

    def _load_meta_data(self, filename=None) -> pd.DataFrame:
        # Load three class arrays (mmap to avoid huge RAM usage)
        eq_path = os.path.join(self._data_dir, "natural_datas.npy")
        ep_path = os.path.join(self._data_dir, "non_natural_datas_blasting.npy")
        co_path = os.path.join(self._data_dir, "non_natural_datas_collapse.npy")

        for p in (eq_path, ep_path, co_path):
            if not os.path.exists(p):
                raise FileNotFoundError(
                    f"[DiTing2Evt3] Required array not found: {p}\n"
                    "Set --data to the prepared directory containing natural_datas.npy and non-natural arrays."
                )

        self._x_eq = np.load(eq_path, mmap_mode="r")
        self._x_ep = np.load(ep_path, mmap_mode="r")
        self._x_co = np.load(co_path, mmap_mode="r")

        n_eq, n_ep, n_co = int(self._x_eq.shape[0]), int(self._x_ep.shape[0]), int(self._x_co.shape[0])
        use_eq = min(n_eq, self._eq_total)
        use_ep = min(n_ep, self._ep_total)
        use_co = min(n_co, self._co_total)

        if (use_eq, use_ep, use_co) != (self._eq_total, self._ep_total, self._co_total):
            logger.warning(
                "[DiTing2Evt3] Available samples are fewer than the configured target counts: "
                f"EQ {n_eq}->{use_eq} (target {self._eq_total}), "
                f"EP {n_ep}->{use_ep} (target {self._ep_total}), "
                f"CO {n_co}->{use_co} (target {self._co_total})."
            )

        # Build per-class indices (deterministic)
        rng = np.random.RandomState(self._seed)
        idx_eq = np.arange(n_eq, dtype=np.int64); rng.shuffle(idx_eq); idx_eq = idx_eq[:use_eq]
        idx_ep = np.arange(n_ep, dtype=np.int64); rng.shuffle(idx_ep); idx_ep = idx_ep[:use_ep]
        idx_co = np.arange(n_co, dtype=np.int64); rng.shuffle(idx_co); idx_co = idx_co[:use_co]

        # Validation counts (cap by available)
        val_eq = min(self._eq_val, len(idx_eq))
        val_ep = min(self._ep_val, len(idx_ep))
        val_co = min(self._co_val, len(idx_co))

        # Split: take first val_x after shuffle for val; rest for train
        rng.shuffle(idx_eq); rng.shuffle(idx_ep); rng.shuffle(idx_co)
        eq_val_idx, eq_tr_idx = idx_eq[:val_eq], idx_eq[val_eq:]
        ep_val_idx, ep_tr_idx = idx_ep[:val_ep], idx_ep[val_ep:]
        co_val_idx, co_tr_idx = idx_co[:val_co], idx_co[val_co:]

        # For test: follow paper, test == val
        if self._mode == "train":
            parts = [("eq", 0, eq_tr_idx), ("ep", 1, ep_tr_idx), ("co", 2, co_tr_idx)]
        elif self._mode == "val":
            parts = [("eq", 0, eq_val_idx), ("ep", 1, ep_val_idx), ("co", 2, co_val_idx)]
        else:  # test
            if self._test_from_val:
                parts = [("eq", 0, eq_val_idx), ("ep", 1, ep_val_idx), ("co", 2, co_val_idx)]
            else:
                # fallback: treat remaining as test if needed (not paper-aligned)
                parts = [("eq", 0, eq_val_idx), ("ep", 1, ep_val_idx), ("co", 2, co_val_idx)]

        rows = []
        for cls_name, cls_id, idxs in parts:
            for ii in idxs.tolist():
                rows.append({"_cls": cls_name, "_evt3": int(cls_id), "_i": int(ii)})

        meta = pd.DataFrame(rows)
        if self._shuffle:
            meta = meta.sample(frac=1, replace=False, random_state=self._seed).reset_index(drop=True)

        # Quick summary
        vc = meta["_evt3"].value_counts().sort_index().to_dict()
        logger.info(f"[DiTing2Evt3] split={self._mode} size={len(meta)} counts={vc} test_from_val={self._test_from_val}")

        return meta

    def _load_event_data(self, idx: int) -> Tuple[dict, Dict]:
        row = self._meta_data.iloc[idx]
        cls = int(row["_evt3"])
        ii = int(row["_i"])

        if cls == 0:
            x = np.asarray(self._x_eq[ii], dtype=np.float32)
        elif cls == 1:
            x = np.asarray(self._x_ep[ii], dtype=np.float32)
        else:
            x = np.asarray(self._x_co[ii], dtype=np.float32)

        # (L,) -> (C,L) with C=3 (replicate Z to N/E to match framework input signature)
        if x.ndim != 1:
            x = x.reshape(-1)
        data = np.stack([x, x, x], axis=0).astype(np.float32, copy=False)

        event = {
            "data": data,
            "evt3": [cls],
            "snr": np.array([10.0, 10.0, 10.0], dtype=np.float32),
        }
        meta = {"_evt3": cls, "_i": ii, "_cls": str(row["_cls"])}
        return event, meta


