"""
DiTing2.0 data 5 data(data se)

data tools/prepare_diting2_evt6.py data meta_evt6*.csv data waves/*.npy: 
data evt6 id value: 
  eq -> 0
  ep -> 1
  co/ss -> 2
  sp -> 3
  se -> 4  (data)
  ot -> 5

evt5 data(data se data 0..4): 
  eq -> 0
  ep -> 1
  co -> 2
  sp -> 3
  ot -> 4
"""

from __future__ import annotations

from .base import DatasetBase
from typing import Tuple
import os
import time
import pandas as pd
import numpy as np
from operator import itemgetter
from utils import logger
from ._factory import register_dataset

__all__ = ["DiTing2Evt5"]


# Open-source note: implementation detail.
_EVT6_TO_EVT5 = {0: 0, 1: 1, 2: 2, 3: 3, 5: 4}


@register_dataset
class DiTing2Evt5(DatasetBase):
    """DiTing2.0 event-type 5-way classification dataset (exclude se)."""

    _name = "diting2_evt5"
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
        **kwargs,
    ):
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
        # Open-source note: implementation detail.
        if filename is None:
            strict_name = f"meta_evt6_{self._mode}.csv"
            strict_path = os.path.join(self._data_dir, strict_name)
            if os.path.exists(strict_path):
                filename = strict_name
                use_ratio_split = False
            else:
                filename = "meta_evt6.csv"
                use_ratio_split = True
        else:
            use_ratio_split = filename == "meta_evt6.csv"

        csv_path = os.path.join(self._data_dir, filename)
        logger.info(f"[DiTing2Evt5] Loading metavalue: {csv_path}")
        meta_df = pd.read_csv(csv_path, low_memory=False)

        # Open-source note: implementation detail.
        for k in meta_df.columns:
            if meta_df[k].dtype in [object, np.object_, "object", "O"]:
                meta_df[k] = meta_df[k].astype(str).str.replace(" ", "")

        # shuffle
        if self._shuffle:
            meta_df = meta_df.sample(frac=1, replace=False, random_state=self._seed)
        meta_df.reset_index(drop=True, inplace=True)

        # Open-source note: implementation detail.
        if use_ratio_split and self._data_split:
            irange = {}
            irange["train"] = [0, int(self._train_size * meta_df.shape[0])]
            irange["val"] = [
                irange["train"][1],
                irange["train"][1] + int(self._val_size * meta_df.shape[0]),
            ]
            irange["test"] = [irange["val"][1], meta_df.shape[0]]

            r = irange[self._mode]
            meta_df = meta_df.iloc[r[0] : r[1], :]
            logger.info(f"[DiTing2Evt5] Data Split: {self._mode}: {r[0]}-{r[1]}")

        # Open-source note: implementation detail.
        if "_evt6" not in meta_df.columns:
            raise KeyError("[DiTing2Evt5] Metadata must contain _evt6; run tools/prepare_diting2_evt6.py first.")
        meta_df["_evt6"] = meta_df["_evt6"].astype(int)
        before = int(len(meta_df))
        meta_df = meta_df.loc[meta_df["_evt6"] != 4].copy()
        after = int(len(meta_df))
        if after != before:
            logger.info(f"[DiTing2Evt5] Drop se(true=4): {before-after} / {before}")
        meta_df["_evt5"] = meta_df["_evt6"].map(_EVT6_TO_EVT5).astype(int)

        # drop missing npy
        if "_npy_path" in meta_df.columns:
            def _abs_npy(p):
                if not isinstance(p, str):
                    p = "" if pd.isna(p) else str(p)
                p = p.strip().replace("\\", "/")
                if not p or p.lower() == "nan":
                    return ""
                return p if os.path.isabs(p) else os.path.join(self._data_dir, p)

            abs_paths = meta_df["_npy_path"].apply(_abs_npy)
            m = abs_paths.apply(os.path.exists)
            missing = int((~m).sum())
            if missing:
                logger.warning(f"[DiTing2Evt5] Missing npy files: {missing} / {len(meta_df)}. Dropped.")
                meta_df = meta_df.loc[m].copy()
                meta_df.reset_index(drop=True, inplace=True)

        return meta_df

    def _load_event_data(self, idx: int) -> Tuple[dict, dict]:
        target = self._meta_data.iloc[idx]
        key, part, npy_rel, evt5 = itemgetter("key", "part", "_npy_path", "_evt5")(target)

        npy_path = str(npy_rel).strip().replace("\\", "/")
        if not os.path.isabs(npy_path):
            npy_path = os.path.join(self._data_dir, npy_path)

        def _load_npy_with_retry(path: str) -> np.ndarray:
            sleeps = [0.2, 0.5, 1.0, 2.0, 4.0, 8.0, 13.0, 21.0, 34.0]
            last_err: Exception | None = None
            for attempt, sleep_s in enumerate(sleeps, start=1):
                try:
                    return np.load(path).astype(np.float32)
                except (FileNotFoundError, OSError) as e:
                    last_err = e
                    if attempt == 1:
                        logger.warning(f"[DiTing2Evt5] npy open failed (will retry): {path} ({type(e).__name__}: {e})")
                    time.sleep(sleep_s)
            assert last_err is not None
            raise last_err

        data = _load_npy_with_retry(npy_path)

        try:
            cls_id = int(evt5)
        except Exception:
            cls_id = -1

        event = {
            "data": data,
            "evt5": [cls_id] if cls_id >= 0 else [],
            "snr": np.array([10.0, 10.0, 10.0], dtype=np.float32),
        }
        return event, target.to_dict()


