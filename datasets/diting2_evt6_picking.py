"""
DiTing2.0 EVT6 xapp picking 子集适配器：

用于把 `meta_evt6_{train,val,test}.csv`（含裁窗对齐后的 p_pick/s_pick）喂给 picking 模型（SeisMoLLM_dpk）。

关键约定：
- 输入波形：`_npy_path` 指向裁窗后的 (3, 8192) npy
- P/S 到时：使用 `p_pick`/`s_pick`（已经是裁窗坐标系，单位=sample）
- 输出 event dict 需要包含：
  - data: np.ndarray (3, in_samples)
  - ppks: List[int]（SeisT/SeisMoLLM picking 口径）
  - spks: List[int]
"""

from __future__ import annotations

from .base import DatasetBase
from typing import Tuple, List
import os
import time
import numpy as np
import pandas as pd
from utils import logger
from ._factory import register_dataset

__all__ = [
    "DiTing2Evt6Picking",
    "DiTing2Evt6PickingCondOracle",
    "DiTing2Evt6PickingCondPredHierSp",
    "DiTing2Evt6PickingCondPredMultiheadW01",
]


def _to_int_or_none(x):
    try:
        if x is None:
            return None
        if isinstance(x, str) and (x.strip() == "" or x.strip().lower() == "nan"):
            return None
        v = int(float(x))
        return v
    except Exception:
        return None


@register_dataset
class DiTing2Evt6Picking(DatasetBase):
    """
    EVT6 picking subset dataset for dpk task.

    Dataset name: `diting2_evt6_picking`
    """

    _name = "diting2_evt6_picking"
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
        # 对齐 EVT6 的 strict split 命名
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
        logger.info(f"[DiTing2Evt6Picking] 加载元数据：{csv_path}")
        meta_df = pd.read_csv(csv_path, low_memory=False)

        # 清理空格
        for k in meta_df.columns:
            if meta_df[k].dtype in [object, np.object_, "object", "O"]:
                meta_df[k] = meta_df[k].astype(str).str.replace(" ", "")

        # shuffle
        if self._shuffle:
            meta_df = meta_df.sample(frac=1, replace=False, random_state=self._seed)
        meta_df.reset_index(drop=True, inplace=True)

        # ratio split（仅对 meta_evt6.csv 生效）
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
            logger.info(f"[DiTing2Evt6Picking] Data Split: {self._mode}: {r[0]}-{r[1]}")

        # drop missing npy
        if "_npy_path" not in meta_df.columns:
            raise KeyError("[DiTing2Evt6Picking] meta 缺少列：_npy_path")

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
            logger.warning(f"[DiTing2Evt6Picking] Missing npy files: {missing} / {len(meta_df)}. Dropped.")
            meta_df = meta_df.loc[m].copy()
            meta_df.reset_index(drop=True, inplace=True)

        return meta_df

    def _load_event_data(self, idx: int) -> Tuple[dict, dict]:
        def _load_npy_with_retry(path: str) -> np.ndarray:
            sleeps = [0.2, 0.5, 1.0, 2.0, 4.0, 8.0, 13.0]
            last_err: Exception | None = None
            for attempt, sleep_s in enumerate(sleeps, start=1):
                try:
                    return np.load(path).astype(np.float32)
                except (FileNotFoundError, OSError) as e:
                    last_err = e
                    if attempt == 1:
                        logger.warning(f"[DiTing2Evt6Picking] npy open failed (will retry): {path} ({type(e).__name__}: {e})")
                    time.sleep(sleep_s)
            assert last_err is not None
            raise last_err

        target = self._meta_data.iloc[int(idx)]
        npy_rel = target.get("_npy_path", "")
        npy_path = str(npy_rel).strip().replace("\\", "/")
        if not os.path.isabs(npy_path):
            npy_path = os.path.join(self._data_dir, npy_path)

        data = _load_npy_with_retry(npy_path)

        p_pick = _to_int_or_none(target.get("p_pick", None))
        s_pick = _to_int_or_none(target.get("s_pick", None))

        ppks: List[int] = [p_pick] if (p_pick is not None and p_pick >= 0) else []
        spks: List[int] = [s_pick] if (s_pick is not None and s_pick >= 0) else []

        event = {
            "data": data,
            "ppks": ppks,
            "spks": spks,
            # 兼容部分可选目标（不用也没事）
            "snr": np.array([10.0, 10.0, 10.0], dtype=np.float32),
        }
        return event, target.to_dict()


@register_dataset
class DiTing2Evt6PickingCondOracle(DiTing2Evt6Picking):
    """
    Oracle-conditioned phase picking dataset:
    use ground-truth evt6 class as `evt6_cond`.
    """

    _name = "diting2_evt6_picking_cond_oracle"
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

    def _load_event_data(self, idx: int) -> Tuple[dict, dict]:
        event, meta = DiTing2Evt6Picking._load_event_data(self, idx=idx)
        # ground-truth evt6 is stored as _evt6 in xapp meta
        evt6 = _to_int_or_none(meta.get("_evt6", None))
        if evt6 is None or evt6 < 0 or evt6 > 5:
            raise ValueError(f"[diting2_evt6_picking_cond_oracle] invalid _evt6={evt6}")
        event["evt6_cond"] = int(evt6)
        return event, meta


@register_dataset
class DiTing2Evt6PickingCondPredHierSp(DiTing2Evt6Picking):
    """
    Predicted-conditioned picking dataset:
    use router hier_sp predicted evt6 class as `evt6_cond`.
    """

    _name = "diting2_evt6_picking_cond_pred_hier_sp"
    _channels = ["z", "n", "e"]
    _sampling_rate = 50

    _pred_col = "pred_evt6_hier_sp"

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

    def _load_event_data(self, idx: int) -> Tuple[dict, dict]:
        event, meta = DiTing2Evt6Picking._load_event_data(self, idx=idx)
        evt6 = _to_int_or_none(meta.get(self._pred_col, None))
        if evt6 is None or evt6 < 0 or evt6 > 5:
            raise ValueError(
                f"[diting2_evt6_picking_cond_pred_hier_sp] missing/invalid column {self._pred_col}, value={evt6}"
            )
        event["evt6_cond"] = int(evt6)
        return event, meta


@register_dataset
class DiTing2Evt6PickingCondPredMultiheadW01(DiTing2Evt6Picking):
    """
    Predicted-conditioned picking dataset:
    use router multihead_w01 predicted evt6 class as `evt6_cond`.
    """

    _name = "diting2_evt6_picking_cond_pred_multihead_w01"
    _channels = ["z", "n", "e"]
    _sampling_rate = 50

    _pred_col = "pred_evt6_multihead_w01"

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

    def _load_event_data(self, idx: int) -> Tuple[dict, dict]:
        event, meta = DiTing2Evt6Picking._load_event_data(self, idx=idx)
        evt6 = _to_int_or_none(meta.get(self._pred_col, None))
        if evt6 is None or evt6 < 0 or evt6 > 5:
            raise ValueError(
                f"[diting2_evt6_picking_cond_pred_multihead_w01] missing/invalid column {self._pred_col}, value={evt6}"
            )
        event["evt6_cond"] = int(evt6)
        return event, meta

