"""
DiTing2.0 事件类型 6 类分类数据集适配器

evtype -> class id (evt6):
  eq -> 0
  ep -> 1
  co/ss -> 2   (CO=坍塌；在 DiTing2.0 的 evtype 编码中通常写成 ss)
  sp -> 3
  se -> 4
  ot -> 5

该数据集读取 tools/prepare_diting2_evt6.py 生成的 meta_evt6.csv + waves/*.npy。
"""

from .base import DatasetBase
from typing import Tuple
import os
import time
import pandas as pd
import numpy as np
from operator import itemgetter
from utils import logger
from ._factory import register_dataset

__all__ = ["DiTing2Evt6"]


@register_dataset
class DiTing2Evt6(DatasetBase):
    """DiTing2.0 event-type 6-way classification dataset."""

    _name = "diting2_evt6"
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
        # 优先读取严格划分文件（若存在）
        # - meta_evt6_train.csv / meta_evt6_val.csv / meta_evt6_test.csv
        # 否则回退到 meta_evt6.csv，并按旧逻辑用 0.8/0.1/0.1 切分
        if filename is None:
            strict_name = f"meta_evt6_{self._mode}.csv"
            strict_path = os.path.join(self._data_dir, strict_name)
            if os.path.exists(strict_path):
                filename = strict_name
                # 使用严格划分文件时，不再执行 ratio split
                use_ratio_split = False
            else:
                filename = "meta_evt6.csv"
                use_ratio_split = True
        else:
            use_ratio_split = filename == "meta_evt6.csv"

        csv_path = os.path.join(self._data_dir, filename)
        logger.info(f"加载元数据：{csv_path}")

        meta_df = pd.read_csv(csv_path, low_memory=False)

        # 清理空格
        for k in meta_df.columns:
            if meta_df[k].dtype in [object, np.object_, "object", "O"]:
                meta_df[k] = meta_df[k].astype(str).str.replace(" ", "")

        # shuffle
        if self._shuffle:
            meta_df = meta_df.sample(frac=1, replace=False, random_state=self._seed)
        meta_df.reset_index(drop=True, inplace=True)

        # split
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
            logger.info(f"Data Split: {self._mode}: {r[0]}-{r[1]}")

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
                logger.warning(f"[DiTing2Evt6] Missing npy files: {missing} / {len(meta_df)}. Dropped.")
                meta_df = meta_df.loc[m].copy()
                meta_df.reset_index(drop=True, inplace=True)

        return meta_df

    def _load_event_data(self, idx: int) -> Tuple[dict, dict]:
        def _load_npy_with_retry(path: str) -> np.ndarray:
            """
            共享盘/NFS 在多进程 DataLoader 下可能出现“瞬时 ENOENT/IO 抖动”（文件实际存在但 open 返回找不到/IO 错）。
            这里做更强的重试（同时捕获 FileNotFoundError 与 OSError），避免训练中途崩溃。
            """
            # 总等待约 0.2+0.5+1+2+4+8+13+21+34 ~= 83.7s
            sleeps = [0.2, 0.5, 1.0, 2.0, 4.0, 8.0, 13.0, 21.0, 34.0]
            last_err: Exception | None = None
            for attempt, sleep_s in enumerate(sleeps, start=1):
                try:
                    return np.load(path).astype(np.float32)
                except (FileNotFoundError, OSError) as e:
                    last_err = e
                    if attempt == 1:
                        logger.warning(f"[DiTing2Evt6] npy open failed (will retry): {path} ({type(e).__name__}: {e})")
                    time.sleep(sleep_s)
            assert last_err is not None
            raise last_err

        # 容错策略：若当前样本文件损坏/缺失，顺序尝试后续样本，直到取到可用样本。
        # 这样可避免训练被少量坏样本直接打断。
        n = len(self._meta_data)
        assert n > 0, "Empty metadata in DiTing2Evt6."
        start = int(idx) % n
        bad_count = 0

        for offset in range(n):
            cand_idx = (start + offset) % n
            target = self._meta_data.iloc[cand_idx]
            key, part, npy_rel, evt6 = itemgetter("key", "part", "_npy_path", "_evt6")(target)

            npy_path = str(npy_rel).strip().replace("\\", "/")
            if not os.path.isabs(npy_path):
                npy_path = os.path.join(self._data_dir, npy_path)

            try:
                data = _load_npy_with_retry(npy_path)
            except (FileNotFoundError, OSError) as e:
                bad_count += 1
                if bad_count <= 3:
                    logger.warning(
                        f"[DiTing2Evt6] Skip bad sample idx={cand_idx}, path={npy_path}, "
                        f"err={type(e).__name__}: {e}"
                    )
                continue

            # event dict: inputs use 'data', labels use 'evt6'
            try:
                cls_id = int(evt6)
            except Exception:
                cls_id = -1

            # Auxiliary coarse label for hierarchical training: sp vs non-sp
            # evt6 mapping: 0 eq,1 ep,2 co,3 sp,4 se,5 ot -> coarse: non-sp(0), sp(1)
            if cls_id == 3:
                cls_sp = 1
            elif cls_id >= 0:
                cls_sp = 0
            else:
                cls_sp = -1

            event = {
                "data": data,
                "evt6": [cls_id] if cls_id >= 0 else [],
                "evt6_sp": [cls_sp] if cls_sp >= 0 else [],
                "snr": np.array([10.0, 10.0, 10.0], dtype=np.float32),
            }
            return event, target.to_dict()

        raise RuntimeError(
            f"[DiTing2Evt6] All samples failed to load in mode={self._mode}. "
            f"Please check dataset files under: {self._data_dir}"
        )



