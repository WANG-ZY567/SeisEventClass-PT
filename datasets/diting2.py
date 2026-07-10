"""
DiTing2.0 数据集适配器

基于 diting.py 修改，适配预处理后的 npy + meta.csv 格式。
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

__all__ = ["DiTing2"]


@register_dataset
class DiTing2(DatasetBase):
    """DiTing2.0 Dataset (预处理后的 npy + meta.csv 格式)"""

    _name = "diting2"
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
        **kwargs
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
        """
        加载元数据 CSV。
        
        CSV 应由 tools/prepare_diting2.py 生成，包含以下列：
        - key, part, ev_id
        - p_pick, s_pick
        - baz, dis
        - evmag, mag_type, st_mag
        - p_motion, p_clarity
        - Z_P_power_snr, N_S_power_snr, E_S_power_snr
        - _npy_path (波形 npy 路径)
        """
        if filename is None:
            # 自动查找 meta_*.csv
            csv_files = [f for f in os.listdir(self._data_dir) if f.startswith('meta_') and f.endswith('.csv')]
            if not csv_files:
                raise FileNotFoundError(f"未找到 meta_*.csv 文件在 {self._data_dir}")
            filename = csv_files[0]
            logger.info(f"自动选择元数据文件：{filename}")
        
        csv_path = os.path.join(self._data_dir, filename)
        logger.info(f"加载元数据：{csv_path}")
        
        meta_df = pd.read_csv(
            csv_path,
            dtype={
                "part": np.int64,
                "key": str,
                "ev_id": np.int64,
                "evmag": str,
                "mag_type": str,
                "p_pick": np.int64,
                "p_clarity": str,
                "p_motion": str,
                "s_pick": np.int64,
                "net": str,
                "sta_id": np.int64,
                "dis": np.float32,
                "st_mag": str,
                "baz": str,
                "Z_P_power_snr": np.float32,
                "N_S_power_snr": np.float32,
                "E_S_power_snr": np.float32,
                "P_residual": str,
                "S_residual": str,
                "_npy_path": str,
            },
            low_memory=False,
        )

        # 清理空格
        for k in meta_df.columns:
            if meta_df[k].dtype in [object, np.object_, "object", "O"]:
                meta_df[k] = meta_df[k].astype(str).str.replace(" ", "")

        # 数据划分
        if self._shuffle:
            meta_df = meta_df.sample(frac=1, replace=False, random_state=self._seed)

        meta_df.reset_index(drop=True, inplace=True)

        if self._data_split:
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

        # 训练前过滤掉缺失的 npy 文件（避免 DataLoader 中途 FileNotFoundError 导致训练崩溃）
        # 兼容两类路径：
        # - 优先使用 `_npy_path`（tools/prepare_diting2.py 生成）
        # - fallback: waves/{key}_{part}.npy 或 waves/{key}.npy
        if "_npy_path" in meta_df.columns:
            def _resolve_npy(row) -> str:
                p = row.get("_npy_path", "")
                # pandas 可能把缺失值读成 nan（float）或 "nan"(str)
                if p is None:
                    p = ""
                if not isinstance(p, str):
                    p = str(p)
                p = p.strip()
                if p and p.lower() != "nan":
                    return p if os.path.isabs(p) else os.path.join(self._data_dir, p)

                key = str(row.get("key", "")).strip()
                part = row.get("part", None)
                try:
                    part_i = int(part) if part is not None and str(part).strip() != "" else None
                except Exception:
                    part_i = None

                # 常见：waves/{key}_{part}.npy（若 key 本身包含 '_' 也没关系）
                if part_i is not None:
                    cand = os.path.join(self._data_dir, "waves", f"{key}_{part_i}.npy")
                    if os.path.exists(cand):
                        return cand

                # 兜底：waves/{key}.npy
                return os.path.join(self._data_dir, "waves", f"{key}.npy")

            npy_paths = meta_df.apply(_resolve_npy, axis=1)
            exists_mask = npy_paths.apply(os.path.exists)
            missing = int((~exists_mask).sum())
            if missing > 0:
                logger.warning(f"[DiTing2] Missing npy files: {missing} / {len(meta_df)}. They will be dropped.")
                meta_df = meta_df.loc[exists_mask].copy()
                meta_df.reset_index(drop=True, inplace=True)

        return meta_df

    def _load_event_data(self, idx: int) -> Tuple[dict, dict]:
        """
        加载事件数据。
        
        与原版 diting.py 的主要区别：
        - 从 npy 文件读取波形，而不是从 HDF5
        - 其他标签处理逻辑保持一致
        """
        target_event = self._meta_data.iloc[idx]
        key = target_event["key"]

        def _load_npy_with_retry(path: str) -> np.ndarray:
            # 共享盘/NFS 在多进程 DataLoader 下偶发“短暂看不到文件”，做轻量重试避免训练中断
            last_err = None
            for attempt in range(6):
                try:
                    return np.load(path).astype(np.float32)
                except FileNotFoundError as e:
                    last_err = e
                    time.sleep([0.2, 0.5, 1.0, 2.0, 4.0, 8.0][attempt])
            raise last_err

        # 从 npy 读取波形（已经是 (3, 8192) 格式）
        npy_path = target_event.get("_npy_path", None)
        if npy_path is None:
            npy_path = ""
        if not isinstance(npy_path, str):
            npy_path = str(npy_path)
        npy_path = npy_path.strip()
        if npy_path and npy_path.lower() != "nan" and (not os.path.isabs(npy_path)):
            npy_path = os.path.join(self._data_dir, npy_path)
        
        if npy_path and os.path.exists(npy_path):
            data = _load_npy_with_retry(npy_path)
        else:
            # Fallback：从 waves 目录读取（兼容 waves/{key}_{part}.npy 与 waves/{key}.npy）
            part = target_event.get("part", None)
            try:
                part_i = int(part) if part is not None and str(part).strip() != "" else None
            except Exception:
                part_i = None

            cand_paths = []
            if part_i is not None:
                cand_paths.append(os.path.join(self._data_dir, "waves", f"{key}_{part_i}.npy"))
            cand_paths.append(os.path.join(self._data_dir, "waves", f"{key}.npy"))

            found = None
            for p in cand_paths:
                if os.path.exists(p):
                    found = p
                    break
            if found is None:
                raise FileNotFoundError(f"Waveform npy not found. Tried: {cand_paths}")
            data = _load_npy_with_retry(found)

        # 提取标签（使用预处理后的 _pmp_bin）
        (
            ppk, spk,
            mag_type, evmag,
            stmag, pmp_bin,
            clarity, baz,
            dis, zpp_snr,
            nsp_snr, esp_snr,
        ) = itemgetter(
            "p_pick", "s_pick",
            "mag_type", "evmag",
            "st_mag", "_pmp_bin",
            "p_clarity", "baz",
            "dis", "Z_P_power_snr",
            "N_S_power_snr", "E_S_power_snr",
        )(
            target_event
        )

        # 震级处理
        try:
            if isinstance(evmag, str):
                evmag = float(evmag)
        except:
            evmag = 0

        try:
            if isinstance(stmag, str):
                stmag = float(stmag)
        except:
            stmag = 0

        # 极性：优先使用预处理后的 `_pmp_bin`（0/1）
        # 注意：这里不再“默认填 0”，因为那会掩盖数据问题；Full5 数据应当 100% 有效。
        motion = None
        try:
            if pd.notnull(pmp_bin) and pmp_bin != "":
                motion = int(float(pmp_bin))
                if motion not in (0, 1):
                    motion = None
        except (ValueError, TypeError, AttributeError):
            motion = None

        # 清晰度
        if pd.notnull(clarity):
            clarity = 0 if clarity.lower() == "i" else 1

        # 方位角
        if pd.notnull(baz):
            if baz == '':
                baz = np.nan
            else:
                baz = float(baz)
                baz = baz % 360

        # 震级类型转换（已在预处理脚本中完成，这里保持一致）
        mag_type_lower = str(mag_type).strip().lower()
        if mag_type_lower == "ms":
            evmag = (evmag + 1.08) / 1.13
            stmag = (stmag + 1.08) / 1.13
        elif mag_type_lower == "mb":
            evmag = (1.17 * evmag + 0.67) / 1.13
            stmag = (1.17 * stmag + 0.67) / 1.13
        elif mag_type_lower in ["ml", ""]:
            pass
        else:
            # 不抛出异常：按 ML 处理即可
            # 注意：该 Dataset 代码可能在 DataLoader worker 进程中运行，
            # 此处不要调用项目自定义 logger（worker 中可能未初始化 logger，导致异常中断训练）。
            pass

        evmag = np.clip(evmag, 0, 8, dtype=np.float32)
        stmag = np.clip(stmag, 0, 8, dtype=np.float32)

        snr = np.array([zpp_snr, nsp_snr, esp_snr])

        # 构造 event 字典（与原版一致）
        # 注意：pmp 是 onehot 类型，必须是非空列表；若为空应在训练参数侧关闭“纯噪声样本增强”。
        event = {
            "data": data,
            "ppks": [ppk] if pd.notnull(ppk) and ppk >= 0 else [],
            "spks": [spk] if pd.notnull(spk) and spk >= 0 else [],
            "emg": [evmag] if pd.notnull(evmag) and evmag > 0 else [],
            "smg": [stmag] if pd.notnull(stmag) and stmag > 0 else [],
            "pmp": [motion] if motion in (0, 1) else [],
            "clr": [clarity] if pd.notnull(clarity) else [],
            "baz": [baz] if pd.notnull(baz) else [],
            "dis": [dis] if pd.notnull(dis) else [],
            "snr": snr,
        }
        target = target_event.to_dict()

        return event, target


@register_dataset
class DiTing2_light(DiTing2):
    """DiTing2.0 轻量版（兼容 diting_light 命名）"""
    _name = "diting2_light"

