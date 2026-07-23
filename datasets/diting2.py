"""
DiTing2.0 data

data diting.py data, data npy + meta.csv data. 
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
    """Prepared DiTing 2.0 waveform dataset."""

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
        metadata CSV. 
        
        CSV data tools/prepare_diting2.py data, value: 
        - key, part, ev_id
        - p_pick, s_pick
        - baz, dis
        - evmag, mag_type, st_mag
        - p_motion, p_clarity
        - Z_P_power_snr, N_S_power_snr, E_S_power_snr
        - _npy_path (data npy data)
        """
        if filename is None:
            # Open-source note: implementation detail.
            csv_files = [f for f in os.listdir(self._data_dir) if f.startswith('meta_') and f.endswith('.csv')]
            if not csv_files:
                raise FileNotFoundError(f"No meta_*.csv file found in {self._data_dir}")
            filename = csv_files[0]
            logger.info(f"Using metadata file: {filename}")
        
        csv_path = os.path.join(self._data_dir, filename)
        logger.info(f"Loading metavalue: {csv_path}")
        
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

        # Open-source note: implementation detail.
        for k in meta_df.columns:
            if meta_df[k].dtype in [object, np.object_, "object", "O"]:
                meta_df[k] = meta_df[k].astype(str).str.replace(" ", "")

        # Open-source note: implementation detail.
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

        # Open-source note: implementation detail.
        # Open-source note: implementation detail.
        # Open-source note: implementation detail.
        # Open-source note: implementation detail.
        if "_npy_path" in meta_df.columns:
            def _resolve_npy(row) -> str:
                p = row.get("_npy_path", "")
                # Open-source note: implementation detail.
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

                # Open-source note: implementation detail.
                if part_i is not None:
                    cand = os.path.join(self._data_dir, "waves", f"{key}_{part_i}.npy")
                    if os.path.exists(cand):
                        return cand

                # Open-source note: implementation detail.
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
        data. 
        
        data diting.py value: 
        - data npy data, data HDF5
        - data
        """
        target_event = self._meta_data.iloc[idx]
        key = target_event["key"]

        def _load_npy_with_retry(path: str) -> np.ndarray:
            # Open-source note: implementation detail.
            last_err = None
            for attempt in range(6):
                try:
                    return np.load(path).astype(np.float32)
                except FileNotFoundError as e:
                    last_err = e
                    time.sleep([0.2, 0.5, 1.0, 2.0, 4.0, 8.0][attempt])
            raise last_err

        # Open-source note: implementation detail.
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
            # Open-source note: implementation detail.
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

        # Open-source note: implementation detail.
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

        # Open-source note: implementation detail.
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

        # Open-source note: implementation detail.
        # Open-source note: implementation detail.
        motion = None
        try:
            if pd.notnull(pmp_bin) and pmp_bin != "":
                motion = int(float(pmp_bin))
                if motion not in (0, 1):
                    motion = None
        except (ValueError, TypeError, AttributeError):
            motion = None

        # Open-source note: implementation detail.
        if pd.notnull(clarity):
            clarity = 0 if clarity.lower() == "i" else 1

        # Open-source note: implementation detail.
        if pd.notnull(baz):
            if baz == '':
                baz = np.nan
            else:
                baz = float(baz)
                baz = baz % 360

        # Open-source note: implementation detail.
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
            # Open-source note: implementation detail.
            # Open-source note: implementation detail.
            # Open-source note: implementation detail.
            pass

        evmag = np.clip(evmag, 0, 8, dtype=np.float32)
        stmag = np.clip(stmag, 0, 8, dtype=np.float32)

        snr = np.array([zpp_snr, nsp_snr, esp_snr])

        # Open-source note: implementation detail.
        # Open-source note: implementation detail.
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
    """Prepared DiTing 2.0 waveform dataset."""
    _name = "diting2_light"

