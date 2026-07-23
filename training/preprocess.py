import argparse
import copy
from operator import itemgetter
from typing import Any, List, Tuple, Union

import json
import numpy as np
from torch.utils.data import Dataset

from datasets import build_dataset
from utils import logger
from config import Config

__all__ = ["Preprocessor", "SeismicDataset"]


def _pad_phases(ppks: list, spks: list, padding_idx: int, num_samples: int) -> Tuple[list, list]:
    padding_idx = abs(padding_idx)
    ppks, spks = sorted(ppks), sorted(spks)
    ppks_, spks_ = ppks.copy(), spks.copy()
    ppk_arr, spk_arr = np.array(ppks), np.array(sorted(spks))
    idx = 0
    while idx < min(len(ppks), len(spks)) and all(ppk_arr[: idx + 1] < spk_arr[-idx - 1 :]):
        idx += 1
    ppks = len(spk_arr[: len(spk_arr) - idx]) * [-padding_idx] + ppks
    spks = spks + len(ppk_arr[idx:]) * [num_samples + padding_idx]
    assert len(ppks) == len(spks), f"Error:{ppks_} -> {ppks},{spks_} -> {spks}"
    return ppks, spks


def _pad_array(s: list, length: int, padding_value: Union[int, float]) -> np.ndarray:
    padding_size = int(length - len(s))
    if padding_size >= 0:
        padded = np.pad(s, (0, padding_size), mode="constant", constant_values=padding_value)
        return padded
    raise Exception(f"`length < len(s)` . Array:{len(s)},Target:{length}")


def copy_loss_targets(event: dict, dataset: str):
    dataset = str(dataset).lower()
    baz = event["baz"].copy()
    dis = event["dis"].copy()
    emg = event["emg"].copy()
    targets = {"baz": baz, "dis": dis, "emg": emg}
    if dataset.startswith("diting"):
        pmp = event["pmp"].copy()
        targets["pmp"] = pmp
    return targets


def get_copy_targets(targets: dict, event: dict, dataset: str):
    dataset = str(dataset).lower()
    event["baz"], event["dis"], event["emg"] = targets["baz"], targets["dis"], targets["emg"]
    if dataset.startswith("diting"):
        event["pmp"] = targets["pmp"]
    return event


class DataPreprocessor:
    """
    data/data(data, data, label data). 
    data, data validate.py / train.py data batch data. 
    """

    def __init__(
        self,
        dataset_name: str,
        data_channels: int,
        sampling_rate: int,
        in_samples: int,
        min_snr: float,
        p_position_ratio: float,
        coda_ratio: float,
        norm_mode: str,
        add_event_rate: float,
        add_noise_rate: float,
        add_gap_rate: float,
        drop_channel_rate: float,
        scale_amplitude_rate: float,
        pre_emphasis_rate: float,
        pre_emphasis_ratio: float,
        max_event_num: float,
        generate_noise_rate: float,
        shift_event_rate: float,
        mask_percent: float,
        noise_percent: float,
        min_event_gap_sec: float,
        soft_label_shape: str,
        soft_label_width: int,
        dtype=np.float32,
    ):
        # Open-source note: implementation detail.
        # Open-source note: implementation detail.
        self.dataset_name = dataset_name.split("_")[0]
        self.sampling_rate = sampling_rate
        self.data_channels = data_channels
        self.in_samples = in_samples
        self.coda_ratio = coda_ratio
        self.norm_mode = norm_mode
        self.min_snr = min_snr
        self.p_position_ratio = p_position_ratio
        self.add_event_rate = add_event_rate
        self.add_noise_rate = add_noise_rate
        self.add_gap_rate = add_gap_rate
        self.drop_channel_rate = drop_channel_rate
        self.scale_amplitude_rate = scale_amplitude_rate
        self.pre_emphasis_rate = pre_emphasis_rate
        self.pre_emphasis_ratio = pre_emphasis_ratio
        self._max_event_num = max_event_num
        self.generate_noise_rate = generate_noise_rate
        self.shift_event_rate = shift_event_rate
        self.mask_percent = mask_percent
        self.noise_percent = noise_percent
        self.min_event_gap = int(min_event_gap_sec * self.sampling_rate)
        self.soft_label_shape = soft_label_shape
        self.soft_label_width = soft_label_width
        self.dtype = dtype

    # Open-source note: implementation detail.
    # Open-source note: implementation detail.

    def _clear_dict_except(self, d: dict, *args) -> None:
        if len(args) > 0:
            for arg in args:
                assert isinstance(arg, str), f"Input arguments must be str, got `{arg}`({type(arg)})"
        for k in set(d) - set(args):
            if isinstance(d[k], (list, dict)):
                d[k].clear()
            elif isinstance(d[k], np.ndarray):
                d[k] = np.array([])
            elif isinstance(d[k], (int, float)):
                d[k] = 0
            elif isinstance(d[k], str):
                d[k] = ""
            else:
                raise TypeError(f"Got `{d[k]}`({type(d[k])})")

    def _normalize(self, data, mode):
        data -= np.mean(data, axis=1, keepdims=True)
        if mode == "max":
            max_data = np.max(data, axis=1, keepdims=True)
            max_data[max_data == 0] = 1
            data /= max_data
        elif mode == "std":
            std_data = np.std(data, axis=1, keepdims=True)
            std_data[std_data == 0] = 1
            data /= std_data
        elif mode == "":
            pass
        return data

    def _augment_waveform(self, value: np.ndarray) -> np.ndarray:
        """
        value: data [C, L] data. 
        data SeismicDataset data"data idx data augmentation=True"data; data. 
        """
        rng = np.random.default_rng()
        data = np.asarray(data, dtype=self.dtype)
        if data.ndim != 2:
            return data
        c, length = data.shape[0], data.shape[1]

        if rng.random() < self.shift_event_rate:
            max_shift = max(1, int(0.05 * length))
            shift = int(rng.integers(-max_shift, max_shift + 1))
            data = np.roll(data, shift, axis=-1)

        if rng.random() < self.add_noise_rate:
            std = float(np.std(data) + 1e-6)
            data = data + rng.normal(0, 0.03 * std, size=data.shape).astype(self.dtype)

        if rng.random() < self.add_gap_rate:
            gap_len = max(1, int(rng.uniform(0.05, 0.2) * length))
            gap_start = int(rng.integers(0, max(1, length - gap_len + 1)))
            data[:, gap_start : gap_start + gap_len] = 0

        if rng.random() < self.drop_channel_rate:
            ch = int(rng.integers(0, c))
            data[ch] = 0

        if rng.random() < self.scale_amplitude_rate:
            data *= float(rng.uniform(0.7, 1.4))

        if rng.random() < self.pre_emphasis_rate:
            alpha = float(self.pre_emphasis_ratio)
            x_prev = np.concatenate(
                [np.zeros((c, 1), dtype=self.dtype), data[:, :-1]], axis=1
            )
            data = data - alpha * x_prev

        if rng.random() < self.generate_noise_rate:
            std = float(np.std(data) + 1e-6)
            data = rng.normal(0, std, size=data.shape).astype(self.dtype)

        return data

    def _make_phase_heatmap(self, centers: List[int]) -> np.ndarray:
        """
        Build soft heatmap for phase picking loss.

        Output: float array with shape [in_samples], range [0,1].
        """
        length = int(self.in_samples)
        t = np.arange(length, dtype=np.float32)

        w = int(self.soft_label_width)
        # avoid degenerate widths
        if w <= 0:
            w = 1

        heat = np.zeros((length,), dtype=np.float32)
        if not centers:
            return heat

        for c in centers:
            try:
                ci = int(c)
            except Exception:
                continue
            if ci < 0 or ci >= length:
                continue

            shape = str(self.soft_label_shape).lower()
            if shape == "gaussian":
                sigma = float(max(1, w))
                heat_c = np.exp(-0.5 * ((t - float(ci)) / sigma) ** 2, dtype=np.float32)
            elif shape == "triangle":
                # linear decay to 0 within +/-w
                heat_c = np.maximum(0.0, 1.0 - (np.abs(t - float(ci)) / float(w)))
            elif shape == "box":
                # constant 1 within +/-w
                heat_c = (np.abs(t - float(ci)) <= float(w)).astype(np.float32)
            elif shape == "sigmoid":
                # smooth step around +/-w
                scale = float(max(1, w / 2))
                dist = np.abs(t - float(ci)) - float(w)
                heat_c = 1.0 / (1.0 + np.exp(dist / scale, dtype=np.float32))
                heat_c = heat_c.astype(np.float32)
            else:
                # fallback: gaussian
                sigma = float(max(1, w))
                heat_c = np.exp(-0.5 * ((t - float(ci)) / sigma) ** 2, dtype=np.float32)

            heat = np.maximum(heat, heat_c.astype(np.float32))

        return heat

    def process(self, event: dict, augmentation: bool = False) -> dict:
        # Open-source note: implementation detail.
        if "data" in event and isinstance(event["data"], np.ndarray):
            event["data"] = event["data"].astype(self.dtype)
            if event["data"].ndim == 2:
                if augmentation:
                    event["data"] = self._augment_waveform(event["data"])
                event["data"] = self._normalize(event["data"], self.norm_mode)
        return event

    def _get_io_item(self, name: str, event: dict):
        # EVT datasets (e.g., diting2_evt6) store waveform as event["data"] with shape [C, L].
        # Map channel names to corresponding slices when present.
        if isinstance(name, str):
            if name in ("z", "n", "e"):
                if "data" in event:
                    idx = {"z": 0, "n": 1, "e": 2}[name]
                    return event["data"][idx]
            if name == "data" and "data" in event:
                return event["data"]
            # Auxiliary targets derived from evt6 label (for multi-auxiliary supervision)
            if name in ("evt6_seot_vs_others", "evt6_se_vs_ot"):
                # evt6 id mapping: 0 eq,1 ep,2 co/ss,3 sp,4 se,5 ot
                if "evt6" not in event:
                    raise KeyError("evt6")
                cls_list = event["evt6"]
                if isinstance(cls_list, list) and len(cls_list) == 1:
                    cls_id = int(cls_list[0])
                elif isinstance(cls_list, (int, np.integer)):
                    cls_id = int(cls_list)
                else:
                    # fallback
                    cls_id = int(cls_list[0]) if hasattr(cls_list, "__len__") and len(cls_list) else -1

                if name == "evt6_seot_vs_others":
                    # binary: {se,ot} vs others
                    return [1] if cls_id in (4, 5) else [0]

                # name == "evt6_se_vs_ot": binary within {se,ot}; for other classes provide a deterministic label.
                # convention: ot->0, se->1
                if cls_id == 4:
                    return [1]
                if cls_id == 5:
                    return [0]
                return [0]
            return event[name]

        # Grouped channel request like ["z","n","e"].
        if isinstance(name, (list, tuple)):
            names = list(name)
            if names == ["z", "n", "e"] and "data" in event:
                return event["data"]
            parts = [self._get_io_item(n, event) for n in names]
            # Stack as [C, L] if each part is [L]
            if all(isinstance(p, np.ndarray) and p.ndim == 1 for p in parts):
                return np.stack(parts, axis=0)
            return tuple(parts)

        raise TypeError(f"Unsupported io item name type: {type(name)}")

    def get_targets_for_loss(self, event: dict, label_names: list):
        targets = []
        for name in label_names:
            # Phase picking supervision for SeisMoLLM_dpk:
            # config labels: [["non","ppk","spk"]]
            # - loss expects [N,3,L] with non/ppk/spk time heatmaps
            if isinstance(name, (list, tuple)) and len(name) > 0:
                if all(isinstance(n, str) for n in name) and all(n in ("non", "ppk", "spk") for n in name):
                    ppks = event.get("ppks", []) or []
                    spks = event.get("spks", []) or []
                    heat_ppk = self._make_phase_heatmap(ppks)
                    heat_spk = self._make_phase_heatmap(spks)
                    heat_non = np.zeros_like(heat_ppk, dtype=np.float32)

                    parts = []
                    for n in name:
                        if n == "non":
                            parts.append(heat_non)
                        elif n == "ppk":
                            parts.append(heat_ppk)
                        elif n == "spk":
                            parts.append(heat_spk)
                        else:
                            raise KeyError(f"Unknown picking label: {n}")

                    tgt = np.stack(parts, axis=0)  # [C, L]
                    targets.append(tgt)
                    continue

            # Label names are strings (evt6/evt6_sp/...) in our EVT configs.
            tgt = self._get_io_item(name, event)
            # Convert class index list -> one-hot (CELoss expects one-hot targets)
            if isinstance(name, str) and name in Config._avl_io_items and Config._avl_io_items[name].get("type") == "onehot":
                num_classes = int(Config._avl_io_items[name].get("num_classes"))
                if isinstance(tgt, list) and len(tgt) == 1:
                    cls = int(tgt[0])
                    oh = np.zeros((num_classes,), dtype=np.float32)
                    oh[cls] = 1.0
                    tgt = oh
                elif isinstance(tgt, (int, np.integer)):
                    cls = int(tgt)
                    oh = np.zeros((num_classes,), dtype=np.float32)
                    oh[cls] = 1.0
                    tgt = oh
            targets.append(tgt)
        if len(targets) > 1:
            return tuple(targets)
        return targets.pop()

    def get_targets_for_metrics(self, event: dict, task_names: list, max_event_num: int):
        targets = {}
        for name in task_names:
            # Phase picking metrics supervision:
            # Metrics expect:
            #  - preds: phases indices [B, topk] (see postprocess._pick_phase)
            #  - targets: phases indices [B, topk]
            # Here we convert event["ppks"]/event["spks"] to padded index lists.
            if name in ("ppk", "spk"):
                topk = int(max_event_num) if max_event_num is not None else 1
                if topk <= 0:
                    topk = 1
                padding_value = int(-1e7)
                centers = event.get("ppks", []) if name == "ppk" else event.get("spks", [])
                centers = centers or []
                centers_int = []
                for c in centers:
                    try:
                        ci = int(c)
                    except Exception:
                        continue
                    centers_int.append(ci)
                centers_int = sorted(centers_int)
                centers_int = centers_int[:topk]
                centers_int = centers_int + [padding_value] * (topk - len(centers_int))
                targets[name] = np.array(centers_int, dtype=np.int64)
                continue

            tgt = self._get_io_item(name, event)
            # Metrics for classification tasks expect one-hot targets (see utils/metrics.py assertions).
            if (
                isinstance(name, str)
                and name in Config._avl_io_items
                and Config._avl_io_items[name].get("type") == "onehot"
            ):
                num_classes = int(Config._avl_io_items[name].get("num_classes"))
                if isinstance(tgt, list) and len(tgt) == 1:
                    cls = int(tgt[0])
                    oh = np.zeros((num_classes,), dtype=np.float32)
                    oh[cls] = 1.0
                    tgt = oh
                elif isinstance(tgt, (int, np.integer)):
                    cls = int(tgt)
                    oh = np.zeros((num_classes,), dtype=np.float32)
                    oh[cls] = 1.0
                    tgt = oh
            targets[name] = tgt
        return targets

    def get_inputs(self, event: dict, input_names: list) -> Union[np.ndarray, tuple]:
        inputs = [self._get_io_item(name=name, event=event) for name in input_names]
        if len(inputs) > 1:
            return tuple(inputs)
        return inputs.pop()


class SeismicDataset(Dataset):
    def __init__(
        self,
        args: argparse.Namespace,
        input_names: list,
        label_names: list,
        task_names: list,
        mode: str,
    ) -> None:
        self._seed = int(args.seed)
        self._mode = mode.lower()
        self._input_names = input_names
        self._label_names = label_names
        self._task_names = task_names
        self._max_event_num = args.max_event_num

        self._augmentation = args.augmentation and self._mode == "train"
        if self._augmentation != args.augmentation:
            logger.warning(f"[{self._mode}]Augmentation -> {self._augmentation}")

        self._dataset = build_dataset(
            dataset_name=args.dataset_name,
            seed=self._seed,
            mode=self._mode,
            data_dir=args.data,
            shuffle=args.shuffle,
            data_split=args.data_split,
            train_size=args.train_size,
            val_size=args.val_size,
        )
        logger.info(self._dataset)

        self._dataset_size = len(self._dataset)
        if self._augmentation:
            logger.warning(f"Data augmentation: Dataset size -> {self._dataset_size *2}")

        self._preprocessor = DataPreprocessor(
            dataset_name=args.dataset_name,
            data_channels=self._dataset.channels(),
            sampling_rate=self._dataset.sampling_rate(),
            in_samples=args.in_samples,
            min_snr=args.min_snr,
            coda_ratio=args.coda_ratio,
            norm_mode=args.norm_mode,
            p_position_ratio=args.p_position_ratio,
            add_event_rate=args.add_event_rate,
            add_noise_rate=args.add_noise_rate,
            add_gap_rate=args.add_gap_rate,
            drop_channel_rate=args.drop_channel_rate,
            scale_amplitude_rate=args.scale_amplitude_rate,
            pre_emphasis_rate=args.pre_emphasis_rate,
            pre_emphasis_ratio=args.pre_emphasis_ratio,
            max_event_num=args.max_event_num,
            generate_noise_rate=args.generate_noise_rate,
            shift_event_rate=args.shift_event_rate,
            mask_percent=args.mask_percent,
            noise_percent=args.noise_percent,
            min_event_gap_sec=args.min_event_gap,
            soft_label_shape=args.label_shape,
            soft_label_width=int(args.label_width * self._dataset.sampling_rate()),
            dtype=np.float32,
        )

    def sampling_rate(self):
        return self._dataset.sampling_rate()

    def data_channels(self):
        return self._dataset.channels()

    def name(self):
        return f"{self._dataset.name()}_{self._mode}"

    def __len__(self) -> int:
        return 2 * self._dataset_size if self._augmentation else self._dataset_size

    def __getitem__(self, idx: int) -> Tuple[Any, Any, Any]:
        event, meta_data = self._dataset[idx % self._dataset_size]
        event = self._preprocessor.process(event=event, augmentation=(self._augmentation and idx >= self._dataset_size))
        inputs = self._preprocessor.get_inputs(event=event, input_names=self._input_names)
        loss_targets = self._preprocessor.get_targets_for_loss(event=event, label_names=self._label_names)
        metrics_targets = self._preprocessor.get_targets_for_metrics(
            event=event, task_names=self._task_names, max_event_num=self._max_event_num
        )
        meta_data_json = json.dumps(meta_data)
        return inputs, loss_targets, metrics_targets, meta_data_json

