from collections import defaultdict
import math
import re
from functools import partial
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch

from models import (
    BCELoss,
    BinaryFocalLoss,
    CELoss,
    ClassBalancedCELoss,
    ConbinationLoss,
    FocalLoss,
    HuberLoss,
    MousaviLoss,
    MSELoss,
    get_model_list,
)


class Config:
    """
    data `logs/*/global.log` data `Configs:` data, data/value: 
    - model registry -> loss / inputs / labels / eval / transforms
    - IO item metadata -> type / metrics / num_classes
    """

    _model_conf_keys = (
        "loss",
        "labels",
        "eval",
        "outputs_transform_for_loss",
        "outputs_transform_for_results",
    )

    # Models (regex keys supported)
    models: Dict[str, Dict[str, Any]] = {
        "phasenet": {
            "loss": partial(CELoss, weight=[[1], [1], [1]]),
            "inputs": [["z", "n", "e"]],
            "labels": [["non", "ppk", "spk"]],
            "eval": ["ppk", "spk"],
            "targets_transform_for_loss": None,
            "outputs_transform_for_loss": None,
            "outputs_transform_for_results": None,
        },
        "eqtransformer": {
            "loss": partial(BCELoss, weight=[[0.05], [0.4], [0.55]]),
            "inputs": [["z", "n", "e"]],
            "labels": [["det", "ppk", "spk"]],
            "eval": ["det", "ppk", "spk"],
            "targets_transform_for_loss": None,
            "outputs_transform_for_loss": None,
            "outputs_transform_for_results": None,
        },
        "magnet": {
            "loss": MousaviLoss,
            "inputs": [["z", "n", "e"]],
            "labels": ["emg"],
            "eval": ["emg"],
            "targets_transform_for_loss": None,
            "outputs_transform_for_loss": None,
            "outputs_transform_for_results": lambda x: x[:, 0].reshape(-1, 1),
        },
        "baz_network": {
            "loss": partial(ConbinationLoss, losses=[MSELoss, MSELoss]),
            "inputs": [["z", "n", "e"]],
            "labels": ["baz"],
            "eval": ["baz"],
            "targets_transform_for_loss": lambda x: (
                (x * math.pi / 180).cos(),
                (x * math.pi / 180).sin(),
            ),
            "outputs_transform_for_loss": None,
            "outputs_transform_for_results": lambda x: torch.atan2(x[1], x[0])
            * 180
            / math.pi,
        },
        "distpt_network": {
            "loss": partial(
                ConbinationLoss,
                losses=[MousaviLoss, MousaviLoss],
                losses_weights=[0.1, 0.9],
            ),
            "inputs": [["z", "n", "e"]],
            "labels": ["dis", "ptt"],
            "eval": ["dis", "ptt"],
            "targets_transform_for_loss": None,
            "outputs_transform_for_loss": None,
            "outputs_transform_for_results": lambda x: (
                x[0][:, 0].reshape(-1, 1),
                x[1][:, 0].reshape(-1, 1),
            ),
        },
        "ditingmotion": {
            "loss": partial(ConbinationLoss, losses=[FocalLoss, FocalLoss]),
            "inputs": [["z", "dz"]],
            "labels": ["clr", "pmp"],
            "eval": ["pmp"],
            "targets_transform_for_loss": None,
            "outputs_transform_for_loss": None,
            "outputs_transform_for_results": lambda xs: [x.softmax(-1) for x in xs],
        },
        "seist_.*?_dpk.*": {
            "loss": partial(BCELoss, weight=[[0.5], [1], [1]]),
            "inputs": [["z", "n", "e"]],
            "labels": [["det", "ppk", "spk"]],
            "eval": ["det", "ppk", "spk"],
            "targets_transform_for_loss": None,
            "outputs_transform_for_loss": None,
            "outputs_transform_for_results": None,
        },
        "seist_.*?_pmp": {
            "loss": partial(CELoss, weight=[1, 1]),
            "inputs": [["z", "n", "e"]],
            "labels": ["pmp"],
            "eval": ["pmp"],
            "targets_transform_for_loss": None,
            "outputs_transform_for_loss": None,
            "outputs_transform_for_results": None,
        },
        "seist_.*?_emg": {
            "loss": HuberLoss,
            "inputs": [["z", "n", "e"]],
            "labels": ["emg"],
            "eval": ["emg"],
            "targets_transform_for_loss": None,
            "outputs_transform_for_loss": None,
            "outputs_transform_for_results": None,
        },
        "seist_.*?_baz": {
            "loss": HuberLoss,
            "inputs": [["z", "n", "e"]],
            "labels": ["baz"],
            "eval": ["baz"],
            "targets_transform_for_loss": None,
            "outputs_transform_for_loss": None,
            "outputs_transform_for_results": None,
        },
        "seist_.*?_dis": {
            "loss": HuberLoss,
            "inputs": [["z", "n", "e"]],
            "labels": ["dis"],
            "eval": ["dis"],
            "targets_transform_for_loss": None,
            "outputs_transform_for_loss": None,
            "outputs_transform_for_results": None,
        },
        "SeisMoLLM_dpk.*": {
            "loss": partial(BCELoss, weight=[[0], [1], [1]]),
            "inputs": [["z", "n", "e"]],
            "labels": [["non", "ppk", "spk"]],
            "eval": ["ppk", "spk"],
            "targets_transform_for_loss": None,
            "outputs_transform_for_loss": None,
            "outputs_transform_for_results": None,
        },
        # Class-conditioned phase picking (evt6_cond injection inside model).
        "SeisMoLLM_dpk_cond.*": {
            "loss": partial(BCELoss, weight=[[0], [1], [1]]),
            "inputs": [["z", "n", "e"], "evt6_cond"],
            "labels": [["non", "ppk", "spk"]],
            "eval": ["ppk", "spk"],
            "targets_transform_for_loss": None,
            "outputs_transform_for_loss": None,
            "outputs_transform_for_results": None,
        },
        # For ablation/diagnosis: early-stop uses SPK instead of PPK.
        "SeisMoLLM_dpk_cond_spkbest.*": {
            "loss": partial(BCELoss, weight=[[0], [1], [1]]),
            "inputs": [["z", "n", "e"], "evt6_cond"],
            "labels": [["non", "ppk", "spk"]],
            "eval": ["spk", "ppk"],
            "targets_transform_for_loss": None,
            "outputs_transform_for_loss": None,
            "outputs_transform_for_results": None,
        },
        "SeisMoLLM_pmp": {
            "loss": partial(CELoss, weight=[1, 1]),
            "inputs": [["z", "n", "e"]],
            "labels": ["pmp"],
            "eval": ["pmp"],
            "targets_transform_for_loss": None,
            "outputs_transform_for_loss": None,
            "outputs_transform_for_results": None,
        },
        # EVT classifiers
        "^SeisMoLLM_evt6$": {
            "loss": partial(CELoss, weight=[1, 1, 1, 1, 1, 1]),
            "inputs": [["z", "n", "e"]],
            "labels": ["evt6"],
            "eval": ["evt6"],
            "targets_transform_for_loss": None,
            "outputs_transform_for_loss": None,
            "outputs_transform_for_results": None,
        },
        # EVT6 channel ablation (single-component vs 3C) for seismic cross-app analysis.
        # These are model-name aliases only; they reuse the same architecture/training loop.
        "^SeisMoLLM_evt6_z$": {
            "loss": partial(CELoss, weight=[1, 1, 1, 1, 1, 1]),
            "inputs": [["z"]],
            "labels": ["evt6"],
            "eval": ["evt6"],
            "targets_transform_for_loss": None,
            "outputs_transform_for_loss": None,
            "outputs_transform_for_results": None,
        },
        "^SeisMoLLM_evt6_n$": {
            "loss": partial(CELoss, weight=[1, 1, 1, 1, 1, 1]),
            "inputs": [["n"]],
            "labels": ["evt6"],
            "eval": ["evt6"],
            "targets_transform_for_loss": None,
            "outputs_transform_for_loss": None,
            "outputs_transform_for_results": None,
        },
        "^SeisMoLLM_evt6_e$": {
            "loss": partial(CELoss, weight=[1, 1, 1, 1, 1, 1]),
            "inputs": [["e"]],
            "labels": ["evt6"],
            "eval": ["evt6"],
            "targets_transform_for_loss": None,
            "outputs_transform_for_loss": None,
            "outputs_transform_for_results": None,
        },
        "^SeisMoLLM_evt6_zne$": {
            "loss": partial(CELoss, weight=[1, 1, 1, 1, 1, 1]),
            "inputs": [["z", "n", "e"]],
            "labels": ["evt6"],
            "eval": ["evt6"],
            "targets_transform_for_loss": None,
            "outputs_transform_for_loss": None,
            "outputs_transform_for_results": None,
        },
        "^SeisMoLLM_evt6_cw$": {
            "loss": partial(CELoss, weight=[1, 1, 1, 1, 1, 1]),
            "inputs": [["z", "n", "e"]],
            "labels": ["evt6"],
            "eval": ["evt6"],
            "targets_transform_for_loss": None,
            "outputs_transform_for_loss": None,
            "outputs_transform_for_results": None,
        },
        "^SeisMoLLM_evt6_ds2$": {
            "loss": partial(CELoss, weight=[1, 1, 1, 1, 1, 1]),
            "inputs": [["z", "n", "e"]],
            "labels": ["evt6"],
            "eval": ["evt6"],
            "targets_transform_for_loss": None,
            "outputs_transform_for_loss": None,
            "outputs_transform_for_results": None,
        },
        "^SeisMoLLM_evt6_ps4$": {
            "loss": partial(CELoss, weight=[1, 1, 1, 1, 1, 1]),
            "inputs": [["z", "n", "e"]],
            "labels": ["evt6"],
            "eval": ["evt6"],
            "targets_transform_for_loss": None,
            "outputs_transform_for_loss": None,
            "outputs_transform_for_results": None,
        },
        "^SeisMoLLM_evt6_ps2$": {
            "loss": partial(CELoss, weight=[1, 1, 1, 1, 1, 1]),
            "inputs": [["z", "n", "e"]],
            "labels": ["evt6"],
            "eval": ["evt6"],
            "targets_transform_for_loss": None,
            "outputs_transform_for_loss": None,
            "outputs_transform_for_results": None,
        },
        "^SeisMoLLM_evt6_attnpool$": {
            "loss": partial(CELoss, weight=[1, 1, 1, 1, 1, 1]),
            "inputs": [["z", "n", "e"]],
            "labels": ["evt6"],
            "eval": ["evt6"],
            "targets_transform_for_loss": None,
            "outputs_transform_for_loss": None,
            "outputs_transform_for_results": None,
        },
        "^SeisMoLLM_evt6_coarse_only$": {
            "loss": partial(CELoss, weight=[1, 1]),
            "inputs": [["z", "n", "e"]],
            "labels": ["evt6_sp"],
            "eval": ["evt6_sp"],
            "targets_transform_for_loss": None,
            "outputs_transform_for_loss": None,
            "outputs_transform_for_results": None,
        },
        "^SeisMoLLM_evt6_hier_sp_phase2$": {
            "loss": partial(CELoss, weight=[1, 1, 1, 1, 1, 1]),
            "inputs": [["z", "n", "e"]],
            "labels": ["evt6"],
            "eval": ["evt6"],
            "targets_transform_for_loss": None,
            "outputs_transform_for_loss": lambda outs: outs[0] if isinstance(outs, (tuple, list)) else outs,
            "outputs_transform_for_results": lambda outs: outs[0] if isinstance(outs, (tuple, list)) else outs,
        },
        "^SeisMoLLM_evt6_hier_sp$": {
            "loss": partial(
                ConbinationLoss,
                losses=[partial(CELoss, weight=[1, 1]), partial(CELoss, weight=[1, 1, 1, 1, 1, 1])],
                losses_weights=[0.3, 1.0],
            ),
            "inputs": [["z", "n", "e"]],
            "labels": ["evt6_sp", "evt6"],
            "eval": ["evt6"],
            "targets_transform_for_loss": None,
            "outputs_transform_for_loss": None,
            "outputs_transform_for_results": None,
        },
        "^SeisMoLLM_evt6_hier_sp_w01$": {
            "loss": partial(
                ConbinationLoss,
                losses=[partial(CELoss, weight=[1, 1]), partial(CELoss, weight=[1, 1, 1, 1, 1, 1])],
                losses_weights=[0.1, 1.0],
            ),
            "inputs": [["z", "n", "e"]],
            "labels": ["evt6_sp", "evt6"],
            "eval": ["evt6"],
            "targets_transform_for_loss": None,
            "outputs_transform_for_loss": None,
            "outputs_transform_for_results": None,
        },
        "^SeisMoLLM_evt6_multihead$": {
            "loss": partial(
                ConbinationLoss,
                losses=[
                    partial(CELoss, weight=[1, 1, 1, 1, 1, 1]),
                    partial(CELoss, weight=[1, 1]),
                    partial(CELoss, weight=[1, 1]),
                    partial(CELoss, weight=[1, 1]),
                ],
                losses_weights=[1.0, 0.3, 0.3, 0.0],
            ),
            "inputs": [["z", "n", "e"]],
            "labels": ["evt6", "evt6_sp", "evt6_seot_vs_others", "evt6_se_vs_ot"],
            "eval": ["evt6"],
            "targets_transform_for_loss": None,
            "outputs_transform_for_loss": None,
            "outputs_transform_for_results": None,
        },
        "^SeisMoLLM_evt6_multihead_w01$": {
            "loss": partial(
                ConbinationLoss,
                losses=[
                    partial(CELoss, weight=[1, 1, 1, 1, 1, 1]),
                    partial(CELoss, weight=[1, 1]),
                    partial(CELoss, weight=[1, 1]),
                    partial(CELoss, weight=[1, 1]),
                ],
                losses_weights=[1.0, 0.1, 0.1, 0.0],
            ),
            "inputs": [["z", "n", "e"]],
            "labels": ["evt6", "evt6_sp", "evt6_seot_vs_others", "evt6_se_vs_ot"],
            "eval": ["evt6"],
            "targets_transform_for_loss": None,
            "outputs_transform_for_loss": None,
            "outputs_transform_for_results": None,
        },
        "^SeisMoLLM_evt6_multihead_w005$": {
            "loss": partial(
                ConbinationLoss,
                losses=[
                    partial(CELoss, weight=[1, 1, 1, 1, 1, 1]),
                    partial(CELoss, weight=[1, 1]),
                    partial(CELoss, weight=[1, 1]),
                    partial(CELoss, weight=[1, 1]),
                ],
                losses_weights=[1.0, 0.05, 0.05, 0.0],
            ),
            "inputs": [["z", "n", "e"]],
            "labels": ["evt6", "evt6_sp", "evt6_seot_vs_others", "evt6_se_vs_ot"],
            "eval": ["evt6"],
            "targets_transform_for_loss": None,
            "outputs_transform_for_loss": None,
            "outputs_transform_for_results": None,
        },
        # Spectrogram stem + same GPT-2 stack / heads as non-spec variants (cross-input ablation).
        "^SeisMoLLM_evt6_hier_sp_specgpt2$": {
            "loss": partial(
                ConbinationLoss,
                losses=[partial(CELoss, weight=[1, 1]), partial(CELoss, weight=[1, 1, 1, 1, 1, 1])],
                losses_weights=[0.3, 1.0],
            ),
            "inputs": [["z", "n", "e"]],
            "labels": ["evt6_sp", "evt6"],
            "eval": ["evt6"],
            "targets_transform_for_loss": None,
            "outputs_transform_for_loss": None,
            "outputs_transform_for_results": None,
        },
        "^SeisMoLLM_evt6_multihead_w01_specgpt2$": {
            "loss": partial(
                ConbinationLoss,
                losses=[
                    partial(CELoss, weight=[1, 1, 1, 1, 1, 1]),
                    partial(CELoss, weight=[1, 1]),
                    partial(CELoss, weight=[1, 1]),
                    partial(CELoss, weight=[1, 1]),
                ],
                losses_weights=[1.0, 0.1, 0.1, 0.0],
            ),
            "inputs": [["z", "n", "e"]],
            "labels": ["evt6", "evt6_sp", "evt6_seot_vs_others", "evt6_se_vs_ot"],
            "eval": ["evt6"],
            "targets_transform_for_loss": None,
            "outputs_transform_for_loss": None,
            "outputs_transform_for_results": None,
        },
        "^SeisMoLLM_evt5$": {
            "loss": partial(CELoss, weight=[1, 1, 1, 1, 1]),
            "inputs": [["z", "n", "e"]],
            "labels": ["evt5"],
            "eval": ["evt5"],
            "targets_transform_for_loss": None,
            "outputs_transform_for_loss": None,
            "outputs_transform_for_results": None,
        },
        "^SeisMoLLM_evt3$": {
            "loss": partial(CELoss, weight=[1, 1, 1]),
            "inputs": [["z", "n", "e"]],
            "labels": ["evt3"],
            "eval": ["evt3"],
            "targets_transform_for_loss": None,
            "outputs_transform_for_loss": None,
            "outputs_transform_for_results": None,
        },
        "^SeisMoLLM_evt3_ps4$": {
            "loss": partial(CELoss, weight=[1, 1, 1]),
            "inputs": [["z", "n", "e"]],
            "labels": ["evt3"],
            "eval": ["evt3"],
            "targets_transform_for_loss": None,
            "outputs_transform_for_loss": None,
            "outputs_transform_for_results": None,
        },
        "^SeisMoLLM_evt3_ps2$": {
            "loss": partial(CELoss, weight=[1, 1, 1]),
            "inputs": [["z", "n", "e"]],
            "labels": ["evt3"],
            "eval": ["evt3"],
            "targets_transform_for_loss": None,
            "outputs_transform_for_loss": None,
            "outputs_transform_for_results": None,
        },
        "SeisMoLLM_emg": {
            "loss": HuberLoss,
            "inputs": [["z", "n", "e"]],
            "labels": ["emg"],
            "eval": ["emg"],
            "targets_transform_for_loss": None,
            "outputs_transform_for_loss": None,
            "outputs_transform_for_results": None,
        },
        "SeisMoLLM_baz": {
            "loss": partial(ConbinationLoss, losses=[HuberLoss, HuberLoss]),
            "inputs": [["z", "n", "e"]],
            "labels": ["baz"],
            "eval": ["baz"],
            "targets_transform_for_loss": lambda x: (
                (x * math.pi / 180).cos(),
                (x * math.pi / 180).sin(),
            ),
            "outputs_transform_for_loss": None,
            "outputs_transform_for_results": lambda x: torch.atan2(x[1], x[0])
            * 180
            / math.pi,
        },
        "SeisMoLLM_dis": {
            "loss": HuberLoss,
            "inputs": [["z", "n", "e"]],
            "labels": ["dis"],
            "eval": ["dis"],
            "targets_transform_for_loss": None,
            "outputs_transform_for_loss": None,
            "outputs_transform_for_results": None,
        },
    }

    _avl_metrics = (
        "precision",
        "recall",
        "f1",
        "accuracy",
        "mean",
        "std",
        "mae",
        "mape",
        "rmse",
        "r2",
    )

    _avl_io_item_types = ("soft", "value", "onehot")

    _avl_io_items: Dict[str, Dict[str, Any]] = {
        "z": {"type": "soft", "metrics": ["mean", "std", "mae"]},
        "n": {"type": "soft", "metrics": ["mean", "std", "mae"]},
        "e": {"type": "soft", "metrics": ["mean", "std", "mae"]},
        "dz": {"type": "soft", "metrics": ["mean", "std", "mae"]},
        "dn": {"type": "soft", "metrics": ["mean", "std", "mae"]},
        "de": {"type": "soft", "metrics": ["mean", "std", "mae"]},
        "non": {"type": "soft", "metrics": []},
        "det": {"type": "soft", "metrics": ["precision", "recall", "f1"]},
        "ppk": {
            "type": "soft",
            "metrics": ["precision", "recall", "f1", "mean", "std", "mae", "mape", "rmse"],
        },
        "spk": {
            "type": "soft",
            "metrics": ["precision", "recall", "f1", "mean", "std", "mae", "mape", "rmse"],
        },
        "ppk+": {"type": "soft", "metrics": []},
        "spk+": {"type": "soft", "metrics": []},
        "det+": {"type": "soft", "metrics": []},
        "ppks": {"type": "value", "metrics": ["mean", "std", "mae", "mape", "rmse", "r2"]},
        "spks": {"type": "value", "metrics": ["mean", "std", "mae", "mape", "rmse", "r2"]},
        "emg": {"type": "value", "metrics": ["mean", "std", "mae", "rmse", "r2"]},
        "smg": {"type": "value", "metrics": ["mean", "std", "mae", "rmse", "r2"]},
        "baz": {"type": "value", "metrics": ["mean", "std", "mae", "rmse", "r2"]},
        "dis": {"type": "value", "metrics": ["mean", "std", "mae", "rmse", "r2"]},
        "ptt": {"type": "value", "metrics": ["mean", "std", "mae", "r2"]},
        "pmp": {"type": "onehot", "metrics": ["precision", "recall", "f1"], "num_classes": 2},
        "clr": {"type": "onehot", "metrics": ["precision", "recall", "f1"], "num_classes": 2},
        "evt6": {
            "type": "onehot",
            "metrics": ["accuracy", "precision", "recall", "f1"],
            "num_classes": 6,
        },
        "evt6_sp": {"type": "onehot", "metrics": ["accuracy"], "num_classes": 2},
        "evt6_seot_vs_others": {"type": "onehot", "metrics": ["accuracy"], "num_classes": 2},
        "evt6_se_vs_ot": {"type": "onehot", "metrics": ["accuracy"], "num_classes": 2},
        "evt5": {
            "type": "onehot",
            "metrics": ["accuracy", "precision", "recall", "f1"],
            "num_classes": 5,
        },
        "evt3": {
            "type": "onehot",
            "metrics": ["accuracy", "precision", "recall", "f1"],
            "num_classes": 3,
        },
    }

    _type_to_ioitems: defaultdict = defaultdict(list)
    for _name, _meta in _avl_io_items.items():
        _type_to_ioitems[_meta["type"]].append(_name)

    @classmethod
    def get_model_list(cls):
        return get_model_list(cls.models)

    @classmethod
    def get_model_config(cls, model_name: str) -> Dict[str, Any]:
        for model, conf in cls.models.items():
            if re.match(f"^{model}$", model_name):
                return conf
        raise KeyError(f"model_name '{model_name}' not found in Config.models")

    @classmethod
    def get_model_config_(cls, model_name: str, *keys):
        conf = cls.get_model_config(model_name)
        return tuple(conf[k] for k in keys)

    @classmethod
    def get_num_inchannels(cls, model_name: str) -> int:
        inputs = cls.get_model_config_(model_name, "inputs")[0]
        # Only count waveform channel inputs.
        # Conditional scalar inputs (e.g., evt6_cond) must not affect in_channels.
        waveform_channels = {"z", "n", "e"}
        ch = 0
        for item in inputs:
            if isinstance(item, (list, tuple)):
                # e.g. ["z","n","e"]
                ch += sum(1 for x in item if isinstance(x, str) and x in waveform_channels)
            elif isinstance(item, str):
                if item in waveform_channels:
                    ch += 1
            # else: ignore (e.g. evt6_cond)
        return int(ch)

    @classmethod
    def get_loss(cls, model_name: str):
        conf = cls.get_model_config(model_name)
        loss = conf["loss"]
        # loss can be a class, a partial, or a callable returning nn.Module.
        if isinstance(loss, type):
            return loss()
        return loss()

    @classmethod
    def get_metrics(cls, task: str) -> List[str]:
        task = str(task)
        meta = cls._avl_io_items.get(task, None)
        if meta is None:
            return []
        return list(meta.get("metrics", []))

    @classmethod
    def get_type(cls, name: str) -> str:
        meta = cls._avl_io_items.get(str(name), None)
        return "" if meta is None else str(meta.get("type", ""))

    @classmethod
    def get_num_classes(cls, name: str) -> int:
        meta = cls._avl_io_items.get(str(name), None)
        if meta is None:
            return 0
        return int(meta.get("num_classes", 0) or 0)

    @classmethod
    def has_io_item(cls, name: str) -> bool:
        return str(name) in cls._avl_io_items


