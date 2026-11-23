<div align="center">

# SeisEventClass-PT

### Six-class seismic event classification with waveform tokenization and pretrained Transformers

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.7.1-ee4c2c.svg)](https://pytorch.org/)

</div>

## Overview

SeisEventClass-PT classifies three-component seismic waveforms into the six
DiTing 2.0 EVT6 catalogue labels: `eq`, `ep`, `ss`, `sp`, `se`, and `ot`.
The paper model combines a multi-scale 1-D convolutional tokenizer, the first
three layers of pretrained GPT-2, LoRA adaptation, and a six-class prediction
head with auxiliary objectives.

The repository contains the model, event-disjoint data preparation tools,
training and evaluation entry points, component ablations, event-level
aggregation, and ComCat zero-shot evaluation utilities. Raw datasets,
generated arrays, pretrained caches, and experiment checkpoints are excluded.

## Paper-to-code map

| Manuscript name | Code entry / variant | Description |
|---|---|---|
| SeisEventClass-PT | `SeisMoLLM_evt6_multihead_w005` | Full paper model |
| SeisEventClass-RandInit | `EVT6_ABLATION=random_lora` | GPT-2 initialized randomly; matched architecture and LoRA |
| SeisEventClass-SingleScale | `EVT6_ABLATION=single_scale` | Single-scale convolutional tokenizer |
| SeisEventClass-NoLoRA | `EVT6_ABLATION=pretrained_no_lora` | Pretrained GPT-2 without LoRA |
| SeisEventClass-FrozenGPT2 | `EVT6_ABLATION=pretrained_frozen` | Pretrained GPT-2 fully frozen |

All five variants use the `w005` multi-head objective. The suffix denotes
auxiliary loss weights of `0.05`, not a different input dataset.

## Repository layout

```text
SeisEventClass-PT/
├── main.py                       # Main train/test entry
├── config.py                     # Model, loss, label, and metric registry
├── datasets/                     # Dataset adapters
├── models/                       # Tokenizer, GPT-2/LoRA model, and baselines
├── training/                     # Train, validation, and test loops
├── tools/                        # Data, analysis, external evaluation, ablations
├── scripts/                      # Reproducible paper train/eval commands
├── docs/                         # Data and reproducibility notes
└── requirements.txt
```

## Installation

```bash
git clone https://github.com/WANG-ZY567/SeisEventClass-PT.git
cd SeisEventClass-PT
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows, activate with `.venv\Scripts\Activate.ps1`. If the pinned PyTorch
wheel does not match your CUDA installation, install the appropriate PyTorch
2.7.1 build first and then install the remaining requirements.

## GPT-2 checkpoint

Download GPT-2 with either helper:

```bash
python download_gpt2_simple.py
# or
python download_gpt2_modelscope.py
```

Point the code to the local checkpoint with:

```bash
export SEISMOLLM_GPT2_PATH=/path/to/gpt2
```

Alternatively, write the path into a local `gpt2_model_path.txt`. That file is
ignored by Git so machine-specific paths are not published.

## Data preparation

DiTing 2.0 is not redistributed here. Obtain the source data from its official
provider and construct the EVT6 benchmark locally. The preprocessing code is
split into the following auditable stages:

1. `tools/build_diting2_evt6_review_protocol.py` filters, windows, and quality
   controls records from the two DiTing 2.0 source files. For the revised
   benchmark, pass `--in_samples 8192 --pre_samples 500 --require_valid_sg`;
   do not rely on the script's historical `pre_samples` default.
2. `tools/create_evt6_bandpass_dataset.py` applies per-component linear
   detrending, a Tukey taper, and a fourth-order 1-20 Hz zero-phase Butterworth
   band-pass filter at 50 Hz.
3. `tools/augment_evt6_natural_classes.py` adds eligible non-`eq` records found
   in the natural-event source, resolves duplicated event identifiers, and
   applies the same signal processing to newly exported records.
4. `tools/resplit_evt6_event_8020.py` creates event-disjoint train, validation,
   and test manifests.
5. `training/preprocess.py` performs per-component demeaning and standard-
   deviation normalization at model input time (`--norm-mode std`).

The prepared directory must contain:

```text
meta_evt6_train.csv
meta_evt6_val.csv
meta_evt6_test.csv
```

and the waveform arrays referenced by `_npy_path`. See
[`docs/PREPROCESSING_ZH.md`](docs/PREPROCESSING_ZH.md),
[`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md), and
[`docs/DATA_ACCESS.md`](docs/DATA_ACCESS.md) before running the pipeline.

## Reproduce the paper model

Linux/macOS:

```bash
export DATA_DIR=/path/to/prepared_evt6
export SEISMOLLM_GPT2_PATH=/path/to/gpt2
bash scripts/train_paper_evt6.sh
```

Windows PowerShell:

```powershell
.\scripts\train_paper_evt6.ps1 `
  -DataDir "D:\data\diting2_evt6" `
  -Gpt2Path "D:\models\gpt2"
```

The scripts encode the paper configuration: 50 Hz, 8,192 samples, three GPT-2
layers, hidden size 768, batch size 16, Adam, cyclic learning rate, class-
balanced cross-entropy (`beta=0.995`), validation-accuracy checkpoint selection,
and random seed `20261107`.

## Evaluate a checkpoint

```bash
DATA_DIR=/path/to/prepared_evt6 \
CHECKPOINT=/path/to/model.pth \
bash scripts/eval_paper_evt6.sh
```

```powershell
.\scripts\eval_paper_evt6.ps1 `
  -DataDir "D:\data\diting2_evt6" `
  -Checkpoint "D:\runs\model.pth" `
  -Gpt2Path "D:\models\gpt2"
```

Generate detailed waveform- and event-level reports from a saved prediction
CSV with `tools/analyze_evt6_predictions.py`.

## Component ablations

Run all matched paper variants with:

```bash
DATA=/path/to/prepared_evt6 \
PY=python \
OUTPUT_ROOT=./outputs \
bash tools/run_evt6_paired_ablation.sh
```

The runner checks parameter compatibility before training and writes a joint
summary for the full model and the four ablations listed above.

## ComCat zero-shot evaluation

The ComCat utilities are under `tools/prepare_comcat_evt6_zeroshot.py` and
`tools/evaluate_comcat_shared_binary.py`. The manuscript reports the
`strict_six_class_output` result: the model retains all six EVT6 outputs, and
ComCat records labelled `eq` or `ep` that are predicted as any other EVT6 class
are counted as incorrect. It is not a six-class ComCat benchmark because
ComCat provides external reference labels only for `eq` and `ep` in this study.

## Public-release checks

Before committing or creating a release archive, run:

```bash
python tools/audit_public_release.py
```

The audit checks for private absolute paths, secrets, data/checkpoint files,
large files, transient environments, and Python syntax errors.

## Data and code availability

Raw DiTing 2.0 and ComCat waveforms, generated arrays, pretrained model caches,
and trained checkpoints are not redistributed because of provider policies and
file-size constraints. Users must obtain the datasets from their original
providers and generate local derivatives with the included tools.

## Acknowledgements

This project builds on the open-source SeisMoLLM and SeisT codebases and adds
the EVT6 model, unified data protocol, event-level evaluation, ablations, and
cross-catalogue analysis.

## Citation

Please cite the associated SeisEventClass-PT manuscript. Until its publication,
also cite the upstream SeisMoLLM work where appropriate:

```bibtex
@misc{wang2025seismollm,
  title={SeisMoLLM: Advancing Seismic Monitoring via Cross-modal Transfer with Pre-trained Large Language Model},
  author={Wang, Xinghao and Liu, Feng and Su, Rui and Wang, Zhihui and Fang, Lihua and Zhou, Lianqing and Bai, Lei and Ouyang, Wanli},
  year={2025},
  eprint={2502.19960},
  archivePrefix={arXiv},
  primaryClass={cs.LG},
  url={https://arxiv.org/abs/2502.19960}
}
```

## License

Released under the [MIT License](LICENSE).
