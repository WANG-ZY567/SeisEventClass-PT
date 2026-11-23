# Reproducibility

This document records the configuration used for the paper model. The
authoritative executable commands are `scripts/train_paper_evt6.sh` and
`scripts/train_paper_evt6.ps1`.

## Environment

- Python 3.10
- PyTorch 2.7.1
- three-component waveform input (`Z`, `N`, `E`)
- 50 Hz sampling rate
- 8,192 samples per component (163.84 s)

Install dependencies with `pip install -r requirements.txt`. Select the
PyTorch wheel appropriate for the CUDA driver on the target machine.

## Dataset contract

The prepared EVT6 directory contains `meta_evt6_train.csv`,
`meta_evt6_val.csv`, and `meta_evt6_test.csv`. Splitting is performed by event:
all retained station records of the same event belong to exactly one split.
Each manifest must provide `_evt6`, `event_uid`, and `_npy_path`.

The paper training manifest contains the following waveform counts in class
order `eq, ep, ss, sp, se, ot`:

```text
4629, 4629, 4287, 1440, 160, 2357
```

Do not substitute counts from an older dataset build when computing the
class-balanced loss.

## Model configuration

| Item | Value |
|---|---|
| Paper model | `SeisMoLLM_evt6_multihead_w005` |
| GPT-2 checkpoint | `gpt2` (base configuration) |
| Active Transformer layers | first 3 layers (indices 0-2) |
| Hidden dimension | 768 |
| Attention heads | 12 (GPT-2 base) |
| Input samples | 8,192 |
| Main task | six-class EVT6 classification |
| Auxiliary loss weights | `0.05, 0.05, 0.0` |
| Random seed | 20261107 |

Set `SEISMOLLM_GPT2_PATH` to the local GPT-2 directory. The code uses
`inputs_embeds`; waveform tokens are continuous embeddings and are not mapped
through a text vocabulary.

## Optimization

| Item | Value |
|---|---|
| Optimizer | Adam |
| Weight decay | 0 |
| Batch size | 16 |
| Data workers | 8 |
| Maximum epochs | 200 |
| Early-stopping patience | 30 |
| Checkpoint criterion | validation accuracy |
| Scheduler | cyclic, `exp_range` |
| Base / maximum / minimum LR | `8e-5` / `1e-3` / `1e-6` |
| Up / down steps | 2,000 / 3,000 |
| Scheduler gamma | 1.0 |
| Primary loss | class-balanced cross-entropy |
| Class-balance beta | 0.995 |
| Weight schedule | fixed |

## Training augmentation

Augmentation is used only for training. The paper settings are:

| Augmentation | Probability |
|---|---:|
| event shift | 0.20 |
| additive noise | 0.40 |
| gap insertion | 0.40 |
| channel dropout | 0.40 |
| amplitude scaling | 0.40 |
| pre-emphasis | 0.40 |
| generated noise | 0.05 |

Pre-emphasis uses a ratio of 0.97. Validation and test evaluation disable
augmentation and test-time augmentation (`tta_times=1`).

## Commands

Linux/macOS:

```bash
DATA_DIR=/path/to/evt6 \
SEISMOLLM_GPT2_PATH=/path/to/gpt2 \
bash scripts/train_paper_evt6.sh
```

Windows PowerShell:

```powershell
.\scripts\train_paper_evt6.ps1 `
  -DataDir "D:\data\evt6" `
  -Gpt2Path "D:\models\gpt2"
```

Evaluation uses the matching `eval_paper_evt6` script. Model selection must be
based on the validation split; the test split is evaluated only after selecting
the checkpoint.

## Matched ablations

`tools/run_evt6_paired_ablation.sh` runs the full model and four matched
variants. In the manuscript they are named:

- `random_lora` -> SeisEventClass-RandInit
- `single_scale` -> SeisEventClass-SingleScale
- `pretrained_no_lora` -> SeisEventClass-NoLoRA
- `pretrained_frozen` -> SeisEventClass-FrozenGPT2

The runner audits parameter counts and trainable GPT-2 tensors before training.

## Release audit

Run `python tools/audit_public_release.py` before publishing. This verifies that
private paths, environments, raw data, model weights, and oversized files have
not entered the repository and compiles all Python sources.
