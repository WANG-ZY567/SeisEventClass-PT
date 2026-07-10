<div align="center">

# SeisEventClass-PT

### Seismic Event Classification with Pretrained Transformers

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![GitHub Repo](https://img.shields.io/badge/GitHub-SeisEventClass--PT-black.svg)](https://github.com/sky6254/SeisEventClass-PT)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.4%2B-ee4c2c.svg)](https://pytorch.org/)

</div>

---

## Overview

**SeisEventClass-PT** is a research codebase for seismic event classification based on waveform tokenization and pretrained Transformer modeling. The repository focuses on the **DiTing 2.0 EVT6** task, where three-component seismic waveforms are classified into six event types.

The core idea is to convert continuous seismic waveforms into learnable token-like representations using convolutional frontends, and then adapt a pretrained Transformer / GPT-2 backbone for event-level classification. This design allows the model to capture discriminative waveform patterns such as phase arrivals, energy release duration, component-wise differences, and coda characteristics.

This repository is organized for reproducible experiments, including EVT6 dataset preparation, supervised training, evaluation, event-level aggregation, transfer protocols, and baseline comparisons.

## Key Features

- **EVT6 seismic event classification** on DiTing 2.0 style data.
- **Pretrained Transformer backbone** adapted from GPT-2 for seismic waveform representation learning.
- **Waveform tokenization frontend** based on multi-scale 1D convolutional embedding.
- **Multiple model variants**, including flat classification, hierarchical heads, multi-auxiliary heads, and CNN / spectrogram baselines.
- **Reproducible data protocol tools** for dataset construction, holdout splits, station-transfer evaluation, and event-level aggregation.
- **Open-source friendly layout**, excluding raw data, checkpoints, model caches, and historical experiment logs.

## EVT6 Label Definition

The EVT6 task follows the `evtype` labels used in the prepared DiTing 2.0 metadata:

| Class ID | Event Code | Description |
|---:|---|---|
| 0 | `eq` | Natural earthquake |
| 1 | `ep` | Explosion / blasting event |
| 2 | `co` / `ss` | Collapse event; `ss` is treated as the DiTing 2.0 alias of collapse |
| 3 | `sp` | DiTing 2.0 event type `sp` |
| 4 | `se` | DiTing 2.0 event type `se` |
| 5 | `ot` | Other event type |

The dataset adapter is implemented in `datasets/diting2_evt6.py`.

## Repository Structure

```text
SeisEventClass-PT/
|-- main.py                         # Main training / evaluation entry
|-- config.py                       # Model, loss, label, and metric registry
|-- requirements.txt                # Python dependencies
|-- datasets/                       # Dataset adapters
|-- models/                         # SeisMoLLM, Transformer, CNN, and baseline models
|-- training/                       # Training, validation, testing, and preprocessing loops
|-- utils/                          # Metrics, logging, visualization, and helper utilities
|-- tools/                          # Data preparation and analysis scripts
|-- scripts/                        # Recommended lightweight train / eval scripts
|-- run_scripts/                    # Original training / testing examples
|-- docs/                           # Dataset, method, leakage, and reproducibility notes
|-- checkpoints/                    # Local checkpoints, ignored by Git
`-- outputs/                        # Local outputs, ignored by Git
```

## Installation

Clone this repository and install the required packages in a clean Python environment:

```bash
git clone https://github.com/sky6254/SeisEventClass-PT.git
cd SeisEventClass-PT
```

```bash
pip install -r requirements.txt
```

The code has been organized around PyTorch-based training. Please install a CUDA-enabled PyTorch build that matches your GPU driver and cluster environment.

## Data Preparation

Raw DiTing 2.0 data and generated waveform arrays are not included in this repository.

Prepare the EVT6 dataset with:

```bash
python tools/prepare_diting2_evt6.py \
  --out_dir /path/to/diting2_evt6 \
  --natural_full5_dir /path/to/diting2_preprocessed \
  --natural_json /path/to/CENC_DiTingv2_natural_earthquake.json \
  --non_h5 /path/to/CENC_DiTingv2_non_natural_earthquake.hdf5 \
  --non_json /path/to/CENC_DiTingv2_non_natural_earthquake.json \
  --val_per_class 300 \
  --test_per_class 300
```

The generated directory is expected to contain metadata files such as:

```text
meta_evt6.csv
meta_evt6_train.csv
meta_evt6_val.csv
meta_evt6_test.csv
selection_record_evt6.json
waves/
waves_non/
```

See `DiTing2.0使用指南.md`, `docs/EVT6_DATASET_REPORT_PAPERALIGN.md`, and `docs/REPRODUCIBILITY.md` for more details.

## GPT-2 / Pretrained Backbone

SeisEventClass-PT adapts GPT-2 as the pretrained Transformer backbone. You can download GPT-2 manually from HuggingFace or use the helper scripts:

```bash
python download_gpt2_simple.py
```

or:

```bash
python download_gpt2_modelscope.py
```

After downloading, make sure the GPT-2 path used by `models/SeisMoLLM.py` points to your local model directory.

## Training

The recommended EVT6 training entry is:

```bash
DATA_DIR=/path/to/diting2_evt6 \
MODEL_NAME=SeisMoLLM_evt6_multihead_w01 \
BATCH_SIZE=64 \
EPOCHS=100 \
bash scripts/train_evt6.sh
```

Equivalent direct command:

```bash
python main.py \
  --mode train_test \
  --model-name SeisMoLLM_evt6_multihead_w01 \
  --dataset-name diting2_evt6 \
  --data /path/to/diting2_evt6 \
  --batch-size 64 \
  --epochs 100 \
  --log-base ./logs
```

For distributed training:

```bash
DATA_DIR=/path/to/diting2_evt6 \
GPUS=4 \
bash scripts/train_evt6.sh
```

## Evaluation

Evaluate a trained checkpoint with:

```bash
DATA_DIR=/path/to/diting2_evt6 \
CHECKPOINT=/path/to/model.pth \
MODEL_NAME=SeisMoLLM_evt6_multihead_w01 \
bash scripts/eval_evt6.sh
```

Or directly:

```bash
python main.py \
  --mode test \
  --model-name SeisMoLLM_evt6_multihead_w01 \
  --dataset-name diting2_evt6 \
  --data /path/to/diting2_evt6 \
  --checkpoint /path/to/model.pth
```

After testing, EVT6 reports and confusion matrices can be generated with:

```bash
python tools/report_evt6_results.py --run_dir /path/to/run_dir
```

## Additional Tools

The `tools/` directory includes scripts for:

- EVT6 dataset construction and protocol variants.
- Station-transfer split generation.
- Event-level prediction aggregation.
- Stratified analysis by event metadata.
- Class sensitivity analysis.
- CNN and spectrogram baseline training.
- PNW / ComCat external evaluation utilities.

See `docs/EVT6_XAPP_PROTOCOLS_AND_METRICS.md` for the extended evaluation protocols.

## Notes for Open-Source Use

This repository intentionally does not include:

- raw DiTing 2.0 data;
- generated waveform arrays;
- pretrained GPT-2 cache files;
- trained checkpoints;
- TensorBoard / W&B logs;
- historical experiment folders.

Please keep datasets under a local data directory and pass the path through `--data`.

## Acknowledgement

This project is developed based on the SeisMoLLM / SeisT-style seismic modeling codebase, with extensions for DiTing 2.0 EVT6 event classification, dataset construction, and event-level evaluation.

We thank the authors of SeisMoLLM and SeisT for their valuable open-source contributions.

## Citation

If this repository is useful for your research, please cite the related SeisMoLLM work and this EVT6 event-classification project when appropriate.

```bibtex
@misc{wang2025seismollmadvancingseismicmonitoring,
      title={SeisMoLLM: Advancing Seismic Monitoring via Cross-modal Transfer with Pre-trained Large Language Model},
      author={Xinghao Wang and Feng Liu and Rui Su and Zhihui Wang and Lihua Fang and Lianqing Zhou and Lei Bai and Wanli Ouyang},
      year={2025},
      eprint={2502.19960},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2502.19960}
}
```

## License

This project is released under the MIT License. See `LICENSE` for details.
