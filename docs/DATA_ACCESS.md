# Data Access and Reproducibility Notes

This repository does not redistribute raw seismic waveform datasets, generated waveform arrays, pretrained model caches, or trained checkpoints.

## DiTing 2.0

DiTing 2.0 data should be obtained through the official data-provider procedures. After obtaining the required natural and non-natural event files, use `tools/prepare_diting2_evt6.py` to construct the EVT6 metadata files and local waveform arrays.

Expected local inputs include:

- `CENC_DiTingv2_natural_earthquake.json`
- `CENC_DiTingv2_non_natural_earthquake.hdf5`
- `CENC_DiTingv2_non_natural_earthquake.json`
- preprocessed natural-event waveform arrays if using an existing local DiTing preprocessing output

Expected prepared outputs include:

- `meta_evt6.csv`
- `meta_evt6_train.csv`
- `meta_evt6_val.csv`
- `meta_evt6_test.csv`
- `selection_record_evt6.json`
- `waves/`
- `waves_non/`

## PNW / ComCat External Evaluation

The Curated Pacific Northwest AI-ready Seismic Dataset is publicly available from the original project repository:

https://github.com/niyiyu/PNW-ML

The formal data descriptor is:

Ni, Y., Hutko, A., Skene, F., Denolle, M., Malone, S., Bodin, P., Hartog, R., and Wright, A. (2023). Curated Pacific Northwest AI-ready seismic dataset. Seismica, 2(1). https://doi.org/10.26443/seismica.v2i1.368

## Reproducibility Boundary

The repository provides source code, dataset-preparation scripts, training/evaluation entry points, and reporting tools. Exact reproduction requires users to obtain the original datasets from their providers and place local paths into the command-line arguments shown in `README.md` and `docs/REPRODUCIBILITY.md`.
