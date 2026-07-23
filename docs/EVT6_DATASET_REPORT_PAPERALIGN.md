# EVT6 Dataset Report

This document summarizes the dataset organization used by the EVT6 experiments.

## Task Definition

EVT6 is formulated as a six-class seismic event classification task over fixed-length three-component waveforms. The label space includes natural earthquakes and multiple non-natural seismic event types.

## Data Construction

Natural and non-natural event branches may come from different sources and preprocessing procedures. To reduce non-physical differences, all records are standardized into a common input format before model training.

Key preprocessing principles include:

- using fixed-length three-component waveform windows;
- preserving event-level identifiers for aggregation and leakage checks;
- applying consistent label mapping;
- separating training, validation, and test subsets by event identity whenever possible.

## Reporting

Dataset reports should include class counts, split counts, station/event summaries, and any filtering criteria used to construct the final benchmark.
