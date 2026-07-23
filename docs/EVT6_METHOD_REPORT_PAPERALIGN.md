# EVT6 Method Report

This document summarizes the method components used in SeisEventClass-PT.

## Framework

SeisEventClass-PT uses a multi-scale convolutional tokenizer to convert three-component waveforms into token sequences. The token sequence is then processed by a pretrained GPT-2 backbone with parameter-efficient adaptation for downstream seismic event classification.

## Modeling Motivation

Different event types produce different waveform evidence, including phase-onset sharpness, inter-component motion patterns, S-wave response, coda development, and long-range temporal organization. The tokenizer captures local waveform morphology, while the Transformer backbone models longer temporal dependencies across the full window.

## Outputs

The classification head predicts EVT6 event-type probabilities. For experiments with multiple records per event, probabilities can be aggregated at event level.
