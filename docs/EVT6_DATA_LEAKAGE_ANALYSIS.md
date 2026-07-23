# EVT6 Data Leakage Analysis

This document summarizes checks used to reduce leakage risk in EVT6 experiments.

## Leakage Risks

Potential leakage may occur when records from the same event, highly similar waveform windows, or duplicated metadata entries are split across training and evaluation subsets.

## Recommended Checks

- Verify that event identifiers do not overlap across splits.
- Check duplicated station-event records.
- Inspect near-duplicate waveform windows when preprocessing produces multiple segments.
- Report split construction rules in the manuscript and README.

## Interpretation

Event-level separation is preferred for evaluating generalization. When a station-level or window-level split is used for a specific experiment, it should be stated clearly and interpreted with caution.
