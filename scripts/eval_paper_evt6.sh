#!/usr/bin/env bash
# Evaluate the paper model on the fixed EVT6 test split.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

: "${DATA_DIR:?Set DATA_DIR to the prepared EVT6 dataset directory}"
: "${CHECKPOINT:?Set CHECKPOINT to a SeisEventClass-PT checkpoint}"
PYTHON_BIN="${PYTHON_BIN:-python}"
LOG_DIR="${LOG_DIR:-./logs/paper_evt6_eval}"
DEVICE="${DEVICE:-cuda:0}"
export SEISMOLLM_GPT2_PATH="${SEISMOLLM_GPT2_PATH:-}"

"$PYTHON_BIN" main.py \
  --mode test \
  --model-name SeisMoLLM_evt6_multihead_w005 \
  --dataset-name diting2_evt6 \
  --data "$DATA_DIR" \
  --checkpoint "$CHECKPOINT" \
  --checkpoint-strict true \
  --log-base "$LOG_DIR" \
  --force-logdir true \
  --device "$DEVICE" \
  --llm GPT2 \
  --pretrain true \
  --freeze true \
  --LLM_start_layer 0 \
  --LLM_layer_num 3 \
  --d_model 768 \
  --seed 20261107 \
  --in-samples 8192 \
  --batch-size 16 \
  --workers 8 \
  --norm-mode std \
  --augmentation false \
  --shuffle false \
  --save-test-results true \
  --save-test-probs true \
  --tta-times 1 \
  --use-tensorboard false
