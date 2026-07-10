#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="${DATA_DIR:-/path/to/diting2_evt6}"
CHECKPOINT="${CHECKPOINT:-/path/to/model.pth}"
LOG_DIR="${LOG_DIR:-./logs}"
MODEL_NAME="${MODEL_NAME:-SeisMoLLM_evt6_multihead_w01}"

python main.py \
  --mode test \
  --model-name "$MODEL_NAME" \
  --dataset-name diting2_evt6 \
  --data "$DATA_DIR" \
  --checkpoint "$CHECKPOINT" \
  --log-base "$LOG_DIR" \
  --batch-size "${BATCH_SIZE:-64}" \
  --workers "${WORKERS:-8}"
