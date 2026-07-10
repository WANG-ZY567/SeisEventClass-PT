#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="${DATA_DIR:-/path/to/diting2_evt6}"
LOG_DIR="${LOG_DIR:-./logs}"
MODEL_NAME="${MODEL_NAME:-SeisMoLLM_evt6_multihead_w01}"
GPUS="${GPUS:-1}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-6}"

if [[ "$GPUS" -gt 1 ]]; then
  torchrun --nnodes 1 --nproc_per_node "$GPUS" --master_port "${MASTER_PORT:-10000}" main.py \
    --mode train_test \
    --model-name "$MODEL_NAME" \
    --dataset-name diting2_evt6 \
    --data "$DATA_DIR" \
    --log-base "$LOG_DIR" \
    --batch-size "${BATCH_SIZE:-64}" \
    --epochs "${EPOCHS:-100}" \
    --workers "${WORKERS:-8}"
else
  python main.py \
    --mode train_test \
    --model-name "$MODEL_NAME" \
    --dataset-name diting2_evt6 \
    --data "$DATA_DIR" \
    --log-base "$LOG_DIR" \
    --batch-size "${BATCH_SIZE:-64}" \
    --epochs "${EPOCHS:-100}" \
    --workers "${WORKERS:-8}"
fi
