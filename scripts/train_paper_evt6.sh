#!/usr/bin/env bash
# Paper-aligned SeisEventClass-PT training on the prepared EVT6 split.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

: "${DATA_DIR:?Set DATA_DIR to the prepared EVT6 dataset directory}"
PYTHON_BIN="${PYTHON_BIN:-python}"
LOG_DIR="${LOG_DIR:-./logs/paper_evt6}"
DEVICE="${DEVICE:-cuda:0}"
SEISMOLLM_GPT2_PATH="${SEISMOLLM_GPT2_PATH:-}"
export SEISMOLLM_GPT2_PATH
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

"$PYTHON_BIN" main.py \
  --mode train_test \
  --model-name SeisMoLLM_evt6_multihead_w005 \
  --dataset-name diting2_evt6 \
  --data "$DATA_DIR" \
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
  --epochs 200 \
  --patience 30 \
  --optim Adam \
  --weight-decay 0 \
  --best-metric accuracy \
  --loss-name class_balanced_ce \
  --cb-beta 0.995 \
  --cb-samples-per-class 4629,4629,4287,1440,160,2357 \
  --cb-weight-schedule fixed \
  --use-lr-scheduler true \
  --lr-scheduler-name cyclic \
  --lr-scheduler-mode exp_range \
  --lr-scheduler-gamma 1.0 \
  --base-lr 8e-5 \
  --max-lr 1e-3 \
  --min-lr 1e-6 \
  --warmup-steps 2000 \
  --down-steps 3000 \
  --augmentation true \
  --shift-event-rate 0.2 \
  --add-noise-rate 0.4 \
  --add-gap-rate 0.4 \
  --drop-channel-rate 0.4 \
  --scale-amplitude-rate 0.4 \
  --pre-emphasis-rate 0.4 \
  --pre-emphasis-ratio 0.97 \
  --generate-noise-rate 0.05 \
  --norm-mode std \
  --save-test-results true \
  --save-test-probs true \
  --tta-times 1 \
  --use-tensorboard true
