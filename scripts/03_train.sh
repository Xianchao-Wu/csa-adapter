#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_repo_env.sh"
CACHE_DIR="${CACHE_DIR:-cache/earnings22_large_v3}"
python -m csa_adapter.longform.train \
  --train-cache "$CACHE_DIR/train" --valid-cache "$CACHE_DIR/validation" \
  --output "${RUN_DIR:-runs/earnings22_csa_v021}" \
  --model "${MODEL:-openai/whisper-large-v3}" --revision "${REVISION:-main}" \
  --epochs "${EPOCHS:-99}" --max-steps "${MAX_STEPS:-1500}" \
  --grad-accum "${GRAD_ACCUM:-8}" --lr "${LR:-1e-4}" --seed "${SEED:-42}" \
  --history-segments "${HISTORY_SEGMENTS:-64}" \
  --compression-rate "${COMPRESSION_RATE:-8}" \
  --max-memory "${MAX_MEMORY:-4096}" --top-k "${TOP_K:-16}" \
  --warmup-steps "${WARMUP_STEPS:-0}" \
  --alpha-init "${ALPHA_INIT:-0.01}" --alpha-max "${ALPHA_MAX:-0.10}" \
  --gate-bias-init "${GATE_BIAS_INIT:--2.0}" \
  --residual-ratio-cap "${RESIDUAL_RATIO_CAP:-0.25}" \
  --valid-examples "${VALID_EXAMPLES:-256}" \
  --validate-every "${VALIDATE_EVERY:-250}" \
  --diagnostics-every "${DIAGNOSTICS_EVERY:-10}" \
  --feature-lru "${FEATURE_LRU:-256}" \
  --save-validation-checkpoints \
  "$@"
