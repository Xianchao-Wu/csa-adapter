#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
CACHE_DIR="${CACHE_DIR:-cache/earnings22_large_v3}"
python -m csa_adapter.longform.train \
  --train-cache "$CACHE_DIR/train" --valid-cache "$CACHE_DIR/validation" \
  --output "${RUN_DIR:-runs/earnings22_csa}" \
  --model "${MODEL:-openai/whisper-large-v3}" --revision "${REVISION:-main}" \
  --epochs 3 --grad-accum 8 --lr 1e-4 --seed "${SEED:-42}" \
  --history-segments 64 --compression-rate 8 --max-memory 4096 --top-k 16 \
  --warmup-steps 100 --valid-examples 256 "$@"
