#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_repo_env.sh"
DATA_DIR="${DATA_DIR:-data/earnings22_622}"
RUN_DIR="${RUN_DIR:-runs/earnings22_csa_v021}"
MANIFEST="${MANIFEST:-$DATA_DIR/test.full.jsonl}"
EVAL_DIR="${EVAL_DIR:-$RUN_DIR/eval_test}"
common=(
  --manifest "$MANIFEST"
  --model "${MODEL:-openai/whisper-large-v3}"
  --revision "${REVISION:-main}"
  --diagnostics
)

python -m csa_adapter.longform.evaluate "${common[@]}" \
  --output "$EVAL_DIR/baseline_text_off" "$@"
python -m csa_adapter.longform.evaluate "${common[@]}" \
  --adapter "$RUN_DIR/best_adapter" --memory-mode history \
  --output "$EVAL_DIR/csa_text_off" "$@"
python -m csa_adapter.longform.evaluate "${common[@]}" \
  --adapter "$RUN_DIR/best_adapter" --memory-mode reset \
  --output "$EVAL_DIR/csa_reset_text_off" "$@"

# Text history is intentionally secondary: it is an inference ablation, not the
# model-selection protocol.  v0.2.1 uses only the immediately previous chunk by default.
python -m csa_adapter.longform.evaluate "${common[@]}" --text-history \
  --output "$EVAL_DIR/baseline_text_on" "$@"
python -m csa_adapter.longform.evaluate "${common[@]}" --text-history \
  --adapter "$RUN_DIR/best_adapter" --memory-mode history \
  --output "$EVAL_DIR/csa_text_on" "$@"
