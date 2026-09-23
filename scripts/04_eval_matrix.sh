#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
DATA_DIR="${DATA_DIR:-data/earnings22_622}"
RUN_DIR="${RUN_DIR:-runs/earnings22_csa}"
MANIFEST="${MANIFEST:-$DATA_DIR/test.full.jsonl}"
EVAL_DIR="${EVAL_DIR:-$RUN_DIR/eval_test}"
common=(--manifest "$MANIFEST" --model "${MODEL:-openai/whisper-large-v3}" --revision "${REVISION:-main}")
for prompt in off on; do
  extra=()
  if [[ "$prompt" == on ]]; then extra+=(--text-history); fi
  python -m csa_adapter.longform.evaluate "${common[@]}" "${extra[@]}" \
    --output "$EVAL_DIR/baseline_text_$prompt" "$@"
  python -m csa_adapter.longform.evaluate "${common[@]}" "${extra[@]}" \
    --adapter "$RUN_DIR/best_adapter" --memory-mode history \
    --output "$EVAL_DIR/csa_text_$prompt" "$@"
done
# Reset ablation: this historical-only adapter becomes exact backbone identity.
python -m csa_adapter.longform.evaluate "${common[@]}" --text-history \
  --adapter "$RUN_DIR/best_adapter" --memory-mode reset \
  --output "$EVAL_DIR/csa_reset_text_on" "$@"
