#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-60}"
# Explicitly skip/recount overlong training segments; full-call test audio is intact.
python -m csa_adapter.longform.data \
  --output "${DATA_DIR:-data/earnings22_622}" \
  --ratios 0.6,0.2,0.2 --seed "${SEED:-42}" --oversize skip \
  --external-earnings21 "$@"
