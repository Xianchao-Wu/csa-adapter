#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-60}"
# Alternative: E22 custom 100 train /25 validation calls; E21 all 44 calls external test.
export DATA_DIR="${DATA_DIR:-data/earnings22_8020}"
export CACHE_DIR="${CACHE_DIR:-cache/earnings22_8020_large_v3}"
export RUN_DIR="${RUN_DIR:-runs/earnings22_8020_csa}"
python -m csa_adapter.longform.data --output "$DATA_DIR" \
  --ratios 0.8,0.2,0 --seed "${SEED:-42}" --oversize skip --external-earnings21
bash scripts/02_cache_features.sh
bash scripts/03_train.sh "$@"
bash scripts/05_eval_earnings21.sh
