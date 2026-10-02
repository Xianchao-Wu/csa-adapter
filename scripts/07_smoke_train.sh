#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_repo_env.sh"
# Requires prepared encoder caches; verifies real data + decoder gradient pipeline.
export RUN_DIR="${RUN_DIR:-runs/smoke_csa}"
bash scripts/03_train.sh --max-steps 3 --grad-accum 1 --history-segments 2 \
  --valid-examples 2 --warmup-steps 1 "$@"
