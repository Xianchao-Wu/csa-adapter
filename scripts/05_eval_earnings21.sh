#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_repo_env.sh"
export MANIFEST="${DATA_DIR:-data/earnings22_622}/earnings21.full.jsonl"
export EVAL_DIR="${RUN_DIR:-runs/earnings22_csa}/eval_earnings21"
bash scripts/04_eval_matrix.sh "$@"
