#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_repo_env.sh"
BASE_RUN="${RUN_DIR:-runs/routing}"
for mode in direct warm dense; do
  extra=()
  case "$mode" in
    direct) extra+=(--warmup-steps 0);;
    warm) extra+=(--warmup-steps 100);;
    dense) extra+=(--dense-always --warmup-steps 0);;
  esac
  RUN_DIR="${BASE_RUN}_$mode" bash scripts/03_train.sh "${extra[@]}" "$@"
done
