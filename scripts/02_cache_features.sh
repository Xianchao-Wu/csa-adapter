#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_repo_env.sh"
DATA_DIR="${DATA_DIR:-data/earnings22_622}"
CACHE_DIR="${CACHE_DIR:-cache/earnings22_large_v3}"
MODEL="${MODEL:-openai/whisper-large-v3}"
# One GPU by default. For 8 GPUs: GPUS=0,1,2,3,4,5,6,7 bash scripts/02_cache_features.sh
IFS=',' read -ra gpu_ids <<< "${GPUS:-0}"
for split in train validation; do
  pids=()
  for i in "${!gpu_ids[@]}"; do
    CUDA_VISIBLE_DEVICES="${gpu_ids[$i]}" python -m csa_adapter.longform.cache \
      --manifest "$DATA_DIR/$split.jsonl" --output "$CACHE_DIR/$split" \
      --model "$MODEL" --revision "${REVISION:-main}" \
      --shards "${#gpu_ids[@]}" --shard-index "$i" "$@" &
    pids+=("$!")
  done
  status=0
  for pid in "${pids[@]}"; do
    wait "$pid" || status=1
  done
  if [[ "$status" != 0 ]]; then exit "$status"; fi
done
