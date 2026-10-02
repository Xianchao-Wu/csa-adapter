#!/usr/bin/env bash
# Fast stability screen: default hard-sparse vs old warm100, four seeds each.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_repo_env.sh"
python scripts/preflight_v021.py
IFS=',' read -ra GPU_IDS <<< "${GPUS:-0,1,2,3,4,5,6,7}"
[[ "${#GPU_IDS[@]}" -eq 8 ]] || { echo "Need 8 GPUs" >&2; exit 2; }
DATA_DIR="${DATA_DIR:-data/earnings22_622}"
CACHE_DIR="${CACHE_DIR:-cache/earnings22_large_v3}"
RUN_ROOT="${RUN_ROOT:-runs/h100_v021_stability}"
MODEL="${MODEL:-openai/whisper-large-v3}"
REVISION="${REVISION:-main}"
MAX_STEPS="${MAX_STEPS:-500}"
mkdir -p "$RUN_ROOT/logs"

names=(default_s42 default_s43 default_s44 default_s45 warm100_s42 warm100_s43 warm100_s44 warm100_s45)
seeds=(42 43 44 45 42 43 44 45)
pids=()
for i in "${!GPU_IDS[@]}"; do
  gpu="${GPU_IDS[$i]}"; run="${names[$i]}"; seed="${seeds[$i]}"
  extra=(--warmup-steps 0)
  [[ "$run" == warm100_* ]] && extra=(--warmup-steps 100)
  CUDA_VISIBLE_DEVICES="$gpu" python -m csa_adapter.longform.train \
    --train-cache "$CACHE_DIR/train" --valid-cache "$CACHE_DIR/validation" \
    --output "$RUN_ROOT/$run" --model "$MODEL" --revision "$REVISION" \
    --epochs 99 --max-steps "$MAX_STEPS" --seed "$seed" --grad-accum 8 --lr 1e-4 \
    --history-segments 64 --compression-rate 8 --max-memory 4096 --top-k 16 \
    --alpha-init 0.01 --alpha-max 0.10 --gate-bias-init -2.0 --residual-ratio-cap 0.25 \
    --valid-examples 128 --validate-every 250 --diagnostics-every 10 --feature-lru 256 \
    "${extra[@]}" >"$RUN_ROOT/logs/$run.train.log" 2>&1 &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
[[ "$status" == 0 ]] || exit 3

pids=()
for i in "${!GPU_IDS[@]}"; do
  gpu="${GPU_IDS[$i]}"; run="${names[$i]}"
  CUDA_VISIBLE_DEVICES="$gpu" python -m csa_adapter.longform.evaluate \
    --manifest "$DATA_DIR/validation.full.jsonl" --max-calls "${MAX_CALLS:-5}" \
    --output "$RUN_ROOT/$run/eval_stability" --model "$MODEL" --revision "$REVISION" \
    --adapter "$RUN_ROOT/$run/best_adapter" --memory-mode history --diagnostics \
    >"$RUN_ROOT/logs/$run.eval.log" 2>&1 &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
[[ "$status" == 0 ]] || exit 4
python scripts/summarize_v021.py "$RUN_ROOT" | tee "$RUN_ROOT/stability_summary.tsv"
