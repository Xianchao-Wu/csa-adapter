#!/usr/bin/env bash
# v0.2.1 H100 priority experiment: 4 families x 2 seeds on 8 independent GPUs.
# Each run is single-GPU; all eight runs execute concurrently.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_repo_env.sh"
python scripts/preflight_v021.py

IFS=',' read -ra GPU_IDS <<< "${GPUS:-0,1,2,3,4,5,6,7}"
if [[ "${#GPU_IDS[@]}" -ne 8 ]]; then
  echo "Expected exactly 8 GPU ids in GPUS, got ${#GPU_IDS[@]}: ${GPU_IDS[*]}" >&2
  exit 2
fi

DATA_DIR="${DATA_DIR:-data/earnings22_622}"
CACHE_DIR="${CACHE_DIR:-cache/earnings22_large_v3}"
RUN_ROOT="${RUN_ROOT:-runs/h100_v021_priority}"
MODEL="${MODEL:-openai/whisper-large-v3}"
REVISION="${REVISION:-main}"
MAX_STEPS="${MAX_STEPS:-1500}"
CHECKPOINT_SELECT_MAX_CALLS="${CHECKPOINT_SELECT_MAX_CALLS:-0}"
mkdir -p "$RUN_ROOT/logs"

# Main v0.2.1 default is hard sparse routing from step 1.
# warm100 preserves the old dense->sparse strategy as an explicit ablation.
RUN_NAMES=(
  default_s42 default_s43
  warm100_s42 warm100_s43
  dense_s42 dense_s43
  mean_s42 mean_s43
)
SEEDS=(42 43 42 43 42 43 42 43)
EXTRAS=(
  "--warmup-steps 0"
  "--warmup-steps 0"
  "--warmup-steps 100"
  "--warmup-steps 100"
  "--dense-always --warmup-steps 0"
  "--dense-always --warmup-steps 0"
  "--compressor mean --warmup-steps 0"
  "--compressor mean --warmup-steps 0"
)

# -----------------------------------------------------------------------------
# Stage 1: train eight independent runs in parallel.
# -----------------------------------------------------------------------------
pids=()
for i in "${!GPU_IDS[@]}"; do
  gpu="${GPU_IDS[$i]}"
  run="${RUN_NAMES[$i]}"
  seed="${SEEDS[$i]}"
  if [[ -e "$RUN_ROOT/$run/run.json" ]]; then
    echo "Refusing to overwrite existing run: $RUN_ROOT/$run" >&2
    exit 3
  fi
  echo "[train] GPU=$gpu run=$run seed=$seed extras=${EXTRAS[$i]}"
  # EXTRAS intentionally word-splits a small trusted flag string defined above.
  # shellcheck disable=SC2086
  CUDA_VISIBLE_DEVICES="$gpu" python -m csa_adapter.longform.train \
    --train-cache "$CACHE_DIR/train" --valid-cache "$CACHE_DIR/validation" \
    --output "$RUN_ROOT/$run" --model "$MODEL" --revision "$REVISION" \
    --epochs 99 --max-steps "$MAX_STEPS" --grad-accum 8 --lr 1e-4 --seed "$seed" \
    --history-segments 64 --compression-rate 8 --max-memory 4096 --top-k 16 \
    --alpha-init 0.01 --alpha-max 0.10 --gate-bias-init -2.0 \
    --residual-ratio-cap 0.25 --valid-examples 256 --validate-every 250 \
    --diagnostics-every 10 --feature-lru 256 --save-validation-checkpoints \
    ${EXTRAS[$i]} \
    >"$RUN_ROOT/logs/${run}.train.log" 2>&1 &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
[[ "$status" == 0 ]] || {
  echo "At least one training job failed; inspect $RUN_ROOT/logs" >&2
  exit 4
}

# -----------------------------------------------------------------------------
# Stage 2: select a checkpoint within each run by long-form validation WER-N.
# This is deliberate: teacher-forced NLL alone did not predict v0.2 free-running
# stability.  Each GPU evaluates the checkpoints belonging to its own run.
# -----------------------------------------------------------------------------
pids=()
for i in "${!GPU_IDS[@]}"; do
  gpu="${GPU_IDS[$i]}"
  run="${RUN_NAMES[$i]}"
  (
    extra=()
    [[ "$run" == dense_* ]] && extra+=(--dense-reading)
    shopt -s nullglob
    ckpts=("$RUN_ROOT/$run"/checkpoints/step-*)
    [[ "${#ckpts[@]}" -gt 0 ]] || {
      echo "No validation checkpoints found for $run" >&2
      exit 20
    }
    for ckpt in "${ckpts[@]}"; do
      tag="$(basename "$ckpt")"
      max_args=()
      [[ "$CHECKPOINT_SELECT_MAX_CALLS" != 0 ]] && \
        max_args+=(--max-calls "$CHECKPOINT_SELECT_MAX_CALLS")
      echo "[ckpt-valid] GPU=$gpu run=$run checkpoint=$tag"
      CUDA_VISIBLE_DEVICES="$gpu" python -m csa_adapter.longform.evaluate \
        --manifest "$DATA_DIR/validation.full.jsonl" \
        --output "$RUN_ROOT/$run/checkpoint_eval/$tag" \
        --model "$MODEL" --revision "$REVISION" \
        --adapter "$ckpt" --memory-mode history --diagnostics \
        "${max_args[@]}" "${extra[@]}" \
        >"$RUN_ROOT/logs/${run}.${tag}.valid.log" 2>&1
    done
    python - "$RUN_ROOT/$run" <<'PY'
import json
import sys
from pathlib import Path

run = Path(sys.argv[1])
values = []
for p in sorted((run / "checkpoint_eval").glob("step-*/metrics.json")):
    d = json.loads(p.read_text())
    values.append((d["normalized_micro_wer"], p.parent.name))
if not values:
    raise SystemExit("no checkpoint metrics")
score, tag = min(values)
(run / "selected_checkpoint.txt").write_text(tag + "\n")
(run / "selected_checkpoint.json").write_text(
    json.dumps({"checkpoint": tag, "validation_micro_wer": score}, indent=2)
)
print(run.name, tag, score)
PY
  ) &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
[[ "$status" == 0 ]] || {
  echo "At least one checkpoint-selection job failed" >&2
  exit 5
}

# -----------------------------------------------------------------------------
# Stage 3: select one seed per family, again on long-form validation WER-N.
# -----------------------------------------------------------------------------
python - "$RUN_ROOT" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
families = {"default": [], "warm100": [], "dense": [], "mean": []}
for family in families:
    for seed in (42, 43):
        run = f"{family}_s{seed}"
        d = json.loads((root / run / "selected_checkpoint.json").read_text())
        families[family].append((d["validation_micro_wer"], run, d["checkpoint"]))
lines = ["family\tselected_run\tselected_checkpoint\tvalidation_micro_wer"]
for family, values in families.items():
    score, run, ckpt = min(values)
    lines.append(f"{family}\t{run}\t{ckpt}\t{score:.10f}")
(root / "selected_runs.tsv").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
PY

selected_run() {
  awk -F '\t' -v f="$1" '$1==f {print $2}' "$RUN_ROOT/selected_runs.tsv"
}
selected_ckpt() {
  awk -F '\t' -v f="$1" '$1==f {print $3}' "$RUN_ROOT/selected_runs.tsv"
}
default_run="$(selected_run default)"; default_ckpt="$(selected_ckpt default)"
warm_run="$(selected_run warm100)"; warm_ckpt="$(selected_ckpt warm100)"
dense_run="$(selected_run dense)"; dense_ckpt="$(selected_ckpt dense)"
mean_run="$(selected_run mean)"; mean_ckpt="$(selected_ckpt mean)"
default_adapter="$RUN_ROOT/$default_run/checkpoints/$default_ckpt"
warm_adapter="$RUN_ROOT/$warm_run/checkpoints/$warm_ckpt"
dense_adapter="$RUN_ROOT/$dense_run/checkpoints/$dense_ckpt"
mean_adapter="$RUN_ROOT/$mean_run/checkpoints/$mean_ckpt"

# -----------------------------------------------------------------------------
# Stage 4: E22 test matrix.  Primary comparisons keep text history OFF.
# -----------------------------------------------------------------------------
TEST_NAMES=(
  baseline_text_off baseline_text_on
  default_text_off default_reset_text_off default_text_on
  warm100_text_off dense_text_off mean_text_off
)
pids=()
for i in "${!TEST_NAMES[@]}"; do
  gpu="${GPU_IDS[$i]}"
  name="${TEST_NAMES[$i]}"
  args=()
  case "$name" in
    baseline_text_off) ;;
    baseline_text_on) args+=(--text-history);;
    default_text_off) args+=(--adapter "$default_adapter" --memory-mode history);;
    default_reset_text_off) args+=(--adapter "$default_adapter" --memory-mode reset);;
    default_text_on) args+=(--adapter "$default_adapter" --memory-mode history --text-history);;
    warm100_text_off) args+=(--adapter "$warm_adapter" --memory-mode history);;
    dense_text_off) args+=(--adapter "$dense_adapter" --memory-mode history --dense-reading);;
    mean_text_off) args+=(--adapter "$mean_adapter" --memory-mode history);;
  esac
  CUDA_VISIBLE_DEVICES="$gpu" python -m csa_adapter.longform.evaluate \
    --manifest "$DATA_DIR/test.full.jsonl" --output "$RUN_ROOT/test/$name" \
    --model "$MODEL" --revision "$REVISION" --diagnostics "${args[@]}" \
    >"$RUN_ROOT/logs/test_${name}.log" 2>&1 &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
[[ "$status" == 0 ]] || { echo "At least one E22 test job failed" >&2; exit 6; }

# -----------------------------------------------------------------------------
# Stage 5: external E21 evaluation, primary comparisons text-history OFF.
# -----------------------------------------------------------------------------
E21_NAMES=(baseline_text_off default_text_off warm100_text_off dense_text_off mean_text_off)
pids=()
for i in "${!E21_NAMES[@]}"; do
  gpu="${GPU_IDS[$i]}"
  name="${E21_NAMES[$i]}"
  args=()
  case "$name" in
    baseline_text_off) ;;
    default_text_off) args+=(--adapter "$default_adapter" --memory-mode history);;
    warm100_text_off) args+=(--adapter "$warm_adapter" --memory-mode history);;
    dense_text_off) args+=(--adapter "$dense_adapter" --memory-mode history --dense-reading);;
    mean_text_off) args+=(--adapter "$mean_adapter" --memory-mode history);;
  esac
  CUDA_VISIBLE_DEVICES="$gpu" python -m csa_adapter.longform.evaluate \
    --manifest "$DATA_DIR/earnings21.full.jsonl" --output "$RUN_ROOT/e21/$name" \
    --model "$MODEL" --revision "$REVISION" --diagnostics "${args[@]}" \
    >"$RUN_ROOT/logs/e21_${name}.log" 2>&1 &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
[[ "$status" == 0 ]] || { echo "At least one E21 job failed" >&2; exit 7; }

python scripts/summarize_v021.py "$RUN_ROOT" | tee "$RUN_ROOT/results_summary.tsv"
echo "Done. Primary selection: $RUN_ROOT/selected_runs.tsv"
echo "Summary: $RUN_ROOT/results_summary.tsv"
