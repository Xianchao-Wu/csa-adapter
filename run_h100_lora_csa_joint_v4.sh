#!/usr/bin/env bash
# H100 LoRA + CSA joint experiment for CSA-Adapter v0.2.1
#
# Sequential protocol:
#   Whisper-large-v3
#     -> LoRA domain adaptation
#     -> merge LoRA into the base model
#     -> RE-CACHE train/validation features from the LoRA-merged model
#     -> train CSA on those matching caches
#     -> select CSA checkpoint by long-form validation WER-N
#     -> full validation / Earnings-22 test / Earnings-21 external test
#
# IMPORTANT:
# Never reuse the original Whisper cache after LoRA. The LoRA model changes
# q_proj/v_proj weights, including encoder attention, so the acoustic cache
# must be regenerated from the LoRA-merged model.
#
# Smoke test:
#   RANKS=8 SEEDS=42 bash run_h100_lora_csa_joint_v1.sh
#
# Full experiment:
#   RANKS=8,16,32,64 SEEDS=42,43,44 bash run_h100_lora_csa_joint_v1.sh
#
# Resume examples:
#   STAGES=cache,csa,eval,summary bash run_h100_lora_csa_joint_v1.sh
#   STAGES=eval,summary bash run_h100_lora_csa_joint_v1.sh

set -Eeuo pipefail

REPO="${REPO:-$(pwd)}"
cd "$REPO"

[[ -f scripts/_repo_env.sh ]] || {
  echo "ERROR: run this from the CSA-Adapter v0.2.1 repository root" >&2
  exit 1
}
source scripts/_repo_env.sh
python scripts/preflight_v021.py

PY_DRIVER="${PY_DRIVER:-train_eval_lora_csa_joint_v4.py}"
LORA_TRAIN_SCRIPT="${LORA_TRAIN_SCRIPT:-train_whisper_lora_earnings_v2.py}"

MODEL="${MODEL:-openai/whisper-large-v3}"
REVISION="${REVISION:-main}"
DATA_DIR="${DATA_DIR:-data/earnings22_622}"

LORA_ROOT="${LORA_ROOT:-runs/whisper_lora_earnings22_622}"
CACHE_ROOT="${CACHE_ROOT:-cache/earnings22_lora_csa_h100_v1}"
JOINT_ROOT="${JOINT_ROOT:-runs/h100_lora_csa_joint_v1}"

RANKS="${RANKS:-8,16,32,64}"
# Conservative first run. For paper statistics use SEEDS=42,43,44.
SEEDS="${SEEDS:-42}"

GPU_LIST="${GPUS:-0,1,2,3,4,5,6,7}"
CACHE_GPU_LIST="${CACHE_GPUS:-$GPU_LIST}"

# Every stage is resume-safe.
STAGES="${STAGES:-lora,cache,csa,eval,summary}"

# LoRA hyperparameters: same standalone LoRA sweep defaults.
LORA_EPOCHS="${LORA_EPOCHS:-3}"
LORA_MAX_STEPS="${LORA_MAX_STEPS:--1}"
LORA_BATCH="${LORA_BATCH:-8}"
LORA_EVAL_BATCH="${LORA_EVAL_BATCH:-8}"
LORA_GRAD_ACCUM="${LORA_GRAD_ACCUM:-4}"
LORA_LR="${LORA_LR:-1e-4}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"

# CSA: same H100 v4 default anchor.
CSA_MAX_STEPS="${CSA_MAX_STEPS:-1500}"
CSA_EPOCHS="${CSA_EPOCHS:-99}"
CSA_GRAD_ACCUM="${CSA_GRAD_ACCUM:-8}"
CSA_LR="${CSA_LR:-1e-4}"
VALID_EXAMPLES="${VALID_EXAMPLES:-256}"
VALIDATE_EVERY="${VALIDATE_EVERY:-250}"
DIAGNOSTICS_EVERY="${DIAGNOSTICS_EVERY:-10}"
FEATURE_LRU="${FEATURE_LRU:-256}"
CHECKPOINT_SELECT_MAX_CALLS="${CHECKPOINT_SELECT_MAX_CALLS:-8}"

CSA_JOBS_PER_GPU="${CSA_JOBS_PER_GPU:-1}"
EVAL_JOBS_PER_GPU="${EVAL_JOBS_PER_GPU:-1}"
REQUIRE_IDLE_GPUS="${REQUIRE_IDLE_GPUS:-1}"
FORCE_RECACHE="${FORCE_RECACHE:-0}"
EVAL_E21="${EVAL_E21:-1}"

mkdir -p "$JOINT_ROOT/logs" "$CACHE_ROOT"

has_stage() { [[ ",$STAGES," == *",$1,"* ]]; }
log() { printf '[%(%F %T)T] %s\n' -1 "$*"; }
die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

[[ -f "$PY_DRIVER" ]] || die "missing Python driver: $PY_DRIVER"

for f in \
  "$DATA_DIR/train.jsonl" \
  "$DATA_DIR/validation.jsonl" \
  "$DATA_DIR/validation.full.jsonl" \
  "$DATA_DIR/test.full.jsonl" \
  "$DATA_DIR/earnings21.full.jsonl"
do
  [[ -f "$f" ]] || die "missing data file: $f"
done

IFS=',' read -r -a RANK_ARR <<< "$RANKS"
IFS=',' read -r -a SEED_ARR <<< "$SEEDS"
IFS=',' read -r -a GPUS_ARR <<< "$GPU_LIST"

[[ "${#GPUS_ARR[@]}" -ge 1 ]] || die "GPUS cannot be empty"
[[ "$CSA_JOBS_PER_GPU" =~ ^[1-9][0-9]*$ ]] || die "CSA_JOBS_PER_GPU must be >=1"
[[ "$EVAL_JOBS_PER_GPU" =~ ^[1-9][0-9]*$ ]] || die "EVAL_JOBS_PER_GPU must be >=1"

if [[ "$REQUIRE_IDLE_GPUS" == 1 ]]; then
  for gpu in "${GPUS_ARR[@]}"; do
    used="$(nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader 2>/dev/null \
      | sed '/^[[:space:]]*$/d' | wc -l)"
    [[ "$used" -eq 0 ]] || die \
      "GPU $gpu already has a compute process; use REQUIRE_IDLE_GPUS=0 only if intentional"
  done
fi

RUN_NAMES=()
RUN_RANKS=()
RUN_SEEDS=()
for rank in "${RANK_ARR[@]}"; do
  case "$rank" in 8|16|32|64) ;; *) die "unsupported LoRA rank: $rank" ;; esac
  for seed in "${SEED_ARR[@]}"; do
    RUN_NAMES+=("rank${rank}_s${seed}")
    RUN_RANKS+=("$rank")
    RUN_SEEDS+=("$seed")
  done
done

{
  echo -e "run\tlora_rank\tseed"
  for i in "${!RUN_NAMES[@]}"; do
    echo -e "${RUN_NAMES[$i]}\t${RUN_RANKS[$i]}\t${RUN_SEEDS[$i]}"
  done
} > "$JOINT_ROOT/run_manifest.tsv"

log "LoRA+CSA H100 v1: ${#RUN_NAMES[@]} run(s)"
column -t -s $'\t' "$JOINT_ROOT/run_manifest.tsv" 2>/dev/null || cat "$JOINT_ROOT/run_manifest.tsv"

declare -a PIDS=() JOB_NAMES=()

stop_children() {
  local p
  for p in "${PIDS[@]:-}"; do kill "$p" 2>/dev/null || true; done
}
trap stop_children EXIT INT TERM

wait_all() {
  local bad=0 i
  for i in "${!PIDS[@]}"; do
    if wait "${PIDS[$i]}"; then
      log "finished: ${JOB_NAMES[$i]}"
    else
      log "FAILED: ${JOB_NAMES[$i]}"
      bad=1
    fi
  done
  PIDS=()
  JOB_NAMES=()
  [[ "$bad" -eq 0 ]] || die "one or more jobs failed; inspect $JOINT_ROOT/logs"
}

wait_when_full() {
  local limit="$1"
  (( ${#PIDS[@]} < limit )) || wait_all
}

cache_complete() {
  python - "$1" <<'PY'
import json
import pathlib
import sys
root = pathlib.Path(sys.argv[1])
first = root / "metadata-000.json"
if not first.is_file(): raise SystemExit(1)
try:
    ident = json.loads(first.read_text())
    shards = int(ident["shards"])
except (OSError, ValueError, KeyError, json.JSONDecodeError):
    raise SystemExit(1)
if shards <= 0: raise SystemExit(1)
for i in range(shards):
    m = root / f"metadata-{i:03d}.json"
    x = root / f"index-{i:03d}.jsonl"
    if not m.is_file() or not x.is_file(): raise SystemExit(1)
    try:
        if json.loads(m.read_text()) != ident: raise SystemExit(1)
    except (OSError, json.JSONDecodeError):
        raise SystemExit(1)
PY
}

cache_has_metadata() { compgen -G "$1/metadata-*.json" >/dev/null; }

lora_fingerprint() {
  python - "$1" <<'PYFP'
import hashlib
import pathlib
import sys
root = pathlib.Path(sys.argv[1])
candidates = [
    root / "train_summary.json",
    root / "adapter" / "adapter_config.json",
    root / "adapter" / "adapter_model.safetensors",
    root / "adapter" / "adapter.safetensors",
]
h = hashlib.sha256()
seen = 0
for p in candidates:
    if p.is_file():
        seen += 1
        h.update(str(p.relative_to(root)).encode())
        with p.open("rb") as f:
            while True:
                b = f.read(1024 * 1024)
                if not b:
                    break
                h.update(b)
if seen == 0:
    raise SystemExit("cannot fingerprint LoRA run: no adapter/summary files found")
print(h.hexdigest())
PYFP
}

verify_cache_fingerprint() {
  local run="$1" cache_dir="$2"
  local current marker
  current="$(lora_fingerprint "$LORA_ROOT/$run")"
  marker="$cache_dir/.lora_fingerprint"
  [[ -f "$marker" ]] || return 1
  [[ "$(tr -d '\r\n' < "$marker")" == "$current" ]]
}

# -----------------------------------------------------------------------------
# Stage 1: LoRA
# -----------------------------------------------------------------------------
if has_stage lora; then
  [[ -f "$LORA_TRAIN_SCRIPT" ]] || die "missing LoRA trainer: $LORA_TRAIN_SCRIPT"

  for i in "${!RUN_NAMES[@]}"; do
    run="${RUN_NAMES[$i]}"
    rank="${RUN_RANKS[$i]}"
    seed="${RUN_SEEDS[$i]}"
    gpu="${GPUS_ARR[$((i % ${#GPUS_ARR[@]}))]}"
    out="$LORA_ROOT/$run"

    if [[ -f "$out/merged_model/config.json" && -f "$out/train_summary.json" ]]; then
      log "reuse LoRA merged model: $run"
      continue
    fi

    log "launch GPU $gpu: LoRA $run"
    (
      export CUDA_VISIBLE_DEVICES="$gpu"
      python "$LORA_TRAIN_SCRIPT" \
        --model "$MODEL" \
        --revision "$REVISION" \
        --train-manifest "$DATA_DIR/train.jsonl" \
        --valid-manifest "$DATA_DIR/validation.jsonl" \
        --output-dir "$out" \
        --rank "$rank" \
        --seed "$seed" \
        --epochs "$LORA_EPOCHS" \
        --max-steps "$LORA_MAX_STEPS" \
        --learning-rate "$LORA_LR" \
        --train-batch-size "$LORA_BATCH" \
        --eval-batch-size "$LORA_EVAL_BATCH" \
        --grad-accum "$LORA_GRAD_ACCUM" \
        --lora-dropout "$LORA_DROPOUT" \
        --bf16 --tf32 --merge
    ) >"$JOINT_ROOT/logs/lora_${run}.log" 2>&1 &

    PIDS+=("$!")
    JOB_NAMES+=("lora_$run")
    wait_when_full "${#GPUS_ARR[@]}"
  done
  [[ "${#PIDS[@]}" -eq 0 ]] || wait_all
fi

if has_stage cache || has_stage csa || has_stage eval; then
  for run in "${RUN_NAMES[@]}"; do
    [[ -f "$LORA_ROOT/$run/merged_model/config.json" ]] || \
      die "LoRA merged model missing for $run: $LORA_ROOT/$run/merged_model"
  done
fi

# -----------------------------------------------------------------------------
# Stage 2: re-cache FROM EACH LoRA-merged model.
# Serial across LoRA runs because 02_cache_features.sh parallelizes internally.
# -----------------------------------------------------------------------------
if has_stage cache; then
  for run in "${RUN_NAMES[@]}"; do
    lora_model="$LORA_ROOT/$run/merged_model"
    cache_dir="$CACHE_ROOT/$run"

    if cache_complete "$cache_dir/train" && cache_complete "$cache_dir/validation"; then
      if verify_cache_fingerprint "$run" "$cache_dir"; then
        log "reuse complete LoRA-derived cache with matching fingerprint: $run"
        continue
      fi
      if [[ "$FORCE_RECACHE" == 1 ]]; then
        log "FORCE_RECACHE=1: cache fingerprint missing/mismatched; remove $cache_dir"
        rm -rf "$cache_dir"
      else
        die "complete cache exists for $run but LoRA fingerprint is missing/mismatched; rerun with FORCE_RECACHE=1"
      fi
    fi

    if cache_has_metadata "$cache_dir/train" || cache_has_metadata "$cache_dir/validation"; then
      if [[ "$FORCE_RECACHE" == 1 ]]; then
        log "FORCE_RECACHE=1: remove incomplete/stale cache $cache_dir"
        rm -rf "$cache_dir"
      else
        die "incomplete cache for $run at $cache_dir; inspect it or rerun with FORCE_RECACHE=1"
      fi
    fi

    log "cache LoRA features: $run -> $cache_dir"
    MODEL="$lora_model" \
    REVISION=main \
    DATA_DIR="$DATA_DIR" \
    CACHE_DIR="$cache_dir" \
    GPUS="$CACHE_GPU_LIST" \
      bash scripts/02_cache_features.sh \
      >"$JOINT_ROOT/logs/cache_${run}.log" 2>&1

    cache_complete "$cache_dir/train" || die "train cache incomplete after caching: $run"
    cache_complete "$cache_dir/validation" || die "validation cache incomplete after caching: $run"
    lora_fingerprint "$LORA_ROOT/$run" > "$cache_dir/.lora_fingerprint"
  done
fi

# -----------------------------------------------------------------------------
# Stage 3: CSA training on matching LoRA-derived cache.
# -----------------------------------------------------------------------------
if has_stage csa; then
  capacity=$(( ${#GPUS_ARR[@]} * CSA_JOBS_PER_GPU ))

  for i in "${!RUN_NAMES[@]}"; do
    run="${RUN_NAMES[$i]}"
    seed="${RUN_SEEDS[$i]}"
    gpu="${GPUS_ARR[$((i % ${#GPUS_ARR[@]}))]}"
    lora_model="$LORA_ROOT/$run/merged_model"
    cache_dir="$CACHE_ROOT/$run"
    out="$JOINT_ROOT/$run"

    cache_complete "$cache_dir/train" || die "missing complete train cache for $run"
    cache_complete "$cache_dir/validation" || die "missing complete valid cache for $run"
    verify_cache_fingerprint "$run" "$cache_dir" || \
      die "LoRA/cache fingerprint mismatch for $run; run STAGES=cache with FORCE_RECACHE=1"

    final_ckpt="$out/checkpoints/step-$(printf '%06d' "$CSA_MAX_STEPS")/adapter.safetensors"
    if [[ -f "$final_ckpt" ]]; then
      log "reuse completed CSA training: $run"
      continue
    fi

    log "launch GPU $gpu: CSA on LoRA base $run"
    (
      export CUDA_VISIBLE_DEVICES="$gpu"
      python "$PY_DRIVER" run \
        --stage train \
        --run-name "$run" \
        --lora-model "$lora_model" \
        --train-cache "$cache_dir/train" \
        --valid-cache "$cache_dir/validation" \
        --output "$out" \
        --revision main \
        --validation-full "$DATA_DIR/validation.full.jsonl" \
        --test-full "$DATA_DIR/test.full.jsonl" \
        --e21-full "$DATA_DIR/earnings21.full.jsonl" \
        --seed "$seed" \
        --epochs "$CSA_EPOCHS" \
        --max-steps "$CSA_MAX_STEPS" \
        --grad-accum "$CSA_GRAD_ACCUM" \
        --lr "$CSA_LR" \
        --valid-examples "$VALID_EXAMPLES" \
        --validate-every "$VALIDATE_EVERY" \
        --diagnostics-every "$DIAGNOSTICS_EVERY" \
        --feature-lru "$FEATURE_LRU" \
        --checkpoint-select-max-calls "$CHECKPOINT_SELECT_MAX_CALLS"
    ) >"$JOINT_ROOT/logs/csa_train_${run}.log" 2>&1 &

    PIDS+=("$!")
    JOB_NAMES+=("csa_$run")
    wait_when_full "$capacity"
  done
  [[ "${#PIDS[@]}" -eq 0 ]] || wait_all
fi

# -----------------------------------------------------------------------------
# Stage 4: paired evaluation. First evaluate LoRA-only with the exact same
# fixed-window long-form evaluator, then evaluate LoRA+CSA.
# -----------------------------------------------------------------------------
if has_stage eval; then
  capacity=$(( ${#GPUS_ARR[@]} * EVAL_JOBS_PER_GPU ))

  # 4a. LoRA-only: needed for a paired complementarity comparison.
  for i in "${!RUN_NAMES[@]}"; do
    run="${RUN_NAMES[$i]}"
    gpu="${GPUS_ARR[$((i % ${#GPUS_ARR[@]}))]}"
    lora_model="$LORA_ROOT/$run/merged_model"

    args=(
      python "$PY_DRIVER" lora-eval
      --lora-model "$lora_model"
      --output "$LORA_ROOT/$run"
      --revision main
      --validation-full "$DATA_DIR/validation.full.jsonl"
      --test-full "$DATA_DIR/test.full.jsonl"
      --e21-full "$DATA_DIR/earnings21.full.jsonl"
    )
    [[ "$EVAL_E21" == 1 ]] || args+=(--skip-e21)

    log "launch GPU $gpu: evaluate LoRA-only $run"
    (
      export CUDA_VISIBLE_DEVICES="$gpu"
      "${args[@]}"
    ) >"$JOINT_ROOT/logs/lora_eval_${run}.log" 2>&1 &

    PIDS+=("$!")
    JOB_NAMES+=("lora_eval_$run")
    wait_when_full "$capacity"
  done
  [[ "${#PIDS[@]}" -eq 0 ]] || wait_all

  # 4b. LoRA+CSA: checkpoint screen, selection and full evaluation.
  for i in "${!RUN_NAMES[@]}"; do
    run="${RUN_NAMES[$i]}"
    seed="${RUN_SEEDS[$i]}"
    gpu="${GPUS_ARR[$((i % ${#GPUS_ARR[@]}))]}"
    lora_model="$LORA_ROOT/$run/merged_model"
    cache_dir="$CACHE_ROOT/$run"
    out="$JOINT_ROOT/$run"

    args=(
      python "$PY_DRIVER" run
      --stage evaluate
      --run-name "$run"
      --lora-model "$lora_model"
      --train-cache "$cache_dir/train"
      --valid-cache "$cache_dir/validation"
      --output "$out"
      --revision main
      --validation-full "$DATA_DIR/validation.full.jsonl"
      --test-full "$DATA_DIR/test.full.jsonl"
      --e21-full "$DATA_DIR/earnings21.full.jsonl"
      --seed "$seed"
      --max-steps "$CSA_MAX_STEPS"
      --checkpoint-select-max-calls "$CHECKPOINT_SELECT_MAX_CALLS"
    )
    [[ "$EVAL_E21" == 1 ]] || args+=(--skip-e21)

    log "launch GPU $gpu: evaluate LoRA+CSA $run"
    (
      export CUDA_VISIBLE_DEVICES="$gpu"
      "${args[@]}"
    ) >"$JOINT_ROOT/logs/joint_eval_${run}.log" 2>&1 &

    PIDS+=("$!")
    JOB_NAMES+=("eval_$run")
    wait_when_full "$capacity"
  done
  [[ "${#PIDS[@]}" -eq 0 ]] || wait_all
fi

# -----------------------------------------------------------------------------
# Stage 5: paired LoRA-only vs LoRA+CSA summary.
# delta = joint - LoRA; negative is better.
# -----------------------------------------------------------------------------
if has_stage summary; then
  python "$PY_DRIVER" summary \
    --joint-root "$JOINT_ROOT" \
    --run-manifest "$JOINT_ROOT/run_manifest.tsv" \
    --lora-only-root "$LORA_ROOT" \
    | tee "$JOINT_ROOT/summary.log"
fi

log "H100 LoRA+CSA joint experiment complete: $JOINT_ROOT"
log "paired per-run : $JOINT_ROOT/joint_run_summary.tsv"
log "aggregate      : $JOINT_ROOT/joint_aggregate_summary.tsv"
log "status         : $JOINT_ROOT/joint_status.json"
