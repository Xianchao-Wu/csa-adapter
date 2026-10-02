#!/usr/bin/env bash
# CSA-Adapter v0.2.1 / 8xH100-80GB priority experiments (v4)
#
# Goals:
#   1) use the v0.2.1 stability defaults (hard sparse from step 1, bounded alpha,
#      conservative gate init, residual RMS safety cap);
#   2) exploit all eight H100s with multiple independent cached-feature trainers/GPU;
#   3) select checkpoints by free-running long-form validation WER-N, text history OFF;
#   4) never cherry-pick a seed on test: report mean/std across all seeds;
#   5) keep text-history and reset-memory experiments as explicit diagnostics.
#
# Run from anywhere inside the v0.2.1 repository; _repo_env.sh changes to repo root.
set -Eeuo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/scripts/_repo_env.sh"
python scripts/preflight_v021.py

MODEL="${MODEL:-openai/whisper-large-v3}"
REVISION="${REVISION:-main}"
E22_REVISION="${E22_REVISION:-0a034f9ed86d33a3859d9025d3e621cf243773ab}"
E21_REVISION="${E21_REVISION:-5956ebd0b2ba89bbff20e6a001415b23b27fd4cb}"
DATA_DIR="${DATA_DIR:-data/earnings22_622}"
CACHE_DIR="${CACHE_DIR:-cache/earnings22_large_v3}"
RUN_ROOT="${RUN_ROOT:-runs/h100_priority_v4}"
LOG_DIR="${LOG_DIR:-$RUN_ROOT/logs}"
GPU_LIST="${GPUS:-0,1,2,3,4,5,6,7}"
CACHE_GPU_LIST="${CACHE_GPUS:-$GPU_LIST}"

# 4 families x 8 seeds = 32 runs = 4 concurrent trainers/H100 by default.
# This is intentional: cached-feature training has a small GPU footprint, while
# the larger seed set directly measures the instability observed in v0.2.
TRAIN_SEEDS="${TRAIN_SEEDS:-42,43,44,45,46,47,48,49}"
TRAIN_JOBS_PER_GPU="${TRAIN_JOBS_PER_GPU:-4}"
EVAL_JOBS_PER_GPU="${EVAL_JOBS_PER_GPU:-1}"

SPLIT_SEED="${SPLIT_SEED:-42}"
MAX_STEPS="${MAX_STEPS:-1500}"
EPOCHS="${EPOCHS:-99}"
GRAD_ACCUM="${GRAD_ACCUM:-8}"
LR="${LR:-1e-4}"
VALID_EXAMPLES="${VALID_EXAMPLES:-256}"
VALIDATE_EVERY="${VALIDATE_EVERY:-250}"
FEATURE_LRU="${FEATURE_LRU:-256}"
DIAGNOSTICS_EVERY="${DIAGNOSTICS_EVERY:-10}"

# Checkpoint screening uses a small fixed number of full calls, then the selected
# checkpoint is evaluated on the complete validation set. 0 = screen every
# checkpoint on the complete validation set (slower but strictest).
CHECKPOINT_SELECT_MAX_CALLS="${CHECKPOINT_SELECT_MAX_CALLS:-8}"

# Primary stages. E21 is intentionally optional; add `external` after E22 is stable.
STAGES="${STAGES:-setup,prepare,cache,train,select,test,diagnostics,summary}"
PLAN_ONLY="${PLAN_ONLY:-0}"
REQUIRE_IDLE_GPUS="${REQUIRE_IDLE_GPUS:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

mkdir -p "$RUN_ROOT" "$LOG_DIR"

has_stage() { [[ ",$STAGES," == *",$1,"* ]]; }
log() { printf '[%(%F %T)T] %s\n' -1 "$*"; }
die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

cache_complete() {
  python - "$1" <<'PY'
import json, pathlib, sys
root=pathlib.Path(sys.argv[1]); first=root/'metadata-000.json'
if not first.is_file(): raise SystemExit(1)
try: ident=json.loads(first.read_text()); shards=int(ident['shards'])
except (OSError, ValueError, KeyError, json.JSONDecodeError): raise SystemExit(1)
if shards <= 0: raise SystemExit(1)
for i in range(shards):
    m=root/f'metadata-{i:03d}.json'; x=root/f'index-{i:03d}.jsonl'
    if not m.is_file() or not x.is_file(): raise SystemExit(1)
    try:
        if json.loads(m.read_text()) != ident: raise SystemExit(1)
    except (OSError, json.JSONDecodeError): raise SystemExit(1)
PY
}
cache_has_metadata() { compgen -G "$1/metadata-*.json" >/dev/null; }

IFS=',' read -r -a GPUS_ARR <<< "$GPU_LIST"
IFS=',' read -r -a SEEDS <<< "$TRAIN_SEEDS"
[[ "${#GPUS_ARR[@]}" -eq 8 ]] || die "GPUS must contain exactly 8 IDs"
[[ "${#SEEDS[@]}" -ge 2 ]] || die "TRAIN_SEEDS must contain at least 2 seeds"
[[ "$TRAIN_JOBS_PER_GPU" =~ ^[1-9][0-9]*$ ]] || die "TRAIN_JOBS_PER_GPU must be >=1"
[[ "$EVAL_JOBS_PER_GPU" =~ ^[1-9][0-9]*$ ]] || die "EVAL_JOBS_PER_GPU must be >=1"

if [[ "$REQUIRE_IDLE_GPUS" == 1 ]]; then
  for gpu in "${GPUS_ARR[@]}"; do
    used=$(nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader 2>/dev/null | sed '/^[[:space:]]*$/d' | wc -l)
    [[ "$used" -eq 0 ]] || die "GPU $gpu already has a compute process; set REQUIRE_IDLE_GPUS=0 only if intentional"
  done
fi

declare -a PIDS=() NAMES=()
stop_children() { local p; for p in "${PIDS[@]:-}"; do kill "$p" 2>/dev/null || true; done; }
trap stop_children EXIT INT TERM

launch() {
  local name="$1" gpu="$2" logfile="$3"; shift 3
  log "launch GPU $gpu: $name"
  CUDA_VISIBLE_DEVICES="$gpu" "$@" >"$logfile" 2>&1 &
  PIDS+=("$!"); NAMES+=("$name")
}

launch_fn() {
  local name="$1" gpu="$2" logfile="$3" fn="$4"; shift 4
  log "launch GPU $gpu: $name"
  ( export CUDA_VISIBLE_DEVICES="$gpu"; "$fn" "$@" ) >"$logfile" 2>&1 &
  PIDS+=("$!"); NAMES+=("$name")
}

wait_all() {
  local bad=0 i
  for i in "${!PIDS[@]}"; do
    if wait "${PIDS[$i]}"; then log "finished: ${NAMES[$i]}"; else log "FAILED: ${NAMES[$i]}"; bad=1; fi
  done
  PIDS=(); NAMES=()
  [[ "$bad" -eq 0 ]] || die "one or more jobs failed; inspect $LOG_DIR"
}

# This deliberately waits for a whole wave, rather than starting a replacement
# job immediately, so each GPU receives a predictable number of simultaneous jobs.
wait_when_full() { local limit="$1"; (( ${#PIDS[@]} < limit )) || wait_all; }

# -----------------------------------------------------------------------------
# Experiment matrix.
# v0.2 hard_direct becomes the v0.2.1 default. The old v0.2 default is renamed
# warm100 so the dense->sparse warm-up remains a transparent ablation.
# -----------------------------------------------------------------------------
NAMES_CFG=(default warm100 dense mean)
RUN_NAMES=(); RUN_CONFIGS=(); RUN_SEEDS=()
for config in "${NAMES_CFG[@]}"; do
  for seed in "${SEEDS[@]}"; do
    RUN_NAMES+=("${config}_s${seed}")
    RUN_CONFIGS+=("$config")
    RUN_SEEDS+=("$seed")
  done
done

config_args() {
  case "$1" in
    default) printf '%s\n' --compressor event --warmup-steps 0 ;;
    warm100) printf '%s\n' --compressor event --warmup-steps 100 ;;
    dense)   printf '%s\n' --compressor event --dense-always --warmup-steps 0 ;;
    mean)    printf '%s\n' --compressor mean --warmup-steps 0 ;;
    *) die "unknown config: $1" ;;
  esac
}

{
  echo -e "run\tconfig\tseed"
  for i in "${!RUN_NAMES[@]}"; do
    echo -e "${RUN_NAMES[$i]}\t${RUN_CONFIGS[$i]}\t${RUN_SEEDS[$i]}"
  done
} > "$RUN_ROOT/run_manifest.tsv"

train_capacity=$(( ${#GPUS_ARR[@]} * TRAIN_JOBS_PER_GPU ))
eval_capacity=$(( ${#GPUS_ARR[@]} * EVAL_JOBS_PER_GPU ))
log "H100 priority v4: ${#RUN_NAMES[@]} runs = ${#NAMES_CFG[@]} configs x ${#SEEDS[@]} seeds"
log "training capacity: $train_capacity concurrent jobs (${TRAIN_JOBS_PER_GPU}/GPU on ${#GPUS_ARR[@]} H100s)"
log "evaluation capacity: $eval_capacity concurrent jobs (${EVAL_JOBS_PER_GPU}/GPU)"
column -t -s $'\t' "$RUN_ROOT/run_manifest.tsv" 2>/dev/null || cat "$RUN_ROOT/run_manifest.tsv"
[[ "$PLAN_ONLY" == 1 ]] && { log "PLAN_ONLY=1; exiting"; exit 0; }

# -----------------------------------------------------------------------------
# Dataset/cache setup.
# -----------------------------------------------------------------------------
if has_stage setup; then
  bash scripts/00_setup.sh 2>&1 | tee "$LOG_DIR/00_setup.log"
fi

if has_stage prepare; then
  if [[ -f "$DATA_DIR/split_calls.json" && -f "$DATA_DIR/preparation_report.json" ]]; then
    log "reuse prepared data: $DATA_DIR"
  else
    DATA_DIR="$DATA_DIR" SEED="$SPLIT_SEED" bash scripts/01_prepare_earnings22.sh \
      --revision "$E22_REVISION" --external-revision "$E21_REVISION" \
      2>&1 | tee "$LOG_DIR/01_prepare.log"
  fi
fi

if has_stage cache; then
  [[ -f "$DATA_DIR/train.jsonl" && -f "$DATA_DIR/validation.jsonl" ]] || die "missing prepared manifests"
  if cache_complete "$CACHE_DIR/train" && cache_complete "$CACHE_DIR/validation"; then
    log "reuse complete feature cache: $CACHE_DIR"
  elif cache_has_metadata "$CACHE_DIR/train" || cache_has_metadata "$CACHE_DIR/validation"; then
    die "incomplete/inconsistent cache under $CACHE_DIR; preserve it for inspection and use a fresh CACHE_DIR"
  else
    DATA_DIR="$DATA_DIR" CACHE_DIR="$CACHE_DIR" MODEL="$MODEL" REVISION="$REVISION" \
      GPUS="$CACHE_GPU_LIST" bash scripts/02_cache_features.sh 2>&1 | tee "$LOG_DIR/02_cache.log"
  fi
fi

# -----------------------------------------------------------------------------
# Train: 32 cached-feature jobs by default, packed 4/H100.
# -----------------------------------------------------------------------------
if has_stage train; then
  [[ -d "$CACHE_DIR/train" && -d "$CACHE_DIR/validation" ]] || die "missing feature caches"
  for i in "${!RUN_NAMES[@]}"; do
    name="${RUN_NAMES[$i]}"; config="${RUN_CONFIGS[$i]}"; seed="${RUN_SEEDS[$i]}"; run="$RUN_ROOT/$name"
    final_ckpt="$run/checkpoints/step-$(printf '%06d' "$MAX_STEPS")/adapter.safetensors"
    if [[ -f "$final_ckpt" ]]; then log "skip completed training: $name"; continue; fi
    [[ ! -f "$run/run.json" ]] || die "incomplete run exists: $run; optimizer resume is unsupported"
    mapfile -t extra < <(config_args "$config")
    gpu="${GPUS_ARR[$((i % ${#GPUS_ARR[@]}))]}"
    launch "train_$name" "$gpu" "$LOG_DIR/train_${name}.log" \
      python -m csa_adapter.longform.train \
      --train-cache "$CACHE_DIR/train" --valid-cache "$CACHE_DIR/validation" \
      --output "$run" --model "$MODEL" --revision "$REVISION" \
      --seed "$seed" --epochs "$EPOCHS" --max-steps "$MAX_STEPS" \
      --grad-accum "$GRAD_ACCUM" --lr "$LR" \
      --history-segments 64 --compression-rate 8 --max-memory 4096 --top-k 16 --rank 16 \
      --gate diagonal --alpha-init 0.01 --alpha-max 0.10 --gate-bias-init -2.0 \
      --residual-ratio-cap 0.25 --valid-examples "$VALID_EXAMPLES" \
      --validate-every "$VALIDATE_EVERY" --diagnostics-every "$DIAGNOSTICS_EVERY" \
      --feature-lru "$FEATURE_LRU" --save-validation-checkpoints \
      "${extra[@]}"
    wait_when_full "$train_capacity"
  done
  [[ "${#PIDS[@]}" -eq 0 ]] || wait_all
fi

# -----------------------------------------------------------------------------
# Checkpoint selection helper: screen every saved checkpoint using free-running
# long-form validation, text history OFF. This directly guards against the v0.2
# failure mode where teacher-forced NLL was healthy but later segments were empty.
# -----------------------------------------------------------------------------
select_one_run() {
  local name="$1" run="$RUN_ROOT/$name" ckpt tag out
  local config="${name%%_s[0-9]*}"
  shopt -s nullglob
  local ckpts=("$run"/checkpoints/step-*)
  [[ "${#ckpts[@]}" -gt 0 ]] || return 20
  for ckpt in "${ckpts[@]}"; do
    tag="$(basename "$ckpt")"; out="$run/checkpoint_screen/$tag"
    [[ -f "$out/metrics.json" ]] && continue
    local args=(--manifest "$DATA_DIR/validation.full.jsonl" --output "$out" \
                --model "$MODEL" --revision "$REVISION" --adapter "$ckpt" \
                --memory-mode history --diagnostics)
    [[ "$config" == dense ]] && args+=(--dense-reading)
    [[ "$CHECKPOINT_SELECT_MAX_CALLS" != 0 ]] && args+=(--max-calls "$CHECKPOINT_SELECT_MAX_CALLS")
    python -m csa_adapter.longform.evaluate "${args[@]}" \
      >"$LOG_DIR/${name}.${tag}.screen.log" 2>&1
  done
  python - "$run" "$CHECKPOINT_SELECT_MAX_CALLS" <<'PY'
import json, sys
from pathlib import Path
run=Path(sys.argv[1]); max_calls=int(sys.argv[2]); vals=[]
for p in sorted((run/'checkpoint_screen').glob('step-*/metrics.json')):
    d=json.loads(p.read_text())
    vals.append((float(d['normalized_micro_wer']), p.parent.name,
                 int(d.get('empty_hypothesis_chunks', 0))))
if not vals: raise SystemExit('no checkpoint-screen metrics')
# WER-N is the primary selector; empty chunks are a deterministic tie-breaker.
score, tag, empty=min(vals, key=lambda x:(x[0], x[2], x[1]))
(run/'selected_checkpoint.txt').write_text(tag+'\n')
(run/'selected_checkpoint.json').write_text(json.dumps({
    'checkpoint':tag,
    'screen_validation_micro_wer':score,
    'screen_empty_hypothesis_chunks':empty,
    'screen_max_calls':max_calls,
    'selection_text_history':False,
    'selection_memory_mode':'history'
}, indent=2)+'\n')
print(run.name, tag, score, 'empty=', empty)
PY
}

full_validation_one_run() {
  local name="$1" run="$RUN_ROOT/$name" out="$run/eval_validation_full" tag adapter config
  config="${name%%_s[0-9]*}"
  tag="$(tr -d '\r\n' < "$run/selected_checkpoint.txt")"; adapter="$run/checkpoints/$tag"
  [[ -f "$out/metrics.json" ]] && return 0
  local args=(--manifest "$DATA_DIR/validation.full.jsonl" --output "$out" \
              --model "$MODEL" --revision "$REVISION" --adapter "$adapter" \
              --memory-mode history --diagnostics)
  [[ "$config" == dense ]] && args+=(--dense-reading)
  python -m csa_adapter.longform.evaluate "${args[@]}"
}

# No seed is selected here. Every seed proceeds independently to full validation
# and test; final reporting uses mean/std across seeds.
if has_stage select; then
  [[ -f "$DATA_DIR/validation.full.jsonl" ]] || die "missing $DATA_DIR/validation.full.jsonl"
  for i in "${!RUN_NAMES[@]}"; do
    name="${RUN_NAMES[$i]}"; gpu="${GPUS_ARR[$((i % ${#GPUS_ARR[@]}))]}"
    launch_fn "select_$name" "$gpu" "$LOG_DIR/select_${name}.log" select_one_run "$name"
    wait_when_full "$eval_capacity"
  done
  [[ "${#PIDS[@]}" -eq 0 ]] || wait_all

  for i in "${!RUN_NAMES[@]}"; do
    name="${RUN_NAMES[$i]}"; gpu="${GPUS_ARR[$((i % ${#GPUS_ARR[@]}))]}"
    launch_fn "valfull_$name" "$gpu" "$LOG_DIR/valfull_${name}.log" full_validation_one_run "$name"
    wait_when_full "$eval_capacity"
  done
  [[ "${#PIDS[@]}" -eq 0 ]] || wait_all
fi

# -----------------------------------------------------------------------------
# E22 test: every seed, selected only by validation. Primary evaluation keeps
# text history OFF so acoustic memory is isolated from the known text-history
# instability seen in v0.2.
# -----------------------------------------------------------------------------
eval_one_run() {
  local name="$1" corpus="$2" run="$RUN_ROOT/$name" tag adapter manifest out config
  config="${name%%_s[0-9]*}"
  tag="$(tr -d '\r\n' < "$run/selected_checkpoint.txt")"; adapter="$run/checkpoints/$tag"
  case "$corpus" in
    test) manifest="$DATA_DIR/test.full.jsonl"; out="$run/eval_test" ;;
    e21)  manifest="$DATA_DIR/earnings21.full.jsonl"; out="$run/eval_e21" ;;
    *) return 31 ;;
  esac
  [[ -f "$out/metrics.json" ]] && return 0
  local args=(--manifest "$manifest" --output "$out" --model "$MODEL" --revision "$REVISION" \
              --adapter "$adapter" --memory-mode history --diagnostics)
  [[ "$config" == dense ]] && args+=(--dense-reading)
  python -m csa_adapter.longform.evaluate "${args[@]}"
}

if has_stage test; then
  [[ -f "$DATA_DIR/test.full.jsonl" ]] || die "missing E22 test manifest"
  for i in "${!RUN_NAMES[@]}"; do
    name="${RUN_NAMES[$i]}"; gpu="${GPUS_ARR[$((i % ${#GPUS_ARR[@]}))]}"
    launch_fn "test_$name" "$gpu" "$LOG_DIR/test_${name}.log" eval_one_run "$name" test
    wait_when_full "$eval_capacity"
  done
  [[ "${#PIDS[@]}" -eq 0 ]] || wait_all
fi

# -----------------------------------------------------------------------------
# Explicit diagnostics (not used to select model/seed):
#   baseline text off/on;
#   default CSA history+text-off (already eval_test), reset+text-off, history+text-on.
# This cleanly separates acoustic persistence from decoder text history.
# -----------------------------------------------------------------------------
eval_baseline() {
  local name="$1" text_history="$2" out="$RUN_ROOT/diagnostics/$name"
  [[ -f "$out/metrics.json" ]] && return 0
  local args=(--manifest "$DATA_DIR/test.full.jsonl" --output "$out" --model "$MODEL" --revision "$REVISION")
  [[ "$text_history" == 1 ]] && args+=(--text-history)
  python -m csa_adapter.longform.evaluate "${args[@]}"
}

eval_default_diag() {
  local name="$1" mode="$2" run="$RUN_ROOT/$name" tag adapter out
  tag="$(tr -d '\r\n' < "$run/selected_checkpoint.txt")"; adapter="$run/checkpoints/$tag"
  out="$run/eval_diag_${mode}"
  [[ -f "$out/metrics.json" ]] && return 0
  local args=(--manifest "$DATA_DIR/test.full.jsonl" --output "$out" --model "$MODEL" --revision "$REVISION" \
              --adapter "$adapter" --diagnostics)
  case "$mode" in
    reset_text_off) args+=(--memory-mode reset) ;;
    history_text_on) args+=(--memory-mode history --text-history) ;;
    *) return 41 ;;
  esac
  python -m csa_adapter.longform.evaluate "${args[@]}"
}

if has_stage diagnostics; then
  [[ -f "$DATA_DIR/test.full.jsonl" ]] || die "missing E22 test manifest"
  launch_fn "baseline_text_off" "${GPUS_ARR[0]}" "$LOG_DIR/baseline_text_off.log" eval_baseline baseline_text_off 0
  launch_fn "baseline_text_on"  "${GPUS_ARR[1]}" "$LOG_DIR/baseline_text_on.log"  eval_baseline baseline_text_on 1
  wait_when_full "$eval_capacity"

  default_runs=()
  for i in "${!RUN_NAMES[@]}"; do [[ "${RUN_CONFIGS[$i]}" == default ]] && default_runs+=("${RUN_NAMES[$i]}"); done
  diag_i=0
  for name in "${default_runs[@]}"; do
    for mode in reset_text_off history_text_on; do
      gpu="${GPUS_ARR[$((diag_i % ${#GPUS_ARR[@]}))]}"; diag_i=$((diag_i+1))
      launch_fn "diag_${name}_${mode}" "$gpu" "$LOG_DIR/diag_${name}_${mode}.log" eval_default_diag "$name" "$mode"
      wait_when_full "$eval_capacity"
    done
  done
  [[ "${#PIDS[@]}" -eq 0 ]] || wait_all
fi

# Optional E21: every per-seed checkpoint, still text-history OFF. Run only after
# E22 is stable: STAGES=external,summary bash scripts/run_h100_priority_experiments_v4.sh
if has_stage external; then
  [[ -f "$DATA_DIR/earnings21.full.jsonl" ]] || die "missing Earnings-21 manifest"
  for i in "${!RUN_NAMES[@]}"; do
    name="${RUN_NAMES[$i]}"; gpu="${GPUS_ARR[$((i % ${#GPUS_ARR[@]}))]}"
    launch_fn "e21_$name" "$gpu" "$LOG_DIR/e21_${name}.log" eval_one_run "$name" e21
    wait_when_full "$eval_capacity"
  done
  [[ "${#PIDS[@]}" -eq 0 ]] || wait_all
fi

# -----------------------------------------------------------------------------
# Summaries: per-seed and mean/std tables; diagnostics are reported separately.
# -----------------------------------------------------------------------------
summarize() {
python - "$RUN_ROOT" <<'PY'
import csv, json, statistics, sys
from collections import defaultdict
from pathlib import Path
root=Path(sys.argv[1])
rows=list(csv.DictReader((root/'run_manifest.tsv').open(), delimiter='\t'))

def load(path):
    return json.loads(path.read_text()) if path.is_file() else {}
def fmt(x):
    if x is None: return '-'
    if isinstance(x, float): return f'{x:.8f}'
    return str(x)
def ms(vals):
    vals=[float(x) for x in vals if x is not None]
    if not vals: return None,None,0
    return statistics.mean(vals), (statistics.stdev(vals) if len(vals)>1 else 0.0), len(vals)

run_lines=[]; grouped=defaultdict(list)
for r in rows:
    rd=root/r['run']; sel=load(rd/'selected_checkpoint.json'); cfg=load(rd/'run.json')
    val=load(rd/'eval_validation_full'/'metrics.json')
    test=load(rd/'eval_test'/'metrics.json')
    e21=load(rd/'eval_e21'/'metrics.json')
    rec={
      **r,
      'checkpoint':sel.get('checkpoint'),
      'screen_wer':sel.get('screen_validation_micro_wer'),
      'trainable_parameters':cfg.get('trainable_parameters'),
      'val_wer':val.get('normalized_micro_wer'),
      'test_wer':test.get('normalized_micro_wer'),
      'e21_wer':e21.get('normalized_micro_wer'),
      'test_rtf':test.get('rtf'),
      'test_empty':test.get('empty_hypothesis_chunks'),
      'test_alpha':test.get('diag_mean_alpha_effective'),
      'test_gate':test.get('diag_mean_gate_mean'),
      'test_residual_pre':test.get('diag_mean_residual_ratio_pre_cap'),
      'test_residual':test.get('diag_mean_residual_ratio'),
      'test_clip':test.get('diag_mean_residual_clip_fraction'),
      'test_cosine':test.get('diag_mean_adapted_cosine'),
      'test_entropy':test.get('diag_mean_retrieval_entropy'),
    }
    run_lines.append(rec); grouped[r['config']].append(rec)

cols=['run','config','seed','checkpoint','screen_wer','trainable_parameters','val_wer','test_wer','e21_wer',
      'test_rtf','test_empty','test_alpha','test_gate','test_residual_pre','test_residual','test_clip','test_cosine','test_entropy']
with (root/'run_summary.tsv').open('w') as o:
    o.write('\t'.join(cols)+'\n')
    for x in run_lines: o.write('\t'.join(fmt(x.get(c)) for c in cols)+'\n')

acols=['config','n_val','val_mean','val_std','n_test','test_mean','test_std','n_e21','e21_mean','e21_std',
       'test_rtf_mean','test_empty_mean','test_residual_mean','test_clip_mean','test_cosine_mean']
with (root/'aggregate_summary.tsv').open('w') as o:
    o.write('\t'.join(acols)+'\n')
    for config in dict.fromkeys(r['config'] for r in rows):
        xs=grouped[config]
        vm,vs,nv=ms([x['val_wer'] for x in xs]); tm,ts,nt=ms([x['test_wer'] for x in xs]); em,es,ne=ms([x['e21_wer'] for x in xs])
        rm,_,_=ms([x['test_rtf'] for x in xs]); xm,_,_=ms([x['test_empty'] for x in xs])
        drm,_,_=ms([x['test_residual'] for x in xs]); cm,_,_=ms([x['test_clip'] for x in xs]); cosm,_,_=ms([x['test_cosine'] for x in xs])
        vals=[config,nv,vm,vs,nt,tm,ts,ne,em,es,rm,xm,drm,cm,cosm]
        o.write('\t'.join(fmt(v) for v in vals)+'\n')

# Diagnostic table: baseline + default-seed persistence/text-history tests.
diag=[]
for b in ('baseline_text_off','baseline_text_on'):
    d=load(root/'diagnostics'/b/'metrics.json')
    if d: diag.append((b,'-',d.get('normalized_micro_wer'),d.get('rtf'),d.get('empty_hypothesis_chunks')))
for r in rows:
    if r['config'] != 'default': continue
    rd=root/r['run']
    main=load(rd/'eval_test'/'metrics.json')
    reset=load(rd/'eval_diag_reset_text_off'/'metrics.json')
    text=load(rd/'eval_diag_history_text_on'/'metrics.json')
    for label,d in [('history_text_off',main),('reset_text_off',reset),('history_text_on',text)]:
        if d: diag.append((label,r['seed'],d.get('normalized_micro_wer'),d.get('rtf'),d.get('empty_hypothesis_chunks')))
with (root/'diagnostic_summary.tsv').open('w') as o:
    o.write('setting\tseed\twer_n\trtf\tempty_chunks\n')
    for x in diag: o.write('\t'.join(fmt(v) for v in x)+'\n')

print('=== aggregate_summary.tsv ===')
print((root/'aggregate_summary.tsv').read_text(), end='')
if (root/'diagnostic_summary.tsv').is_file():
    print('=== diagnostic_summary.tsv ===')
    print((root/'diagnostic_summary.tsv').read_text(), end='')
PY
}

if has_stage summary; then summarize | tee "$RUN_ROOT/summary.log"; fi

log "H100 priority v4 requested stages complete: $RUN_ROOT"
log "per-seed results : $RUN_ROOT/run_summary.tsv"
log "aggregate results: $RUN_ROOT/aggregate_summary.tsv"
log "diagnostics      : $RUN_ROOT/diagnostic_summary.tsv"
