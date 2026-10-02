#!/usr/bin/env bash
# CSA-Adapter v0.2.1 / H100 priority experiments v4 -- evaluation-only recovery
#
# Reuses already trained checkpoints under runs/h100_priority_v4.
# NO setup / prepare / cache / train stage is executed here.
#
# Default pipeline:
#   checkpoint screening -> full validation -> E22 test -> diagnostics -> summary
# Optional E21:
#   STAGES=external,summary bash run_h100_priority_experiments_v4_recovery.sh
#
# Important fixes vs original v4:
#   * avoid Bash dynamic-scope collisions by never reusing caller-local `name`
#   * split dependent local assignments across statements
#   * fail immediately on evaluator failure and print the relevant log tail
#   * verify selected checkpoint / adapter files before evaluation
#   * resume safely by skipping outputs that already contain metrics.json
#   * support RUN_FILTER for one-run smoke tests

set -Eeuo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/scripts/_repo_env.sh"
python scripts/preflight_v021.py

MODEL="${MODEL:-openai/whisper-large-v3}"
REVISION="${REVISION:-main}"
DATA_DIR="${DATA_DIR:-data/earnings22_622}"
RUN_ROOT="${RUN_ROOT:-runs/h100_priority_v4}"
LOG_DIR="${LOG_DIR:-$RUN_ROOT/logs}"
GPU_LIST="${GPUS:-0,1,2,3,4,5,6,7}"
TRAIN_SEEDS="${TRAIN_SEEDS:-42,43,44,45,46,47,48,49}"
EVAL_JOBS_PER_GPU="${EVAL_JOBS_PER_GPU:-1}"
CHECKPOINT_SELECT_MAX_CALLS="${CHECKPOINT_SELECT_MAX_CALLS:-8}"
STAGES="${STAGES:-select,test,diagnostics,summary}"
RUN_FILTER="${RUN_FILTER:-}"
REQUIRE_IDLE_GPUS="${REQUIRE_IDLE_GPUS:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

mkdir -p "$RUN_ROOT" "$LOG_DIR"

has_stage() { [[ ",$STAGES," == *",$1,"* ]]; }
log() { printf '[%(%F %T)T] %s\n' -1 "$*"; }
die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

IFS=',' read -r -a GPUS_ARR <<< "$GPU_LIST"
IFS=',' read -r -a SEEDS <<< "$TRAIN_SEEDS"
[[ "${#GPUS_ARR[@]}" -ge 1 ]] || die "GPUS must contain >=1 GPU ID"
[[ "${#SEEDS[@]}" -ge 1 ]] || die "TRAIN_SEEDS must contain >=1 seed"
[[ "$EVAL_JOBS_PER_GPU" =~ ^[1-9][0-9]*$ ]] || die "EVAL_JOBS_PER_GPU must be >=1"

if [[ "$REQUIRE_IDLE_GPUS" == 1 ]]; then
  for gpu in "${GPUS_ARR[@]}"; do
    used=$(nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader 2>/dev/null \
      | sed '/^[[:space:]]*$/d' | wc -l)
    [[ "$used" -eq 0 ]] || die "GPU $gpu already has a compute process; set REQUIRE_IDLE_GPUS=0 only if intentional"
  done
fi

declare -a PIDS=() JOB_NAMES=()
stop_children() {
  local p
  for p in "${PIDS[@]:-}"; do
    kill "$p" 2>/dev/null || true
  done
}
trap stop_children EXIT INT TERM

launch_fn() {
  local job_name="$1"
  local gpu="$2"
  local logfile="$3"
  local fn="$4"
  shift 4

  log "launch GPU $gpu: $job_name"
  (
    export CUDA_VISIBLE_DEVICES="$gpu"
    "$fn" "$@"
  ) >"$logfile" 2>&1 &

  PIDS+=("$!")
  JOB_NAMES+=("$job_name")
}

wait_all() {
  local bad=0
  local i
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
  [[ "$bad" -eq 0 ]] || die "one or more jobs failed; inspect $LOG_DIR"
}

wait_when_full() {
  local limit="$1"
  (( ${#PIDS[@]} < limit )) || wait_all
}

# -----------------------------------------------------------------------------
# Experiment matrix: must match the original H100 v4 training script.
# -----------------------------------------------------------------------------
NAMES_CFG=(default warm100 dense mean)
RUN_NAMES=()
RUN_CONFIGS=()
RUN_SEEDS=()

for config in "${NAMES_CFG[@]}"; do
  for seed in "${SEEDS[@]}"; do
    run_name="${config}_s${seed}"
    if [[ -n "$RUN_FILTER" && "$run_name" != $RUN_FILTER ]]; then
      continue
    fi
    RUN_NAMES+=("$run_name")
    RUN_CONFIGS+=("$config")
    RUN_SEEDS+=("$seed")
  done
done

[[ "${#RUN_NAMES[@]}" -gt 0 ]] || die "RUN_FILTER matched no runs: ${RUN_FILTER:-<empty>}"

eval_capacity=$(( ${#GPUS_ARR[@]} * EVAL_JOBS_PER_GPU ))
log "H100 v4 recovery: ${#RUN_NAMES[@]} run(s), ${#GPUS_ARR[@]} GPU(s), eval capacity=$eval_capacity"
log "stages: $STAGES"
[[ -n "$RUN_FILTER" ]] && log "RUN_FILTER=$RUN_FILTER"

# Do not overwrite the canonical run_manifest.tsv for a filtered smoke test.
if [[ -z "$RUN_FILTER" ]]; then
  {
    echo -e "run\tconfig\tseed"
    for i in "${!RUN_NAMES[@]}"; do
      echo -e "${RUN_NAMES[$i]}\t${RUN_CONFIGS[$i]}\t${RUN_SEEDS[$i]}"
    done
  } > "$RUN_ROOT/run_manifest.tsv"
fi

# -----------------------------------------------------------------------------
# Preflight existing trained artifacts. No training is performed.
# -----------------------------------------------------------------------------
preflight_existing_runs() {
  local missing=0
  local i run_name run_dir count

  for i in "${!RUN_NAMES[@]}"; do
    run_name="${RUN_NAMES[$i]}"
    run_dir="$RUN_ROOT/$run_name"

    if [[ ! -f "$run_dir/run.json" ]]; then
      echo "MISSING run.json: $run_dir/run.json" >&2
      missing=1
      continue
    fi

    shopt -s nullglob
    local ckpts=("$run_dir"/checkpoints/step-*)
    count=${#ckpts[@]}
    shopt -u nullglob

    if (( count == 0 )); then
      echo "MISSING checkpoints: $run_dir/checkpoints/step-*" >&2
      missing=1
    else
      log "preflight $run_name: $count checkpoint(s)"
    fi
  done

  [[ "$missing" -eq 0 ]] || die "existing-run preflight failed; training outputs are incomplete"
}

preflight_existing_runs

# -----------------------------------------------------------------------------
# Checkpoint selection: WER-N on long-form validation, text history OFF.
# Dense config retains --dense-reading exactly as in original v4.
# -----------------------------------------------------------------------------
select_one_run() {
  local run_name="$1"
  local run_dir="$RUN_ROOT/$run_name"
  local config="${run_name%%_s[0-9]*}"
  local ckpt tag out screen_log

  echo "[select] run=$run_name"
  echo "[select] run_dir=$run_dir"
  echo "[select] config=$config"
  echo "[select] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

  shopt -s nullglob
  local ckpts=("$run_dir"/checkpoints/step-*)
  shopt -u nullglob

  echo "[select] found ${#ckpts[@]} checkpoint(s)"
  if (( ${#ckpts[@]} == 0 )); then
    echo "ERROR: no checkpoints under $run_dir/checkpoints" >&2
    return 20
  fi

  for ckpt in "${ckpts[@]}"; do
    tag="$(basename "$ckpt")"
    out="$run_dir/checkpoint_screen/$tag"
    screen_log="$LOG_DIR/${run_name}.${tag}.screen.log"

    if [[ -f "$out/metrics.json" ]]; then
      echo "[select] skip existing metrics: $run_name / $tag"
      continue
    fi

    [[ -f "$ckpt/adapter.safetensors" ]] || {
      echo "ERROR: missing adapter: $ckpt/adapter.safetensors" >&2
      return 21
    }

    local args=(
      --manifest "$DATA_DIR/validation.full.jsonl"
      --output "$out"
      --model "$MODEL"
      --revision "$REVISION"
      --adapter "$ckpt"
      --memory-mode history
      --diagnostics
    )
    [[ "$config" == dense ]] && args+=(--dense-reading)
    [[ "$CHECKPOINT_SELECT_MAX_CALLS" != 0 ]] && args+=(--max-calls "$CHECKPOINT_SELECT_MAX_CALLS")

    echo "[select] evaluating $run_name / $tag"
    if ! python -m csa_adapter.longform.evaluate "${args[@]}" >"$screen_log" 2>&1; then
      echo "ERROR: checkpoint evaluation failed: $run_name / $tag" >&2
      echo "------ $screen_log (tail) ------" >&2
      tail -100 "$screen_log" >&2 || true
      echo "--------------------------------" >&2
      return 22
    fi

    if [[ ! -f "$out/metrics.json" ]]; then
      echo "ERROR: evaluator returned success but metrics.json is missing: $out/metrics.json" >&2
      echo "------ $screen_log (tail) ------" >&2
      tail -100 "$screen_log" >&2 || true
      echo "--------------------------------" >&2
      return 23
    fi
  done

  python - "$run_dir" "$CHECKPOINT_SELECT_MAX_CALLS" <<'PY'
import json, sys
from pathlib import Path

run = Path(sys.argv[1])
max_calls = int(sys.argv[2])
vals = []

for p in sorted((run / 'checkpoint_screen').glob('step-*/metrics.json')):
    d = json.loads(p.read_text())
    vals.append((
        float(d['normalized_micro_wer']),
        p.parent.name,
        int(d.get('empty_hypothesis_chunks', 0)),
    ))

if not vals:
    raise SystemExit('no checkpoint-screen metrics')

# WER-N primary; empty chunks deterministic tie-breaker; tag final tie-breaker.
score, tag, empty = min(vals, key=lambda x: (x[0], x[2], x[1]))

(run / 'selected_checkpoint.txt').write_text(tag + '\n')
(run / 'selected_checkpoint.json').write_text(json.dumps({
    'checkpoint': tag,
    'screen_validation_micro_wer': score,
    'screen_empty_hypothesis_chunks': empty,
    'screen_max_calls': max_calls,
    'selection_text_history': False,
    'selection_memory_mode': 'history',
}, indent=2) + '\n')

print(run.name, tag, score, 'empty=', empty)
PY
}

full_validation_one_run() {
  local run_name="$1"
  local run_dir="$RUN_ROOT/$run_name"
  local out="$run_dir/eval_validation_full"
  local config="${run_name%%_s[0-9]*}"
  local tag adapter

  [[ -f "$run_dir/selected_checkpoint.txt" ]] || {
    echo "ERROR: missing selected_checkpoint.txt: $run_name" >&2
    return 30
  }

  tag="$(tr -d '\r\n' < "$run_dir/selected_checkpoint.txt")"
  adapter="$run_dir/checkpoints/$tag"
  [[ -f "$adapter/adapter.safetensors" ]] || {
    echo "ERROR: selected adapter missing: $adapter/adapter.safetensors" >&2
    return 31
  }

  if [[ -f "$out/metrics.json" ]]; then
    echo "[val-full] skip existing metrics: $run_name"
    return 0
  fi

  local args=(
    --manifest "$DATA_DIR/validation.full.jsonl"
    --output "$out"
    --model "$MODEL"
    --revision "$REVISION"
    --adapter "$adapter"
    --memory-mode history
    --diagnostics
  )
  [[ "$config" == dense ]] && args+=(--dense-reading)

  echo "[val-full] $run_name / $tag"
  python -m csa_adapter.longform.evaluate "${args[@]}"
  [[ -f "$out/metrics.json" ]] || {
    echo "ERROR: full validation metrics missing: $out/metrics.json" >&2
    return 32
  }
}

if has_stage select; then
  [[ -f "$DATA_DIR/validation.full.jsonl" ]] || die "missing $DATA_DIR/validation.full.jsonl"

  for i in "${!RUN_NAMES[@]}"; do
    run_name="${RUN_NAMES[$i]}"
    gpu="${GPUS_ARR[$((i % ${#GPUS_ARR[@]}))]}"
    launch_fn "select_$run_name" "$gpu" "$LOG_DIR/select_${run_name}.log" select_one_run "$run_name"
    wait_when_full "$eval_capacity"
  done
  [[ "${#PIDS[@]}" -eq 0 ]] || wait_all

  for i in "${!RUN_NAMES[@]}"; do
    run_name="${RUN_NAMES[$i]}"
    gpu="${GPUS_ARR[$((i % ${#GPUS_ARR[@]}))]}"
    launch_fn "valfull_$run_name" "$gpu" "$LOG_DIR/valfull_${run_name}.log" full_validation_one_run "$run_name"
    wait_when_full "$eval_capacity"
  done
  [[ "${#PIDS[@]}" -eq 0 ]] || wait_all
fi

# -----------------------------------------------------------------------------
# E22 / E21 evaluation of validation-selected checkpoint.
# -----------------------------------------------------------------------------
eval_one_run() {
  local run_name="$1"
  local corpus="$2"
  local run_dir="$RUN_ROOT/$run_name"
  local config="${run_name%%_s[0-9]*}"
  local tag adapter manifest out

  [[ -f "$run_dir/selected_checkpoint.txt" ]] || {
    echo "ERROR: missing selected_checkpoint.txt: $run_name" >&2
    return 40
  }

  tag="$(tr -d '\r\n' < "$run_dir/selected_checkpoint.txt")"
  adapter="$run_dir/checkpoints/$tag"
  [[ -f "$adapter/adapter.safetensors" ]] || {
    echo "ERROR: selected adapter missing: $adapter/adapter.safetensors" >&2
    return 41
  }

  case "$corpus" in
    test)
      manifest="$DATA_DIR/test.full.jsonl"
      out="$run_dir/eval_test"
      ;;
    e21)
      manifest="$DATA_DIR/earnings21.full.jsonl"
      out="$run_dir/eval_e21"
      ;;
    *)
      echo "ERROR: unknown corpus: $corpus" >&2
      return 42
      ;;
  esac

  [[ -f "$manifest" ]] || {
    echo "ERROR: missing manifest: $manifest" >&2
    return 43
  }

  if [[ -f "$out/metrics.json" ]]; then
    echo "[$corpus] skip existing metrics: $run_name"
    return 0
  fi

  local args=(
    --manifest "$manifest"
    --output "$out"
    --model "$MODEL"
    --revision "$REVISION"
    --adapter "$adapter"
    --memory-mode history
    --diagnostics
  )
  [[ "$config" == dense ]] && args+=(--dense-reading)

  echo "[$corpus] $run_name / $tag"
  python -m csa_adapter.longform.evaluate "${args[@]}"
  [[ -f "$out/metrics.json" ]] || {
    echo "ERROR: metrics missing after $corpus evaluation: $out/metrics.json" >&2
    return 44
  }
}

if has_stage test; then
  [[ -f "$DATA_DIR/test.full.jsonl" ]] || die "missing E22 test manifest: $DATA_DIR/test.full.jsonl"
  for i in "${!RUN_NAMES[@]}"; do
    run_name="${RUN_NAMES[$i]}"
    gpu="${GPUS_ARR[$((i % ${#GPUS_ARR[@]}))]}"
    launch_fn "test_$run_name" "$gpu" "$LOG_DIR/test_${run_name}.log" eval_one_run "$run_name" test
    wait_when_full "$eval_capacity"
  done
  [[ "${#PIDS[@]}" -eq 0 ]] || wait_all
fi

# -----------------------------------------------------------------------------
# Diagnostics retained from original H100 v4.
# -----------------------------------------------------------------------------
eval_baseline() {
  local diag_name="$1"
  local text_history="$2"
  local out="$RUN_ROOT/diagnostics/$diag_name"

  if [[ -f "$out/metrics.json" ]]; then
    echo "[diag] skip existing baseline metrics: $diag_name"
    return 0
  fi

  local args=(
    --manifest "$DATA_DIR/test.full.jsonl"
    --output "$out"
    --model "$MODEL"
    --revision "$REVISION"
  )
  [[ "$text_history" == 1 ]] && args+=(--text-history)

  python -m csa_adapter.longform.evaluate "${args[@]}"
  [[ -f "$out/metrics.json" ]] || {
    echo "ERROR: baseline diagnostic metrics missing: $out/metrics.json" >&2
    return 50
  }
}

eval_default_diag() {
  local run_name="$1"
  local mode="$2"
  local run_dir="$RUN_ROOT/$run_name"
  local tag adapter out

  [[ -f "$run_dir/selected_checkpoint.txt" ]] || {
    echo "ERROR: missing selected checkpoint for diagnostic: $run_name" >&2
    return 51
  }

  tag="$(tr -d '\r\n' < "$run_dir/selected_checkpoint.txt")"
  adapter="$run_dir/checkpoints/$tag"
  out="$run_dir/eval_diag_${mode}"

  [[ -f "$adapter/adapter.safetensors" ]] || {
    echo "ERROR: selected adapter missing: $adapter/adapter.safetensors" >&2
    return 52
  }

  if [[ -f "$out/metrics.json" ]]; then
    echo "[diag] skip existing: $run_name / $mode"
    return 0
  fi

  local args=(
    --manifest "$DATA_DIR/test.full.jsonl"
    --output "$out"
    --model "$MODEL"
    --revision "$REVISION"
    --adapter "$adapter"
    --diagnostics
  )

  case "$mode" in
    reset_text_off) args+=(--memory-mode reset) ;;
    history_text_on) args+=(--memory-mode history --text-history) ;;
    *)
      echo "ERROR: unknown diagnostic mode: $mode" >&2
      return 53
      ;;
  esac

  python -m csa_adapter.longform.evaluate "${args[@]}"
  [[ -f "$out/metrics.json" ]] || {
    echo "ERROR: diagnostic metrics missing: $out/metrics.json" >&2
    return 54
  }
}

if has_stage diagnostics; then
  [[ -f "$DATA_DIR/test.full.jsonl" ]] || die "missing E22 test manifest: $DATA_DIR/test.full.jsonl"

  # Baseline diagnostics are global, so only run them for unfiltered/full recovery.
  if [[ -z "$RUN_FILTER" ]]; then
    launch_fn "baseline_text_off" "${GPUS_ARR[0]}" "$LOG_DIR/baseline_text_off.log" eval_baseline baseline_text_off 0
    if (( ${#GPUS_ARR[@]} >= 2 )); then
      launch_fn "baseline_text_on" "${GPUS_ARR[1]}" "$LOG_DIR/baseline_text_on.log" eval_baseline baseline_text_on 1
    else
      launch_fn "baseline_text_on" "${GPUS_ARR[0]}" "$LOG_DIR/baseline_text_on.log" eval_baseline baseline_text_on 1
    fi
    wait_when_full "$eval_capacity"
  fi

  default_runs=()
  for i in "${!RUN_NAMES[@]}"; do
    [[ "${RUN_CONFIGS[$i]}" == default ]] && default_runs+=("${RUN_NAMES[$i]}")
  done

  diag_i=0
  for run_name in "${default_runs[@]}"; do
    for mode in reset_text_off history_text_on; do
      gpu="${GPUS_ARR[$((diag_i % ${#GPUS_ARR[@]}))]}"
      diag_i=$((diag_i + 1))
      launch_fn "diag_${run_name}_${mode}" "$gpu" "$LOG_DIR/diag_${run_name}_${mode}.log" \
        eval_default_diag "$run_name" "$mode"
      wait_when_full "$eval_capacity"
    done
  done
  [[ "${#PIDS[@]}" -eq 0 ]] || wait_all
fi

if has_stage external; then
  [[ -f "$DATA_DIR/earnings21.full.jsonl" ]] || die "missing Earnings-21 manifest: $DATA_DIR/earnings21.full.jsonl"
  for i in "${!RUN_NAMES[@]}"; do
    run_name="${RUN_NAMES[$i]}"
    gpu="${GPUS_ARR[$((i % ${#GPUS_ARR[@]}))]}"
    launch_fn "e21_$run_name" "$gpu" "$LOG_DIR/e21_${run_name}.log" eval_one_run "$run_name" e21
    wait_when_full "$eval_capacity"
  done
  [[ "${#PIDS[@]}" -eq 0 ]] || wait_all
fi

# -----------------------------------------------------------------------------
# Summary. For full runs this reproduces the original H100 tables and adds status.
# For RUN_FILTER smoke tests it writes a separate filtered summary without replacing
# the canonical run_manifest.tsv.
# -----------------------------------------------------------------------------
summarize() {
  python - "$RUN_ROOT" "$RUN_FILTER" "$TRAIN_SEEDS" <<'PY'
import csv, json, statistics, sys
from collections import defaultdict
from pathlib import Path

root = Path(sys.argv[1])
run_filter = sys.argv[2]
seeds = [x for x in sys.argv[3].split(',') if x]
configs = ['default', 'warm100', 'dense', 'mean']

if run_filter:
    rows=[]
    for config in configs:
        for seed in seeds:
            name=f'{config}_s{seed}'
            if name == run_filter:
                rows.append({'run':name,'config':config,'seed':seed})
    suffix='.filtered'
else:
    manifest=root/'run_manifest.tsv'
    if not manifest.is_file():
        raise SystemExit(f'missing {manifest}')
    rows=list(csv.DictReader(manifest.open(), delimiter='\t'))
    suffix=''

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

run_lines=[]
grouped=defaultdict(list)
for r in rows:
    rd=root/r['run']
    sel=load(rd/'selected_checkpoint.json')
    cfg=load(rd/'run.json')
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
    run_lines.append(rec)
    grouped[r['config']].append(rec)

cols=['run','config','seed','checkpoint','screen_wer','trainable_parameters','val_wer','test_wer','e21_wer',
      'test_rtf','test_empty','test_alpha','test_gate','test_residual_pre','test_residual','test_clip','test_cosine','test_entropy']
run_summary=root/f'run_summary{suffix}.tsv'
with run_summary.open('w') as o:
    o.write('\t'.join(cols)+'\n')
    for x in run_lines:
        o.write('\t'.join(fmt(x.get(c)) for c in cols)+'\n')

acols=['config','n_val','val_mean','val_std','n_test','test_mean','test_std','n_e21','e21_mean','e21_std',
       'test_rtf_mean','test_empty_mean','test_residual_mean','test_clip_mean','test_cosine_mean']
agg_summary=root/f'aggregate_summary{suffix}.tsv'
with agg_summary.open('w') as o:
    o.write('\t'.join(acols)+'\n')
    for config in configs:
        xs=grouped.get(config, [])
        if not xs: continue
        vm,vs,nv=ms([x['val_wer'] for x in xs])
        tm,ts,nt=ms([x['test_wer'] for x in xs])
        em,es,ne=ms([x['e21_wer'] for x in xs])
        rm,_,_=ms([x['test_rtf'] for x in xs])
        xm,_,_=ms([x['test_empty'] for x in xs])
        drm,_,_=ms([x['test_residual'] for x in xs])
        cm,_,_=ms([x['test_clip'] for x in xs])
        cosm,_,_=ms([x['test_cosine'] for x in xs])
        vals=[config,nv,vm,vs,nt,tm,ts,ne,em,es,rm,xm,drm,cm,cosm]
        o.write('\t'.join(fmt(v) for v in vals)+'\n')

# Diagnostics table.
diag=[]
for b in ('baseline_text_off','baseline_text_on'):
    d=load(root/'diagnostics'/b/'metrics.json')
    if d:
        diag.append((b,'-',d.get('normalized_micro_wer'),d.get('rtf'),d.get('empty_hypothesis_chunks')))
for r in rows:
    if r['config'] != 'default': continue
    rd=root/r['run']
    main=load(rd/'eval_test'/'metrics.json')
    reset=load(rd/'eval_diag_reset_text_off'/'metrics.json')
    text=load(rd/'eval_diag_history_text_on'/'metrics.json')
    for label,d in [('history_text_off',main),('reset_text_off',reset),('history_text_on',text)]:
        if d:
            diag.append((label,r['seed'],d.get('normalized_micro_wer'),d.get('rtf'),d.get('empty_hypothesis_chunks')))
diag_summary=root/f'diagnostic_summary{suffix}.tsv'
with diag_summary.open('w') as o:
    o.write('setting\tseed\twer_n\trtf\tempty_chunks\n')
    for x in diag:
        o.write('\t'.join(fmt(v) for v in x)+'\n')

status={
    'runs_total': len(rows),
    'selected': sum((root/r['run']/'selected_checkpoint.json').is_file() for r in rows),
    'full_validation_metrics': sum((root/r['run']/'eval_validation_full'/'metrics.json').is_file() for r in rows),
    'e22_test_metrics': sum((root/r['run']/'eval_test'/'metrics.json').is_file() for r in rows),
    'e21_test_metrics': sum((root/r['run']/'eval_e21'/'metrics.json').is_file() for r in rows),
    'default_reset_diag_metrics': sum((root/r['run']/'eval_diag_reset_text_off'/'metrics.json').is_file() for r in rows if r['config']=='default'),
    'default_history_text_on_diag_metrics': sum((root/r['run']/'eval_diag_history_text_on'/'metrics.json').is_file() for r in rows if r['config']=='default'),
}
status_path=root/f'evaluation_status{suffix}.json'
status_path.write_text(json.dumps(status, indent=2)+'\n')

print('=== evaluation status ===')
print(json.dumps(status, indent=2))
print('=== aggregate summary ===')
print(agg_summary.read_text(), end='')
if diag_summary.is_file():
    print('=== diagnostic summary ===')
    print(diag_summary.read_text(), end='')
PY
}

if has_stage summary; then
  summarize | tee "$RUN_ROOT/summary${RUN_FILTER:+.filtered}.log"
fi

log "H100 v4 recovery complete: $RUN_ROOT"
if [[ -z "$RUN_FILTER" ]]; then
  log "per-seed results : $RUN_ROOT/run_summary.tsv"
  log "aggregate results: $RUN_ROOT/aggregate_summary.tsv"
  log "diagnostics      : $RUN_ROOT/diagnostic_summary.tsv"
  log "status           : $RUN_ROOT/evaluation_status.json"
else
  log "filtered results : $RUN_ROOT/run_summary.filtered.tsv"
  log "filtered status  : $RUN_ROOT/evaluation_status.filtered.json"
fi
