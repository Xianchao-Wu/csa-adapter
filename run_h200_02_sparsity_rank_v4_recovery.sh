#!/usr/bin/env bash
# CSA-Adapter v0.2.1 / DGX-H200-02
# Evaluation-only recovery script for top-k sparsity and low-rank indexing sweeps.
# Reuses existing checkpoints and never retrains.
set -Eeuo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/scripts/_repo_env.sh"
python scripts/preflight_v021.py

MODEL="${MODEL:-openai/whisper-large-v3}"
REVISION="${REVISION:-main}"
DATA_DIR="${DATA_DIR:-data/earnings22_622}"
RUN_ROOT="${RUN_ROOT:-runs/h200_02_sparsity_rank_v4}"
LOG_DIR="${LOG_DIR:-$RUN_ROOT/logs}"
GPU_LIST="${GPUS:-0,1,2,3,4,5,6,7}"
TRAIN_SEEDS="${TRAIN_SEEDS:-42,43,44}"
CHECKPOINT_SELECT_MAX_CALLS="${CHECKPOINT_SELECT_MAX_CALLS:-8}"
STAGES="${STAGES:-select,test,external,summary}"
RUN_FILTER="${RUN_FILTER:-}"
REQUIRE_IDLE_GPUS="${REQUIRE_IDLE_GPUS:-1}"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
mkdir -p "$RUN_ROOT" "$LOG_DIR"

has_stage() { [[ ",$STAGES," == *",$1,"* ]]; }
log() { printf '[%(%F %T)T] %s\n' -1 "$*"; }
die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

IFS=',' read -r -a GPUS_ARR <<< "$GPU_LIST"
IFS=',' read -r -a SEEDS <<< "$TRAIN_SEEDS"
[[ "${#GPUS_ARR[@]}" -ge 1 ]] || die "GPUS must contain at least one GPU ID"
[[ "${#SEEDS[@]}" -ge 1 ]] || die "TRAIN_SEEDS must contain >=1 seed"

if [[ "$REQUIRE_IDLE_GPUS" == 1 ]]; then
  for gpu in "${GPUS_ARR[@]}"; do
    used=$(nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader 2>/dev/null | sed '/^[[:space:]]*$/d' | wc -l)
    [[ "$used" -eq 0 ]] || die "GPU $gpu already has a compute process; set REQUIRE_IDLE_GPUS=0 only if intentional"
  done
fi

declare -a PIDS=() JOB_NAMES=()
stop_children() { local p; for p in "${PIDS[@]:-}"; do kill "$p" 2>/dev/null || true; done; }
trap stop_children EXIT INT TERM

launch_fn() {
  local job_name="$1"
  local gpu="$2"
  local logfile="$3"
  local fn="$4"
  shift 4
  log "launch GPU $gpu: $job_name"
  ( export CUDA_VISIBLE_DEVICES="$gpu"; "$fn" "$@" ) >"$logfile" 2>&1 &
  PIDS+=("$!")
  JOB_NAMES+=("$job_name")
}

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
  [[ "$bad" -eq 0 ]] || die "one or more jobs failed; inspect $LOG_DIR"
}

wait_when_full() { local limit="$1"; (( ${#PIDS[@]} < limit )) || wait_all; }

NAMES_CFG=(topk4 topk8 topk16 topk32 topk64 rank8 rank32 rank64)
RUN_NAMES=(); RUN_CONFIGS=(); RUN_SEEDS=()
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
[[ "${#RUN_NAMES[@]}" -gt 0 ]] || die "RUN_FILTER matched no runs"

{
  echo -e "run\tconfig\tseed"
  for i in "${!RUN_NAMES[@]}"; do
    echo -e "${RUN_NAMES[$i]}\t${RUN_CONFIGS[$i]}\t${RUN_SEEDS[$i]}"
  done
} > "$RUN_ROOT/run_manifest.tsv"

log "H200-02 recovery: ${#RUN_NAMES[@]} run(s); GPUs=${GPU_LIST}; stages=${STAGES}"
column -t -s $'\t' "$RUN_ROOT/run_manifest.tsv" 2>/dev/null || cat "$RUN_ROOT/run_manifest.tsv"

preflight_existing_runs() {
  local missing=0 run_name run_dir n_ckpt
  for run_name in "${RUN_NAMES[@]}"; do
    run_dir="$RUN_ROOT/$run_name"
    if [[ ! -f "$run_dir/run.json" ]]; then
      echo "ERROR: missing run.json: $run_dir/run.json" >&2
      missing=1
      continue
    fi
    shopt -s nullglob
    local ckpts=("$run_dir"/checkpoints/step-*)
    shopt -u nullglob
    n_ckpt="${#ckpts[@]}"
    if [[ "$n_ckpt" -eq 0 ]]; then
      echo "ERROR: no checkpoints found: $run_dir/checkpoints/step-*" >&2
      missing=1
    else
      log "preflight: $run_name -> $n_ckpt checkpoint(s)"
    fi
  done
  [[ "$missing" -eq 0 ]] || die "existing trained runs are incomplete"
}
preflight_existing_runs

select_one_run() {
  local run_name="$1"
  local run_dir="$RUN_ROOT/$run_name"
  local ckpt tag out_dir
  local args=()

  echo "[select] run=$run_name"
  echo "[select] run_dir=$run_dir"
  echo "[select] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

  shopt -s nullglob
  local ckpts=("$run_dir"/checkpoints/step-*)
  shopt -u nullglob
  echo "[select] found ${#ckpts[@]} checkpoint(s)"
  [[ "${#ckpts[@]}" -gt 0 ]] || { echo "ERROR: no checkpoints found under $run_dir/checkpoints" >&2; return 20; }

  for ckpt in "${ckpts[@]}"; do
    tag="$(basename "$ckpt")"
    out_dir="$run_dir/checkpoint_screen/$tag"
    if [[ -f "$out_dir/metrics.json" ]]; then
      echo "[select] skip existing metrics: $run_name / $tag"
      continue
    fi

    args=(--manifest "$DATA_DIR/validation.full.jsonl" --output "$out_dir" --model "$MODEL" --revision "$REVISION" --adapter "$ckpt" --memory-mode history --diagnostics)
    [[ "$CHECKPOINT_SELECT_MAX_CALLS" != 0 ]] && args+=(--max-calls "$CHECKPOINT_SELECT_MAX_CALLS")

    echo "[select] evaluating $run_name / $tag"
    if ! python -m csa_adapter.longform.evaluate "${args[@]}" >"$LOG_DIR/${run_name}.${tag}.screen.log" 2>&1; then
      echo "ERROR: checkpoint evaluation failed: $run_name / $tag" >&2
      tail -80 "$LOG_DIR/${run_name}.${tag}.screen.log" >&2 || true
      return 21
    fi
    [[ -f "$out_dir/metrics.json" ]] || { echo "ERROR: evaluator finished but metrics.json missing: $out_dir" >&2; return 22; }
  done

  python - "$run_dir" "$CHECKPOINT_SELECT_MAX_CALLS" <<'PY'
import json, sys
from pathlib import Path
run=Path(sys.argv[1]); max_calls=int(sys.argv[2]); vals=[]
for p in sorted((run/'checkpoint_screen').glob('step-*/metrics.json')):
    d=json.loads(p.read_text())
    vals.append((float(d['normalized_micro_wer']), int(d.get('empty_hypothesis_chunks',0)), p.parent.name))
if not vals: raise SystemExit('no checkpoint-screen metrics')
score, empty, tag=min(vals, key=lambda x:(x[0],x[1],x[2]))
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
  local run_name="$1"
  local run_dir="$RUN_ROOT/$run_name"
  local out_dir="$run_dir/eval_validation_full"
  local tag adapter
  [[ -f "$run_dir/selected_checkpoint.txt" ]] || { echo "ERROR: missing selected_checkpoint.txt for $run_name" >&2; return 30; }
  tag="$(tr -d '\r\n' < "$run_dir/selected_checkpoint.txt")"
  adapter="$run_dir/checkpoints/$tag"
  [[ -f "$adapter/adapter.safetensors" ]] || { echo "ERROR: selected adapter missing: $adapter/adapter.safetensors" >&2; return 31; }
  [[ -f "$out_dir/metrics.json" ]] && { echo "[valfull] skip existing metrics: $run_name"; return 0; }
  echo "[valfull] run=$run_name checkpoint=$tag"
  python -m csa_adapter.longform.evaluate --manifest "$DATA_DIR/validation.full.jsonl" --output "$out_dir" --model "$MODEL" --revision "$REVISION" --adapter "$adapter" --memory-mode history --diagnostics
  [[ -f "$out_dir/metrics.json" ]] || { echo "ERROR: full validation finished without metrics.json: $out_dir" >&2; return 32; }
}

eval_one_run() {
  local run_name="$1"
  local corpus="$2"
  local run_dir="$RUN_ROOT/$run_name"
  local tag adapter manifest out_dir
  [[ -f "$run_dir/selected_checkpoint.txt" ]] || { echo "ERROR: missing selected_checkpoint.txt for $run_name" >&2; return 40; }
  tag="$(tr -d '\r\n' < "$run_dir/selected_checkpoint.txt")"
  adapter="$run_dir/checkpoints/$tag"
  [[ -f "$adapter/adapter.safetensors" ]] || { echo "ERROR: selected adapter missing: $adapter/adapter.safetensors" >&2; return 41; }
  case "$corpus" in
    test) manifest="$DATA_DIR/test.full.jsonl"; out_dir="$run_dir/eval_test" ;;
    e21) manifest="$DATA_DIR/earnings21.full.jsonl"; out_dir="$run_dir/eval_e21" ;;
    *) echo "ERROR: unknown corpus: $corpus" >&2; return 42 ;;
  esac
  [[ -f "$manifest" ]] || { echo "ERROR: missing manifest: $manifest" >&2; return 43; }
  [[ -f "$out_dir/metrics.json" ]] && { echo "[eval:$corpus] skip existing metrics: $run_name"; return 0; }
  echo "[eval:$corpus] run=$run_name checkpoint=$tag"
  python -m csa_adapter.longform.evaluate --manifest "$manifest" --output "$out_dir" --model "$MODEL" --revision "$REVISION" --adapter "$adapter" --memory-mode history --diagnostics
  [[ -f "$out_dir/metrics.json" ]] || { echo "ERROR: evaluation finished without metrics.json: $out_dir" >&2; return 44; }
}

if has_stage select; then
  [[ -f "$DATA_DIR/validation.full.jsonl" ]] || die "missing full validation manifest: $DATA_DIR/validation.full.jsonl"
  for i in "${!RUN_NAMES[@]}"; do
    run_name="${RUN_NAMES[$i]}"; gpu="${GPUS_ARR[$((i % ${#GPUS_ARR[@]}))]}"
    launch_fn "select_$run_name" "$gpu" "$LOG_DIR/select_${run_name}.log" select_one_run "$run_name"
    wait_when_full "${#GPUS_ARR[@]}"
  done
  [[ "${#PIDS[@]}" -eq 0 ]] || wait_all

  for i in "${!RUN_NAMES[@]}"; do
    run_name="${RUN_NAMES[$i]}"; gpu="${GPUS_ARR[$((i % ${#GPUS_ARR[@]}))]}"
    launch_fn "valfull_$run_name" "$gpu" "$LOG_DIR/valfull_${run_name}.log" full_validation_one_run "$run_name"
    wait_when_full "${#GPUS_ARR[@]}"
  done
  [[ "${#PIDS[@]}" -eq 0 ]] || wait_all
fi

if has_stage test; then
  [[ -f "$DATA_DIR/test.full.jsonl" ]] || die "missing E22 test manifest: $DATA_DIR/test.full.jsonl"
  for i in "${!RUN_NAMES[@]}"; do
    run_name="${RUN_NAMES[$i]}"; gpu="${GPUS_ARR[$((i % ${#GPUS_ARR[@]}))]}"
    launch_fn "test_$run_name" "$gpu" "$LOG_DIR/test_${run_name}.log" eval_one_run "$run_name" test
    wait_when_full "${#GPUS_ARR[@]}"
  done
  [[ "${#PIDS[@]}" -eq 0 ]] || wait_all
fi

if has_stage external; then
  [[ -f "$DATA_DIR/earnings21.full.jsonl" ]] || die "missing Earnings-21 manifest: $DATA_DIR/earnings21.full.jsonl"
  for i in "${!RUN_NAMES[@]}"; do
    run_name="${RUN_NAMES[$i]}"; gpu="${GPUS_ARR[$((i % ${#GPUS_ARR[@]}))]}"
    launch_fn "e21_$run_name" "$gpu" "$LOG_DIR/e21_${run_name}.log" eval_one_run "$run_name" e21
    wait_when_full "${#GPUS_ARR[@]}"
  done
  [[ "${#PIDS[@]}" -eq 0 ]] || wait_all
fi

summarize() {
python - "$RUN_ROOT" <<'PY'
import csv, json, statistics, sys
from collections import defaultdict
from pathlib import Path
root=Path(sys.argv[1]); rows=list(csv.DictReader((root/'run_manifest.tsv').open(), delimiter='\t'))
def load(p): return json.loads(p.read_text()) if p.is_file() else {}
def fmt(x): return '-' if x is None else (f'{x:.8f}' if isinstance(x,float) else str(x))
def ms(vals):
    vals=[float(x) for x in vals if x is not None]
    if not vals: return None,None,0
    return statistics.mean(vals), (statistics.stdev(vals) if len(vals)>1 else 0.0), len(vals)
run_lines=[]; grouped=defaultdict(list)
for r in rows:
    rd=root/r['run']; sel=load(rd/'selected_checkpoint.json'); cfg=load(rd/'run.json')
    val=load(rd/'eval_validation_full'/'metrics.json'); test=load(rd/'eval_test'/'metrics.json'); e21=load(rd/'eval_e21'/'metrics.json')
    rec={**r,
         'checkpoint':sel.get('checkpoint'),
         'screen_wer':sel.get('screen_validation_micro_wer'),
         'trainable_parameters':cfg.get('trainable_parameters'),
         'val_wer':val.get('normalized_micro_wer'),
         'test_wer':test.get('normalized_micro_wer'),
         'e21_wer':e21.get('normalized_micro_wer'),
         'test_rtf':test.get('rtf'),
         'test_empty':test.get('empty_hypothesis_chunks'),
         'test_residual':test.get('diag_mean_residual_ratio'),
         'test_clip':test.get('diag_mean_residual_clip_fraction'),
         'test_cosine':test.get('diag_mean_adapted_cosine'),
         'test_entropy':test.get('diag_mean_retrieval_entropy')}
    run_lines.append(rec); grouped[r['config']].append(rec)
cols=['run','config','seed','checkpoint','screen_wer','trainable_parameters','val_wer','test_wer','e21_wer','test_rtf','test_empty','test_residual','test_clip','test_cosine','test_entropy']
with (root/'run_summary.tsv').open('w') as o:
    o.write('\t'.join(cols)+'\n')
    for x in run_lines: o.write('\t'.join(fmt(x.get(c)) for c in cols)+'\n')
acols=['config','n_val','val_mean','val_std','n_test','test_mean','test_std','n_e21','e21_mean','e21_std','test_rtf_mean','test_empty_mean','test_residual_mean','test_clip_mean','test_cosine_mean','test_entropy_mean']
with (root/'aggregate_summary.tsv').open('w') as o:
    o.write('\t'.join(acols)+'\n')
    for config in dict.fromkeys(r['config'] for r in rows):
        xs=grouped[config]
        vm,vs,nv=ms([x['val_wer'] for x in xs]); tm,ts,nt=ms([x['test_wer'] for x in xs]); em,es,ne=ms([x['e21_wer'] for x in xs])
        rm,_,_=ms([x['test_rtf'] for x in xs]); xm,_,_=ms([x['test_empty'] for x in xs]); drm,_,_=ms([x['test_residual'] for x in xs]); cm,_,_=ms([x['test_clip'] for x in xs]); cosm,_,_=ms([x['test_cosine'] for x in xs]); entm,_,_=ms([x['test_entropy'] for x in xs])
        vals=[config,nv,vm,vs,nt,tm,ts,ne,em,es,rm,xm,drm,cm,cosm,entm]
        o.write('\t'.join(fmt(v) for v in vals)+'\n')
status={
  'runs_total':len(rows),
  'selected':sum((root/r['run']/'selected_checkpoint.json').is_file() for r in rows),
  'full_validation_metrics':sum((root/r['run']/'eval_validation_full'/'metrics.json').is_file() for r in rows),
  'e22_test_metrics':sum((root/r['run']/'eval_test'/'metrics.json').is_file() for r in rows),
  'e21_test_metrics':sum((root/r['run']/'eval_e21'/'metrics.json').is_file() for r in rows),
  'checkpoint_screen_metrics':sum(1 for r in rows for _ in (root/r['run']/'checkpoint_screen').glob('step-*/metrics.json')),
}
(root/'evaluation_status.json').write_text(json.dumps(status, indent=2)+'\n')
print('=== evaluation_status.json ==='); print(json.dumps(status, indent=2))
print('=== aggregate_summary.tsv ==='); print((root/'aggregate_summary.tsv').read_text(), end='')
PY
}

if has_stage summary; then summarize | tee "$RUN_ROOT/summary.log"; fi
log "H200-02 recovery requested stages complete: $RUN_ROOT"
log "per-seed results : $RUN_ROOT/run_summary.tsv"
log "aggregate results: $RUN_ROOT/aggregate_summary.tsv"
log "status           : $RUN_ROOT/evaluation_status.json"
