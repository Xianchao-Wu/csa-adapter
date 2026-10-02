#!/usr/bin/env bash
# CSA-Adapter v0.2.1 / DGX-H200-01
# Memory horizon, compression-rate, and gate ablations on all 8 H200 GPUs.
# Primary model/checkpoint selection uses text-history OFF and long-form WER-N.
set -Eeuo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/scripts/_repo_env.sh"
python scripts/preflight_v021.py

MODEL="${MODEL:-openai/whisper-large-v3}"
REVISION="${REVISION:-main}"
E22_REVISION="${E22_REVISION:-0a034f9ed86d33a3859d9025d3e621cf243773ab}"
E21_REVISION="${E21_REVISION:-5956ebd0b2ba89bbff20e6a001415b23b27fd4cb}"
DATA_DIR="${DATA_DIR:-data/earnings22_622}"
CACHE_DIR="${CACHE_DIR:-cache/earnings22_large_v3}"
RUN_ROOT="${RUN_ROOT:-runs/h200_01_memory_compression_v4}"
LOG_DIR="${LOG_DIR:-$RUN_ROOT/logs}"
GPU_LIST="${GPUS:-0,1,2,3,4,5,6,7}"
CACHE_GPU_LIST="${CACHE_GPUS:-0,1,2,3,4,5,6,7,0,1,2,3,4,5,6,7,0,1,2,3,4,5,6,7,0,1,2,3,4,5,6,7}"
TRAIN_SEEDS="${TRAIN_SEEDS:-42,43,44}"
SPLIT_SEED="${SPLIT_SEED:-42}"
MAX_STEPS="${MAX_STEPS:-1500}"
EPOCHS="${EPOCHS:-99}"
GRAD_ACCUM="${GRAD_ACCUM:-8}"
LR="${LR:-1e-4}"
VALID_EXAMPLES="${VALID_EXAMPLES:-256}"
VALIDATE_EVERY="${VALIDATE_EVERY:-250}"
FEATURE_LRU="${FEATURE_LRU:-256}"
DIAGNOSTICS_EVERY="${DIAGNOSTICS_EVERY:-10}"
CHECKPOINT_SELECT_MAX_CALLS="${CHECKPOINT_SELECT_MAX_CALLS:-8}"
TRAIN_JOBS_PER_GPU="${TRAIN_JOBS_PER_GPU:-3}"
STAGES="${STAGES:-setup,prepare,cache,train,select,test,summary}"
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
[[ "${#SEEDS[@]}" -ge 1 ]] || die "TRAIN_SEEDS must contain >=1 seed"
[[ "$TRAIN_JOBS_PER_GPU" =~ ^[1-9][0-9]*$ ]] || die "TRAIN_JOBS_PER_GPU must be >=1"

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
wait_when_full() { local limit="$1"; (( ${#PIDS[@]} < limit )) || wait_all; }

# Shared default anchor enables clean one-factor-at-a-time curves.
NAMES_CFG=(mem512 mem2048 default mem8192 mem16384 comp_c4 comp_c16 no_gate)
RUN_NAMES=(); RUN_CONFIGS=(); RUN_SEEDS=()
for config in "${NAMES_CFG[@]}"; do
  for seed in "${SEEDS[@]}"; do
    RUN_NAMES+=("${config}_s${seed}"); RUN_CONFIGS+=("$config"); RUN_SEEDS+=("$seed")
  done
done

config_args() {
  case "$1" in
    mem512)   printf '%s\n' --compression-rate 8  --max-memory 512   --history-segments 64  ;;
    mem2048)  printf '%s\n' --compression-rate 8  --max-memory 2048  --history-segments 64  ;;
    default)  printf '%s\n' --compression-rate 8  --max-memory 4096  --history-segments 64  ;;
    mem8192)  printf '%s\n' --compression-rate 8  --max-memory 8192  --history-segments 64  ;;
    mem16384) printf '%s\n' --compression-rate 8  --max-memory 16384 --history-segments 128 ;;
    # Match temporal FIFO horizon to C=8/L=4096 while changing compression rate.
    comp_c4)  printf '%s\n' --compression-rate 4  --max-memory 8192  --history-segments 64  ;;
    comp_c16) printf '%s\n' --compression-rate 16 --max-memory 2048  --history-segments 64  ;;
    no_gate)  printf '%s\n' --compression-rate 8  --max-memory 4096  --history-segments 64 --gate none ;;
    *) die "unknown config: $1" ;;
  esac
}

{
  echo -e "run\tconfig\tseed"
  for i in "${!RUN_NAMES[@]}"; do echo -e "${RUN_NAMES[$i]}\t${RUN_CONFIGS[$i]}\t${RUN_SEEDS[$i]}"; done
} > "$RUN_ROOT/run_manifest.tsv"
log "H200-01 v4: ${#RUN_NAMES[@]} runs; 8 GPUs; max $TRAIN_JOBS_PER_GPU training processes/GPU"
column -t -s $'\t' "$RUN_ROOT/run_manifest.tsv" 2>/dev/null || cat "$RUN_ROOT/run_manifest.tsv"
[[ "$PLAN_ONLY" == 1 ]] && { log "PLAN_ONLY=1; exiting"; exit 0; }

if has_stage setup; then bash scripts/00_setup.sh 2>&1 | tee "$LOG_DIR/00_setup.log"; fi

if has_stage prepare; then
  if [[ -f "$DATA_DIR/split_calls.json" && -f "$DATA_DIR/preparation_report.json" ]]; then
    log "reuse prepared data: $DATA_DIR"
  else
    DATA_DIR="$DATA_DIR" SEED="$SPLIT_SEED" bash scripts/01_prepare_earnings22.sh \
      --revision "$E22_REVISION" --external-revision "$E21_REVISION" 2>&1 | tee "$LOG_DIR/01_prepare.log"
  fi
fi

if has_stage cache; then
  [[ -f "$DATA_DIR/train.jsonl" && -f "$DATA_DIR/validation.jsonl" ]] || die "missing prepared manifests"
  if cache_complete "$CACHE_DIR/train" && cache_complete "$CACHE_DIR/validation"; then
    log "reuse complete feature cache: $CACHE_DIR"
  elif cache_has_metadata "$CACHE_DIR/train" || cache_has_metadata "$CACHE_DIR/validation"; then
    die "incomplete/inconsistent cache under $CACHE_DIR; use a fresh CACHE_DIR"
  else
    DATA_DIR="$DATA_DIR" CACHE_DIR="$CACHE_DIR" MODEL="$MODEL" REVISION="$REVISION" \
      GPUS="$CACHE_GPU_LIST" bash scripts/02_cache_features.sh 2>&1 | tee "$LOG_DIR/02_cache.log"
  fi
fi

if has_stage train; then
  [[ -d "$CACHE_DIR/train" && -d "$CACHE_DIR/validation" ]] || die "missing feature caches"
  max_parallel=$(( ${#GPUS_ARR[@]} * TRAIN_JOBS_PER_GPU ))
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
      --seed "$seed" --epochs "$EPOCHS" --max-steps "$MAX_STEPS" --grad-accum "$GRAD_ACCUM" --lr "$LR" \
      --compressor event --top-k 16 --rank 16 --warmup-steps 0 \
      --alpha-init 0.01 --alpha-max 0.10 --gate-bias-init -2.0 --residual-ratio-cap 0.25 \
      --valid-examples "$VALID_EXAMPLES" --validate-every "$VALIDATE_EVERY" \
      --diagnostics-every "$DIAGNOSTICS_EVERY" --feature-lru "$FEATURE_LRU" --save-validation-checkpoints \
      "${extra[@]}"
    wait_when_full "$max_parallel"
  done
  [[ "${#PIDS[@]}" -eq 0 ]] || wait_all
fi

select_one_run() {
  local name="$1" run="$RUN_ROOT/$name" ckpt tag out
  shopt -s nullglob
  local ckpts=("$run"/checkpoints/step-*)
  [[ "${#ckpts[@]}" -gt 0 ]] || return 20
  for ckpt in "${ckpts[@]}"; do
    tag="$(basename "$ckpt")"; out="$run/checkpoint_screen/$tag"
    [[ -f "$out/metrics.json" ]] && continue
    local args=(--manifest "$DATA_DIR/validation.full.jsonl" --output "$out" --model "$MODEL" --revision "$REVISION" \
                --adapter "$ckpt" --memory-mode history --diagnostics)
    [[ "$CHECKPOINT_SELECT_MAX_CALLS" != 0 ]] && args+=(--max-calls "$CHECKPOINT_SELECT_MAX_CALLS")
    python -m csa_adapter.longform.evaluate "${args[@]}" >"$LOG_DIR/${name}.${tag}.screen.log" 2>&1
  done
  python - "$run" "$CHECKPOINT_SELECT_MAX_CALLS" <<'PY'
import json, sys
from pathlib import Path
run=Path(sys.argv[1]); max_calls=int(sys.argv[2]); vals=[]
for p in sorted((run/'checkpoint_screen').glob('step-*/metrics.json')):
    d=json.loads(p.read_text()); vals.append((d['normalized_micro_wer'], p.parent.name))
if not vals: raise SystemExit('no checkpoint-screen metrics')
score, tag=min(vals)
(run/'selected_checkpoint.txt').write_text(tag+'\n')
(run/'selected_checkpoint.json').write_text(json.dumps({
    'checkpoint':tag, 'screen_validation_micro_wer':score,
    'screen_max_calls':max_calls, 'selection_text_history':False
}, indent=2)+'\n')
print(run.name, tag, score)
PY
}

full_validation_one_run() {
  local name="$1" run="$RUN_ROOT/$name" out="$run/eval_validation_full" tag adapter
  tag="$(tr -d '\r\n' < "$run/selected_checkpoint.txt")"; adapter="$run/checkpoints/$tag"
  [[ -f "$out/metrics.json" ]] && return 0
  python -m csa_adapter.longform.evaluate --manifest "$DATA_DIR/validation.full.jsonl" --output "$out" \
    --model "$MODEL" --revision "$REVISION" --adapter "$adapter" --memory-mode history --diagnostics
}

eval_one_run() {
  local name="$1" corpus="$2" run="$RUN_ROOT/$name" tag adapter manifest out
  tag="$(tr -d '\r\n' < "$run/selected_checkpoint.txt")"; adapter="$run/checkpoints/$tag"
  case "$corpus" in
    test) manifest="$DATA_DIR/test.full.jsonl"; out="$run/eval_test" ;;
    e21) manifest="$DATA_DIR/earnings21.full.jsonl"; out="$run/eval_e21" ;;
    *) return 31 ;;
  esac
  [[ -f "$out/metrics.json" ]] && return 0
  python -m csa_adapter.longform.evaluate --manifest "$manifest" --output "$out" \
    --model "$MODEL" --revision "$REVISION" --adapter "$adapter" --memory-mode history --diagnostics
}

if has_stage select; then
  [[ -f "$DATA_DIR/validation.full.jsonl" ]] || die "missing full validation manifest"
  # Screen checkpoints: one process/GPU, 8 runs per wave.
  for i in "${!RUN_NAMES[@]}"; do
    name="${RUN_NAMES[$i]}"; gpu="${GPUS_ARR[$((i % ${#GPUS_ARR[@]}))]}"
    launch_fn "select_$name" "$gpu" "$LOG_DIR/select_${name}.log" select_one_run "$name"
    wait_when_full "${#GPUS_ARR[@]}"
  done
  [[ "${#PIDS[@]}" -eq 0 ]] || wait_all
  # Full validation only for the screened checkpoint, again one process/GPU.
  for i in "${!RUN_NAMES[@]}"; do
    name="${RUN_NAMES[$i]}"; gpu="${GPUS_ARR[$((i % ${#GPUS_ARR[@]}))]}"
    launch_fn "valfull_$name" "$gpu" "$LOG_DIR/valfull_${name}.log" full_validation_one_run "$name"
    wait_when_full "${#GPUS_ARR[@]}"
  done
  [[ "${#PIDS[@]}" -eq 0 ]] || wait_all
fi

if has_stage test; then
  [[ -f "$DATA_DIR/test.full.jsonl" ]] || die "missing E22 test manifest"
  for i in "${!RUN_NAMES[@]}"; do
    name="${RUN_NAMES[$i]}"; gpu="${GPUS_ARR[$((i % ${#GPUS_ARR[@]}))]}"
    launch_fn "test_$name" "$gpu" "$LOG_DIR/test_${name}.log" eval_one_run "$name" test
    wait_when_full "${#GPUS_ARR[@]}"
  done
  [[ "${#PIDS[@]}" -eq 0 ]] || wait_all
fi

# Optional: STAGES=external,summary evaluates all selected per-seed checkpoints on E21.
if has_stage external; then
  [[ -f "$DATA_DIR/earnings21.full.jsonl" ]] || die "missing Earnings-21 manifest"
  for i in "${!RUN_NAMES[@]}"; do
    name="${RUN_NAMES[$i]}"; gpu="${GPUS_ARR[$((i % ${#GPUS_ARR[@]}))]}"
    launch_fn "e21_$name" "$gpu" "$LOG_DIR/e21_${name}.log" eval_one_run "$name" e21
    wait_when_full "${#GPUS_ARR[@]}"
  done
  [[ "${#PIDS[@]}" -eq 0 ]] || wait_all
fi

summarize() {
python - "$RUN_ROOT" <<'PY'
import csv, json, statistics, sys
from collections import defaultdict
from pathlib import Path
root=Path(sys.argv[1])
rows=list(csv.DictReader((root/'run_manifest.tsv').open(), delimiter='\t'))

def load(path):
    return json.loads(path.read_text()) if path.is_file() else {}
def f(x):
    return '-' if x is None else (f'{x:.8f}' if isinstance(x,float) else str(x))

run_lines=[]; grouped=defaultdict(list)
for r in rows:
    rd=root/r['run']; sel=load(rd/'selected_checkpoint.json'); cfg=load(rd/'run.json')
    val=load(rd/'eval_validation_full'/'metrics.json'); test=load(rd/'eval_test'/'metrics.json'); e21=load(rd/'eval_e21'/'metrics.json')
    rec={
      **r, 'checkpoint':sel.get('checkpoint'), 'trainable_parameters':cfg.get('trainable_parameters'),
      'val_wer':val.get('normalized_micro_wer'), 'test_wer':test.get('normalized_micro_wer'),
      'e21_wer':e21.get('normalized_micro_wer'), 'test_rtf':test.get('rtf'),
      'test_empty':test.get('empty_hypothesis_chunks'),
      'test_residual':test.get('diag_mean_residual_ratio'),
      'test_clip':test.get('diag_mean_residual_clip_fraction'),
      'test_cosine':test.get('diag_mean_adapted_cosine')}
    run_lines.append(rec); grouped[r['config']].append(rec)
cols=['run','config','seed','checkpoint','trainable_parameters','val_wer','test_wer','e21_wer','test_rtf','test_empty','test_residual','test_clip','test_cosine']
with (root/'run_summary.tsv').open('w') as o:
    o.write('\t'.join(cols)+'\n')
    for x in run_lines: o.write('\t'.join(f(x.get(c)) for c in cols)+'\n')

def ms(vals):
    vals=[float(x) for x in vals if x is not None]
    if not vals: return (None,None,0)
    return (statistics.mean(vals), statistics.stdev(vals) if len(vals)>1 else 0.0, len(vals))
acols=['config','n','val_mean','val_std','test_mean','test_std','e21_mean','e21_std','test_rtf_mean','test_empty_mean']
with (root/'aggregate_summary.tsv').open('w') as o:
    o.write('\t'.join(acols)+'\n')
    for config in dict.fromkeys(r['config'] for r in rows):
        xs=grouped[config]; vm,vs,n=ms([x['val_wer'] for x in xs]); tm,ts,_=ms([x['test_wer'] for x in xs]); em,es,_=ms([x['e21_wer'] for x in xs]); rm,_,_=ms([x['test_rtf'] for x in xs]); xm,_,_=ms([x['test_empty'] for x in xs])
        vals=[config,n,vm,vs,tm,ts,em,es,rm,xm]
        o.write('\t'.join(f(v) for v in vals)+'\n')
print((root/'aggregate_summary.tsv').read_text())
PY
}

if has_stage summary; then summarize | tee "$RUN_ROOT/summary.log"; fi
log "H200-01 v4 requested stages complete: $RUN_ROOT"
