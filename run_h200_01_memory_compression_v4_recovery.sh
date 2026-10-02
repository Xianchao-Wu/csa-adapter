#!/usr/bin/env bash
# CSA-Adapter v0.2.1 / DGX-H200-01
# Evaluation-only recovery for run_h200_01_memory_compression_v4.sh.
#
# IMPORTANT:
#   - NEVER trains or rewrites trained adapters/checkpoints.
#   - Reuses runs/h200_01_memory_compression_v4/*/checkpoints/step-*.
#   - Fixes the hidden Bash dynamic-scope/local-assignment bug from v4.
#   - Resumable: existing metrics.json files are skipped automatically.
#
# Default pipeline:
#   checkpoint screening -> select best checkpoint -> full validation
#   -> Earnings-22 test -> Earnings-21 external test -> summaries
#
# Typical usage:
#   bash run_h200_01_memory_compression_v4_recovery.sh
#
# Resume only selected stages:
#   STAGES=select bash run_h200_01_memory_compression_v4_recovery.sh
#   STAGES=test,external,summary bash run_h200_01_memory_compression_v4_recovery.sh
#
# Smoke test one run only:
#   RUN_FILTER=mem512_s42 STAGES=select bash run_h200_01_memory_compression_v4_recovery.sh
#
# Use only a subset of GPUs:
#   GPUS=0,1,2,3 bash run_h200_01_memory_compression_v4_recovery.sh

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

source "$SCRIPT_DIR/scripts/_repo_env.sh"
python "$SCRIPT_DIR/scripts/preflight_v021.py"

MODEL="${MODEL:-openai/whisper-large-v3}"
REVISION="${REVISION:-main}"
DATA_DIR="${DATA_DIR:-data/earnings22_622}"
RUN_ROOT="${RUN_ROOT:-runs/h200_01_memory_compression_v4}"
LOG_DIR="${LOG_DIR:-$RUN_ROOT/logs}"
GPU_LIST="${GPUS:-0,1,2,3,4,5,6,7}"
TRAIN_SEEDS="${TRAIN_SEEDS:-42,43,44}"

# Number of complete validation calls used for checkpoint screening.
# 0 means use the complete validation.full.jsonl for every checkpoint.
CHECKPOINT_SELECT_MAX_CALLS="${CHECKPOINT_SELECT_MAX_CALLS:-8}"

# Recovery defaults to every post-training stage.
STAGES="${STAGES:-select,test,external,summary}"

# Optional exact run name, e.g. mem512_s42. Empty = all runs.
RUN_FILTER="${RUN_FILTER:-}"

# Safety: refuse to share GPUs with existing compute jobs unless explicitly disabled.
REQUIRE_IDLE_GPUS="${REQUIRE_IDLE_GPUS:-1}"

# Keep PyTorch allocator behavior consistent with v4.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

mkdir -p "$RUN_ROOT" "$LOG_DIR"

has_stage() { [[ ",$STAGES," == *",$1,"* ]]; }
log() { printf '[%(%F %T)T] %s\n' -1 "$*"; }
die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

# -----------------------------------------------------------------------------
# GPU / run setup
# -----------------------------------------------------------------------------
IFS=',' read -r -a GPUS_ARR <<< "$GPU_LIST"
IFS=',' read -r -a SEEDS <<< "$TRAIN_SEEDS"

[[ "${#GPUS_ARR[@]}" -ge 1 ]] || die "GPUS must contain at least one GPU ID"
[[ "${#SEEDS[@]}" -ge 1 ]] || die "TRAIN_SEEDS must contain at least one seed"
[[ "$CHECKPOINT_SELECT_MAX_CALLS" =~ ^[0-9]+$ ]] || \
  die "CHECKPOINT_SELECT_MAX_CALLS must be a non-negative integer"

if [[ "$REQUIRE_IDLE_GPUS" == 1 ]]; then
  for gpu in "${GPUS_ARR[@]}"; do
    used="$(nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader 2>/dev/null \
      | sed '/^[[:space:]]*$/d' | wc -l)"
    [[ "$used" -eq 0 ]] || \
      die "GPU $gpu already has a compute process; set REQUIRE_IDLE_GPUS=0 only if intentional"
  done
fi

# Same experiment grid as v4.
NAMES_CFG=(mem512 mem2048 default mem8192 mem16384 comp_c4 comp_c16 no_gate)
RUN_NAMES=()
RUN_CONFIGS=()
RUN_SEEDS=()

for config in "${NAMES_CFG[@]}"; do
  for seed in "${SEEDS[@]}"; do
    candidate="${config}_s${seed}"
    if [[ -n "$RUN_FILTER" && "$candidate" != "$RUN_FILTER" ]]; then
      continue
    fi
    RUN_NAMES+=("$candidate")
    RUN_CONFIGS+=("$config")
    RUN_SEEDS+=("$seed")
  done
done

[[ "${#RUN_NAMES[@]}" -gt 0 ]] || die "RUN_FILTER='$RUN_FILTER' matched no runs"

# Preserve a full manifest for normal all-run recovery.  For a smoke RUN_FILTER,
# do not overwrite the original v4 manifest used by the final all-run summary.
if [[ -z "$RUN_FILTER" ]]; then
  {
    echo -e "run\tconfig\tseed"
    for i in "${!RUN_NAMES[@]}"; do
      echo -e "${RUN_NAMES[$i]}\t${RUN_CONFIGS[$i]}\t${RUN_SEEDS[$i]}"
    done
  } > "$RUN_ROOT/run_manifest.tsv"
fi

log "evaluation-only recovery: ${#RUN_NAMES[@]} run(s), ${#GPUS_ARR[@]} GPU(s)"
log "STAGES=$STAGES"
log "RUN_ROOT=$RUN_ROOT"
log "DATA_DIR=$DATA_DIR"
log "CHECKPOINT_SELECT_MAX_CALLS=$CHECKPOINT_SELECT_MAX_CALLS"
[[ -n "$RUN_FILTER" ]] && log "RUN_FILTER=$RUN_FILTER"

# -----------------------------------------------------------------------------
# Preflight checks for existing trained runs.  NO TRAINING occurs in this script.
# -----------------------------------------------------------------------------
check_existing_run() {
  local run_name="$1"
  local run_dir
  local -a ckpts

  run_dir="$RUN_ROOT/$run_name"
  [[ -d "$run_dir" ]] || die "missing trained run directory: $run_dir"
  [[ -f "$run_dir/run.json" ]] || die "missing run.json: $run_dir/run.json"

  shopt -s nullglob
  ckpts=("$run_dir"/checkpoints/step-*)
  shopt -u nullglob

  [[ "${#ckpts[@]}" -gt 0 ]] || die "no checkpoints found: $run_dir/checkpoints/step-*"

  local ckpt
  for ckpt in "${ckpts[@]}"; do
    [[ -f "$ckpt/adapter.safetensors" ]] || die "missing adapter.safetensors: $ckpt"
    [[ -f "$ckpt/adapter_config.json" ]] || die "missing adapter_config.json: $ckpt"
  done

  log "preflight OK: $run_name (${#ckpts[@]} checkpoints)"
}

for run_name in "${RUN_NAMES[@]}"; do
  check_existing_run "$run_name"
done

if has_stage select; then
  [[ -f "$DATA_DIR/validation.full.jsonl" ]] || \
    die "missing full validation manifest: $DATA_DIR/validation.full.jsonl"
fi
if has_stage test; then
  [[ -f "$DATA_DIR/test.full.jsonl" ]] || \
    die "missing E22 test manifest: $DATA_DIR/test.full.jsonl"
fi
if has_stage external; then
  [[ -f "$DATA_DIR/earnings21.full.jsonl" ]] || \
    die "missing Earnings-21 manifest: $DATA_DIR/earnings21.full.jsonl"
fi

# -----------------------------------------------------------------------------
# Background-job orchestration.
#
# IMPORTANT BUG FIX:
# v4 used local variable `name` in launch_fn() and in called functions that also
# initialized `run="$RUN_ROOT/$name"` on the SAME `local` statement.  Bash uses
# dynamic scoping for local variables; therefore the caller's `name` could leak
# into RHS expansion (e.g. select_mem512_s42 instead of mem512_s42).
#
# This recovery script:
#   1) uses job_name in launch_fn(), and
#   2) initializes run_name/run_dir/output_dir on separate statements everywhere.
# -----------------------------------------------------------------------------
declare -a PIDS=()
declare -a JOB_NAMES=()

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
      log "FAILED: ${JOB_NAMES[$i]} (see $LOG_DIR)"
      bad=1
    fi
  done

  PIDS=()
  JOB_NAMES=()
  [[ "$bad" -eq 0 ]] || die "one or more evaluation jobs failed; inspect $LOG_DIR"
}

wait_when_full() {
  local limit="$1"
  (( ${#PIDS[@]} < limit )) || wait_all
}

# -----------------------------------------------------------------------------
# Stage 1: checkpoint screening + best-checkpoint selection
# -----------------------------------------------------------------------------
select_one_run() {
  local run_name="$1"
  local run_dir
  local ckpt tag output_dir screen_log
  local -a ckpts args

  run_dir="$RUN_ROOT/$run_name"

  echo "[select] run=$run_name"
  echo "[select] run_dir=$run_dir"
  echo "[select] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

  shopt -s nullglob
  ckpts=("$run_dir"/checkpoints/step-*)
  shopt -u nullglob

  echo "[select] found ${#ckpts[@]} checkpoint(s)"
  [[ "${#ckpts[@]}" -gt 0 ]] || {
    echo "ERROR: no checkpoints under $run_dir/checkpoints" >&2
    return 20
  }

  for ckpt in "${ckpts[@]}"; do
    tag="$(basename "$ckpt")"
    output_dir="$run_dir/checkpoint_screen/$tag"
    screen_log="$LOG_DIR/${run_name}.${tag}.screen.log"

    if [[ -f "$output_dir/metrics.json" ]]; then
      echo "[select] skip existing metrics: $run_name / $tag"
      continue
    fi

    args=(
      --manifest "$DATA_DIR/validation.full.jsonl"
      --output "$output_dir"
      --model "$MODEL"
      --revision "$REVISION"
      --adapter "$ckpt"
      --memory-mode history
      --diagnostics
    )

    if [[ "$CHECKPOINT_SELECT_MAX_CALLS" != 0 ]]; then
      args+=(--max-calls "$CHECKPOINT_SELECT_MAX_CALLS")
    fi

    echo "[select] evaluating $run_name / $tag"
    echo "[select] evaluator log: $screen_log"

    if ! python -m csa_adapter.longform.evaluate "${args[@]}" >"$screen_log" 2>&1; then
      echo "ERROR: checkpoint screening failed: $run_name / $tag" >&2
      echo "----- tail: $screen_log -----" >&2
      tail -100 "$screen_log" >&2 || true
      echo "--------------------------------" >&2
      return 21
    fi

    if [[ ! -f "$output_dir/metrics.json" ]]; then
      echo "ERROR: evaluator returned success but metrics.json is missing: $output_dir/metrics.json" >&2
      echo "----- tail: $screen_log -----" >&2
      tail -100 "$screen_log" >&2 || true
      echo "--------------------------------" >&2
      return 22
    fi
  done

  # Select minimum normalized_micro_wer from all successfully screened checkpoints.
  python - "$run_dir" "$CHECKPOINT_SELECT_MAX_CALLS" <<'PY'
import json
import math
import sys
from pathlib import Path

run = Path(sys.argv[1])
max_calls = int(sys.argv[2])
vals = []

for p in sorted((run / "checkpoint_screen").glob("step-*/metrics.json")):
    try:
        d = json.loads(p.read_text())
        score = float(d["normalized_micro_wer"])
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
        raise SystemExit(f"invalid checkpoint-screen metrics: {p}: {exc}")
    if not math.isfinite(score):
        raise SystemExit(f"non-finite normalized_micro_wer in {p}: {score}")
    vals.append((score, p.parent.name))

if not vals:
    raise SystemExit("no checkpoint-screen metrics")

score, tag = min(vals, key=lambda x: (x[0], x[1]))

(run / "selected_checkpoint.txt").write_text(tag + "\n")
(run / "selected_checkpoint.json").write_text(
    json.dumps(
        {
            "checkpoint": tag,
            "screen_validation_micro_wer": score,
            "screen_max_calls": max_calls,
            "selection_text_history": False,
            "num_screened_checkpoints": len(vals),
        },
        indent=2,
    )
    + "\n"
)

print(f"[select] selected {run.name}: {tag}, WER-N={score:.8f}, screened={len(vals)}")
PY
}

# -----------------------------------------------------------------------------
# Stage 1b: full validation for the selected checkpoint
# -----------------------------------------------------------------------------
full_validation_one_run() {
  local run_name="$1"
  local run_dir
  local output_dir
  local tag adapter

  run_dir="$RUN_ROOT/$run_name"
  output_dir="$run_dir/eval_validation_full"

  [[ -f "$run_dir/selected_checkpoint.txt" ]] || {
    echo "ERROR: selected checkpoint missing for $run_name" >&2
    return 23
  }

  tag="$(tr -d '\r\n' < "$run_dir/selected_checkpoint.txt")"
  adapter="$run_dir/checkpoints/$tag"

  [[ -f "$adapter/adapter.safetensors" ]] || {
    echo "ERROR: selected adapter missing: $adapter/adapter.safetensors" >&2
    return 24
  }

  if [[ -f "$output_dir/metrics.json" ]]; then
    echo "[valfull] skip existing metrics: $run_name"
    return 0
  fi

  echo "[valfull] run=$run_name checkpoint=$tag gpu=${CUDA_VISIBLE_DEVICES:-unset}"

  python -m csa_adapter.longform.evaluate \
    --manifest "$DATA_DIR/validation.full.jsonl" \
    --output "$output_dir" \
    --model "$MODEL" \
    --revision "$REVISION" \
    --adapter "$adapter" \
    --memory-mode history \
    --diagnostics

  [[ -f "$output_dir/metrics.json" ]] || {
    echo "ERROR: full validation finished without metrics.json: $output_dir" >&2
    return 25
  }
}

# -----------------------------------------------------------------------------
# Stage 2/3: Earnings-22 test and Earnings-21 external evaluation
# -----------------------------------------------------------------------------
eval_one_run() {
  local run_name="$1"
  local corpus="$2"
  local run_dir
  local tag adapter manifest output_dir

  run_dir="$RUN_ROOT/$run_name"

  [[ -f "$run_dir/selected_checkpoint.txt" ]] || {
    echo "ERROR: selected checkpoint missing for $run_name; run STAGES=select first" >&2
    return 30
  }

  tag="$(tr -d '\r\n' < "$run_dir/selected_checkpoint.txt")"
  adapter="$run_dir/checkpoints/$tag"

  [[ -f "$adapter/adapter.safetensors" ]] || {
    echo "ERROR: selected adapter missing: $adapter/adapter.safetensors" >&2
    return 32
  }

  case "$corpus" in
    test)
      manifest="$DATA_DIR/test.full.jsonl"
      output_dir="$run_dir/eval_test"
      ;;
    e21)
      manifest="$DATA_DIR/earnings21.full.jsonl"
      output_dir="$run_dir/eval_e21"
      ;;
    *)
      echo "ERROR: unknown corpus: $corpus" >&2
      return 31
      ;;
  esac

  [[ -f "$manifest" ]] || {
    echo "ERROR: manifest missing: $manifest" >&2
    return 33
  }

  if [[ -f "$output_dir/metrics.json" ]]; then
    echo "[$corpus] skip existing metrics: $run_name"
    return 0
  fi

  echo "[$corpus] run=$run_name checkpoint=$tag gpu=${CUDA_VISIBLE_DEVICES:-unset}"

  python -m csa_adapter.longform.evaluate \
    --manifest "$manifest" \
    --output "$output_dir" \
    --model "$MODEL" \
    --revision "$REVISION" \
    --adapter "$adapter" \
    --memory-mode history \
    --diagnostics

  [[ -f "$output_dir/metrics.json" ]] || {
    echo "ERROR: evaluation finished without metrics.json: $output_dir" >&2
    return 34
  }
}

# -----------------------------------------------------------------------------
# Execute requested stages
# -----------------------------------------------------------------------------
if has_stage select; then
  log "stage select: checkpoint screening"

  for i in "${!RUN_NAMES[@]}"; do
    run_name="${RUN_NAMES[$i]}"
    gpu="${GPUS_ARR[$((i % ${#GPUS_ARR[@]}))]}"

    launch_fn \
      "select_$run_name" \
      "$gpu" \
      "$LOG_DIR/select_${run_name}.log" \
      select_one_run "$run_name"

    wait_when_full "${#GPUS_ARR[@]}"
  done
  [[ "${#PIDS[@]}" -eq 0 ]] || wait_all

  log "stage select: full validation of selected checkpoints"

  for i in "${!RUN_NAMES[@]}"; do
    run_name="${RUN_NAMES[$i]}"
    gpu="${GPUS_ARR[$((i % ${#GPUS_ARR[@]}))]}"

    launch_fn \
      "valfull_$run_name" \
      "$gpu" \
      "$LOG_DIR/valfull_${run_name}.log" \
      full_validation_one_run "$run_name"

    wait_when_full "${#GPUS_ARR[@]}"
  done
  [[ "${#PIDS[@]}" -eq 0 ]] || wait_all
fi

if has_stage test; then
  log "stage test: Earnings-22 full test"

  for i in "${!RUN_NAMES[@]}"; do
    run_name="${RUN_NAMES[$i]}"
    gpu="${GPUS_ARR[$((i % ${#GPUS_ARR[@]}))]}"

    launch_fn \
      "test_$run_name" \
      "$gpu" \
      "$LOG_DIR/test_${run_name}.log" \
      eval_one_run "$run_name" test

    wait_when_full "${#GPUS_ARR[@]}"
  done
  [[ "${#PIDS[@]}" -eq 0 ]] || wait_all
fi

if has_stage external; then
  log "stage external: Earnings-21 full external test"

  for i in "${!RUN_NAMES[@]}"; do
    run_name="${RUN_NAMES[$i]}"
    gpu="${GPUS_ARR[$((i % ${#GPUS_ARR[@]}))]}"

    launch_fn \
      "e21_$run_name" \
      "$gpu" \
      "$LOG_DIR/e21_${run_name}.log" \
      eval_one_run "$run_name" e21

    wait_when_full "${#GPUS_ARR[@]}"
  done
  [[ "${#PIDS[@]}" -eq 0 ]] || wait_all
fi

# -----------------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------------
summarize() {
  python - "$RUN_ROOT" <<'PY'
import csv
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

root = Path(sys.argv[1])
manifest = root / "run_manifest.tsv"
if not manifest.is_file():
    raise SystemExit(f"missing run manifest: {manifest}")

rows = list(csv.DictReader(manifest.open(), delimiter="\t"))

def load(path: Path):
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}

def fmt(x):
    if x is None:
        return "-"
    if isinstance(x, float):
        return f"{x:.8f}"
    return str(x)

run_lines = []
grouped = defaultdict(list)

for r in rows:
    rd = root / r["run"]
    sel = load(rd / "selected_checkpoint.json")
    cfg = load(rd / "run.json")
    val = load(rd / "eval_validation_full" / "metrics.json")
    test = load(rd / "eval_test" / "metrics.json")
    e21 = load(rd / "eval_e21" / "metrics.json")

    rec = {
        **r,
        "checkpoint": sel.get("checkpoint"),
        "screen_val_wer": sel.get("screen_validation_micro_wer"),
        "screen_n": sel.get("num_screened_checkpoints"),
        "trainable_parameters": cfg.get("trainable_parameters"),
        "val_wer": val.get("normalized_micro_wer"),
        "test_wer": test.get("normalized_micro_wer"),
        "e21_wer": e21.get("normalized_micro_wer"),
        "test_rtf": test.get("rtf"),
        "test_empty": test.get("empty_hypothesis_chunks"),
        "test_residual": test.get("diag_mean_residual_ratio"),
        "test_clip": test.get("diag_mean_residual_clip_fraction"),
        "test_cosine": test.get("diag_mean_adapted_cosine"),
    }
    run_lines.append(rec)
    grouped[r["config"]].append(rec)

cols = [
    "run", "config", "seed", "checkpoint", "screen_val_wer", "screen_n",
    "trainable_parameters", "val_wer", "test_wer", "e21_wer", "test_rtf",
    "test_empty", "test_residual", "test_clip", "test_cosine",
]

with (root / "run_summary.tsv").open("w") as o:
    o.write("\t".join(cols) + "\n")
    for x in run_lines:
        o.write("\t".join(fmt(x.get(c)) for c in cols) + "\n")

def mean_std(vals):
    vals = [float(x) for x in vals if x is not None]
    if not vals:
        return None, None, 0
    return (
        statistics.mean(vals),
        statistics.stdev(vals) if len(vals) > 1 else 0.0,
        len(vals),
    )

acols = [
    "config", "n_val", "n_test", "n_e21",
    "val_mean", "val_std", "test_mean", "test_std",
    "e21_mean", "e21_std", "test_rtf_mean", "test_empty_mean",
    "test_residual_mean", "test_clip_mean", "test_cosine_mean",
]

with (root / "aggregate_summary.tsv").open("w") as o:
    o.write("\t".join(acols) + "\n")

    for config in dict.fromkeys(r["config"] for r in rows):
        xs = grouped[config]
        vm, vs, nv = mean_std([x["val_wer"] for x in xs])
        tm, ts, nt = mean_std([x["test_wer"] for x in xs])
        em, es, ne = mean_std([x["e21_wer"] for x in xs])
        rm, _, _ = mean_std([x["test_rtf"] for x in xs])
        xm, _, _ = mean_std([x["test_empty"] for x in xs])
        residual, _, _ = mean_std([x["test_residual"] for x in xs])
        clip, _, _ = mean_std([x["test_clip"] for x in xs])
        cosine, _, _ = mean_std([x["test_cosine"] for x in xs])

        vals = [
            config, nv, nt, ne,
            vm, vs, tm, ts,
            em, es, rm, xm,
            residual, clip, cosine,
        ]
        o.write("\t".join(fmt(v) for v in vals) + "\n")

# Machine-readable completeness report.
status = {
    "runs_total": len(rows),
    "selected": sum(bool(x.get("checkpoint")) for x in run_lines),
    "full_validation_metrics": sum(x.get("val_wer") is not None for x in run_lines),
    "e22_test_metrics": sum(x.get("test_wer") is not None for x in run_lines),
    "e21_test_metrics": sum(x.get("e21_wer") is not None for x in run_lines),
}
(root / "evaluation_status.json").write_text(json.dumps(status, indent=2) + "\n")

print("=== evaluation status ===")
print(json.dumps(status, indent=2))
print("\n=== aggregate summary ===")
print((root / "aggregate_summary.tsv").read_text())
PY
}

if has_stage summary; then
  if [[ -n "$RUN_FILTER" ]]; then
    log "RUN_FILTER is set; summary uses the existing full run_manifest.tsv and may include unfinished runs"
  fi
  summarize | tee "$RUN_ROOT/summary.log"
fi

log "evaluation-only recovery complete: $RUN_ROOT"
