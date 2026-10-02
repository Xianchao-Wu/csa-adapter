#!/usr/bin/env bash
# Whisper-large-v3 LoRA rank sweep on the same Earnings-22/Earnings-21 data
# used by the CSA-Adapter H100/H200 experiments.
#
# Default:
#   ranks = 8,16,32,64
#   seeds = 42,43,44 (12 runs total across four ranks)
#   independent single-GPU LoRA trainings run in parallel
#   final evaluation reuses csa_adapter.longform.evaluate
#
# Expected data:
#   data/earnings22_622/train.jsonl
#   data/earnings22_622/validation.jsonl
#   data/earnings22_622/validation.full.jsonl
#   data/earnings22_622/test.full.jsonl
#   data/earnings22_622/earnings21.full.jsonl
#
# Run from the CSA-Adapter v0.2.1 repository root.

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# If this script lives outside repo root, set REPO explicitly.
REPO="${REPO:-$PWD}"
cd "$REPO"

MODEL="${MODEL:-openai/whisper-large-v3}"
REVISION="${REVISION:-main}"
DATA_DIR="${DATA_DIR:-data/earnings22_622}"
OUT_ROOT="${OUT_ROOT:-runs/whisper_lora_earnings22_622}"

RANKS="${RANKS:-8,16,32,64}"
SEEDS="${SEEDS:-42,43,44}"
GPUS="${GPUS:-0,1,2,3,4,5,6,7}"

EPOCHS="${EPOCHS:-3}"
MAX_STEPS="${MAX_STEPS:--1}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-8}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-8}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
LR="${LR:-1e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
WARMUP_RATIO="${WARMUP_RATIO:-0.05}"
EVAL_STEPS="${EVAL_STEPS:-250}"
SAVE_STEPS="${SAVE_STEPS:-250}"
LOGGING_STEPS="${LOGGING_STEPS:-25}"
DATALOADER_WORKERS="${DATALOADER_WORKERS:-4}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"

# H100/H200: BF16 + TF32 are sensible defaults.
BF16="${BF16:-1}"
TF32="${TF32:-1}"

# Stages can be rerun independently.
STAGES="${STAGES:-train,eval,summary}"

mkdir -p "$OUT_ROOT/logs"

has_stage() { [[ ",$STAGES," == *",$1,"* ]]; }
log() { printf '[%(%F %T)T] %s\n' -1 "$*"; }
die() { echo "ERROR: $*" >&2; exit 1; }

IFS=',' read -r -a RANK_ARR <<< "$RANKS"
IFS=',' read -r -a SEED_ARR <<< "$SEEDS"
IFS=',' read -r -a GPU_ARR <<< "$GPUS"

[[ "${#GPU_ARR[@]}" -ge 1 ]] || die "GPUS cannot be empty"

# ---------------------------------------------------------------------------
# Dependency / data preflight
# ---------------------------------------------------------------------------
python - <<'PY'
import importlib
mods = ["torch", "transformers", "peft", "soundfile", "scipy"]
missing = []
for m in mods:
    try:
        importlib.import_module(m)
    except Exception as e:
        missing.append((m, repr(e)))
if missing:
    print("Missing/broken Python dependencies:")
    for x in missing:
        print("  ", x)
    print("\nSuggested install:")
    print("  python -m pip install -U peft soundfile scipy")
    raise SystemExit(1)
import torch, transformers, peft
print("torch       :", torch.__version__)
print("transformers:", transformers.__version__)
print("peft        :", peft.__version__)
print("cuda        :", torch.cuda.is_available(), torch.cuda.device_count())
PY

for f in \
  "$DATA_DIR/train.jsonl" \
  "$DATA_DIR/validation.jsonl" \
  "$DATA_DIR/validation.full.jsonl" \
  "$DATA_DIR/test.full.jsonl" \
  "$DATA_DIR/earnings21.full.jsonl"
do
  [[ -f "$f" ]] || die "missing data file: $f"
done

[[ -f train_whisper_lora_earnings.py ]] || \
  die "missing train_whisper_lora_earnings.py in repo root"

# ---------------------------------------------------------------------------
# Build all run names: rank{8,16,32,64}_s{seed}
# ---------------------------------------------------------------------------
RUN_NAMES=()
RUN_RANKS=()
RUN_SEEDS=()

for rank in "${RANK_ARR[@]}"; do
  case "$rank" in 8|16|32|64) ;; *) die "unsupported rank: $rank" ;; esac
  for seed in "${SEED_ARR[@]}"; do
    RUN_NAMES+=("rank${rank}_s${seed}")
    RUN_RANKS+=("$rank")
    RUN_SEEDS+=("$seed")
  done
done

{
  echo -e "run\trank\tseed"
  for i in "${!RUN_NAMES[@]}"; do
    echo -e "${RUN_NAMES[$i]}\t${RUN_RANKS[$i]}\t${RUN_SEEDS[$i]}"
  done
} > "$OUT_ROOT/run_manifest.tsv"

log "LoRA sweep: ${#RUN_NAMES[@]} run(s)"
column -t -s $'\t' "$OUT_ROOT/run_manifest.tsv" 2>/dev/null || cat "$OUT_ROOT/run_manifest.tsv"

declare -a PIDS=() JOBS=()

wait_all() {
  local bad=0 i
  for i in "${!PIDS[@]}"; do
    if wait "${PIDS[$i]}"; then
      log "finished: ${JOBS[$i]}"
    else
      log "FAILED: ${JOBS[$i]}"
      bad=1
    fi
  done
  PIDS=()
  JOBS=()
  [[ "$bad" -eq 0 ]] || die "one or more jobs failed; inspect $OUT_ROOT/logs"
}

wait_when_full() {
  local limit="$1"
  (( ${#PIDS[@]} < limit )) || wait_all
}

# ---------------------------------------------------------------------------
# Train LoRA. One independent rank/seed run per GPU.
# ---------------------------------------------------------------------------
if has_stage train; then
  for i in "${!RUN_NAMES[@]}"; do
    run="${RUN_NAMES[$i]}"
    rank="${RUN_RANKS[$i]}"
    seed="${RUN_SEEDS[$i]}"
    gpu="${GPU_ARR[$((i % ${#GPU_ARR[@]}))]}"
    out="$OUT_ROOT/$run"

    if [[ -f "$out/merged_model/config.json" && -f "$out/train_summary.json" ]]; then
      log "skip completed training: $run"
      continue
    fi

    cmd=(
      python train_whisper_lora_earnings.py
      --model "$MODEL"
      --revision "$REVISION"
      --train-manifest "$DATA_DIR/train.jsonl"
      --valid-manifest "$DATA_DIR/validation.jsonl"
      --output-dir "$out"
      --rank "$rank"
      --seed "$seed"
      --epochs "$EPOCHS"
      --max-steps "$MAX_STEPS"
      --learning-rate "$LR"
      --weight-decay "$WEIGHT_DECAY"
      --warmup-ratio "$WARMUP_RATIO"
      --train-batch-size "$TRAIN_BATCH_SIZE"
      --eval-batch-size "$EVAL_BATCH_SIZE"
      --grad-accum "$GRAD_ACCUM"
      --eval-steps "$EVAL_STEPS"
      --save-steps "$SAVE_STEPS"
      --logging-steps "$LOGGING_STEPS"
      --dataloader-workers "$DATALOADER_WORKERS"
      --lora-dropout "$LORA_DROPOUT"
      --gradient-checkpointing
      --merge
    )
    [[ "$BF16" == 1 ]] && cmd+=(--bf16)
    [[ "$TF32" == 1 ]] && cmd+=(--tf32)

    log "launch GPU $gpu: train $run (rank=$rank seed=$seed)"
    (
      export CUDA_VISIBLE_DEVICES="$gpu"
      "${cmd[@]}"
    ) >"$OUT_ROOT/logs/train_${run}.log" 2>&1 &

    PIDS+=("$!")
    JOBS+=("train_$run")
    wait_when_full "${#GPU_ARR[@]}"
  done
  [[ "${#PIDS[@]}" -eq 0 ]] || wait_all
fi

# ---------------------------------------------------------------------------
# Long-form evaluation with the EXISTING CSA evaluator.
# This keeps the evaluation protocol and normalized_micro_wer identical.
# Each run gets one GPU and evaluates val -> E22 test -> E21 test sequentially.
# ---------------------------------------------------------------------------
eval_run() {
  local run="$1"
  local model_dir="$OUT_ROOT/$run/merged_model"

  [[ -f "$model_dir/config.json" ]] || {
    echo "ERROR: merged model missing: $model_dir" >&2
    return 20
  }

  local name manifest out

  for name in validation e22 e21; do
    case "$name" in
      validation)
        manifest="$DATA_DIR/validation.full.jsonl"
        out="$OUT_ROOT/$run/eval_validation_full"
        ;;
      e22)
        manifest="$DATA_DIR/test.full.jsonl"
        out="$OUT_ROOT/$run/eval_test"
        ;;
      e21)
        manifest="$DATA_DIR/earnings21.full.jsonl"
        out="$OUT_ROOT/$run/eval_e21"
        ;;
    esac

    if [[ -f "$out/metrics.json" ]]; then
      echo "[eval] skip existing $run / $name"
      continue
    fi

    echo "[eval] $run / $name"
    python -m csa_adapter.longform.evaluate \
      --manifest "$manifest" \
      --output "$out" \
      --model "$model_dir" \
      --revision main

    [[ -f "$out/metrics.json" ]] || {
      echo "ERROR: metrics.json missing after $run / $name" >&2
      return 21
    }
  done
}

if has_stage eval; then
  for i in "${!RUN_NAMES[@]}"; do
    run="${RUN_NAMES[$i]}"
    gpu="${GPU_ARR[$((i % ${#GPU_ARR[@]}))]}"

    log "launch GPU $gpu: eval $run"
    (
      export CUDA_VISIBLE_DEVICES="$gpu"
      eval_run "$run"
    ) >"$OUT_ROOT/logs/eval_${run}.log" 2>&1 &

    PIDS+=("$!")
    JOBS+=("eval_$run")
    wait_when_full "${#GPU_ARR[@]}"
  done
  [[ "${#PIDS[@]}" -eq 0 ]] || wait_all
fi

# ---------------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------------
if has_stage summary; then
python - "$OUT_ROOT" <<'PY'
import csv
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

root = Path(sys.argv[1])
rows = list(csv.DictReader((root / "run_manifest.tsv").open(), delimiter="\t"))

def load(p):
    return json.loads(p.read_text()) if p.is_file() else {}

def fmt(v):
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.8f}"
    return str(v)

def ms(vals):
    vals = [float(x) for x in vals if x is not None]
    if not vals:
        return None, None, 0
    return statistics.mean(vals), (
        statistics.stdev(vals) if len(vals) > 1 else 0.0
    ), len(vals)

per_run = []
groups = defaultdict(list)

for r in rows:
    rd = root / r["run"]
    tr = load(rd / "train_summary.json")
    va = load(rd / "eval_validation_full" / "metrics.json")
    te = load(rd / "eval_test" / "metrics.json")
    e21 = load(rd / "eval_e21" / "metrics.json")

    rec = {
        **r,
        "trainable_parameters": tr.get("trainable_parameters"),
        "best_eval_loss": tr.get("best_metric_eval_loss"),
        "val_wer": va.get("normalized_micro_wer"),
        "test_wer": te.get("normalized_micro_wer"),
        "e21_wer": e21.get("normalized_micro_wer"),
        "test_rtf": te.get("rtf"),
        "test_empty": te.get("empty_hypothesis_chunks"),
    }
    per_run.append(rec)
    groups[r["rank"]].append(rec)

cols = [
    "run", "rank", "seed", "trainable_parameters", "best_eval_loss",
    "val_wer", "test_wer", "e21_wer", "test_rtf", "test_empty",
]

with (root / "run_summary.tsv").open("w") as f:
    f.write("\t".join(cols) + "\n")
    for x in per_run:
        f.write("\t".join(fmt(x.get(c)) for c in cols) + "\n")

acols = [
    "rank", "n",
    "val_mean", "val_std",
    "test_mean", "test_std",
    "e21_mean", "e21_std",
    "test_rtf_mean",
]
with (root / "aggregate_summary.tsv").open("w") as f:
    f.write("\t".join(acols) + "\n")
    for rank in dict.fromkeys(r["rank"] for r in rows):
        xs = groups[rank]
        vm, vs, n = ms([x["val_wer"] for x in xs])
        tm, ts, _ = ms([x["test_wer"] for x in xs])
        em, es, _ = ms([x["e21_wer"] for x in xs])
        rm, _, _ = ms([x["test_rtf"] for x in xs])
        vals = [rank, n, vm, vs, tm, ts, em, es, rm]
        f.write("\t".join(fmt(v) for v in vals) + "\n")

status = {
    "runs_total": len(rows),
    "trained_and_merged": sum(
        (root / r["run"] / "merged_model" / "config.json").is_file()
        for r in rows
    ),
    "validation_metrics": sum(
        (root / r["run"] / "eval_validation_full" / "metrics.json").is_file()
        for r in rows
    ),
    "e22_metrics": sum(
        (root / r["run"] / "eval_test" / "metrics.json").is_file()
        for r in rows
    ),
    "e21_metrics": sum(
        (root / r["run"] / "eval_e21" / "metrics.json").is_file()
        for r in rows
    ),
}
(root / "status.json").write_text(json.dumps(status, indent=2) + "\n")

print("=== status.json ===")
print(json.dumps(status, indent=2))
print("=== aggregate_summary.tsv ===")
print((root / "aggregate_summary.tsv").read_text(), end="")
PY
fi

log "LoRA sweep complete: $OUT_ROOT"
log "per-run results : $OUT_ROOT/run_summary.tsv"
log "aggregate       : $OUT_ROOT/aggregate_summary.tsv"
