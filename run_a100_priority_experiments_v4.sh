#!/usr/bin/env bash
# CSA-Adapter v0.2.1-fixed: DGX-A100 8x80GB main reproducibility pipeline.
#
# Goals:
#   1) use all eight A100 GPUs;
#   2) reproduce the v0.2.1 default (event compressor, hard sparse from step 1);
#   3) run eight independent seeds instead of selecting a lucky seed;
#   4) select a checkpoint WITHIN each seed by free-running long-form validation WER-N;
#   5) keep text-history OFF for the primary result;
#   6) evaluate all selected seeds on E22 test and Earnings-21;
#   7) run a small persistence/text-history diagnostic matrix.
#
# Recommended:
#   GPUS=0,1,2,3,4,5,6,7 bash scripts/run_a100_priority_experiments_v4.sh
#
# Plan only:
#   PLAN_ONLY=1 bash scripts/run_a100_priority_experiments_v4.sh
#
# Resume selected stages:
#   STAGES=train,select,val,test,external,diagnostics,summary \
#     bash scripts/run_a100_priority_experiments_v4.sh

set -Eeuo pipefail

# -----------------------------------------------------------------------------
# Robust repository bootstrap: works whether this file is under scripts/ or at
# the repository root.
# -----------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "$SCRIPT_DIR/_repo_env.sh" ]]; then
    # Normal/recommended location: repo/scripts/
    source "$SCRIPT_DIR/_repo_env.sh"
elif [[ -f "$SCRIPT_DIR/scripts/_repo_env.sh" ]]; then
    # Also support placing this script at repo root.
    source "$SCRIPT_DIR/scripts/_repo_env.sh"
else
    echo "ERROR: cannot locate _repo_env.sh from $SCRIPT_DIR" >&2
    exit 2
fi

cd "$CSA_REPO_ROOT"
python scripts/preflight_v021.py

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
MODEL="${MODEL:-openai/whisper-large-v3}"
REVISION="${REVISION:-main}"
DATA_DIR="${DATA_DIR:-data/earnings22_622}"
CACHE_DIR="${CACHE_DIR:-cache/earnings22_large_v3}"
RUN_ROOT="${RUN_ROOT:-runs/a100_priority_v4}"
LOG_DIR="${LOG_DIR:-$RUN_ROOT/logs}"

GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
TRAIN_SEEDS="${TRAIN_SEEDS:-42,43,44,45,46,47,48,49}"

MAX_STEPS="${MAX_STEPS:-1500}"
EPOCHS="${EPOCHS:-99}"
GRAD_ACCUM="${GRAD_ACCUM:-8}"
LR="${LR:-1e-4}"
VALID_EXAMPLES="${VALID_EXAMPLES:-256}"
VALIDATE_EVERY="${VALIDATE_EVERY:-250}"
FEATURE_LRU="${FEATURE_LRU:-256}"
DIAGNOSTICS_EVERY="${DIAGNOSTICS_EVERY:-10}"

# v0.2.1 stability/default configuration.
HISTORY_SEGMENTS="${HISTORY_SEGMENTS:-64}"
COMPRESSION_RATE="${COMPRESSION_RATE:-8}"
MAX_MEMORY="${MAX_MEMORY:-4096}"
TOP_K="${TOP_K:-16}"
RANK="${RANK:-16}"
ALPHA_INIT="${ALPHA_INIT:-0.01}"
ALPHA_MAX="${ALPHA_MAX:-0.10}"
GATE_BIAS_INIT="${GATE_BIAS_INIT:--2.0}"
RESIDUAL_RATIO_CAP="${RESIDUAL_RATIO_CAP:-0.25}"

# 0 = use the complete validation.full.jsonl for every saved checkpoint.
# Set e.g. 8 for a faster checkpoint screen, followed by full validation of the
# selected checkpoint in the "val" stage.
CHECKPOINT_SELECT_MAX_CALLS="${CHECKPOINT_SELECT_MAX_CALLS:-8}"

# Default pipeline. "smoke" runs only on GPU0 before launching the 8 real runs.
STAGES="${STAGES:-setup,prepare,cache,smoke,train,select,val,test,external,diagnostics,summary}"
PLAN_ONLY="${PLAN_ONLY:-0}"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

mkdir -p "$RUN_ROOT" "$LOG_DIR"

has_stage() { [[ ",$STAGES," == *",$1,"* ]]; }
log() { printf '[%(%F %T)T] %s\n' -1 "$*"; }
die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

IFS=',' read -r -a GPU_IDS <<< "$GPUS"
IFS=',' read -r -a SEEDS <<< "$TRAIN_SEEDS"

[[ "${#GPU_IDS[@]}" -eq 8 ]] || die "GPUS must contain exactly 8 GPU IDs; got: $GPUS"
[[ "${#SEEDS[@]}" -eq 8 ]] || die "TRAIN_SEEDS must contain exactly 8 seeds; got: $TRAIN_SEEDS"

# Ensure GPU IDs are unique.
declare -A _seen_gpu=()
for gpu in "${GPU_IDS[@]}"; do
    [[ -z "${_seen_gpu[$gpu]+x}" ]] || die "duplicate GPU id: $gpu"
    _seen_gpu[$gpu]=1
done

# Manifest of the eight independent runs.
RUN_NAMES=()
for seed in "${SEEDS[@]}"; do
    RUN_NAMES+=("default_s${seed}")
done

cat >"$RUN_ROOT/run_manifest.tsv" <<EOF
run	seed	gpu
EOF
for i in "${!RUN_NAMES[@]}"; do
    printf '%s\t%s\t%s\n' "${RUN_NAMES[$i]}" "${SEEDS[$i]}" "${GPU_IDS[$i]}" \
        >>"$RUN_ROOT/run_manifest.tsv"
done

log "A100 v4: ${#RUN_NAMES[@]} default runs on ${#GPU_IDS[@]} GPUs"
log "seeds: ${SEEDS[*]}"
log "primary setting: event/C=$COMPRESSION_RATE/L=$MAX_MEMORY/top-k=$TOP_K/rank=$RANK, hard sparse, text-history OFF"
log "stages: $STAGES"

if [[ "$PLAN_ONLY" == 1 ]]; then
    cat "$RUN_ROOT/run_manifest.tsv"
    log "PLAN_ONLY=1; no setup/cache/train/evaluation launched."
    exit 0
fi

# -----------------------------------------------------------------------------
# Process helpers
# -----------------------------------------------------------------------------
declare -a PIDS=() JOB_NAMES=()

stop_children() {
    local p
    for p in "${PIDS[@]:-}"; do
        kill "$p" 2>/dev/null || true
    done
}
trap stop_children INT TERM

launch() {
    local name="$1" gpu="$2" logfile="$3"
    shift 3
    log "launch physical GPU $gpu: $name"
    CUDA_VISIBLE_DEVICES="$gpu" "$@" >"$logfile" 2>&1 &
    PIDS+=("$!")
    JOB_NAMES+=("$name")
}

wait_all() {
    local bad=0 i
    for i in "${!PIDS[@]}"; do
        if wait "${PIDS[$i]}"; then
            log "finished: ${JOB_NAMES[$i]}"
        else
            log "FAILED: ${JOB_NAMES[$i]} (see log)"
            bad=1
        fi
    done
    PIDS=()
    JOB_NAMES=()
    [[ "$bad" -eq 0 ]] || die "one or more parallel jobs failed; inspect $LOG_DIR"
}

# -----------------------------------------------------------------------------
# Cache integrity helpers. Never silently rewrite a cache with another shard
# identity (the source of the earlier 'Cache identity changed' failure).
# -----------------------------------------------------------------------------
cache_complete() {
    python - "$1" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
first = root / "metadata-000.json"
if not first.is_file():
    raise SystemExit(1)
try:
    identity = json.loads(first.read_text())
    shards = int(identity["shards"])
except (OSError, ValueError, KeyError, json.JSONDecodeError):
    raise SystemExit(1)
if shards <= 0:
    raise SystemExit(1)
for index in range(shards):
    meta = root / f"metadata-{index:03d}.json"
    rows = root / f"index-{index:03d}.jsonl"
    if not meta.is_file() or not rows.is_file():
        raise SystemExit(1)
    try:
        if json.loads(meta.read_text()) != identity:
            raise SystemExit(1)
    except (OSError, json.JSONDecodeError):
        raise SystemExit(1)
raise SystemExit(0)
PY
}

cache_has_metadata() {
    compgen -G "$1/metadata-*.json" >/dev/null
}

# -----------------------------------------------------------------------------
# setup / data preparation / 8-GPU encoder feature caching
# -----------------------------------------------------------------------------
if has_stage setup; then
    log "setup + repository tests"
    bash scripts/00_setup.sh 2>&1 | tee "$LOG_DIR/00_setup.log"
    python scripts/preflight_v021.py
fi

if has_stage prepare; then
    if [[ -f "$DATA_DIR/split_calls.json" && -f "$DATA_DIR/preparation_report.json" ]]; then
        log "reuse prepared dataset: $DATA_DIR"
    else
        log "prepare Earnings-22/Earnings-21 manifests"
        DATA_DIR="$DATA_DIR" SEED=42 \
            bash scripts/01_prepare_earnings22.sh \
            2>&1 | tee "$LOG_DIR/01_prepare.log"
    fi
fi

if has_stage cache; then
    [[ -f "$DATA_DIR/train.jsonl" && -f "$DATA_DIR/validation.jsonl" ]] || \
        die "missing train/validation manifests under $DATA_DIR"

    if cache_complete "$CACHE_DIR/train" && cache_complete "$CACHE_DIR/validation"; then
        log "reuse complete feature cache: $CACHE_DIR"
    elif cache_has_metadata "$CACHE_DIR/train" || cache_has_metadata "$CACHE_DIR/validation"; then
        die "incomplete/inconsistent cache under $CACHE_DIR; keep it for inspection and use a fresh CACHE_DIR"
    else
        log "cache frozen Whisper encoder outputs with all 8 A100 GPUs"
        DATA_DIR="$DATA_DIR" CACHE_DIR="$CACHE_DIR" MODEL="$MODEL" REVISION="$REVISION" \
            GPUS="$GPUS" bash scripts/02_cache_features.sh \
            2>&1 | tee "$LOG_DIR/02_cache_features.log"
    fi
fi

# -----------------------------------------------------------------------------
# Smoke test on real cached data before the eight full runs.
# -----------------------------------------------------------------------------
if has_stage smoke; then
    [[ -d "$CACHE_DIR/train" && -d "$CACHE_DIR/validation" ]] || \
        die "feature cache missing under $CACHE_DIR"
    log "run v0.2.1 smoke training on GPU ${GPU_IDS[0]}"
    CUDA_VISIBLE_DEVICES="${GPU_IDS[0]}" \
        CACHE_DIR="$CACHE_DIR" MODEL="$MODEL" REVISION="$REVISION" \
        bash scripts/07_smoke_train.sh \
        2>&1 | tee "$LOG_DIR/07_smoke_train.log"
    python scripts/preflight_v021.py
fi

# -----------------------------------------------------------------------------
# Train 8 independent v0.2.1 default seeds: one full trainer per A100.
# A100-80GB has ample memory for this cached-feature trainer; using one process
# per GPU avoids oversubscription and gives clean, reproducible timing.
# -----------------------------------------------------------------------------
if has_stage train; then
    [[ -d "$CACHE_DIR/train" && -d "$CACHE_DIR/validation" ]] || \
        die "feature cache missing under $CACHE_DIR"

    for i in "${!RUN_NAMES[@]}"; do
        run="${RUN_NAMES[$i]}"
        seed="${SEEDS[$i]}"
        gpu="${GPU_IDS[$i]}"
        out="$RUN_ROOT/$run"

        if [[ -f "$out/run.json" ]]; then
            # Allow a completed run to be resumed into selection/evaluation.
            if compgen -G "$out/checkpoints/step-*/adapter.safetensors" >/dev/null; then
                log "skip existing completed training run: $run"
                continue
            fi
            die "existing incomplete run detected: $out"
        fi

        launch "train_$run" "$gpu" "$LOG_DIR/${run}.train.log" \
            python -m csa_adapter.longform.train \
            --train-cache "$CACHE_DIR/train" \
            --valid-cache "$CACHE_DIR/validation" \
            --output "$out" \
            --model "$MODEL" --revision "$REVISION" \
            --epochs "$EPOCHS" --max-steps "$MAX_STEPS" \
            --grad-accum "$GRAD_ACCUM" --lr "$LR" --seed "$seed" \
            --history-segments "$HISTORY_SEGMENTS" \
            --compression-rate "$COMPRESSION_RATE" \
            --max-memory "$MAX_MEMORY" \
            --top-k "$TOP_K" --rank "$RANK" \
            --compressor event --warmup-steps 0 \
            --alpha-init "$ALPHA_INIT" --alpha-max "$ALPHA_MAX" \
            --gate-bias-init "$GATE_BIAS_INIT" \
            --residual-ratio-cap "$RESIDUAL_RATIO_CAP" \
            --valid-examples "$VALID_EXAMPLES" \
            --validate-every "$VALIDATE_EVERY" \
            --diagnostics-every "$DIAGNOSTICS_EVERY" \
            --feature-lru "$FEATURE_LRU" \
            --save-validation-checkpoints
    done
    wait_all
fi

# -----------------------------------------------------------------------------
# Select a checkpoint WITHIN each seed by free-running long-form validation.
# Primary selection is acoustic-memory-only: text-history OFF.
# -----------------------------------------------------------------------------
if has_stage select; then
    manifest="$DATA_DIR/validation.full.jsonl"
    [[ -f "$manifest" ]] || die "missing $manifest"

    for i in "${!RUN_NAMES[@]}"; do
        run="${RUN_NAMES[$i]}"
        gpu="${GPU_IDS[$i]}"
        run_dir="$RUN_ROOT/$run"

        (
            shopt -s nullglob
            ckpts=("$run_dir"/checkpoints/step-*)
            [[ "${#ckpts[@]}" -gt 0 ]] || {
                echo "No validation checkpoints found for $run" >&2
                exit 20
            }

            for ckpt in "${ckpts[@]}"; do
                tag="$(basename "$ckpt")"
                output="$run_dir/checkpoint_eval/$tag"
                if [[ -f "$output/metrics.json" ]]; then
                    echo "[ckpt-valid] reuse $run $tag"
                    continue
                fi

                max_args=()
                if [[ "$CHECKPOINT_SELECT_MAX_CALLS" != 0 ]]; then
                    max_args+=(--max-calls "$CHECKPOINT_SELECT_MAX_CALLS")
                fi

                echo "[ckpt-valid] GPU=$gpu run=$run checkpoint=$tag"
                CUDA_VISIBLE_DEVICES="$gpu" python -m csa_adapter.longform.evaluate \
                    --manifest "$manifest" \
                    --output "$output" \
                    --model "$MODEL" --revision "$REVISION" \
                    --adapter "$ckpt" \
                    --memory-mode history \
                    --diagnostics \
                    "${max_args[@]}" \
                    >"$LOG_DIR/${run}.${tag}.select.log" 2>&1
            done

            python - "$run_dir" <<'PY'
import json
import sys
from pathlib import Path

run = Path(sys.argv[1])
values = []
for p in sorted((run / "checkpoint_eval").glob("step-*/metrics.json")):
    d = json.loads(p.read_text())
    values.append((float(d["normalized_micro_wer"]), p.parent.name))
if not values:
    raise SystemExit(f"no checkpoint metrics for {run}")
score, tag = min(values)
(run / "selected_checkpoint.txt").write_text(tag + "\n")
(run / "selected_checkpoint.json").write_text(
    json.dumps({"checkpoint": tag, "selection_validation_micro_wer": score}, indent=2) + "\n"
)
print(f"{run.name}\t{tag}\t{score:.10f}")
PY
        ) &
        PIDS+=("$!")
        JOB_NAMES+=("select_$run")
    done
    wait_all
fi

# -----------------------------------------------------------------------------
# Full validation of the selected checkpoint for every seed.
# This is separate from the small checkpoint-screening subset.
# -----------------------------------------------------------------------------
if has_stage val; then
    manifest="$DATA_DIR/validation.full.jsonl"
    [[ -f "$manifest" ]] || die "missing $manifest"

    for i in "${!RUN_NAMES[@]}"; do
        run="${RUN_NAMES[$i]}"
        gpu="${GPU_IDS[$i]}"
        run_dir="$RUN_ROOT/$run"
        [[ -f "$run_dir/selected_checkpoint.txt" ]] || die "missing selected checkpoint for $run"
        tag="$(tr -d '\r\n' < "$run_dir/selected_checkpoint.txt")"
        adapter="$run_dir/checkpoints/$tag"
        output="$run_dir/eval_validation_text_off"

        if [[ -f "$output/metrics.json" ]]; then
            log "reuse full validation: $run"
            continue
        fi

        launch "val_$run" "$gpu" "$LOG_DIR/${run}.val.log" \
            python -m csa_adapter.longform.evaluate \
            --manifest "$manifest" \
            --output "$output" \
            --model "$MODEL" --revision "$REVISION" \
            --adapter "$adapter" \
            --memory-mode history \
            --diagnostics
    done
    wait_all
fi

# -----------------------------------------------------------------------------
# E22 test: all 8 seeds in parallel. No seed selection.
# -----------------------------------------------------------------------------
if has_stage test; then
    manifest="$DATA_DIR/test.full.jsonl"
    [[ -f "$manifest" ]] || die "missing $manifest"

    for i in "${!RUN_NAMES[@]}"; do
        run="${RUN_NAMES[$i]}"
        gpu="${GPU_IDS[$i]}"
        run_dir="$RUN_ROOT/$run"
        [[ -f "$run_dir/selected_checkpoint.txt" ]] || die "missing selected checkpoint for $run"
        tag="$(tr -d '\r\n' < "$run_dir/selected_checkpoint.txt")"
        adapter="$run_dir/checkpoints/$tag"
        output="$run_dir/eval_test_text_off"

        if [[ -f "$output/metrics.json" ]]; then
            log "reuse E22 test: $run"
            continue
        fi

        launch "test_$run" "$gpu" "$LOG_DIR/${run}.test.log" \
            python -m csa_adapter.longform.evaluate \
            --manifest "$manifest" \
            --output "$output" \
            --model "$MODEL" --revision "$REVISION" \
            --adapter "$adapter" \
            --memory-mode history \
            --diagnostics
    done
    wait_all
fi

# -----------------------------------------------------------------------------
# Earnings-21 external test: all 8 seeds in parallel, text-history OFF.
# -----------------------------------------------------------------------------
if has_stage external; then
    manifest="$DATA_DIR/earnings21.full.jsonl"
    [[ -f "$manifest" ]] || \
        die "missing $manifest; prepare Earnings-21 before running external stage"

    for i in "${!RUN_NAMES[@]}"; do
        run="${RUN_NAMES[$i]}"
        gpu="${GPU_IDS[$i]}"
        run_dir="$RUN_ROOT/$run"
        [[ -f "$run_dir/selected_checkpoint.txt" ]] || die "missing selected checkpoint for $run"
        tag="$(tr -d '\r\n' < "$run_dir/selected_checkpoint.txt")"
        adapter="$run_dir/checkpoints/$tag"
        output="$run_dir/eval_e21_text_off"

        if [[ -f "$output/metrics.json" ]]; then
            log "reuse E21: $run"
            continue
        fi

        launch "e21_$run" "$gpu" "$LOG_DIR/${run}.e21.log" \
            python -m csa_adapter.longform.evaluate \
            --manifest "$manifest" \
            --output "$output" \
            --model "$MODEL" --revision "$REVISION" \
            --adapter "$adapter" \
            --memory-mode history \
            --diagnostics
    done
    wait_all
fi

# -----------------------------------------------------------------------------
# Diagnostic matrix.
# Use seed 42 as the fixed diagnostic adapter; primary paper results above still
# report all 8 seeds. The matrix separates acoustic persistence from text prompt.
# Eight A100s are used at once.
# -----------------------------------------------------------------------------
if has_stage diagnostics; then
    diag_run="default_s${SEEDS[0]}"
    diag_dir="$RUN_ROOT/$diag_run"
    [[ -f "$diag_dir/selected_checkpoint.txt" ]] || die "missing selected checkpoint for $diag_run"
    diag_tag="$(tr -d '\r\n' < "$diag_dir/selected_checkpoint.txt")"
    diag_adapter="$diag_dir/checkpoints/$diag_tag"

    DIAG_NAMES=(
        e22_baseline_text_off
        e22_baseline_text_on
        e22_csa_history_text_off
        e22_csa_reset_text_off
        e22_csa_history_text_on
        e21_baseline_text_off
        e21_baseline_text_on
        e21_csa_history_text_off
    )

    for i in "${!DIAG_NAMES[@]}"; do
        name="${DIAG_NAMES[$i]}"
        gpu="${GPU_IDS[$i]}"
        args=()
        case "$name" in
            e22_baseline_text_off)
                manifest="$DATA_DIR/test.full.jsonl"
                ;;
            e22_baseline_text_on)
                manifest="$DATA_DIR/test.full.jsonl"
                args+=(--text-history)
                ;;
            e22_csa_history_text_off)
                manifest="$DATA_DIR/test.full.jsonl"
                args+=(--adapter "$diag_adapter" --memory-mode history)
                ;;
            e22_csa_reset_text_off)
                manifest="$DATA_DIR/test.full.jsonl"
                args+=(--adapter "$diag_adapter" --memory-mode reset)
                ;;
            e22_csa_history_text_on)
                manifest="$DATA_DIR/test.full.jsonl"
                args+=(--adapter "$diag_adapter" --memory-mode history --text-history)
                ;;
            e21_baseline_text_off)
                manifest="$DATA_DIR/earnings21.full.jsonl"
                ;;
            e21_baseline_text_on)
                manifest="$DATA_DIR/earnings21.full.jsonl"
                args+=(--text-history)
                ;;
            e21_csa_history_text_off)
                manifest="$DATA_DIR/earnings21.full.jsonl"
                args+=(--adapter "$diag_adapter" --memory-mode history)
                ;;
        esac
        [[ -f "$manifest" ]] || die "missing diagnostic manifest: $manifest"
        output="$RUN_ROOT/diagnostics/$name"

        if [[ -f "$output/metrics.json" ]]; then
            log "reuse diagnostic: $name"
            continue
        fi

        launch "diag_$name" "$gpu" "$LOG_DIR/diag_${name}.log" \
            python -m csa_adapter.longform.evaluate \
            --manifest "$manifest" \
            --output "$output" \
            --model "$MODEL" --revision "$REVISION" \
            --diagnostics \
            "${args[@]}"
    done
    wait_all
fi

# -----------------------------------------------------------------------------
# Summary: per-seed + mean/std. No seed is discarded.
# -----------------------------------------------------------------------------
if has_stage summary; then
    python - "$RUN_ROOT" "${RUN_NAMES[@]}" <<'PY'
import json
import math
import statistics
import sys
from pathlib import Path

root = Path(sys.argv[1])
runs = sys.argv[2:]

def metric(path):
    if not path.is_file():
        return None
    d = json.loads(path.read_text())
    return {
        "micro": float(d["normalized_micro_wer"]),
        "macro": float(d["normalized_macro_call_wer"]),
        "rtf": float(d["rtf"]),
    }

rows = []
for run in runs:
    r = root / run
    seed = run.rsplit("_s", 1)[-1]
    ckpt = (r / "selected_checkpoint.txt").read_text().strip() if (r / "selected_checkpoint.txt").is_file() else ""
    val = metric(r / "eval_validation_text_off" / "metrics.json")
    test = metric(r / "eval_test_text_off" / "metrics.json")
    e21 = metric(r / "eval_e21_text_off" / "metrics.json")
    rows.append((run, seed, ckpt, val, test, e21))

out = root / "run_summary.tsv"
with out.open("w") as f:
    f.write("run\tseed\tcheckpoint\tval_micro\tval_macro\tval_rtf\t"
            "test_micro\ttest_macro\ttest_rtf\te21_micro\te21_macro\te21_rtf\n")
    for run, seed, ckpt, val, test, e21 in rows:
        def vals(x):
            return ("", "", "") if x is None else (
                f"{x['micro']:.10f}", f"{x['macro']:.10f}", f"{x['rtf']:.10f}"
            )
        f.write("\t".join([run, seed, ckpt, *vals(val), *vals(test), *vals(e21)]) + "\n")

def stats_for(key, field="micro"):
    vals = []
    for _, _, _, val, test, e21 in rows:
        x = {"val": val, "test": test, "e21": e21}[key]
        if x is not None:
            vals.append(x[field])
    if not vals:
        return 0, math.nan, math.nan
    mean = statistics.mean(vals)
    std = statistics.stdev(vals) if len(vals) > 1 else 0.0
    return len(vals), mean, std

agg = root / "aggregate_summary.tsv"
with agg.open("w") as f:
    f.write("dataset\tn\tmicro_mean\tmicro_std\tmacro_mean\tmacro_std\trtf_mean\trtf_std\n")
    for key in ("val", "test", "e21"):
        nm, mm, sm = stats_for(key, "micro")
        _, ma, sa = stats_for(key, "macro")
        _, mr, sr = stats_for(key, "rtf")
        f.write(f"{key}\t{nm}\t{mm:.10f}\t{sm:.10f}\t{ma:.10f}\t{sa:.10f}\t{mr:.10f}\t{sr:.10f}\n")

diag_root = root / "diagnostics"
diag_out = root / "diagnostic_summary.tsv"
with diag_out.open("w") as f:
    f.write("name\tnormalized_micro_wer\tnormalized_macro_call_wer\trtf\n")
    if diag_root.is_dir():
        for p in sorted(diag_root.glob("*/metrics.json")):
            d = json.loads(p.read_text())
            f.write(
                f"{p.parent.name}\t{float(d['normalized_micro_wer']):.10f}\t"
                f"{float(d['normalized_macro_call_wer']):.10f}\t{float(d['rtf']):.10f}\n"
            )

print(out.read_text(), end="")
print("\nAggregate:")
print(agg.read_text(), end="")
if diag_out.is_file():
    print("\nDiagnostics:")
    print(diag_out.read_text(), end="")
PY

    # Also keep the repository's generic diagnostic summarizer output.
    python scripts/summarize_v021.py "$RUN_ROOT" \
        >"$RUN_ROOT/summarize_v021.tsv" 2>"$RUN_ROOT/summarize_v021.stderr" || true
fi

log "A100 v4 requested stages complete."
log "Per-seed summary : $RUN_ROOT/run_summary.tsv"
log "Mean/std summary : $RUN_ROOT/aggregate_summary.tsv"
log "Diagnostics      : $RUN_ROOT/diagnostic_summary.tsv"
