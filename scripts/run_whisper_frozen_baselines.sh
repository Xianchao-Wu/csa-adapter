#!/usr/bin/env bash
set -Eeuo pipefail

# Frozen Whisper-large-v3 baselines for CSA-Adapter paper.
# No CSA checkpoint, no training, and no selected_runs.tsv are required.
# Run from the csa-adapter repository root after data preparation.
#
# Default jobs:
#   Earnings-22 validation: text off / text on
#   Earnings-22 test:       text off / text on
#   Earnings-21 external:   text off / text on
#
# Example:
#   bash run_whisper_frozen_baselines.sh
#   GPU_LIST=0,1,2,3 DATASETS=test,e21 bash run_whisper_frozen_baselines.sh

MODEL="${MODEL:-openai/whisper-large-v3}"
REVISION="${REVISION:-main}"
DATA_DIR="${DATA_DIR:-data/earnings22_622}"
OUT_ROOT="${OUT_ROOT:-runs/frozen_baselines/whisper_large_v3}"
LOG_DIR="${LOG_DIR:-logs/frozen_baselines/whisper_large_v3}"
GPU_LIST="${GPU_LIST:-0,1,2,3,4,5}"
DATASETS="${DATASETS:-validation,test,e21}"
MODES="${MODES:-text_off,text_on}"

mkdir -p "$OUT_ROOT" "$LOG_DIR"

log() { printf '[%(%F %T)T] %s\n' -1 "$*"; }
die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

python - <<'PY' >/dev/null 2>&1 || {
import csa_adapter.longform.evaluate  # noqa: F401
PY
  die "cannot import csa_adapter.longform.evaluate; run scripts/00_setup.sh first"
}

IFS=',' read -r -a GPUS <<< "$GPU_LIST"
[[ ${#GPUS[@]} -ge 1 ]] || die "GPU_LIST is empty"

for gpu in "${GPUS[@]}"; do
  nvidia-smi -i "$gpu" --query-gpu=index --format=csv,noheader >/dev/null 2>&1 || die "GPU $gpu is not visible"
done

declare -a PIDS=() NAMES=()
cleanup() {
  local p
  for p in "${PIDS[@]:-}"; do kill "$p" 2>/dev/null || true; done
}
trap cleanup EXIT INT TERM

launch() {
  local gpu="$1" name="$2" manifest="$3" out="$4"; shift 4
  if [[ -f "$out/metrics.json" ]]; then
    log "skip completed: $name -> $out/metrics.json"
    return
  fi
  mkdir -p "$out"
  log "launch GPU $gpu: $name"
  CUDA_VISIBLE_DEVICES="$gpu" \
    python -m csa_adapter.longform.evaluate \
      --manifest "$manifest" \
      --output "$out" \
      --model "$MODEL" \
      --revision "$REVISION" \
      "$@" \
      >"$LOG_DIR/${name}.log" 2>&1 &
  PIDS+=("$!")
  NAMES+=("$name")
}

manifest_for() {
  case "$1" in
    validation) printf '%s\n' "$DATA_DIR/validation.full.jsonl" ;;
    test)       printf '%s\n' "$DATA_DIR/test.full.jsonl" ;;
    e21)        printf '%s\n' "$DATA_DIR/earnings21.full.jsonl" ;;
    *) die "unknown DATASET: $1" ;;
  esac
}

job=0
IFS=',' read -r -a DS <<< "$DATASETS"
IFS=',' read -r -a MS <<< "$MODES"
for ds in "${DS[@]}"; do
  manifest="$(manifest_for "$ds")"
  [[ -f "$manifest" ]] || die "missing manifest: $manifest"
  for mode in "${MS[@]}"; do
    case "$mode" in
      text_off) extra=() ;;
      text_on)  extra=(--text-history) ;;
      *) die "unknown MODE: $mode" ;;
    esac
    gpu="${GPUS[$((job % ${#GPUS[@]}))]}"
    launch "$gpu" "${ds}_${mode}" "$manifest" "$OUT_ROOT/${ds}_${mode}" "${extra[@]}"
    job=$((job + 1))
  done
done

bad=0
for i in "${!PIDS[@]}"; do
  if wait "${PIDS[$i]}"; then
    log "finished: ${NAMES[$i]}"
  else
    log "FAILED: ${NAMES[$i]} (see $LOG_DIR/${NAMES[$i]}.log)"
    bad=1
  fi
done
trap - EXIT INT TERM
[[ "$bad" -eq 0 ]] || exit 1

python - "$OUT_ROOT" "${DS[@]}" -- "${MS[@]}" <<'PY' | tee "$OUT_ROOT/summary.tsv"
import json, pathlib, sys
root = pathlib.Path(sys.argv[1])
sep = sys.argv.index('--')
datasets = sys.argv[2:sep]
modes = sys.argv[sep+1:]
print("backbone\tdataset\tmode\tnormalized_micro_wer\tnormalized_macro_call_wer\trtf")
for ds in datasets:
    for mode in modes:
        p = root / f"{ds}_{mode}" / "metrics.json"
        if not p.exists():
            continue
        m = json.loads(p.read_text())
        print("whisper-large-v3\t{}\t{}\t{}\t{}\t{}".format(
            ds, mode,
            m.get("normalized_micro_wer", "NA"),
            m.get("normalized_macro_call_wer", "NA"),
            m.get("rtf", "NA"),
        ))
PY

log "done. Summary: $OUT_ROOT/summary.tsv"
