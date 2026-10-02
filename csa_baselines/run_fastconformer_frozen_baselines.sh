#!/usr/bin/env bash
set -Eeuo pipefail

# Frozen FastConformer baselines for Earnings-22/Earnings-21.
# Run from csa-adapter repo root after data preparation.
# Uses segment manifests when present because they are much safer/faster than
# feeding hour-long calls directly to NeMo. Falls back to *.full.jsonl.

MODEL="${MODEL:-nvidia/stt_en_fastconformer_hybrid_large_pc}"
DATA_DIR="${DATA_DIR:-data/earnings22_622}"
OUT_ROOT="${OUT_ROOT:-runs/frozen_baselines/fastconformer_hybrid_large_pc}"
LOG_DIR="${LOG_DIR:-logs/frozen_baselines/fastconformer_hybrid_large_pc}"
GPU_LIST="${GPU_LIST:-0,1,2}"
DATASETS="${DATASETS:-validation,test,e21}"
DECODER="${DECODER:-rnnt}"
BATCH_SIZE="${BATCH_SIZE:-32}"
NUM_WORKERS="${NUM_WORKERS:-4}"
PY_SCRIPT="${PY_SCRIPT:-scripts/eval_fastconformer_frozen.py}"

mkdir -p "$OUT_ROOT" "$LOG_DIR"
log() { printf '[%(%F %T)T] %s\n' -1 "$*"; }
die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

[[ -f "$PY_SCRIPT" ]] || die "missing $PY_SCRIPT; copy eval_fastconformer_frozen.py to scripts/ first"
IFS=',' read -r -a GPUS <<< "$GPU_LIST"
[[ ${#GPUS[@]} -ge 1 ]] || die "GPU_LIST is empty"

pick_manifest() {
  case "$1" in
    validation)
      [[ -f "$DATA_DIR/validation.jsonl" ]] && { echo "$DATA_DIR/validation.jsonl"; return; }
      echo "$DATA_DIR/validation.full.jsonl" ;;
    test)
      [[ -f "$DATA_DIR/test.jsonl" ]] && { echo "$DATA_DIR/test.jsonl"; return; }
      echo "$DATA_DIR/test.full.jsonl" ;;
    e21)
      # Use a prepared segmented E21 manifest if one exists; otherwise the
      # Python runner can expand common nested full-call manifest formats.
      for p in "$DATA_DIR/earnings21.jsonl" "$DATA_DIR/earnings21.segments.jsonl" "$DATA_DIR/earnings21.full.jsonl"; do
        [[ -f "$p" ]] && { echo "$p"; return; }
      done
      echo "$DATA_DIR/earnings21.full.jsonl" ;;
    *) die "unknown dataset: $1" ;;
  esac
}

declare -a PIDS=() NAMES=()
cleanup() { local p; for p in "${PIDS[@]:-}"; do kill "$p" 2>/dev/null || true; done; }
trap cleanup EXIT INT TERM

IFS=',' read -r -a DS <<< "$DATASETS"
for i in "${!DS[@]}"; do
  ds="${DS[$i]}"
  manifest="$(pick_manifest "$ds")"
  [[ -f "$manifest" ]] || die "missing manifest for $ds: $manifest"
  out="$OUT_ROOT/${ds}_${DECODER}"
  if [[ -f "$out/metrics.json" ]]; then
    log "skip completed: $ds"
    continue
  fi
  gpu="${GPUS[$((i % ${#GPUS[@]}))]}"
  log "launch GPU $gpu: FastConformer $ds ($DECODER), manifest=$manifest"
  CUDA_VISIBLE_DEVICES="$gpu" python "$PY_SCRIPT" \
    --manifest "$manifest" \
    --output "$out" \
    --model "$MODEL" \
    --decoder "$DECODER" \
    --batch-size "$BATCH_SIZE" \
    --num-workers "$NUM_WORKERS" \
    >"$LOG_DIR/${ds}_${DECODER}.log" 2>&1 &
  PIDS+=("$!")
  NAMES+=("$ds")
done

bad=0
for i in "${!PIDS[@]}"; do
  if wait "${PIDS[$i]}"; then log "finished: ${NAMES[$i]}";
  else log "FAILED: ${NAMES[$i]} (see log)"; bad=1; fi
done
trap - EXIT INT TERM
[[ "$bad" -eq 0 ]] || exit 1

python - "$OUT_ROOT" "$DECODER" "${DS[@]}" <<'PY' | tee "$OUT_ROOT/summary.tsv"
import json, pathlib, sys
root = pathlib.Path(sys.argv[1]); decoder = sys.argv[2]; datasets = sys.argv[3:]
print("backbone\tdataset\tdecoder\tnormalized_micro_wer\tnormalized_macro_call_wer\trtf\tpeak_gpu_memory_gb")
for ds in datasets:
    p = root / f"{ds}_{decoder}" / "metrics.json"
    if not p.exists(): continue
    m = json.loads(p.read_text())
    print("fastconformer\t{}\t{}\t{}\t{}\t{}\t{}".format(
        ds, decoder, m.get("normalized_micro_wer","NA"),
        m.get("normalized_macro_call_wer","NA"), m.get("rtf","NA"),
        m.get("peak_gpu_memory_gb","NA")))
PY

log "done. Summary: $OUT_ROOT/summary.tsv"
