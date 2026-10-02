# Quick frozen baselines for CSA-Adapter

## What counts as a baseline?

In the paper, every `Frozen` row is a base-model baseline: all pretrained ASR
weights are frozen and no CSA/LoRA parameters are trained.

For Whisper there are two useful frozen conditions:

- `Frozen, text off`: no prior transcript prompt.
- `Frozen, text on`: predicted text history is supplied by the existing long-form evaluator.

For FastConformer the initial frozen baseline has no Whisper-style decoder text-history row.
Use the hybrid model's RNN-T branch by default; optionally report CTC as an auxiliary decoder control.

## 1. Whisper-large-v3 baseline

Copy the runner into the repository root and run:

```bash
bash run_whisper_frozen_baselines.sh
```

Default input directory:

```text
data/earnings22_622/
  validation.full.jsonl
  test.full.jsonl
  earnings21.full.jsonl
```

Default outputs:

```text
runs/frozen_baselines/whisper_large_v3/
  validation_text_off/metrics.json
  validation_text_on/metrics.json
  test_text_off/metrics.json
  test_text_on/metrics.json
  e21_text_off/metrics.json
  e21_text_on/metrics.json
  summary.tsv
```

This is intentionally independent of CSA training and does **not** require
`selected_runs.tsv` or any adapter checkpoint.

## 2. FastConformer baseline

Place the Python evaluator at:

```text
scripts/eval_fastconformer_frozen.py
```

and the shell runner in the repository root, then run:

```bash
bash run_fastconformer_frozen_baselines.sh
```

Recommended environment is an NVIDIA NeMo/Speech container with NeMo ASR and
PyTorch available. The default model is:

```text
nvidia/stt_en_fastconformer_hybrid_large_pc
```

The runner prefers `validation.jsonl` / `test.jsonl` (bounded segments) if they
exist, because this is safer and faster than passing whole earnings calls to
NeMo. For E21 it searches `earnings21.jsonl`,
`earnings21.segments.jsonl`, then `earnings21.full.jsonl`.

Default decoder is RNN-T. To run CTC too:

```bash
DECODER=ctc GPU_LIST=3,4,5 bash run_fastconformer_frozen_baselines.sh
```

## 3. First smoke test

Before committing all calls, test FastConformer on a few entries:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/eval_fastconformer_frozen.py \
  --manifest data/earnings22_622/test.jsonl \
  --output runs/smoke_fastconformer \
  --batch-size 8 \
  --max-items 16
```

If that succeeds, run the full wrapper.

## Metric note

Whisper's existing CSA evaluator remains the canonical scorer for Whisper and
produces `normalized_micro_wer`, `normalized_macro_call_wer`, and `rtf`.
The FastConformer script uses the same metric names and a documented
model-independent English normalization. Before final submission, use one
shared normalizer for predictions from all backbones if the repository's exact
Whisper normalizer differs; the prediction JSONL is saved so rescoring does not
require rerunning ASR inference.
