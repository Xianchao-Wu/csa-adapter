# CSA-Adapter v0.2.1 — persistent acoustic memory for segmented Whisper ASR

> **v0.2.1 stability patch.** This version adds bounded residual strength, a per-token residual RMS safety cap, negative gate initialization, segment-level diagnostics, and long-form checkpoint selection. Existing v0.2 frozen-encoder feature caches can be reused, but v0.2 adapter checkpoints must be retrained. See `PATCH_NOTES_v0.2.1.md`.

Research prototype: frozen Whisper + trainable compressed historical acoustic memory.
This release implements the **cross-segment** design discussed in the CSA-Adapter draft.
It is not a pretrained model, a validated accuracy claim, or an implementation of
DeepSeek's attention kernels. No teacher model or FinASR-Bench is required.

## Quick start (Linux / NVIDIA GPU)

Keep the CUDA PyTorch in your NVIDIA container; do not install a CPU wheel over it.
Python 3.10+; a BF16-capable GPU (A100/H100 recommended); internet access to Hugging Face.
Dependencies are pinned to Transformers 4.57.6 and Datasets 3.6.0 to stabilize APIs.
Audio uses SoundFile/libsndfile and SciPy, not torchaudio or torchcodec. A recent
SoundFile wheel includes MP3 decoding; install system libsndfile/ffmpeg if your custom
build cannot decode the source MP3s. FFmpeg is not automatically called by this code.

```bash
unzip csa-adapter-v0.2.1.zip
cd csa-adapter-v0.2.1
bash scripts/00_setup.sh
bash scripts/01_prepare_earnings22.sh
GPUS=0,1,2,3,4,5,6,7 bash scripts/02_cache_features.sh
CUDA_VISIBLE_DEVICES=0 bash scripts/07_smoke_train.sh
CUDA_VISIBLE_DEVICES=0 bash scripts/03_train.sh
CUDA_VISIBLE_DEVICES=0 bash scripts/04_eval_matrix.sh
CUDA_VISIBLE_DEVICES=0 bash scripts/05_eval_earnings21.sh
```

`02` defaults to one GPU; multiple GPUs independently cache disjoint shards.
`03` trains on **one GPU**, microbatch 1 and gradient accumulation 8. It is not DDP.
Start with the smoke train before launching a multi-hour run. Cache preparation can
resume on matching source/model metadata; training does **not** resume optimizer state.
Use a new RUN_DIR for each training run. Evaluation also refuses to overwrite an
existing calls.jsonl. Output files are created locally on your machine.

To use a smaller backbone for the first engineering run, consistently export:

```bash
export MODEL=openai/whisper-small
export CACHE_DIR=cache/earnings22_small
export RUN_DIR=runs/earnings22_small_csa
```

Do this before caching, training and evaluation. If `REVISION` is set it is passed to
all model operations; checkpoints/cache metadata reject mismatched backbone revisions.
Do not mix caches from different models or change cache sharding when resuming.
A local model path is also supported; its contents must remain unchanged.

## Data protocols

| Recipe | Train | Validation | Test |
|---|---|---|---|
| Default E22 6:2:2 | 75 E22 calls | 25 E22 calls | 25 E22 calls, full audio |
| External evaluation after default training | same 75 E22 | same 25 E22 | all 44 E21 calls, full audio |
| Alternative E22 8:2 → E21 | 100 E22 calls | 25 E22 calls | all 44 E21 calls, full audio |

Counts assume the public 125-call E22 and 44-call E21 snapshots. Grouping is by
`file_id`, **never random segments**. Ratios are by call count, not duration.
The seeded permutation operates on sorted call IDs; split_calls.json is authoritative.
These are **custom adaptation splits**. E22 originally provides a test-only corpus;
our 25-call result must not be reported as its official full-test benchmark.
Call-disjoint does not guarantee company-, speaker- or accent-disjoint data. E21 is
an external corpus test, not a guaranteed unseen-company or unseen-speaker test.
Possible pretraining exposure of Whisper is not ruled out by this protocol.

Alternative executable recipe (new prepared data/cache/run directories):

```bash
GPUS=0,1,2,3,4,5,6,7 CUDA_VISIBLE_DEVICES=0 \
  bash scripts/06_earnings22_80_20_to_earnings21.sh
```

The default preparation downloads E22 full audio and optional E21 full audio. It
projects only alignment/text columns from the E22 chunked Parquet dataset instead
of decoding its much larger audio column. Backend range/prefetch behavior can still
increase network transfer. Training uses original full-recording waveform slices,
packed along consecutive aligned segments to at most 28 seconds; no packing across
calls, omitted source segments, overlaps, or gaps greater than one second.

An individual aligned segment longer than 28 seconds cannot be split safely without
word timestamps. The recipe explicitly uses `--oversize skip` for these **training/
segmented-validation** records. The default Python API is strict (`--oversize error`).
`excluded_segments.jsonl` and `preparation_report.json` report exclusions and coverage.
Full-call evaluation loses **no source audio** and uses the full reference transcript.
Alignment errors remain a dataset limitation. Empty or inconsistent source timestamps
are reported; overlapping source spans are excluded and reported. A final temporal
audit rejects any overlap remaining in the prepared segments.

Prepared files:

- train.jsonl / validation.jsonl / test.jsonl: chronological packed segment records.
- train.full.jsonl / validation.full.jsonl / test.full.jsonl: original full-call audio.
- earnings21.full.jsonl: external full-call evaluation.
- split_calls.json: exact recording IDs; preparation_report.json: provenance and hours.

No datasets or model weights are bundled. Consult upstream dataset terms before use;
repository code license does not relicense audio or transcripts. Sources:
[E22 dataset card](https://huggingface.co/datasets/distil-whisper/earnings22),
[E21 dataset card](https://huggingface.co/datasets/distil-whisper/earnings21),
[original Rev repository](https://github.com/revdotcom/speech-datasets),
[Earnings-22 paper](https://arxiv.org/abs/2203.15591).

## Architecture and gradients

`src/csa_adapter/longform/memory.py` implements one adapter **after the final frozen
encoder output**, before the frozen decoder. The backbone keeps its <=30-second
acoustic window. The adapter's queries see only previously committed segments.

1. Encode current audio independently; retain the valid frame count.
2. Query history, select exact top-k memory entries, read their projected values.
3. Apply output projection and gate, add the scaled correction to valid frames only.
4. Decode from the adapted encoder output.
5. After successful decoding, compress the **unadapted** current encoder states and
   commit them to this call's FIFO memory. Padding is never stored.
6. Explicitly reset at a call boundary. Duplicate, overlapping and out-of-order
   commits raise errors. Forward does not mutate memory, so retries cannot write twice.

The mean and event branches use independent W_c and W_e. A single learned scoring
vector u is shared across windows. The default gate is `sigmoid(w_g * LN(H) + b_g)`
(elementwise, 2D parameters), instead of the earlier dense D-by-D gate. The diagonal gate starts with bias -2.0 and the effective alpha starts at 0.01.
The effective alpha is smoothly bounded and the applied residual has a per-token RMS
safety cap; empty history still yields an exact identity. No time-distance bias or content-aware memory eviction is included.

Training recompresses historical **detached raw encoder features**, using the current
writer parameters on every step. Compressed Z is not detached. Frozen decoder forward
is not under no_grad, so current ASR loss reaches the query/key/value projections,
writer, layer norm and gate. The encoder itself is cached and never backpropagated.
Only the current segment's labels supervise an example; historical labels are not
inputs. There is no backpropagation through prior decoding or optimizer steps.

The default history is the previous 64 retained segments, capped at 4096 memory
entries. At 50 encoder frames/s and C=8, 4096 entries cover approximately 10.9 minutes
of continuously stored speech; packing gaps and per-segment ceiling alter this.
Training history can be shorter than inference retention; increase --history-segments
or reduce --max-memory when testing matched history spans. It does not preserve an
entire multi-hour call once FIFO eviction occurs. Longer retention requires a capacity
sweep or a future hierarchical retention policy.

Exact search scores all stored compressed keys in tiles under no_grad, then recomputes
selected scores differentiably. This avoids retaining every search tile in the
training graph; it does not eliminate O(T*L*d_i) scoring. Selected key/value tensors
still consume O(T*k*d) storage. Dense warm-up has higher memory cost and uses all
current history entries. Start with --history-segments 8 --max-memory 1024 if necessary.

## Training, model selection and ablations

Best checkpoint is selected on a fixed seeded subset of 256 validation segments using
**token-weighted teacher-forced NLL**, with hard routing (except --dense-always).
This uses reference transcriptions as normal ASR targets, **not a teacher model**.
Use `--valid-examples 0` for the full eligible segmented validation set. NLL selection
is not identical to selecting long-form WER. For publication, compare candidates on
validation.full.jsonl using the target decoding protocol, then evaluate test once.
Never choose hyperparameters/checkpoints on the test set.

```bash
# Validate the selected checkpoint on full recordings before final testing.
MANIFEST=data/earnings22_622/validation.full.jsonl \
EVAL_DIR=runs/earnings22_csa/eval_validation bash scripts/04_eval_matrix.sh

# Separate training runs: hard top-k, warm-up then top-k, always dense.
bash scripts/08_compare_routing.sh

# Compressor/gate ablations: separate run, same data split and seed.
RUN_DIR=runs/mean_csa bash scripts/03_train.sh --compressor mean
RUN_DIR=runs/no_gate_csa bash scripts/03_train.sh --gate none
```

`best_adapter/` and `last_adapter/` contain safetensors weights and configuration only;
no backbone, optimizer, or session memory. Scripts record source/model revisions,
parameter counts, training logs, validation examples and decoding options.
New v0.2 persistent checkpoints are intentionally incompatible with v0.1 layer-patch
checkpoints. Retrain this architecture rather than silently loading unmatched keys.

## Full-call evaluation

The evaluator reads every sample of each full waveform, with nonoverlapping fixed
30-second windows (last window is padded only for encoder input). It is a controlled
segmented-ASR protocol, **not** the native Whisper timestamp-seek/fallback implementation.
It has no VAD, word-boundary overlap stitching, or temperature fallback. Boundary
errors affect every compared system under the same segmentation.

The matrix includes baseline text-history off/on and CSA history text-history off/on.
Text history is a secondary inference ablation. By default it uses at most 64 tokens from the **immediately previous predicted chunk** as a bounded prompt; `--text-history-mode rolling` restores bounded accumulated history.
This is not a byte-for-byte reproduction of native condition_on_previous_text reset
semantics. It is disabled during training and evaluated as an inference ablation.
The reset-memory ablation is an exact identity for this historical-only adapter;
it tests dependence on memory but is not a separately trained local adapter baseline.
`--memory-mode local` is available only as an inference diagnostic, not a trained
local-CSA baseline. No LoRA baseline is claimed to be implemented in these new recipes.

Outputs: calls.jsonl (references, hypotheses, errors, time, memory), segments.jsonl
(time spans and text), metrics.json (English-normalized micro WER, macro call WER,
raw macro call WER, RTF). `possible_token_cap_chunks` flags generation without a final
EOS; inspect it before reporting WER. Increase --max-new-tokens within the model's
448-position decoder budget, accounting for prompts and language/task tokens.

Timing excludes file decode/model load, includes encoding, memory operations and
text generation; per-call peak GPU allocated bytes include model weights. No speedup
over the original Whisper is promised: CSA adds cross-segment computation. Compare
against dense historical reading for sparse-memory efficiency claims.

For retrieval-specific numeric/entity metrics, annotate targets or add a separate
scorer; ordinary WER does not prove correct entity retrieval. FER is intentionally
not required in this release. Fixed-window ASR is segment-causal, not low-latency
frame-causal streaming within a 30-second window.

## Storage and practical defaults

The cache stores full 1500-position final encoder outputs in FP16, including padded
positions for decoder compatibility, plus valid lengths for memory. Whisper-large-v3
uses about 3.84 MB per packed segment (1500*1280*2 bytes), plus small metadata overhead.
For 10,000 packed segments budget roughly 38.4 GB. Exact total is reported by the
number of prepared segments; allocate substantial local NVMe space. The same frozen
cache can be reused across compression/rank/gate/routing ablations with the same model.
The CPU feature LRU defaults to 128 segments (~0.5 GB at large-v3); it is configurable.

## GitHub

The ZIP has one top-level csa-adapter-v0.2.1 directory and includes tests, scripts,
pyproject.toml, CI and .gitignore. Data, caches, outputs and model weights are ignored.

```bash
git init
git add .
git commit -m "Add persistent acoustic memory CSA adapter and Earnings recipes"
git branch -M main
git remote add origin git@github.com:YOUR_USER/csa-adapter.git
git push -u origin main
```

See VALIDATION.md for what was actually tested. Legacy layer-patch utilities and the
generic dataset registry remain for reference; legacy CLI execution is disabled to
avoid the old >30-second truncation path. Use the new commands/scripts above.