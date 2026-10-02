# CSA-Adapter v0.2.1 stability patch

## Why this patch exists

The v0.2 H100 priority run exposed a specific long-form failure mode:

- the first 30 s chunk decoded normally;
- once historical acoustic memory was available, some checkpoints emitted empty hypotheses;
- teacher-forced training NLL remained healthy, so training loss alone did not detect the failure;
- the old dense-to-sparse warm-up was also much less stable than one hard-sparse seed;
- text-history-on evaluation was substantially worse than text-history-off and therefore should not be used for model selection.

v0.2.1 does not hide empty hypotheses or silently fall back to the backbone.  It adds stability controls and diagnostics so failures remain scientifically visible.

## Code changes

### 1. Bounded acoustic residual

`PersistentCSA.alpha` remains a learned scalar parameter, but the applied scalar is

```text
alpha_eff = alpha_max * tanh(alpha_raw / alpha_max)
```

with defaults `alpha_init=0.01` and `alpha_max=0.10`.

### 2. Negative gate bias at initialization

The diagonal gate now starts with `gate_bias_init=-2.0` instead of 0, so the initial gate is about 0.119 rather than 0.5.  The adapter therefore starts closer to the frozen Whisper representation while retaining nonzero gradients.

### 3. Per-token residual RMS safety cap

The applied residual is limited to a fraction of the frozen hidden-state RMS:

```text
RMS(delta_t) <= residual_ratio_cap * RMS(H_t)
```

Default: `residual_ratio_cap=0.25`.

The cap scale is detached: it is a safety envelope, not a trainable normalization path.

### 4. Smaller output-projection initialization

The historical readout projection now uses the same small-normal style as the lightweight low-rank projections instead of PyTorch's much larger default Linear initialization.

### 5. Segment-level diagnostics

With `--diagnostics`, `segments.jsonl` contains:

- memory entries before/after the segment;
- raw/effective alpha;
- gate mean/max and saturation fraction;
- base/readout/delta RMS;
- residual ratio before/after the cap;
- fraction of frames/tokens clipped by the safety cap;
- cosine between frozen and adapted encoder states;
- retrieval entropy;
- prompt token count and actual prompt text.

`metrics.json` aggregates the main diagnostics.

### 6. Long-form checkpoint selection

Training can save every validation checkpoint with `--save-validation-checkpoints`.
The H100 priority script evaluates every saved checkpoint on `validation.full.jsonl` with text history OFF, selects a checkpoint within each run by long-form WER-N, and only then selects the seed within each family.

This removes a v0.2 weakness: teacher-forced validation NLL was not sufficient to detect free-running long-form collapse.

### 7. Hard sparse routing is the new default

`--warmup-steps` now defaults to `0`.
The old 100-step dense-to-sparse warm-up remains available as the `warm100` ablation.

### 8. Text history is explicitly secondary

Text history is not used for model/checkpoint selection.  When enabled, the default prompt is only the immediately previous decoded chunk and is bounded to 64 text tokens.  Rolling history remains available via `--text-history-mode rolling`.

### 9. Checkpoint format bump

v0.2.1 writes `persistent-csa-v0.2.1`.  Old v0.2 adapter checkpoints are intentionally rejected because the residual semantics changed.  Reuse the frozen feature caches, but retrain adapters.

## Recommended H100 sequence

Existing v0.2 feature caches are reusable because the frozen Whisper encoder and cache format did not change.

First run the inexpensive stability screen:

```bash
cd /workspace/asr/csa-adapter-v0.2.1
export PYTHONPATH=$PWD/src:$PYTHONPATH
GPUS=0,1,2,3,4,5,6,7 \
  bash scripts/run_h100_v021_stability_8gpu.sh
```

Inspect:

```bash
column -t -s $'\t' runs/h100_v021_stability/stability_summary.tsv
```

Then run the full priority experiment:

```bash
GPUS=0,1,2,3,4,5,6,7 \
  bash scripts/run_h100_v021_priority_8gpu.sh
```

Important outputs:

```text
runs/h100_v021_priority/selected_runs.tsv
runs/h100_v021_priority/results_summary.tsv
runs/h100_v021_priority/*/selected_checkpoint.json
runs/h100_v021_priority/test/*/metrics.json
runs/h100_v021_priority/e21/*/metrics.json
```

For the publication run, keep `CHECKPOINT_SELECT_MAX_CALLS=0` (all validation calls).  For a quick engineering pass, for example:

```bash
CHECKPOINT_SELECT_MAX_CALLS=5 MAX_STEPS=500 \
GPUS=0,1,2,3,4,5,6,7 \
  RUN_ROOT=runs/h100_v021_quick \
  bash scripts/run_h100_v021_priority_8gpu.sh
```

## What to look for

A healthy run should have:

- very few or zero `empty_hypothesis_chunks`;
- `diag_mean_residual_ratio` comfortably below the configured 0.25 cap;
- `diag_mean_adapted_cosine` close to 1;
- low `diag_mean_residual_clip_fraction` after training stabilizes;
- similar behavior across seeds rather than one seed near 0.99 WER-N and another near normal ASR WER.

If clipping is active on most frames, do not simply increase the cap.  Inspect alpha, gate, retrieval entropy, and checkpoint evolution first.

## Packaging hotfix: force the current checkout on PYTHONPATH

The original v0.2.1 H100 shell scripts could accidentally resolve an older
`csa_adapter` already installed in site-packages.  This presented as
`train.py: error: unrecognized arguments: --alpha-init ...` even though the
v0.2.1 source file contained those flags.

All repository shell scripts now source `scripts/_repo_env.sh`, which prepends
`$REPO/src` to `PYTHONPATH`.  The two H100 entrypoints also run
`scripts/preflight_v021.py` before launching GPU jobs; it fails immediately if
Python would resolve a different checkout or if the expected v0.2.1 CLI flags
are absent.
