# Validation record — v0.2.1 patch

## Executed in the delivery environment

The delivery container has Python 3.13.5 and PyTorch 2.10.0+cpu.  It does not have network access and does not contain the pinned Transformers/JiWER dependencies, so a real Whisper/H100 end-to-end run cannot be executed here.

The following checks were executed locally on the delivered source tree:

- Python compilation for all `src/`, `tests/`, and Python scripts.
- Bash syntax checks for every shell script.
- Existing dependency-light pytest suite: config/module tests passed.
- Focused executable `PersistentCSA` tests covering:
  - exact identity with empty history;
  - sparse historical reading;
  - differentiable historical writer while frozen raw history remains detached;
  - smooth alpha bound;
  - forced residual-cap activation and numerical bound verification;
  - finite diagnostics;
  - FIFO capacity and call reset.
- Checkpoint-format/save-load logic was reviewed and the format was intentionally bumped to v0.2.1.
- H100 scripts were syntax checked and use eight independent single-GPU jobs rather than DDP.

## Must be validated on the user's H100 machine

The following require the user's installed Transformers 4.57.6, Whisper-large-v3 weights, prepared Earnings manifests/caches, and CUDA/BF16 hardware:

1. `bash scripts/run_h100_v021_stability_8gpu.sh`
2. inspect `stability_summary.tsv` for empty chunks and residual diagnostics;
3. if stable, `bash scripts/run_h100_v021_priority_8gpu.sh`;
4. inspect long-form validation checkpoint selection before using E22/E21 test numbers.

No new accuracy claim is made by this patch before those GPU runs complete.
