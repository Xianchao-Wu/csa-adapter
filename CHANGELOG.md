# v0.2.1

- Bound learned persistent-memory residual strength with a smooth `tanh` parameterization.
- Add configurable per-token residual RMS cap to prevent historical memory from destroying frozen Whisper states.
- Initialize the diagonal gate with a negative bias and use a smaller output-projection initialization.
- Add per-segment memory/residual/gate/cosine/retrieval diagnostics and aggregate metrics.
- Make hard sparse routing (`warmup_steps=0`) the stable default; retain dense-to-sparse warm-up as an ablation.
- Save optional periodic validation checkpoints and select checkpoints by long-form validation WER in the new H100 workflow.
- Keep text history out of model selection; default text prompting uses only the immediately previous chunk and 64 tokens.
- Add 8xH100 stability and priority experiment scripts plus a result summarizer.
- Bump checkpoint format to `persistent-csa-v0.2.1`; v0.2 adapter checkpoints require retraining, while frozen feature caches remain reusable.

# v0.2

- Historical-only adapter after final Whisper encoder output, explicit per-call FIFO state.
- Training recomputes differentiable memory from frozen raw-feature caches.
- Exact tiled no-grad top-k plus differentiable selected-score recomputation.
- Dense routing warm-up, independent mean/event projections, diagonal lightweight gate.
- Full-call E22 custom call-disjoint 6:2:2 and 8:2→E21 preparation and executable recipes.
- Reference-independent full-audio evaluation, text-history ablations, WER/RTF/provenance logs.
- No silent long-audio/label truncation in the supported pipeline.
- Legacy v0.1 CLI disabled. Event checkpoints changed shape; persistent v0.2 requires retraining.
