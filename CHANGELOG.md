# v0.2

- Historical-only adapter after final Whisper encoder output, explicit per-call FIFO state.
- Training recomputes differentiable memory from frozen raw-feature caches.
- Exact tiled no-grad top-k plus differentiable selected-score recomputation.
- Dense routing warm-up, independent mean/event projections, diagonal lightweight gate.
- Full-call E22 custom call-disjoint 6:2:2 and 8:2→E21 preparation and executable recipes.
- Reference-independent full-audio evaluation, text-history ablations, WER/RTF/provenance logs.
- No silent long-audio/label truncation in the supported pipeline.
- Legacy v0.1 CLI disabled. Event checkpoints changed shape; persistent v0.2 requires retraining.
