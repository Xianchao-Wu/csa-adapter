# Validation record — v0.2

Executed in the delivery environment (CPU, Python 3.12, torch 2.14.0+cpu,
Transformers 4.57.6, Datasets 3.6.0). CUDA/PyTorch supplied by the user's container
is retained by the setup script; BF16 GPU behavior has not been benchmarked here.

- 18 pytest tests passed, including legacy basic-module tests.
- Exact chunked top-k matches dense ranking; selecting all memories matches dense reading.
- Current ASR loss reaches historical compressor, event scorer, query and key parameters;
  detached historical backbone features and frozen Whisper weights receive no gradients.
- Real Hugging Face Whisper decoder forward/backward and generate exercised on a tiny
  random configuration, including prompt_ids and precomputed encoder_outputs.
- Empty-memory identity, untouched padding, FIFO retention, duplicate/overlap rejection,
  call reset, checkpoint round trip, backbone mismatch and incomplete cache guards tested.
- Five-call synthetic data preparation exercised chronological packing, call-level
  splitting, explicit oversize exclusion reports and preservation of full test waveforms.
- End-to-end CLI smoke run with a tiny random Whisper and the public Whisper tokenizer:
  two 3-segment caches -> two optimizer steps (dense then hard top-k) -> best/last
  safetensors -> full-waveform decoding across three windows with history + text prompts.
  This is a functionality check, NOT an ASR accuracy measurement.
- Ruff checks, Python compilation and Bash syntax checks passed.

Source verification: public E22/E21 dataset cards and API revision metadata were read.
Actual E22 chunked rows were fetched and confirmed to have file_id, segment_id,
transcription, start_ts and end_ts; their storage order was not chronological.
The final synchronous projected Parquet reader fetched actual E22 metadata and
exited normally; local Parquet projection is also unit-tested. Complete 125-call
audio preparation/download has NOT been run end-to-end here. Source loading requires
reachable Hugging Face file/CDN endpoints on the experiment machine.

No full Earnings training, pretrained large-v3 accuracy evaluation, GPU throughput
measurement or comparison against LoRA was performed. No numerical accuracy gains
are claimed. Use scripts/07_smoke_train.sh before the full training run and inspect
preparation_report.json, validation logs and generation token-cap diagnostics.

Known deliberate scope limits: fixed nonoverlapping evaluation windows instead of
native timestamp-seek; single-GPU training; FIFO instead of hierarchical retention;
no optimizer resume; no automatic numeric/entity scoring; no v0.1 checkpoint migration.
