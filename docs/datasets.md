# Legacy dataset registry (v0.1 reference)

These generic registry entries are not the validated v0.2 persistent-memory recipes.
Use the Earnings-22/Earnings-21 preparation and scripts in the root README.
Other datasets need call IDs, ordered timestamps, and aligned segment labels before
use with historical-memory training. No automatic schema validation is claimed here.

# Dataset strategy

CSA-Adapter needs several dataset types because no single benchmark validates
domain adaptation, long-range retrieval, source-domain retention, and agentic
speech recognition at the same time.

## Tier 1: core claim

### FinASR-Bench

Use for multilingual financial adaptation and report WER/CER, normalized error,
SemDist, and FER. Its controlled financial entities and facts directly measure
the target domain, but synthetic data alone is not sufficient evidence for
real long-form robustness.

### Earnings-21 and Earnings-22

- <https://huggingface.co/datasets/distil-whisper/earnings21>
- <https://huggingface.co/datasets/distil-whisper/earnings22>

These are the most important external tests. Earnings-22 contains approximately
119 hours across 125 full English earnings calls and speakers from multiple
accent regions. Use the chunked configuration for ordinary Whisper evaluation;
use whole calls or reconstructed ordered chunks for a persistent-memory model.

### AMI Meeting Corpus

- <https://huggingface.co/datasets/edinburghcstr/ami>

AMI contains roughly 100 hours of meetings with close-talking and far-field
microphone conditions. The Hugging Face version is already segmented. Group by
`meeting_id` and sort by `begin_time` when constructing controlled long-form
sequences. Avoid treating every IHM channel as a single conversation stream;
SDM is the cleaner first long-form condition.

## Tier 2: generalization and scale

### SPGISpeech

- <https://huggingface.co/datasets/kensho/spgispeech>

SPGISpeech contains about 5,000 hours of professionally transcribed financial
speech, but samples are short slices. It is excellent for financial adaptation
and cross-corpus transfer, not a standalone long-range retrieval test. Access
is gated and its terms must be accepted before use.

### GigaSpeech

- <https://huggingface.co/datasets/speechcolab/gigaspeech>

GigaSpeech covers audiobooks, podcasts, and online videos. Start with the XS
configuration. Access conditions apply. Its `audio_id`, `begin_time`, and
`end_time` fields make it possible to group ordered segments from a common
source, although the hosted audio samples are segmented.

### TED-LIUM 3

- <https://huggingface.co/datasets/LIUM/tedlium>

Use TED talks as a clean discourse-level transfer test. It is less
domain-specific than financial calls, making it useful for determining whether
CSA learns a general long-context mechanism.

## Tier 3: agentic and downstream evaluation

### SLURP

- Paper: <https://aclanthology.org/2020.emnlp-main.588/>
- Dataset: <https://huggingface.co/datasets/qmeeus/slurp>

SLURP spans 18 voice-assistant domains and provides scenarios, actions, and
entities. Report transcription WER plus exact entity/action accuracy. For the
dual-bank experiment, construct a context bank from the active tool schema and
candidate entities, and include difficult irrelevant candidates.

### MInDS-14

- <https://huggingface.co/datasets/PolyAI/minds14>

MInDS-14 contains banking-related spoken intents in multiple language varieties.
It is a good bridge between financial ASR and agent routing. It is short-form,
so it evaluates context alignment rather than long-range acoustic retrieval.

### SLUE

- <https://huggingface.co/datasets/asapp/slue>

SLUE connects ASR with spoken named-entity recognition and sentiment. Use it to
test whether lower WER from CSA improves downstream semantics instead of merely
changing surface forms.

## Mandatory control sets

- LibriSpeech test-clean and test-other: general-domain retention.
- Common Voice: accent and recording-condition robustness.
- FLEURS: multilingual regression checks under a shared evaluation design.

## Recommended reporting matrix

| Claim | Train | Test | Primary metrics |
|---|---|---|---|
| Fast financial adaptation | FinASR train | FinASR held-out | WER/CER, FER, GPU-hours |
| Real financial transfer | FinASR synthetic | Earnings-21/22 | WER, entity/number error |
| Cross-corpus financial transfer | SPGISpeech or FinASR | Earnings-22 + FinASR real | WER, FER |
| Long-range retrieval | controlled long calls | 30 s/1 m/3 m/5+ m sets | FER vs distance, recall@k |
| Meeting robustness | AMI train/dev | AMI IHM/SDM | WER, RTF, memory |
| Agent-context alignment | SLURP/MInDS-14 | clean, distractor, stale context | entity/action accuracy, false bias rate |
| Retention | financial adaptation only | LibriSpeech/Common Voice | WER degradation |

Dataset cards and licenses remain authoritative. Registry entries are
convenience defaults, not a substitute for checking upstream access terms.

