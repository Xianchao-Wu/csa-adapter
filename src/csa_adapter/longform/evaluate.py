"""Full-call, reference-independent fixed-window decoding with explicit memory.

This is a controlled segmented ASR protocol, not Whisper's timestamp-seek loop.
No ground-truth segmentation/text is read by the decoding loop.
"""

import argparse
import json
import time
from pathlib import Path

import jiwer
import numpy as np
import torch
from transformers.modeling_outputs import BaseModelOutput
from transformers.models.whisper.english_normalizer import EnglishTextNormalizer

from .data import read_jsonl, write_jsonl
from .memory import MemoryBank
from .runtime import adapt_valid, amp, encode, load_adapter, load_backbone, model_identity, read_wav


def windows(audio, seconds):
    size = round(seconds * 16000)
    if not 0 < size <= 30 * 16000:
        raise ValueError("chunk-seconds must be in (0,30]")
    for start in range(0, len(audio), size):
        end = min(len(audio), start + size)
        yield start / 16000, end / 16000, audio[start:end]


def summarize(rows):
    counts = {
        k: sum(r[k] for r in rows) for k in ("substitutions", "deletions", "insertions", "hits")
    }
    denom = counts["substitutions"] + counts["deletions"] + counts["hits"]
    return dict(
        calls=len(rows),
        normalized_micro_wer=(counts["substitutions"] + counts["deletions"] + counts["insertions"])
        / max(1, denom),
        normalized_macro_call_wer=float(np.mean([r["wer_normalized"] for r in rows])),
        raw_macro_call_wer=float(np.mean([r["wer_raw"] for r in rows])),
        rtf=sum(r["decode_seconds"] for r in rows) / sum(r["audio_seconds"] for r in rows),
        **counts,
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", required=True, help="*.full.jsonl, one complete call per row")
    p.add_argument("--output", required=True)
    p.add_argument("--model", default="openai/whisper-large-v3")
    p.add_argument("--revision", default="main")
    p.add_argument("--adapter")
    p.add_argument("--device", default="cuda")
    p.add_argument("--chunk-seconds", type=float, default=30.0)
    p.add_argument("--text-history", action="store_true")
    p.add_argument("--prompt-tokens", type=int, default=128)
    p.add_argument("--memory-mode", choices=["history", "reset", "local"], default="history")
    p.add_argument("--dense-reading", action="store_true")
    p.add_argument("--max-calls", type=int, default=0)
    p.add_argument("--num-beams", type=int, default=1)
    p.add_argument("--max-new-tokens", type=int, default=256)
    args = p.parse_args()
    if not 0 < args.chunk_seconds <= 30 or not 1 <= args.prompt_tokens <= 160:
        p.error("chunk-seconds in (0,30], prompt-tokens in [1,160]")
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "calls.jsonl").exists():
        raise FileExistsError("Evaluation exists; choose a new output directory")
    model, proc = load_backbone(args.model, args.device, args.revision)
    identity = model_identity(model, args.model)
    adapter = load_adapter(args.adapter, identity, args.device) if args.adapter else None
    normalizer = EnglishTextNormalizer(proc.tokenizer.english_spelling_normalizer)
    records = read_jsonl(args.manifest)
    if args.max_calls:
        records = records[: args.max_calls]
    if not records or len({r["call_id"] for r in records}) != len(records):
        raise ValueError("Need nonempty full-call manifest with unique call_id")
    results, segments = [], []
    (out / "run.json").write_text(
        json.dumps(
            dict(
                vars(args),
                backbone=identity,
                protocol="fixed_nonoverlapping_windows_no_reference_boundaries",
                text_prompt="bounded previous predicted text, not native transcribe prompt-reset logic",
            ),
            indent=2,
        )
    )
    for row in records:
        audio = read_wav(row["audio_filepath"])
        bank = MemoryBank(adapter.cfg.max_memory if adapter else 1)
        bank.reset(row["call_id"])
        hypotheses, prompt_text, cap_hits = [], "", 0
        if args.device.startswith("cuda"):
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        began = time.perf_counter()
        for idx, (start, end, chunk) in enumerate(windows(audio, args.chunk_seconds)):
            with torch.no_grad(), amp(args.device):
                raw, valid = encode(model, proc, chunk, args.device)
                adapted = raw
                if adapter:
                    if args.memory_mode == "history":
                        memory = bank.memory
                    elif args.memory_mode == "local":
                        memory = adapter.compress(raw[:, :valid])
                    else:
                        memory = None
                    adapted = adapt_valid(adapter, raw, valid, memory, dense=args.dense_reading)
                kwargs = dict(
                    encoder_outputs=BaseModelOutput(last_hidden_state=adapted),
                    language="english",
                    task="transcribe",
                    return_timestamps=False,
                    num_beams=args.num_beams,
                    do_sample=False,
                    max_new_tokens=args.max_new_tokens,
                )
                if args.text_history and prompt_text:
                    ids = proc.tokenizer.encode(prompt_text, add_special_tokens=False)[
                        -args.prompt_tokens :
                    ]
                    trimmed = proc.tokenizer.decode(ids)
                    kwargs["prompt_ids"] = proc.get_prompt_ids(trimmed, return_tensors="pt").to(
                        args.device
                    )
                generated = model.generate(**kwargs)
                # Count missing EOS conservatively: these chunks may have hit generation limits.
                eos = model.config.eos_token_id
                cap_hits += int(generated[0, -1].item() != eos)
                text = proc.batch_decode(generated, skip_special_tokens=True)[0].strip()
                # Explicit commit after successful generation; never write adapted states.
                if adapter and args.memory_mode == "history":
                    bank.commit(
                        row["call_id"], str(idx), start, end, adapter.compress(raw[:, :valid])
                    )
            hypotheses.append(text)
            prompt_text = " ".join(hypotheses)
            # Bound retained text state as well as the model prompt.
            prompt_ids = proc.tokenizer.encode(prompt_text, add_special_tokens=False)[
                -args.prompt_tokens :
            ]
            prompt_text = proc.tokenizer.decode(prompt_ids)
            segments.append(
                dict(
                    call_id=row["call_id"],
                    segment_id=idx,
                    start=start,
                    end=end,
                    hypothesis=text,
                    memory_entries=0 if bank.memory is None else bank.memory.shape[1],
                )
            )
        if args.device.startswith("cuda"):
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - began
        hypothesis = " ".join(hypotheses)
        # Reference text is accessed only AFTER all chunks have been decoded.
        reference = row["text"]
        score = jiwer.process_words(normalizer(reference), normalizer(hypothesis))
        result = dict(
            call_id=row["call_id"],
            reference=reference,
            hypothesis=hypothesis,
            wer_normalized=score.wer,
            wer_raw=jiwer.wer(reference, hypothesis),
            substitutions=score.substitutions,
            deletions=score.deletions,
            insertions=score.insertions,
            hits=score.hits,
            decode_seconds=elapsed,
            audio_seconds=len(audio) / 16000,
            chunks=len(hypotheses),
            possible_token_cap_chunks=cap_hits,
            peak_gpu_bytes=torch.cuda.max_memory_allocated()
            if args.device.startswith("cuda")
            else 0,
            memory_entries=0 if bank.memory is None else bank.memory.shape[1],
        )
        results.append(result)
        with open(out / "calls.jsonl", "a") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
        print(
            f"{row['call_id']}: WER-N={score.wer:.4f}, RTF={elapsed / (len(audio) / 16000):.3f}",
            flush=True,
        )
    write_jsonl(out / "segments.jsonl", segments)
    report = summarize(results)
    (out / "metrics.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
