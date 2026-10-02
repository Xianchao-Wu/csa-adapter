#!/usr/bin/env python3
"""Frozen FastConformer baseline for CSA-Adapter paper.

Supports ordinary segment manifests and several common long-form manifest shapes.
Outputs predictions.jsonl and metrics.json.

Recommended model:
  nvidia/stt_en_fastconformer_hybrid_large_pc

The script uses the RNN-T branch by default. Set --decoder ctc for the auxiliary
CTC branch. No training is performed.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import string
import tempfile
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--model", default="nvidia/stt_en_fastconformer_hybrid_large_pc")
    p.add_argument("--decoder", choices=["rnnt", "ctc"], default="rnnt")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--device", default="cuda")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--max-items", type=int, default=0,
                   help="Debug only: stop after N flattened utterances (0=all).")
    return p.parse_args()


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"bad JSON at {path}:{lineno}: {e}") from e
    return rows


def first_of(d: Dict[str, Any], names: Sequence[str], default=None):
    for n in names:
        if n in d and d[n] is not None:
            return d[n]
    return default


def clean_call_id(row: Dict[str, Any], audio: str, fallback: str) -> str:
    x = first_of(row, ["call_id", "recording_id", "session_id", "conversation_id", "group_id"])
    if x is not None:
        return str(x)
    if audio:
        return Path(audio).stem
    return fallback


def duration_from_file(path: str) -> Optional[float]:
    try:
        import soundfile as sf
        info = sf.info(path)
        return float(info.frames) / float(info.samplerate)
    except Exception:
        return None


def segment_bounds(seg: Dict[str, Any]) -> Tuple[Optional[float], Optional[float]]:
    start = first_of(seg, ["start", "start_time", "offset", "begin", "begin_time"])
    end = first_of(seg, ["end", "end_time", "stop", "stop_time"])
    dur = first_of(seg, ["duration", "duration_s", "length_seconds"])
    start = None if start is None else float(start)
    if end is not None:
        end = float(end)
    elif start is not None and dur is not None:
        end = start + float(dur)
    return start, end


def materialize_slice(parent_audio: str, start: float, end: float, out_wav: Path) -> Tuple[str, float]:
    import soundfile as sf
    if end <= start:
        raise ValueError(f"invalid slice {start}..{end} for {parent_audio}")
    info = sf.info(parent_audio)
    sr = info.samplerate
    s0 = max(0, int(round(start * sr)))
    s1 = min(info.frames, int(round(end * sr)))
    audio, _ = sf.read(parent_audio, start=s0, stop=s1, dtype="float32", always_2d=False)
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_wav), audio, sr)
    return str(out_wav), (s1 - s0) / float(sr)


def flatten_manifest(rows: List[Dict[str, Any]], temp_root: Path) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    for row_idx, row in enumerate(rows):
        parent_audio = str(first_of(row, ["audio_filepath", "audio_path", "wav_path", "audio"], "") or "")
        parent_call = clean_call_id(row, parent_audio, f"call_{row_idx:05d}")
        segments = row.get("segments")

        if isinstance(segments, list) and segments:
            for seg_idx, seg in enumerate(segments):
                if not isinstance(seg, dict):
                    raise TypeError(f"segments[{seg_idx}] in row {row_idx} is not an object")
                audio = str(first_of(seg, ["audio_filepath", "audio_path", "wav_path", "audio"], "") or "")
                ref = str(first_of(seg, ["text", "transcript", "reference", "ref"], "") or "")
                start, end = segment_bounds(seg)
                dur = first_of(seg, ["duration", "duration_s", "length_seconds"])
                if audio:
                    if dur is None:
                        dur = duration_from_file(audio)
                elif parent_audio and start is not None and end is not None:
                    wav = temp_root / f"row{row_idx:05d}_seg{seg_idx:05d}.wav"
                    audio, dur2 = materialize_slice(parent_audio, start, end, wav)
                    if dur is None:
                        dur = dur2
                else:
                    raise ValueError(
                        f"cannot resolve audio for row {row_idx} segment {seg_idx}; "
                        "need segment audio_filepath or parent audio + start/end"
                    )
                items.append({
                    "audio_filepath": audio,
                    "text": ref,
                    "call_id": str(first_of(seg, ["call_id", "recording_id", "session_id"], parent_call)),
                    "segment_index": int(first_of(seg, ["segment_index", "segment_id", "index"], seg_idx)),
                    "duration": None if dur is None else float(dur),
                    "source_row": row_idx,
                })
        else:
            audio = parent_audio
            if not audio:
                raise ValueError(f"row {row_idx} has no audio filepath")
            ref = str(first_of(row, ["text", "transcript", "reference", "ref"], "") or "")
            dur = first_of(row, ["duration", "duration_s", "length_seconds"])
            if dur is None:
                dur = duration_from_file(audio)
            items.append({
                "audio_filepath": audio,
                "text": ref,
                "call_id": parent_call,
                #"segment_index": int(first_of(row, ["segment_index", "segment_id", "index"], row_idx)),
                "segment_index": row_idx,
                "segment_id": str(first_of(row, ["segment_id", "id"], f"segment-{row_idx:07d}")),
                "duration": None if dur is None else float(dur),
                "source_row": row_idx,
            })
    return items


# Lightweight English normalizer for a model-independent WER-N.
# It deliberately avoids model-specific tokenizers. The exact reference metric
# used in the paper should use the same normalization for all backbones.
_PUNCT = str.maketrans({c: " " for c in string.punctuation})

def normalize_text(x: str) -> str:
    x = x.lower().replace("’", "'").replace("–", "-").replace("—", "-")
    x = x.translate(_PUNCT)
    x = re.sub(r"\s+", " ", x).strip()
    return x


def edit_counts(ref_words: Sequence[str], hyp_words: Sequence[str]) -> Tuple[int, int, int]:
    # DP with (cost, S, D, I), keeping counts for the chosen min-cost path.
    m, n = len(ref_words), len(hyp_words)
    prev = [(j, 0, 0, j) for j in range(n + 1)]
    for i in range(1, m + 1):
        cur = [(i, 0, i, 0)] + [(0, 0, 0, 0)] * n
        for j in range(1, n + 1):
            if ref_words[i - 1] == hyp_words[j - 1]:
                cur[j] = prev[j - 1]
            else:
                sub = (prev[j - 1][0] + 1, prev[j - 1][1] + 1, prev[j - 1][2], prev[j - 1][3])
                dele = (prev[j][0] + 1, prev[j][1], prev[j][2] + 1, prev[j][3])
                ins = (cur[j - 1][0] + 1, cur[j - 1][1], cur[j - 1][2], cur[j - 1][3] + 1)
                cur[j] = min(sub, dele, ins, key=lambda z: z[0])
        prev = cur
    _, s, d, ins = prev[n]
    return s, d, ins


def score_pairs(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    total_s = total_d = total_i = total_n = 0
    by_call: Dict[str, List[Tuple[str, str]]] = defaultdict(list)

    for x in items:
        ref = normalize_text(str(x.get("text", "")))
        hyp = normalize_text(str(x.get("prediction", "")))
        rw, hw = ref.split(), hyp.split()
        s, d, i = edit_counts(rw, hw)
        total_s += s; total_d += d; total_i += i; total_n += len(rw)
        by_call[str(x["call_id"])].append((ref, hyp))

    micro = (total_s + total_d + total_i) / max(1, total_n)
    call_wers = []
    for call_id, pairs in by_call.items():
        ref = " ".join(r for r, _ in pairs)
        hyp = " ".join(h for _, h in pairs)
        rw, hw = ref.split(), hyp.split()
        s, d, i = edit_counts(rw, hw)
        call_wers.append((s + d + i) / max(1, len(rw)))

    return {
        "normalization": "lowercase + punctuation-to-space + whitespace-collapse",
        "normalized_micro_wer": micro,
        "normalized_macro_call_wer": sum(call_wers) / max(1, len(call_wers)),
        "num_calls": len(call_wers),
        "num_utterances": len(items),
        "reference_words": total_n,
        "substitutions": total_s,
        "deletions": total_d,
        "insertions": total_i,
    }


def extract_texts(transcriptions: Any) -> List[str]:
    # NeMo may return List[str], List[Hypothesis], or for hybrid paths a tuple.
    if isinstance(transcriptions, tuple):
        transcriptions = transcriptions[0]
    out = []
    for x in transcriptions:
        if isinstance(x, str):
            out.append(x)
        elif hasattr(x, "text"):
            out.append(str(x.text))
        elif isinstance(x, dict) and "text" in x:
            out.append(str(x["text"]))
        else:
            out.append(str(x))
    return out


def main() -> None:
    args = parse_args()
    manifest = Path(args.manifest)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    pred_path = output / "predictions.jsonl"
    metrics_path = output / "metrics.json"

    if metrics_path.exists() and not args.overwrite:
        print(metrics_path.read_text())
        return

    try:
        from nemo.collections.asr.models import ASRModel
    except Exception as e:
        raise SystemExit(
            "NeMo ASR is required. Use an NVIDIA NeMo/Speech container or install nemo_toolkit[asr].\n"
            f"Import error: {e}"
        )

    rows = read_jsonl(manifest)
    with tempfile.TemporaryDirectory(prefix="fastconformer_baseline_") as td:
        items = flatten_manifest(rows, Path(td))
        items.sort(key=lambda x: (x["call_id"], x["segment_index"]))
        if args.max_items > 0:
            items = items[: args.max_items]

        missing = [x["audio_filepath"] for x in items if not Path(x["audio_filepath"]).is_file()]
        if missing:
            raise FileNotFoundError(f"missing audio file(s), first: {missing[0]}")

        print(f"Loading {args.model} ...")
        model = ASRModel.from_pretrained(model_name=args.model)
        if hasattr(model, "change_decoding_strategy"):
            # Hybrid RNN-T/CTC checkpoints support this exact interface.
            try:
                model.change_decoding_strategy(decoder_type=args.decoder)
            except TypeError:
                # Older NeMo releases may require positional/keyword variations.
                if args.decoder != "rnnt":
                    raise

        if args.device.startswith("cuda") and torch.cuda.is_available():
            model = model.cuda()
        model.eval()

        audio_paths = [x["audio_filepath"] for x in items]
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        with torch.inference_mode():
            trans = model.transcribe(
                audio=audio_paths,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                verbose=True,
            )
        wall = time.perf_counter() - t0
        preds = extract_texts(trans)
        if len(preds) != len(items):
            raise RuntimeError(f"got {len(preds)} predictions for {len(items)} inputs")

        total_audio = 0.0
        unknown_duration = 0
        for x, pred in zip(items, preds):
            x["prediction"] = pred
            if x.get("duration") is None:
                d = duration_from_file(x["audio_filepath"])
                x["duration"] = d
            if x.get("duration") is None:
                unknown_duration += 1
            else:
                total_audio += float(x["duration"])

        with pred_path.open("w", encoding="utf-8") as f:
            for x in items:
                f.write(json.dumps(x, ensure_ascii=False) + "\n")

        metrics = score_pairs(items)
        metrics.update({
            "model": args.model,
            "decoder": args.decoder,
            "manifest": str(manifest),
            "wall_seconds": wall,
            "audio_seconds": total_audio,
            "rtf": wall / total_audio if total_audio > 0 else None,
            "unknown_duration_items": unknown_duration,
            "peak_gpu_memory_gb": (
                torch.cuda.max_memory_allocated() / (1024 ** 3)
                if torch.cuda.is_available() else None
            ),
        })
        metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
