"""Call-disjoint preparation, byte decoding without torchaudio/torchcodec."""

import argparse
import hashlib
import io
import json
import math
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows):
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def decode_audio(audio):
    source = io.BytesIO(audio["bytes"]) if audio.get("bytes") is not None else audio["path"]
    x, sr = sf.read(source, dtype="float32", always_2d=True)
    x = x.mean(-1)
    if sr != 16000:
        g = math.gcd(sr, 16000)
        x = resample_poly(x, 16000 // g, sr // g).astype(np.float32)
    if not len(x) or not np.isfinite(x).all():
        raise ValueError("Empty/nonfinite audio")
    return x


def split_calls(ids, ratios=(0.6, 0.2, 0.2), seed=42):
    if len(ratios) != 3 or min(ratios) < 0 or abs(sum(ratios) - 1) > 1e-6:
        raise ValueError("Need three nonnegative ratios summing to one")
    ids = sorted(set(map(str, ids)))
    random.Random(seed).shuffle(ids)
    ntrain, nvalid = int(len(ids) * ratios[0]), int(len(ids) * ratios[1])
    return dict(
        train=sorted(ids[:ntrain]),
        validation=sorted(ids[ntrain : ntrain + nvalid]),
        test=sorted(ids[ntrain + nvalid :]),
    )


def groups(rows):
    grouped = defaultdict(list)
    seen = set()
    for row in rows:
        key = (row["call_id"], row["segment_id"])
        if key in seen:
            raise ValueError(f"Duplicate segment: {key}")
        seen.add(key)
        grouped[row["call_id"]].append(row)
    for seq in grouped.values():
        seq.sort(key=lambda r: (r["start"], r["end"], r["segment_id"]))
        for a, b in zip(seq, seq[1:]):
            if b["start"] < a["end"] - 1e-4:
                raise ValueError(f"Overlapping segments in {a['call_id']}")
    return dict(grouped)


def hf_rows(dataset, config, revision):
    from datasets import Audio, load_dataset

    ds = load_dataset(dataset, config, split="test", revision=revision, streaming=True)
    return ds.cast_column("audio", Audio(decode=False))


def hf_segment_metadata(revision):
    # Synchronous Parquet column projection; explicit close avoids leaving Arrow
    # streaming prefetch workers alive at interpreter shutdown on partial reads.
    import pyarrow.parquet as pq
    from huggingface_hub import HfApi, HfFileSystem

    names = HfApi().list_repo_files(
        "distil-whisper/earnings22", repo_type="dataset", revision=revision
    )
    files = sorted(
        name for name in names if name.startswith("chunked/") and name.endswith(".parquet")
    )
    if not files:
        raise ValueError("No chunked parquet files at selected revision")
    fs = HfFileSystem()
    columns = ["file_id", "segment_id", "transcription", "start_ts", "end_ts"]
    for name in files:
        print(f"Reading alignment metadata: {name}", flush=True)
        with fs.open(
            f"datasets/distil-whisper/earnings22@{revision}/{name}",
            "rb",
            block_size=65536,
            cache_type="none",
        ) as f:
            table = pq.ParquetFile(f).read(columns=columns, use_threads=False)
        yield from table.to_pylist()


def pack_segments(rows, max_seconds=28.0, max_gap=1.0):
    # Never join across calls, large gaps, or a rejected/omitted source segment.
    packed = []
    for cid, seq in groups(rows).items():
        pending = None
        for row in seq:
            if (
                pending is not None
                and row["end"] - pending["start"] <= max_seconds
                and row["start"] - pending["end"] <= max_gap
                and row["source_order"] == pending["last_source_order"] + 1
            ):
                pending["end"] = row["end"]
                pending["text"] += " " + row["text"]
                pending["source_segments"].append(row["segment_id"])
                pending["last_source_order"] = row["source_order"]
            else:
                if pending is not None:
                    packed.append(pending)
                pending = dict(
                    row, source_segments=[row["segment_id"]], last_source_order=row["source_order"]
                )
        if pending is not None:
            packed.append(pending)
    return packed


def prepare():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True)
    p.add_argument("--ratios", default="0.6,0.2,0.2")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--revision", default="main")
    p.add_argument("--max-seconds", type=float, default=28.0)
    p.add_argument("--oversize", choices=["error", "skip"], default="error")
    p.add_argument("--external-earnings21", action="store_true")
    p.add_argument("--external-revision", default="main")
    args = p.parse_args()
    if not 0 < args.max_seconds <= 30:
        p.error("--max-seconds must be in (0,30]")
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out / "split_calls.json").exists():
        raise FileExistsError("Prepared output already exists; use a new output directory")
    audio_dir = out / "audio"
    audio_dir.mkdir(exist_ok=True)
    from huggingface_hub import HfApi

    rev = HfApi().dataset_info("distil-whisper/earnings22", revision=args.revision).sha
    full, metadata = [], {}
    for row in hf_rows("distil-whisper/earnings22", "full", rev):
        cid = str(row["file_id"])
        token = hashlib.sha256(cid.encode()).hexdigest()[:20]
        path = audio_dir / f"e22-full-{token}.flac"
        x = decode_audio(row["audio"])
        sf.write(path, x, 16000, format="FLAC")
        full.append(
            dict(
                call_id=cid,
                audio_filepath=str(path),
                text=row["transcription"],
                duration=len(x) / 16000,
                corpus="earnings22",
            )
        )
        metadata[cid] = {
            k: row.get(k) for k in ("ticker_symbol", "country_by_ticker", "language_family")
        }
        print(f"Full call {cid}: {len(x) / 16000:.1f}s", flush=True)
    split = split_calls(
        [r["call_id"] for r in full], tuple(map(float, args.ratios.split(","))), args.seed
    )
    membership = {cid: name for name, ids in split.items() for cid in ids}
    segments, rejected, seen_ids = [], [], set()
    source_rows = []
    source_order = defaultdict(int)
    full_by_id = {r["call_id"]: r for r in full}
    # Use alignment metadata to cut original full audio. Training labels guide
    # training boundaries only; full-call evaluation uses fixed acoustic windows.
    for row in hf_segment_metadata(rev):
        cid = str(row["file_id"])
        if cid not in membership:
            raise ValueError(f"Chunked call absent from full: {cid}")
        seen_ids.add(cid)
        source_rows.append(
            dict(
                call_id=cid,
                segment_id=str(row["segment_id"]),
                text=str(row["transcription"]),
                start=float(row["start_ts"]),
                end=float(row["end_ts"]),
            )
        )
    # Sort before assigning per-call source order; parquet storage order is irrelevant.
    source_rows.sort(key=lambda r: (r["call_id"], r["start"], r["end"], r["segment_id"]))
    retained = []
    retained_end = {}
    for row in source_rows:
        cid = row["call_id"]
        row["source_order"] = source_order[cid]
        source_order[cid] += 1
        duration = row["end"] - row["start"]
        reason = None
        if duration > args.max_seconds:
            reason = "oversize"
        elif not row["text"].strip():
            reason = "empty_text"
        elif duration <= 0 or row["start"] < 0 or row["end"] > full_by_id[cid]["duration"] + 0.05:
            reason = "invalid_time"
        if reason is None and row["start"] < retained_end.get(cid, -float("inf")) - 1e-4:
            reason = "overlapping_alignment"
        if reason:
            rejected.append(
                dict(
                    call_id=cid,
                    segment_id=row["segment_id"],
                    duration=max(0, duration),
                    reason=reason,
                )
            )
            if reason == "oversize" and args.oversize == "error":
                raise ValueError(
                    f"{cid}: {duration:.2f}s > limit. Use explicit --oversize skip; never truncate labels."
                )
        else:
            retained.append(row)
            retained_end[cid] = row["end"]
    for i, row in enumerate(pack_segments(retained, args.max_seconds)):
        cid = row["call_id"]
        start_sample = round(row["start"] * 16000)
        end_sample = min(round(row["end"] * 16000), round(full_by_id[cid]["duration"] * 16000))
        x, sr = sf.read(
            full_by_id[cid]["audio_filepath"], start=start_sample, stop=end_sample, dtype="float32"
        )
        if sr != 16000 or not 0 < len(x) <= 30 * 16000:
            raise ValueError("Invalid packed waveform")
        path = audio_dir / f"e22-packed-{i:07d}.flac"
        sf.write(path, x, sr, format="FLAC")
        segments.append(
            dict(
                call_id=cid,
                segment_id=f"packed-{i:07d}",
                audio_filepath=str(path),
                text=row["text"],
                start=start_sample / 16000,
                end=end_sample / 16000,
                duration=len(x) / 16000,
                corpus="earnings22",
                source_segments=row["source_segments"],
            )
        )
    groups(segments)
    if seen_ids != set(membership):
        raise ValueError("Full/chunked call coverage differs")
    for name, ids in split.items():
        write_jsonl(out / f"{name}.jsonl", [r for r in segments if r["call_id"] in ids])
        write_jsonl(out / f"{name}.full.jsonl", [r for r in full if r["call_id"] in ids])
    write_jsonl(out / "excluded_segments.jsonl", rejected)
    report = dict(
        protocol="custom_call_disjoint_earnings22_not_official_test",
        seed=args.seed,
        ratios=args.ratios,
        source_revision=rev,
        call_metadata=metadata,
        segmentation="aligned_metadata_packed_full_audio",
        max_segment_seconds=args.max_seconds,
        source_segments=len(source_rows),
        retained_source_segments=len(retained),
        total_calls=len(full),
        retained_segments=len(segments),
        excluded_segments=len(rejected),
        full_hours=sum(r["duration"] for r in full) / 3600,
        retained_segment_hours=sum(r["duration"] for r in segments) / 3600,
        excluded_segment_hours=sum(r["duration"] for r in rejected) / 3600,
        splits={
            name: dict(
                calls=len(ids),
                segments=sum(r["call_id"] in ids for r in segments),
                full_hours=sum(r["duration"] for r in full if r["call_id"] in ids) / 3600,
                excluded_segments=sum(r["call_id"] in ids for r in rejected),
            )
            for name, ids in split.items()
        },
    )
    if args.external_earnings21:
        rev21 = (
            HfApi().dataset_info("distil-whisper/earnings21", revision=args.external_revision).sha
        )
        ext = []
        for row in hf_rows("distil-whisper/earnings21", "full", rev21):
            cid = str(row["file_id"])
            token = hashlib.sha256(cid.encode()).hexdigest()[:20]
            path = audio_dir / f"e21-full-{token}.flac"
            x = decode_audio(row["audio"])
            sf.write(path, x, 16000, format="FLAC")
            ext.append(
                dict(
                    call_id="e21-" + cid,
                    audio_filepath=str(path),
                    text=row["transcription"],
                    duration=len(x) / 16000,
                    corpus="earnings21",
                )
            )
        write_jsonl(out / "earnings21.full.jsonl", ext)
        report["earnings21_revision"] = rev21
        report["earnings21_calls"] = len(ext)
    (out / "split_calls.json").write_text(json.dumps(split, indent=2))
    (out / "preparation_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report["splits"], indent=2))


if __name__ == "__main__":
    prepare()
