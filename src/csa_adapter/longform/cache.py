"""Cache full padded encoder outputs once; keep valid lengths for memory writes."""

import argparse
import hashlib
import json
from pathlib import Path

import torch

from .data import read_jsonl, write_jsonl
from .runtime import encode, load_backbone, model_identity, read_wav


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--model", default="openai/whisper-large-v3")
    p.add_argument("--revision", default="main")
    p.add_argument("--device", default="cuda")
    p.add_argument("--shards", type=int, default=1)
    p.add_argument("--shard-index", type=int, default=0)
    args = p.parse_args()
    if args.shards < 1 or not 0 <= args.shard_index < args.shards:
        p.error("Invalid shard")
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    model, proc = load_backbone(args.model, args.device, args.revision)
    identity = model_identity(model, args.model)
    rows = read_jsonl(args.manifest)
    manifest_hash = hashlib.sha256(Path(args.manifest).read_bytes()).hexdigest()
    meta = dict(
        backbone=identity,
        manifest_sha256=manifest_hash,
        dtype="float16",
        processor_revision=getattr(proc.tokenizer, "init_kwargs", {}).get("_commit_hash"),
        shards=args.shards,
    )
    meta_file = out / f"metadata-{args.shard_index:03d}.json"
    if meta_file.exists() and json.loads(meta_file.read_text()) != meta:
        raise ValueError("Cache identity changed; use a fresh cache directory")
    meta_file.write_text(json.dumps(meta, indent=2))
    index = []
    for i, row in enumerate(rows):
        if i % args.shards != args.shard_index:
            continue
        token = hashlib.sha256((row["call_id"] + "\0" + row["segment_id"]).encode()).hexdigest()
        file = out / f"{token}.pt"
        if not file.exists():
            h, n = encode(model, proc, read_wav(row["audio_filepath"]), args.device)
            tmp = file.with_suffix(".tmp")
            torch.save(dict(hidden=h[0].cpu().half(), nvalid=n), tmp)
            tmp.replace(file)
        index.append(dict(row, cache_file=str(file)))
        if len(index) % 100 == 0:
            print(f"Cached shard {args.shard_index}: {len(index)}", flush=True)
    write_jsonl(out / f"index-{args.shard_index:03d}.jsonl", index)
    print(f"Done: {len(index)} segments", flush=True)


if __name__ == "__main__":
    main()
