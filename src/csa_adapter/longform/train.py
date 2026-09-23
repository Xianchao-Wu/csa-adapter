"""Single-GPU cached-feature training with differentiable historical writing."""

import argparse
import json
import random
from collections import OrderedDict
from pathlib import Path

import torch

from .data import groups, read_jsonl
from .memory import MemoryConfig, PersistentCSA
from .runtime import (
    adapt_valid,
    amp,
    asr_loss,
    labels_for,
    load_backbone,
    model_identity,
    save_adapter,
)


def load_index(directory, identity):
    directory = Path(directory)
    metas = sorted(directory.glob("metadata-*.json"))
    if not metas:
        raise ValueError(f"No cache metadata in {directory}")
    meta = json.loads(metas[0].read_text())
    if meta["backbone"] != identity:
        raise ValueError("Cache/backbone mismatch")
    rows = []
    for i in range(meta["shards"]):
        m = directory / f"metadata-{i:03d}.json"
        ix = directory / f"index-{i:03d}.jsonl"
        if not m.exists() or not ix.exists() or json.loads(m.read_text()) != meta:
            raise ValueError("Incomplete or inconsistent cache shards")
        rows.extend(read_jsonl(ix))
    return groups(rows)


class Features:
    def __init__(self, capacity=128):
        self.cache = OrderedDict()
        self.capacity = capacity

    def get(self, row):
        path = row["cache_file"]
        if path not in self.cache:
            self.cache[path] = torch.load(path, map_location="cpu", weights_only=True)
        self.cache.move_to_end(path)
        value = self.cache[path]
        while len(self.cache) > self.capacity:
            self.cache.popitem(last=False)
        return value


def examples(calls):
    # Every segment except the first has at least one genuine historical segment.
    return [(cid, i) for cid, rows in sorted(calls.items()) for i in range(1, len(rows))]


def loss_for(example, calls, features, adapter, model, proc, args, dense):
    cid, i = example
    seq = calls[cid]
    current = features.get(seq[i])
    raw = current["hidden"].unsqueeze(0).to(args.device, dtype=next(model.parameters()).dtype)
    nvalid = current["nvalid"]
    history = []
    for row in seq[max(0, i - args.history_segments) : i]:
        if row["end"] > seq[i]["start"] + 1e-4:
            raise ValueError("Future/overlapping history")
        h = features.get(row)
        history.append(h["hidden"][: h["nvalid"]].unsqueeze(0).to(args.device, dtype=raw.dtype))
    labels = labels_for(proc, seq[i]["text"], model)
    with amp(args.device):
        memory = adapter.historical_memory(history)
        adapted = adapt_valid(
            adapter, raw, nvalid, memory, dense=dense, temperature=args.temperature
        )
        loss = asr_loss(model, adapted, labels)
    return loss, labels.numel()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train-cache", required=True)
    p.add_argument("--valid-cache", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--model", default="openai/whisper-large-v3")
    p.add_argument("--revision", default="main")
    p.add_argument("--device", default="cuda")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--max-steps", type=int, default=0)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--history-segments", type=int, default=64)
    p.add_argument("--compression-rate", type=int, default=8)
    p.add_argument("--top-k", type=int, default=16)
    p.add_argument("--max-memory", type=int, default=4096)
    p.add_argument("--adapter-dim", type=int, default=128)
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--compressor", choices=["mean", "event", "depthwise"], default="event")
    p.add_argument("--gate", choices=["diagonal", "none"], default="diagonal")
    p.add_argument("--warmup-steps", type=int, default=100)
    p.add_argument("--dense-always", action="store_true")
    p.add_argument(
        "--valid-examples",
        type=int,
        default=256,
        help="0 = all eligible validation segments; positive = fixed seeded subset",
    )
    p.add_argument("--validate-every", type=int, default=250)
    p.add_argument("--feature-lru", type=int, default=128)
    args = p.parse_args()
    if min(args.epochs, args.grad_accum, args.history_segments, args.validate_every) <= 0:
        p.error("epochs, grad-accum, history-segments, validate-every must be positive")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "run.json").exists():
        raise FileExistsError(
            "Run exists; choose a new output path (optimizer resume not implemented)"
        )
    model, proc = load_backbone(args.model, args.device, args.revision)
    identity = model_identity(model, args.model)
    train = load_index(args.train_cache, identity)
    valid = load_index(args.valid_cache, identity)
    if set(train) & set(valid):
        raise ValueError("Train/validation call leakage")
    adapter = PersistentCSA(
        MemoryConfig(
            model_dim=model.config.d_model,
            adapter_dim=args.adapter_dim,
            rank=args.rank,
            compression_rate=args.compression_rate,
            top_k=args.top_k,
            max_memory=args.max_memory,
            compressor=args.compressor,
            gate=args.gate,
        )
    ).to(args.device)
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=args.lr, weight_decay=0.01)
    features = Features(args.feature_lru)
    train_examples, valid_examples = examples(train), examples(valid)
    if not train_examples or not valid_examples:
        raise ValueError(
            "Need >=2 chronological retained segments in training and validation calls"
        )
    random.Random(args.seed).shuffle(valid_examples)
    if args.valid_examples:
        valid_examples = valid_examples[: args.valid_examples]
    info = dict(
        vars(args),
        backbone=identity,
        trainable_parameters=sum(p.numel() for p in adapter.parameters()),
        validation_selection="seeded fixed segment subset; teacher-forced NLL, no text history",
    )
    (out / "run.json").write_text(json.dumps(info, indent=2))
    (out / "validation_examples.json").write_text(json.dumps(valid_examples, indent=2))
    best, step = float("inf"), 0

    def validate():
        nonlocal best
        adapter.eval()
        total, count = 0.0, 0
        with torch.no_grad():
            for ex in valid_examples:
                loss, tokens = loss_for(
                    ex, valid, features, adapter, model, proc, args, args.dense_always
                )
                total += loss.item() * tokens
                count += tokens
        score = total / count
        record = dict(step=step, validation_nll=score, examples=len(valid_examples), tokens=count)
        with open(out / "validation.jsonl", "a") as f:
            f.write(json.dumps(record) + "\n")
        print(record, flush=True)
        save_adapter(
            out / "last_adapter", adapter, identity, dict(info, step=step, validation_nll=score)
        )
        if score < best:
            best = score
            save_adapter(
                out / "best_adapter", adapter, identity, dict(info, step=step, validation_nll=score)
            )
        adapter.train()

    args.temperature = 1.0
    for epoch in range(args.epochs):
        random.shuffle(train_examples)
        for start in range(0, len(train_examples), args.grad_accum):
            batch = train_examples[start : start + args.grad_accum]
            adapter.train()
            optimizer.zero_grad(set_to_none=True)
            dense = args.dense_always or step < args.warmup_steps
            args.temperature = (
                1.0 + max(0.0, 1 - step / max(1, args.warmup_steps)) if dense else 1.0
            )
            loss_sum = 0.0
            for ex in batch:
                loss, _ = loss_for(ex, train, features, adapter, model, proc, args, dense)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Nonfinite loss at {ex}")
                (loss / len(batch)).backward()
                loss_sum += loss.item()
            torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            step += 1
            record = dict(step=step, epoch=epoch, loss=loss_sum / len(batch), dense=dense)
            with open(out / "train.jsonl", "a") as f:
                f.write(json.dumps(record) + "\n")
            if step % 10 == 0:
                print(record, flush=True)
            if step % args.validate_every == 0:
                args.temperature = 1.0
                validate()
            if args.max_steps and step >= args.max_steps:
                args.temperature = 1.0
                validate()
                return
        args.temperature = 1.0
        validate()
    print(f"Best adapter: {out / 'best_adapter'}", flush=True)


if __name__ == "__main__":
    main()
