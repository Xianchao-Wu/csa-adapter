#!/usr/bin/env python3
"""Summarize v0.2.1 segment-level failure/stability diagnostics."""

import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("segments", help="path to segments.jsonl")
    p.add_argument("--top", type=int, default=20)
    args = p.parse_args()
    path = Path(args.segments)
    rows = [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
    if not rows:
        raise SystemExit("no segment rows")

    empty = [r for r in rows if not r.get("hypothesis", "").strip()]
    with_memory = [r for r in rows if r.get("memory_entries_before", 0) > 0]
    stats = [(r, r.get("adapter_stats") or {}) for r in rows]
    ratios = [(s.get("residual_ratio_pre_cap", 0.0), r, s) for r, s in stats if s]
    clips = [(s.get("residual_clip_fraction", 0.0), r, s) for r, s in stats if s]

    print(f"segments: {len(rows)}")
    print(f"segments_with_history: {len(with_memory)}")
    print(f"empty_hypotheses: {len(empty)}")
    if with_memory:
        print(
            "empty_after_history: "
            f"{sum(not r.get('hypothesis','').strip() for r in with_memory)}/{len(with_memory)}"
        )

    if ratios:
        vals = [x[0] for x in ratios]
        print(f"residual_ratio_pre_cap mean={sum(vals)/len(vals):.6f} max={max(vals):.6f}")
    if clips:
        vals = [x[0] for x in clips]
        print(f"residual_clip_fraction mean={sum(vals)/len(vals):.6f} max={max(vals):.6f}")

    print("\nMost suspicious residual segments:")
    for value, r, s in sorted(ratios, reverse=True, key=lambda x: x[0])[: args.top]:
        print(
            f"call={r['call_id']} seg={r['segment_id']} "
            f"mem={r.get('memory_entries_before',0)} empty={not bool(r.get('hypothesis','').strip())} "
            f"pre_ratio={value:.5f} ratio={s.get('residual_ratio',0):.5f} "
            f"clip={s.get('residual_clip_fraction',0):.3f} "
            f"cos={s.get('adapted_cosine',0):.5f} gate={s.get('gate_mean',0):.4f} "
            f"alpha={s.get('alpha_effective',0):.5f} entropy={s.get('retrieval_entropy',0):.4f}"
        )

    if empty:
        print("\nEmpty hypotheses:")
        for r in empty[: args.top]:
            s = r.get("adapter_stats") or {}
            print(
                f"call={r['call_id']} seg={r['segment_id']} "
                f"mem={r.get('memory_entries_before',0)} "
                f"pre_ratio={s.get('residual_ratio_pre_cap','-')} "
                f"ratio={s.get('residual_ratio','-')} clip={s.get('residual_clip_fraction','-')}"
            )


if __name__ == "__main__":
    main()
