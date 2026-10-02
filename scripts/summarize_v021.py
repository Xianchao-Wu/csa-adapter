#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def fmt(x):
    if x is None:
        return "-"
    if isinstance(x, float):
        return f"{x:.6f}"
    return str(x)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("root", nargs="?", default="runs/h100_v021_priority")
    args = p.parse_args()
    root = Path(args.root)
    keys = [
        "normalized_micro_wer",
        "rtf",
        "empty_hypothesis_chunks",
        "diag_mean_alpha_effective",
        "diag_mean_gate_mean",
        "diag_mean_residual_ratio_pre_cap",
        "diag_mean_residual_ratio",
        "diag_max_residual_ratio",
        "diag_mean_residual_clip_fraction",
        "diag_mean_adapted_cosine",
    ]
    print("experiment\t" + "\t".join(keys))
    for path in sorted(root.rglob("metrics.json")):
        data = json.loads(path.read_text())
        name = str(path.parent.relative_to(root))
        print(name + "\t" + "\t".join(fmt(data.get(k)) for k in keys))


if __name__ == "__main__":
    main()
