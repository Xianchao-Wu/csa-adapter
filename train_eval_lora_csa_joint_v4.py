#!/usr/bin/env python3
"""Sequential LoRA + CSA experiment driver for CSA-Adapter v0.2.1.

Protocol:
  merged Whisper LoRA model -> matching feature cache -> CSA training
  -> long-form checkpoint selection -> full validation -> E22 test -> E21 test.

This script deliberately reuses the repository's existing
`csa_adapter.longform.train` and `csa_adapter.longform.evaluate` modules.
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


def run_cmd(cmd: Sequence[str], *, log_file: Optional[Path] = None) -> None:
    printable = " ".join(str(x) for x in cmd)
    print(f"[cmd] {printable}", flush=True)
    if log_file is None:
        subprocess.run(list(cmd), check=True)
        return
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with log_file.open("w", encoding="utf-8") as f:
        f.write(f"[cmd] {printable}\n")
        f.flush()
        p = subprocess.Popen(
            list(cmd), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
        )
        assert p.stdout is not None
        for line in p.stdout:
            sys.stdout.write(line)
            f.write(line)
        rc = p.wait()
        if rc != 0:
            raise subprocess.CalledProcessError(rc, list(cmd))


def require_file(path: Path, what: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{what} not found: {path}")


def require_dir(path: Path, what: str) -> None:
    if not path.is_dir():
        raise FileNotFoundError(f"{what} not found: {path}")


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def adapter_is_valid(path: Path) -> bool:
    return path.is_dir() and (path / "adapter.safetensors").is_file() and (path / "adapter_config.json").is_file()


def list_checkpoints(run_dir: Path) -> List[Path]:
    return [p for p in sorted((run_dir / "checkpoints").glob("step-*")) if adapter_is_valid(p)]


def _cache_backbone_identity(cache_dir: Path) -> Dict[str, Any]:
    meta = cache_dir / "metadata-000.json"
    require_file(meta, f"cache metadata for {cache_dir}")
    d = json.loads(meta.read_text(encoding="utf-8"))
    bb = d.get("backbone")
    if not isinstance(bb, dict):
        raise RuntimeError(f"cache metadata has no backbone object: {meta}")
    return bb


def validate_cache_backbone_args(args: argparse.Namespace) -> None:
    """Fail early if the train command cannot reproduce the cache backbone identity."""
    train_cache = Path(args.train_cache).resolve()
    valid_cache = Path(args.valid_cache).resolve()
    train_bb = _cache_backbone_identity(train_cache)
    valid_bb = _cache_backbone_identity(valid_cache)

    if train_bb != valid_bb:
        raise RuntimeError(
            "train/validation cache backbone identities differ:\n"
            f"train={json.dumps(train_bb, sort_keys=True)}\n"
            f"valid={json.dumps(valid_bb, sort_keys=True)}"
        )

    cached_model = train_bb.get("model_id")
    if cached_model != args.lora_model:
        raise RuntimeError(
            "LoRA model identifier does not match cache metadata. "
            "Use exactly the same MODEL string that was used during caching.\n"
            f"cache model_id={cached_model!r}\n"
            f"train --model ={args.lora_model!r}"
        )

    # For local HF model directories the cache code canonicalizes revision to null.
    # Do not reject args.revision='main' here; csa_adapter resolves local revisions itself.
    print(
        "[identity-preflight] cache backbone="
        + json.dumps(train_bb, sort_keys=True),
        flush=True,
    )


def build_train_command(args: argparse.Namespace) -> List[str]:
    return [
        sys.executable, "-m", "csa_adapter.longform.train",
        "--train-cache", str(Path(args.train_cache).resolve()),
        "--valid-cache", str(Path(args.valid_cache).resolve()),
        "--output", str(Path(args.output).resolve()),
        # Preserve the exact model string used by the cache builder.
        # Relative->absolute normalization can trigger a false cache/backbone mismatch.
        "--model", args.lora_model,
        "--revision", args.revision,
        "--seed", str(args.seed),
        "--epochs", str(args.epochs),
        "--max-steps", str(args.max_steps),
        "--grad-accum", str(args.grad_accum),
        "--lr", str(args.lr),
        "--history-segments", str(args.history_segments),
        "--compression-rate", str(args.compression_rate),
        "--max-memory", str(args.max_memory),
        "--compressor", args.compressor,
        "--top-k", str(args.top_k),
        "--rank", str(args.csa_rank),
        "--gate", args.gate,
        "--warmup-steps", str(args.warmup_steps),
        "--alpha-init", str(args.alpha_init),
        "--alpha-max", str(args.alpha_max),
        "--gate-bias-init", str(args.gate_bias_init),
        "--residual-ratio-cap", str(args.residual_ratio_cap),
        "--valid-examples", str(args.valid_examples),
        "--validate-every", str(args.validate_every),
        "--diagnostics-every", str(args.diagnostics_every),
        "--feature-lru", str(args.feature_lru),
        "--save-validation-checkpoints",
    ]


def train_csa(args: argparse.Namespace) -> None:
    lora_model = Path(args.lora_model).resolve()
    train_cache = Path(args.train_cache).resolve()
    valid_cache = Path(args.valid_cache).resolve()
    run_dir = Path(args.output).resolve()

    require_file(lora_model / "config.json", "LoRA merged model config")
    require_dir(train_cache, "LoRA-derived train cache")
    require_dir(valid_cache, "LoRA-derived validation cache")
    validate_cache_backbone_args(args)

    final_ckpt = run_dir / "checkpoints" / f"step-{args.max_steps:06d}" / "adapter.safetensors"
    if final_ckpt.is_file():
        print(f"[train] completed already: {final_ckpt}", flush=True)
        return

    if (run_dir / "run.json").is_file():
        raise RuntimeError(
            f"incomplete CSA run already exists: {run_dir}\n"
            "Final checkpoint is missing. Preserve it for inspection and move/delete "
            "that run directory before intentionally restarting."
        )

    run_dir.parent.mkdir(parents=True, exist_ok=True)

    print(f"[identity] train --model string : {args.lora_model}", flush=True)
    print(f"[identity] resolved model path   : {lora_model}", flush=True)
    print(f"[identity] revision              : {args.revision}", flush=True)
    for split_name, cache_root in (("train", train_cache), ("validation", valid_cache)):
        meta = cache_root / "metadata-000.json"
        if meta.is_file():
            try:
                d = json.loads(meta.read_text(encoding="utf-8"))
                print(
                    f"[identity] {split_name} metadata-000.json: "
                    + json.dumps(d, sort_keys=True),
                    flush=True,
                )
            except Exception as e:
                print(f"[identity] WARNING: cannot parse {meta}: {e}", flush=True)

    cmd = build_train_command(args)
    if "--model" not in cmd:
        raise RuntimeError("internal error: generated CSA train command is missing --model")
    mi = cmd.index("--model")
    if mi + 1 >= len(cmd) or cmd[mi + 1] != args.lora_model:
        raise RuntimeError(
            f"internal error: generated --model is wrong: {cmd[mi + 1] if mi + 1 < len(cmd) else None!r}"
        )
    print(f"[identity-preflight] generated --model={cmd[mi + 1]!r}", flush=True)
    run_cmd(cmd)
    require_file(final_ckpt, "final CSA checkpoint")

    meta = {
        "joint_type": "LoRA_then_CSA",
        "run_name": args.run_name,
        "lora_model": str(lora_model),
        "train_cache": str(train_cache),
        "valid_cache": str(valid_cache),
        "seed": args.seed,
        "csa_config": {
            "compressor": args.compressor,
            "top_k": args.top_k,
            "rank": args.csa_rank,
            "history_segments": args.history_segments,
            "compression_rate": args.compression_rate,
            "max_memory": args.max_memory,
            "gate": args.gate,
            "warmup_steps": args.warmup_steps,
            "alpha_init": args.alpha_init,
            "alpha_max": args.alpha_max,
            "gate_bias_init": args.gate_bias_init,
            "residual_ratio_cap": args.residual_ratio_cap,
        },
    }
    (run_dir / "joint_experiment.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")


def eval_command(*, manifest: Path, output: Path, model_id: str, revision: str,
                 adapter: Path, diagnostics: bool = True, max_calls: int = 0) -> List[str]:
    """Preserve the exact backbone identity string used during CSA training.

    For local models, relative and absolute paths can point to the same files
    while still being different CSA backbone identities.
    """
    cmd = [
        sys.executable, "-m", "csa_adapter.longform.evaluate",
        "--manifest", str(manifest), "--output", str(output),
        "--model", model_id, "--revision", revision,
        "--adapter", str(adapter), "--memory-mode", "history",
    ]
    if diagnostics:
        cmd.append("--diagnostics")
    if max_calls:
        cmd.extend(["--max-calls", str(max_calls)])
    return cmd


def screen_and_select(args: argparse.Namespace) -> Tuple[str, float]:
    run_dir = Path(args.output).resolve()
    model_id = args.lora_model
    model_path = Path(args.lora_model).resolve()
    validation_full = Path(args.validation_full).resolve()
    require_file(model_path / "config.json", "LoRA merged model config")
    require_file(validation_full, "full validation manifest")

    print(f"[eval-identity] --model string      : {model_id}", flush=True)
    print(f"[eval-identity] resolved model path : {model_path}", flush=True)

    checkpoints = list_checkpoints(run_dir)
    if not checkpoints:
        raise RuntimeError(f"no valid CSA checkpoints under {run_dir / 'checkpoints'}")

    print(f"[select] {args.run_name}: {len(checkpoints)} checkpoints; max_calls={args.checkpoint_select_max_calls}", flush=True)
    print_adapter_identity(checkpoints[0], model_id)
    for ckpt in checkpoints:
        tag = ckpt.name
        out_dir = run_dir / "checkpoint_screen" / tag
        metrics = out_dir / "metrics.json"
        if metrics.is_file():
            print(f"[select] skip existing {tag}", flush=True)
            continue
        run_cmd(
            eval_command(
                manifest=validation_full, output=out_dir, model_id=model_id,
                revision=args.revision, adapter=ckpt, diagnostics=True,
                max_calls=args.checkpoint_select_max_calls,
            ),
            log_file=run_dir / "logs" / f"{tag}.screen.log",
        )
        require_file(metrics, f"checkpoint-screen metrics for {tag}")

    vals: List[Tuple[float, int, str]] = []
    for p in sorted((run_dir / "checkpoint_screen").glob("step-*/metrics.json")):
        d = load_json(p)
        vals.append((float(d["normalized_micro_wer"]), int(d.get("empty_hypothesis_chunks", 0)), p.parent.name))
    if not vals:
        raise RuntimeError("no checkpoint-screen metrics found")

    score, empty, tag = min(vals, key=lambda x: (x[0], x[1], x[2]))
    (run_dir / "selected_checkpoint.txt").write_text(tag + "\n", encoding="utf-8")
    (run_dir / "selected_checkpoint.json").write_text(
        json.dumps({
            "checkpoint": tag,
            "screen_validation_micro_wer": score,
            "screen_empty_hypothesis_chunks": empty,
            "screen_max_calls": args.checkpoint_select_max_calls,
            "selection_text_history": False,
            "selection_memory_mode": "history",
            "base_model": model_id,
            "joint_type": "LoRA_then_CSA",
        }, indent=2) + "\n", encoding="utf-8")
    print(f"[select] selected {tag}: WER-N={score:.8f}, empty={empty}", flush=True)
    return tag, score


def evaluate_selected(args: argparse.Namespace) -> None:
    run_dir = Path(args.output).resolve()
    model_id = args.lora_model
    model_path = Path(args.lora_model).resolve()
    require_file(model_path / "config.json", "LoRA merged model config")
    selected_txt = run_dir / "selected_checkpoint.txt"
    require_file(selected_txt, "selected checkpoint file")
    tag = selected_txt.read_text(encoding="utf-8").strip()
    adapter = run_dir / "checkpoints" / tag
    if not adapter_is_valid(adapter):
        raise RuntimeError(f"invalid selected CSA adapter: {adapter}")

    tasks = [
        ("validation", Path(args.validation_full).resolve(), run_dir / "eval_validation_full"),
        ("e22", Path(args.test_full).resolve(), run_dir / "eval_test"),
    ]
    if not args.skip_e21:
        tasks.append(("e21", Path(args.e21_full).resolve(), run_dir / "eval_e21"))

    for label, manifest, out_dir in tasks:
        require_file(manifest, f"{label} manifest")
        metrics = out_dir / "metrics.json"
        if metrics.is_file():
            print(f"[eval] skip existing {args.run_name}/{label}", flush=True)
            continue
        print(f"[eval] {args.run_name}/{label} with {tag}", flush=True)
        run_cmd(
            eval_command(manifest=manifest, output=out_dir, model_id=model_id,
                         revision=args.revision, adapter=adapter, diagnostics=True),
            log_file=run_dir / "logs" / f"eval_{label}.log",
        )
        require_file(metrics, f"{label} metrics")


def run_stage(args: argparse.Namespace) -> None:
    if args.stage in ("train", "all"):
        train_csa(args)
    if args.stage in ("evaluate", "all"):
        screen_and_select(args)
        evaluate_selected(args)



def evaluate_lora_only(args: argparse.Namespace) -> None:
    model_id = args.lora_model
    model_path = Path(args.lora_model).resolve()
    out_root = Path(args.output).resolve()
    require_file(model_path / "config.json", "LoRA merged model config")

    tasks = [
        ("validation", Path(args.validation_full).resolve(), out_root / "eval_validation_full"),
        ("e22", Path(args.test_full).resolve(), out_root / "eval_test"),
    ]
    if not args.skip_e21:
        tasks.append(("e21", Path(args.e21_full).resolve(), out_root / "eval_e21"))

    for label, manifest, out_dir in tasks:
        require_file(manifest, f"{label} manifest")
        metrics = out_dir / "metrics.json"
        if metrics.is_file():
            print(f"[lora-eval] skip existing {label}", flush=True)
            continue
        cmd = [
            sys.executable, "-m", "csa_adapter.longform.evaluate",
            "--manifest", str(manifest),
            "--output", str(out_dir),
            "--model", model_id,
            "--revision", args.revision,
        ]
        print(f"[lora-eval] {label}", flush=True)
        run_cmd(cmd, log_file=out_root / "logs" / f"eval_{label}.log")
        require_file(metrics, f"LoRA-only {label} metrics")

def mean_std(vals: Iterable[Optional[float]]) -> Tuple[Optional[float], Optional[float], int]:
    xs = [float(x) for x in vals if x is not None]
    if not xs:
        return None, None, 0
    return statistics.mean(xs), (statistics.stdev(xs) if len(xs) > 1 else 0.0), len(xs)


def summary_stage(args: argparse.Namespace) -> None:
    joint_root = Path(args.joint_root).resolve()
    lora_root = Path(args.lora_only_root).resolve()
    manifest = Path(args.run_manifest).resolve()
    require_file(manifest, "joint run manifest")
    rows = list(csv.DictReader(manifest.open(encoding="utf-8"), delimiter="\t"))
    out_rows: List[Dict[str, Any]] = []

    for r in rows:
        run = r["run"]
        joint = joint_root / run
        lora = lora_root / run
        tr = load_json(joint / "run.json")
        sel = load_json(joint / "selected_checkpoint.json")
        jv = load_json(joint / "eval_validation_full" / "metrics.json")
        jt = load_json(joint / "eval_test" / "metrics.json")
        je = load_json(joint / "eval_e21" / "metrics.json")
        lv = load_json(lora / "eval_validation_full" / "metrics.json")
        lt = load_json(lora / "eval_test" / "metrics.json")
        le = load_json(lora / "eval_e21" / "metrics.json")

        def metric(d: Dict[str, Any], k: str = "normalized_micro_wer"):
            v = d.get(k)
            return None if v is None else float(v)

        lora_val, lora_test, lora_e21 = metric(lv), metric(lt), metric(le)
        joint_val, joint_test, joint_e21 = metric(jv), metric(jt), metric(je)
        out_rows.append({
            **r,
            "csa_checkpoint": sel.get("checkpoint"),
            "csa_trainable_parameters": tr.get("trainable_parameters"),
            "lora_val_wer": lora_val,
            "joint_val_wer": joint_val,
            "delta_val_joint_minus_lora": None if lora_val is None or joint_val is None else joint_val - lora_val,
            "lora_test_wer": lora_test,
            "joint_test_wer": joint_test,
            "delta_test_joint_minus_lora": None if lora_test is None or joint_test is None else joint_test - lora_test,
            "lora_e21_wer": lora_e21,
            "joint_e21_wer": joint_e21,
            "delta_e21_joint_minus_lora": None if lora_e21 is None or joint_e21 is None else joint_e21 - lora_e21,
            "joint_test_rtf": metric(jt, "rtf"),
            "joint_test_empty": jt.get("empty_hypothesis_chunks"),
            "joint_residual": jt.get("diag_mean_residual_ratio"),
            "joint_clip": jt.get("diag_mean_residual_clip_fraction"),
            "joint_cosine": jt.get("diag_mean_adapted_cosine"),
        })

    cols = [
        "run", "lora_rank", "seed", "csa_checkpoint", "csa_trainable_parameters",
        "lora_val_wer", "joint_val_wer", "delta_val_joint_minus_lora",
        "lora_test_wer", "joint_test_wer", "delta_test_joint_minus_lora",
        "lora_e21_wer", "joint_e21_wer", "delta_e21_joint_minus_lora",
        "joint_test_rtf", "joint_test_empty", "joint_residual", "joint_clip", "joint_cosine",
    ]

    def fmt(v: Any) -> str:
        if v is None: return "-"
        if isinstance(v, float): return f"{v:.8f}"
        return str(v)

    with (joint_root / "joint_run_summary.tsv").open("w", encoding="utf-8") as f:
        f.write("\t".join(cols) + "\n")
        for r in out_rows:
            f.write("\t".join(fmt(r.get(c)) for c in cols) + "\n")

    groups: Dict[str, List[Dict[str, Any]]] = {}
    for r in out_rows:
        groups.setdefault(str(r["lora_rank"]), []).append(r)

    acols = [
        "lora_rank", "n", "lora_test_mean", "lora_test_std",
        "joint_test_mean", "joint_test_std", "delta_test_mean",
        "lora_e21_mean", "lora_e21_std", "joint_e21_mean", "joint_e21_std", "delta_e21_mean",
    ]
    with (joint_root / "joint_aggregate_summary.tsv").open("w", encoding="utf-8") as f:
        f.write("\t".join(acols) + "\n")
        for rank, xs in groups.items():
            ltm, lts, n = mean_std(x["lora_test_wer"] for x in xs)
            jtm, jts, _ = mean_std(x["joint_test_wer"] for x in xs)
            dtm, _, _ = mean_std(x["delta_test_joint_minus_lora"] for x in xs)
            lem, les, _ = mean_std(x["lora_e21_wer"] for x in xs)
            jem, jes, _ = mean_std(x["joint_e21_wer"] for x in xs)
            dem, _, _ = mean_std(x["delta_e21_joint_minus_lora"] for x in xs)
            vals = [rank, n, ltm, lts, jtm, jts, dtm, lem, les, jem, jes, dem]
            f.write("\t".join(fmt(v) for v in vals) + "\n")

    status = {
        "runs_total": len(rows),
        "joint_selected": sum((joint_root / r["run"] / "selected_checkpoint.json").is_file() for r in rows),
        "joint_val_metrics": sum((joint_root / r["run"] / "eval_validation_full" / "metrics.json").is_file() for r in rows),
        "joint_e22_metrics": sum((joint_root / r["run"] / "eval_test" / "metrics.json").is_file() for r in rows),
        "joint_e21_metrics": sum((joint_root / r["run"] / "eval_e21" / "metrics.json").is_file() for r in rows),
        "lora_only_e22_metrics": sum((lora_root / r["run"] / "eval_test" / "metrics.json").is_file() for r in rows),
    }
    (joint_root / "joint_status.json").write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
    print("=== joint_status.json ===")
    print(json.dumps(status, indent=2))
    print("=== joint_aggregate_summary.tsv ===")
    print((joint_root / "joint_aggregate_summary.tsv").read_text(), end="")


def add_common_run_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--stage", choices=("train", "evaluate", "all"), required=True)
    p.add_argument("--run-name", required=True)
    p.add_argument("--lora-model", required=True)
    p.add_argument("--train-cache", required=True)
    p.add_argument("--valid-cache", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--revision", default="main")
    p.add_argument("--validation-full", default="data/earnings22_622/validation.full.jsonl")
    p.add_argument("--test-full", default="data/earnings22_622/test.full.jsonl")
    p.add_argument("--e21-full", default="data/earnings22_622/earnings21.full.jsonl")
    p.add_argument("--skip-e21", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=99)
    p.add_argument("--max-steps", type=int, default=1500)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--history-segments", type=int, default=64)
    p.add_argument("--compression-rate", type=int, default=8)
    p.add_argument("--max-memory", type=int, default=4096)
    p.add_argument("--compressor", choices=("event", "mean"), default="event")
    p.add_argument("--top-k", type=int, default=16)
    p.add_argument("--csa-rank", type=int, default=16)
    p.add_argument("--gate", default="diagonal")
    p.add_argument("--warmup-steps", type=int, default=0)
    p.add_argument("--alpha-init", type=float, default=0.01)
    p.add_argument("--alpha-max", type=float, default=0.10)
    p.add_argument("--gate-bias-init", type=float, default=-2.0)
    p.add_argument("--residual-ratio-cap", type=float, default=0.25)
    p.add_argument("--valid-examples", type=int, default=256)
    p.add_argument("--validate-every", type=int, default=250)
    p.add_argument("--diagnostics-every", type=int, default=10)
    p.add_argument("--feature-lru", type=int, default=256)
    p.add_argument("--checkpoint-select-max-calls", type=int, default=8)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="command", required=True)
    runp = sub.add_parser("run", help="train/evaluate one LoRA+CSA run")
    add_common_run_args(runp)
    lep = sub.add_parser("lora-eval", help="evaluate a merged LoRA model with the same long-form protocol")
    lep.add_argument("--lora-model", required=True)
    lep.add_argument("--output", required=True)
    lep.add_argument("--revision", default="main")
    lep.add_argument("--validation-full", default="data/earnings22_622/validation.full.jsonl")
    lep.add_argument("--test-full", default="data/earnings22_622/test.full.jsonl")
    lep.add_argument("--e21-full", default="data/earnings22_622/earnings21.full.jsonl")
    lep.add_argument("--skip-e21", action="store_true")

    sump = sub.add_parser("summary", help="compare LoRA-only vs LoRA+CSA")
    sump.add_argument("--joint-root", required=True)
    sump.add_argument("--run-manifest", required=True)
    sump.add_argument("--lora-only-root", required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "run":
        run_stage(args)
    elif args.command == "lora-eval":
        evaluate_lora_only(args)
    elif args.command == "summary":
        summary_stage(args)
    else:
        raise AssertionError(args.command)


if __name__ == "__main__":
    main()
