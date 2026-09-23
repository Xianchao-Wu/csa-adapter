from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from jiwer import cer, wer
from torch.utils.data import DataLoader
from transformers import AutoProcessor, WhisperForConditionalGeneration

from csa_adapter.data.registry import load_asr_dataset
from csa_adapter.whisper import load_csa_adapter


def main() -> None:
    raise RuntimeError(
        "Legacy v0.1 CLI disabled: use csa_adapter.longform.train/evaluate; see README"
    )
    p = argparse.ArgumentParser(description="Evaluate a Whisper CSA-Adapter")
    p.add_argument("--model", default="openai/whisper-large-v3")
    p.add_argument("--adapter", required=True)
    p.add_argument("--dataset", default="earnings22")
    p.add_argument("--split")
    p.add_argument("--hf-id")
    p.add_argument("--subset")
    p.add_argument("--local-jsonl")
    p.add_argument("--audio-column")
    p.add_argument("--text-column")
    p.add_argument("--language")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--max-samples", type=int)
    p.add_argument("--max-new-tokens", type=int, default=448)
    p.add_argument("--metric", choices=["wer", "cer", "both"], default="both")
    p.add_argument("--output", default="predictions.jsonl")
    p.add_argument("--trust-remote-code", action="store_true")
    args = p.parse_args()

    processor = AutoProcessor.from_pretrained(args.model)
    model = WhisperForConditionalGeneration.from_pretrained(args.model)
    load_csa_adapter(model, args.adapter)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    if args.language:
        model.generation_config.language = args.language
    model.generation_config.task = "transcribe"

    dataset, audio_col, text_col = load_asr_dataset(
        args.dataset,
        split=args.split,
        hf_id=args.hf_id,
        subset=args.subset,
        local_jsonl=args.local_jsonl,
        audio_column=args.audio_column,
        text_column=args.text_column,
        trust_remote_code=args.trust_remote_code,
    )
    if args.max_samples:
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))

    def collate(rows):
        audios = [row[audio_col] for row in rows]
        inputs = processor.feature_extractor(
            [a["array"] for a in audios], sampling_rate=16000, return_tensors="pt"
        )
        return inputs.input_features, [str(row[text_col]) for row in rows]

    records, references, hypotheses = [], [], []
    for features, refs in DataLoader(dataset, batch_size=args.batch_size, collate_fn=collate):
        with torch.inference_mode():
            ids = model.generate(features.to(device), max_new_tokens=args.max_new_tokens)
        hyps = processor.batch_decode(ids, skip_special_tokens=True)
        for ref, hyp in zip(refs, hyps):
            records.append({"reference": ref, "hypothesis": hyp})
            references.append(ref)
            hypotheses.append(hyp)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    metrics = {}
    if args.metric in {"wer", "both"}:
        metrics["wer"] = wer(references, hypotheses)
    if args.metric in {"cer", "both"}:
        metrics["cer"] = cer(references, hypotheses)
    metrics["num_samples"] = len(records)
    print(json.dumps(metrics, indent=2))
    output.with_suffix(".metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
