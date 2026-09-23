from __future__ import annotations

import argparse
import json
from pathlib import Path

from transformers import (
    AutoProcessor,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    WhisperForConditionalGeneration,
)

from csa_adapter.config import CSAAdapterConfig
from csa_adapter.data.collator import WhisperDataCollator
from csa_adapter.data.registry import load_asr_dataset
from csa_adapter.whisper import adapter_parameter_count, patch_whisper_encoder, save_csa_adapter


def parse_layers(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train CSA-Adapter on a frozen Whisper backbone")
    p.add_argument("--model", default="openai/whisper-large-v3")
    p.add_argument("--dataset", default="finasr")
    p.add_argument("--train-split", default="train")
    p.add_argument("--eval-split", default="validation")
    p.add_argument("--hf-id")
    p.add_argument("--subset")
    p.add_argument("--train-jsonl")
    p.add_argument("--eval-jsonl")
    p.add_argument("--audio-column")
    p.add_argument("--text-column")
    p.add_argument("--language")
    p.add_argument("--task", default="transcribe")
    p.add_argument("--output-dir", default="outputs/csa-whisper")
    p.add_argument("--layers", default="-1")
    p.add_argument("--adapter-dim", type=int, default=128)
    p.add_argument("--index-dim", type=int, default=64)
    p.add_argument("--value-dim", type=int, default=64)
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--compression-rate", type=int, default=4)
    p.add_argument("--top-k", type=int, default=8)
    p.add_argument("--compressor", choices=["mean", "depthwise", "event"], default="mean")
    p.add_argument("--query-chunk-size", type=int, default=256)
    p.add_argument("--memory-chunk-size", type=int, default=512)
    p.add_argument("--causal", action="store_true")
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--epochs", type=float, default=3.0)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--max-train-samples", type=int)
    p.add_argument("--max-eval-samples", type=int)
    p.add_argument("--gradient-checkpointing", action="store_true")
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--trust-remote-code", action="store_true")
    return p


def main() -> None:
    raise RuntimeError(
        "Legacy v0.1 CLI disabled: use csa_adapter.longform.train/evaluate; see README"
    )
    args = build_parser().parse_args()
    processor = AutoProcessor.from_pretrained(args.model)
    model = WhisperForConditionalGeneration.from_pretrained(args.model)
    model.config.use_cache = False
    if args.language:
        model.generation_config.language = args.language
    model.generation_config.task = args.task

    cfg = CSAAdapterConfig(
        model_dim=model.config.d_model,
        adapter_dim=args.adapter_dim,
        index_dim=args.index_dim,
        value_dim=args.value_dim,
        rank=args.rank,
        num_heads=args.heads,
        compression_rate=args.compression_rate,
        top_k=args.top_k,
        query_chunk_size=args.query_chunk_size,
        memory_chunk_size=args.memory_chunk_size,
        compressor=args.compressor,
        causal=args.causal,
        layer_indices=parse_layers(args.layers),
    )
    patch_whisper_encoder(model, cfg)
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    trainable, total = adapter_parameter_count(model)
    print(f"trainable={trainable:,} total={total:,} ratio={100 * trainable / total:.4f}%")

    common = dict(
        name=args.dataset,
        hf_id=args.hf_id,
        subset=args.subset,
        audio_column=args.audio_column,
        text_column=args.text_column,
        trust_remote_code=args.trust_remote_code,
    )
    train_ds, audio_col, text_col = load_asr_dataset(
        **common, split=args.train_split, local_jsonl=args.train_jsonl
    )
    eval_ds, eval_audio_col, eval_text_col = load_asr_dataset(
        **common, split=args.eval_split, local_jsonl=args.eval_jsonl
    )
    if args.max_train_samples:
        train_ds = train_ds.select(range(min(args.max_train_samples, len(train_ds))))
    if args.max_eval_samples:
        eval_ds = eval_ds.select(range(min(args.max_eval_samples, len(eval_ds))))

    def prepare(example, a_col: str, t_col: str):
        audio = example[a_col]
        example["input_features"] = processor.feature_extractor(
            audio["array"], sampling_rate=audio["sampling_rate"]
        ).input_features[0]
        example["labels"] = processor.tokenizer(str(example[t_col])).input_ids
        return example

    train_ds = train_ds.map(
        lambda x: prepare(x, audio_col, text_col),
        remove_columns=train_ds.column_names,
        desc="Extracting train features",
    )
    eval_ds = eval_ds.map(
        lambda x: prepare(x, eval_audio_col, eval_text_col),
        remove_columns=eval_ds.column_names,
        desc="Extracting eval features",
    )

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=10,
        predict_with_generate=False,
        remove_unused_columns=False,
        fp16=args.fp16,
        bf16=args.bf16,
        report_to="none",
    )
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=WhisperDataCollator(processor),
    )
    trainer.train()
    adapter_dir = Path(args.output_dir) / "final_adapter"
    save_csa_adapter(model, adapter_dir)
    (adapter_dir / "training_args.json").write_text(
        json.dumps(vars(args), indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
