#!/usr/bin/env python3
"""
LoRA fine-tuning for Whisper-large-v3 on the exact Earnings-22 6:2:2 manifests
used by the CSA-Adapter experiments.

Training:
  data/earnings22_622/train.jsonl
Development loss:
  data/earnings22_622/validation.jsonl

Final long-form evaluation is intentionally NOT reimplemented here.
After training, this script merges the LoRA weights into a standalone
Hugging Face Whisper model. The accompanying bash script evaluates that
merged model with `python -m csa_adapter.longform.evaluate`, so the LoRA
baseline uses the same long-form protocol and WER-N implementation as CSA.

Expected manifest row (minimum):
  {"audio_filepath": "...wav", "text": "reference transcript"}

Also accepts `audio` or `path` as the audio-path key, and
`transcript` or `reference` as the text key.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly
from torch.utils.data import Dataset

from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)


AUDIO_KEYS = ("audio_filepath", "audio", "path", "wav", "wav_path")
TEXT_KEYS = ("text", "transcript", "reference")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="openai/whisper-large-v3")
    p.add_argument("--revision", default="main")
    p.add_argument("--train-manifest", required=True)
    p.add_argument("--valid-manifest", required=True)
    p.add_argument("--output-dir", required=True)

    p.add_argument("--rank", type=int, required=True, choices=(8, 16, 32, 64))
    p.add_argument("--lora-alpha", type=int, default=0,
                   help="0 => 2 * rank")
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--target-modules", default="q_proj,v_proj",
                   help="Comma-separated PEFT target module suffixes.")

    p.add_argument("--language", default="english")
    p.add_argument("--task", default="transcribe")
    p.add_argument("--sampling-rate", type=int, default=16000)

    p.add_argument("--epochs", type=float, default=3.0)
    p.add_argument("--max-steps", type=int, default=-1)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-ratio", type=float, default=0.05)

    p.add_argument("--train-batch-size", type=int, default=8)
    p.add_argument("--eval-batch-size", type=int, default=8)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--gradient-checkpointing", action="store_true", default=True)
    p.add_argument("--no-gradient-checkpointing",
                   dest="gradient_checkpointing", action="store_false")

    p.add_argument("--eval-steps", type=int, default=250)
    p.add_argument("--save-steps", type=int, default=250)
    p.add_argument("--logging-steps", type=int, default=25)
    p.add_argument("--save-total-limit", type=int, default=2)
    p.add_argument("--dataloader-workers", type=int, default=4)

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--tf32", action="store_true")

    p.add_argument("--resume-from-checkpoint", default=None)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--merge", action="store_true", default=True)
    p.add_argument("--no-merge", dest="merge", action="store_false")
    return p.parse_args()


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {e}") from e
            rows.append(row)
    if not rows:
        raise ValueError(f"empty manifest: {path}")
    return rows


def first_present(row: Dict[str, Any], keys) -> Any:
    for k in keys:
        if k in row and row[k] not in (None, ""):
            return row[k]
    return None


class JsonlSpeechDataset(Dataset):
    def __init__(
        self,
        manifest: str,
        processor: WhisperProcessor,
        sampling_rate: int = 16000,
    ):
        self.manifest = Path(manifest).resolve()
        self.base_dir = self.manifest.parent
        self.rows = read_jsonl(self.manifest)
        self.processor = processor
        self.sampling_rate = sampling_rate

        # Validate schema up front, without loading all audio.
        for i, row in enumerate(self.rows):
            ap = first_present(row, AUDIO_KEYS)
            txt = first_present(row, TEXT_KEYS)
            if ap is None:
                raise KeyError(
                    f"{self.manifest}: row {i} has none of audio keys {AUDIO_KEYS}"
                )
            if txt is None:
                raise KeyError(
                    f"{self.manifest}: row {i} has none of text keys {TEXT_KEYS}"
                )

    def __len__(self) -> int:
        return len(self.rows)

    def _resolve_audio_path(self, row: Dict[str, Any]) -> Path:
        value = first_present(row, AUDIO_KEYS)
        if isinstance(value, dict):
            value = value.get("path") or value.get("audio_filepath")
        if not isinstance(value, str):
            raise TypeError(f"unsupported audio field: {type(value)}")
        p = Path(value)
        if not p.is_absolute():
            p = self.base_dir / p
        return p

    def _load_audio(self, path: Path):
        audio, sr = sf.read(str(path), dtype="float32", always_2d=False)
        if audio.ndim == 2:
            audio = audio.mean(axis=1)
        if sr != self.sampling_rate:
            g = math.gcd(int(sr), int(self.sampling_rate))
            audio = resample_poly(
                audio,
                self.sampling_rate // g,
                int(sr) // g,
            ).astype(np.float32, copy=False)
        return audio

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.rows[idx]
        audio_path = self._resolve_audio_path(row)
        if not audio_path.is_file():
            raise FileNotFoundError(audio_path)

        audio = self._load_audio(audio_path)
        text = str(first_present(row, TEXT_KEYS))

        # WhisperFeatureExtractor pads/truncates to Whisper's 30 s input window.
        input_features = self.processor.feature_extractor(
            audio,
            sampling_rate=self.sampling_rate,
            return_tensors="np",
        ).input_features[0]

        labels = self.processor.tokenizer(
            text,
            add_special_tokens=True,
        ).input_ids

        return {
            "input_features": input_features,
            "labels": labels,
        }


@dataclass
class DataCollatorSpeechSeq2Seq:
    processor: WhisperProcessor
    decoder_start_token_id: int

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        input_features = [
            {"input_features": f["input_features"]} for f in features
        ]
        batch = self.processor.feature_extractor.pad(
            input_features, return_tensors="pt"
        )

        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(
            label_features, return_tensors="pt"
        )
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch["attention_mask"].ne(1), -100
        )

        # Whisper internally supplies decoder_start_token_id.
        if (
            labels.shape[1] > 0
            and (labels[:, 0] == self.decoder_start_token_id).all().item()
        ):
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch


def count_params(model):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def main() -> None:
    args = parse_args()

    if args.bf16 and args.fp16:
        raise ValueError("choose at most one of --bf16 and --fp16")

    out = Path(args.output_dir).resolve()
    trainer_dir = out / "trainer"
    adapter_dir = out / "adapter"
    merged_dir = out / "merged_model"

    if out.exists() and args.overwrite:
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    processor = WhisperProcessor.from_pretrained(
        args.model,
        revision=args.revision,
        language=args.language,
        task=args.task,
    )
    # Explicitly fix the tokenizer prompt for English transcription.
    if hasattr(processor.tokenizer, "set_prefix_tokens"):
        processor.tokenizer.set_prefix_tokens(
            language=args.language,
            task=args.task,
        )

    model = WhisperForConditionalGeneration.from_pretrained(
        args.model,
        revision=args.revision,
        torch_dtype=torch.bfloat16 if args.bf16 else None,
    )
    model.config.use_cache = False
    model.generation_config.language = args.language
    model.generation_config.task = args.task

    alpha = args.lora_alpha if args.lora_alpha > 0 else 2 * args.rank
    targets = [x.strip() for x in args.target_modules.split(",") if x.strip()]
    if not targets:
        raise ValueError("--target-modules cannot be empty")

    lora_cfg = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=args.rank,
        lora_alpha=alpha,
        lora_dropout=args.lora_dropout,
        target_modules=targets,
        bias="none",
    )
    model = get_peft_model(model, lora_cfg)

    trainable, total = count_params(model)
    print(
        f"LoRA rank={args.rank} alpha={alpha} targets={targets}\n"
        f"trainable={trainable:,} / total={total:,} "
        f"({100.0 * trainable / total:.4f}%)",
        flush=True,
    )

    train_ds = JsonlSpeechDataset(
        args.train_manifest, processor, args.sampling_rate
    )
    valid_ds = JsonlSpeechDataset(
        args.valid_manifest, processor, args.sampling_rate
    )
    print(
        f"train examples={len(train_ds):,}; valid examples={len(valid_ds):,}",
        flush=True,
    )

    collator = DataCollatorSpeechSeq2Seq(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
    )

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(trainer_dir),
        overwrite_output_dir=False,
        do_train=True,
        do_eval=True,
        eval_strategy="steps",
        save_strategy="steps",
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="linear",
        optim="adamw_torch_fused",
        bf16=args.bf16,
        fp16=args.fp16,
        tf32=args.tf32,
        gradient_checkpointing=args.gradient_checkpointing,
        gradient_checkpointing_kwargs=(
            {"use_reentrant": False} if args.gradient_checkpointing else None
        ),
        remove_unused_columns=False,
        dataloader_num_workers=args.dataloader_workers,
        dataloader_pin_memory=True,
        report_to=[],
        predict_with_generate=False,
        seed=args.seed,
        data_seed=args.seed,
        ddp_find_unused_parameters=False,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=valid_ds,
        data_collator=collator,
        processing_class=processor,
    )

    train_result = trainer.train(
        resume_from_checkpoint=args.resume_from_checkpoint
    )
    trainer.accelerator.wait_for_everyone()

    # Save the best-loaded LoRA adapter.
    if trainer.is_world_process_zero():
        adapter_dir.mkdir(parents=True, exist_ok=True)
        trainer.model.save_pretrained(
            adapter_dir,
            safe_serialization=True,
        )
        processor.save_pretrained(adapter_dir)

        summary = {
            "base_model": args.model,
            "revision": args.revision,
            "rank": args.rank,
            "lora_alpha": alpha,
            "lora_dropout": args.lora_dropout,
            "target_modules": targets,
            "seed": args.seed,
            "train_examples": len(train_ds),
            "valid_examples": len(valid_ds),
            "trainable_parameters": trainable,
            "total_parameters_with_adapter": total,
            "best_model_checkpoint": trainer.state.best_model_checkpoint,
            "best_metric_eval_loss": trainer.state.best_metric,
            "train_metrics": train_result.metrics,
            "train_manifest": str(Path(args.train_manifest).resolve()),
            "valid_manifest": str(Path(args.valid_manifest).resolve()),
        }
        (out / "train_summary.json").write_text(
            json.dumps(summary, indent=2, default=str) + "\n"
        )

    trainer.accelerator.wait_for_everyone()

    # Merge only on global rank 0. This produces a plain Whisper model that the
    # existing CSA long-form evaluator can load through --model.
    if args.merge and trainer.is_world_process_zero():
        print(f"Merging LoRA adapter into base model -> {merged_dir}", flush=True)

        # Free the training copy first to reduce peak GPU/CPU memory.
        del trainer
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        base = WhisperForConditionalGeneration.from_pretrained(
            args.model,
            revision=args.revision,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
        )
        peft_model = PeftModel.from_pretrained(base, str(adapter_dir))
        merged = peft_model.merge_and_unload(safe_merge=True)
        merged.generation_config.language = args.language
        merged.generation_config.task = args.task

        merged_dir.mkdir(parents=True, exist_ok=True)
        merged.save_pretrained(
            merged_dir,
            safe_serialization=True,
            max_shard_size="5GB",
        )
        processor.save_pretrained(merged_dir)
        print(f"Merged model saved: {merged_dir}", flush=True)

    if trainer.is_world_process_zero() if 'trainer' in locals() else True:
        print(f"Training complete: {out}", flush=True)


if __name__ == "__main__":
    main()
