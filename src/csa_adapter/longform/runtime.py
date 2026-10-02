import contextlib
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from transformers import AutoProcessor, WhisperForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutput

from .memory import MemoryConfig, PersistentCSA


ADAPTER_FORMAT = "persistent-csa-v0.2.1"


def amp(device):
    return (
        torch.autocast("cuda", dtype=torch.bfloat16)
        if str(device).startswith("cuda")
        else contextlib.nullcontext()
    )


def load_backbone(model_id, device, revision="main"):
    if str(device).startswith("cuda") and not torch.cuda.is_bf16_supported():
        raise RuntimeError("This GPU recipe requires bf16; use --device cpu for functional checks")
    processor = AutoProcessor.from_pretrained(model_id, revision=revision)
    processor.tokenizer.set_prefix_tokens(
        language="english", task="transcribe", predict_timestamps=False
    )
    model = (
        WhisperForConditionalGeneration.from_pretrained(
            model_id,
            revision=revision,
            attn_implementation="sdpa",
            dtype=torch.bfloat16 if str(device).startswith("cuda") else torch.float32,
        )
        .to(device)
        .eval()
    )
    model.requires_grad_(False)
    model.generation_config.forced_decoder_ids = None
    return model, processor


def model_identity(model, model_id):
    return dict(
        model_id=str(model_id),
        revision=getattr(model.config, "_commit_hash", None),
        d_model=model.config.d_model,
        encoder_layers=model.config.encoder_layers,
        max_source_positions=model.config.max_source_positions,
    )


def read_wav(path):
    x, sr = sf.read(path, dtype="float32", always_2d=True)
    if sr != 16000:
        raise ValueError("Prepared audio must be 16kHz")
    return x.mean(-1)


@torch.no_grad()
def encode(model, processor, audio, device):
    if len(audio) > 30 * 16000 or len(audio) == 0:
        raise ValueError("encode expects nonempty audio <=30 seconds; no silent truncation")
    if not np.isfinite(audio).all():
        raise ValueError("Nonfinite waveform")
    f = processor.feature_extractor(audio, sampling_rate=16000, return_tensors="pt").input_features
    with amp(device):
        h = model.model.encoder(
            f.to(device=device, dtype=next(model.parameters()).dtype)
        ).last_hidden_state
    nvalid = min(h.shape[1], (len(audio) + 319) // 320)
    return h, nvalid


def adapt_valid(
    adapter,
    hidden,
    nvalid,
    memory,
    dense=False,
    temperature=1.0,
    return_diagnostics=False,
):
    # Padded encoder states are retained for compatibility with the frozen decoder,
    # but never written into acoustic memory or modified by the adapter.
    if not 0 <= nvalid <= hidden.shape[1]:
        raise ValueError("nvalid is outside encoder sequence length")
    result = adapter(
        hidden[:, :nvalid],
        memory,
        dense=dense,
        temperature=temperature,
        return_diagnostics=return_diagnostics,
    )
    if return_diagnostics:
        changed, stats = result
        return torch.cat([changed, hidden[:, nvalid:]], dim=1), stats
    return torch.cat([result, hidden[:, nvalid:]], dim=1)


def labels_for(processor, text, model):
    ids = processor.tokenizer(text).input_ids
    if ids and ids[0] == model.config.decoder_start_token_id:
        ids = ids[1:]
    if len(ids) > model.config.max_target_positions:
        raise ValueError(
            f"Transcript has {len(ids)} tokens, exceeding decoder limit; no truncation"
        )
    return torch.tensor([ids], dtype=torch.long, device=next(model.parameters()).device)


def asr_loss(model, adapted, labels):
    # Frozen decoder must NOT run under no_grad: input gradients train the adapter.
    return model(
        encoder_outputs=BaseModelOutput(last_hidden_state=adapted), labels=labels, use_cache=False
    ).loss


def save_adapter(path, adapter, identity, info):
    from safetensors.torch import save_file

    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    save_file(
        {k: v.detach().cpu().contiguous() for k, v in adapter.state_dict().items()},
        str(path / "adapter.safetensors"),
    )
    (path / "adapter_config.json").write_text(
        json.dumps(
            dict(
                format=ADAPTER_FORMAT,
                config=asdict(adapter.cfg),
                backbone=identity,
                training=info,
            ),
            indent=2,
        )
    )


def load_adapter(path, identity, device):
    from safetensors.torch import load_file

    path = Path(path)
    data = json.loads((path / "adapter_config.json").read_text())
    if data.get("format") != ADAPTER_FORMAT:
        raise ValueError(
            f"Adapter format {data.get('format')!r} is not {ADAPTER_FORMAT!r}. "
            "v0.2.1 changes residual stability semantics; retrain instead of silently migrating."
        )
    if data["backbone"] != identity:
        raise ValueError("Adapter/backbone identity mismatch; use exact model and revision")
    module = PersistentCSA(MemoryConfig(**data["config"])).to(device)
    module.load_state_dict(load_file(str(path / "adapter.safetensors")), strict=True)
    if not all(torch.isfinite(p).all() for p in module.parameters()):
        raise FloatingPointError("Adapter checkpoint contains non-finite parameters")
    return module.eval()
