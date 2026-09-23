from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from safetensors.torch import load_file, save_file
from torch import Tensor, nn

from .config import CSAAdapterConfig
from .modules import CSAAdapter


class WhisperEncoderLayerWithCSA(nn.Module):
    """Version-tolerant wrapper around a Hugging Face Whisper encoder layer."""

    def __init__(self, base_layer: nn.Module, adapter: CSAAdapter) -> None:
        super().__init__()
        self.base_layer = base_layer
        self.csa_adapter = adapter
        self._context_bank: Tensor | None = None
        self._context_mask: Tensor | None = None

    def set_context(self, states: Tensor | None, mask: Tensor | None = None) -> None:
        self._context_bank = states
        self._context_mask = mask

    def forward(self, hidden_states: Tensor, *args: Any, **kwargs: Any) -> Any:
        result = self.base_layer(hidden_states, *args, **kwargs)
        base_hidden = result[0] if isinstance(result, tuple) else result

        # Whisper's internal attention mask is commonly 4-D/additive. The CSA
        # module accepts only a 2-D valid-frame mask, so use it only when safe.
        mask = kwargs.get("attention_mask")
        if mask is not None and mask.ndim != 2:
            mask = None
        correction = self.csa_adapter(
            base_hidden,
            padding_mask=mask,
            context_bank=self._context_bank,
            context_mask=self._context_mask,
        )
        adapted = base_hidden + correction
        if isinstance(result, tuple):
            return (adapted, *result[1:])
        return adapted


def _resolve_layers(indices: list[int], number_of_layers: int) -> list[int]:
    resolved = sorted({i if i >= 0 else number_of_layers + i for i in indices})
    if not resolved or resolved[0] < 0 or resolved[-1] >= number_of_layers:
        raise IndexError(f"layer_indices={indices} invalid for {number_of_layers} layers")
    return resolved


def patch_whisper_encoder(model: nn.Module, cfg: CSAAdapterConfig) -> nn.Module:
    """Freeze ``model`` and inject CSA branches into selected encoder layers."""
    encoder = model.model.encoder
    cfg.model_dim = int(model.config.d_model)
    for parameter in model.parameters():
        parameter.requires_grad = False
    indices = _resolve_layers(cfg.layer_indices, len(encoder.layers))
    for index in indices:
        layer = encoder.layers[index]
        if isinstance(layer, WhisperEncoderLayerWithCSA):
            raise ValueError(f"encoder layer {index} is already patched")
        encoder.layers[index] = WhisperEncoderLayerWithCSA(layer, CSAAdapter(cfg))
    model.csa_adapter_config = cfg.to_dict()
    return model


def iter_csa_adapters(model: nn.Module):
    for name, module in model.named_modules():
        if isinstance(module, CSAAdapter):
            yield name, module


def adapter_parameter_count(model: nn.Module) -> tuple[int, int]:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def set_agent_context(model: nn.Module, states: Tensor | None, mask: Tensor | None = None) -> None:
    """Set precomputed agent-context embeddings on all patched layers."""
    for module in model.modules():
        if isinstance(module, WhisperEncoderLayerWithCSA):
            module.set_context(states, mask)


def _adapter_state_dict(model: nn.Module) -> dict[str, Tensor]:
    return {
        name: tensor.detach().cpu().contiguous()
        for name, tensor in model.state_dict().items()
        if ".csa_adapter." in name
    }


def save_csa_adapter(model: nn.Module, output_dir: str | Path) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    cfg = getattr(model, "csa_adapter_config", None)
    if cfg is None:
        raise ValueError("model has not been patched with CSA-Adapter")
    (output / "adapter_config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    save_file(_adapter_state_dict(model), str(output / "adapter_model.safetensors"))


def load_csa_adapter(model: nn.Module, adapter_dir: str | Path) -> nn.Module:
    source = Path(adapter_dir)
    cfg = CSAAdapterConfig.from_dict(
        json.loads((source / "adapter_config.json").read_text(encoding="utf-8"))
    )
    patch_whisper_encoder(model, cfg)
    state = load_file(str(source / "adapter_model.safetensors"))
    missing, unexpected = model.load_state_dict(state, strict=False)
    adapter_missing = [name for name in missing if ".csa_adapter." in name]
    if adapter_missing or unexpected:
        raise RuntimeError(
            f"adapter load mismatch: missing={adapter_missing}, unexpected={unexpected}"
        )
    return model
