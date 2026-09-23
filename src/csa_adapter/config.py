from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class CSAAdapterConfig:
    """Configuration for a CSA residual branch.

    ``model_dim`` is normally filled from the backbone. ``layer_indices`` may
    contain negative indices and is resolved when the model is patched.
    """

    model_dim: int = 1280
    adapter_dim: int = 128
    index_dim: int = 64
    value_dim: int = 64
    rank: int = 16
    num_heads: int = 4
    compression_rate: int = 4
    top_k: int = 8
    query_chunk_size: int = 256
    memory_chunk_size: int = 512
    compressor: str = "mean"  # mean | depthwise | event
    dropout: float = 0.0
    causal: bool = False
    layer_indices: list[int] = field(default_factory=lambda: [-1])
    use_context_bank: bool = False
    context_dim: int | None = None

    def __post_init__(self) -> None:
        positive = {
            "model_dim": self.model_dim,
            "adapter_dim": self.adapter_dim,
            "index_dim": self.index_dim,
            "value_dim": self.value_dim,
            "rank": self.rank,
            "num_heads": self.num_heads,
            "compression_rate": self.compression_rate,
            "top_k": self.top_k,
            "query_chunk_size": self.query_chunk_size,
            "memory_chunk_size": self.memory_chunk_size,
        }
        for name, value in positive.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
        if self.compressor not in {"mean", "depthwise", "event"}:
            raise ValueError("compressor must be one of: mean, depthwise, event")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "CSAAdapterConfig":
        return cls(**values)
