"""Historical-only CSA after the final frozen Whisper encoder output.

No state is mutated by forward: commit only after decoding succeeds.
Hard selection runs without autograd; selected scores are recomputed with
explicit gradients. This is the exact piecewise gradient of hard top-k.
"""

import math
from dataclasses import asdict, dataclass

import torch
from torch import nn

from ..config import CSAAdapterConfig
from ..modules import LowRankProjection, WindowCompressor


@dataclass
class MemoryConfig:
    model_dim: int = 1280
    adapter_dim: int = 128
    index_dim: int = 32
    value_dim: int = 32
    rank: int = 16
    heads: int = 4
    compression_rate: int = 8
    top_k: int = 16
    max_memory: int = 4096
    query_chunk: int = 128
    memory_chunk: int = 512
    compressor: str = "event"
    gate: str = "diagonal"  # 2D parameters, rather than D^2 + D
    alpha_init: float = 0.01

    def __post_init__(self):
        for key, value in asdict(self).items():
            if isinstance(value, int) and value <= 0:
                raise ValueError(f"{key} must be positive")
        if self.gate not in {"diagonal", "none"}:
            raise ValueError("gate must be diagonal or none")


class PersistentCSA(nn.Module):
    def __init__(self, cfg: MemoryConfig):
        super().__init__()
        self.cfg = cfg
        self.norm = nn.LayerNorm(cfg.model_dim)
        self.compressor = WindowCompressor(
            CSAAdapterConfig(
                model_dim=cfg.model_dim,
                adapter_dim=cfg.adapter_dim,
                compression_rate=cfg.compression_rate,
                compressor=cfg.compressor,
            )
        )
        self.q = LowRankProjection(cfg.model_dim, cfg.heads * cfg.index_dim, cfg.rank)
        self.k = LowRankProjection(cfg.adapter_dim, cfg.heads * cfg.index_dim, cfg.rank)
        self.v = LowRankProjection(cfg.adapter_dim, cfg.heads * cfg.value_dim, cfg.rank)
        self.out = nn.Linear(cfg.heads * cfg.value_dim, cfg.model_dim, bias=False)
        self.alpha = nn.Parameter(torch.tensor(cfg.alpha_init))
        if cfg.gate == "diagonal":
            self.gate_weight = nn.Parameter(torch.zeros(cfg.model_dim))
            self.gate_bias = nn.Parameter(torch.zeros(cfg.model_dim))

    def compress(self, raw_hidden):
        """[1,T_valid,D] -> [1,ceil(T_valid/C),Da]; gradients reach writer."""
        z, _, _ = self.compressor(self.norm(raw_hidden))
        return z

    def historical_memory(self, raw_history):
        """Recompute writer from detached *raw* features at every train step.

        Do NOT detach the resulting z: current ASR loss must train the writer.
        Windows never straddle segment boundaries. Old raw segments are dropped
        before projection when their windows cannot survive the FIFO cap.
        """
        chosen, count = [], 0
        for h in reversed(raw_history):
            chosen.append(h)
            count += math.ceil(h.shape[1] / self.cfg.compression_rate)
            if count >= self.cfg.max_memory:
                break
        if not chosen:
            return None
        return torch.cat([self.compress(h.detach()) for h in reversed(chosen)], 1)[
            :, -self.cfg.max_memory :
        ]

    def heads(self, x, dim):
        return x.reshape(x.shape[0], x.shape[1], self.cfg.heads, dim).transpose(1, 2)

    @staticmethod
    def gather(x, idx):
        b = torch.arange(x.shape[0], device=x.device)[:, None, None, None]
        h = torch.arange(x.shape[1], device=x.device)[None, :, None, None]
        return x[b, h, idx]

    @torch.no_grad()
    def select(self, q, keys):
        best, indices = None, None
        for start in range(0, keys.shape[2], self.cfg.memory_chunk):
            block = keys[:, :, start : start + self.cfg.memory_chunk]
            score = torch.matmul(q.float(), block.float().transpose(-1, -2))
            ix = torch.arange(start, start + block.shape[2], device=q.device)
            ix = ix.view(1, 1, 1, -1).expand_as(score)
            if best is not None:
                score, ix = torch.cat([best, score], -1), torch.cat([indices, ix], -1)
            best, pos = score.topk(min(self.cfg.top_k, score.shape[-1]), dim=-1)
            indices = ix.gather(-1, pos)
        return indices

    def forward(self, raw_hidden, memory=None, *, dense=False, temperature=1.0):
        """Return adapted [B,T,D], preserving padded encoder tail as supplied.

        Memory has no invalid/padded slots; mini-batching uses one call at a time.
        Empty history is an exact identity. No current segment writes occur here.
        """
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        if memory is None or memory.shape[1] == 0:
            return raw_hidden
        hidden = self.norm(raw_hidden)
        q = self.heads(self.q(hidden), self.cfg.index_dim)
        keys = self.heads(self.k(memory), self.cfg.index_dim)
        values = self.heads(self.v(memory), self.cfg.value_dim)
        outputs = []
        for start in range(0, q.shape[2], self.cfg.query_chunk):
            qq = q[:, :, start : start + self.cfg.query_chunk]
            if dense:
                scores = torch.matmul(qq.float(), keys.float().transpose(-1, -2))
                p = (scores / (math.sqrt(self.cfg.index_dim) * temperature)).softmax(-1)
                out = torch.matmul(p.to(values.dtype), values)
            else:
                idx = self.select(qq, keys)
                selected_keys = self.gather(keys, idx)
                scores = (qq.float().unsqueeze(-2) * selected_keys.float()).sum(-1)
                p = (scores / (math.sqrt(self.cfg.index_dim) * temperature)).softmax(-1)
                out = (p.to(values.dtype).unsqueeze(-1) * self.gather(values, idx)).sum(-2)
            outputs.append(out)
        out = (
            torch.cat(outputs, 2)
            .transpose(1, 2)
            .reshape(raw_hidden.shape[0], raw_hidden.shape[1], -1)
        )
        gate = (
            1.0
            if self.cfg.gate == "none"
            else torch.sigmoid(hidden * self.gate_weight + self.gate_bias)
        )
        return raw_hidden + self.alpha * gate * self.out(out)


class MemoryBank:
    """One ordered call per state object. No persistent state in checkpoints."""

    def __init__(self, max_memory):
        self.max_memory = max_memory
        self.reset(None)

    def reset(self, call_id):
        self.call_id = call_id
        self.memory = None
        self.committed = set()
        self.last_end = -float("inf")

    def commit(self, call_id, segment_id, start, end, z):
        if call_id != self.call_id:
            raise ValueError("Call changed: explicitly reset memory before decoding")
        if segment_id in self.committed:
            raise ValueError("Duplicate segment commit")
        if start < self.last_end - 1e-4 or end <= start:
            raise ValueError("Overlapping or out-of-order commit")
        if z.shape[0] != 1:
            raise ValueError("One session per MemoryBank")
        z = z.detach()
        self.memory = z if self.memory is None else torch.cat([self.memory, z], 1)
        self.memory = self.memory[:, -self.max_memory :]
        self.committed.add(segment_id)
        self.last_end = end
