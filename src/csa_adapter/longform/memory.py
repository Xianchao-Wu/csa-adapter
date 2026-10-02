"""Historical-only persistent CSA after the final frozen Whisper encoder output.

v0.2.1 adds two stability guarantees for long-form decoding:

1. the learned scalar residual strength is smoothly bounded; and
2. the *applied* residual is limited to a configurable fraction of the
   per-token RMS of the frozen encoder representation.

Forward never mutates persistent state.  Memory is committed only after a
segment has decoded successfully.
"""

import math
from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.nn import functional as F

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
    gate: str = "diagonal"  # O(D) parameters, rather than D^2 + D

    # Stability controls.  alpha is stored as an unconstrained raw parameter,
    # but forward maps it through tanh to [-alpha_max, alpha_max].
    alpha_init: float = 0.01
    alpha_max: float = 0.10
    gate_bias_init: float = -2.0
    residual_ratio_cap: float = 0.25
    residual_eps: float = 1e-6

    def __post_init__(self):
        positive_ints = {
            k: v for k, v in asdict(self).items() if isinstance(v, int) and not isinstance(v, bool)
        }
        for key, value in positive_ints.items():
            if value <= 0:
                raise ValueError(f"{key} must be positive")
        if self.gate not in {"diagonal", "none"}:
            raise ValueError("gate must be diagonal or none")
        if self.compressor not in {"mean", "event", "depthwise"}:
            raise ValueError("compressor must be mean, event, or depthwise")
        if not math.isfinite(self.alpha_init):
            raise ValueError("alpha_init must be finite")
        if not math.isfinite(self.alpha_max) or self.alpha_max <= 0:
            raise ValueError("alpha_max must be finite and positive")
        if abs(self.alpha_init) >= self.alpha_max:
            raise ValueError("abs(alpha_init) must be smaller than alpha_max")
        if not math.isfinite(self.gate_bias_init):
            raise ValueError("gate_bias_init must be finite")
        if not math.isfinite(self.residual_ratio_cap) or self.residual_ratio_cap < 0:
            raise ValueError("residual_ratio_cap must be finite and non-negative")
        if not math.isfinite(self.residual_eps) or self.residual_eps <= 0:
            raise ValueError("residual_eps must be finite and positive")


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

        # Store alpha in the same state-dict key as v0.2, while making the
        # effective value exactly cfg.alpha_init at initialization.
        raw_alpha = cfg.alpha_max * math.atanh(cfg.alpha_init / cfg.alpha_max)
        self.alpha = nn.Parameter(torch.tensor(raw_alpha, dtype=torch.float32))

        if cfg.gate == "diagonal":
            self.gate_weight = nn.Parameter(torch.zeros(cfg.model_dim))
            self.gate_bias = nn.Parameter(torch.full((cfg.model_dim,), cfg.gate_bias_init))

        # Smaller output initialization makes the adapter start close to the
        # frozen backbone while still allowing gradients to all writer/reader
        # parameters because alpha_init is non-zero.
        nn.init.normal_(
            self.out.weight,
            std=0.02 / math.sqrt(max(1, cfg.heads * cfg.value_dim)),
        )

    def effective_alpha(self):
        return self.cfg.alpha_max * torch.tanh(self.alpha / self.cfg.alpha_max)

    def compress(self, raw_hidden):
        """[1,T_valid,D] -> [1,ceil(T_valid/C),Da]; gradients reach writer."""
        if raw_hidden.ndim != 3 or raw_hidden.shape[0] != 1:
            raise ValueError("compress expects [1,T,D]")
        z, _, _ = self.compressor(self.norm(raw_hidden))
        if not torch.isfinite(z).all():
            raise FloatingPointError("Non-finite compressed acoustic memory")
        return z

    def historical_memory(self, raw_history):
        """Recompute writer from detached *raw* historical encoder features.

        The detached backbone features are immutable, but the newly compressed
        memory remains differentiable with respect to the current writer.
        Windows never straddle segment boundaries.  Old raw segments are
        dropped before projection when they cannot survive the FIFO cap.
        """
        chosen, count = [], 0
        for h in reversed(raw_history):
            chosen.append(h)
            count += math.ceil(h.shape[1] / self.cfg.compression_rate)
            if count >= self.cfg.max_memory:
                break
        if not chosen:
            return None
        memory = torch.cat([self.compress(h.detach()) for h in reversed(chosen)], 1)
        return memory[:, -self.cfg.max_memory :]

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
        if indices is None:
            raise ValueError("Cannot select from empty memory")
        return indices

    def _apply_residual_cap(self, raw_hidden, delta):
        """Limit each token's residual RMS relative to its frozen state RMS.

        The limiter scale is detached.  It acts as a safety envelope rather
        than a learnable normalization and therefore does not create a path by
        which the optimizer can reduce the base-state norm.
        """
        if self.cfg.residual_ratio_cap <= 0:
            scale = torch.ones_like(delta[..., :1])
            return delta, scale
        base_rms = raw_hidden.float().square().mean(-1, keepdim=True).sqrt()
        delta_rms = delta.float().square().mean(-1, keepdim=True).sqrt()
        limit = self.cfg.residual_ratio_cap * base_rms
        scale = (limit / (delta_rms + self.cfg.residual_eps)).clamp(max=1.0).detach()
        return delta * scale.to(delta.dtype), scale

    def _diagnostics(
        self,
        raw_hidden,
        readout,
        gate,
        delta_pre_cap,
        delta,
        cap_scale,
        memory_entries,
        entropy_sum,
        entropy_count,
        dense,
    ):
        with torch.no_grad():
            eps = self.cfg.residual_eps
            base_rms = raw_hidden.float().square().mean().sqrt()
            readout_rms = readout.float().square().mean().sqrt()
            pre_rms = delta_pre_cap.float().square().mean().sqrt()
            delta_rms = delta.float().square().mean().sqrt()
            adapted = raw_hidden.float() + delta.float()
            cosine = F.cosine_similarity(raw_hidden.float(), adapted, dim=-1).mean()
            entropy = entropy_sum / max(1, entropy_count)
            return {
                "memory_entries_before": int(memory_entries),
                "dense_reading": bool(dense),
                "alpha_raw": float(self.alpha.detach().float().cpu()),
                "alpha_effective": float(self.effective_alpha().detach().float().cpu()),
                "gate_mean": float(gate.float().mean().cpu()),
                "gate_max": float(gate.float().max().cpu()),
                "gate_saturation_gt_0_9": float((gate.float() > 0.9).float().mean().cpu()),
                "base_rms": float(base_rms.cpu()),
                "readout_rms": float(readout_rms.cpu()),
                "delta_pre_cap_rms": float(pre_rms.cpu()),
                "delta_rms": float(delta_rms.cpu()),
                "residual_ratio_pre_cap": float((pre_rms / (base_rms + eps)).cpu()),
                "residual_ratio": float((delta_rms / (base_rms + eps)).cpu()),
                "residual_clip_fraction": float((cap_scale.float() < 0.999999).float().mean().cpu()),
                "adapted_cosine": float(cosine.cpu()),
                "retrieval_entropy": float(entropy),
            }

    def forward(
        self,
        raw_hidden,
        memory=None,
        *,
        dense=False,
        temperature=1.0,
        return_diagnostics=False,
    ):
        """Return adapted [B,T,D], preserving historical-only causality.

        Empty history is an exact identity.  No current-segment write occurs in
        forward; callers commit compressed *raw* states only after decoding.
        """
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        if raw_hidden.ndim != 3:
            raise ValueError("raw_hidden must have shape [B,T,D]")
        if memory is None or memory.shape[1] == 0:
            if not return_diagnostics:
                return raw_hidden
            return raw_hidden, {
                "memory_entries_before": 0,
                "dense_reading": bool(dense),
                "alpha_raw": float(self.alpha.detach().float().cpu()),
                "alpha_effective": float(self.effective_alpha().detach().float().cpu()),
                "gate_mean": 0.0,
                "gate_max": 0.0,
                "gate_saturation_gt_0_9": 0.0,
                "base_rms": float(raw_hidden.detach().float().square().mean().sqrt().cpu()),
                "readout_rms": 0.0,
                "delta_pre_cap_rms": 0.0,
                "delta_rms": 0.0,
                "residual_ratio_pre_cap": 0.0,
                "residual_ratio": 0.0,
                "residual_clip_fraction": 0.0,
                "adapted_cosine": 1.0,
                "retrieval_entropy": 0.0,
            }
        if memory.ndim != 3 or memory.shape[0] != raw_hidden.shape[0]:
            raise ValueError("memory must have shape [B,M,Da] with matching batch")
        if not torch.isfinite(memory).all():
            raise FloatingPointError("Non-finite persistent acoustic memory")

        hidden = self.norm(raw_hidden)
        q = self.heads(self.q(hidden), self.cfg.index_dim)
        keys = self.heads(self.k(memory), self.cfg.index_dim)
        values = self.heads(self.v(memory), self.cfg.value_dim)
        outputs = []
        entropy_sum = 0.0
        entropy_count = 0

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
            if return_diagnostics:
                pp = p.detach().float().clamp_min(1e-12)
                entropy_sum += float((-(pp * pp.log()).sum(-1)).mean().cpu())
                entropy_count += 1
            outputs.append(out)

        out = (
            torch.cat(outputs, 2)
            .transpose(1, 2)
            .reshape(raw_hidden.shape[0], raw_hidden.shape[1], -1)
        )
        readout = self.out(out)
        gate = (
            torch.ones_like(raw_hidden)
            if self.cfg.gate == "none"
            else torch.sigmoid(hidden * self.gate_weight + self.gate_bias)
        )
        delta_pre_cap = self.effective_alpha().to(readout.dtype) * gate * readout
        delta, cap_scale = self._apply_residual_cap(raw_hidden, delta_pre_cap)
        adapted = raw_hidden + delta

        if not torch.isfinite(adapted).all():
            raise FloatingPointError("Non-finite adapted encoder state")
        if not return_diagnostics:
            return adapted
        stats = self._diagnostics(
            raw_hidden,
            readout,
            gate,
            delta_pre_cap,
            delta,
            cap_scale,
            memory.shape[1],
            entropy_sum,
            entropy_count,
            dense,
        )
        return adapted, stats


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
        if z.ndim != 3 or z.shape[0] != 1:
            raise ValueError("One session per MemoryBank; z must be [1,M,Da]")
        if not torch.isfinite(z).all():
            raise FloatingPointError("Cannot commit non-finite memory")
        z = z.detach()
        self.memory = z if self.memory is None else torch.cat([self.memory, z], 1)
        self.memory = self.memory[:, -self.max_memory :]
        self.committed.add(segment_id)
        self.last_end = end
