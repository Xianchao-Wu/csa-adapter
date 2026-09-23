from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .config import CSAAdapterConfig


class LowRankProjection(nn.Module):
    """A rank-r projection ``x @ A.T @ B.T`` without a dense base matrix."""

    def __init__(self, in_features: int, out_features: int, rank: int) -> None:
        super().__init__()
        self.down = nn.Linear(in_features, rank, bias=False)
        self.up = nn.Linear(rank, out_features, bias=False)
        nn.init.kaiming_uniform_(self.down.weight, a=math.sqrt(5))
        nn.init.normal_(self.up.weight, std=0.02 / math.sqrt(max(rank, 1)))

    def forward(self, x: Tensor) -> Tensor:
        return self.up(self.down(x))


class WindowCompressor(nn.Module):
    """Compress ``[B,T,D]`` to ``[B,ceil(T/C),Da]``.

    It returns a boolean memory mask and the inclusive end position of every
    source window. The latter is used to prevent future leakage in causal mode.
    """

    def __init__(self, cfg: CSAAdapterConfig) -> None:
        super().__init__()
        self.rate = cfg.compression_rate
        self.kind = cfg.compressor
        self.proj = nn.Linear(cfg.model_dim, cfg.adapter_dim, bias=False)
        self.event_score = nn.Linear(cfg.model_dim, 1, bias=False) if self.kind == "event" else None
        self.event_proj = (
            nn.Linear(cfg.model_dim, cfg.adapter_dim, bias=False) if self.kind == "event" else None
        )
        self.depthwise = None
        if self.kind == "depthwise":
            self.depthwise = nn.Conv1d(
                cfg.model_dim,
                cfg.model_dim,
                kernel_size=self.rate,
                stride=self.rate,
                groups=cfg.model_dim,
                bias=False,
            )

    def forward(
        self, hidden: Tensor, padding_mask: Tensor | None = None
    ) -> tuple[Tensor, Tensor, Tensor]:
        batch, time, dim = hidden.shape
        if padding_mask is None:
            padding_mask = torch.ones(batch, time, device=hidden.device, dtype=torch.bool)
        else:
            padding_mask = padding_mask.to(device=hidden.device, dtype=torch.bool)
            if padding_mask.shape != (batch, time):
                raise ValueError(f"padding_mask must have shape {(batch, time)}")

        windows = math.ceil(time / self.rate)
        padded_time = windows * self.rate
        pad = padded_time - time
        x = F.pad(hidden, (0, 0, 0, pad))
        mask = F.pad(padding_mask, (0, pad), value=False)
        xw = x.view(batch, windows, self.rate, dim)
        mw = mask.view(batch, windows, self.rate)
        counts = mw.sum(dim=-1, keepdim=True).clamp_min(1).to(x.dtype)

        if self.kind == "depthwise":
            # Mask padded frames before the strided depthwise convolution.
            conv = self.depthwise((x * mask.unsqueeze(-1)).transpose(1, 2)).transpose(1, 2)
            pooled = conv
        else:
            mean = (xw * mw.unsqueeze(-1)).sum(dim=2) / counts
            if self.kind == "event":
                logits = self.event_score(xw).squeeze(-1)
                logits = logits.masked_fill(~mw, torch.finfo(logits.dtype).min)
                weights = torch.softmax(logits, dim=-1)
                weights = torch.nan_to_num(weights)
                event = (weights.unsqueeze(-1) * xw).sum(dim=2)
                pooled = mean
            else:
                pooled = mean

        memory_mask = mw.any(dim=-1)
        window_end = torch.arange(windows, device=hidden.device) * self.rate + self.rate - 1
        window_end = window_end.clamp_max(time - 1)
        compressed = self.proj(pooled)
        if self.kind == "event":
            compressed = compressed + self.event_proj(event)
        compressed = compressed * memory_mask.unsqueeze(-1)
        return compressed, memory_mask, window_end


@dataclass
class CSARouting:
    indices: Tensor  # [B,H,T,K]
    probabilities: Tensor  # [B,H,T,K]
    valid: Tensor  # [B,H,T,K]


class CSAAdapter(nn.Module):
    """Compressed sparse attention residual for an ASR encoder.

    The index search is exact but chunked over queries and memory, so it never
    materializes the complete ``[B,H,T,M]`` score tensor.
    """

    def __init__(self, cfg: CSAAdapterConfig) -> None:
        super().__init__()
        self.cfg = cfg
        h = cfg.num_heads
        self.compressor = WindowCompressor(cfg)
        self.norm = nn.LayerNorm(cfg.model_dim)
        self.q_proj = LowRankProjection(cfg.model_dim, h * cfg.index_dim, cfg.rank)
        self.k_proj = LowRankProjection(cfg.adapter_dim, h * cfg.index_dim, cfg.rank)
        self.v_proj = LowRankProjection(cfg.adapter_dim, h * cfg.value_dim, cfg.rank)
        self.context_k = None
        self.context_v = None
        if cfg.use_context_bank:
            context_dim = cfg.context_dim or cfg.model_dim
            self.context_k = LowRankProjection(context_dim, h * cfg.index_dim, cfg.rank)
            self.context_v = LowRankProjection(context_dim, h * cfg.value_dim, cfg.rank)
        self.out_proj = nn.Linear(h * cfg.value_dim, cfg.model_dim, bias=False)
        self.gate = nn.Linear(cfg.model_dim, cfg.model_dim)
        self.dropout = nn.Dropout(cfg.dropout)
        self.alpha = nn.Parameter(torch.zeros(()))
        # Do not zero both alpha and out_proj: that would block all gradients.
        nn.init.normal_(self.out_proj.weight, std=0.02 / math.sqrt(h * cfg.value_dim))
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)

    def _split_heads(self, x: Tensor, head_dim: int) -> Tensor:
        batch, time, _ = x.shape
        return x.view(batch, time, self.cfg.num_heads, head_dim).transpose(1, 2)

    def _chunked_topk(
        self,
        q: Tensor,
        k: Tensor,
        memory_mask: Tensor,
        memory_end: Tensor,
        query_offset: int,
    ) -> tuple[Tensor, Tensor]:
        """Return exact top-k scores/indices for one query chunk."""
        _, _, q_len, _ = q.shape
        best_scores: Tensor | None = None
        best_indices: Tensor | None = None
        scale = self.cfg.index_dim**-0.5

        for start in range(0, k.size(2), self.cfg.memory_chunk_size):
            end = min(start + self.cfg.memory_chunk_size, k.size(2))
            scores = torch.einsum("bhqd,bhmd->bhqm", q, k[:, :, start:end]) * scale
            valid = memory_mask[:, None, None, start:end]
            if self.cfg.causal:
                query_pos = torch.arange(query_offset, query_offset + q_len, device=q.device).view(
                    1, 1, q_len, 1
                )
                causal_valid = memory_end[start:end].view(1, 1, 1, -1) <= query_pos
                valid = valid & causal_valid
            scores = scores.masked_fill(~valid, float("-inf"))
            indices = torch.arange(start, end, device=q.device).view(1, 1, 1, -1)
            indices = indices.expand_as(scores)

            if best_scores is not None:
                scores = torch.cat((best_scores, scores), dim=-1)
                indices = torch.cat((best_indices, indices), dim=-1)
            take = min(self.cfg.top_k, scores.size(-1))
            best_scores, positions = torch.topk(scores, k=take, dim=-1)
            best_indices = torch.gather(indices, dim=-1, index=positions)
        assert best_scores is not None and best_indices is not None
        return best_scores, best_indices

    @staticmethod
    def _gather_values(values: Tensor, indices: Tensor) -> Tensor:
        # Advanced indexing avoids expanding values to [B,H,T,M,Dv].
        batch, heads, _, _ = values.shape
        b = torch.arange(batch, device=values.device)[:, None, None, None]
        h = torch.arange(heads, device=values.device)[None, :, None, None]
        return values[b, h, indices]

    def forward(
        self,
        hidden: Tensor,
        padding_mask: Tensor | None = None,
        context_bank: Tensor | None = None,
        context_mask: Tensor | None = None,
        return_routing: bool = False,
    ) -> Tensor | tuple[Tensor, CSARouting]:
        residual_input = hidden
        hidden = self.norm(hidden)
        memory, memory_mask, memory_end = self.compressor(hidden, padding_mask)

        q = self._split_heads(self.q_proj(hidden), self.cfg.index_dim)
        k = self._split_heads(self.k_proj(memory), self.cfg.index_dim)
        v = self._split_heads(self.v_proj(memory), self.cfg.value_dim)

        if context_bank is not None:
            if self.context_k is None or self.context_v is None:
                raise ValueError("context_bank was provided but use_context_bank=False")
            ck = self._split_heads(self.context_k(context_bank), self.cfg.index_dim)
            cv = self._split_heads(self.context_v(context_bank), self.cfg.value_dim)
            if context_mask is None:
                context_mask = torch.ones(
                    context_bank.shape[:2], device=hidden.device, dtype=torch.bool
                )
            k = torch.cat((k, ck), dim=2)
            v = torch.cat((v, cv), dim=2)
            memory_mask = torch.cat((memory_mask, context_mask.to(torch.bool)), dim=1)
            # Context is supplied before recognition and is always visible.
            context_end = torch.full(
                (context_bank.size(1),), -1, device=hidden.device, dtype=memory_end.dtype
            )
            memory_end = torch.cat((memory_end, context_end), dim=0)

        outputs, all_idx, all_prob, all_valid = [], [], [], []
        for start in range(0, hidden.size(1), self.cfg.query_chunk_size):
            end = min(start + self.cfg.query_chunk_size, hidden.size(1))
            scores, indices = self._chunked_topk(
                q[:, :, start:end], k, memory_mask, memory_end, start
            )
            valid = torch.isfinite(scores)
            safe_scores = scores.masked_fill(~valid, -1e4)
            probs = torch.softmax(safe_scores.float(), dim=-1).to(scores.dtype)
            probs = probs * valid.to(probs.dtype)
            probs = (probs.float() / probs.float().sum(dim=-1, keepdim=True).clamp_min(1e-9)).to(
                scores.dtype
            )
            selected = self._gather_values(v, indices)
            out = (probs.unsqueeze(-1) * selected).sum(dim=-2)
            outputs.append(out)
            if return_routing:
                all_idx.append(indices)
                all_prob.append(probs)
                all_valid.append(valid)

        out = torch.cat(outputs, dim=2).transpose(1, 2).contiguous()
        out = out.view(hidden.size(0), hidden.size(1), -1)
        correction = torch.sigmoid(self.gate(residual_input)) * self.out_proj(out)
        correction = self.alpha * self.dropout(correction)
        if padding_mask is not None:
            correction = correction * padding_mask.unsqueeze(-1).to(correction.dtype)

        if not return_routing:
            return correction
        routing = CSARouting(
            indices=torch.cat(all_idx, dim=2),
            probabilities=torch.cat(all_prob, dim=2),
            valid=torch.cat(all_valid, dim=2),
        )
        return correction, routing
