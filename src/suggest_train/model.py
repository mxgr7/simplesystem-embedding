"""Small decoder-only LM for autocomplete.

Reference shape (from the plan):
  * 6 layers × 384 hidden × 6 heads × 4× FFN ≈ 13.7M trainable params
  * RoPE positional encoding on Q/K
  * Tied input/output embeddings
  * 64-token context
  * Pre-norm RMSNorm, GELU FFN

We use ``torch.nn.functional.scaled_dot_product_attention`` so the model
runs efficiently on the 4090 (FlashAttention kernels) without us hand-
writing a kernel.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class LMConfig:
    vocab_size: int
    n_layers: int = 6
    d_model: int = 384
    n_heads: int = 6
    d_ff: int | None = None  # default: 4 * d_model
    max_seq_len: int = 64
    dropout: float = 0.0
    rope_base: float = 10000.0
    pad_id: int = 0
    bos_id: int = 2
    eos_id: int = 3
    sep_id: int = 4

    @property
    def head_dim(self) -> int:
        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model={self.d_model} not divisible by n_heads={self.n_heads}"
            )
        return self.d_model // self.n_heads

    @property
    def ffn_dim(self) -> int:
        return self.d_ff if self.d_ff is not None else 4 * self.d_model


def build_rope_cache(
    seq_len: int, head_dim: int, base: float, device, dtype
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pre-compute the RoPE cos / sin tables for ``seq_len`` positions."""
    if head_dim % 2 != 0:
        raise ValueError("RoPE requires an even head_dim")
    inv_freq = 1.0 / (
        base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim)
    )
    t = torch.arange(seq_len, device=device).float()
    freqs = torch.einsum("i,j->ij", t, inv_freq)  # (seq_len, head_dim/2)
    cos = freqs.cos().to(dtype)
    sin = freqs.sin().to(dtype)
    return cos, sin


def apply_rope(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    """Apply RoPE rotation to ``x`` of shape (B, H, T, D).

    ``cos`` / ``sin`` are (T, D/2) tables; we broadcast across batch and head.
    """
    # Split last dim into pairs.
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    cos = cos.unsqueeze(0).unsqueeze(0)  # (1, 1, T, D/2)
    sin = sin.unsqueeze(0).unsqueeze(0)
    rotated_x1 = x1 * cos - x2 * sin
    rotated_x2 = x2 * cos + x1 * sin
    out = torch.empty_like(x)
    out[..., 0::2] = rotated_x1
    out[..., 1::2] = rotated_x2
    return out


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * rms).to(x.dtype) * self.weight


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: LMConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.dropout = cfg.dropout

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        b, t, _ = x.shape
        h = self.cfg.n_heads
        d = self.cfg.head_dim
        qkv = self.qkv(x).view(b, t, 3, h, d).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = apply_rope(q, cos[:t], sin[:t])
        k = apply_rope(k, cos[:t], sin[:t])
        # SDPA handles causal masking + optional key-padding mask.
        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=attn_mask is None,
        )
        out = out.transpose(1, 2).contiguous().view(b, t, self.cfg.d_model)
        return self.proj(out)


class FeedForward(nn.Module):
    def __init__(self, cfg: LMConfig) -> None:
        super().__init__()
        ff = cfg.ffn_dim
        self.fc1 = nn.Linear(cfg.d_model, ff, bias=False)
        self.fc2 = nn.Linear(ff, cfg.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(x)))


class Block(nn.Module):
    def __init__(self, cfg: LMConfig) -> None:
        super().__init__()
        self.norm1 = RMSNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.norm2 = RMSNorm(cfg.d_model)
        self.ffn = FeedForward(cfg)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), cos, sin, attn_mask)
        x = x + self.ffn(self.norm2(x))
        return x


class SuggestLM(nn.Module):
    def __init__(self, cfg: LMConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.norm = RMSNorm(cfg.d_model)
        # Tied I/O: lm_head shares weights with the input embedding table.
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.embed.weight

        self._rope_cache: dict[
            tuple[torch.device, torch.dtype, int],
            tuple[torch.Tensor, torch.Tensor],
        ] = {}

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def rope(
        self, seq_len: int, device, dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        key = (device, dtype, self.cfg.max_seq_len)
        cached = self._rope_cache.get(key)
        if cached is None:
            cached = build_rope_cache(
                self.cfg.max_seq_len, self.cfg.head_dim, self.cfg.rope_base,
                device=device, dtype=dtype,
            )
            self._rope_cache[key] = cached
        cos, sin = cached
        return cos[:seq_len], sin[:seq_len]

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        b, t = input_ids.shape
        if t > self.cfg.max_seq_len:
            raise ValueError(
                f"input length {t} exceeds max_seq_len {self.cfg.max_seq_len}"
            )
        x = self.embed(input_ids)
        cos, sin = self.rope(t, x.device, x.dtype)

        attn_mask = None
        if attention_mask is not None:
            # Build a (B, 1, T, T) additive bias = causal AND non-pad.
            kpm = attention_mask[:, None, None, :].to(dtype=torch.bool)
            causal = torch.ones(
                t, t, dtype=torch.bool, device=input_ids.device
            ).tril()
            allowed = kpm & causal[None, None, :, :]
            attn_mask = torch.zeros(
                b, 1, t, t, dtype=x.dtype, device=input_ids.device
            )
            attn_mask = attn_mask.masked_fill(~allowed, float("-inf"))

        for block in self.blocks:
            x = block(x, cos, sin, attn_mask)
        x = self.norm(x)
        return self.lm_head(x)
