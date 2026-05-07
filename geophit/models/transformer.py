"""Stage 2 hierarchical discrete-token Transformer.

Three modules:

1. ``ContextEncoder`` (E_T): Transformer encoder that maps the historical real
   sequence ``X_p`` to contextualised embeddings ``H_p``.
2. ``BaseDecoder``: autoregressively generates trend tokens ``s_down`` (Eq. 9)
   conditioned on ``H_p`` via cross-attention.
3. ``SelfConditionedDecoder``: autoregressively generates fine-grained target
   tokens ``s_pred`` (Eq. 10) conditioned on both ``s_down`` (trend
   cross-attention) and ``H_p`` (temporal cross-attention).

Greedy sampling is used at inference, matching the paper's deterministic
deployment mode.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 4096) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class ContextEncoder(nn.Module):
    """Transformer encoder over continuous historical channels."""

    def __init__(
        self,
        d_in: int,
        d_model: int = 256,
        num_heads: int = 8,
        num_layers: int = 4,
        dropout: float = 0.1,
        max_len: int = 4096,
    ) -> None:
        super().__init__()
        self.in_proj = nn.Linear(d_in, d_model)
        self.pos = SinusoidalPositionalEncoding(d_model, max_len=max_len)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x_hist: torch.Tensor) -> torch.Tensor:
        h = self.in_proj(x_hist)
        h = self.pos(h)
        h = self.encoder(h)
        return self.norm(h)


def _causal_mask(length: int, device: torch.device) -> torch.Tensor:
    return torch.triu(
        torch.ones(length, length, device=device, dtype=torch.bool), diagonal=1
    )


class _CrossAttnDecoderLayer(nn.Module):
    """Single decoder layer with one or two cross-attention sources."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float,
        num_cross: int = 1,
    ) -> None:
        super().__init__()
        self.num_cross = num_cross
        self.self_attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )
        self.cross_attns = nn.ModuleList(
            [
                nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
                for _ in range(num_cross)
            ]
        )
        self.norm_self = nn.LayerNorm(d_model)
        self.norm_cross = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(num_cross)])
        self.norm_ff = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        memories: list[torch.Tensor],
        causal_mask: torch.Tensor,
    ) -> torch.Tensor:
        if len(memories) != self.num_cross:
            raise ValueError(
                f"expected {self.num_cross} memories, got {len(memories)}"
            )
        h = self.norm_self(x)
        sa, _ = self.self_attn(h, h, h, attn_mask=causal_mask, need_weights=False)
        x = x + self.dropout(sa)
        for ln, attn, mem in zip(self.norm_cross, self.cross_attns, memories):
            h = ln(x)
            ca, _ = attn(h, mem, mem, need_weights=False)
            x = x + self.dropout(ca)
        h = self.norm_ff(x)
        return x + self.dropout(self.ff(h))


class _TokenDecoder(nn.Module):
    """Generic causal decoder operating on discrete tokens."""

    def __init__(
        self,
        codebook_size: int,
        d_model: int,
        num_heads: int,
        num_layers: int,
        dropout: float,
        num_cross: int,
        max_len: int = 4096,
    ) -> None:
        super().__init__()
        self.codebook_size = codebook_size
        self.bos_id = codebook_size  # extra token for start-of-sequence
        self.token_embed = nn.Embedding(codebook_size + 1, d_model)
        self.pos = SinusoidalPositionalEncoding(d_model, max_len=max_len)
        self.layers = nn.ModuleList(
            [
                _CrossAttnDecoderLayer(d_model, num_heads, dropout, num_cross=num_cross)
                for _ in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, codebook_size)

    def _embed_input(self, tokens: torch.Tensor) -> torch.Tensor:
        h = self.token_embed(tokens)
        return self.pos(h)

    def forward(
        self, target_tokens: torch.Tensor, memories: list[torch.Tensor]
    ) -> torch.Tensor:
        """Teacher-forced forward pass.

        ``target_tokens`` shape ``(B, S)``. A BOS token is prepended internally
        and the final position is dropped, so the returned logits have shape
        ``(B, S, K)`` and align with the original token positions.
        """
        b, s = target_tokens.shape
        bos = torch.full(
            (b, 1), self.bos_id, device=target_tokens.device, dtype=torch.long
        )
        inp = torch.cat([bos, target_tokens[:, :-1]], dim=1)
        h = self._embed_input(inp)
        mask = _causal_mask(h.size(1), h.device)
        for layer in self.layers:
            h = layer(h, memories=memories, causal_mask=mask)
        h = self.norm(h)
        return self.head(h)

    @torch.no_grad()
    def generate(
        self, length: int, memories: list[torch.Tensor], greedy: bool = True
    ) -> torch.Tensor:
        b = memories[0].size(0)
        device = memories[0].device
        out = torch.full((b, 0), 0, dtype=torch.long, device=device)
        cur = torch.full((b, 1), self.bos_id, dtype=torch.long, device=device)
        for _ in range(length):
            h = self._embed_input(cur)
            mask = _causal_mask(h.size(1), device)
            for layer in self.layers:
                h = layer(h, memories=memories, causal_mask=mask)
            h = self.norm(h)
            logits = self.head(h[:, -1:])  # (B, 1, K)
            if greedy:
                nxt = logits.argmax(dim=-1)  # (B, 1)
            else:
                probs = logits.softmax(dim=-1).squeeze(1)
                nxt = torch.multinomial(probs, num_samples=1)
            out = torch.cat([out, nxt], dim=1)
            cur = torch.cat([cur, nxt], dim=1)
        return out


class BaseDecoder(_TokenDecoder):
    """Generates trend tokens conditioned on H_p (single cross-attention)."""

    def __init__(
        self,
        codebook_size: int,
        d_model: int = 256,
        num_heads: int = 8,
        num_layers: int = 4,
        dropout: float = 0.1,
        max_len: int = 4096,
    ) -> None:
        super().__init__(
            codebook_size=codebook_size,
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
            num_cross=1,
            max_len=max_len,
        )


class SelfConditionedDecoder(_TokenDecoder):
    """Generates target tokens conditioned on (s_down, H_p)."""

    def __init__(
        self,
        codebook_size: int,
        d_model: int = 256,
        num_heads: int = 8,
        num_layers: int = 4,
        dropout: float = 0.1,
        trend_codebook_size: int | None = None,
        max_len: int = 4096,
    ) -> None:
        super().__init__(
            codebook_size=codebook_size,
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
            num_cross=2,
            max_len=max_len,
        )
        trend_size = trend_codebook_size or codebook_size
        self.trend_token_embed = nn.Embedding(trend_size, d_model)

    def encode_trend_tokens(self, trend_tokens: torch.Tensor) -> torch.Tensor:
        h = self.trend_token_embed(trend_tokens)
        return self.pos(h)


class HierarchicalTransformer(nn.Module):
    """Wires the three Stage 2 modules together."""

    def __init__(
        self,
        d_in: int,
        codebook_size_target: int,
        codebook_size_trend: int,
        d_model: int = 256,
        num_heads: int = 8,
        num_layers: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.context_encoder = ContextEncoder(
            d_in=d_in,
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
        )
        self.base_decoder = BaseDecoder(
            codebook_size=codebook_size_trend,
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
        )
        self.self_cond_decoder = SelfConditionedDecoder(
            codebook_size=codebook_size_target,
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
            trend_codebook_size=codebook_size_trend,
        )

    def forward(
        self,
        x_hist: torch.Tensor,
        trend_tokens: torch.Tensor,
        target_tokens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h_p = self.context_encoder(x_hist)
        base_logits = self.base_decoder(trend_tokens, memories=[h_p])
        trend_emb = self.self_cond_decoder.encode_trend_tokens(trend_tokens)
        sc_logits = self.self_cond_decoder(target_tokens, memories=[trend_emb, h_p])
        return base_logits, sc_logits

    @torch.no_grad()
    def generate(
        self,
        x_hist: torch.Tensor,
        trend_len: int,
        target_len: int,
        greedy: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h_p = self.context_encoder(x_hist)
        s_down = self.base_decoder.generate(trend_len, memories=[h_p], greedy=greedy)
        trend_emb = self.self_cond_decoder.encode_trend_tokens(s_down)
        s_pred = self.self_cond_decoder.generate(
            target_len, memories=[trend_emb, h_p], greedy=greedy
        )
        return s_down, s_pred
