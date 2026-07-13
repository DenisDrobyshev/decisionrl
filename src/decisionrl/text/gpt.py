"""A minimal character-level GPT language model + tokenizer + SFT training.

Small enough to train on a CPU in seconds, but a real autoregressive Transformer
LM — the substrate for :mod:`decisionrl.text.rlhf` (RLHF fine-tuning).
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["CharTokenizer", "CharGPT", "sft_train"]


class CharTokenizer:
    """Character-level tokenizer built from a corpus."""

    def __init__(self, text: str) -> None:
        self.chars = sorted(set(text))
        self.stoi = {c: i for i, c in enumerate(self.chars)}
        self.itos = dict(enumerate(self.chars))
        self.vocab_size = len(self.chars)

    def encode(self, s: str) -> List[int]:
        return [self.stoi[c] for c in s if c in self.stoi]

    def decode(self, ids) -> str:
        return "".join(self.itos[int(i)] for i in ids)


class _Block(nn.Module):
    def __init__(self, n_embd: int, n_head: int, dropout: float) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = nn.MultiheadAttention(n_embd, n_head, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd), nn.GELU(), nn.Linear(4 * n_embd, n_embd), nn.Dropout(dropout)
        )

    def forward(self, x, attn_mask):
        h = self.ln1(x)
        a, _ = self.attn(h, h, h, attn_mask=attn_mask, need_weights=False)
        x = x + a
        return x + self.mlp(self.ln2(x))


class CharGPT(nn.Module):
    def __init__(self, vocab_size: int, block_size: int = 64, n_embd: int = 64,
                 n_head: int = 4, n_layer: int = 3, dropout: float = 0.1) -> None:
        super().__init__()
        self.block_size = block_size
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([_Block(n_embd, n_head, dropout) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        b, t = idx.shape
        pos = torch.arange(t, device=idx.device)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos)[None])
        mask = torch.triu(torch.full((t, t), float("-inf"), device=idx.device), diagonal=1)
        for block in self.blocks:
            x = block(x, mask)
        return self.head(self.ln_f(x))

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0,
                 deterministic: bool = False) -> torch.Tensor:
        for _ in range(max_new_tokens):
            logits = self(idx[:, -self.block_size:])[:, -1, :] / max(temperature, 1e-6)
            probs = F.softmax(logits, dim=-1)
            nxt = probs.argmax(dim=-1, keepdim=True) if deterministic else torch.multinomial(probs, 1)
            idx = torch.cat([idx, nxt], dim=1)
        return idx


def sft_train(model: CharGPT, tokenizer: CharTokenizer, text: str, n_iters: int = 2000,
              batch_size: int = 32, learning_rate: float = 3e-3, device: str = "cpu",
              seed: Optional[int] = None) -> dict:
    """Supervised next-character pre-training on ``text``."""
    from collections import deque

    rng = np.random.default_rng(seed)
    data = np.array(tokenizer.encode(text), dtype=np.int64)
    block = model.block_size
    opt = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model.train()
    losses: deque = deque(maxlen=100)
    for _ in range(n_iters):
        ix = rng.integers(0, len(data) - block - 1, size=batch_size)
        x = torch.as_tensor(np.stack([data[i: i + block] for i in ix]), device=device)
        y = torch.as_tensor(np.stack([data[i + 1: i + 1 + block] for i in ix]), device=device)
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(float(loss.item()))
    return {"loss": float(np.mean(losses))}
