"""RLHF fine-tuning of a language model (PPO/GRPO-style with a KL penalty).

The same loop industry uses to align LLMs, on a tiny char-level GPT: sample
completions, score them with a reward, and update the policy toward higher-reward
generations while a **KL penalty keeps it close to the reference (SFT) model** so
it does not drift into gibberish. Uses group-normalized advantages (GRPO) — no
value network — and Schulman's ``k3`` KL estimator.
"""

from __future__ import annotations

import copy
from typing import Callable, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from .gpt import CharGPT, CharTokenizer

__all__ = ["rlhf_finetune", "char_frequency_reward", "lexicon_reward"]


def _sequence_logprobs(model: CharGPT, seq: torch.Tensor, prompt_len: int) -> torch.Tensor:
    """Per-token log-probs of the *generated* tokens (positions >= prompt_len)."""
    logits = model(seq[:, :-1])
    logp = F.log_softmax(logits, dim=-1)
    targets = seq[:, 1:]
    token_logp = logp.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    return token_logp[:, prompt_len - 1:]


def char_frequency_reward(target: str) -> Callable[[str], float]:
    """Reward = fraction of characters in ``target`` (a simple steering signal)."""
    def reward(text: str) -> float:
        if not text:
            return 0.0
        return sum(text.count(c) for c in target) / len(text)
    return reward


def lexicon_reward(positive: Sequence[str], negative: Sequence[str] = ()) -> Callable[[str], float]:
    """Reward = (#positive words) - (#negative words), e.g. for sentiment steering."""
    pos, neg = set(positive), set(negative)
    def reward(text: str) -> float:
        words = text.split()
        return sum(w in pos for w in words) - sum(w in neg for w in words)
    return reward


def rlhf_finetune(
    model: CharGPT,
    tokenizer: CharTokenizer,
    reward_fn: Callable[[str], float],
    prompt: str = " ",
    gen_len: int = 32,
    iters: int = 100,
    group_size: int = 16,
    learning_rate: float = 1e-3,
    kl_coef: float = 0.1,
    temperature: float = 1.0,
    reference: Optional[CharGPT] = None,
    device: str = "cpu",
    seed: Optional[int] = None,
) -> dict:
    """RLHF-finetune ``model`` to maximize ``reward_fn`` with a KL penalty.

    Returns ``{"history": [...], "before": r0, "after": rN}`` of mean rewards.
    """
    torch.manual_seed(seed if seed is not None else 0)
    ref = reference if reference is not None else copy.deepcopy(model)
    ref.eval()
    for p in ref.parameters():
        p.requires_grad_(False)

    prompt_ids = tokenizer.encode(prompt) or [0]
    prompt_len = len(prompt_ids)
    prompt_t = torch.tensor(prompt_ids, device=device, dtype=torch.long)
    opt = torch.optim.Adam(model.parameters(), lr=learning_rate)

    history = []
    for _ in range(iters):
        model.eval()
        idx0 = prompt_t.unsqueeze(0).repeat(group_size, 1)
        with torch.no_grad():
            seq = model.generate(idx0, gen_len, temperature=temperature)
        texts = [tokenizer.decode(seq[i, prompt_len:]) for i in range(group_size)]
        rewards = np.array([float(reward_fn(t)) for t in texts], dtype=np.float64)
        history.append(float(rewards.mean()))

        adv = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        adv_t = torch.as_tensor(adv, dtype=torch.float32, device=device).unsqueeze(1)

        model.train()
        logp = _sequence_logprobs(model, seq, prompt_len)
        with torch.no_grad():
            ref_logp = _sequence_logprobs(ref, seq, prompt_len)
        pg_loss = -(adv_t * logp).sum(dim=1).mean()
        log_ratio = ref_logp - logp
        kl = (torch.exp(log_ratio) - log_ratio - 1).mean()  # k3 estimator, >= 0
        loss = pg_loss + kl_coef * kl

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

    return {"history": history, "before": history[0], "after": history[-1]}
