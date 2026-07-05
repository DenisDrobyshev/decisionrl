"""Policy evaluation utilities."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from ..core.env import Env

__all__ = ["evaluate_policy"]


def evaluate_policy(
    agent,
    env: Env,
    n_episodes: int = 10,
    deterministic: bool = True,
    seed: Optional[int] = None,
) -> Tuple[float, float]:
    """Run ``n_episodes`` and return ``(mean_return, std_return)``.

    Uses undiscounted episode returns, the standard reporting metric.
    """
    returns = []
    for ep in range(n_episodes):
        ep_seed = None if seed is None else seed + ep
        obs, _ = env.reset(seed=ep_seed)
        agent.reset_states()  # reset RNN hidden state (no-op for memoryless agents)
        done = False
        total = 0.0
        while not done:
            action = agent.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, _ = env.step(action)
            total += reward
            done = terminated or truncated
        returns.append(total)
    return float(np.mean(returns)), float(np.std(returns))
