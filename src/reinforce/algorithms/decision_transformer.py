"""Decision Transformer (Chen et al., 2021): offline RL as sequence modeling.

Instead of learning a value function or policy gradient, Decision Transformer
casts control as *conditional sequence modeling*: a causal GPT is trained on
offline trajectories tokenized as ``(return-to-go, state, action, ...)`` and
learns to predict the action that, in the data, preceded a given desired return.
At test time you **condition on a target return** and the model autoregressively
produces actions to achieve it.

Training is purely supervised on a :class:`~reinforce.data.TrajectoryDataset`
(no bootstrapping, no environment interaction). Discrete and continuous actions
are both supported. Because the standard evaluation loop must decrement the
return-to-go by the reward actually received, use :meth:`evaluate` for faithful
return-conditioned rollouts; :meth:`predict` also satisfies the generic agent
interface (holding the return target fixed).
"""

from __future__ import annotations

from collections import deque
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..core.agent import BaseAgent
from ..core.env import Env
from ..core.spaces import is_discrete
from ..data import TrajectoryDataset
from ..utils.torch_utils import get_device, to_tensor

__all__ = ["DecisionTransformer"]


class _Block(nn.Module):
    """A pre-norm Transformer block with causal self-attention."""

    def __init__(self, dim: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, n_heads, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim), nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        h = self.ln1(x)
        a, _ = self.attn(h, h, h, attn_mask=attn_mask, need_weights=False)
        x = x + a
        x = x + self.mlp(self.ln2(x))
        return x


class _GPT(nn.Module):
    """Minimal GPT over interleaved (return, state, action) tokens."""

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        discrete: bool,
        n_actions: int,
        embed_dim: int,
        n_layers: int,
        n_heads: int,
        dropout: float,
        max_ep_len: int,
    ) -> None:
        super().__init__()
        self.discrete = discrete
        self.embed_dim = embed_dim
        self.embed_timestep = nn.Embedding(max_ep_len, embed_dim)
        self.embed_return = nn.Linear(1, embed_dim)
        self.embed_state = nn.Linear(obs_dim, embed_dim)
        if discrete:
            self.embed_action = nn.Embedding(n_actions, embed_dim)
            self.action_head = nn.Linear(embed_dim, n_actions)
        else:
            self.embed_action = nn.Linear(act_dim, embed_dim)
            self.action_head = nn.Linear(embed_dim, act_dim)
        self.ln = nn.LayerNorm(embed_dim)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([_Block(embed_dim, n_heads, dropout) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(embed_dim)

    def forward(self, states, actions, returns_to_go, timesteps):
        b, k = states.shape[0], states.shape[1]
        time_emb = self.embed_timestep(timesteps)
        state_emb = self.embed_state(states) + time_emb
        return_emb = self.embed_return(returns_to_go) + time_emb
        if self.discrete:
            action_emb = self.embed_action(actions) + time_emb
        else:
            action_emb = self.embed_action(actions) + time_emb

        # Interleave to (R_0, s_0, a_0, R_1, s_1, a_1, ...) => length 3K.
        tokens = torch.stack([return_emb, state_emb, action_emb], dim=2).reshape(b, 3 * k, self.embed_dim)
        tokens = self.drop(self.ln(tokens))

        seq_len = 3 * k
        causal = torch.triu(
            torch.full((seq_len, seq_len), float("-inf"), device=tokens.device), diagonal=1
        )
        x = tokens
        for block in self.blocks:
            x = block(x, causal)
        x = self.ln_f(x)

        # Predict the action at each state-token position (index 1 of each triple).
        x = x.reshape(b, k, 3, self.embed_dim)
        return self.action_head(x[:, :, 1, :])


class DecisionTransformer(BaseAgent):
    def __init__(
        self,
        env: Env,
        context_len: int = 20,
        embed_dim: int = 128,
        n_layers: int = 3,
        n_heads: int = 1,
        dropout: float = 0.1,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        max_ep_len: int = 1000,
        target_return: float = 0.0,
        device: str = "auto",
        seed: Optional[int] = None,
        **kwargs,
    ) -> None:
        super().__init__(env, seed=seed, **kwargs)
        self.device = get_device(device)
        self.context_len = int(context_len)
        self.max_ep_len = int(max_ep_len)
        self.target_return = float(target_return)
        self.discrete = is_discrete(self.action_space)
        self.obs_dim = int(np.prod(self.observation_space.shape))
        if self.discrete:
            self.n_actions = int(self.action_space.n)
            self.act_dim = 1
        else:
            self.n_actions = 0
            self.act_dim = int(self.action_space.shape[0])
            self.action_low = np.asarray(self.action_space.low, dtype=np.float32)
            self.action_high = np.asarray(self.action_space.high, dtype=np.float32)

        self.model = _GPT(
            self.obs_dim, self.act_dim, self.discrete, self.n_actions,
            embed_dim, n_layers, n_heads, dropout, self.max_ep_len,
        ).to(self.device)
        self._cfg = dict(
            context_len=self.context_len, embed_dim=embed_dim, n_layers=n_layers,
            n_heads=n_heads, dropout=dropout, max_ep_len=self.max_ep_len,
            target_return=self.target_return,
        )
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        self._reset_history()

    # -- offline training --------------------------------------------------
    def learn_offline(
        self,
        dataset: TrajectoryDataset,
        n_iters: int = 2000,
        batch_size: int = 64,
        log_interval: int = 0,
    ) -> "DecisionTransformer":
        """Train by supervised sequence modeling on a :class:`TrajectoryDataset`."""
        self.model.train()
        losses: deque = deque(maxlen=100)
        for it in range(n_iters):
            states, actions, rtg, timesteps, mask = dataset.sample(
                batch_size, self.context_len, self.max_ep_len
            )
            preds = self.model(states, actions, rtg, timesteps)
            m = mask.reshape(-1) > 0
            if self.discrete:
                logits = preds.reshape(-1, self.n_actions)[m]
                target = actions.reshape(-1)[m]
                loss = F.cross_entropy(logits, target)
            else:
                pred = preds.reshape(-1, self.act_dim)[m]
                target = actions.reshape(-1, self.act_dim)[m]
                loss = F.mse_loss(pred, target)

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.25)
            self.optimizer.step()
            losses.append(float(loss.item()))
            self.num_timesteps += 1
            if log_interval and (it + 1) % log_interval == 0:
                self.logger.record("train/loss", float(np.mean(losses)))
                self.logger.dump(it + 1)
        return self

    def learn(self, *args, **kwargs):  # pragma: no cover - guard
        raise NotImplementedError(
            "DecisionTransformer is offline; use learn_offline(dataset, n_iters)."
        )

    # -- return-conditioned inference --------------------------------------
    def _reset_history(self) -> None:
        self._states: list = []
        self._actions: list = []
        self._rtgs: list = []
        self._t = 0
        self._return_so_far = 0.0
        self._eval_target: Optional[float] = None

    def reset_states(self) -> None:
        self._reset_history()

    def _placeholder_action(self):
        return 0 if self.discrete else np.zeros(self.act_dim, dtype=np.float32)

    @torch.no_grad()
    def predict(self, obs, deterministic: bool = True):
        self.model.eval()
        target = self.target_return if self._eval_target is None else self._eval_target
        rtg = target - self._return_so_far
        self._states.append(np.asarray(obs, dtype=np.float32).reshape(-1))
        self._rtgs.append(float(rtg))
        self._actions.append(self._placeholder_action())

        K = self.context_len
        states = np.asarray(self._states[-K:], dtype=np.float32)
        rtgs = np.asarray(self._rtgs[-K:], dtype=np.float32).reshape(-1, 1)
        L = states.shape[0]
        timesteps = np.clip(
            np.arange(max(0, self._t - L + 1), self._t + 1), 0, self.max_ep_len - 1
        )
        if self.discrete:
            actions = np.asarray(self._actions[-K:], dtype=np.int64).reshape(1, L)
            actions_t = to_tensor(actions, self.device, dtype=torch.long)
        else:
            actions = np.asarray(self._actions[-K:], dtype=np.float32).reshape(1, L, self.act_dim)
            actions_t = to_tensor(actions, self.device)

        preds = self.model(
            to_tensor(states.reshape(1, L, self.obs_dim), self.device),
            actions_t,
            to_tensor(rtgs.reshape(1, L, 1), self.device),
            to_tensor(timesteps.reshape(1, L), self.device, dtype=torch.long),
        )
        last = preds[0, -1]
        if self.discrete:
            action = int(last.argmax().item()) if deterministic else int(
                torch.distributions.Categorical(logits=last).sample().item()
            )
        else:
            action = np.clip(last.cpu().numpy(), self.action_low, self.action_high)

        self._actions[-1] = action
        self._t += 1
        return action

    def _observe_reward(self, reward: float) -> None:
        self._return_so_far += float(reward)

    @torch.no_grad()
    def evaluate(
        self,
        env: Env,
        target_return: Optional[float] = None,
        n_episodes: int = 10,
        seed: Optional[int] = None,
    ) -> tuple:
        """Return-conditioned rollout: decrements the target by each reward.

        Returns ``(mean_return, std_return)`` measured on the true environment
        reward. This is the faithful Decision Transformer evaluation.
        """
        target = self.target_return if target_return is None else float(target_return)
        returns = []
        for ep in range(n_episodes):
            ep_seed = None if seed is None else seed + ep
            obs, _ = env.reset(seed=ep_seed)
            self._reset_history()
            self._eval_target = target
            done = False
            total = 0.0
            while not done:
                action = self.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, _ = env.step(action)
                self._observe_reward(reward)
                total += reward
                done = terminated or truncated
            returns.append(total)
        return float(np.mean(returns)), float(np.std(returns))

    # -- persistence -------------------------------------------------------
    def save(self, path: str) -> None:
        torch.save({"model": self.model.state_dict(), "config": self._cfg}, path)

    @classmethod
    def load(cls, path: str, env: Env = None, device: str = "auto", **kwargs) -> "DecisionTransformer":
        checkpoint = torch.load(path, map_location=get_device(device), weights_only=False)
        agent = cls(env, device=device, **{**checkpoint["config"], **kwargs})
        agent.model.load_state_dict(checkpoint["model"])
        return agent
