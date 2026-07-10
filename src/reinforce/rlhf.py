"""Reinforcement learning from (preference) feedback — RLHF, classic-control style.

The same recipe that aligns large language models, on small control tasks:

1. **Collect segments** of behaviour (:func:`collect_segments`).
2. **Get preferences** over pairs of segments — here from a synthetic teacher that
   prefers the higher-true-return segment (:func:`synthetic_preferences`); swap in
   real human labels by building a :class:`PreferenceDataset` directly.
3. **Train a reward model** on the Bradley-Terry preference likelihood
   (:func:`train_reward_model`): ``P(A ≻ B) = σ(R_A − R_B)`` where ``R`` is the sum
   of the model's per-step rewards over a segment.
4. **Optimize a policy** against the learned reward by wrapping the environment in
   :class:`RewardModelWrapper` and training *any* agent on it — the true reward is
   never shown to the policy, only the learned one.

Pairs naturally with :class:`~reinforce.algorithms.GRPO`, the critic-free,
group-relative policy-optimization method used for LLM RLHF.
"""

from __future__ import annotations

import copy
from collections import deque
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .core.agent import BaseAgent
from .core.env import Env, Wrapper
from .core.spaces import is_discrete
from .networks.mlp import build_mlp
from .networks.policies import CategoricalActor, GaussianActor
from .utils.torch_utils import get_device, to_tensor

__all__ = [
    "RewardModel",
    "Segment",
    "PreferenceDataset",
    "collect_segments",
    "synthetic_preferences",
    "train_reward_model",
    "RewardModelWrapper",
    "DPO",
]


class RewardModel(nn.Module):
    """Learned scalar reward ``r(s)`` or ``r(s, a)`` trained from preferences."""

    def __init__(
        self,
        obs_dim: int,
        action_space=None,
        use_action: bool = True,
        hidden_sizes=(64, 64),
        learning_rate: float = 1e-3,
        device: str = "auto",
    ) -> None:
        super().__init__()
        self.device = get_device(device)
        self.use_action = bool(use_action) and action_space is not None
        self.discrete = is_discrete(action_space) if action_space is not None else False
        if self.use_action:
            self.action_dim = int(action_space.n) if self.discrete else int(action_space.shape[0])
        else:
            self.action_dim = 0
        self.net = build_mlp(obs_dim + self.action_dim, 1, hidden_sizes).to(self.device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=learning_rate)

    def _encode_actions(self, actions: torch.Tensor) -> torch.Tensor:
        if self.discrete:
            return F.one_hot(actions.long().reshape(-1), self.action_dim).float()
        return actions.reshape(actions.shape[0], -1).float()

    def forward(self, obs: torch.Tensor, actions: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Per-row reward for a batch of observations ``(N, obs_dim)`` -> ``(N,)``."""
        x = obs.reshape(obs.shape[0], -1)
        if self.use_action:
            x = torch.cat([x, self._encode_actions(actions)], dim=-1)
        return self.net(x).squeeze(-1)

    @torch.no_grad()
    def reward(self, obs, action=None) -> float:
        obs_t = to_tensor(np.asarray(obs, dtype=np.float32).reshape(1, -1), self.device)
        act_t = None
        if self.use_action:
            act_t = to_tensor(np.asarray(action).reshape(1, -1), self.device)
        return float(self.forward(obs_t, act_t).item())

    @torch.no_grad()
    def predict_rewards(self, obs: np.ndarray, actions: Optional[np.ndarray] = None) -> np.ndarray:
        obs_t = to_tensor(np.asarray(obs, dtype=np.float32), self.device)
        act_t = to_tensor(np.asarray(actions), self.device) if self.use_action else None
        return self.forward(obs_t, act_t).cpu().numpy()


@dataclass
class Segment:
    """A fixed-length behaviour snippet with its true (environment) return."""

    obs: np.ndarray  # (T, obs_dim)
    actions: np.ndarray  # (T, act_dim) or (T,)
    true_return: float


class PreferenceDataset:
    """Pairs of segments with a label: ``1`` if the first is preferred, else ``0``."""

    def __init__(self, pairs: List[Tuple[Segment, Segment, float]], seed: Optional[int] = None) -> None:
        self.pairs = list(pairs)
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.pairs)

    def _stack(self, segments: List[Segment], device):
        obs = np.stack([s.obs for s in segments]).astype(np.float32)  # (B, T, obs_dim)
        acts = np.stack([s.actions for s in segments])  # (B, T[, act_dim])
        return to_tensor(obs, device), to_tensor(acts, device)

    def sample(self, batch_size: int, device):
        idx = self.rng.integers(0, len(self.pairs), size=min(batch_size, len(self.pairs)))
        seg_a = [self.pairs[i][0] for i in idx]
        seg_b = [self.pairs[i][1] for i in idx]
        labels = np.array([self.pairs[i][2] for i in idx], dtype=np.float32)
        obs_a, act_a = self._stack(seg_a, device)
        obs_b, act_b = self._stack(seg_b, device)
        return obs_a, act_a, obs_b, act_b, to_tensor(labels, device)


def collect_segments(
    env: Env,
    policy: Callable[[np.ndarray], object],
    n_segments: int,
    seg_len: int = 25,
    seed: Optional[int] = None,
) -> List[Segment]:
    """Roll out ``policy`` and slice the stream into fixed-length labelled segments."""
    segments: List[Segment] = []
    obs, _ = env.reset(seed=seed)
    buf_o, buf_a, buf_r = [], [], []
    while len(segments) < n_segments:
        action = policy(obs)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        buf_o.append(np.asarray(obs, dtype=np.float32))
        buf_a.append(np.asarray(action, dtype=np.float32))
        buf_r.append(float(reward))
        obs = next_obs
        if len(buf_o) == seg_len:
            segments.append(
                Segment(np.asarray(buf_o), np.asarray(buf_a), float(np.sum(buf_r)))
            )
            buf_o, buf_a, buf_r = [], [], []
        if terminated or truncated:
            obs, _ = env.reset()
            buf_o, buf_a, buf_r = [], [], []  # segments do not straddle episodes
    return segments


def synthetic_preferences(
    segments: List[Segment],
    n_pairs: int,
    rational: bool = True,
    temperature: float = 1.0,
    seed: Optional[int] = None,
) -> PreferenceDataset:
    """Build preferences from a synthetic teacher that prefers higher true return.

    ``rational=True`` labels deterministically by which segment has the higher
    return; otherwise labels are sampled from the Bradley-Terry model
    ``P(A ≻ B) = σ((G_A − G_B) / temperature)`` (a noisier, more human-like teacher).
    """
    rng = np.random.default_rng(seed)
    pairs: List[Tuple[Segment, Segment, float]] = []
    n = len(segments)
    for _ in range(n_pairs):
        i, j = rng.integers(0, n, size=2)
        a, b = segments[i], segments[j]
        if rational:
            label = 1.0 if a.true_return >= b.true_return else 0.0
        else:
            p = 1.0 / (1.0 + np.exp(-(a.true_return - b.true_return) / temperature))
            label = float(rng.random() < p)
        pairs.append((a, b, label))
    return PreferenceDataset(pairs, seed=seed)


def train_reward_model(
    model: RewardModel,
    dataset: PreferenceDataset,
    n_iters: int = 500,
    batch_size: int = 32,
    log_interval: int = 0,
    logger=None,
) -> dict:
    """Fit ``model`` to the Bradley-Terry preference likelihood; returns metrics."""
    losses: deque = deque(maxlen=50)
    accs: deque = deque(maxlen=50)
    for it in range(n_iters):
        obs_a, act_a, obs_b, act_b, labels = dataset.sample(batch_size, model.device)
        # Sum per-step model rewards into a segment "return".
        b, t = obs_a.shape[0], obs_a.shape[1]
        ret_a = model(obs_a.reshape(b * t, -1), act_a.reshape(b * t, -1) if model.use_action else None)
        ret_b = model(obs_b.reshape(b * t, -1), act_b.reshape(b * t, -1) if model.use_action else None)
        ret_a = ret_a.reshape(b, t).sum(dim=1)
        ret_b = ret_b.reshape(b, t).sum(dim=1)

        logits = ret_a - ret_b  # P(A preferred) = sigmoid(logits)
        loss = F.binary_cross_entropy_with_logits(logits, labels)
        model.optimizer.zero_grad()
        loss.backward()
        model.optimizer.step()

        with torch.no_grad():
            acc = (((logits > 0).float() == labels).float().mean().item())
        losses.append(float(loss.item()))
        accs.append(acc)
        if logger is not None and log_interval and (it + 1) % log_interval == 0:
            logger.record("reward_model/loss", float(np.mean(losses)))
            logger.record("reward_model/accuracy", float(np.mean(accs)))
            logger.dump(it + 1)
    return {"loss": float(np.mean(losses)), "accuracy": float(np.mean(accs))}


class RewardModelWrapper(Wrapper):
    """Replace the environment reward with a learned :class:`RewardModel` reward.

    The policy sees only the learned reward; the true environment reward is kept in
    ``info["true_reward"]`` so evaluation can still measure real performance.
    """

    def __init__(self, env: Env, reward_model: RewardModel) -> None:
        super().__init__(env)
        self.reward_model = reward_model
        self._last_obs: Optional[np.ndarray] = None

    def reset(self, *, seed: Optional[int] = None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        self._last_obs = np.asarray(obs, dtype=np.float32)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        learned = self.reward_model.reward(self._last_obs, action)
        info = dict(info)
        info["true_reward"] = float(reward)
        self._last_obs = np.asarray(obs, dtype=np.float32)
        return obs, learned, terminated, truncated, info


class DPO(BaseAgent):
    """Direct Preference Optimization (Rafailov et al., 2023) for control.

    Optimizes a policy *directly* from preference pairs against a frozen reference
    policy — no separate reward model, no RL loop. For two segments with a label
    (winner ``w``, loser ``l``) the loss is::

        L = -log sigma( beta * [ (logpi(w) - logref(w)) - (logpi(l) - logref(l)) ] )

    where ``logpi(seg)`` is the summed log-probability of the segment's actions
    under the policy. Discrete and continuous action spaces are supported. This is
    the LLM-alignment method that skips the reward model that :class:`RewardModel`
    + :class:`~reinforce.algorithms.GRPO` would otherwise learn.
    """

    def __init__(
        self,
        env: Env,
        beta: float = 0.1,
        learning_rate: float = 3e-4,
        hidden_sizes=(64, 64),
        activation: str = "tanh",
        device: str = "auto",
        seed: Optional[int] = None,
        **kwargs,
    ) -> None:
        super().__init__(env, seed=seed, **kwargs)
        self.device = get_device(device)
        self.beta = float(beta)
        self.hidden_sizes = tuple(hidden_sizes)
        self.discrete = is_discrete(self.action_space)
        self.obs_dim = int(np.prod(self.observation_space.shape))
        act_fn = {"tanh": nn.Tanh, "relu": nn.ReLU}[activation]
        if self.discrete:
            self.actor = CategoricalActor(self.obs_dim, int(self.action_space.n), hidden_sizes, act_fn)
        else:
            self.act_dim = int(self.action_space.shape[0])
            self.action_low = np.asarray(self.action_space.low, dtype=np.float32)
            self.action_high = np.asarray(self.action_space.high, dtype=np.float32)
            self.actor = GaussianActor(self.obs_dim, self.act_dim, hidden_sizes, act_fn)
        self.actor.to(self.device)
        self.reference = copy.deepcopy(self.actor)
        for p in self.reference.parameters():
            p.requires_grad_(False)
        self.optimizer = torch.optim.Adam(self.actor.parameters(), lr=learning_rate)

    def _segment_logprob(self, actor, obs_t, act_t) -> torch.Tensor:
        b, t = obs_t.shape[0], obs_t.shape[1]
        dist = actor(obs_t.reshape(b * t, self.obs_dim))
        if self.discrete:
            lp = dist.log_prob(act_t.reshape(b * t).long())
        else:
            lp = dist.log_prob(act_t.reshape(b * t, self.act_dim)).sum(dim=-1)
        return lp.reshape(b, t).sum(dim=1)  # summed log-prob per segment

    def train(self, dataset: "PreferenceDataset", n_iters: int = 500, batch_size: int = 32,
              log_interval: int = 0) -> dict:
        """Fit the policy to preferences via the DPO loss. Returns metrics."""
        losses: deque = deque(maxlen=50)
        accs: deque = deque(maxlen=50)
        for it in range(n_iters):
            obs_a, act_a, obs_b, act_b, labels = dataset.sample(batch_size, self.device)
            with torch.no_grad():
                ref_a = self._segment_logprob(self.reference, obs_a, act_a)
                ref_b = self._segment_logprob(self.reference, obs_b, act_b)
            pi_a = self._segment_logprob(self.actor, obs_a, act_a)
            pi_b = self._segment_logprob(self.actor, obs_b, act_b)

            sign = 2.0 * labels - 1.0  # label 1 (a preferred) -> +1, else -1
            logits = self.beta * sign * ((pi_a - ref_a) - (pi_b - ref_b))
            loss = -F.logsigmoid(logits).mean()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            losses.append(float(loss.item()))
            accs.append(float((logits > 0).float().mean().item()))
            if log_interval and (it + 1) % log_interval == 0:
                self.logger.record("dpo/loss", float(np.mean(losses)))
                self.logger.record("dpo/accuracy", float(np.mean(accs)))
                self.logger.dump(it + 1)
        return {"loss": float(np.mean(losses)), "accuracy": float(np.mean(accs))}

    @torch.no_grad()
    def predict(self, obs, deterministic: bool = True):
        obs_t = to_tensor(np.asarray(obs, dtype=np.float32).reshape(1, -1), self.device)
        dist = self.actor(obs_t)
        if self.discrete:
            action = dist.probs.argmax(dim=-1) if deterministic else dist.sample()
            return int(action.item())
        action = dist.mean if deterministic else dist.sample()
        return np.clip(action.cpu().numpy()[0], self.action_low, self.action_high)

    def learn(self, *args, **kwargs):  # pragma: no cover - guard
        raise NotImplementedError("DPO is preference-based; use train(preference_dataset).")

    def save(self, path: str) -> None:
        torch.save(
            {"actor": self.actor.state_dict(),
             "config": dict(beta=self.beta, hidden_sizes=self.hidden_sizes)},
            path,
        )

    @classmethod
    def load(cls, path: str, env: Env = None, device: str = "auto", **kwargs) -> "DPO":
        checkpoint = torch.load(path, map_location=get_device(device), weights_only=False)
        agent = cls(env, device=device, **{**checkpoint["config"], **kwargs})
        agent.actor.load_state_dict(checkpoint["actor"])
        agent.reference.load_state_dict(checkpoint["actor"])
        return agent
