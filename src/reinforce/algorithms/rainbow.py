"""Rainbow DQN (Hessel et al., 2018): the combined value-based agent.

Integrates six DQN improvements: Double Q-learning, Dueling architecture,
Prioritized Experience Replay, multi-step (n-step) returns, Distributional RL
(C51) and Noisy Nets (for exploration, replacing epsilon-greedy). Reuses C51's
categorical projection and DQN's training loop.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from ..networks.noisy import RainbowNetwork
from ..utils.torch_utils import to_tensor
from .c51 import C51

__all__ = ["Rainbow"]


class Rainbow(C51):
    def __init__(
        self,
        env,
        n_atoms: int = 51,
        v_min: float = -10.0,
        v_max: float = 10.0,
        noisy_sigma: float = 0.5,
        n_step: int = 3,
        **kwargs,
    ) -> None:
        self.noisy_sigma = float(noisy_sigma)
        # Rainbow bundles double + PER + n-step; noisy nets replace epsilon-greedy.
        kwargs.setdefault("double_q", True)
        kwargs.setdefault("prioritized", True)
        kwargs.setdefault("epsilon_start", 0.0)
        kwargs.setdefault("epsilon_end", 0.0)
        super().__init__(env, n_atoms=n_atoms, v_min=v_min, v_max=v_max, n_step=n_step, **kwargs)

    def _build_networks(self, learning_rate: float) -> None:
        self.support = torch.linspace(self.v_min, self.v_max, self.n_atoms, device=self.device)
        self.delta_z = (self.v_max - self.v_min) / (self.n_atoms - 1)
        self.q_net = RainbowNetwork(self.obs_dim, self.n_actions, self.n_atoms,
                                    self.hidden_sizes, self.noisy_sigma).to(self.device)
        self.target_net = RainbowNetwork(self.obs_dim, self.n_actions, self.n_atoms,
                                         self.hidden_sizes, self.noisy_sigma).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        # target stays in train() mode so its noisy layers use resampled noise
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=learning_rate)

    @torch.no_grad()
    def predict(self, obs, deterministic: bool = True) -> int:
        obs_t = to_tensor(np.asarray(obs).reshape(1, *self.obs_shape), self.device)
        if deterministic:
            was_training = self.q_net.training
            self.q_net.eval()  # NoisyLinear uses mean weights (deterministic)
            action = int(self._q_values(obs_t).argmax(dim=1).item())
            if was_training:
                self.q_net.train()
            return action
        self.q_net.reset_noise()
        return int(self._q_values(obs_t).argmax(dim=1).item())

    def _train_step(self, beta: float) -> float:
        self.q_net.reset_noise()
        self.target_net.reset_noise()
        if self.prioritized:
            batch = self.buffer.sample(self.batch_size, beta=beta)
        else:
            batch = self.buffer.sample(self.batch_size)

        m = self._project(batch)
        bsz = batch.obs.shape[0]
        log_p = F.log_softmax(self.q_net(batch.obs), dim=2)[torch.arange(bsz), batch.actions]
        per_sample = -(m * log_p).sum(dim=1)  # cross-entropy per transition
        loss = (batch.weights * per_sample).mean() if self.prioritized else per_sample.mean()

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), self.max_grad_norm)
        self.optimizer.step()

        if self.prioritized:
            self.buffer.update_priorities(batch.indices, per_sample.detach().cpu().numpy())
        return float(loss.item())

    def _config(self) -> dict:
        cfg = super()._config()
        cfg.update(noisy_sigma=self.noisy_sigma)
        return cfg
