"""C51: Categorical / distributional DQN (Bellemare et al., 2017).

Instead of a single Q-value, the network predicts a categorical distribution
over a fixed set of return "atoms" per action. The distributional Bellman target
is projected back onto the support, and the loss is the cross-entropy between the
projected target and the predicted distribution. Reuses DQN's training loop.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from ..networks.q_networks import CategoricalQNetwork
from ..utils.torch_utils import to_tensor
from .dqn import DQN

__all__ = ["C51"]


class C51(DQN):
    def __init__(
        self,
        env,
        n_atoms: int = 51,
        v_min: float = -10.0,
        v_max: float = 10.0,
        **kwargs,
    ) -> None:
        self.n_atoms = int(n_atoms)
        self.v_min = float(v_min)
        self.v_max = float(v_max)
        # Dueling/categorical are mutually exclusive here.
        kwargs.setdefault("dueling", False)
        super().__init__(env, **kwargs)

    def _build_networks(self, learning_rate: float) -> None:
        self.support = torch.linspace(self.v_min, self.v_max, self.n_atoms, device=self.device)
        self.delta_z = (self.v_max - self.v_min) / (self.n_atoms - 1)
        self.q_net = CategoricalQNetwork(self.obs_dim, self.n_actions, self.n_atoms, self.hidden_sizes).to(self.device)
        self.target_net = CategoricalQNetwork(self.obs_dim, self.n_actions, self.n_atoms, self.hidden_sizes).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=learning_rate)

    def _q_values(self, obs_t: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(self.q_net(obs_t), dim=2)
        return (probs * self.support).sum(dim=2)

    @torch.no_grad()
    def predict(self, obs, deterministic: bool = True) -> int:
        if not deterministic and self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.n_actions))
        obs_t = to_tensor(np.asarray(obs).reshape(1, -1), self.device)
        return int(self._q_values(obs_t).argmax(dim=1).item())

    def _train_step(self, beta: float) -> float:
        batch = self.buffer.sample(self.batch_size)
        bsz = self.batch_size

        with torch.no_grad():
            next_probs = F.softmax(self.target_net(batch.next_obs), dim=2)  # (B, A, N)
            next_q = (next_probs * self.support).sum(dim=2)  # (B, A)
            if self.double_q:
                online_q = self._q_values(batch.next_obs)
                next_actions = online_q.argmax(dim=1)
            else:
                next_actions = next_q.argmax(dim=1)
            next_dist = next_probs[torch.arange(bsz), next_actions]  # (B, N)

            tz = (
                batch.rewards.unsqueeze(1)
                + batch.discounts.unsqueeze(1) * (1.0 - batch.dones.unsqueeze(1)) * self.support.unsqueeze(0)
            ).clamp(self.v_min, self.v_max)
            b = (tz - self.v_min) / self.delta_z
            lower = b.floor().long()
            upper = b.ceil().long()
            # avoid losing probability mass when b lands exactly on an atom
            lower[(upper > 0) & (lower == upper)] -= 1
            upper[(lower < (self.n_atoms - 1)) & (lower == upper)] += 1

            m = torch.zeros(bsz, self.n_atoms, device=self.device)
            offset = (torch.arange(bsz, device=self.device) * self.n_atoms).unsqueeze(1)
            m.view(-1).index_add_(0, (lower + offset).view(-1), (next_dist * (upper.float() - b)).view(-1))
            m.view(-1).index_add_(0, (upper + offset).view(-1), (next_dist * (b - lower.float())).view(-1))

        log_p = F.log_softmax(self.q_net(batch.obs), dim=2)[torch.arange(bsz), batch.actions]  # (B, N)
        loss = -(m * log_p).sum(dim=1).mean()

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), self.max_grad_norm)
        self.optimizer.step()
        return float(loss.item())

    def _config(self) -> dict:
        cfg = super()._config()
        cfg.update(n_atoms=self.n_atoms, v_min=self.v_min, v_max=self.v_max)
        return cfg
