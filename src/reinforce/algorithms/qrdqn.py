"""QR-DQN: Quantile Regression DQN (Dabney et al., 2018).

Learns ``n_quantiles`` of the return distribution per action and trains them with
the quantile-Huber (pinball) loss. The greedy action maximizes the mean of the
predicted quantiles. Reuses DQN's training loop via the ``_build_networks`` hook.
"""

from __future__ import annotations

import numpy as np
import torch

from ..networks.q_networks import QuantileQNetwork
from ..utils.torch_utils import to_tensor
from .dqn import DQN

__all__ = ["QRDQN"]


class QRDQN(DQN):
    def __init__(self, env, n_quantiles: int = 200, kappa: float = 1.0, **kwargs) -> None:
        self.n_quantiles = int(n_quantiles)
        self.kappa = float(kappa)
        kwargs.setdefault("dueling", False)
        super().__init__(env, **kwargs)

    def _build_networks(self, learning_rate: float) -> None:
        # midpoint quantile fractions tau_i = (i + 0.5) / N
        # (note: distinct from DQN's polyak coefficient self.tau)
        self.quantile_taus = (torch.arange(self.n_quantiles, device=self.device) + 0.5) / self.n_quantiles
        self.q_net = QuantileQNetwork(self.obs_dim, self.n_actions, self.n_quantiles, self.hidden_sizes).to(self.device)
        self.target_net = QuantileQNetwork(self.obs_dim, self.n_actions, self.n_quantiles, self.hidden_sizes).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=learning_rate)

    @torch.no_grad()
    def predict(self, obs, deterministic: bool = True) -> int:
        if not deterministic and self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.n_actions))
        obs_t = to_tensor(np.asarray(obs).reshape(1, *self.obs_shape), self.device)
        q = self.q_net(obs_t).mean(dim=2)  # (1, A)
        return int(q.argmax(dim=1).item())

    def _quantile_huber_loss(self, current: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # current: (B, N), target: (B, N)
        delta = target.unsqueeze(1) - current.unsqueeze(2)  # (B, N_current i, N_target j)
        abs_delta = delta.abs()
        huber = torch.where(abs_delta <= self.kappa, 0.5 * delta.pow(2), self.kappa * (abs_delta - 0.5 * self.kappa))
        weight = (self.quantile_taus.view(1, -1, 1) - (delta.detach() < 0).float()).abs()
        # mean over target quantiles, sum over current quantiles, mean over batch
        return (weight * huber / self.kappa).mean(dim=2).sum(dim=1).mean()

    def _train_step(self, beta: float) -> float:
        batch = self.buffer.sample(self.batch_size)
        bsz = self.batch_size

        with torch.no_grad():
            next_quant = self.target_net(batch.next_obs)  # (B, A, N)
            if self.double_q:
                next_actions = self.q_net(batch.next_obs).mean(dim=2).argmax(dim=1)
            else:
                next_actions = next_quant.mean(dim=2).argmax(dim=1)
            next_quant_sel = next_quant[torch.arange(bsz), next_actions]  # (B, N)
            target = (
                batch.rewards.unsqueeze(1)
                + batch.discounts.unsqueeze(1) * (1.0 - batch.dones.unsqueeze(1)) * next_quant_sel
            )  # (B, N)

        current = self.q_net(batch.obs)[torch.arange(bsz), batch.actions]  # (B, N)
        loss = self._quantile_huber_loss(current, target)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), self.max_grad_norm)
        self.optimizer.step()
        return float(loss.item())

    def _config(self) -> dict:
        cfg = super()._config()
        cfg.update(n_quantiles=self.n_quantiles, kappa=self.kappa)
        return cfg
