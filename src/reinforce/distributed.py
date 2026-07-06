"""Distributed actors via real processes, feeding a central V-trace learner.

This is a genuine multi-process IMPALA-style architecture (discrete actions):
several **actor processes** each hold their own environment and a local copy of
the policy, run inference locally, and stream fixed-length trajectories to the
**learner** (the main process). The learner performs V-trace updates and
broadcasts fresh weights back to the actors each iteration.

Because it uses the ``spawn`` start method, the environment factory must be
picklable (a module-level function or a class), not a lambda.
"""

from __future__ import annotations

import contextlib
import multiprocessing as mp
from typing import Callable, Optional, Sequence

import numpy as np
import torch

from .core.env import Env
from .core.spaces import is_discrete
from .networks.policies import CategoricalActor
from .networks.value import VNetwork
from .utils.logger import Logger
from .utils.torch_utils import get_device, to_tensor

__all__ = ["DistributedActorLearner"]


def _actor_worker(remote, parent_remote, env_fn, obs_dim, n_actions, hidden_sizes, n_steps) -> None:
    parent_remote.close()
    env = env_fn()
    actor = CategoricalActor(obs_dim, n_actions, hidden_sizes)
    actor.eval()
    obs, _ = env.reset()
    ep_start = 1.0
    try:
        while True:
            cmd, data = remote.recv()
            if cmd == "close":
                remote.close()
                break
            actor.load_state_dict(data)
            obs_b, act_b, logp_b, rew_b, start_b = [], [], [], [], []
            for _ in range(n_steps):
                with torch.no_grad():
                    dist = actor(torch.as_tensor(np.asarray(obs, np.float32)).unsqueeze(0))
                    action = int(dist.sample().item())
                    logp = float(dist.log_prob(torch.tensor([action])).item())
                obs_b.append(np.asarray(obs, np.float32))
                act_b.append(action)
                logp_b.append(logp)
                start_b.append(ep_start)
                obs, reward, terminated, truncated, _ = env.step(action)
                rew_b.append(float(reward))
                done = terminated or truncated
                ep_start = float(done)
                if done:
                    obs, _ = env.reset()
            remote.send(dict(
                obs=np.asarray(obs_b, np.float32), actions=np.asarray(act_b, np.int64),
                logp=np.asarray(logp_b, np.float32), rewards=np.asarray(rew_b, np.float32),
                starts=np.asarray(start_b, np.float32),
                last_obs=np.asarray(obs, np.float32), last_done=float(ep_start),
            ))
    except (KeyboardInterrupt, EOFError):  # pragma: no cover
        pass
    finally:
        env.close()


class DistributedActorLearner:
    def __init__(
        self,
        env_fn: Callable[[], Env],
        num_actors: int = 4,
        n_steps: int = 32,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        ent_coef: float = 0.01,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        rho_bar: float = 1.0,
        c_bar: float = 1.0,
        hidden_sizes: Sequence[int] = (64, 64),
        device: str = "cpu",
        seed: Optional[int] = None,
        logger: Optional[Logger] = None,
    ) -> None:
        probe = env_fn()
        assert is_discrete(probe.action_space), "DistributedActorLearner supports discrete actions"
        self.obs_dim = int(np.prod(probe.observation_space.shape))
        self.n_actions = int(probe.action_space.n)
        probe.close()

        self.device = get_device(device)
        self.num_actors = int(num_actors)
        self.n_steps = int(n_steps)
        self.gamma, self.ent_coef, self.vf_coef = float(gamma), float(ent_coef), float(vf_coef)
        self.max_grad_norm, self.rho_bar, self.c_bar = float(max_grad_norm), float(rho_bar), float(c_bar)
        self.hidden_sizes = tuple(hidden_sizes)
        self.logger = logger if logger is not None else Logger()
        self.num_timesteps = 0

        self.actor = CategoricalActor(self.obs_dim, self.n_actions, self.hidden_sizes).to(self.device)
        self.critic = VNetwork(self.obs_dim, self.hidden_sizes).to(self.device)
        self.optimizer = torch.optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()), lr=learning_rate
        )

        ctx = mp.get_context("spawn")
        self.remotes, work_remotes = zip(*[ctx.Pipe() for _ in range(self.num_actors)])
        self.processes = []
        for wr, r in zip(work_remotes, self.remotes):
            p = ctx.Process(target=_actor_worker,
                            args=(wr, r, env_fn, self.obs_dim, self.n_actions, self.hidden_sizes, self.n_steps),
                            daemon=True)
            p.start()
            self.processes.append(p)
            wr.close()
        self.closed = False

    @torch.no_grad()
    def predict(self, obs, deterministic: bool = True) -> int:
        dist = self.actor(to_tensor(np.asarray(obs, np.float32).reshape(1, -1), self.device))
        action = dist.probs.argmax(-1) if deterministic else dist.sample()
        return int(action.item())

    def reset_states(self) -> None:  # for evaluate_policy compatibility
        pass

    def _weights(self):
        return {k: v.detach().cpu() for k, v in self.actor.state_dict().items()}

    def learn(self, total_steps: int, log_interval: int = 10) -> "DistributedActorLearner":
        iteration = 0
        while self.num_timesteps < total_steps:
            weights = self._weights()
            for remote in self.remotes:
                remote.send(("collect", weights))
            trajs = [remote.recv() for remote in self.remotes]

            T, A = self.n_steps, self.num_actors
            obs = to_tensor(np.stack([t["obs"] for t in trajs], axis=1).reshape(T * A, -1), self.device)
            actions = torch.as_tensor(np.stack([t["actions"] for t in trajs], axis=1).reshape(T * A),
                                      device=self.device, dtype=torch.long)
            behavior_logp = to_tensor(np.stack([t["logp"] for t in trajs], axis=1), self.device)
            rewards = to_tensor(np.stack([t["rewards"] for t in trajs], axis=1), self.device)
            starts = to_tensor(np.stack([t["starts"] for t in trajs], axis=1), self.device)
            last_obs = to_tensor(np.stack([t["last_obs"] for t in trajs], axis=0), self.device)
            last_done = to_tensor(np.asarray([t["last_done"] for t in trajs], np.float32), self.device)

            dist = self.actor(obs)
            log_probs = dist.log_prob(actions).reshape(T, A)
            entropy = dist.entropy().reshape(T, A)
            values = self.critic(obs).reshape(T, A)
            with torch.no_grad():
                bootstrap = self.critic(last_obs)
                rho = torch.exp(log_probs - behavior_logp)
                clipped_rho = rho.clamp(max=self.rho_bar)
                c = rho.clamp(max=self.c_bar)
                v = values.detach()
                vs = torch.zeros_like(v)
                pg_adv = torch.zeros_like(v)
                carry = torch.zeros(A, device=self.device)
                for t in reversed(range(T)):
                    if t == T - 1:
                        nnt, next_v, next_vs = 1.0 - last_done, bootstrap, bootstrap
                    else:
                        nnt, next_v, next_vs = 1.0 - starts[t + 1], v[t + 1], vs[t + 1]
                    delta = clipped_rho[t] * (rewards[t] + self.gamma * nnt * next_v - v[t])
                    carry = delta + self.gamma * nnt * c[t] * carry
                    vs[t] = v[t] + carry
                    pg_adv[t] = clipped_rho[t] * (rewards[t] + self.gamma * nnt * next_vs - v[t])

            policy_loss = -(pg_adv * log_probs).mean()
            value_loss = ((vs - values) ** 2).mean()
            loss = policy_loss + self.vf_coef * value_loss - self.ent_coef * entropy.mean()
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(self.actor.parameters()) + list(self.critic.parameters()), self.max_grad_norm
            )
            self.optimizer.step()

            self.num_timesteps += T * A
            iteration += 1
            if log_interval and iteration % log_interval == 0:
                self.logger.record("rollout/reward_mean", float(rewards.mean().item()))
                self.logger.record("train/policy_loss", float(policy_loss.item()))
                self.logger.dump(self.num_timesteps)
        return self

    def close(self) -> None:
        if self.closed:
            return
        for remote in self.remotes:
            with contextlib.suppress(BrokenPipeError, OSError):
                remote.send(("close", None))
        for p in self.processes:
            p.join(timeout=5)
        for p in self.processes:
            if p.is_alive():  # pragma: no cover
                p.terminate()
        self.closed = True

    def __del__(self):  # pragma: no cover
        self.close()
