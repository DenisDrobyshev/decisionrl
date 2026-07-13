"""Neuroevolution: train a policy network with a gradient-free optimizer.

The policy is a small tanh-MLP whose flattened weights are optimized directly to
maximize episode return by any :class:`~decisionrl.evolution.base.BlackBoxOptimizer`
(CEM, CMA-ES, GA, PSO, ...). No gradients, no replay buffer — just rollouts as a
fitness function. Implements the standard ``predict / learn / save / load`` API.
"""

from __future__ import annotations

from collections import deque
from typing import Optional, Sequence

import numpy as np

from ..core.agent import BaseAgent
from ..core.env import Env
from ..core.spaces import is_discrete
from .evolutionary import (
    ARS,
    CEM,
    CMAES,
    DifferentialEvolution,
    GeneticAlgorithm,
    OpenAIES,
    SimulatedAnnealing,
)
from .swarm import PSO, ArtificialBeeColony, BatAlgorithm, FireflyAlgorithm, GreyWolfOptimizer

__all__ = ["NeuroevolutionAgent", "OPTIMIZERS"]

OPTIMIZERS = {
    "cem": CEM,
    "cmaes": CMAES,
    "de": DifferentialEvolution,
    "ga": GeneticAlgorithm,
    "es": OpenAIES,
    "ars": ARS,
    "sa": SimulatedAnnealing,
    "pso": PSO,
    "firefly": FireflyAlgorithm,
    "abc": ArtificialBeeColony,
    "gwo": GreyWolfOptimizer,
    "bat": BatAlgorithm,
}


class NeuroevolutionAgent(BaseAgent):
    def __init__(
        self,
        env: Env,
        optimizer: str = "cem",
        popsize: Optional[int] = None,
        hidden_sizes: Sequence[int] = (32,),
        episodes_per_eval: int = 1,
        max_ep_steps: int = 1000,
        seed: Optional[int] = None,
        logger=None,
        **optimizer_kwargs,
    ) -> None:
        super().__init__(env, seed=seed, logger=logger)
        if optimizer not in OPTIMIZERS:
            raise KeyError(f"unknown optimizer {optimizer!r}; available: {sorted(OPTIMIZERS)}")
        self.optimizer_name = optimizer
        self.hidden_sizes = tuple(hidden_sizes)
        self.episodes_per_eval = int(episodes_per_eval)
        self.max_ep_steps = int(max_ep_steps)

        self.discrete = is_discrete(self.action_space)
        self.obs_dim = int(np.prod(self.observation_space.shape))
        if self.discrete:
            self.out_dim = int(self.action_space.n)
        else:
            self.out_dim = int(self.action_space.shape[0])
            self.action_low = np.asarray(self.action_space.low, dtype=np.float64)
            self.action_high = np.asarray(self.action_space.high, dtype=np.float64)

        self._shapes = self._layer_shapes()
        self.param_dim = int(sum(w * h + h for w, h in self._shapes))
        kwargs = dict(optimizer_kwargs)
        if popsize is not None:
            kwargs["popsize"] = popsize
        self.optimizer = OPTIMIZERS[optimizer](self.param_dim, seed=seed, **kwargs)
        self.params = np.zeros(self.param_dim)  # current best-known policy weights

    def _layer_shapes(self):
        sizes = [self.obs_dim, *self.hidden_sizes, self.out_dim]
        return [(sizes[i], sizes[i + 1]) for i in range(len(sizes) - 1)]

    def _forward(self, params: np.ndarray, obs: np.ndarray) -> np.ndarray:
        x = np.asarray(obs, dtype=np.float64).reshape(-1)
        idx = 0
        for li, (win, wout) in enumerate(self._shapes):
            w = params[idx: idx + win * wout].reshape(win, wout)
            idx += win * wout
            b = params[idx: idx + wout]
            idx += wout
            x = x @ w + b
            if li < len(self._shapes) - 1:
                x = np.tanh(x)
        return x

    def _action(self, params: np.ndarray, obs: np.ndarray):
        out = self._forward(params, obs)
        if self.discrete:
            return int(np.argmax(out))
        scale = (self.action_high - self.action_low) / 2.0
        bias = (self.action_high + self.action_low) / 2.0
        return np.clip(np.tanh(out) * scale + bias, self.action_low, self.action_high)

    def predict(self, obs, deterministic: bool = True):
        return self._action(self.params, obs)

    def _fitness(self, params: np.ndarray) -> float:
        """Return the negative mean episode return (for minimization)."""
        total = 0.0
        for _ in range(self.episodes_per_eval):
            obs, _ = self.env.reset()
            done, steps = False, 0
            while not done and steps < self.max_ep_steps:
                obs, reward, terminated, truncated, _ = self.env.step(self._action(params, obs))
                total += reward
                self.num_timesteps += 1
                steps += 1
                done = terminated or truncated
        return -total / self.episodes_per_eval

    def learn(self, total_steps: int, callback=None, log_interval: int = 10) -> "NeuroevolutionAgent":
        if callback is not None:
            callback.on_training_start(self)
        returns_window: deque = deque(maxlen=20)
        self.history_: list = getattr(self, "history_", [])  # (timesteps, best_return) per gen
        generation = 0
        while self.num_timesteps < total_steps:
            population = self.optimizer.ask()
            fitnesses = np.array([self._fitness(p) for p in population], dtype=np.float64)
            self.optimizer.tell(population, fitnesses)
            if self.optimizer.best_x is not None:
                self.params = self.optimizer.best_x.copy()
            returns_window.append(-float(fitnesses.min()))
            self.history_.append((self.num_timesteps, -float(self.optimizer.best_f)))
            generation += 1
            if callback is not None and not callback.on_step():
                break
            if generation % log_interval == 0:
                self.logger.record("evo/best_return", -self.optimizer.best_f)
                self.logger.record("evo/gen_return_mean", float(np.mean(returns_window)))
                self.logger.dump(self.num_timesteps)
        if callback is not None:
            callback.on_training_end()
        return self

    def save(self, path: str) -> None:
        import numpy as _np

        _np.savez(
            path,
            params=self.params,
            config=_np.array(
                {"optimizer": self.optimizer_name, "hidden_sizes": self.hidden_sizes,
                 "episodes_per_eval": self.episodes_per_eval, "max_ep_steps": self.max_ep_steps},
                dtype=object,
            ),
        )

    @classmethod
    def load(cls, path: str, env: Env = None, **kwargs) -> "NeuroevolutionAgent":
        data = np.load(path if path.endswith(".npz") else path + ".npz", allow_pickle=True)
        cfg = data["config"].item()
        agent = cls(env, optimizer=cfg["optimizer"], hidden_sizes=cfg["hidden_sizes"],
                    episodes_per_eval=cfg["episodes_per_eval"], max_ep_steps=cfg["max_ep_steps"], **kwargs)
        agent.params = data["params"]
        return agent
