"""A synchronous vectorized environment.

Runs several environment copies in the same process and batches their
observations/rewards. It uses Gymnasium's *autoreset* convention: when a
sub-environment finishes, it is reset immediately and the terminal observation
is preserved under ``infos["final_observation"][i]`` so that on-policy
algorithms can bootstrap correctly on truncation.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np

from ..core.env import Env

__all__ = ["SyncVectorEnv"]


class SyncVectorEnv:
    def __init__(self, env_fns: Sequence[Callable[[], Env]]) -> None:
        assert len(env_fns) >= 1, "need at least one env"
        self.envs: List[Env] = [fn() for fn in env_fns]
        self.num_envs = len(self.envs)
        self.single_observation_space = self.envs[0].observation_space
        self.single_action_space = self.envs[0].action_space
        # Aliases so single-env-style code still finds the spaces.
        self.observation_space = self.single_observation_space
        self.action_space = self.single_action_space

    def _stack(self, obs_list: List[Any]) -> np.ndarray:
        return np.asarray(obs_list)

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None):
        obs_list: List[Any] = []
        infos: Dict[str, Any] = {}
        for i, env in enumerate(self.envs):
            env_seed = None if seed is None else seed + i
            obs, _info = env.reset(seed=env_seed, options=options)
            obs_list.append(obs)
        return self._stack(obs_list), infos

    def step(self, actions: Sequence[Any]):
        obs_list: List[Any] = []
        rewards = np.zeros(self.num_envs, dtype=np.float32)
        terminateds = np.zeros(self.num_envs, dtype=bool)
        truncateds = np.zeros(self.num_envs, dtype=bool)
        final_obs: List[Any] = [None] * self.num_envs
        final_info: List[Any] = [None] * self.num_envs
        any_final = False

        for i, (env, action) in enumerate(zip(self.envs, actions)):
            obs, reward, terminated, truncated, info = env.step(action)
            rewards[i] = reward
            terminateds[i] = terminated
            truncateds[i] = truncated
            if terminated or truncated:
                any_final = True
                final_obs[i] = obs
                final_info[i] = info
                obs, _ = env.reset()
            obs_list.append(obs)

        infos: Dict[str, Any] = {}
        if any_final:
            infos["final_observation"] = final_obs
            infos["final_info"] = final_info
        return self._stack(obs_list), rewards, terminateds, truncateds, infos

    def close(self) -> None:
        for env in self.envs:
            env.close()
