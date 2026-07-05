"""A subprocess-based vectorized environment for true parallel data collection.

Each sub-environment runs in its own process, so environment stepping happens in
parallel with policy inference and gradient updates. The API matches
:class:`~reinforce.wrappers.vector.SyncVectorEnv` (including autoreset and
``infos["final_observation"]``), so on-policy agents use it unchanged.

Because it uses the ``spawn`` start method (for cross-platform consistency), the
environment factories passed in **must be picklable** - use module-level
functions or :func:`functools.partial`, not lambdas or closures.
"""

from __future__ import annotations

import contextlib
import multiprocessing as mp
from typing import Any, Callable, Dict, List, Sequence

import numpy as np

from ..core.env import Env

__all__ = ["AsyncVectorEnv"]


def _worker(remote, parent_remote, env_fn) -> None:
    parent_remote.close()
    env = env_fn()
    try:
        while True:
            cmd, data = remote.recv()
            if cmd == "step":
                obs, reward, terminated, truncated, info = env.step(data)
                if terminated or truncated:
                    info = dict(info)
                    info["final_observation"] = obs
                    obs, _ = env.reset()
                remote.send((obs, reward, terminated, truncated, info))
            elif cmd == "reset":
                seed, options = data
                obs, info = env.reset(seed=seed, options=options)
                remote.send((obs, info))
            elif cmd == "close":
                remote.close()
                break
            else:  # pragma: no cover - defensive
                raise RuntimeError(f"unknown command {cmd!r}")
    except (KeyboardInterrupt, EOFError):  # pragma: no cover
        pass
    finally:
        env.close()


class AsyncVectorEnv:
    def __init__(self, env_fns: Sequence[Callable[[], Env]]) -> None:
        assert len(env_fns) >= 1, "need at least one env"
        self.num_envs = len(env_fns)
        ctx = mp.get_context("spawn")

        # Read spaces once in the parent to avoid an extra round-trip.
        probe = env_fns[0]()
        self.single_observation_space = probe.observation_space
        self.single_action_space = probe.action_space
        self.observation_space = self.single_observation_space
        self.action_space = self.single_action_space
        probe.close()

        self.remotes, work_remotes = zip(*[ctx.Pipe() for _ in range(self.num_envs)])
        self.processes: List[mp.Process] = []
        for work_remote, remote, fn in zip(work_remotes, self.remotes, env_fns):
            p = ctx.Process(target=_worker, args=(work_remote, remote, fn), daemon=True)
            p.start()
            self.processes.append(p)
            work_remote.close()
        self.closed = False

    def reset(self, *, seed=None, options=None):
        for i, remote in enumerate(self.remotes):
            env_seed = None if seed is None else seed + i
            remote.send(("reset", (env_seed, options)))
        results = [remote.recv() for remote in self.remotes]
        return np.asarray([r[0] for r in results]), {}

    def step(self, actions: Sequence[Any]):
        for remote, action in zip(self.remotes, actions):
            remote.send(("step", action))
        results = [remote.recv() for remote in self.remotes]

        obs = np.asarray([r[0] for r in results])
        rewards = np.array([r[1] for r in results], dtype=np.float32)
        terminateds = np.array([r[2] for r in results], dtype=bool)
        truncateds = np.array([r[3] for r in results], dtype=bool)

        infos: Dict[str, Any] = {}
        final_obs: List[Any] = [None] * self.num_envs
        final_info: List[Any] = [None] * self.num_envs
        any_final = False
        for i, r in enumerate(results):
            info = r[4]
            if "final_observation" in info:
                any_final = True
                final_obs[i] = info.pop("final_observation")
                final_info[i] = info
        if any_final:
            infos["final_observation"] = final_obs
            infos["final_info"] = final_info
        return obs, rewards, terminateds, truncateds, infos

    def close(self) -> None:
        if self.closed:
            return
        for remote in self.remotes:
            with contextlib.suppress(BrokenPipeError, OSError):  # pragma: no cover
                remote.send(("close", None))
        for p in self.processes:
            p.join(timeout=5)
        for p in self.processes:
            if p.is_alive():  # pragma: no cover
                p.terminate()
        self.closed = True

    def __del__(self):  # pragma: no cover
        self.close()
