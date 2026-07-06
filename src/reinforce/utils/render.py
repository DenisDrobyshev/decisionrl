"""Rendering helpers: figure-to-array and GIF recording of episodes.

Rendering is optional and pulls in matplotlib/Pillow lazily, so the core stays
dependency-light. Environments that support it expose ``render_rgb() -> (H,W,3)``.
"""

from __future__ import annotations

from typing import List

import numpy as np

__all__ = ["fig_to_rgb", "record_gif"]


def fig_to_rgb(fig) -> np.ndarray:
    """Rasterize a matplotlib figure to an ``(H, W, 3)`` uint8 array."""
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())
    return buf[..., :3].copy()


def record_gif(agent, env, path: str, max_steps: int = 500, fps: int = 30,
               deterministic: bool = True, seed: int = 0) -> str:
    """Roll out ``agent`` in ``env`` and save the episode as an animated GIF.

    Requires ``env.render_rgb()`` and Pillow. Returns the output path.
    """
    from PIL import Image

    if not hasattr(env, "render_rgb"):
        raise TypeError(f"{type(env).__name__} does not implement render_rgb()")

    frames: List[np.ndarray] = []
    obs, _ = env.reset(seed=seed)
    if hasattr(agent, "reset_states"):
        agent.reset_states()
    frames.append(env.render_rgb())
    done, steps = False, 0
    while not done and steps < max_steps:
        action = agent.predict(obs, deterministic=deterministic)
        obs, _, terminated, truncated, _ = env.step(action)
        frames.append(env.render_rgb())
        done = terminated or truncated
        steps += 1

    images = [Image.fromarray(f) for f in frames]
    images[0].save(path, save_all=True, append_images=images[1:],
                   duration=int(1000 / fps), loop=0, optimize=True)
    return path
