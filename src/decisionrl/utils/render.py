"""Rendering helpers: figure-to-array and GIF recording of episodes.

Rendering is optional and pulls in matplotlib/Pillow lazily, so the core stays
dependency-light. Environments that support it expose ``render_rgb() -> (H,W,3)``.
"""

from __future__ import annotations

from typing import List

import numpy as np

__all__ = ["fig_to_rgb", "line_frame", "bars_frame", "record_gif"]


def fig_to_rgb(fig) -> np.ndarray:
    """Rasterize a matplotlib figure to an ``(H, W, 3)`` uint8 array."""
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())
    return buf[..., :3].copy()


def line_frame(series, title=None, xlim=None, ylim=None, hspans=None,
               figsize=(4.2, 3.0), dpi=64) -> np.ndarray:
    """Rasterize a small multi-line plot to ``(H, W, 3)`` — a reusable render_rgb body.

    ``series`` is a list of ``(label, values, color)``; ``hspans`` an optional list
    of ``(lo, hi, color)`` shaded horizontal bands (e.g. a comfort/target zone).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    for lo, hi, color in (hspans or []):
        ax.axhspan(lo, hi, color=color, alpha=0.15)
    for label, values, color in series:
        ax.plot(range(len(values)), values, label=label, color=color, lw=2)
    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)
    if title:
        ax.set_title(title, fontsize=10, fontweight="bold")
    ax.legend(fontsize=7, loc="upper right")
    fig.tight_layout()
    frame = fig_to_rgb(fig)
    plt.close(fig)
    return frame


def bars_frame(labels, values, ymax, ymin=0.0, colors=None, title=None,
               figsize=(3.6, 3.0), dpi=64) -> np.ndarray:
    """Rasterize a small bar chart of the current state to ``(H, W, 3)``.

    A reusable ``render_rgb`` body for envs whose state is a few scalar levels
    (inventory, queue occupancy, battery charge, ...).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    x = range(len(values))
    ax.bar(x, values, color=colors or "#2563eb")
    ax.axhline(0, color="#334155", lw=0.8)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(ymin, ymax)
    if title:
        ax.set_title(title, fontsize=10, fontweight="bold")
    fig.tight_layout()
    frame = fig_to_rgb(fig)
    plt.close(fig)
    return frame


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
