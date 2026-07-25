"""Visualise what a learned inventory policy actually does, and how it differs from the
classical rules.

Trains a DQN on :class:`~decisionrl.envs.NonstationaryInventory` and plots the order
quantity it chooses across the whole state space (inventory on one axis, recent demand on
the other), next to the two classical rules on the same grid:

* fixed base-stock  -> order depends only on inventory (horizontal bands; blind to demand)
* adaptive tracking -> order rises with recent demand and falls with inventory
* learned policy     -> should resemble the adaptive rule, discovered from reward alone

Seeing the learned rule as a picture is a simple, honest form of explainability: it makes
plain that the agent learned to raise its order-up-to level when recent demand is high,
which is exactly why it beats a policy that commits to one level.

Run: python examples/policy_heatmap.py [--steps 100000] [--device auto] [--out PATH]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from decisionrl import baselines as B
from decisionrl.algorithms import DQN
from decisionrl.envs import NonstationaryInventory
from decisionrl.utils import Logger, set_seed


def order_grid(action_fn, max_inv: int, max_order: int) -> np.ndarray:
    """Order quantity chosen at each (inventory, recent-demand) state."""
    grid = np.zeros((max_inv + 1, max_order + 1))
    for inv in range(max_inv + 1):
        for demand in range(max_order + 1):
            obs = np.array([inv / max_inv, min(demand / max_order, 1.0)], dtype=np.float32)
            grid[inv, demand] = action_fn(obs)
    return grid


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=100_000)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default="docs/assets/policy_heatmap.png")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    env = NonstationaryInventory()
    max_inv, max_order = env.max_inventory, env.max_order

    set_seed(0)
    agent = DQN(NonstationaryInventory(), seed=0, logger=Logger(verbose=0), device=args.device,
                learning_rate=5e-4, buffer_size=50_000, learning_starts=1000,
                target_update_interval=500)
    agent.learn(args.steps)

    level, _ = B.best_base_stock(NonstationaryInventory, seed=100)
    safety, _ = B.best_tracking_base_stock(NonstationaryInventory, seed=100)
    fixed_p, track_p = B.base_stock(level), B.tracking_base_stock(safety)

    grids = {
        "Fixed base-stock": order_grid(lambda o: fixed_p(env, o), max_inv, max_order),
        "Adaptive tracking": order_grid(lambda o: track_p(env, o), max_inv, max_order),
        "Learned policy (DQN)": order_grid(
            lambda o: agent.predict(o, deterministic=True), max_inv, max_order),
    }

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), constrained_layout=True)
    im = None
    for ax, (title, grid) in zip(axes, grids.items()):
        im = ax.imshow(grid, origin="lower", aspect="auto", cmap="viridis",
                       vmin=0, vmax=max_order)
        ax.set_title(title)
        ax.set_xlabel("recent demand")
        ax.set_ylabel("inventory on hand")
    fig.colorbar(im, ax=axes, label="order quantity", shrink=0.85)
    fig.suptitle("Order quantity by state: the learned policy tracks recent demand",
                 fontsize=12)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
