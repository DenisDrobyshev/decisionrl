"""Applied case study on real data: inventory control under real US consumer demand.

The demand series is quarterly US real personal consumption expenditure (1959-2009),
a genuine, trending, recession-punctuated demand signal shipped with statsmodels and
saved to ``examples/data/us_consumption.csv``. Rescaled to a single product's demand,
it grows several-fold across the decades, so no single order-up-to level serves both the
low-demand early era and the high-demand later era. A base-stock policy must commit to
one level; a learned policy that reads recent demand can track the era and adapt.

This script finds the best fixed base-stock (an exhaustive 1-D search, the exact optimum
within that family), trains an agent over several seeds, and reports the interquartile
mean with a 95% bootstrap confidence interval so the comparison is not a single-seed
fluke.

Run: python examples/real_data_case_study.py [--seeds 5] [--steps 100000] [--device auto]
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import time
from pathlib import Path

from decisionrl.algorithms import DQN
from decisionrl.baselines import best_base_stock
from decisionrl.envs import DatasetDemandInventory
from decisionrl.evaluation import bootstrap_ci, iqm
from decisionrl.training import evaluate_policy
from decisionrl.utils import Logger, set_seed

DATA = Path(__file__).parent / "data" / "us_consumption.csv"


def load_demand_series(path: Path = DATA):
    with open(path, newline="") as f:
        return [float(row["realcons"]) for row in csv.DictReader(f)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--steps", type=int, default=100_000)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    series = load_demand_series()
    print(f"loaded {len(series)} quarters of real US consumption "
          f"({min(series):.0f}..{max(series):.0f}), rescaled to product demand")

    def make_env():
        return DatasetDemandInventory(demand_series=series)

    level, base_value = best_base_stock(make_env, seed=100)
    print(f"best fixed base-stock: order up to {level:.0f}  ->  {base_value:.1f} return")

    t0 = time.time()
    scores = []
    for s in range(args.seeds):
        set_seed(s)
        agent = DQN(make_env(), seed=s, logger=Logger(verbose=0), device=args.device,
                    learning_rate=5e-4, buffer_size=50_000, learning_starts=1000,
                    target_update_interval=500)
        agent.learn(args.steps)
        score = evaluate_policy(agent, make_env(), n_episodes=50, seed=100)[0]
        scores.append(score)
        print(f"  seed {s}: {score:.1f}")

    point, lo, hi = bootstrap_ci(scores, aggregate=iqm, reps=5000, seed=0)
    lift = 100.0 * (point - base_value) / abs(base_value)
    print(f"\nlearned policy IQM {point:.1f} [95% CI {lo:.1f}, {hi:.1f}]  "
          f"vs base-stock {base_value:.1f}  ({lift:+.0f}%)")
    print(f"{args.seeds} seeds, {args.steps} steps each, {time.time() - t0:.0f}s")

    out = {"dataset": "US real personal consumption expenditure (statsmodels macrodata)",
           "n_quarters": len(series), "base_stock_level": level,
           "base_stock_value": base_value, "rl_iqm": point, "ci_low": lo, "ci_high": hi,
           "rl_mean": statistics.mean(scores), "seed_values": scores,
           "seeds": args.seeds, "steps": args.steps}
    Path("real_data_case_study.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
