"""RLHF + GRPO: align a policy from preferences, LLM-style, on control tasks.

Two demos:

1. **RLHF pipeline** on PointMass — learn a reward model from *preferences*
   (a synthetic teacher that prefers higher-return behaviour), then optimize a
   policy against the *learned* reward and measure the recovered *true* return.
2. **GRPO** on CartPole — the critic-free, group-relative policy-optimization
   method used to align language models, learning from scratch on the true reward.

Run: python examples/rlhf_grpo.py
"""

from __future__ import annotations

import numpy as np

from decisionrl.algorithms import GRPO, SAC
from decisionrl.envs import CartPole, PointMass
from decisionrl.rlhf import (
    RewardModel,
    RewardModelWrapper,
    collect_segments,
    synthetic_preferences,
    train_reward_model,
)
from decisionrl.training import evaluate_policy
from decisionrl.utils import Logger, set_seed


def rlhf_pointmass() -> None:
    set_seed(0)
    env = PointMass()
    print("[RLHF] collecting behaviour segments and preferences ...")
    segments = collect_segments(env, lambda o: env.action_space.sample(), 150, seg_len=25, seed=0)
    prefs = synthetic_preferences(segments, n_pairs=1000, rational=True, seed=1)

    reward_model = RewardModel(obs_dim=2, action_space=env.action_space, use_action=False)
    metrics = train_reward_model(reward_model, prefs, n_iters=600, batch_size=32)
    print(f"[RLHF] reward model: loss={metrics['loss']:.3f}  pref-accuracy={metrics['accuracy']:.2f}")

    # How well does the learned reward match the true reward r(s) = -||s|| ?
    grid = np.random.default_rng(2).uniform(-1, 1, size=(500, 2)).astype(np.float32)
    corr = float(np.corrcoef(-np.linalg.norm(grid, axis=1), reward_model.predict_rewards(grid))[0, 1])
    print(f"[RLHF] corr(learned reward, true reward) = {corr:.3f}")

    print("[RLHF] optimizing SAC against the LEARNED reward ...")
    agent = SAC(RewardModelWrapper(PointMass(), reward_model), seed=0, logger=Logger(verbose=0))
    agent.learn(20_000)
    true_mean, _ = evaluate_policy(agent, PointMass(), n_episodes=20, seed=100)
    rand_returns = []
    e = PointMass()
    for ep in range(20):
        o, _ = e.reset(seed=100 + ep)
        done, tot = False, 0.0
        while not done:
            o, r, term, trunc, _ = e.step(e.action_space.sample())
            tot += r
            done = term or trunc
        rand_returns.append(tot)
    print(f"[RLHF] TRUE return — trained-on-learned-reward: {true_mean:.2f}  vs  random: {np.mean(rand_returns):.2f}")


def grpo_cartpole() -> None:
    set_seed(0)
    print("\n[GRPO] training critic-free GRPO on CartPole ...")
    agent = GRPO(
        CartPole(), group_size=8, groups_per_update=4, n_epochs=4,
        learning_rate=1e-3, seed=0, logger=Logger(verbose=0),
    )
    before, _ = evaluate_policy(agent, CartPole(), n_episodes=10, seed=100)
    agent.learn(30_000)
    after, _ = evaluate_policy(agent, CartPole(), n_episodes=10, seed=100)
    print(f"[GRPO] CartPole return: before={before:.1f}  ->  after={after:.1f}")


if __name__ == "__main__":
    rlhf_pointmass()
    grpo_cartpole()
