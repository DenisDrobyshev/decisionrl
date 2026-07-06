"""Multi-agent RL: self-play on Rock-Paper-Scissors and independent PPO on a
cooperative coordination game.

Run: python examples/multiagent_selfplay.py
"""

import numpy as np

from reinforce.multiagent import CoordinationGame, MultiAgentPPO, RockPaperScissors
from reinforce.utils import set_seed


def main() -> None:
    # --- Self-play on Rock-Paper-Scissors (competitive, zero-sum) ---
    set_seed(0)
    rps = MultiAgentPPO(RockPaperScissors(), shared_policy=True, n_steps=256, ent_coef=0.01, seed=0)
    rps.learn(40_000)
    probs = rps.policy_probs("player_0", np.zeros(1, np.float32))
    print(f"RPS self-play strategy (rock/paper/scissors): {np.round(probs, 2)}")

    # --- Independent PPO on a cooperative coordination game ---
    set_seed(0)
    coord = MultiAgentPPO(CoordinationGame(n_agents=2, n_actions=3), shared_policy=False,
                          n_steps=128, ent_coef=0.01, seed=0)
    coord.learn(20_000)
    env = CoordinationGame(n_agents=2, n_actions=3)
    obs, _ = env.reset()
    actions = {a: coord.predict(obs[a], agent=a) for a in env.agents}
    _, rewards, _, _, _ = env.step(actions)
    print(f"Coordination greedy actions={actions}, joint reward={rewards['agent_0']}")


if __name__ == "__main__":
    main()
