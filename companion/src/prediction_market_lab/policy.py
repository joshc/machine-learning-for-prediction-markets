"""One-step contextual bandit: deliberately restricted synthetic mechanics."""

from dataclasses import dataclass
from math import isfinite
from random import Random
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class LoggedAction:
    context: int
    action: int
    reward: float
    propensity: float

    def __post_init__(self) -> None:
        if self.context not in (0, 1) or self.action not in (0, 1):
            raise ValueError("this simulator has two contexts and wait/buy actions")
        if not isfinite(self.reward) or not isfinite(self.propensity) or not 0 < self.propensity <= 1:
            raise ValueError("finite reward and known positive logging propensity required")


def simulate_logged_policy(rounds: int = 400, seed: int = 20260910,
                           *, adverse_selection: bool = False) -> tuple[LoggedAction, ...]:
    if isinstance(rounds, bool) or not isinstance(rounds, int) or rounds < 1:
        raise ValueError("positive integer simulation horizon required")
    rng, records = Random(seed), []
    for _ in range(rounds):
        context = rng.randrange(2)
        action = rng.randrange(2)
        p = (0.65 if context else 0.35) - (0.30 if adverse_selection else 0)
        payout = int(rng.random() < p)
        reward = payout - 0.51 if action else 0.0
        records.append(LoggedAction(context, action, reward, 0.5))
    return tuple(records)


def learn_tabular_policy(log: Sequence[LoggedAction]) -> tuple[int, int]:
    if not log:
        raise ValueError("empty bandit training log")
    policy = []
    for context in (0, 1):
        rewards = [[row.reward for row in log if row.context == context and row.action == action]
                   for action in (0, 1)]
        if any(not arm for arm in rewards):
            raise ValueError("training support missing for an action/context")
        means = [float(np.mean(arm)) for arm in rewards]
        policy.append(1 if means[1] > means[0] else 0)
    return tuple(policy)


def ips_value(log: Sequence[LoggedAction], policy: Sequence[int]) -> dict[str, float | int]:
    """Ordinary inverse-propensity estimate, not a deployment or uncertainty guarantee."""
    if len(policy) != 2 or any(action not in (0, 1) for action in policy) or not log:
        raise ValueError("two-action context policy and nonempty held-out log required")
    for context in (0, 1):
        if not any(row.context == context and row.action == policy[context] for row in log):
            raise ValueError("off-policy support absent for a target context/action")
    weights = np.asarray([1 / row.propensity if row.action == policy[row.context] else 0 for row in log])
    rewards = np.asarray([row.reward for row in log])
    estimate = float(np.mean(weights * rewards))
    ess = float(weights.sum() ** 2 / np.sum(weights ** 2))
    return {"ips_reward_per_round": estimate, "effective_weight_sample_size": ess, "logged_rounds": len(log)}
