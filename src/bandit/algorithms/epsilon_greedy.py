"""Epsilon-Greedy bandit algorithm.

With probability *epsilon* the algorithm explores by selecting a
candidate uniformly at random.  With probability *1 - epsilon* it
exploits by choosing the arm with the highest observed average reward.
"""

from collections import defaultdict

import numpy as np

from bandit.algorithms.base import BanditAlgorithm


class EpsilonGreedy(BanditAlgorithm):
    """Epsilon-Greedy exploration/exploitation strategy.

    Maintains a running mean reward for each arm and uses an
    epsilon-probability coin flip to decide between exploration
    (random) and exploitation (best known arm).

    Args:
        epsilon: Probability of exploring (0.0 = pure greedy,
            1.0 = pure random).
        seed: Optional RNG seed for reproducibility.
    """

    def __init__(
        self,
        epsilon: float = 0.1,
        seed: int | None = None,
    ) -> None:
        self._epsilon = epsilon
        self._rng = np.random.default_rng(seed)
        self._total_reward: dict[str, float] = defaultdict(float)
        self._pull_count: dict[str, int] = defaultdict(int)

    def select_arm(
        self,
        arm_ids: list[str],
        context: np.ndarray | None = None,
    ) -> str:
        """Pick an arm using epsilon-greedy logic.

        Args:
            arm_ids: Candidate article IDs for this round.
            context: Ignored by this algorithm.

        Returns:
            The chosen article ID.
        """
        if self._rng.random() < self._epsilon:
            idx = self._rng.integers(len(arm_ids))
            return arm_ids[idx]

        # Exploit: pick the arm with the highest average reward.
        best_arm = arm_ids[0]
        best_avg = self._avg_reward(arm_ids[0])
        for arm in arm_ids[1:]:
            avg = self._avg_reward(arm)
            if avg > best_avg:
                best_avg = avg
                best_arm = arm
        return best_arm

    def update(
        self,
        arm_id: str,
        reward: float,
        context: np.ndarray | None = None,
    ) -> None:
        """Record the observed reward for an arm.

        Args:
            arm_id: The recommended article ID.
            reward: The observed reward (1.0 for click, 0.0 otherwise).
            context: Optional context vector (unused).
        """
        self._total_reward[arm_id] += reward
        self._pull_count[arm_id] += 1

    def _avg_reward(self, arm_id: str) -> float:
        """Return the running average reward for *arm_id*.

        Arms that have never been pulled return 0.0.
        """
        count = self._pull_count[arm_id]
        if count == 0:
            return 0.0
        return self._total_reward[arm_id] / count
