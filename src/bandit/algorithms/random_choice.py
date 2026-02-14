"""RandomChoice bandit algorithm.

Selects an arm uniformly at random from the available candidates.
This serves as the simplest baseline — it ignores all reward history
and context, providing a lower bound on expected performance.
"""

import numpy as np

from bandit.algorithms.base import BanditAlgorithm


class RandomChoice(BanditAlgorithm):
    """Uniform-random arm selection.

    This algorithm picks each candidate with equal probability,
    regardless of past rewards.  It is useful as a no-learning
    baseline when evaluating smarter algorithms.

    Args:
        seed: Optional RNG seed for reproducibility.
    """

    def __init__(self, seed: int | None = None) -> None:
        self._rng = np.random.default_rng(seed)

    def select_arm(
        self,
        arm_ids: list[str],
        context: np.ndarray | None = None,
    ) -> str:
        """Return a uniformly random candidate.

        Args:
            arm_ids: Candidate article IDs for this round.
            context: Ignored by this algorithm.

        Returns:
            A randomly chosen article ID.
        """
        idx = self._rng.integers(len(arm_ids))
        return arm_ids[idx]

    def update(
        self,
        arm_id: str,
        reward: float,
        context: np.ndarray | None = None,
    ) -> None:
        """No-op — RandomChoice does not learn from rewards.

        Args:
            arm_id: The recommended article ID (unused).
            reward: The observed reward (unused).
            context: Optional context vector (unused).
        """
