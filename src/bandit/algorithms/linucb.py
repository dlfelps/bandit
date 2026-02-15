"""LinUCB contextual bandit algorithm (disjoint model).

Implements the disjoint linear Upper Confidence Bound algorithm from
Li et al., "A Contextual-Bandit Approach to Personalized News Article
Recommendation" (WWW 2010).

Each arm maintains its own linear model.  At decision time the
algorithm computes an upper confidence bound on the expected reward
for each candidate and selects the arm with the highest UCB.
"""

import numpy as np

from bandit.algorithms.base import BanditAlgorithm


class LinUCB(BanditAlgorithm):
    """Disjoint LinUCB with per-arm linear models.

    For each arm *a* the algorithm maintains:
        - A_a: (d x d) design matrix, initialised to the identity.
        - b_a: (d,) reward-weighted feature accumulator, initialised
          to zeros.

    The predicted reward plus exploration bonus is:
        UCB_a = theta_a^T x_a + alpha * sqrt(x_a^T A_a^{-1} x_a)
    where theta_a = A_a^{-1} b_a.

    Args:
        alpha: Exploration parameter controlling the width of the
            confidence interval.  Higher values encourage more
            exploration.
        seed: Optional RNG seed for reproducibility (used when
            falling back to random selection without context).
    """

    def __init__(
        self,
        alpha: float = 1.0,
        seed: int | None = None,
    ) -> None:
        self._alpha = alpha
        self._rng = np.random.default_rng(seed)
        self._d: int | None = None
        self._A: dict[str, np.ndarray] = {}
        self._b: dict[str, np.ndarray] = {}

    def _init_arm(self, arm_id: str) -> None:
        """Lazily initialise the model for *arm_id*."""
        assert self._d is not None
        self._A[arm_id] = np.eye(self._d)
        self._b[arm_id] = np.zeros(self._d)

    def _ensure_dim(self, x: np.ndarray) -> None:
        """Set dimensionality from the first feature vector seen."""
        if self._d is None:
            self._d = x.shape[0]

    def select_arm(
        self,
        arm_ids: list[str],
        context: dict[str, np.ndarray] | None = None,
    ) -> str:
        """Select the arm with the highest UCB score.

        Falls back to uniform random selection when no context is
        provided.

        Args:
            arm_ids: Candidate article IDs for this round.
            context: Dict mapping each arm_id to its feature vector.

        Returns:
            The article ID with the highest UCB score.
        """
        if context is None:
            idx = self._rng.integers(len(arm_ids))
            return arm_ids[idx]

        best_arm = arm_ids[0]
        best_ucb = -np.inf

        for arm in arm_ids:
            x = context[arm]
            self._ensure_dim(x)
            if arm not in self._A:
                self._init_arm(arm)

            theta = np.linalg.solve(self._A[arm], self._b[arm])
            bonus = self._alpha * np.sqrt(
                x @ np.linalg.solve(self._A[arm], x)
            )
            ucb = theta @ x + bonus

            if ucb > best_ucb:
                best_ucb = ucb
                best_arm = arm

        return best_arm

    def update(
        self,
        arm_id: str,
        reward: float,
        context: np.ndarray | None = None,
    ) -> None:
        """Update the linear model for the selected arm.

        Args:
            arm_id: The recommended article ID.
            reward: The observed reward (1.0 for click, 0.0 otherwise).
            context: The feature vector used when this arm was selected.
        """
        if context is None:
            return

        self._ensure_dim(context)
        if arm_id not in self._A:
            self._init_arm(arm_id)

        self._A[arm_id] += np.outer(context, context)
        self._b[arm_id] += reward * context
