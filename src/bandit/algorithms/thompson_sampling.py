"""Thompson Sampling bandit algorithm using a Beta-Bernoulli model.

For each arm the algorithm maintains a Beta(alpha, beta) posterior
distribution over the arm's true click probability.  At decision
time it draws one sample from each arm's posterior and selects the
arm with the highest sampled value.

Mathematical background:
    - Prior: Beta(1, 1) (uniform over [0, 1]) for each unseen arm.
    - Likelihood: Bernoulli (reward is 1.0 for click, 0.0 otherwise).
    - Posterior update (conjugate):
        alpha <- alpha + reward
        beta  <- beta  + (1 - reward)

This naturally balances exploration and exploitation: arms with
uncertain posteriors (low alpha + beta) will occasionally produce
high samples, prompting exploration, while arms with strong evidence
of high reward will be sampled high consistently.
"""

from collections import defaultdict

import numpy as np

from bandit.algorithms.base import BanditAlgorithm


class ThompsonSampling(BanditAlgorithm):
    """Beta-Bernoulli Thompson Sampling.

    Maintains a Beta posterior for each arm and selects the arm
    whose posterior sample is largest.  This provides a principled
    Bayesian approach to the exploration/exploitation trade-off.

    Args:
        seed: Optional RNG seed for reproducibility.
    """

    def __init__(self, seed: int | None = None) -> None:
        self._rng = np.random.default_rng(seed)
        # Beta posterior parameters: default prior is Beta(1, 1)
        self._alpha: dict[str, float] = defaultdict(lambda: 1.0)
        self._beta: dict[str, float] = defaultdict(lambda: 1.0)

    def select_arm(
        self,
        arm_ids: list[str],
        context: np.ndarray | None = None,
    ) -> str:
        """Sample from each arm's Beta posterior; pick the highest.

        Args:
            arm_ids: Candidate article IDs for this round.
            context: Ignored by this algorithm.

        Returns:
            The article ID with the highest posterior sample.
        """
        best_arm = arm_ids[0]
        best_sample = self._rng.beta(
            self._alpha[arm_ids[0]], self._beta[arm_ids[0]]
        )

        for arm in arm_ids[1:]:
            sample = self._rng.beta(
                self._alpha[arm], self._beta[arm]
            )
            if sample > best_sample:
                best_sample = sample
                best_arm = arm

        return best_arm

    def update(
        self,
        arm_id: str,
        reward: float,
        context: np.ndarray | None = None,
    ) -> None:
        """Update the Beta posterior for the selected arm.

        Uses the conjugate update rule:
            alpha <- alpha + reward
            beta  <- beta  + (1 - reward)

        Args:
            arm_id: The recommended article ID.
            reward: The observed reward (1.0 for click, 0.0 otherwise).
            context: Optional context vector (unused).
        """
        self._alpha[arm_id] += reward
        self._beta[arm_id] += 1.0 - reward
