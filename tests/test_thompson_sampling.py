"""Tests for the ThompsonSampling bandit algorithm.

Verifies Bayesian posterior sampling behaviour: the algorithm should
maintain a Beta(alpha, beta) posterior for each arm and select the
arm whose sampled value is highest. Arms with more observed clicks
should be favoured over time.
"""

import numpy as np
import pytest

from bandit.algorithms.base import BanditAlgorithm
from bandit.algorithms.thompson_sampling import ThompsonSampling


class TestThompsonSamplingInterface:
    """Test that ThompsonSampling conforms to BanditAlgorithm."""

    def test_is_bandit_algorithm_subclass(self) -> None:
        """ThompsonSampling should inherit from BanditAlgorithm."""
        algo = ThompsonSampling()
        assert isinstance(algo, BanditAlgorithm)

    def test_name_property(self) -> None:
        """name should return 'ThompsonSampling'."""
        algo = ThompsonSampling()
        assert algo.name == "ThompsonSampling"


class TestThompsonSamplingSelection:
    """Test arm selection via posterior sampling."""

    def test_returns_valid_arm(self) -> None:
        """select_arm should return one of the candidates."""
        algo = ThompsonSampling(seed=42)
        candidates = ["A", "B", "C"]
        for _ in range(20):
            assert algo.select_arm(candidates) in candidates

    def test_favours_high_reward_arm_after_updates(self) -> None:
        """An arm with many clicks should be selected most often."""
        algo = ThompsonSampling(seed=0)

        # Arm A: 20 clicks out of 20 (perfect)
        for _ in range(20):
            algo.update("A", 1.0)
        # Arm B: 0 clicks out of 20 (never clicked)
        for _ in range(20):
            algo.update("B", 0.0)

        candidates = ["A", "B"]
        counts = {"A": 0, "B": 0}
        for _ in range(500):
            counts[algo.select_arm(candidates)] += 1

        # A should dominate (>90% of selections)
        assert counts["A"] > 450, (
            f"Arm A selected {counts['A']}/500 times, expected >450"
        )

    def test_explores_uncertain_arms(self) -> None:
        """With no prior data, all arms should be explored."""
        algo = ThompsonSampling(seed=42)
        candidates = ["A", "B", "C", "D"]

        counts: dict[str, int] = {c: 0 for c in candidates}
        for _ in range(2000):
            counts[algo.select_arm(candidates)] += 1

        # Each arm should be selected a reasonable number of times.
        for arm, count in counts.items():
            assert count > 200, (
                f"Arm {arm} selected {count}/2000 times, "
                "expected >200 with uniform prior"
            )

    def test_single_candidate(self) -> None:
        """With one candidate, that candidate must be returned."""
        algo = ThompsonSampling(seed=42)
        assert algo.select_arm(["A"]) == "A"


class TestThompsonSamplingUpdate:
    """Test Beta posterior updates."""

    def test_update_records_success(self) -> None:
        """A reward of 1.0 should increase the arm's alpha."""
        algo = ThompsonSampling(seed=42)
        algo.update("A", 1.0)
        # After one success: alpha=2, beta=1 (from prior alpha=1, beta=1)
        assert algo._alpha["A"] == 2.0
        assert algo._beta["A"] == 1.0

    def test_update_records_failure(self) -> None:
        """A reward of 0.0 should increase the arm's beta."""
        algo = ThompsonSampling(seed=42)
        algo.update("A", 0.0)
        # After one failure: alpha=1, beta=2
        assert algo._alpha["A"] == 1.0
        assert algo._beta["A"] == 2.0

    def test_multiple_updates_accumulate(self) -> None:
        """Multiple updates should accumulate in the posterior."""
        algo = ThompsonSampling(seed=42)
        algo.update("A", 1.0)
        algo.update("A", 1.0)
        algo.update("A", 0.0)
        # alpha = 1 + 2 = 3, beta = 1 + 1 = 2
        assert algo._alpha["A"] == 3.0
        assert algo._beta["A"] == 2.0

    def test_update_accepts_context(self) -> None:
        """update() should accept an optional context vector."""
        algo = ThompsonSampling()
        algo.update("A", 1.0, context=np.array([1.0, 2.0]))


class TestThompsonSamplingEdgeCases:
    """Test edge cases and parameter handling."""

    def test_unseen_arms_use_uniform_prior(self) -> None:
        """Arms with no data should use Beta(1, 1) = Uniform."""
        algo = ThompsonSampling(seed=42)
        # No updates — alpha and beta default to 1.0
        assert algo._alpha["NEW_ARM"] == 1.0
        assert algo._beta["NEW_ARM"] == 1.0

    def test_select_arm_accepts_context(self) -> None:
        """select_arm should accept an optional context vector."""
        algo = ThompsonSampling(seed=42)
        context = np.array([0.5, 0.3])
        selected = algo.select_arm(["A", "B"], context=context)
        assert selected in ["A", "B"]

    def test_seed_produces_reproducible_results(self) -> None:
        """Two instances with the same seed should agree."""
        algo_a = ThompsonSampling(seed=7)
        algo_b = ThompsonSampling(seed=7)
        candidates = ["A", "B", "C"]

        for _ in range(50):
            sel_a = algo_a.select_arm(candidates)
            sel_b = algo_b.select_arm(candidates)
            assert sel_a == sel_b
            algo_a.update(sel_a, 1.0)
            algo_b.update(sel_b, 1.0)
