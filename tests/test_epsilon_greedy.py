"""Tests for the EpsilonGreedy bandit algorithm.

Verifies exploration/exploitation behaviour: the algorithm should
exploit (pick the best-known arm) with probability 1-epsilon and
explore (pick a random arm) with probability epsilon.
"""

import numpy as np
import pytest

from bandit.algorithms.base import BanditAlgorithm
from bandit.algorithms.epsilon_greedy import EpsilonGreedy


class TestEpsilonGreedyInterface:
    """Test that EpsilonGreedy conforms to BanditAlgorithm."""

    def test_is_bandit_algorithm_subclass(self) -> None:
        """EpsilonGreedy should inherit from BanditAlgorithm."""
        algo = EpsilonGreedy(epsilon=0.1)
        assert isinstance(algo, BanditAlgorithm)

    def test_name_property(self) -> None:
        """name should return 'EpsilonGreedy'."""
        algo = EpsilonGreedy(epsilon=0.1)
        assert algo.name == "EpsilonGreedy"


class TestEpsilonGreedyExploration:
    """Test exploration (random selection) behaviour."""

    def test_epsilon_one_selects_uniformly(self) -> None:
        """With epsilon=1.0 the algorithm should explore every time."""
        algo = EpsilonGreedy(epsilon=1.0, seed=0)
        candidates = ["A", "B", "C", "D"]

        # Give arm A a high reward so greedy would always pick it.
        algo.update("A", 1.0)

        counts: dict[str, int] = {c: 0 for c in candidates}
        n_trials = 4000
        for _ in range(n_trials):
            counts[algo.select_arm(candidates)] += 1

        # All arms should be selected roughly equally.
        expected = n_trials / len(candidates)
        for arm, count in counts.items():
            assert count == pytest.approx(expected, rel=0.15), (
                f"Arm {arm} selected {count} times, expected ~{expected}"
            )

    def test_returns_valid_arm_during_exploration(self) -> None:
        """Even during exploration, the selected arm must be valid."""
        algo = EpsilonGreedy(epsilon=1.0, seed=42)
        candidates = ["X", "Y"]
        for _ in range(20):
            assert algo.select_arm(candidates) in candidates


class TestEpsilonGreedyExploitation:
    """Test exploitation (greedy) behaviour."""

    def test_epsilon_zero_always_picks_best_arm(self) -> None:
        """With epsilon=0 the algorithm should always exploit."""
        algo = EpsilonGreedy(epsilon=0.0, seed=42)

        # Teach the algorithm that B has a higher average reward.
        algo.update("A", 0.0)
        algo.update("B", 1.0)

        candidates = ["A", "B"]
        for _ in range(20):
            assert algo.select_arm(candidates) == "B"

    def test_greedy_picks_highest_average_reward(self) -> None:
        """Greedy selection should choose the arm with the best mean."""
        algo = EpsilonGreedy(epsilon=0.0, seed=42)
        algo.update("A", 1.0)
        algo.update("A", 0.0)  # avg 0.5
        algo.update("B", 1.0)
        algo.update("B", 1.0)  # avg 1.0
        algo.update("C", 0.0)  # avg 0.0

        candidates = ["A", "B", "C"]
        assert algo.select_arm(candidates) == "B"


class TestEpsilonGreedyUpdate:
    """Test reward tracking via update()."""

    def test_update_records_reward(self) -> None:
        """After updates, the algorithm should track arm statistics."""
        algo = EpsilonGreedy(epsilon=0.0, seed=42)
        algo.update("A", 1.0)
        algo.update("A", 0.0)
        algo.update("B", 1.0)

        # Greedy should pick B (avg 1.0) over A (avg 0.5).
        assert algo.select_arm(["A", "B"]) == "B"

    def test_update_accepts_context(self) -> None:
        """update() should accept an optional context vector."""
        algo = EpsilonGreedy(epsilon=0.1)
        algo.update("A", 1.0, context=np.array([1.0, 2.0]))


class TestEpsilonGreedyEdgeCases:
    """Test edge cases and cold-start behaviour."""

    def test_unseen_arms_default_to_zero(self) -> None:
        """Arms with no prior data should have zero estimated reward."""
        algo = EpsilonGreedy(epsilon=0.0, seed=42)
        algo.update("A", 1.0)

        # Only A has been seen; B/C default to 0.  Greedy picks A.
        assert algo.select_arm(["A", "B", "C"]) == "A"

    def test_all_arms_unseen_returns_valid_arm(self) -> None:
        """When no arm has been seen, select_arm should still work."""
        algo = EpsilonGreedy(epsilon=0.0, seed=42)
        candidates = ["A", "B", "C"]
        selected = algo.select_arm(candidates)
        assert selected in candidates

    def test_single_candidate(self) -> None:
        """With one candidate, that candidate must be returned."""
        algo = EpsilonGreedy(epsilon=0.5, seed=42)
        assert algo.select_arm(["A"]) == "A"

    def test_select_arm_accepts_context(self) -> None:
        """select_arm should accept an optional context vector."""
        algo = EpsilonGreedy(epsilon=0.1, seed=42)
        context = np.array([0.5, 0.3])
        selected = algo.select_arm(["A", "B"], context=context)
        assert selected in ["A", "B"]

    def test_seed_produces_reproducible_results(self) -> None:
        """Two instances with the same seed and updates should agree."""
        algo_a = EpsilonGreedy(epsilon=0.3, seed=7)
        algo_b = EpsilonGreedy(epsilon=0.3, seed=7)
        candidates = ["A", "B", "C"]

        for _ in range(50):
            sel_a = algo_a.select_arm(candidates)
            sel_b = algo_b.select_arm(candidates)
            assert sel_a == sel_b
            # Keep them in sync.
            algo_a.update(sel_a, 1.0)
            algo_b.update(sel_b, 1.0)
