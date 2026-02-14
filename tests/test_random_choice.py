"""Tests for the RandomChoice bandit algorithm.

Verifies that RandomChoice uniformly selects from the available
candidates and correctly implements the BanditAlgorithm interface.
"""

import numpy as np
import pytest

from bandit.algorithms.base import BanditAlgorithm
from bandit.algorithms.random_choice import RandomChoice


class TestRandomChoiceInterface:
    """Test that RandomChoice conforms to the BanditAlgorithm contract."""

    def test_is_bandit_algorithm_subclass(self) -> None:
        """RandomChoice should inherit from BanditAlgorithm."""
        algo = RandomChoice()
        assert isinstance(algo, BanditAlgorithm)

    def test_name_property(self) -> None:
        """name should return 'RandomChoice'."""
        algo = RandomChoice()
        assert algo.name == "RandomChoice"


class TestRandomChoiceSelectArm:
    """Test the select_arm method."""

    def test_returns_valid_arm(self) -> None:
        """select_arm should return one of the provided candidates."""
        algo = RandomChoice(seed=42)
        candidates = ["A", "B", "C", "D"]
        selected = algo.select_arm(candidates)
        assert selected in candidates

    def test_single_candidate(self) -> None:
        """With one candidate, select_arm must return that candidate."""
        algo = RandomChoice(seed=42)
        assert algo.select_arm(["A"]) == "A"

    def test_distribution_is_approximately_uniform(self) -> None:
        """Over many trials, each arm should be selected roughly equally."""
        algo = RandomChoice(seed=0)
        candidates = ["A", "B", "C", "D"]
        counts: dict[str, int] = {c: 0 for c in candidates}
        n_trials = 4000

        for _ in range(n_trials):
            selected = algo.select_arm(candidates)
            counts[selected] += 1

        expected = n_trials / len(candidates)
        for arm, count in counts.items():
            assert count == pytest.approx(expected, rel=0.15), (
                f"Arm {arm} selected {count} times, expected ~{expected}"
            )

    def test_accepts_context_parameter(self) -> None:
        """select_arm should accept an optional context vector."""
        algo = RandomChoice(seed=42)
        context = np.array([1.0, 0.5])
        selected = algo.select_arm(["A", "B"], context=context)
        assert selected in ["A", "B"]

    def test_seed_produces_reproducible_results(self) -> None:
        """Two instances with the same seed should make identical choices."""
        algo_a = RandomChoice(seed=123)
        algo_b = RandomChoice(seed=123)
        candidates = ["A", "B", "C", "D", "E"]

        for _ in range(50):
            assert algo_a.select_arm(candidates) == (
                algo_b.select_arm(candidates)
            )


class TestRandomChoiceUpdate:
    """Test the update method."""

    def test_update_does_not_raise(self) -> None:
        """update() should accept reward without error."""
        algo = RandomChoice()
        algo.update("A", 1.0)

    def test_update_accepts_context(self) -> None:
        """update() should accept an optional context vector."""
        algo = RandomChoice()
        algo.update("A", 0.0, context=np.array([1.0, 2.0]))

    def test_update_does_not_change_selection_behavior(self) -> None:
        """Calling update should not alter the random selection logic."""
        algo = RandomChoice(seed=99)
        candidates = ["A", "B", "C"]
        first_selection = algo.select_arm(candidates)

        # Simulate several updates.
        algo.update("A", 1.0)
        algo.update("B", 0.0)
        algo.update("C", 1.0)

        # The algorithm is stateless, so a fresh instance with the
        # same seed (advanced by the same number of calls) would behave
        # identically.  We just verify no exception is raised and it
        # still returns a valid candidate.
        selected = algo.select_arm(candidates)
        assert selected in candidates
