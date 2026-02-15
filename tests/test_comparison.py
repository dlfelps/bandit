"""Tests for the algorithm comparison runner.

Verifies that compare_algorithms() correctly runs multiple bandit
algorithms against the same dataset and returns structured results
for each one.
"""

import numpy as np

from bandit.algorithms.base import BanditAlgorithm
from bandit.simulation.comparison import compare_algorithms


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _FirstArmAlgorithm(BanditAlgorithm):
    """Deterministic algorithm that always picks the first candidate."""

    def select_arm(
        self,
        arm_ids: list[str],
        context: np.ndarray | None = None,
    ) -> str:
        return arm_ids[0]

    def update(
        self,
        arm_id: str,
        reward: float,
        context: np.ndarray | None = None,
    ) -> None:
        pass


class _LastArmAlgorithm(BanditAlgorithm):
    """Deterministic algorithm that always picks the last candidate."""

    def select_arm(
        self,
        arm_ids: list[str],
        context: np.ndarray | None = None,
    ) -> str:
        return arm_ids[-1]

    def update(
        self,
        arm_id: str,
        reward: float,
        context: np.ndarray | None = None,
    ) -> None:
        pass


def _make_impressions(
    specs: list[tuple[list[str], dict[str, float]]],
) -> list[dict]:
    """Build impression dicts from (candidates, rewards) specs."""
    return [
        {
            "user_id": f"U{i:03d}",
            "candidates": candidates,
            "rewards": rewards,
        }
        for i, (candidates, rewards) in enumerate(specs)
    ]


class _FakeLoader:
    """Minimal object that quacks like MINDDataLoader."""

    def __init__(self, impressions: list[dict]) -> None:
        self._impressions = impressions

    @property
    def impressions(self) -> list[dict]:
        return self._impressions

    def __iter__(self):
        return iter(self._impressions)

    def __len__(self) -> int:
        return len(self._impressions)


_IMPRESSIONS = _make_impressions([
    (["A", "B"], {"A": 1.0, "B": 0.0}),
    (["C", "D"], {"C": 0.0, "D": 1.0}),
    (["E", "F"], {"E": 1.0, "F": 0.0}),
])


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestCompareAlgorithmsReturnStructure:
    """Test the shape and keys of results from compare_algorithms."""

    def test_returns_list_with_one_result_per_algorithm(
        self,
    ) -> None:
        """Should return exactly one result dict per algorithm."""
        loader = _FakeLoader(_IMPRESSIONS)
        results = compare_algorithms(
            [_FirstArmAlgorithm(), _LastArmAlgorithm()], loader
        )
        assert len(results) == 2

    def test_each_result_has_required_keys(self) -> None:
        """Each result dict should have the standard metrics keys."""
        loader = _FakeLoader(_IMPRESSIONS)
        results = compare_algorithms(
            [_FirstArmAlgorithm()], loader
        )
        expected_keys = {
            "algorithm",
            "total_impressions",
            "total_clicks",
            "click_through_rate",
            "history",
        }
        assert set(results[0].keys()) == expected_keys

    def test_result_identifies_algorithm_by_name(self) -> None:
        """Each result should carry the algorithm's class name."""
        loader = _FakeLoader(_IMPRESSIONS)
        results = compare_algorithms(
            [_FirstArmAlgorithm(), _LastArmAlgorithm()], loader
        )
        assert results[0]["algorithm"] == "_FirstArmAlgorithm"
        assert results[1]["algorithm"] == "_LastArmAlgorithm"


class TestCompareAlgorithmsMetrics:
    """Test that metrics are correctly computed for each algorithm."""

    def test_both_algorithms_process_same_impression_count(
        self,
    ) -> None:
        """All algorithms should run over the full dataset."""
        loader = _FakeLoader(_IMPRESSIONS)
        results = compare_algorithms(
            [_FirstArmAlgorithm(), _LastArmAlgorithm()], loader
        )
        assert results[0]["total_impressions"] == 3
        assert results[1]["total_impressions"] == 3

    def test_different_algorithms_can_have_different_clicks(
        self,
    ) -> None:
        """Algorithms picking different arms may get different rewards."""
        # FirstArm picks A(1), C(0), E(1) -> 2 clicks
        # LastArm picks B(0), D(1), F(0) -> 1 click
        loader = _FakeLoader(_IMPRESSIONS)
        results = compare_algorithms(
            [_FirstArmAlgorithm(), _LastArmAlgorithm()], loader
        )
        assert results[0]["total_clicks"] == 2
        assert results[1]["total_clicks"] == 1

    def test_click_through_rates_reflect_selections(self) -> None:
        """CTR should equal total_clicks / total_impressions."""
        loader = _FakeLoader(_IMPRESSIONS)
        results = compare_algorithms(
            [_FirstArmAlgorithm(), _LastArmAlgorithm()], loader
        )
        assert results[0]["click_through_rate"] == 2 / 3
        assert results[1]["click_through_rate"] == 1 / 3


class TestCompareAlgorithmsEdgeCases:
    """Test edge cases for the comparison runner."""

    def test_empty_algorithm_list_returns_empty(self) -> None:
        """No algorithms means no results."""
        loader = _FakeLoader(_IMPRESSIONS)
        results = compare_algorithms([], loader)
        assert results == []

    def test_single_algorithm(self) -> None:
        """Should work with just one algorithm."""
        loader = _FakeLoader(_IMPRESSIONS)
        results = compare_algorithms(
            [_FirstArmAlgorithm()], loader
        )
        assert len(results) == 1
        assert results[0]["total_impressions"] == 3
