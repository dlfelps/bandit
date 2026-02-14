"""Tests for the SimulationEngine.

Verifies that the engine correctly coordinates the loop between a
BanditAlgorithm and a MINDDataLoader: presenting candidates, recording
selections, collecting rewards, and tracking cumulative metrics.
"""

import numpy as np
import pytest

from bandit.algorithms.base import BanditAlgorithm
from bandit.simulation.engine import SimulationEngine


# ---------------------------------------------------------------------------
# Helpers: minimal concrete algorithm and fake data loader
# ---------------------------------------------------------------------------


class FirstArmAlgorithm(BanditAlgorithm):
    """Deterministic algorithm that always picks the first candidate."""

    def __init__(self) -> None:
        self.update_calls: list[tuple[str, float]] = []

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
        self.update_calls.append((arm_id, reward))


def _make_impressions(
    specs: list[tuple[list[str], dict[str, float]]],
) -> list[dict]:
    """Build a list of impression dicts from (candidates, rewards) specs."""
    return [
        {
            "user_id": f"U{i:03d}",
            "candidates": candidates,
            "rewards": rewards,
        }
        for i, (candidates, rewards) in enumerate(specs)
    ]


class FakeLoader:
    """Minimal object that quacks like MINDDataLoader for testing."""

    def __init__(self, impressions: list[dict]) -> None:
        self._impressions = impressions

    @property
    def impressions(self) -> list[dict]:
        return self._impressions

    def __iter__(self):
        return iter(self._impressions)

    def __len__(self) -> int:
        return len(self._impressions)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestSimulationEngineInit:
    """Test engine creation and parameter validation."""

    def test_creates_with_algorithm_and_loader(self) -> None:
        """Should accept a BanditAlgorithm and a data loader."""
        algo = FirstArmAlgorithm()
        loader = FakeLoader(_make_impressions([]))
        engine = SimulationEngine(algorithm=algo, data_loader=loader)
        assert engine is not None

    def test_exposes_algorithm(self) -> None:
        """Should store and expose the algorithm."""
        algo = FirstArmAlgorithm()
        loader = FakeLoader(_make_impressions([]))
        engine = SimulationEngine(algorithm=algo, data_loader=loader)
        assert engine.algorithm is algo


class TestSimulationRun:
    """Test the core run() simulation loop."""

    def test_run_returns_results_dict(self) -> None:
        """run() should return a dict with summary metrics."""
        algo = FirstArmAlgorithm()
        impressions = _make_impressions([
            (["A", "B"], {"A": 1.0, "B": 0.0}),
        ])
        engine = SimulationEngine(
            algorithm=algo, data_loader=FakeLoader(impressions)
        )
        results = engine.run()
        assert isinstance(results, dict)

    def test_run_tracks_total_impressions(self) -> None:
        """results should include the total number of rounds processed."""
        impressions = _make_impressions([
            (["A", "B"], {"A": 1.0, "B": 0.0}),
            (["C", "D"], {"C": 0.0, "D": 1.0}),
            (["E", "F"], {"E": 0.0, "F": 0.0}),
        ])
        engine = SimulationEngine(
            algorithm=FirstArmAlgorithm(),
            data_loader=FakeLoader(impressions),
        )
        results = engine.run()
        assert results["total_impressions"] == 3

    def test_run_tracks_total_clicks(self) -> None:
        """results should count the number of times the chosen arm was clicked."""
        # FirstArmAlgorithm always picks the first candidate.
        # Round 1: picks A (reward 1.0) -> click
        # Round 2: picks C (reward 0.0) -> no click
        impressions = _make_impressions([
            (["A", "B"], {"A": 1.0, "B": 0.0}),
            (["C", "D"], {"C": 0.0, "D": 1.0}),
        ])
        engine = SimulationEngine(
            algorithm=FirstArmAlgorithm(),
            data_loader=FakeLoader(impressions),
        )
        results = engine.run()
        assert results["total_clicks"] == 1

    def test_run_computes_click_through_rate(self) -> None:
        """results should include CTR = total_clicks / total_impressions."""
        impressions = _make_impressions([
            (["A", "B"], {"A": 1.0, "B": 0.0}),
            (["C", "D"], {"C": 0.0, "D": 1.0}),
            (["E", "F"], {"E": 1.0, "F": 0.0}),
            (["G", "H"], {"G": 0.0, "H": 1.0}),
        ])
        engine = SimulationEngine(
            algorithm=FirstArmAlgorithm(),
            data_loader=FakeLoader(impressions),
        )
        results = engine.run()
        # FirstArm picks A (1), C (0), E (1), G (0) -> 2 clicks / 4 = 0.5
        assert results["click_through_rate"] == pytest.approx(0.5)

    def test_run_returns_algorithm_name(self) -> None:
        """results should include the algorithm's name for identification."""
        engine = SimulationEngine(
            algorithm=FirstArmAlgorithm(),
            data_loader=FakeLoader(_make_impressions([
                (["A"], {"A": 0.0}),
            ])),
        )
        results = engine.run()
        assert results["algorithm"] == "FirstArmAlgorithm"

    def test_run_records_per_round_history(self) -> None:
        """results should include a list of per-round records."""
        impressions = _make_impressions([
            (["A", "B"], {"A": 1.0, "B": 0.0}),
            (["C", "D"], {"C": 0.0, "D": 1.0}),
        ])
        engine = SimulationEngine(
            algorithm=FirstArmAlgorithm(),
            data_loader=FakeLoader(impressions),
        )
        results = engine.run()
        history = results["history"]
        assert len(history) == 2
        assert history[0]["selected_arm"] == "A"
        assert history[0]["reward"] == 1.0
        assert history[1]["selected_arm"] == "C"
        assert history[1]["reward"] == 0.0


class TestAlgorithmInteraction:
    """Test that the engine correctly calls select_arm and update."""

    def test_algorithm_update_called_each_round(self) -> None:
        """update() should be called once per impression round."""
        algo = FirstArmAlgorithm()
        impressions = _make_impressions([
            (["A", "B"], {"A": 1.0, "B": 0.0}),
            (["C", "D"], {"C": 0.0, "D": 1.0}),
            (["E", "F"], {"E": 0.0, "F": 0.0}),
        ])
        engine = SimulationEngine(
            algorithm=algo, data_loader=FakeLoader(impressions)
        )
        engine.run()
        assert len(algo.update_calls) == 3

    def test_algorithm_receives_correct_rewards(self) -> None:
        """update() should receive the reward for the selected arm."""
        algo = FirstArmAlgorithm()
        impressions = _make_impressions([
            (["A", "B"], {"A": 1.0, "B": 0.0}),
            (["C", "D"], {"C": 0.0, "D": 1.0}),
        ])
        engine = SimulationEngine(
            algorithm=algo, data_loader=FakeLoader(impressions)
        )
        engine.run()
        assert algo.update_calls[0] == ("A", 1.0)
        assert algo.update_calls[1] == ("C", 0.0)


class TestEdgeCases:
    """Test edge cases in the simulation loop."""

    def test_empty_loader_returns_zero_metrics(self) -> None:
        """An empty data loader should produce zero impressions and CTR."""
        engine = SimulationEngine(
            algorithm=FirstArmAlgorithm(),
            data_loader=FakeLoader([]),
        )
        results = engine.run()
        assert results["total_impressions"] == 0
        assert results["total_clicks"] == 0
        assert results["click_through_rate"] == 0.0

    def test_single_candidate_impression(self) -> None:
        """Should handle impressions with only one candidate article."""
        impressions = _make_impressions([
            (["A"], {"A": 1.0}),
        ])
        engine = SimulationEngine(
            algorithm=FirstArmAlgorithm(),
            data_loader=FakeLoader(impressions),
        )
        results = engine.run()
        assert results["total_clicks"] == 1
        assert results["total_impressions"] == 1
