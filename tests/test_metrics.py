"""Tests for metric logging utilities.

Verifies that simulation results are correctly converted to DataFrames
and persisted as CSV files.
"""

import pandas as pd
import pytest

from bandit.utils.metrics import results_to_dataframe, save_results_csv


def _make_results(
    algorithm: str,
    rewards: list[float],
) -> dict:
    """Build a minimal results dict for testing."""
    history = [
        {
            "round": i + 1,
            "user_id": f"U{i:03d}",
            "selected_arm": f"N{i:03d}",
            "reward": r,
        }
        for i, r in enumerate(rewards)
    ]
    total_clicks = int(sum(rewards))
    total_impressions = len(rewards)
    return {
        "algorithm": algorithm,
        "total_impressions": total_impressions,
        "total_clicks": total_clicks,
        "click_through_rate": (
            total_clicks / total_impressions if total_impressions else 0.0
        ),
        "history": history,
    }


class TestResultsToDataframe:
    """Test results_to_dataframe conversion."""

    def test_returns_dataframe(self) -> None:
        """Should return a pandas DataFrame."""
        results = _make_results("Test", [1.0, 0.0])
        df = results_to_dataframe(results)
        assert isinstance(df, pd.DataFrame)

    def test_has_expected_columns(self) -> None:
        """DataFrame should have the correct column names."""
        results = _make_results("Test", [1.0, 0.0])
        df = results_to_dataframe(results)
        expected = {
            "algorithm",
            "round",
            "user_id",
            "selected_arm",
            "reward",
            "cumulative_ctr",
        }
        assert set(df.columns) == expected

    def test_row_count_matches_history(self) -> None:
        """One row per impression round."""
        results = _make_results("Test", [1.0, 0.0, 1.0])
        df = results_to_dataframe(results)
        assert len(df) == 3

    def test_cumulative_ctr_computed_correctly(self) -> None:
        """cumulative_ctr should reflect running clicks / rounds."""
        results = _make_results("Test", [1.0, 0.0, 1.0, 0.0])
        df = results_to_dataframe(results)
        # Round 1: 1/1=1.0, Round 2: 1/2=0.5, Round 3: 2/3≈0.667,
        # Round 4: 2/4=0.5
        expected = [1.0, 0.5, 2.0 / 3.0, 0.5]
        assert df["cumulative_ctr"].tolist() == pytest.approx(expected)

    def test_algorithm_name_propagated(self) -> None:
        """Every row should carry the algorithm name."""
        results = _make_results("RandomChoice", [1.0, 0.0])
        df = results_to_dataframe(results)
        assert (df["algorithm"] == "RandomChoice").all()


class TestSaveResultsCsv:
    """Test CSV persistence."""

    def test_creates_csv_file(self, tmp_path) -> None:
        """Should create a CSV file at the given path."""
        df = results_to_dataframe(_make_results("Test", [1.0]))
        out = save_results_csv([df], tmp_path / "results.csv")
        assert out.exists()

    def test_csv_is_readable(self, tmp_path) -> None:
        """The CSV should be re-loadable by pandas."""
        df = results_to_dataframe(_make_results("Test", [1.0, 0.0]))
        out = save_results_csv([df], tmp_path / "results.csv")
        loaded = pd.read_csv(out)
        assert len(loaded) == 2

    def test_combines_multiple_algorithms(self, tmp_path) -> None:
        """Multiple DataFrames should be concatenated in one CSV."""
        df_a = results_to_dataframe(_make_results("AlgoA", [1.0, 0.0]))
        df_b = results_to_dataframe(_make_results("AlgoB", [0.0, 1.0]))
        out = save_results_csv([df_a, df_b], tmp_path / "results.csv")
        loaded = pd.read_csv(out)
        assert len(loaded) == 4
        assert set(loaded["algorithm"]) == {"AlgoA", "AlgoB"}

    def test_creates_parent_directories(self, tmp_path) -> None:
        """Should create intermediate directories if they don't exist."""
        df = results_to_dataframe(_make_results("Test", [1.0]))
        out = save_results_csv(
            [df], tmp_path / "sub" / "dir" / "results.csv"
        )
        assert out.exists()
