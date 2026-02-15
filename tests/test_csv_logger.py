"""Tests for CSV metric logging.

Verifies that simulation results can be saved to CSV files in a
structured format suitable for post-experiment analysis.
"""

import pandas as pd
import pytest

from bandit.metrics.csv_logger import save_results


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_result(
    algorithm_name: str = "TestAlgo",
    num_rounds: int = 3,
) -> dict:
    """Build a fake simulation result dict for testing."""
    history = [
        {
            "round": i + 1,
            "user_id": f"U{i:03d}",
            "selected_arm": f"N{i:03d}",
            "reward": float(i % 2),
        }
        for i in range(num_rounds)
    ]
    total_clicks = sum(int(h["reward"]) for h in history)
    return {
        "algorithm": algorithm_name,
        "total_impressions": num_rounds,
        "total_clicks": total_clicks,
        "click_through_rate": (
            total_clicks / num_rounds if num_rounds > 0 else 0.0
        ),
        "history": history,
    }


# ---------------------------------------------------------------------------
# Tests: per-algorithm history CSV
# ---------------------------------------------------------------------------


class TestSaveHistoryCSV:
    """Test per-round history CSV output."""

    def test_creates_history_csv_file(self, tmp_path) -> None:
        """save_results should create a history CSV for each algo."""
        results = [_make_result("RandomChoice")]
        save_results(results, tmp_path)
        assert (tmp_path / "history_RandomChoice.csv").exists()

    def test_history_csv_has_expected_columns(
        self, tmp_path
    ) -> None:
        """History CSV should have the four per-round columns."""
        results = [_make_result("TestAlgo")]
        save_results(results, tmp_path)
        df = pd.read_csv(tmp_path / "history_TestAlgo.csv")
        assert list(df.columns) == [
            "round",
            "user_id",
            "selected_arm",
            "reward",
        ]

    def test_history_csv_row_count_matches(
        self, tmp_path
    ) -> None:
        """Number of CSV rows should match history length."""
        results = [_make_result("TestAlgo", num_rounds=5)]
        save_results(results, tmp_path)
        df = pd.read_csv(tmp_path / "history_TestAlgo.csv")
        assert len(df) == 5

    def test_history_csv_values_are_correct(
        self, tmp_path
    ) -> None:
        """CSV values should match the input history data."""
        results = [_make_result("TestAlgo", num_rounds=3)]
        save_results(results, tmp_path)
        df = pd.read_csv(tmp_path / "history_TestAlgo.csv")
        # Round 0: reward 0.0, Round 1: reward 1.0, Round 2: 0.0
        assert df.iloc[0]["round"] == 1
        assert df.iloc[0]["user_id"] == "U000"
        assert df.iloc[0]["selected_arm"] == "N000"
        assert df.iloc[0]["reward"] == 0.0
        assert df.iloc[1]["reward"] == 1.0

    def test_creates_separate_history_per_algorithm(
        self, tmp_path
    ) -> None:
        """Each algorithm should get its own history CSV."""
        results = [_make_result("AlgoA"), _make_result("AlgoB")]
        save_results(results, tmp_path)
        assert (tmp_path / "history_AlgoA.csv").exists()
        assert (tmp_path / "history_AlgoB.csv").exists()


# ---------------------------------------------------------------------------
# Tests: aggregate summary CSV
# ---------------------------------------------------------------------------


class TestSaveSummaryCSV:
    """Test aggregate summary CSV output."""

    def test_creates_summary_csv(self, tmp_path) -> None:
        """save_results should create a summary.csv file."""
        results = [_make_result("A"), _make_result("B")]
        save_results(results, tmp_path)
        assert (tmp_path / "summary.csv").exists()

    def test_summary_has_expected_columns(
        self, tmp_path
    ) -> None:
        """Summary CSV should have the four aggregate columns."""
        results = [_make_result("A")]
        save_results(results, tmp_path)
        df = pd.read_csv(tmp_path / "summary.csv")
        assert list(df.columns) == [
            "algorithm",
            "total_impressions",
            "total_clicks",
            "click_through_rate",
        ]

    def test_summary_has_one_row_per_algorithm(
        self, tmp_path
    ) -> None:
        """Summary CSV should have exactly one row per algorithm."""
        results = [_make_result("A"), _make_result("B")]
        save_results(results, tmp_path)
        df = pd.read_csv(tmp_path / "summary.csv")
        assert len(df) == 2

    def test_summary_values_are_correct(
        self, tmp_path
    ) -> None:
        """Summary values should match the input result metrics."""
        result = _make_result("TestAlgo", num_rounds=4)
        save_results([result], tmp_path)
        df = pd.read_csv(tmp_path / "summary.csv")
        row = df.iloc[0]
        assert row["algorithm"] == "TestAlgo"
        assert row["total_impressions"] == 4
        assert row["total_clicks"] == result["total_clicks"]
        assert row["click_through_rate"] == pytest.approx(
            result["click_through_rate"]
        )


# ---------------------------------------------------------------------------
# Tests: edge cases
# ---------------------------------------------------------------------------


class TestSaveResultsEdgeCases:
    """Test edge cases for save_results."""

    def test_creates_output_directory_if_missing(
        self, tmp_path
    ) -> None:
        """Should create the output directory when it doesn't exist."""
        output_dir = tmp_path / "nested" / "output"
        results = [_make_result("A")]
        save_results(results, output_dir)
        assert (output_dir / "summary.csv").exists()
