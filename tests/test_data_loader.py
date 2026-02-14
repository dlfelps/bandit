"""Tests for the MINDDataLoader.

Verifies that the data loader correctly parses MIND dataset files
(behaviors.tsv and news.tsv) and produces structured impression rounds
suitable for the simulation engine.
"""

from pathlib import Path

import pandas as pd
import pytest

from bandit.data.loader import MINDDataLoader


@pytest.fixture
def data_dir() -> Path:
    """Return the path to the MIND small dev dataset."""
    return Path(__file__).resolve().parent.parent / "data" / "MINDsmall_dev"


@pytest.fixture
def loader(data_dir: Path) -> MINDDataLoader:
    """Create a MINDDataLoader pointed at the small dev dataset."""
    return MINDDataLoader(data_dir)


class TestMINDDataLoaderInit:
    """Test MINDDataLoader initialization and file loading."""

    def test_raises_if_directory_missing(self, tmp_path: Path) -> None:
        """Should raise FileNotFoundError for a nonexistent directory."""
        with pytest.raises(FileNotFoundError):
            MINDDataLoader(tmp_path / "nonexistent")

    def test_raises_if_behaviors_file_missing(self, tmp_path: Path) -> None:
        """Should raise FileNotFoundError when behaviors.tsv is absent."""
        (tmp_path / "news.tsv").write_text("N001\tsports\tfootball\tTitle\tAbstract\turl\t[]\t[]\n")
        with pytest.raises(FileNotFoundError):
            MINDDataLoader(tmp_path)

    def test_raises_if_news_file_missing(self, tmp_path: Path) -> None:
        """Should raise FileNotFoundError when news.tsv is absent."""
        (tmp_path / "behaviors.tsv").write_text("1\tU001\t11/15/2019\tN001\tN002-1 N003-0\n")
        with pytest.raises(FileNotFoundError):
            MINDDataLoader(tmp_path)

    def test_loads_from_valid_directory(self, loader: MINDDataLoader) -> None:
        """Should initialize without error from the dev dataset."""
        assert loader is not None


class TestNewsMetadata:
    """Test that news metadata is parsed correctly."""

    def test_news_returns_dataframe(self, loader: MINDDataLoader) -> None:
        """news property should return a pandas DataFrame."""
        assert isinstance(loader.news, pd.DataFrame)

    def test_news_has_expected_columns(self, loader: MINDDataLoader) -> None:
        """news DataFrame should have article_id, category, subcategory, title columns."""
        for col in ["article_id", "category", "subcategory", "title"]:
            assert col in loader.news.columns

    def test_news_has_rows(self, loader: MINDDataLoader) -> None:
        """news DataFrame should not be empty."""
        assert len(loader.news) > 0

    def test_news_article_ids_are_unique(self, loader: MINDDataLoader) -> None:
        """Each article should appear exactly once in the news table."""
        assert loader.news["article_id"].is_unique


class TestImpressions:
    """Test that impression logs are parsed into structured rounds."""

    def test_impressions_returns_list(self, loader: MINDDataLoader) -> None:
        """impressions property should return a list."""
        assert isinstance(loader.impressions, list)

    def test_impressions_not_empty(self, loader: MINDDataLoader) -> None:
        """Should contain at least one impression round."""
        assert len(loader.impressions) > 0

    def test_impression_has_required_keys(self, loader: MINDDataLoader) -> None:
        """Each impression dict should have user_id, candidates, and rewards."""
        impression = loader.impressions[0]
        assert "user_id" in impression
        assert "candidates" in impression
        assert "rewards" in impression

    def test_candidates_is_list_of_strings(self, loader: MINDDataLoader) -> None:
        """candidates should be a list of article ID strings."""
        impression = loader.impressions[0]
        assert isinstance(impression["candidates"], list)
        assert all(isinstance(c, str) for c in impression["candidates"])

    def test_rewards_maps_candidates_to_binary(self, loader: MINDDataLoader) -> None:
        """rewards should map each candidate to 0.0 or 1.0."""
        impression = loader.impressions[0]
        rewards = impression["rewards"]
        assert isinstance(rewards, dict)
        for candidate in impression["candidates"]:
            assert candidate in rewards
            assert rewards[candidate] in (0.0, 1.0)

    def test_each_impression_has_at_least_one_candidate(
        self, loader: MINDDataLoader
    ) -> None:
        """Every impression should have at least one candidate article."""
        for impression in loader.impressions:
            assert len(impression["candidates"]) >= 1

    def test_impression_count_matches_behaviors(
        self, loader: MINDDataLoader, data_dir: Path
    ) -> None:
        """Number of impressions should match lines in behaviors.tsv."""
        line_count = sum(1 for _ in (data_dir / "behaviors.tsv").open())
        assert len(loader.impressions) == line_count


class TestIteration:
    """Test that the loader supports sequential iteration over rounds."""

    def test_loader_is_iterable(self, loader: MINDDataLoader) -> None:
        """MINDDataLoader should support iteration."""
        iterator = iter(loader)
        first = next(iterator)
        assert "user_id" in first

    def test_iteration_yields_all_impressions(self, loader: MINDDataLoader) -> None:
        """Iterating should yield the same number as len(impressions)."""
        count = sum(1 for _ in loader)
        assert count == len(loader.impressions)

    def test_len_returns_impression_count(self, loader: MINDDataLoader) -> None:
        """len() should return the number of impressions."""
        assert len(loader) == len(loader.impressions)
