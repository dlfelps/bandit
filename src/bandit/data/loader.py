"""MIND dataset loader for the bandit simulation.

Parses the Microsoft News Dataset (MIND) files — behaviors.tsv and
news.tsv — into structured Python objects that the simulation engine
can iterate over sequentially.

MIND dataset schema:
    news.tsv columns (tab-separated):
        article_id, category, subcategory, title, abstract, url,
        title_entities, abstract_entities

    behaviors.tsv columns (tab-separated):
        impression_id, user_id, timestamp, click_history, impressions
        where impressions are space-separated "article_id-label" pairs
        (label: 1 = clicked, 0 = not clicked)
"""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Column names for the MIND dataset TSV files (no headers in the files)
_NEWS_COLUMNS = [
    "article_id",
    "category",
    "subcategory",
    "title",
    "abstract",
    "url",
    "title_entities",
    "abstract_entities",
]

_BEHAVIORS_COLUMNS = [
    "impression_id",
    "user_id",
    "timestamp",
    "click_history",
    "impressions",
]


class MINDDataLoader:
    """Load and parse MIND dataset files into simulation-ready rounds.

    Each impression round represents a single user visit where the
    recommender must choose one article from a set of candidates.
    The reward is determined by whether the user actually clicked
    the recommended article (1.0) or not (0.0).

    Attributes:
        news: DataFrame of article metadata (id, category, title, etc.).
        impressions: List of dicts, each with keys:
            - user_id (str): The user who generated this impression.
            - candidates (list[str]): Article IDs shown to the user.
            - rewards (dict[str, float]): Mapping of article_id -> click
              label (1.0 or 0.0).
    """

    def __init__(
        self,
        data_dir: str | Path,
        max_impressions: int | None = None,
    ) -> None:
        """Initialize the loader by reading news and behaviors files.

        Args:
            data_dir: Path to the directory containing news.tsv and
                behaviors.tsv.
            max_impressions: If set, only load the first N impression
                rows from behaviors.tsv.  Useful for sampling from
                large datasets without reading the entire file.

        Raises:
            FileNotFoundError: If the directory or required files do
                not exist.
        """
        data_dir = Path(data_dir)
        if not data_dir.is_dir():
            raise FileNotFoundError(f"Data directory not found: {data_dir}")

        news_path = data_dir / "news.tsv"
        behaviors_path = data_dir / "behaviors.tsv"

        if not news_path.exists():
            raise FileNotFoundError(f"News file not found: {news_path}")
        if not behaviors_path.exists():
            raise FileNotFoundError(
                f"Behaviors file not found: {behaviors_path}"
            )

        self._news = self._load_news(news_path)
        self._article_features = self._build_article_features()
        self._impressions = self._parse_behaviors(
            behaviors_path, max_impressions
        )

    @property
    def news(self) -> pd.DataFrame:
        """Return the news article metadata DataFrame."""
        return self._news

    @property
    def article_features(self) -> dict[str, np.ndarray]:
        """Return per-article one-hot category feature vectors."""
        return self._article_features

    @property
    def impressions(self) -> list[dict[str, Any]]:
        """Return the list of parsed impression rounds."""
        return self._impressions

    def _load_news(self, path: Path) -> pd.DataFrame:
        """Parse news.tsv into a DataFrame with named columns.

        Args:
            path: Path to the news.tsv file.

        Returns:
            DataFrame with columns: article_id, category, subcategory,
            title, abstract, url, title_entities, abstract_entities.
        """
        return pd.read_csv(
            path,
            sep="\t",
            header=None,
            names=_NEWS_COLUMNS,
        )

    def _build_article_features(self) -> dict[str, np.ndarray]:
        """Build one-hot subcategory feature vectors for each article.

        Uses the top-30 most frequent subcategories plus an "other"
        bucket, giving a 31-dimensional feature vector per article.
        Subcategories are far more granular than categories (285 vs
        18 values), capturing distinctions like football_nfl vs
        basketball_nba rather than just "sports".

        Returns:
            Dict mapping article_id to a numpy array of shape
            (num_subcategories,) with a 1.0 at the subcategory index.
        """
        top_n = 30
        subcat_counts = self._news["subcategory"].value_counts()
        top_subcats = list(subcat_counts.head(top_n).index)
        self._feature_labels = top_subcats + ["_other"]
        self._subcat_to_idx = {
            sc: i for i, sc in enumerate(top_subcats)
        }
        self._other_idx = top_n
        d = top_n + 1

        features: dict[str, np.ndarray] = {}
        for _, row in self._news.iterrows():
            vec = np.zeros(d)
            subcat = row["subcategory"]
            idx = self._subcat_to_idx.get(subcat, self._other_idx)
            vec[idx] = 1.0
            features[row["article_id"]] = vec
        return features

    def _build_user_profile(
        self, click_history: str
    ) -> np.ndarray:
        """Build a user profile vector from their click history.

        Computes a normalised subcategory-frequency vector over the
        articles the user has previously clicked.  If the history is
        empty or contains no known articles the result is a zero
        vector.

        Args:
            click_history: Space-separated article IDs the user
                clicked before this impression.

        Returns:
            Array of shape (num_subcategories,) with values in [0, 1].
        """
        d = len(self._feature_labels)
        profile = np.zeros(d)

        if not isinstance(click_history, str) or not click_history.strip():
            return profile

        for aid in click_history.split():
            if aid in self._article_features:
                profile += self._article_features[aid]

        total = profile.sum()
        if total > 0:
            profile /= total
        return profile

    def _parse_behaviors(
        self,
        path: Path,
        max_impressions: int | None = None,
    ) -> list[dict[str, Any]]:
        """Parse behaviors.tsv into a list of impression round dicts.

        Each impression string like "N001-1 N002-0 N003-0" is split
        into candidate article IDs and their corresponding click labels.

        Args:
            path: Path to the behaviors.tsv file.
            max_impressions: If set, only read the first N rows.

        Returns:
            List of dicts with keys: user_id, candidates, rewards.
        """
        behaviors_df = pd.read_csv(
            path,
            sep="\t",
            header=None,
            names=_BEHAVIORS_COLUMNS,
            nrows=max_impressions,
        )

        rounds: list[dict[str, Any]] = []
        for _, row in behaviors_df.iterrows():
            impression_pairs = str(row["impressions"]).split()
            candidates: list[str] = []
            rewards: dict[str, float] = {}

            for pair in impression_pairs:
                article_id, label = pair.rsplit("-", maxsplit=1)
                candidates.append(article_id)
                rewards[article_id] = float(label)

            user_profile = self._build_user_profile(
                row["click_history"]
            )
            contexts = {}
            for aid in candidates:
                if aid in self._article_features:
                    article_vec = self._article_features[aid]
                    contexts[aid] = np.concatenate(
                        [user_profile, article_vec]
                    )

            rounds.append(
                {
                    "user_id": str(row["user_id"]),
                    "candidates": candidates,
                    "rewards": rewards,
                    "contexts": contexts,
                }
            )

        return rounds

    def __iter__(self):
        """Iterate over impression rounds sequentially."""
        return iter(self._impressions)

    def __len__(self) -> int:
        """Return the number of impression rounds."""
        return len(self._impressions)
