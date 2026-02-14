"""Generate a synthetic MIND-format dataset for development and testing.

The MIND dataset consists of two key TSV files:
- behaviors.tsv: User impression logs with click/no-click labels.
- news.tsv: News article metadata (ID, category, title, etc.).

This script generates data that mirrors the real MIND schema so the
pipeline can be developed and tested without downloading the full dataset.
"""

import os
import random

# Seed for reproducibility
random.seed(42)

OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data",
    "MINDsmall_dev",
)

NUM_NEWS_ARTICLES = 200
NUM_USERS = 50
NUM_IMPRESSIONS = 500
CATEGORIES = [
    "news", "sports", "entertainment", "finance", "lifestyle",
    "health", "autos", "travel", "foodanddrink", "weather",
]
SUBCATEGORIES = {
    "news": ["usanews", "worldnews", "politicsandelections"],
    "sports": ["football", "basketball", "baseball", "soccer"],
    "entertainment": ["movies", "tv", "music", "celebrity"],
    "finance": ["markets", "personalfinance", "realestate"],
    "lifestyle": ["fashion", "wellness", "relationships"],
    "health": ["medical", "nutrition", "mentalhealth"],
    "autos": ["autoreviews", "autonews", "electric"],
    "travel": ["destinations", "tips", "airlines"],
    "foodanddrink": ["recipes", "restaurants", "wine"],
    "weather": ["forecast", "severeweather", "climate"],
}
TITLE_WORDS = [
    "breaking", "new", "top", "big", "latest", "update", "report",
    "analysis", "review", "guide", "how", "why", "what", "best",
    "worst", "first", "last", "major", "key", "critical",
]
NOUNS = [
    "market", "team", "player", "movie", "stock", "deal", "policy",
    "study", "election", "company", "event", "game", "show", "trend",
    "discovery", "breakthrough", "crisis", "reform", "debate", "plan",
]


def generate_title() -> str:
    """Generate a random news-like title."""
    num_words = random.randint(4, 8)
    words = [random.choice(TITLE_WORDS + NOUNS) for _ in range(num_words)]
    return " ".join(words).capitalize()


def generate_news_tsv(filepath: str) -> list[str]:
    """Generate news.tsv with article metadata.

    MIND news.tsv format (tab-separated):
    NewsID  Category  SubCategory  Title  Abstract  URL  TitleEntities  AbstractEntities

    Returns:
        List of news article IDs.
    """
    news_ids = []
    with open(filepath, "w") as f:
        for i in range(NUM_NEWS_ARTICLES):
            news_id = f"N{i + 1:05d}"
            news_ids.append(news_id)
            category = random.choice(CATEGORIES)
            subcategory = random.choice(SUBCATEGORIES[category])
            title = generate_title()
            abstract = f"Abstract for article {news_id}."
            url = f"https://example.com/{news_id}"
            # Simplified entities (empty for synthetic data)
            title_entities = "[]"
            abstract_entities = "[]"
            row = "\t".join([
                news_id, category, subcategory, title, abstract,
                url, title_entities, abstract_entities,
            ])
            f.write(row + "\n")
    return news_ids


def generate_behaviors_tsv(
    filepath: str, news_ids: list[str]
) -> None:
    """Generate behaviors.tsv with user impression logs.

    MIND behaviors.tsv format (tab-separated):
    ImpressionID  UserID  Time  History  Impressions

    Impressions column format: "NewsID-Click NewsID-Click ..."
    where Click is 1 (clicked) or 0 (not clicked).
    """
    with open(filepath, "w") as f:
        for imp_idx in range(NUM_IMPRESSIONS):
            impression_id = imp_idx + 1
            user_id = f"U{random.randint(1, NUM_USERS):05d}"
            time_str = f"11/15/2019 {random.randint(0, 23)}:{random.randint(0, 59):02d}:00 AM"

            # User click history (random subset of articles)
            history_size = random.randint(3, 15)
            history = random.sample(news_ids, min(history_size, len(news_ids)))
            history_str = " ".join(history)

            # Impression candidates (3-10 articles shown to user)
            num_candidates = random.randint(3, 10)
            candidates = random.sample(news_ids, num_candidates)

            # Exactly one article is clicked per impression
            clicked_idx = random.randint(0, num_candidates - 1)
            impressions = []
            for j, article_id in enumerate(candidates):
                click_label = 1 if j == clicked_idx else 0
                impressions.append(f"{article_id}-{click_label}")
            impressions_str = " ".join(impressions)

            row = "\t".join([
                str(impression_id),
                user_id,
                time_str,
                history_str,
                impressions_str,
            ])
            f.write(row + "\n")


def main() -> None:
    """Generate the synthetic MIND dataset."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    news_path = os.path.join(OUTPUT_DIR, "news.tsv")
    behaviors_path = os.path.join(OUTPUT_DIR, "behaviors.tsv")

    print(f"Generating synthetic MIND dataset in: {OUTPUT_DIR}")
    news_ids = generate_news_tsv(news_path)
    print(f"  Created {news_path} with {NUM_NEWS_ARTICLES} articles")

    generate_behaviors_tsv(behaviors_path, news_ids)
    print(f"  Created {behaviors_path} with {NUM_IMPRESSIONS} impressions")
    print("Done.")


if __name__ == "__main__":
    main()
