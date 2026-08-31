"""
VADER sentiment analysis engine.

Reads unprocessed social media posts from MongoDB, classifies them using
the VADER compound score, and writes predictions back in batches.

The actual_sentiment field is never used as an input to prediction.
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from pymongo import UpdateOne
from pymongo.database import Database

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (  # noqa: E402
    BATCH_SIZE,
    COLLECTION_SOCIAL_POSTS,
    SENTIMENT_RESULTS_CSV,
    VADER_NEGATIVE_THRESHOLD,
    VADER_POSITIVE_THRESHOLD,
    ensure_directories,
)
from src.database.mongodb_connection import get_database, upsert_metadata  # noqa: E402
from src.logging_setup import get_logger  # noqa: E402

logger = get_logger(__name__)


def ensure_vader_lexicon() -> None:
    """Download the VADER lexicon if it is not already available."""
    try:
        SentimentIntensityAnalyzer()
    except LookupError:
        logger.info("Downloading NLTK vader_lexicon...")
        import nltk

        nltk.download("vader_lexicon", quiet=True)


def classify_compound(compound: float) -> str:
    """
    Map a VADER compound score to Positive, Negative, or Neutral.

    compound >= 0.05  → Positive
    compound <= -0.05 → Negative
    otherwise         → Neutral
    """
    if compound >= VADER_POSITIVE_THRESHOLD:
        return "Positive"
    if compound <= VADER_NEGATIVE_THRESHOLD:
        return "Negative"
    return "Neutral"


def analyse_text(text: str, analyzer: SentimentIntensityAnalyzer | None = None) -> dict[str, Any]:
    """
    Analyse a single piece of text and return label plus scores.

    Uses cleaned_text (or any caller-supplied string). Does not consult
    actual sentiment labels.
    """
    if analyzer is None:
        ensure_vader_lexicon()
        analyzer = SentimentIntensityAnalyzer()

    scores = analyzer.polarity_scores(text or "")
    compound = float(scores["compound"])
    return {
        "predicted_sentiment": classify_compound(compound),
        "sentiment_score": compound,
        "positive_score": float(scores["pos"]),
        "negative_score": float(scores["neg"]),
        "neutral_score": float(scores["neu"]),
    }


def _iter_unprocessed_posts(
    database: Database,
    batch_size: int,
    force: bool,
) -> Iterable[list[dict[str, Any]]]:
    """Yield batches of posts that still need sentiment analysis."""
    collection = database[COLLECTION_SOCIAL_POSTS]
    query: dict[str, Any] = {} if force else {"processed": False}
    cursor = collection.find(query, no_cursor_timeout=False).batch_size(batch_size)

    batch: list[dict[str, Any]] = []
    for document in cursor:
        batch.append(document)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def process_posts(
    database: Database | None = None,
    batch_size: int = BATCH_SIZE,
    force: bool = False,
) -> dict[str, int]:
    """
    Run VADER on MongoDB posts and store predicted_sentiment + sentiment_score.

    Args:
        force: If True, re-analyse posts that are already marked processed.
    """
    ensure_vader_lexicon()
    analyzer = SentimentIntensityAnalyzer()
    db = database if database is not None else get_database()
    collection = db[COLLECTION_SOCIAL_POSTS]
    now = datetime.now(timezone.utc)

    processed_count = 0
    for batch in _iter_unprocessed_posts(db, batch_size, force=force):
        operations = []
        for post in batch:
            result = analyse_text(post.get("cleaned_text") or "", analyzer=analyzer)
            operations.append(
                UpdateOne(
                    {"_id": post["_id"]},
                    {
                        "$set": {
                            "predicted_sentiment": result["predicted_sentiment"],
                            "sentiment_score": result["sentiment_score"],
                            "processed": True,
                            "updated_at": now,
                        }
                    },
                )
            )
        if operations:
            collection.bulk_write(operations, ordered=False)
            processed_count += len(operations)
            logger.info("Analysed %s posts so far...", processed_count)

    stats = {"processed": processed_count}
    logger.info("Sentiment analysis complete: %s", stats)
    upsert_metadata(
        "last_sentiment_stats",
        {**stats, "completed_at": datetime.now(timezone.utc).isoformat()},
        database=db,
    )
    return stats


def export_sentiment_results(database: Database | None = None, path: Path = SENTIMENT_RESULTS_CSV) -> Path:
    """Export processed posts to results/sentiment_results.csv."""
    db = database if database is not None else get_database()
    posts = list(
        db[COLLECTION_SOCIAL_POSTS].find(
            {"processed": True},
            {
                "_id": 0,
                "post_id": 1,
                "cleaned_text": 1,
                "actual_sentiment": 1,
                "predicted_sentiment": 1,
                "sentiment_score": 1,
                "dataset_split": 1,
            },
        )
    )
    ensure_directories()
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(posts).to_csv(path, index=False, encoding="utf-8")
    logger.info("Wrote %s sentiment rows to %s", len(posts), path)
    return path


def run(force: bool = False, database: Database | None = None) -> dict[str, int]:
    """Run sentiment analysis and export a CSV of results."""
    stats = process_posts(database=database, force=force)
    export_sentiment_results(database=database)
    return stats


if __name__ == "__main__":
    run()
