"""
Explore the raw social media dataset and persist summary statistics.

Analyses:
    - total records
    - sentiment counts (positive / negative / neutral)
    - missing values and duplicate records
    - text length statistics
    - sentiment and split distributions
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import EXPLORATION_RESULTS_JSON, RAW_CSV_PATH, ensure_directories  # noqa: E402
from src.logging_setup import get_logger  # noqa: E402

logger = get_logger(__name__)


def load_raw_dataset(path: Path = RAW_CSV_PATH) -> pd.DataFrame:
    """Load the raw CSV produced by the download step."""
    if not path.exists():
        raise FileNotFoundError(
            f"Raw dataset not found at {path}. Run the download step first."
        )
    df = pd.read_csv(path)
    logger.info("Loaded raw dataset from %s (%s rows)", path, len(df))
    return df


def explore_dataset(df: pd.DataFrame) -> dict[str, Any]:
    """Compute exploration statistics from the raw dataset."""
    text_lengths = df["text"].fillna("").astype(str).str.len()

    stats: dict[str, Any] = {
        "total_records": int(len(df)),
        "positive_posts": int((df["actual_sentiment"] == "Positive").sum()),
        "negative_posts": int((df["actual_sentiment"] == "Negative").sum()),
        "neutral_posts": int((df["actual_sentiment"] == "Neutral").sum()),
        "missing_values": {col: int(df[col].isna().sum()) for col in df.columns},
        "duplicate_records": int(df.duplicated().sum()),
        "duplicate_texts": int(df["text"].duplicated().sum()),
        "average_text_length": float(text_lengths.mean()) if len(df) else 0.0,
        "minimum_text_length": int(text_lengths.min()) if len(df) else 0,
        "maximum_text_length": int(text_lengths.max()) if len(df) else 0,
        "sentiment_distribution": df["actual_sentiment"].value_counts().to_dict(),
        "split_distribution": df["split"].value_counts().to_dict(),
        "columns": list(df.columns),
    }
    return stats


def save_exploration_results(stats: dict[str, Any], path: Path = EXPLORATION_RESULTS_JSON) -> Path:
    """Write exploration statistics to JSON."""
    ensure_directories()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(stats, handle, indent=2, default=str)
    logger.info("Saved exploration results to %s", path)
    return path


def print_summary(stats: dict[str, Any]) -> None:
    """Print a human-readable summary of exploration statistics."""
    logger.info("=== Dataset Exploration Summary ===")
    logger.info("Total records: %s", stats["total_records"])
    logger.info("Positive: %s | Negative: %s | Neutral: %s",
                stats["positive_posts"], stats["negative_posts"], stats["neutral_posts"])
    logger.info("Missing values: %s", stats["missing_values"])
    logger.info("Duplicate rows: %s | Duplicate texts: %s",
                stats["duplicate_records"], stats["duplicate_texts"])
    logger.info("Text length — avg: %.2f, min: %s, max: %s",
                stats["average_text_length"],
                stats["minimum_text_length"],
                stats["maximum_text_length"])
    logger.info("Sentiment distribution: %s", stats["sentiment_distribution"])
    logger.info("Split distribution: %s", stats["split_distribution"])


def run() -> dict[str, Any]:
    """Run exploration on the raw CSV and persist results."""
    df = load_raw_dataset()
    stats = explore_dataset(df)
    print_summary(stats)
    save_exploration_results(stats)
    return stats


if __name__ == "__main__":
    run()
