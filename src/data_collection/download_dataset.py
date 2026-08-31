"""
Download the TweetEval sentiment dataset and save it as a raw CSV.

Dataset: Hugging Face `tweet_eval` / configuration `sentiment`
Labels: 0 = Negative, 1 = Neutral, 2 = Positive

Output columns:
    post_id, text, actual_sentiment, split
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from datasets import load_dataset

# Allow running this file directly: python src/data_collection/download_dataset.py
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (  # noqa: E402
    DATASET_CONFIG,
    DATASET_NAME,
    LABEL_MAP,
    RAW_CSV_PATH,
    ensure_directories,
)
from src.logging_setup import get_logger  # noqa: E402

logger = get_logger(__name__)


def download_tweeteval_sentiment() -> pd.DataFrame:
    """
    Download all TweetEval sentiment splits and return a combined DataFrame.

    The actual sentiment label is preserved for later evaluation but is never
    used as an input to the prediction pipeline.
    """
    logger.info("Downloading dataset '%s' (config='%s') from Hugging Face...", DATASET_NAME, DATASET_CONFIG)
    dataset = load_dataset(DATASET_NAME, DATASET_CONFIG)

    frames: list[pd.DataFrame] = []
    post_id = 1

    for split_name in ("train", "validation", "test"):
        if split_name not in dataset:
            logger.warning("Split '%s' is not available; skipping.", split_name)
            continue

        split = dataset[split_name]
        records = []
        for row in split:
            records.append(
                {
                    "post_id": post_id,
                    "text": row["text"],
                    "actual_sentiment": LABEL_MAP[int(row["label"])],
                    "split": split_name,
                }
            )
            post_id += 1

        split_df = pd.DataFrame(records)
        logger.info("Loaded %s split: %s records", split_name, len(split_df))
        frames.append(split_df)

    if not frames:
        raise RuntimeError("No TweetEval sentiment splits were loaded.")

    combined = pd.concat(frames, ignore_index=True)
    logger.info("Combined dataset size: %s records", len(combined))
    return combined


def save_raw_csv(df: pd.DataFrame, output_path: Path = RAW_CSV_PATH) -> Path:
    """Save the raw dataset to CSV."""
    ensure_directories()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8")
    logger.info("Saved raw dataset to %s", output_path)
    return output_path


def run() -> Path:
    """Download the dataset and write the raw CSV. Returns the output path."""
    df = download_tweeteval_sentiment()
    return save_raw_csv(df)


if __name__ == "__main__":
    run()
