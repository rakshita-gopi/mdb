"""
Social media text preprocessing pipeline.

Cleaning is intentionally light so that sentiment-bearing tokens
(including emojis) are preserved for VADER analysis.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import CLEANED_CSV_PATH, RAW_CSV_PATH, ensure_directories  # noqa: E402
from src.logging_setup import get_logger  # noqa: E402

logger = get_logger(__name__)

# Compiled patterns used by the cleaning pipeline
URL_PATTERN = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
MENTION_PATTERN = re.compile(r"@\w+")
RETWEET_PATTERN = re.compile(r"\brt\b[:\s]*", re.IGNORECASE)
WHITESPACE_PATTERN = re.compile(r"\s+")


def handle_missing_text(text: object) -> str:
    """Replace missing or non-string values with an empty string."""
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return ""
    return str(text)


def remove_urls(text: str) -> str:
    """Remove HTTP(S) and www URLs."""
    return URL_PATTERN.sub(" ", text)


def remove_mentions(text: str) -> str:
    """Remove @username mentions."""
    return MENTION_PATTERN.sub(" ", text)


def remove_retweet_indicator(text: str) -> str:
    """Strip leading/embedded RT retweet markers."""
    return RETWEET_PATTERN.sub(" ", text)


def remove_special_characters(text: str) -> str:
    """
    Remove control characters and uncommon symbols.

    Emojis and common punctuation (!, ?, #, emoticons) are kept because
    they contribute to social media sentiment.
    """
    # Drop ASCII control characters but keep printable text and emojis
    cleaned_chars: list[str] = []
    for char in text:
        code = ord(char)
        if code < 32 or code == 127:
            cleaned_chars.append(" ")
            continue
        cleaned_chars.append(char)
    return "".join(cleaned_chars)


def normalize_whitespace(text: str) -> str:
    """Collapse repeated whitespace and trim ends."""
    return WHITESPACE_PATTERN.sub(" ", text).strip()


def to_lowercase(text: str) -> str:
    """Lowercase Latin letters. Emojis and non-cased characters are unchanged."""
    return text.lower()


def clean_text(text: object) -> str:
    """
    Apply the full cleaning pipeline to a single social media post.

    Order:
        missing → URLs → mentions → RT → special/control chars → lowercase → whitespace
    """
    cleaned = handle_missing_text(text)
    cleaned = remove_urls(cleaned)
    cleaned = remove_mentions(cleaned)
    cleaned = remove_retweet_indicator(cleaned)
    cleaned = remove_special_characters(cleaned)
    cleaned = to_lowercase(cleaned)
    cleaned = normalize_whitespace(cleaned)
    return cleaned


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform a raw dataset DataFrame into the processed schema.

    Output columns: post_id, original_text, cleaned_text, actual_sentiment, split
    """
    required = {"post_id", "text", "actual_sentiment", "split"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Raw dataset is missing required columns: {sorted(missing)}")

    processed = pd.DataFrame(
        {
            "post_id": df["post_id"],
            "original_text": df["text"].map(handle_missing_text),
            "cleaned_text": df["text"].map(clean_text),
            "actual_sentiment": df["actual_sentiment"],
            "split": df["split"],
        }
    )
    logger.info("Preprocessed %s records", len(processed))
    return processed


def save_cleaned_csv(df: pd.DataFrame, output_path: Path = CLEANED_CSV_PATH) -> Path:
    """Save the processed dataset to CSV."""
    ensure_directories()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8")
    logger.info("Saved cleaned dataset to %s", output_path)
    return output_path


def run(input_path: Path = RAW_CSV_PATH, output_path: Path = CLEANED_CSV_PATH) -> Path:
    """Load the raw CSV, preprocess it, and write the cleaned CSV."""
    if not input_path.exists():
        raise FileNotFoundError(
            f"Raw dataset not found at {input_path}. Run the download step first."
        )
    raw_df = pd.read_csv(input_path)
    logger.info("Loaded %s raw records from %s", len(raw_df), input_path)
    processed = preprocess_dataframe(raw_df)
    return save_cleaned_csv(processed, output_path)


if __name__ == "__main__":
    run()
