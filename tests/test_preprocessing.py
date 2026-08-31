"""Tests for the social media text cleaning pipeline."""

from src.preprocessing.preprocess import (
    clean_text,
    handle_missing_text,
    normalize_whitespace,
    preprocess_dataframe,
    remove_mentions,
    remove_retweet_indicator,
    remove_urls,
    to_lowercase,
)
import pandas as pd


def test_handle_missing_text() -> None:
    assert handle_missing_text(None) == ""
    assert handle_missing_text(float("nan")) == ""
    assert handle_missing_text("hello") == "hello"


def test_remove_urls() -> None:
    text = "Check https://example.com/path now and www.test.org too"
    cleaned = remove_urls(text)
    assert "https://" not in cleaned
    assert "www.test.org" not in cleaned


def test_remove_mentions() -> None:
    assert "@alice" not in remove_mentions("hello @alice how are you")


def test_remove_retweet_indicator() -> None:
    cleaned = remove_retweet_indicator("RT @user: great news")
    assert not cleaned.lower().startswith("rt ")


def test_normalize_whitespace() -> None:
    assert normalize_whitespace("  too   many   spaces  ") == "too many spaces"


def test_to_lowercase() -> None:
    assert to_lowercase("I LOVE This") == "i love this"


def test_clean_text_full_pipeline() -> None:
    raw = "RT @shop: I LOVE this product! https://shop.example/x  :)"
    cleaned = clean_text(raw)
    assert "http" not in cleaned
    assert "@shop" not in cleaned
    assert "love" in cleaned
    assert cleaned == cleaned.lower()


def test_clean_text_keeps_emoji() -> None:
    cleaned = clean_text("This is awesome 😍")
    assert "😍" in cleaned


def test_preprocess_dataframe_schema() -> None:
    df = pd.DataFrame(
        {
            "post_id": [1],
            "text": ["I love this!"],
            "actual_sentiment": ["Positive"],
            "split": ["train"],
        }
    )
    out = preprocess_dataframe(df)
    assert list(out.columns) == [
        "post_id",
        "original_text",
        "cleaned_text",
        "actual_sentiment",
        "split",
    ]
    assert out.loc[0, "original_text"] == "I love this!"
    assert "love" in out.loc[0, "cleaned_text"]
