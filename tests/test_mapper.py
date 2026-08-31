"""Tests for the Map phase."""

from src.mapreduce.mapper import map_post, map_posts


def test_map_post_positive() -> None:
    assert map_post({"predicted_sentiment": "Positive"}) == ("Positive", 1)


def test_map_post_negative() -> None:
    assert map_post({"predicted_sentiment": "Negative"}) == ("Negative", 1)


def test_map_post_missing_returns_none() -> None:
    assert map_post({"predicted_sentiment": None}) is None
    assert map_post({}) is None


def test_map_posts_list() -> None:
    posts = [
        {"predicted_sentiment": "Positive"},
        {"predicted_sentiment": "Positive"},
        {"predicted_sentiment": "Negative"},
        {"predicted_sentiment": "Neutral"},
        {"predicted_sentiment": None},
    ]
    pairs = map_posts(posts)
    assert pairs == [
        ("Positive", 1),
        ("Positive", 1),
        ("Negative", 1),
        ("Neutral", 1),
    ]
