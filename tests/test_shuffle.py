"""Tests for the Shuffle and Group phase."""

from src.mapreduce.shuffle import shuffle_and_group


def test_shuffle_groups_by_key() -> None:
    pairs = [
        ("Positive", 1),
        ("Positive", 1),
        ("Negative", 1),
        ("Neutral", 1),
    ]
    grouped = shuffle_and_group(pairs)
    assert grouped["Positive"] == [1, 1]
    assert grouped["Negative"] == [1]
    assert grouped["Neutral"] == [1]


def test_shuffle_empty() -> None:
    assert shuffle_and_group([]) == {}
