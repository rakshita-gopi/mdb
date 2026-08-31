"""Tests for the Reduce phase."""

from src.mapreduce.reducer import reduce_all, reduce_group


def test_reduce_group_sums_values() -> None:
    result = reduce_group("Positive", [1, 1, 1, 1])
    assert result == {"sentiment": "Positive", "count": 4}


def test_reduce_all_counts_and_percentages() -> None:
    grouped = {
        "Positive": [1, 1, 1, 1],
        "Negative": [1, 1],
        "Neutral": [1, 1, 1],
    }
    reduced = reduce_all(grouped)
    assert reduced["total_posts"] == 9
    assert reduced["sentiments"]["Positive"]["count"] == 4
    assert reduced["sentiments"]["Negative"]["count"] == 2
    assert reduced["sentiments"]["Neutral"]["count"] == 3
    assert reduced["sentiments"]["Positive"]["percentage"] == 44.44
    assert reduced["sentiments"]["Negative"]["percentage"] == 22.22
    assert reduced["sentiments"]["Neutral"]["percentage"] == 33.33


def test_reduce_all_fills_missing_labels() -> None:
    reduced = reduce_all({"Positive": [1, 1]})
    assert reduced["sentiments"]["Negative"]["count"] == 0
    assert reduced["sentiments"]["Neutral"]["count"] == 0
    assert reduced["total_posts"] == 2
