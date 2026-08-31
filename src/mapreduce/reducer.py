"""
Reduce phase of the custom MapReduce pipeline.

Aggregates grouped (sentiment → [1, 1, ...]) lists into counts and percentages.
"""

from __future__ import annotations

from typing import Any


def reduce_group(sentiment: str, values: list[int]) -> dict[str, Any]:
    """Sum the values for a single sentiment key."""
    return {
        "sentiment": sentiment,
        "count": int(sum(values)),
    }


def reduce_all(grouped: dict[str, list[int]]) -> dict[str, Any]:
    """
    Reduce every grouped key and attach percentages plus total.

    Input:
        {"Positive": [1, 1, 1, 1], "Negative": [1, 1], "Neutral": [1, 1, 1]}

    Output:
        {
            "total_posts": 9,
            "sentiments": {
                "Positive": {"count": 4, "percentage": 44.44},
                "Negative": {"count": 2, "percentage": 22.22},
                "Neutral": {"count": 3, "percentage": 33.33},
            },
        }
    """
    reduced = {key: reduce_group(key, values) for key, values in grouped.items()}
    total = sum(item["count"] for item in reduced.values())

    sentiments: dict[str, dict[str, Any]] = {}
    for key, item in reduced.items():
        count = item["count"]
        percentage = round((count / total) * 100, 2) if total else 0.0
        sentiments[key] = {"count": count, "percentage": percentage}

    # Guarantee all three labels appear even if a class had zero posts
    for label in ("Positive", "Negative", "Neutral"):
        sentiments.setdefault(label, {"count": 0, "percentage": 0.0})

    return {
        "total_posts": total,
        "sentiments": sentiments,
    }
