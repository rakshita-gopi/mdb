"""
Shuffle and Group phase of the custom MapReduce pipeline.

Groups all intermediate values that share the same key.
"""

from __future__ import annotations

from collections import defaultdict


def shuffle_and_group(pairs: list[tuple[str, int]]) -> dict[str, list[int]]:
    """
    Group mapper output by sentiment key.

    Input:
        [("Positive", 1), ("Positive", 1), ("Negative", 1), ("Neutral", 1)]

    Output:
        {
            "Positive": [1, 1],
            "Negative": [1],
            "Neutral": [1],
        }
    """
    grouped: dict[str, list[int]] = defaultdict(list)
    for key, value in pairs:
        grouped[key].append(value)
    return dict(grouped)
