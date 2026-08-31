"""
Map phase of the custom MapReduce pipeline.

Each processed social media post emits a key-value pair:
    (predicted_sentiment, 1)
"""

from __future__ import annotations

from typing import Any, Iterable


def map_post(post: dict[str, Any]) -> tuple[str, int] | None:
    """
    Map a single post to (sentiment, 1).

    Returns None if the post has no predicted sentiment.
    """
    sentiment = post.get("predicted_sentiment")
    if not sentiment:
        return None
    return (str(sentiment), 1)


def map_posts(posts: Iterable[dict[str, Any]]) -> list[tuple[str, int]]:
    """
    Map a collection of posts to intermediate key-value pairs.

    Example output:
        [("Positive", 1), ("Negative", 1), ("Neutral", 1), ...]
    """
    pairs: list[tuple[str, int]] = []
    for post in posts:
        pair = map_post(post)
        if pair is not None:
            pairs.append(pair)
    return pairs
