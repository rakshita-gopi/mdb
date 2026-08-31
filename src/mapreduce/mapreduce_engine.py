"""
MapReduce engine.

Coordinates Map → Shuffle/Group → Reduce, stores aggregated results
and visualisation samples in MongoDB, and records execution metadata.
"""

from __future__ import annotations

import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pymongo.database import Database

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (  # noqa: E402
    COLLECTION_MAPREDUCE_RESULTS,
    COLLECTION_PROCESSING_RESULTS,
    COLLECTION_SOCIAL_POSTS,
    MAPREDUCE_SAMPLE_SIZE,
)
from src.database.mongodb_connection import get_database, upsert_metadata  # noqa: E402
from src.logging_setup import get_logger  # noqa: E402
from src.mapreduce.mapper import map_posts  # noqa: E402
from src.mapreduce.reducer import reduce_all  # noqa: E402
from src.mapreduce.shuffle import shuffle_and_group  # noqa: E402

logger = get_logger(__name__)


def _new_analysis_id() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"run_{stamp}_{uuid.uuid4().hex[:6]}"


def fetch_processed_posts(database: Database) -> list[dict[str, Any]]:
    """Load processed posts needed by the mapper (sentiment + post_id)."""
    cursor = database[COLLECTION_SOCIAL_POSTS].find(
        {"processed": True, "predicted_sentiment": {"$ne": None}},
        {"_id": 0, "post_id": 1, "predicted_sentiment": 1, "cleaned_text": 1},
    )
    posts = list(cursor)
    logger.info("Fetched %s processed posts for MapReduce", len(posts))
    return posts


def run_mapreduce(
    posts: list[dict[str, Any]] | None = None,
    database: Database | None = None,
    sample_size: int = MAPREDUCE_SAMPLE_SIZE,
) -> dict[str, Any]:
    """
    Execute the full Map → Shuffle → Reduce pipeline.

    Stores:
        - one document per sentiment in mapreduce_results
        - execution metadata + map/shuffle samples in processing_results
        - latest analysis_id in project_metadata
    """
    db = database if database is not None else get_database()
    if posts is None:
        posts = fetch_processed_posts(db)

    if not posts:
        raise RuntimeError(
            "No processed posts found. Run the sentiment analysis step first."
        )

    analysis_id = _new_analysis_id()
    started = time.perf_counter()
    created_at = datetime.now(timezone.utc)

    logger.info("MapReduce %s — MAP phase (%s posts)", analysis_id, len(posts))
    mapped_pairs = map_posts(posts)

    logger.info("MapReduce %s — SHUFFLE AND GROUP phase", analysis_id)
    grouped = shuffle_and_group(mapped_pairs)

    logger.info("MapReduce %s — REDUCE phase", analysis_id)
    reduced = reduce_all(grouped)

    duration = round(time.perf_counter() - started, 4)
    logger.info(
        "MapReduce %s complete in %ss (total_posts=%s)",
        analysis_id,
        duration,
        reduced["total_posts"],
    )

    _store_mapreduce_results(db, analysis_id, reduced, created_at)
    _store_processing_results(
        db,
        analysis_id=analysis_id,
        mapped_pairs=mapped_pairs,
        grouped=grouped,
        reduced=reduced,
        duration=duration,
        created_at=created_at,
        sample_size=sample_size,
        posts=posts,
    )
    upsert_metadata("latest_analysis_id", analysis_id, database=db)
    upsert_metadata(
        "latest_mapreduce",
        {
            "analysis_id": analysis_id,
            "total_posts": reduced["total_posts"],
            "duration_seconds": duration,
            "created_at": created_at.isoformat(),
            "sentiments": reduced["sentiments"],
        },
        database=db,
    )

    return {
        "analysis_id": analysis_id,
        "duration_seconds": duration,
        "created_at": created_at.isoformat(),
        **reduced,
    }


def _store_mapreduce_results(
    database: Database,
    analysis_id: str,
    reduced: dict[str, Any],
    created_at: datetime,
) -> None:
    """Write one aggregated document per sentiment class."""
    collection = database[COLLECTION_MAPREDUCE_RESULTS]
    documents = []
    total = reduced["total_posts"]
    for sentiment, payload in reduced["sentiments"].items():
        documents.append(
            {
                "analysis_id": analysis_id,
                "sentiment": sentiment,
                "count": payload["count"],
                "percentage": payload["percentage"],
                "total_posts": total,
                "created_at": created_at,
            }
        )
    if documents:
        collection.insert_many(documents)
        logger.info("Stored %s mapreduce_results documents for %s", len(documents), analysis_id)


def _store_processing_results(
    database: Database,
    analysis_id: str,
    mapped_pairs: list[tuple[str, int]],
    grouped: dict[str, list[int]],
    reduced: dict[str, Any],
    duration: float,
    created_at: datetime,
    sample_size: int,
    posts: list[dict[str, Any]],
) -> None:
    """
    Persist execution metadata and stage samples for the dashboard.

    Full grouped lists of tens of thousands of 1s are summarised: the
    shuffle sample keeps a short prefix of each group's values so the
    UI can show Positive → [1, 1, 1, ...] without huge documents.
    """
    map_sample = [
        {"key": key, "value": value, "post_id": posts[i].get("post_id"), "text": posts[i].get("cleaned_text", "")[:120]}
        for i, (key, value) in enumerate(mapped_pairs[:sample_size])
    ]

    shuffle_sample = {
        key: values[: min(12, len(values))]
        for key, values in grouped.items()
    }
    shuffle_counts = {key: len(values) for key, values in grouped.items()}

    document = {
        "analysis_id": analysis_id,
        "record_type": "mapreduce_execution",
        "created_at": created_at,
        "duration_seconds": duration,
        "total_records_processed": reduced["total_posts"],
        "map_pair_count": len(mapped_pairs),
        "map_sample": map_sample,
        "shuffle_sample": shuffle_sample,
        "shuffle_group_sizes": shuffle_counts,
        "reduce_results": reduced["sentiments"],
    }
    database[COLLECTION_PROCESSING_RESULTS].insert_one(document)
    logger.info("Stored processing_results for %s", analysis_id)


def run(database: Database | None = None) -> dict[str, Any]:
    """Entry point used by the pipeline runner."""
    return run_mapreduce(database=database)


if __name__ == "__main__":
    result = run()
    logger.info("Final aggregated results: %s", result["sentiments"])
