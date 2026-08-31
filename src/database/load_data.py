"""
Load the processed CSV into MongoDB using batch inserts.

Duplicate records (same post_id) are skipped thanks to a unique index.
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from pymongo.database import Database
from pymongo.errors import BulkWriteError

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import BATCH_SIZE, CLEANED_CSV_PATH, COLLECTION_SOCIAL_POSTS  # noqa: E402
from src.database.mongodb_connection import (  # noqa: E402
    get_database,
    initialise_database,
    upsert_metadata,
)
from src.logging_setup import get_logger  # noqa: E402

logger = get_logger(__name__)


def dataframe_to_documents(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Convert processed CSV rows into social_posts documents."""
    now = datetime.now(timezone.utc)
    documents: list[dict[str, Any]] = []
    for row in df.itertuples(index=False):
        documents.append(
            {
                "post_id": int(row.post_id),
                "original_text": str(row.original_text) if pd.notna(row.original_text) else "",
                "cleaned_text": str(row.cleaned_text) if pd.notna(row.cleaned_text) else "",
                "actual_sentiment": str(row.actual_sentiment) if pd.notna(row.actual_sentiment) else "",
                "predicted_sentiment": None,
                "sentiment_score": None,
                "dataset_split": str(row.split) if pd.notna(row.split) else "",
                "processed": False,
                "created_at": now,
                "updated_at": now,
            }
        )
    return documents


def insert_documents_in_batches(
    documents: list[dict[str, Any]],
    database: Database,
    batch_size: int = BATCH_SIZE,
) -> dict[str, int]:
    """
    Insert documents in batches. Duplicates (unique post_id) are counted, not raised.

    Returns:
        Dict with inserted, duplicates_skipped, and total counts.
    """
    collection = database[COLLECTION_SOCIAL_POSTS]
    inserted = 0
    duplicates = 0

    for start in range(0, len(documents), batch_size):
        batch = documents[start : start + batch_size]
        try:
            result = collection.insert_many(batch, ordered=False)
            inserted += len(result.inserted_ids)
        except BulkWriteError as exc:
            write_errors = exc.details.get("writeErrors", [])
            n_inserted = exc.details.get("nInserted", 0)
            inserted += n_inserted
            # Duplicate key errors have code 11000
            duplicates += sum(1 for error in write_errors if error.get("code") == 11000)
            other_errors = [error for error in write_errors if error.get("code") != 11000]
            if other_errors:
                logger.error("Non-duplicate bulk write errors: %s", other_errors[:3])
                raise

        logger.info(
            "Batch %s–%s processed (running inserted=%s, duplicates=%s)",
            start + 1,
            min(start + batch_size, len(documents)),
            inserted,
            duplicates,
        )

    stats = {
        "total_in_csv": len(documents),
        "inserted": inserted,
        "duplicates_skipped": duplicates,
    }
    logger.info("Insertion complete: %s", stats)
    return stats


def run(
    csv_path: Path = CLEANED_CSV_PATH,
    database: Database | None = None,
    batch_size: int = BATCH_SIZE,
) -> dict[str, int]:
    """
    Load the cleaned CSV into MongoDB.

    Re-running this step is safe: existing post_id values are skipped.
    """
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Cleaned dataset not found at {csv_path}. Run the preprocess step first."
        )

    df = pd.read_csv(csv_path)
    logger.info("Loaded %s cleaned records from %s", len(df), csv_path)

    db = database if database is not None else get_database()
    initialise_database(db)

    documents = dataframe_to_documents(df)
    stats = insert_documents_in_batches(documents, db, batch_size=batch_size)

    upsert_metadata(
        "last_load_stats",
        {**stats, "loaded_at": datetime.now(timezone.utc).isoformat()},
        database=db,
    )
    return stats


if __name__ == "__main__":
    run()
