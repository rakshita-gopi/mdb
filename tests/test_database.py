"""Tests for MongoDB helpers using mongomock (no live server required)."""

from datetime import datetime, timezone

import mongomock
import pandas as pd
from pymongo import ASCENDING

from src.config import COLLECTION_SOCIAL_POSTS
from src.database.load_data import dataframe_to_documents, insert_documents_in_batches
from src.database.mongodb_connection import initialise_database


def _mock_database():
    client = mongomock.MongoClient()
    return client["sentiment_analysis_db"]


def test_initialise_database_creates_collections_and_indexes() -> None:
    db = _mock_database()
    initialise_database(db)
    names = set(db.list_collection_names())
    assert "social_posts" in names
    assert "processing_results" in names
    assert "mapreduce_results" in names
    assert "evaluation_results" in names
    assert "project_metadata" in names

    index_names = db[COLLECTION_SOCIAL_POSTS].index_information()
    assert "uniq_post_id" in index_names


def test_dataframe_to_documents_schema() -> None:
    df = pd.DataFrame(
        {
            "post_id": [1],
            "original_text": ["I love this"],
            "cleaned_text": ["i love this"],
            "actual_sentiment": ["Positive"],
            "split": ["train"],
        }
    )
    docs = dataframe_to_documents(df)
    assert len(docs) == 1
    doc = docs[0]
    assert doc["post_id"] == 1
    assert doc["predicted_sentiment"] is None
    assert doc["sentiment_score"] is None
    assert doc["processed"] is False
    assert doc["dataset_split"] == "train"
    assert isinstance(doc["created_at"], datetime)


def test_batch_insert_skips_duplicates() -> None:
    db = _mock_database()
    initialise_database(db)
    now = datetime.now(timezone.utc)
    docs = [
        {
            "post_id": 1,
            "original_text": "a",
            "cleaned_text": "a",
            "actual_sentiment": "Positive",
            "predicted_sentiment": None,
            "sentiment_score": None,
            "dataset_split": "train",
            "processed": False,
            "created_at": now,
            "updated_at": now,
        },
        {
            "post_id": 2,
            "original_text": "b",
            "cleaned_text": "b",
            "actual_sentiment": "Negative",
            "predicted_sentiment": None,
            "sentiment_score": None,
            "dataset_split": "train",
            "processed": False,
            "created_at": now,
            "updated_at": now,
        },
    ]
    first = insert_documents_in_batches(docs, db, batch_size=10)
    assert first["inserted"] == 2
    assert first["duplicates_skipped"] == 0

    second = insert_documents_in_batches(docs, db, batch_size=10)
    assert second["inserted"] == 0
    assert second["duplicates_skipped"] == 2
    assert db[COLLECTION_SOCIAL_POSTS].count_documents({}) == 2


def test_unique_post_id_index_direction() -> None:
    db = _mock_database()
    initialise_database(db)
    info = db[COLLECTION_SOCIAL_POSTS].index_information()["uniq_post_id"]
    assert info["key"] == [("post_id", ASCENDING)]
    assert info.get("unique") is True
