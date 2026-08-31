"""
MongoDB connection helpers.

Reads connection details from environment variables, initialises the
database and collections, and creates useful indexes.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from pymongo import ASCENDING, MongoClient
from pymongo.collection import Collection
from pymongo.database import Database
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (  # noqa: E402
    COLLECTION_EVALUATION_RESULTS,
    COLLECTION_MAPREDUCE_RESULTS,
    COLLECTION_PROCESSING_RESULTS,
    COLLECTION_PROJECT_METADATA,
    COLLECTION_SOCIAL_POSTS,
    MONGODB_DATABASE,
    MONGODB_URI,
)
from src.logging_setup import get_logger  # noqa: E402

logger = get_logger(__name__)

# How long to wait for a MongoDB server (ms)
SERVER_SELECTION_TIMEOUT_MS = 8000


class MongoDBConnectionError(RuntimeError):
    """Raised when the application cannot connect to MongoDB."""


def get_client(uri: str | None = None, timeout_ms: int = SERVER_SELECTION_TIMEOUT_MS) -> MongoClient:
    """
    Create a MongoClient and verify the connection with ping.

    Args:
        uri: MongoDB connection string. Defaults to MONGODB_URI from config.
        timeout_ms: Server selection timeout in milliseconds.
    """
    connection_uri = uri or MONGODB_URI
    try:
        client: MongoClient = MongoClient(
            connection_uri,
            serverSelectionTimeoutMS=timeout_ms,
        )
        client.admin.command("ping")
        logger.info("Connected to MongoDB at %s", _redact_uri(connection_uri))
        return client
    except (ConnectionFailure, ServerSelectionTimeoutError) as exc:
        raise MongoDBConnectionError(
            "Could not connect to MongoDB. Check that the server is running "
            f"and that MONGODB_URI is correct. Details: {exc}"
        ) from exc


def get_database(client: MongoClient | None = None, name: str | None = None) -> Database:
    """Return the project database, creating the client if needed."""
    active_client = client or get_client()
    db_name = name or MONGODB_DATABASE
    return active_client[db_name]


def get_collection(name: str, database: Database | None = None) -> Collection:
    """Return a named collection from the project database."""
    db = database if database is not None else get_database()
    return db[name]


def initialise_database(database: Database | None = None) -> Database:
    """
    Ensure all project collections exist and create indexes.

    Collections:
        social_posts, processing_results, mapreduce_results,
        evaluation_results, project_metadata
    """
    db = database if database is not None else get_database()

    collection_names = {
        COLLECTION_SOCIAL_POSTS,
        COLLECTION_PROCESSING_RESULTS,
        COLLECTION_MAPREDUCE_RESULTS,
        COLLECTION_EVALUATION_RESULTS,
        COLLECTION_PROJECT_METADATA,
    }
    existing = set(db.list_collection_names())
    for collection_name in collection_names:
        if collection_name not in existing:
            db.create_collection(collection_name)
            logger.info("Created collection '%s'", collection_name)

    _create_indexes(db)
    logger.info("Database '%s' is initialised", db.name)
    return db


def _create_indexes(database: Database) -> None:
    """Create indexes that support lookups and duplicate prevention."""
    posts = database[COLLECTION_SOCIAL_POSTS]
    posts.create_index([("post_id", ASCENDING)], unique=True, name="uniq_post_id")
    posts.create_index([("actual_sentiment", ASCENDING)], name="idx_actual_sentiment")
    posts.create_index([("predicted_sentiment", ASCENDING)], name="idx_predicted_sentiment")
    posts.create_index([("processed", ASCENDING)], name="idx_processed")
    posts.create_index([("dataset_split", ASCENDING)], name="idx_dataset_split")

    mr = database[COLLECTION_MAPREDUCE_RESULTS]
    mr.create_index([("analysis_id", ASCENDING)], name="idx_analysis_id")
    mr.create_index([("sentiment", ASCENDING)], name="idx_mr_sentiment")

    evaluation = database[COLLECTION_EVALUATION_RESULTS]
    evaluation.create_index([("evaluation_id", ASCENDING)], unique=True, name="uniq_evaluation_id")

    processing = database[COLLECTION_PROCESSING_RESULTS]
    processing.create_index([("analysis_id", ASCENDING)], name="idx_processing_analysis_id")

    metadata = database[COLLECTION_PROJECT_METADATA]
    metadata.create_index([("key", ASCENDING)], unique=True, name="uniq_metadata_key")

    logger.info("MongoDB indexes created or already present")


def upsert_metadata(key: str, value: Any, database: Database | None = None) -> None:
    """Store a key/value pair in project_metadata (upsert)."""
    db = database if database is not None else get_database()
    db[COLLECTION_PROJECT_METADATA].update_one(
        {"key": key},
        {"$set": {"key": key, "value": value}},
        upsert=True,
    )


def get_metadata(key: str, database: Database | None = None) -> Any:
    """Read a metadata value by key, or None if missing."""
    db = database if database is not None else get_database()
    document = db[COLLECTION_PROJECT_METADATA].find_one({"key": key})
    if document is None:
        return None
    return document.get("value")


def _redact_uri(uri: str) -> str:
    """Hide credentials if a connection string contains them."""
    if "@" not in uri:
        return uri
    scheme, rest = uri.split("://", 1) if "://" in uri else ("", uri)
    credentials, host = rest.rsplit("@", 1)
    return f"{scheme}://***@{host}" if scheme else f"***@{host}"
