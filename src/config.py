"""
Central project configuration.

Reads MongoDB settings from environment variables and defines
shared paths, collection names, batch sizes, and VADER thresholds.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env from the project root (parent of src/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

RAW_CSV_PATH = RAW_DATA_DIR / "social_media_raw.csv"
CLEANED_CSV_PATH = PROCESSED_DATA_DIR / "social_media_cleaned.csv"

SENTIMENT_RESULTS_CSV = RESULTS_DIR / "sentiment_results.csv"
EVALUATION_RESULTS_JSON = RESULTS_DIR / "evaluation_results.json"
CLASSIFICATION_REPORT_JSON = RESULTS_DIR / "classification_report.json"
CONFUSION_MATRIX_PNG = RESULTS_DIR / "confusion_matrix.png"
EXPLORATION_RESULTS_JSON = RESULTS_DIR / "exploration_results.json"

# ---------------------------------------------------------------------------
# MongoDB
# ---------------------------------------------------------------------------
MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
MONGODB_DATABASE = os.getenv("MONGODB_DATABASE", "sentiment_analysis_db")

COLLECTION_SOCIAL_POSTS = "social_posts"
COLLECTION_PROCESSING_RESULTS = "processing_results"
COLLECTION_MAPREDUCE_RESULTS = "mapreduce_results"
COLLECTION_EVALUATION_RESULTS = "evaluation_results"
COLLECTION_PROJECT_METADATA = "project_metadata"

# ---------------------------------------------------------------------------
# Processing
# ---------------------------------------------------------------------------
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "1000"))
MAPREDUCE_SAMPLE_SIZE = int(os.getenv("MAPREDUCE_SAMPLE_SIZE", "20"))

# VADER compound-score classification thresholds
VADER_POSITIVE_THRESHOLD = 0.05
VADER_NEGATIVE_THRESHOLD = -0.05

SENTIMENT_LABELS = ("Negative", "Neutral", "Positive")
LABEL_MAP = {0: "Negative", 1: "Neutral", 2: "Positive"}

DATASET_NAME = "tweet_eval"
DATASET_CONFIG = "sentiment"


def ensure_directories() -> None:
    """Create data and results directories if they do not exist."""
    for path in (RAW_DATA_DIR, PROCESSED_DATA_DIR, RESULTS_DIR, NOTEBOOKS_DIR):
        path.mkdir(parents=True, exist_ok=True)
