"""
Evaluate predicted sentiment against actual TweetEval labels.

Computes accuracy, precision, recall, and F1 (macro and weighted),
writes artifacts under results/, and stores a summary in MongoDB.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pymongo.database import Database
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (  # noqa: E402
    CLASSIFICATION_REPORT_JSON,
    COLLECTION_EVALUATION_RESULTS,
    COLLECTION_SOCIAL_POSTS,
    CONFUSION_MATRIX_PNG,
    EVALUATION_RESULTS_JSON,
    SENTIMENT_LABELS,
    ensure_directories,
)
from src.database.mongodb_connection import get_database, upsert_metadata  # noqa: E402
from src.logging_setup import get_logger  # noqa: E402

logger = get_logger(__name__)

LABELS = list(SENTIMENT_LABELS)


def fetch_labelled_posts(database: Database) -> tuple[list[str], list[str]]:
    """Return (y_true, y_pred) for processed posts that have both labels."""
    cursor = database[COLLECTION_SOCIAL_POSTS].find(
        {
            "processed": True,
            "predicted_sentiment": {"$ne": None},
            "actual_sentiment": {"$ne": None},
        },
        {"_id": 0, "actual_sentiment": 1, "predicted_sentiment": 1},
    )
    y_true: list[str] = []
    y_pred: list[str] = []
    for document in cursor:
        y_true.append(str(document["actual_sentiment"]))
        y_pred.append(str(document["predicted_sentiment"]))
    logger.info("Loaded %s labelled posts for evaluation", len(y_true))
    return y_true, y_pred


def compute_metrics(y_true: list[str], y_pred: list[str]) -> dict[str, Any]:
    """Compute multi-class classification metrics."""
    if not y_true:
        raise RuntimeError("No labelled posts available for evaluation.")

    report_dict = classification_report(
        y_true,
        y_pred,
        labels=LABELS,
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(y_true, y_pred, labels=LABELS)

    metrics = {
        "n_samples": len(y_true),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0)),
        "precision_weighted": float(precision_score(y_true, y_pred, labels=LABELS, average="weighted", zero_division=0)),
        "recall_weighted": float(recall_score(y_true, y_pred, labels=LABELS, average="weighted", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, labels=LABELS, average="weighted", zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0)),
        "classification_report": report_dict,
        "confusion_matrix": matrix.tolist(),
        "labels": LABELS,
    }
    return metrics


def save_confusion_matrix_plot(matrix: list[list[int]], labels: list[str], path: Path = CONFUSION_MATRIX_PNG) -> Path:
    """Render and save a confusion-matrix heatmap."""
    array = np.array(matrix)
    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(array, interpolation="nearest", cmap="Blues")
    fig.colorbar(image, ax=ax)
    ax.set(
        xticks=np.arange(len(labels)),
        yticks=np.arange(len(labels)),
        xticklabels=labels,
        yticklabels=labels,
        ylabel="Actual sentiment",
        xlabel="Predicted sentiment",
        title="Confusion Matrix",
    )
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    thresh = array.max() / 2 if array.size else 0
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            ax.text(
                j,
                i,
                format(array[i, j], "d"),
                ha="center",
                va="center",
                color="white" if array[i, j] > thresh else "black",
            )
    fig.tight_layout()
    ensure_directories()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)
    logger.info("Saved confusion matrix image to %s", path)
    return path


def save_json(payload: dict[str, Any], path: Path) -> Path:
    """Write a JSON artifact under results/."""
    ensure_directories()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    logger.info("Saved %s", path)
    return path


def store_evaluation_in_mongodb(
    metrics: dict[str, Any],
    database: Database,
    evaluation_id: str,
    created_at: datetime,
) -> None:
    """Insert the evaluation summary into evaluation_results."""
    document = {
        "evaluation_id": evaluation_id,
        "accuracy": metrics["accuracy"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1_score": metrics["f1_score"],
        "precision_macro": metrics["precision_macro"],
        "recall_macro": metrics["recall_macro"],
        "f1_macro": metrics["f1_macro"],
        "precision_weighted": metrics["precision_weighted"],
        "recall_weighted": metrics["recall_weighted"],
        "f1_weighted": metrics["f1_weighted"],
        "n_samples": metrics["n_samples"],
        "classification_report": metrics["classification_report"],
        "confusion_matrix": metrics["confusion_matrix"],
        "labels": metrics["labels"],
        "created_at": created_at,
    }
    database[COLLECTION_EVALUATION_RESULTS].insert_one(document)
    upsert_metadata("latest_evaluation_id", evaluation_id, database=database)
    upsert_metadata(
        "latest_evaluation",
        {
            "evaluation_id": evaluation_id,
            "accuracy": metrics["accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1_score": metrics["f1_score"],
            "n_samples": metrics["n_samples"],
            "created_at": created_at.isoformat(),
        },
        database=database,
    )
    logger.info("Stored evaluation_results document %s", evaluation_id)


def run(database: Database | None = None) -> dict[str, Any]:
    """Run evaluation, write files, and persist results in MongoDB."""
    db = database if database is not None else get_database()
    y_true, y_pred = fetch_labelled_posts(db)
    metrics = compute_metrics(y_true, y_pred)

    created_at = datetime.now(timezone.utc)
    evaluation_id = f"evaluation_{created_at.strftime('%Y%m%d_%H%M%S')}"

    file_payload = {
        "evaluation_id": evaluation_id,
        "created_at": created_at.isoformat(),
        **{k: v for k, v in metrics.items() if k != "classification_report"},
    }
    save_json(file_payload, EVALUATION_RESULTS_JSON)
    save_json(metrics["classification_report"], CLASSIFICATION_REPORT_JSON)
    save_confusion_matrix_plot(metrics["confusion_matrix"], metrics["labels"])
    store_evaluation_in_mongodb(metrics, db, evaluation_id, created_at)

    logger.info(
        "Evaluation %s — accuracy=%.4f precision=%.4f recall=%.4f f1=%.4f",
        evaluation_id,
        metrics["accuracy"],
        metrics["precision"],
        metrics["recall"],
        metrics["f1_score"],
    )
    return {"evaluation_id": evaluation_id, **metrics}


if __name__ == "__main__":
    run()
