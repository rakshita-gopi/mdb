"""
End-to-end pipeline runner.

Examples:
    python run_pipeline.py --all
    python run_pipeline.py --step download
    python run_pipeline.py --step preprocess
    python run_pipeline.py --step load
    python run_pipeline.py --step sentiment
    python run_pipeline.py --step mapreduce
    python run_pipeline.py --step evaluate
    python run_pipeline.py --step sentiment --force
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import ensure_directories  # noqa: E402
from src.logging_setup import get_logger  # noqa: E402

logger = get_logger("pipeline")

STEPS = ("download", "explore", "preprocess", "load", "sentiment", "mapreduce", "evaluate")


def step_download() -> None:
    from src.data_collection.download_dataset import run

    run()


def step_explore() -> None:
    from src.data_collection.explore_dataset import run

    run()


def step_preprocess() -> None:
    from src.preprocessing.preprocess import run

    run()


def step_load() -> None:
    from src.database.load_data import run

    run()


def step_sentiment(force: bool = False) -> None:
    from src.sentiment_analysis.sentiment_analyzer import run

    run(force=force)


def step_mapreduce() -> None:
    from src.mapreduce.mapreduce_engine import run

    run()


def step_evaluate() -> None:
    from src.evaluation.evaluate_model import run

    run()


def run_step(name: str, force: bool = False) -> None:
    logger.info("======== STEP: %s ========", name)
    if name == "download":
        step_download()
    elif name == "explore":
        step_explore()
    elif name == "preprocess":
        step_preprocess()
    elif name == "load":
        step_load()
    elif name == "sentiment":
        step_sentiment(force=force)
    elif name == "mapreduce":
        step_mapreduce()
    elif name == "evaluate":
        step_evaluate()
    else:
        raise ValueError(f"Unknown step '{name}'. Choose from: {', '.join(STEPS)}")
    logger.info("======== DONE: %s ========", name)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sentiment Analysis + MapReduce pipeline runner",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--all", action="store_true", help="Run every pipeline step in order")
    group.add_argument(
        "--step",
        choices=STEPS,
        help="Run a single step: " + ", ".join(STEPS),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run sentiment analysis on posts that are already processed",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    ensure_directories()
    try:
        if args.all:
            for name in STEPS:
                run_step(name, force=args.force)
        else:
            run_step(args.step, force=args.force)
    except Exception as exc:  # noqa: BLE001 — top-level CLI handler
        logger.error("Pipeline failed: %s", exc)
        return 1
    logger.info("Pipeline finished successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
