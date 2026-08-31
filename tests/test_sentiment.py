"""Tests for VADER classification logic (no MongoDB required)."""

from src.sentiment_analysis.sentiment_analyzer import analyse_text, classify_compound


def test_classify_positive() -> None:
    assert classify_compound(0.82) == "Positive"
    assert classify_compound(0.05) == "Positive"


def test_classify_negative() -> None:
    assert classify_compound(-0.82) == "Negative"
    assert classify_compound(-0.05) == "Negative"


def test_classify_neutral() -> None:
    assert classify_compound(0.0) == "Neutral"
    assert classify_compound(0.04) == "Neutral"
    assert classify_compound(-0.04) == "Neutral"


def test_analyse_positive_text() -> None:
    result = analyse_text("I absolutely love this product!")
    assert result["predicted_sentiment"] == "Positive"
    assert result["sentiment_score"] >= 0.05


def test_analyse_negative_text() -> None:
    result = analyse_text("This service is terrible and I hate it.")
    assert result["predicted_sentiment"] == "Negative"
    assert result["sentiment_score"] <= -0.05


def test_analyse_does_not_use_label() -> None:
    """Prediction depends only on text, not on any actual_sentiment field."""
    result = analyse_text("The product is okay")
    assert "actual_sentiment" not in result
    assert result["predicted_sentiment"] in {"Positive", "Negative", "Neutral"}
