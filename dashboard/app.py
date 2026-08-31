"""
Streamlit dashboard for the Sentiment Analysis + MapReduce project.

All metrics and tables are loaded from MongoDB. If a pipeline stage has
not been run yet, the page shows a clear empty state instead of fake data.
"""

from __future__ import annotations

import html
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (  # noqa: E402
    COLLECTION_EVALUATION_RESULTS,
    COLLECTION_MAPREDUCE_RESULTS,
    COLLECTION_PROCESSING_RESULTS,
    COLLECTION_SOCIAL_POSTS,
    CONFUSION_MATRIX_PNG,
    MONGODB_DATABASE,
)
from src.database.mongodb_connection import (  # noqa: E402
    MongoDBConnectionError,
    get_client,
    get_database,
    get_metadata,
)
from src.sentiment_analysis.sentiment_analyzer import analyse_text  # noqa: E402

try:
    from dashboard.gsap_runtime import inject_gsap
except ImportError:  # streamlit run dashboard/app.py puts this folder first on sys.path
    from gsap_runtime import inject_gsap

st.set_page_config(
    page_title="Sentiment Analysis · MapReduce",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

SENTIMENT_COLORS = {
    "Positive": "#34d399",
    "Negative": "#f87171",
    "Neutral": "#94a3b8",
}
SENTIMENT_SOFT = {
    "Positive": "rgba(52, 211, 153, 0.18)",
    "Negative": "rgba(248, 113, 113, 0.18)",
    "Neutral": "rgba(148, 163, 184, 0.18)",
}

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: "Plus Jakarta Sans", "Segoe UI", sans-serif;
}

.stApp {
    background:
        radial-gradient(1200px 500px at -10% -10%, rgba(56, 189, 248, 0.16), transparent 55%),
        radial-gradient(900px 420px at 110% 0%, rgba(52, 211, 153, 0.10), transparent 50%),
        radial-gradient(800px 500px at 80% 110%, rgba(167, 139, 250, 0.10), transparent 55%);
}

[data-testid="stHeader"] { background: transparent; }

.block-container {
    padding-top: 1.1rem;
    padding-bottom: 2.4rem;
}

.hero h1 .gsap-ch {
    display: inline-block;
    will-change: transform, opacity;
}

[data-testid="stIFrame"] {
    height: 0 !important;
    min-height: 0 !important;
    margin: 0 !important;
    padding: 0 !important;
    overflow: hidden !important;
    border: 0 !important;
}

@keyframes pulseDot {
    0%, 100% { box-shadow: 0 0 0 0 rgba(52, 211, 153, 0.55); }
    70% { box-shadow: 0 0 0 8px rgba(52, 211, 153, 0); }
}
@keyframes shimmer {
    0% { background-position: 0% 50%; }
    100% { background-position: 200% 50%; }
}

section[data-testid="stSidebar"] {
    border-right: 1px solid rgba(148, 163, 184, 0.18);
    backdrop-filter: blur(14px);
}
section[data-testid="stSidebar"] .stRadio > div {
    gap: 0.28rem;
}
section[data-testid="stSidebar"] label[data-baseweb="radio"] {
    padding: 0.62rem 0.8rem;
    border-radius: 12px;
    border: 1px solid transparent;
    transition: transform 0.22s ease, background 0.22s ease, border-color 0.22s ease;
}
section[data-testid="stSidebar"] label[data-baseweb="radio"]:hover {
    transform: translateX(4px);
    background: rgba(56, 189, 248, 0.10);
    border-color: rgba(56, 189, 248, 0.22);
}

.brand {
    padding: 0.15rem 0.15rem 1rem 0.15rem;
}
.brand-kicker {
    font-size: 0.72rem;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    opacity: 0.65;
    font-weight: 700;
}
.brand-title {
    font-size: 1.28rem;
    font-weight: 800;
    line-height: 1.2;
    margin: 0.2rem 0 0.45rem 0;
    background: linear-gradient(90deg, #38bdf8, #34d399, #a78bfa, #38bdf8);
    background-size: 220% 100%;
    -webkit-background-clip: text;
    background-clip: text;
    color: transparent;
    animation: shimmer 7s linear infinite;
}
.status-pill {
    display: inline-flex;
    align-items: center;
    gap: 0.45rem;
    font-size: 0.78rem;
    padding: 0.28rem 0.7rem;
    border-radius: 999px;
    background: rgba(52, 211, 153, 0.12);
    border: 1px solid rgba(52, 211, 153, 0.28);
}
.status-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: #34d399;
    animation: pulseDot 1.6s ease infinite;
}

.hero {
    position: relative;
    overflow: hidden;
    border-radius: 22px;
    padding: 1.45rem 1.6rem 1.35rem 1.6rem;
    margin-bottom: 1.15rem;
    border: 1px solid rgba(148, 163, 184, 0.22);
    background:
        linear-gradient(135deg, rgba(56, 189, 248, 0.12), rgba(167, 139, 250, 0.08) 45%, rgba(52, 211, 153, 0.08));
    box-shadow: 0 18px 50px rgba(0, 0, 0, 0.12);
}
.hero:after {
    content: "";
    position: absolute;
    width: 180px;
    height: 180px;
    right: -30px;
    top: -40px;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(56, 189, 248, 0.28), transparent 70%);
}
.hero-kicker {
    font-size: 0.75rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    font-weight: 700;
    opacity: 0.7;
}
.hero h1 {
    margin: 0.25rem 0 0.4rem 0;
    font-size: 2rem;
    font-weight: 800;
    letter-spacing: -0.03em;
}
.hero p {
    margin: 0;
    max-width: 70ch;
    opacity: 0.82;
    line-height: 1.55;
}

.kpi-grid {
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 0.85rem;
    margin: 0.2rem 0 1.1rem 0;
}
@media (max-width: 1100px) {
    .kpi-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
}
.kpi-card {
    border-radius: 18px;
    padding: 1rem 1.05rem 0.95rem 1.05rem;
    border: 1px solid rgba(148, 163, 184, 0.22);
    background: rgba(15, 23, 42, 0.18);
    backdrop-filter: blur(10px);
}
.kpi-label {
    font-size: 0.78rem;
    opacity: 0.68;
    font-weight: 600;
}
.kpi-value {
    font-size: 1.42rem;
    font-weight: 800;
    letter-spacing: -0.03em;
    margin: 0.22rem 0 0.18rem 0;
    word-break: break-word;
}
.kpi-hint {
    font-size: 0.75rem;
    opacity: 0.55;
}

.tech-grid {
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 0.75rem;
    margin-bottom: 0.4rem;
}
@media (max-width: 1100px) {
    .tech-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
}
.tech-card {
    border-radius: 16px;
    padding: 0.9rem 0.95rem;
    border: 1px solid rgba(148, 163, 184, 0.2);
    background: rgba(255, 255, 255, 0.03);
}
.tech-k { font-size: 0.72rem; opacity: 0.6; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em; }
.tech-v { font-size: 0.98rem; font-weight: 700; margin-top: 0.2rem; }

.pipeline {
    display: flex;
    align-items: center;
    gap: 0.55rem;
    flex-wrap: wrap;
    margin: 0.3rem 0 1rem 0;
}
.pipe-step {
    padding: 0.55rem 0.9rem;
    border-radius: 999px;
    font-weight: 700;
    font-size: 0.86rem;
    border: 1px solid rgba(56, 189, 248, 0.28);
    background: rgba(56, 189, 248, 0.10);
}
.pipe-step:nth-child(3) { background: rgba(167, 139, 250, 0.12); border-color: rgba(167, 139, 250, 0.3); }
.pipe-step:nth-child(5) { background: rgba(52, 211, 153, 0.12); border-color: rgba(52, 211, 153, 0.3); }
.pipe-arrow {
    opacity: 0.7;
    font-weight: 800;
    display: inline-block;
}

.empty-note {
    padding: 1.2rem 1.35rem;
    border: 1px dashed rgba(148, 163, 184, 0.45);
    border-radius: 16px;
    background: rgba(148, 163, 184, 0.06);
}
.section-title {
    font-size: 1.18rem;
    font-weight: 800;
    letter-spacing: -0.02em;
    margin: 0.4rem 0 0.7rem 0;
}
.result-card {
    border-radius: 18px;
    padding: 1.05rem 1.15rem;
    border: 1px solid rgba(148, 163, 184, 0.22);
}
.score-track {
    height: 10px;
    border-radius: 999px;
    background: rgba(148, 163, 184, 0.2);
    overflow: hidden;
    margin-top: 0.55rem;
}
.score-fill {
    height: 100%;
    border-radius: 999px;
}

div[data-testid="stPlotlyChart"] {
    min-height: 280px;
}
.stButton > button {
    border-radius: 12px;
    font-weight: 700;
    transition: transform 0.18s ease, box-shadow 0.18s ease;
}
.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 18px rgba(56, 189, 248, 0.22);
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

NAV_PAGES = [
    "Project overview",
    "Dataset overview",
    "Sentiment analysis",
    "MapReduce visualization",
    "Analytics",
    "Model evaluation",
]
NAV_ICONS = {
    "Project overview": "🏠",
    "Dataset overview": "🗃️",
    "Sentiment analysis": "💬",
    "MapReduce visualization": "🗺️",
    "Analytics": "📈",
    "Model evaluation": "🎯",
}


@st.cache_resource(show_spinner=False)
def _connect():
    client = get_client()
    return get_database(client)


def get_db():
    """Return the MongoDB database or None if the server is unavailable."""
    try:
        return _connect()
    except MongoDBConnectionError as exc:
        st.error(
            f"Cannot connect to MongoDB ({MONGODB_DATABASE}). "
            "Start the local server and confirm `.env` settings."
        )
        st.caption(str(exc))
        return None


def empty_state(message: str, command: str) -> None:
    st.markdown(
        f'<div class="empty-note">{html.escape(message)}<br><br>'
        f"<code>{html.escape(command)}</code></div>",
        unsafe_allow_html=True,
    )


def page_header(title: str, subtitle: str, kicker: str = "Academic dashboard") -> None:
    st.markdown(
        f"""
        <div class="hero">
            <div class="hero-kicker">{html.escape(kicker)}</div>
            <h1>{html.escape(title)}</h1>
            <p>{html.escape(subtitle)}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def kpi_cards(items: list[tuple[str, Any, str]]) -> None:
    cards = []
    for label, value, hint in items:
        cards.append(
            "<div class='kpi-card'>"
            f"<div class='kpi-label'>{html.escape(str(label))}</div>"
            f"<div class='kpi-value'>{html.escape(str(value))}</div>"
            f"<div class='kpi-hint'>{html.escape(str(hint))}</div>"
            "</div>"
        )
    st.markdown(f"<div class='kpi-grid'>{''.join(cards)}</div>", unsafe_allow_html=True)


def pipeline_banner(steps: list[str]) -> None:
    parts: list[str] = []
    for i, step in enumerate(steps):
        if i:
            parts.append('<span class="pipe-arrow">→</span>')
        parts.append(f'<span class="pipe-step">{html.escape(step)}</span>')
    st.markdown(f'<div class="pipeline">{"".join(parts)}</div>', unsafe_allow_html=True)


def section_title(text: str) -> None:
    st.markdown(f'<div class="section-title">{html.escape(text)}</div>', unsafe_allow_html=True)


def _style_figure(fig, height: int = 360):
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=52, b=24, l=12, r=12),
        height=height,
        font=dict(family="Plus Jakarta Sans, Segoe UI, sans-serif", size=13),
        title=dict(font=dict(size=16)),
        hoverlabel=dict(font_size=13, namelength=-1),
        xaxis=dict(showgrid=False, zeroline=False),
        yaxis=dict(gridcolor="rgba(148,163,184,0.18)", zeroline=False),
    )
    return fig


def show_chart(fig, height: int = 360) -> None:
    st.plotly_chart(
        _style_figure(fig, height=height),
        width="stretch",
        theme="streamlit",
        config={"displayModeBar": False, "responsive": True},
    )


def sentiment_bar(counts: dict[str, int], title: str):
    frame = pd.DataFrame({"Sentiment": list(counts.keys()), "Count": list(counts.values())})
    fig = px.bar(
        frame,
        x="Sentiment",
        y="Count",
        color="Sentiment",
        color_discrete_map=SENTIMENT_COLORS,
        title=title,
        text="Count",
    )
    fig.update_traces(
        texttemplate="%{text:,}",
        textposition="outside",
        marker_line_width=0,
        hovertemplate="<b>%{x}</b><br>Count: %{y:,}<extra></extra>",
    )
    fig.update_layout(showlegend=False, yaxis_title=None, xaxis_title=None, bargap=0.38)
    return fig


def sentiment_pie(counts: dict[str, int], title: str):
    frame = pd.DataFrame({"Sentiment": list(counts.keys()), "Count": list(counts.values())})
    fig = px.pie(
        frame,
        names="Sentiment",
        values="Count",
        color="Sentiment",
        color_discrete_map=SENTIMENT_COLORS,
        title=title,
        hole=0.58,
    )
    fig.update_traces(
        textinfo="percent+label",
        pull=[0.03] * len(frame),
        hovertemplate="<b>%{label}</b><br>%{value:,} posts (%{percent})<extra></extra>",
    )
    fig.update_layout(showlegend=False)
    return fig


def mapreduce_sankey(reduce_results: dict[str, Any]):
    order = ["Positive", "Neutral", "Negative"]
    counts = [int((reduce_results.get(label) or {}).get("count", 0)) for label in order]
    labels = [
        "MAP  (sentiment, 1)",
        "Shuffle  Positive",
        "Shuffle  Neutral",
        "Shuffle  Negative",
        "REDUCE  Positive",
        "REDUCE  Neutral",
        "REDUCE  Negative",
    ]
    node_colors = [
        "#38bdf8",
        SENTIMENT_COLORS["Positive"],
        SENTIMENT_COLORS["Neutral"],
        SENTIMENT_COLORS["Negative"],
        SENTIMENT_COLORS["Positive"],
        SENTIMENT_COLORS["Neutral"],
        SENTIMENT_COLORS["Negative"],
    ]
    fig = go.Figure(
        data=[
            go.Sankey(
                arrangement="snap",
                node=dict(pad=18, thickness=20, label=labels, color=node_colors, line=dict(width=0)),
                link=dict(
                    source=[0, 0, 0, 1, 2, 3],
                    target=[1, 2, 3, 4, 5, 6],
                    value=counts + counts,
                    color=[
                        "rgba(52,211,153,0.35)",
                        "rgba(148,163,184,0.35)",
                        "rgba(248,113,113,0.35)",
                        "rgba(52,211,153,0.55)",
                        "rgba(148,163,184,0.55)",
                        "rgba(248,113,113,0.55)",
                    ],
                ),
            )
        ]
    )
    fig.update_layout(title_text="Map → Shuffle → Reduce data flow", font_size=13)
    return fig


def gauge_chart(title: str, value: float) -> go.Figure:
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=round(value * 100, 1),
            number={"suffix": "%"},
            title={"text": title, "font": {"size": 15}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 0},
                "bar": {"color": "#38bdf8", "thickness": 0.28},
                "bgcolor": "rgba(148,163,184,0.12)",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 40], "color": "rgba(248,113,113,0.18)"},
                    {"range": [40, 70], "color": "rgba(250,204,21,0.16)"},
                    {"range": [70, 100], "color": "rgba(52,211,153,0.18)"},
                ],
                "threshold": {"line": {"color": "#34d399", "width": 3}, "value": value * 100},
            },
        )
    )
    return fig


def live_result_card(result: dict[str, Any]) -> None:
    label = str(result["predicted_sentiment"])
    score = float(result["sentiment_score"])
    color = SENTIMENT_COLORS.get(label, "#94a3b8")
    fill = max(0.0, min(1.0, (score + 1) / 2)) * 100
    st.markdown(
        f"""
        <div class="result-card" style="background:{SENTIMENT_SOFT.get(label, 'rgba(148,163,184,0.12)')};
             border-color:{color}55;">
            <div class="kpi-label">Live VADER result</div>
            <div class="kpi-value" style="color:{color};">{html.escape(label)}</div>
            <div class="kpi-hint">Compound score {score:+.3f} · mapped with thresholds ±0.05</div>
            <div class="score-track">
                <div class="score-fill" style="width:{fill:.1f}%; background:{color};"></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def page_overview(db) -> None:
    page_header(
        "Sentiment Analysis for Social Media Data",
        "MongoDB storage, VADER scoring, a custom Python MapReduce pipeline, and live analytics — built as a complete academic demonstration.",
        "TweetEval · MongoDB · MapReduce",
    )
    pipeline_banner(["Dataset", "Clean", "MongoDB", "VADER", "Map", "Shuffle", "Reduce", "Evaluate"])

    posts = db[COLLECTION_SOCIAL_POSTS]
    total = posts.count_documents({})
    processed = posts.count_documents({"processed": True})
    latest = get_metadata("latest_mapreduce", database=db) or {}
    pct = f"{(processed / total * 100):.1f}%" if total else "0%"

    kpi_cards(
        [
            ("Total posts", f"{total:,}", "Loaded into social_posts"),
            ("Processed posts", f"{processed:,}", f"{pct} analysed by VADER"),
            ("Latest MapReduce", latest.get("analysis_id", "Not run"), "Unique analysis_id"),
            ("Records reduced", f"{latest.get('total_posts', 0):,}", f"{latest.get('duration_seconds', 0)} s runtime"),
        ]
    )

    section_title("Technology stack")
    stack = [
        ("Language", "Python"),
        ("Dataset", "TweetEval (Hugging Face)"),
        ("Database", "MongoDB + PyMongo"),
        ("NLP", "NLTK VADER"),
        ("MapReduce", "Custom Python"),
        ("Evaluation", "Scikit-learn"),
        ("Dashboard", "Streamlit + Plotly"),
        ("Config", "python-dotenv"),
    ]
    cards = "".join(
        f"<div class='tech-card'><div class='tech-k'>{html.escape(k)}</div>"
        f"<div class='tech-v'>{html.escape(v)}</div></div>"
        for k, v in stack
    )
    st.markdown(f"<div class='tech-grid'>{cards}</div>", unsafe_allow_html=True)

    if latest:
        section_title("Latest MapReduce execution")
        sentiments = latest.get("sentiments") or {}
        counts = {k: int((v or {}).get("count", 0)) for k, v in sentiments.items()}
        if counts:
            left, right = st.columns(2)
            with left:
                show_chart(sentiment_pie(counts, "Latest reduced sentiment"))
            with right:
                show_chart(sentiment_bar(counts, "Latest class counts"))
        with st.expander("Execution metadata", expanded=False):
            st.json(
                {
                    "analysis_id": latest.get("analysis_id"),
                    "total_posts": latest.get("total_posts"),
                    "duration_seconds": latest.get("duration_seconds"),
                    "created_at": latest.get("created_at"),
                    "sentiments": latest.get("sentiments"),
                }
            )
    else:
        empty_state("No MapReduce run has been stored yet.", "python run_pipeline.py --step mapreduce")


def page_dataset(db) -> None:
    page_header(
        "Dataset overview",
        "Inspect TweetEval posts stored in MongoDB: size, splits, actual labels, and sample rows.",
        "social_posts collection",
    )
    posts = db[COLLECTION_SOCIAL_POSTS]
    total = posts.count_documents({})
    if total == 0:
        empty_state("No social media posts are in MongoDB yet.", "python run_pipeline.py --step load")
        return

    sample = list(posts.find({}, {"_id": 0, "created_at": 0, "updated_at": 0}).limit(25))
    frame = pd.DataFrame(sample)
    missing = {
        "original_text": posts.count_documents({"original_text": {"$in": [None, ""]}}),
        "cleaned_text": posts.count_documents({"cleaned_text": {"$in": [None, ""]}}),
        "actual_sentiment": posts.count_documents({"actual_sentiment": {"$in": [None, ""]}}),
    }
    pipeline_text_dup = [
        {"$group": {"_id": "$original_text", "n": {"$sum": 1}}},
        {"$match": {"n": {"$gt": 1}}},
        {"$count": "duplicates"},
    ]
    dup_result = list(posts.aggregate(pipeline_text_dup))
    duplicate_texts = dup_result[0]["duplicates"] if dup_result else 0
    actual_counts = {
        row["_id"]: row["count"]
        for row in posts.aggregate([{"$group": {"_id": "$actual_sentiment", "count": {"$sum": 1}}}])
        if row["_id"]
    }
    split_counts = {
        row["_id"]: row["count"]
        for row in posts.aggregate([{"$group": {"_id": "$dataset_split", "count": {"$sum": 1}}}])
        if row["_id"]
    }

    kpi_cards(
        [
            ("Dataset size", f"{total:,}", "Train + validation + test"),
            ("Duplicate texts", f"{duplicate_texts:,}", "Repeated original_text"),
            ("Empty original_text", missing["original_text"], "Missing source posts"),
            ("Splits", len(split_counts), ", ".join(sorted(split_counts)) or "—"),
        ]
    )

    left, right = st.columns(2)
    with left:
        show_chart(sentiment_bar(actual_counts, "Actual sentiment distribution"))
    with right:
        show_chart(sentiment_pie(split_counts, "Dataset split distribution"))

    section_title("Sample social media posts")
    display_cols = [
        c
        for c in ["post_id", "original_text", "cleaned_text", "actual_sentiment", "dataset_split", "processed"]
        if c in frame.columns
    ]
    st.dataframe(frame[display_cols], width="stretch", hide_index=True, height=360)
    with st.expander("Missing-data statistics"):
        st.json(missing)


def page_sentiment(db) -> None:
    page_header(
        "Sentiment analysis",
        "Search processed posts, filter by VADER label, and try the same engine on custom text.",
        "NLTK VADER · compound thresholds ±0.05",
    )
    posts = db[COLLECTION_SOCIAL_POSTS]
    processed = posts.count_documents({"processed": True})
    if processed == 0:
        empty_state("No posts have been analysed yet.", "python run_pipeline.py --step sentiment")
        return

    predicted_counts = {
        row["_id"]: row["count"]
        for row in posts.aggregate(
            [{"$match": {"processed": True}}, {"$group": {"_id": "$predicted_sentiment", "count": {"$sum": 1}}}]
        )
        if row["_id"]
    }
    kpi_cards(
        [
            ("Processed", f"{processed:,}", "Marked processed=true"),
            ("Predicted positive", f"{predicted_counts.get('Positive', 0):,}", "compound ≥ 0.05"),
            ("Predicted neutral", f"{predicted_counts.get('Neutral', 0):,}", "between thresholds"),
            ("Predicted negative", f"{predicted_counts.get('Negative', 0):,}", "compound ≤ -0.05"),
        ]
    )

    col_f1, col_f2 = st.columns([2, 1])
    query_text = col_f1.text_input("Search cleaned text", placeholder="e.g. love, terrible, service")
    sentiment_filter = col_f2.selectbox("Predicted sentiment", ["All", "Positive", "Negative", "Neutral"])

    query: dict[str, Any] = {"processed": True}
    if sentiment_filter != "All":
        query["predicted_sentiment"] = sentiment_filter
    if query_text.strip():
        query["cleaned_text"] = {"$regex": query_text.strip(), "$options": "i"}

    rows = list(
        posts.find(
            query,
            {
                "_id": 0,
                "post_id": 1,
                "cleaned_text": 1,
                "predicted_sentiment": 1,
                "sentiment_score": 1,
                "actual_sentiment": 1,
                "dataset_split": 1,
            },
        ).limit(200)
    )
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True, height=380)

    section_title("Live sentiment check")
    st.caption("Uses the same VADER engine as the pipeline. This does not write to MongoDB.")
    custom = st.text_area("Enter a social media post", placeholder="I love this new product!")
    if st.button("Analyse text", type="primary") and custom.strip():
        result = analyse_text(custom.strip())
        live_result_card(result)
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Predicted", result["predicted_sentiment"])
        m2.metric("Compound", f"{result['sentiment_score']:.3f}")
        m3.metric("Positive", f"{result['positive_score']:.3f}")
        m4.metric("Negative", f"{result['negative_score']:.3f}")
        show_chart(
            gauge_chart("Compound mapped to 0–100", (float(result["sentiment_score"]) + 1) / 2),
            height=280,
        )


def page_mapreduce(db) -> None:
    page_header(
        "MapReduce visualization",
        "The academic core: mapper, shuffle/group, and reducer run in Python. MongoDB only stores posts and the aggregated output.",
        "Custom Python MapReduce",
    )
    pipeline_banner(["MAP (sentiment, 1)", "SHUFFLE & GROUP", "REDUCE counts + %"])

    latest_id = get_metadata("latest_analysis_id", database=db)
    execution = None
    if latest_id:
        execution = db[COLLECTION_PROCESSING_RESULTS].find_one(
            {"analysis_id": latest_id, "record_type": "mapreduce_execution"}
        )
    if not execution:
        empty_state("No MapReduce execution metadata is available yet.", "python run_pipeline.py --step mapreduce")
        return

    kpi_cards(
        [
            ("Analysis ID", execution.get("analysis_id"), "Unique run identifier"),
            ("Records processed", f"{execution.get('total_records_processed', 0):,}", "Processed social posts"),
            ("Duration", f"{execution.get('duration_seconds')} s", "Engine wall time"),
            ("Map pairs", f"{execution.get('map_pair_count', 0):,}", "Intermediate key-value pairs"),
        ]
    )

    map_sample = execution.get("map_sample") or []
    shuffle_sample = execution.get("shuffle_sample") or {}
    shuffle_sizes = execution.get("shuffle_group_sizes") or {}
    reduce_results = execution.get("reduce_results") or {}
    counts = {k: int(v.get("count", 0)) for k, v in reduce_results.items()}

    section_title("Pipeline flow")
    show_chart(mapreduce_sankey(reduce_results), height=430)

    c1, c2, c3 = st.columns(3)
    with c1:
        with st.container(border=True):
            st.subheader("1. MAP")
            st.caption("Each processed post emits a key-value pair `(sentiment, 1)`.")
            sample_keys = [item.get("key") or "Unknown" for item in map_sample]
            if sample_keys:
                sample_counts = pd.Series(sample_keys, name="Sentiment").value_counts().to_dict()
                show_chart(sentiment_bar(sample_counts, "Sample mapper output"), height=280)
            st.code(
                "\n".join(f"({item.get('key')}, {item.get('value')})" for item in map_sample[:12])
                or "(no sample pairs)",
                language="text",
            )
    with c2:
        with st.container(border=True):
            st.subheader("2. SHUFFLE AND GROUP")
            st.caption("All values that share a key are collected into a list.")
            group_counts = {key: int(size) for key, size in shuffle_sizes.items()}
            if not group_counts:
                group_counts = {key: len(values) for key, values in shuffle_sample.items()}
            if group_counts:
                show_chart(sentiment_bar(group_counts, "Grouped list sizes"), height=280)
            lines = []
            for key, values in shuffle_sample.items():
                shown = ",".join(str(v) for v in values[:8])
                extra = shuffle_sizes.get(key, len(values))
                suffix = f" … ×{extra:,}" if extra > len(values) else ""
                lines.append(f"{key} → [{shown}]{suffix}")
            st.code("\n".join(lines) or "(no grouped lists)", language="text")
    with c3:
        with st.container(border=True):
            st.subheader("3. REDUCE")
            st.caption("Each list is summed to a count. Percentages use the total.")
            if counts:
                show_chart(sentiment_pie(counts, "Reduced sentiment share"), height=280)
            lines = [
                f"{key} → {payload.get('count', 0):,} ({payload.get('percentage', 0)}%)"
                for key, payload in reduce_results.items()
            ]
            st.code("\n".join(lines) or "(no reduce output)", language="text")

    section_title("Map sample with source text")
    if map_sample:
        st.dataframe(pd.DataFrame(map_sample), width="stretch", hide_index=True, height=280)

    section_title("Final sentiment counts")
    left, right = st.columns(2)
    with left:
        show_chart(sentiment_bar(counts, "Reduced sentiment counts"))
    with right:
        show_chart(sentiment_pie(counts, "Reduced sentiment share"))


def page_analytics(db) -> None:
    page_header(
        "Analytics",
        "Aggregated Positive, Negative, and Neutral counts from the latest MapReduce run in MongoDB.",
        "mapreduce_results collection",
    )
    latest_id = get_metadata("latest_analysis_id", database=db)
    if not latest_id:
        empty_state("Analytics are populated from MongoDB MapReduce results.", "python run_pipeline.py --step mapreduce")
        return

    rows = list(db[COLLECTION_MAPREDUCE_RESULTS].find({"analysis_id": latest_id}, {"_id": 0}))
    if not rows:
        empty_state("MapReduce result documents were not found for the latest run.", "python run_pipeline.py --step mapreduce")
        return

    frame = pd.DataFrame(rows)
    counts = {row["sentiment"]: int(row["count"]) for row in rows}
    total = int(rows[0].get("total_posts") or sum(counts.values()))
    pos = counts.get("Positive", 0)
    neg = counts.get("Negative", 0)
    neu = counts.get("Neutral", 0)

    kpi_cards(
        [
            ("Positive", f"{pos:,}", f"{(pos / total * 100) if total else 0:.2f}% of posts"),
            ("Negative", f"{neg:,}", f"{(neg / total * 100) if total else 0:.2f}% of posts"),
            ("Neutral", f"{neu:,}", f"{(neu / total * 100) if total else 0:.2f}% of posts"),
            ("Total", f"{total:,}", "From latest analysis_id"),
        ]
    )

    left, right = st.columns(2)
    with left:
        show_chart(sentiment_pie(counts, "Sentiment percentages"))
    with right:
        show_chart(sentiment_bar(counts, "Sentiment counts"))

    pct_frame = pd.DataFrame(
        {
            "Sentiment": list(counts.keys()),
            "Percentage": [round((c / total) * 100, 2) if total else 0 for c in counts.values()],
        }
    )
    fig = px.bar(
        pct_frame,
        x="Sentiment",
        y="Percentage",
        color="Sentiment",
        color_discrete_map=SENTIMENT_COLORS,
        title="Sentiment percentage distribution",
        text="Percentage",
    )
    fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside", marker_line_width=0)
    fig.update_layout(showlegend=False, height=360, bargap=0.38)
    show_chart(fig)

    section_title("Raw MapReduce documents")
    st.dataframe(frame, width="stretch", hide_index=True)


def page_evaluation(db) -> None:
    page_header(
        "Model evaluation",
        "VADER predictions compared with TweetEval actual_sentiment using multi-class sklearn metrics.",
        "evaluation_results collection",
    )
    latest_id = get_metadata("latest_evaluation_id", database=db)
    document = None
    if latest_id:
        document = db[COLLECTION_EVALUATION_RESULTS].find_one({"evaluation_id": latest_id}, {"_id": 0})
    if not document:
        empty_state("No evaluation results are stored in MongoDB yet.", "python run_pipeline.py --step evaluate")
        return

    kpi_cards(
        [
            ("Accuracy", f"{document.get('accuracy', 0):.3f}", "Overall correct labels"),
            ("Precision", f"{document.get('precision', 0):.3f}", "Macro average"),
            ("Recall", f"{document.get('recall', 0):.3f}", "Macro average"),
            ("F1 score", f"{document.get('f1_score', 0):.3f}", "Macro average"),
        ]
    )
    st.caption(
        f"Evaluation ID `{document.get('evaluation_id')}` · "
        f"{document.get('n_samples', 0):,} labelled posts. "
        "VADER is a lexicon method, so scores below 80% against TweetEval labels are expected."
    )

    g1, g2, g3, g4 = st.columns(4)
    with g1:
        show_chart(gauge_chart("Accuracy", float(document.get("accuracy", 0))), height=250)
    with g2:
        show_chart(gauge_chart("Precision", float(document.get("precision", 0))), height=250)
    with g3:
        show_chart(gauge_chart("Recall", float(document.get("recall", 0))), height=250)
    with g4:
        show_chart(gauge_chart("F1 score", float(document.get("f1_score", 0))), height=250)

    report = document.get("classification_report") or {}
    report_rows = []
    for label, values in report.items():
        if not isinstance(values, dict):
            report_rows.append({"label": label, "precision": None, "recall": None, "f1-score": None, "support": values})
            continue
        report_rows.append({"label": label, **values})
    if report_rows:
        section_title("Classification report")
        st.dataframe(pd.DataFrame(report_rows), width="stretch", hide_index=True)

    matrix = document.get("confusion_matrix") or []
    labels = document.get("labels") or ["Negative", "Neutral", "Positive"]
    if matrix:
        section_title("Confusion matrix")
        heat = pd.DataFrame(matrix, index=[f"Actual {item}" for item in labels], columns=[f"Pred {item}" for item in labels])
        fig = px.imshow(
            heat,
            text_auto=True,
            color_continuous_scale="Tealgrn",
            aspect="auto",
            title="Actual vs predicted",
        )
        fig.update_layout(height=420)
        show_chart(fig, height=420)

    png = CONFUSION_MATRIX_PNG
    if png.exists():
        with st.expander("Saved confusion-matrix image"):
            st.image(str(png.resolve()), caption="results/confusion_matrix.png", width="stretch")

    section_title("Actual vs predicted counts")
    posts = db[COLLECTION_SOCIAL_POSTS]
    actual = {
        row["_id"]: row["count"]
        for row in posts.aggregate(
            [{"$match": {"processed": True}}, {"$group": {"_id": "$actual_sentiment", "count": {"$sum": 1}}}]
        )
        if row["_id"]
    }
    predicted = {
        row["_id"]: row["count"]
        for row in posts.aggregate(
            [{"$match": {"processed": True}}, {"$group": {"_id": "$predicted_sentiment", "count": {"$sum": 1}}}]
        )
        if row["_id"]
    }
    compare = pd.DataFrame(
        {
            "Sentiment": ["Positive", "Negative", "Neutral"],
            "Actual": [actual.get("Positive", 0), actual.get("Negative", 0), actual.get("Neutral", 0)],
            "Predicted": [predicted.get("Positive", 0), predicted.get("Negative", 0), predicted.get("Neutral", 0)],
        }
    )
    fig = px.bar(
        compare.melt(id_vars="Sentiment", var_name="Source", value_name="Count"),
        x="Sentiment",
        y="Count",
        color="Source",
        barmode="group",
        title="Actual labels vs VADER predictions",
        color_discrete_map={"Actual": "#38bdf8", "Predicted": "#a78bfa"},
        text="Count",
    )
    fig.update_traces(texttemplate="%{text:,}", textposition="outside", marker_line_width=0)
    fig.update_layout(height=400, bargap=0.28)
    show_chart(fig, height=400)


def main() -> None:
    st.sidebar.markdown(
        f"""
        <div class="brand">
            <div class="brand-kicker">Academic project</div>
            <div class="brand-title">Sentiment MapReduce</div>
            <div class="status-pill"><span class="status-dot"></span> MongoDB {html.escape(MONGODB_DATABASE)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    page = st.sidebar.radio(
        "Go to",
        NAV_PAGES,
        format_func=lambda name: f"{NAV_ICONS.get(name, '•')}  {name}",
        label_visibility="collapsed",
    )
    st.sidebar.markdown("---")
    st.sidebar.caption("Live data from MongoDB. Run `python run_pipeline.py --all` if a page is empty.")
    st.sidebar.caption("Motion powered by [GSAP](https://gsap.com/).")

    db = get_db()
    pages = {
        "Project overview": page_overview,
        "Dataset overview": page_dataset,
        "Sentiment analysis": page_sentiment,
        "MapReduce visualization": page_mapreduce,
        "Analytics": page_analytics,
        "Model evaluation": page_evaluation,
    }
    if db is not None:
        pages[page](db)
    inject_gsap()


if __name__ == "__main__":
    main()
