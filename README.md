# Sentiment Analysis for Social Media Data using MongoDB and MapReduce

An academic end-to-end system that downloads the TweetEval sentiment dataset, cleans social media text, stores posts in MongoDB, predicts sentiment with NLTK VADER, aggregates class counts through a **custom Python MapReduce** pipeline (Map → Shuffle/Group → Reduce), evaluates predictions against original labels, and visualises everything in a Streamlit dashboard.

---

## 1. Project Title

**Sentiment Analysis for Social Media Data using MongoDB and MapReduce**

---

## 2. Project Description

Social media posts are short, noisy, and opinion-rich. This project turns the public [TweetEval sentiment](https://huggingface.co/datasets/cardiffnlp/tweet_eval) corpus into a reproducible analytics pipeline:

1. Extract train / validation / test splits and save a raw CSV.
2. Explore class balance, missing values, duplicates, and text lengths.
3. Clean text without stripping sentiment-bearing tokens such as emojis.
4. Load documents into MongoDB with batch inserts and duplicate protection.
5. Score each post with VADER (compound score thresholds).
6. Run an explicit MapReduce job in Python and store aggregated counts.
7. Compare predicted labels with TweetEval’s `actual_sentiment`.
8. Serve interactive charts and a MapReduce stage walkthrough in Streamlit.

The **actual sentiment label is retained only for evaluation**. It is never passed into the VADER predictor.

---

## 3. Problem Statement

Manually reading thousands of posts cannot produce a reliable picture of public opinion. The project addresses three academic goals at once:

- Store and query social media documents in a document database (MongoDB).
- Classify each post as Positive, Negative, or Neutral using a transparent lexicon model (VADER).
- Demonstrate the MapReduce programming model in Python rather than hiding aggregation inside a single MongoDB query.

---

## 4. Objectives

- Fetch and convert the TweetEval sentiment dataset to CSV.
- Preprocess noisy social media text in a modular, reusable way.
- Persist posts, processing metadata, MapReduce output, and evaluation results in MongoDB.
- Implement Mapper, Shuffle/Group, and Reducer as separate, testable modules.
- Report accuracy, precision, recall, F1, a classification report, and a confusion matrix.
- Provide a six-page Streamlit dashboard driven by live MongoDB data.

---

## 5. Features

- Hugging Face `datasets` download of all TweetEval sentiment splits.
- Raw and cleaned CSV artifacts under `data/`.
- Batch MongoDB inserts with unique `post_id` (safe to re-run).
- VADER batch scoring with configurable thresholds (`≥ 0.05` / `≤ -0.05`).
- Explicit MapReduce engine with `analysis_id`, duration, and stage samples for visualisation.
- Scikit-learn multi-class evaluation written to `results/` and MongoDB.
- Streamlit pages: overview, dataset, sentiment (including live text analysis), MapReduce visualisation, analytics, evaluation.
- Pytest coverage for cleaning, sentiment rules, mapper, shuffle, reducer, and mongomock database operations.

---

## 6. Technology Stack

| Layer | Technology |
| --- | --- |
| Language | Python 3.10+ |
| Dataset | TweetEval sentiment (`tweet_eval` / `sentiment`) |
| Dataset library | Hugging Face `datasets` |
| Data format | CSV |
| Database | MongoDB |
| Driver | PyMongo |
| NLP | NLTK VADER |
| Tables / arrays | Pandas, NumPy |
| Evaluation | Scikit-learn, Matplotlib |
| Dashboard | Streamlit, Plotly |
| Config | python-dotenv |
| Tests | pytest, mongomock |

---

## 7. System Architecture

```text
TweetEval Social Media Dataset
            │
            ▼
Dataset Extraction using Python
            │
            ▼
CSV Dataset Creation
            │
            ▼
Dataset Exploration
            │
            ▼
Data Preprocessing
            │
            ▼
MongoDB Data Storage
            │
            ▼
Sentiment Analysis (VADER)
            │
            ▼
MAP PHASE  (Sentiment, 1)
            │
            ▼
SHUFFLE AND GROUP
            │
            ▼
REDUCE PHASE  (counts + percentages)
            │
            ▼
Store Aggregated Results in MongoDB
            │
            ▼
Model Evaluation
            │
            ▼
Streamlit Dashboard
```

Hybrid data flow:

```text
MongoDB  →  store posts
Python   →  VADER sentiment
Python   →  custom MapReduce (not MongoDB aggregation)
MongoDB  →  store aggregated + evaluation results
Streamlit →  read MongoDB and visualise
```

---

## 8. Project Folder Structure

```text
.
├── data/
│   ├── raw/social_media_raw.csv          (generated)
│   └── processed/social_media_cleaned.csv (generated)
├── src/
│   ├── config.py
│   ├── logging_setup.py
│   ├── data_collection/
│   │   ├── download_dataset.py
│   │   └── explore_dataset.py
│   ├── preprocessing/preprocess.py
│   ├── database/
│   │   ├── mongodb_connection.py
│   │   └── load_data.py
│   ├── sentiment_analysis/sentiment_analyzer.py
│   ├── mapreduce/
│   │   ├── mapper.py
│   │   ├── shuffle.py
│   │   ├── reducer.py
│   │   └── mapreduce_engine.py
│   └── evaluation/evaluate_model.py
├── dashboard/app.py
├── notebooks/dataset_exploration.ipynb
├── results/                              (generated metrics and plots)
├── tests/
├── run_pipeline.py
├── requirements.txt
├── .env.example
└── README.md
```

---

## 9. Installation Instructions

```powershell
cd d:\mdb
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

On macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## 10. Environment Setup

Copy the example file and edit if needed. **Do not commit `.env`.**

```powershell
copy .env.example .env
```

```env
MONGODB_URI=mongodb://localhost:27017
MONGODB_DATABASE=sentiment_analysis_db
```

Optional:

```env
BATCH_SIZE=1000
MAPREDUCE_SAMPLE_SIZE=20
```

---

## 11. MongoDB Setup

This project defaults to a **local** MongoDB server.

1. Install [MongoDB Community Server](https://www.mongodb.com/try/download/community).
2. Start the service so it listens on `mongodb://localhost:27017`.
3. Confirm connectivity (optional): `mongosh` then `db.runCommand({ ping: 1 })`.

The pipeline creates database `sentiment_analysis_db` and collections:

- `social_posts`
- `processing_results`
- `mapreduce_results`
- `evaluation_results`
- `project_metadata`

Indexes include unique `post_id`, plus `actual_sentiment`, `predicted_sentiment`, and `processed`.

---

## 12. Dataset Download Instructions

The download step needs internet access to Hugging Face.

```powershell
python run_pipeline.py --step download
```

This writes `data/raw/social_media_raw.csv` with columns:

```text
post_id, text, actual_sentiment, split
```

Label mapping:

| Numeric label | Readable label |
| --- | --- |
| 0 | Negative |
| 1 | Neutral |
| 2 | Positive |

TweetEval sentiment is on the order of ~60,000 posts across train, validation, and test. The first download may take several minutes.

---

## 13. How to Run the Pipeline

Full run (download → explore → preprocess → load → sentiment → mapreduce → evaluate):

```powershell
python run_pipeline.py --all
```

Individual stages:

```powershell
python run_pipeline.py --step download
python run_pipeline.py --step explore
python run_pipeline.py --step preprocess
python run_pipeline.py --step load
python run_pipeline.py --step sentiment
python run_pipeline.py --step mapreduce
python run_pipeline.py --step evaluate
```

Re-analyse posts that are already marked `processed`:

```powershell
python run_pipeline.py --step sentiment --force
```

Re-running `--step load` does **not** duplicate documents: existing `post_id` values are skipped.

You can also invoke modules directly, for example:

```powershell
python src\data_collection\download_dataset.py
python src\preprocessing\preprocess.py
```

---

## 14. How to Run Streamlit

With MongoDB running and the pipeline completed (or at least the stages you want to inspect):

```powershell
streamlit run dashboard/app.py
```

Open the URL printed in the terminal (typically `http://localhost:8501`).

Sidebar pages:

1. Project overview  
2. Dataset overview  
3. Sentiment analysis (search, filter, live VADER box)  
4. MapReduce visualisation (Map / Shuffle / Reduce)  
5. Analytics (counts from `mapreduce_results`)  
6. Model evaluation  

If a collection is empty, the page explains which pipeline command to run. Values are not hardcoded.

---

## 15. MapReduce Explanation

MongoDB aggregation is **not** used as a substitute for this module. The engine in `src/mapreduce/` is deliberately split into three files.

### Map

Each processed post becomes a key-value pair:

```text
("I love this product", predicted=Positive)  →  (Positive, 1)
("This service is terrible", predicted=Negative) → (Negative, 1)
```

### Shuffle and Group

Pairs with the same key are collected:

```text
Positive → [1, 1, ...]
Negative → [1, ...]
Neutral  → [1, ...]
```

### Reduce

Lists are summed; percentages use the total number of mapped posts:

```text
Positive → count, percentage
Negative → count, percentage
Neutral  → count, percentage
```

Each execution stores:

- `analysis_id` (unique run identifier)
- timestamps and duration
- one document per sentiment in `mapreduce_results`
- map/shuffle **samples** in `processing_results` so the dashboard can show the three stages without dumping tens of thousands of ones into the UI

---

## 16. Screenshots

Add screenshots here after a local demo (faculty / viva). Suggested captures:

- `docs/screenshots/01-overview.png` — metric cards and stack  
- `docs/screenshots/02-dataset.png` — split and actual-sentiment charts  
- `docs/screenshots/03-sentiment.png` — filtered posts + live analyser  
- `docs/screenshots/04-mapreduce.png` — Map / Shuffle / Reduce columns  
- `docs/screenshots/05-analytics.png` — pie and bar charts  
- `docs/screenshots/06-evaluation.png` — confusion matrix and metrics  

*(Placeholders — insert images after you run the dashboard.)*

---

## 17. Evaluation Metrics

Predicted VADER labels are compared with TweetEval `actual_sentiment` using Scikit-learn:

- Accuracy  
- Precision, Recall, F1 (macro — primary dashboard numbers)  
- Weighted precision / recall / F1 (also stored)  
- Per-class classification report  
- Confusion matrix (`results/confusion_matrix.png`)

Artifacts:

```text
results/sentiment_results.csv
results/evaluation_results.json
results/classification_report.json
results/confusion_matrix.png
results/exploration_results.json
```

**Expected academic note:** VADER is a general English lexicon. TweetEval tweets are informal and often sarcastic, so accuracy is typically moderate rather than near-perfect. The evaluation page exists to measure that gap, not to hide it.

---

## 18. Future Enhancements

- Swap VADER for a transformer sentiment classifier while keeping the same MapReduce and MongoDB layers.
- Add per-split evaluation (train vs validation vs test).
- Stream MapReduce over MongoDB cursors in smaller chunks for machines with limited RAM.
- Deploy MongoDB Atlas and parameterise `MONGODB_URI` for cloud demos.
- Export a PDF report of the latest `analysis_id` for submission folders.
- Add user authentication if the dashboard is hosted beyond localhost.

---

## Tests

```powershell
pytest tests/
```

No live MongoDB is required for the unit tests (`mongomock` covers insert/index behaviour).

---

## Limitations

- Custom MapReduce runs in a single Python process (not Hadoop / Spark). That is intentional for the course demonstration.
- Full dataset download requires Hugging Face network access.
- Local MongoDB must be installed and running before `--step load` and the dashboard.
- Intermediate MapReduce lists of size ~60k are summarised for storage; full counts still come from the Reduce output.

---

## License / academic use

Intended as a student / faculty demonstration project. TweetEval is provided by Cardiff NLP / Hugging Face under the dataset’s own terms.
