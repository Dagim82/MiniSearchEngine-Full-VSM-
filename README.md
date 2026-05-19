# Educational Note Search Engine

![Python](https://img.shields.io/badge/Python-3.14-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-orange.svg)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-green.svg)
![Information Retrieval](https://img.shields.io/badge/Information%20Retrieval-IR-purple.svg)

An academic Information Retrieval (IR) system for searching, retrieving, and ranking educational documents such as lecture notes, tutorials, research write-ups, and study resources. This repository demonstrates practical IR techniques using TF–IDF vectorization and Cosine Similarity, wrapped in an interactive Streamlit dashboard for exploration, evaluation, and demonstration.

## Table of Contents

- [Project Description](#project-description)
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Project Structure](#project-structure)
- [System Workflow](#system-workflow)
- [Evaluation Metrics](#evaluation-metrics)
- [Installation](#installation)
- [Usage](#usage)
- [Screenshots](#screenshots)
- [Challenges and Solutions](#challenges-and-solutions)
- [Future Improvements](#future-improvements)
- [Conclusion](#conclusion)
- [Acknowledgements](#acknowledgements)

## Project Description

The Educational Note Search Engine is an academic IR prototype tailored for educational content. It supports indexing and searching collections of lecture notes and study materials. The retrieval pipeline uses TF–IDF vectorization to represent documents and user queries, and Cosine Similarity to compute relevance scores and rank results.

The project is implemented in Python and leverages familiar scientific and ML tooling (Pandas, NumPy, Scikit-learn), with an interactive front-end built in Streamlit so students and researchers can experiment with retrieval settings and view evaluation metrics live.

## Features

- Educational note search: find relevant lecture notes and study resources quickly.
- Ranked document retrieval using TF–IDF and Cosine Similarity.
- Search history tracking and simple query inspection.
- Interactive Streamlit dashboard for parameter tuning and visualizations.
- Evaluation metrics visualization for benchmarking retrieval performance.
- Predefined standard queries and the ability to run custom queries.
- Responsive, education-focused UI for demonstrations and presentations.

## Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core implementation, scripting |
| Pandas | Data manipulation and CSV handling |
| NumPy | Numerical operations |
| Scikit-learn | `TfidfVectorizer` and similarity functions |
| Streamlit | Interactive web dashboard and UI |
| TF–IDF | Term weighting / vectorization model |
| Cosine Similarity | Document-query similarity scoring |

## Project Structure

Typical repository layout and responsibilities:

- `app.py` → Main Streamlit application and UI glue code.
- `ir_system.py` → Retrieval and ranking logic: indexing, search, result formatting, and schema normalization.
- `preprocess.py` → Text cleaning, tokenization and lemmatization utilities.
- `evaluation.py` → Metric calculations and benchmarking helpers (Precision@k, Recall, F1, etc.).
- `benchmark_data.py` → Loaders for `queries.csv` and graded relevance judgments (`relevance.csv`).
- `docs/` → Educational documents and note collections used as the retrieval corpus.
- `queries.csv` → Predefined queries used for benchmarking and demos.
- `relevance.csv` → Graded qrels (relevance judgments) for evaluation experiments.
- `requirements.txt` → Python dependencies for reproducibility.
- `results/` → (optional) Generated evaluation outputs, charts and CSV exports.
- `preprocessing/` → (optional) Extended cleaning and dataset-prep scripts.
- `reports/` → (optional) Generated PDF/HTML reports for experiments and presentations.

> Note: Some optional folders (like `results/`, `preprocessing/`, `reports/`) are suggested locations for outputs and extended tooling. Adjust the layout for your needs.

## System Workflow

1. Load documents from `docs/` (or other configured dataset locations).
2. Preprocess text: cleaning, lowercasing, stopword removal, lemmatization (via `preprocess.py`).
3. Generate TF–IDF vectors for the corpus using `TfidfVectorizer` (Scikit-learn).
4. Accept user queries from the Streamlit UI (predefined or custom).
5. Vectorize the query with the same TF–IDF vocabulary.
6. Calculate Cosine Similarity between the query vector and document vectors.
7. Rank documents by similarity score and display results in the dashboard.

## Evaluation Metrics

The project includes standard retrieval metrics to evaluate and demonstrate system behavior:

- Precision@5 — fraction of relevant documents in the top 5 results.
- Precision@10 — fraction of relevant documents in the top 10 results.
- Recall — fraction of relevant documents retrieved out of all relevant for the query.
- F1-Score — harmonic mean of precision and recall.
- Accuracy — proportion of correctly classified relevant/non-relevant results (used for binary thresholds in demos).

Each metric is explained visually in the dashboard and can be computed across the predefined query set using the graded `relevance.csv` qrels.

## Installation

Clone the repository and install dependencies (recommended inside a virtual environment):

```bash
pip install -r requirements.txt
```

Run the Streamlit app locally:

```bash
streamlit run app.py
```

If you use a virtual environment, activate it first. On Windows PowerShell:

```powershell
& ".venv\Scripts\Activate.ps1"
pip install -r requirements.txt
streamlit run app.py
```

## Usage

- Open the dashboard at the URL printed by Streamlit (typically `http://localhost:8501`).
- Try a predefined query from the sidebar or enter a custom phrase.
- Adjust TF–IDF settings (e.g., `max_features`, `ngram_range`) and re-run the search.
- Use the benchmarking controls to run evaluation across the standard queries and visualize metrics.

## Screenshots

> Add high-quality screenshots to `assets/` and update the links below. These placeholders show recommended views for documentation and presentation.

- Dashboard (search view):

![Dashboard - Search View](assets/screenshot_search.png)

- Results and Evaluation:

![Dashboard - Results](assets/screenshot_results.png)

- Benchmark Summary:

![Dashboard - Benchmark](assets/screenshot_benchmark.png)

## Challenges and Solutions

- DataFrame column mismatch issues
  - Problem: legacy code and different modules expected inconsistent column names (e.g., `Similarity Score` vs `score`).
  - Solution: schema normalization helpers in `ir_system.py` and defensive CSV column cleaning in `benchmark_data.py`.

- Formatting problems in console and UI
  - Problem: display formatting assumed specific DataFrame columns and produced KeyErrors when names differed.
  - Solution: robust column normalization and safe display fallbacks to avoid runtime crashes.

- Debugging the retrieval pipeline
  - Problem: subtle bugs introduced by cached instances and stale references when developing iteratively.
  - Solution: prefer class/static utility functions for frequently changing helpers and restart the app after code updates during development.

## Future Improvements

- Semantic search using transformer-based encoders (e.g., Sentence-BERT) for improved relevance.
- Add PDF/DOCX ingestion with OCR support for scanned lecture notes.
- Implement advanced ranking algorithms (BM25, learning-to-rank).
- Add user authentication, personalized query history, and saved-result collections.
- Deploy online with Docker + CI/CD for reproducible demos and public access.

## Conclusion

This project demonstrates practical Information Retrieval techniques applied to educational content. It is suitable for final-year projects, research demonstrations, and classroom labs. The Streamlit dashboard provides a hands-on environment to experiment with retrieval settings and visualise performance metrics.

## Acknowledgements

- Built with Scikit-learn, Pandas, NumPy and Streamlit.
- Inspiration and design patterns from classic IR curricula and educational demos.

---

If you'd like, I can:

- Add real screenshots in `assets/` and link them in this README.
- Remove debug prints across the codebase and prepare a release-ready branch.
- Create a short `DEMO.md` with step-by-step presentation notes.
