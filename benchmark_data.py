"""Cached loaders for benchmark queries and relevance judgments.

The project uses CSV files instead of Python modules for the benchmark data so
queries and qrels can be edited without changing code. These helpers keep the
I/O cached for Streamlit responsiveness.
"""

from pathlib import Path

import pandas as pd
import streamlit as st


BASE_DIR = Path(__file__).resolve().parent
QUERIES_CSV = BASE_DIR / "queries.csv"
RELEVANCE_CSV = BASE_DIR / "relevance.csv"


@st.cache_data(show_spinner=False)
def load_queries(csv_path=None):
    """Load the predefined benchmark query list from CSV."""
    path = Path(csv_path) if csv_path else QUERIES_CSV
    if not path.exists():
        return []

    frame = pd.read_csv(path)
    # normalize column names: strip spaces and lowercase
    try:
        frame.columns = frame.columns.str.strip().str.lower()
    except Exception:
        frame.columns = [c.strip().lower() for c in frame.columns]
    print("Loaded queries CSV columns:", frame.columns.tolist())

    if "query" not in frame.columns:
        print("Missing column: query in queries CSV")
        return []

    return [str(value).strip() for value in frame["query"].dropna().tolist() if str(value).strip()]


@st.cache_data(show_spinner=False)
def load_relevance_judgments(csv_path=None):
    """Load human relevance judgments from CSV into a query->doc_ids mapping."""
    path = Path(csv_path) if csv_path else RELEVANCE_CSV
    if not path.exists():
        return {}

    frame = pd.read_csv(path)
    # normalize column names: strip spaces and lowercase
    try:
        frame.columns = frame.columns.str.strip().str.lower()
    except Exception:
        frame.columns = [c.strip().lower() for c in frame.columns]
    print("Loaded relevance CSV columns:", frame.columns.tolist())

    # required columns
    if "query" not in frame.columns or "doc_id" not in frame.columns:
        missing = [c for c in ("query", "doc_id") if c not in frame.columns]
        print("Missing columns in relevance CSV:", missing)
        try:
            import streamlit as _st
            _st.error(f"Missing columns in relevance CSV: {missing}")
        except Exception:
            pass
        return {}

    # if relevance column exists, filter non-zero relevance (allow graded relevance)
    if "relevance" in frame.columns:
        try:
            frame["relevance"] = frame["relevance"].fillna(1).astype(int)
            frame = frame[frame["relevance"] > 0]
        except Exception:
            # if conversion fails, keep all rows
            pass

    judgments = {}
    for query, group in frame.groupby("query"):
        judgments[str(query).strip()] = [str(doc_id).strip() for doc_id in group["doc_id"].dropna().tolist() if str(doc_id).strip()]

    return judgments


QUERIES = load_queries()
RELEVANCE_JUDGMENTS = load_relevance_judgments()
