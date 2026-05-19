"""Core information-retrieval logic used by the Streamlit app.

This module loads the document collection, builds the TF-IDF index, ranks
documents for a query, and generates automatic relevance judgments from the
document content itself. The goal is to make evaluation work without manual
ground-truth labels.
"""

from collections import Counter
from pathlib import Path
import time

import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from preprocess import preprocess


@st.cache_data(show_spinner=False)
def load_documents(docs_path=None):
    """Load every .txt file from the docs folder into a document dictionary."""
    docs_folder = Path(docs_path) if docs_path else Path(__file__).resolve().parent / "docs"
    documents = {}

    if not docs_folder.exists():
        return documents

    for file_path in sorted(docs_folder.glob("*.txt")):
        documents[file_path.stem] = file_path.read_text(encoding="utf-8")

    return documents


@st.cache_data(show_spinner=False)
def preprocess_documents(document_items):
    """Preprocess the full corpus once and keep the token lists in memory."""
    return {
        doc_id: preprocess(text)
        for doc_id, text in document_items
    }


class InformationRetrievalSystem:
    """Wrap the document collection, vectorizer, and query helpers."""

    def __init__(self, docs_path=None):
        self.load_error = None
        startup_start = time.perf_counter()

        load_start = time.perf_counter()
        self.documents = load_documents(docs_path)
        self.document_load_time = time.perf_counter() - load_start
        self.doc_ids = list(self.documents.keys())
        self.doc_texts = [self.documents[doc_id] for doc_id in self.doc_ids]
        self._keyword_cache = {}

        if not self.documents:
            docs_folder = Path(docs_path) if docs_path else Path(__file__).resolve().parent / "docs"
            if not docs_folder.exists():
                self.load_error = f"Missing docs folder: {docs_folder}"
            else:
                self.load_error = f"No .txt documents found in: {docs_folder}"
            self.vectorizer = None
            self.doc_tfidf_matrix = None
            self.idf_map = {}
            self.processed_docs = {}
            self.preprocessing_time = 0.0
            self.indexing_time = 0.0
            return

        preprocess_start = time.perf_counter()
        self.processed_docs = preprocess_documents(tuple(self.documents.items()))
        self.preprocessing_time = time.perf_counter() - preprocess_start

        indexed_docs = [" ".join(self.processed_docs[doc_id]) for doc_id in self.doc_ids]

        self.vectorizer = TfidfVectorizer(
            tokenizer=str.split,
            preprocessor=None,
            lowercase=False,
            token_pattern=None,
        )

        index_start = time.perf_counter()
        self.doc_tfidf_matrix = self.vectorizer.fit_transform(indexed_docs)
        self.indexing_time = time.perf_counter() - index_start
        self.idf_map = dict(zip(self.vectorizer.get_feature_names_out(), self.vectorizer.idf_))

        self.total_startup_time = time.perf_counter() - startup_start
        print(f"Documents loaded in {self.document_load_time:.4f} seconds")
        print(f"Documents preprocessed in {self.preprocessing_time:.4f} seconds")
        print(f"TF-IDF built in {self.indexing_time:.4f} seconds")
        print(f"IR system initialized in {self.total_startup_time:.4f} seconds")

    @property
    def document_count(self):
        return len(self.documents)

    @property
    def vocabulary_size(self):
        return len(self.idf_map)

    @property
    def is_ready(self):
        return self.vectorizer is not None and self.doc_tfidf_matrix is not None

    def search(self, query):
        """Return ranked documents for a query and the elapsed execution time."""
        if not self.is_ready or not query.strip():
            return [], 0.0

        start_time = time.perf_counter()
        processed_query = " ".join(preprocess(query))
        query_vector = self.vectorizer.transform([processed_query])
        similarity_scores = cosine_similarity(query_vector, self.doc_tfidf_matrix).flatten()

        scores = {
            self.doc_ids[index]: score
            for index, score in enumerate(similarity_scores)
            if score > 0
        }

        ranked_results = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        execution_time = time.perf_counter() - start_time
        return ranked_results, execution_time

    def get_document_top_terms(self, top_n=3):
        """Return a DataFrame with the highest-TFIDF terms for each document."""
        if not self.is_ready:
            return pd.DataFrame()

        if top_n in self._keyword_cache:
            return self._keyword_cache[top_n].copy()

        feature_names = self.vectorizer.get_feature_names_out()
        rows = []

        for doc_index, doc_id in enumerate(self.doc_ids):
            row = self.doc_tfidf_matrix[doc_index]
            term_weights = sorted(
                zip(row.indices, row.data),
                key=lambda item: item[1],
                reverse=True,
            )
            top_terms = [feature_names[index] for index, weight in term_weights[:top_n] if weight > 0]

            rows.append(
                {
                    "Document": doc_id,
                    "Top Terms": ", ".join(top_terms),
                }
            )

        keyword_df = pd.DataFrame(rows)
        self._keyword_cache[top_n] = keyword_df
        return keyword_df.copy()

    def analyze_query_terms(self, query, top_n=10):
        """Return processed query tokens and TF/IDF tables for the top terms."""
        if not self.is_ready or not query.strip():
            return [], pd.DataFrame(), pd.DataFrame()

        query_terms = preprocess(query)
        query_tf = Counter(query_terms)

        tf_rows = []
        idf_rows = []

        for term in sorted(query_tf, key=lambda term: (-query_tf[term], term))[:top_n]:
            tf_rows.append({"Term": term, "Frequency": query_tf[term]})
            idf_rows.append({"Term": term, "IDF Score": round(self.idf_map.get(term, 0.0), 4)})

        return query_terms, pd.DataFrame(tf_rows), pd.DataFrame(idf_rows)

    @staticmethod
    def normalize_results_schema(df):
        """Normalize any results DataFrame to the display schema used across the app."""
        if df is None or df.empty:
            return pd.DataFrame(columns=["Rank", "Document", "Similarity Score"])

        # debug trace for dataframe schema
        print("normalize_results_schema input columns:", df.columns.tolist())
        print("normalize_results_schema input head:\n", df.head().to_string(index=False))

        try:
            df.columns = df.columns.str.strip().str.lower()
        except Exception:
            df.columns = [str(c).strip().lower() for c in df.columns]

        column_mapping = {
            "rank": "Rank",
            "doc_id": "Document",
            "document": "Document",
            "score": "Similarity Score",
            "similarity": "Similarity Score",
            "similarity score": "Similarity Score",
            "cosine similarity score": "Similarity Score",
            "similarity_score": "Similarity Score",
        }
        df = df.rename(columns=column_mapping)

        required_cols = ["Rank", "Document", "Similarity Score"]
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            print("Missing columns:", missing)
            print("Available columns:", df.columns.tolist())
            for col in missing:
                df[col] = pd.NA

        # keep only the canonical display schema order
        df = df[required_cols]

        print("normalize_results_schema output columns:", df.columns.tolist())
        print("normalize_results_schema output head:\n", df.head().to_string(index=False))
        return df

    @staticmethod
    def build_results_table(results):
        """Convert ranked retrieval results into a DataFrame for display."""
        rows = []
        for index, (doc_id, score) in enumerate(results):
            rows.append({
                "rank": index + 1,
                "doc_id": doc_id,
                "score": round(score, 4),
            })

        return InformationRetrievalSystem.normalize_results_schema(pd.DataFrame(rows))

    @staticmethod
    def format_console_output(report):
        """Print a report-style summary to the console."""
        print("\n========================")
        print("QUERY RESULTS")
        print("========================\n")
        print(f"Query: {report['query']}\n")

        if not report["results_table"].empty:
            results_df = report["results_table"].head(report["top_k"]).copy()
            print("format_console_output raw results columns:", results_df.columns.tolist())
            print("format_console_output raw results head:\n", results_df.head().to_string(index=False))

            normalized_df = InformationRetrievalSystem.normalize_results_schema(results_df)
            required_cols = ["Rank", "Document", "Similarity Score"]
            missing = [c for c in required_cols if c not in normalized_df.columns]
            if missing:
                print("Missing columns:", missing)
                print("Available columns:", normalized_df.columns.tolist())

            print(normalized_df[required_cols].to_string(index=False))
        else:
            print("No matching documents found.")

        print("\nRetrieval Statistics:")
        if not report["statistics_table"].empty:
            print(report["statistics_table"].to_string(index=False))

        print("\n========================")
        print("EVALUATION METRICS")
        print("========================\n")
        print(report["evaluation_table"].to_string(index=False))
