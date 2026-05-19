"""Streamlit application for the Cranfield-style IR evaluation demo.

The interface keeps the TF-IDF retrieval pipeline, but evaluation now comes
from predefined benchmark queries and human-defined relevance judgments. This
matches standard information retrieval evaluation methodology used in research
and TREC-style experiments.
"""

import pandas as pd
import streamlit as st

from benchmark_data import QUERIES, RELEVANCE_JUDGMENTS
from evaluation import benchmark_queries, build_metric_table, evaluate_ranking
from ir_system import InformationRetrievalSystem


st.set_page_config(page_title="Mini Search Engine", page_icon="Search", layout="wide")


# ------------------------------
# Session state helpers
# ------------------------------
def build_empty_report():
    """Create an empty report so the page can render before the first search."""
    return {
        "query": "",
        "query_terms": [],
        "results_table": pd.DataFrame(),
        "query_tf_table": pd.DataFrame(),
        "query_idf_table": pd.DataFrame(),
        "evaluation_table": pd.DataFrame(),
        "statistics_table": pd.DataFrame(),
        "relevance_table": pd.DataFrame(),
        "details_table": pd.DataFrame(),
        "keywords_table": pd.DataFrame(),
        "retrieved_docs": [],
        "relevant_docs": [],
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
        "execution_time": 0.0,
        "top_k": 10,
}


def initialize_state():
    """Initialize the Streamlit session state used across reruns."""
    if "selected_query" not in st.session_state:
        st.session_state.selected_query = QUERIES[0]
    if "search_query" not in st.session_state:
        st.session_state.search_query = QUERIES[0]
    if "last_report" not in st.session_state:
        st.session_state.last_report = build_empty_report()
    if "benchmark_report" not in st.session_state:
        st.session_state.benchmark_report = {
            "per_query": pd.DataFrame(),
            "summary": pd.DataFrame(),
        }
    # Dark mode toggle state (UI icon)
    if "dark_mode" not in st.session_state:
        st.session_state.dark_mode = False


@st.cache_resource(show_spinner=False)
def get_ir_system():
    """Build the IR system once and reuse it across Streamlit reruns."""
    return InformationRetrievalSystem()


def sync_search_query():
    """Copy the selected benchmark query into the search box."""
    st.session_state.search_query = st.session_state.selected_query


initialize_state()
ir_system = get_ir_system()

if ir_system.load_error:
    st.error(ir_system.load_error)
    st.stop()


# ------------------------------
# Dark mode icon + CSS
# ------------------------------
dm_left, dm_right = st.columns([9, 1])
with dm_right:
    # simple moon icon checkbox to toggle dark mode
    st.checkbox("🌙", key="dark_mode", help="Toggle dark mode for the demo UI")

if st.session_state.get("dark_mode", False):
    dark_css = """
    <style>
    /* Basic dark theme overrides for the demo app */
    html, body, .stApp {
        background: #0b1220 !important;
        color: #e6eef8 !important;
    }
    .hero-card { background: linear-gradient(180deg, rgba(15,23,42,0.9), rgba(10,15,30,0.9)) !important; }
    .stButton>button, .stCheckbox>div>label {
        color: #e6eef8 !important;
    }
    .streamlit-expanderHeader { color: #cbd5e1 !important; }
    .stDataFrame td, .stDataFrame th { color: #e6eef8 !important; }
    .stMetric { color: #e6eef8 !important; }
    </style>
    """
    st.markdown(dark_css, unsafe_allow_html=True)


# ------------------------------
# Sidebar: corpus and evaluation summary
# ------------------------------
st.sidebar.title("Mini Search Engine")
st.sidebar.markdown("Professional educational search demo with standard Cranfield evaluation.")

st.sidebar.markdown("### Corpus Statistics")
st.sidebar.metric("Number of Documents", ir_system.document_count)
st.sidebar.metric("Vocabulary Size", ir_system.vocabulary_size)
st.sidebar.metric("Document Load Time (s)", f"{getattr(ir_system, 'document_load_time', 0.0):.4f}")
st.sidebar.metric("Preprocessing Time (s)", f"{getattr(ir_system, 'preprocessing_time', 0.0):.4f}")
st.sidebar.metric("Index Build Time (s)", f"{getattr(ir_system, 'indexing_time', 0.0):.4f}")

st.sidebar.markdown("### Standard Query Set")
st.sidebar.selectbox(
    "Predefined query",
    QUERIES,
    key="selected_query",
    on_change=sync_search_query,
)

st.sidebar.markdown("### Retrieval Controls")
top_k = st.sidebar.selectbox("Top-K Retrieval", [5, 10, 15], index=1)

st.sidebar.markdown("### Performance Notes")
st.sidebar.caption("Search runs only when the Search button is clicked. Benchmark tables are computed on demand.")


# ------------------------------
# Main page header and search controls
# ------------------------------
st.markdown(
        """
        <style>
        .hero-container {
                display: flex;
                justify-content: center;
                margin-top: 8px;
                margin-bottom: 12px;
        }
        .hero-card {
                max-width: 880px;
                padding: 18px 26px;
                border-radius: 10px;
                background: linear-gradient(180deg, rgba(255,255,255,0.95), rgba(250,250,250,0.95));
                box-shadow: 0 6px 18px rgba(20,20,20,0.06);
                text-align: center;
        }
        .hero-title { font-size: 28px; font-weight:700; margin: 6px 0 8px 0; }
        .hero-desc { font-size: 15px; color: #374151; margin-bottom: 8px; }
        .hero-helper { font-size: 13px; color: #6b7280; margin-top: 6px; }
        @media (max-width: 640px) {
                .hero-card { padding: 14px; }
                .hero-title { font-size: 22px; }
        }
        </style>

        <div class="hero-container">
            <div class="hero-card">
                <div class="hero-title">🎓 Welcome to the Educational Note Search Engine</div>
                <div class="hero-desc">Search through academic notes, lecture materials, research content, and educational documents using intelligent information retrieval techniques.</div>
                <div class="hero-helper">Choose a predefined Cranfield-style query or enter your own custom search query to discover the most relevant study materials instantly.</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
)

query = st.text_input(
    "Search Box",
    value=st.session_state.search_query,
    placeholder="Example: machine learning",
    key="search_query",
)

col_search, col_help = st.columns([1, 1])
with col_search:
    search_clicked = st.button("Search")
with col_help:
    st.caption("The search box can be edited, but evaluation uses the human qrels when a matching predefined query is used.")


# ------------------------------
# Run retrieval only when requested
# ------------------------------
if search_clicked:
    current_query = st.session_state.search_query.strip()
    st.session_state.last_report = build_empty_report()

    if not current_query:
        st.warning("Please enter a query before searching.")
    else:
        ranked_results, execution_time = ir_system.search(current_query)
        query_terms, query_tf_df, query_idf_df = ir_system.analyze_query_terms(current_query, top_n=10)
        top_ranked_results = ranked_results[:top_k]
        results_table = ir_system.build_results_table(top_ranked_results)
        results_table = InformationRetrievalSystem.normalize_results_schema(results_table)
        print("app.py results_table columns:", results_table.columns.tolist())
        print("app.py results_table head:\n", results_table.head().to_string(index=False))
        relevant_docs = RELEVANCE_JUDGMENTS.get(current_query, [])
        evaluation_report = evaluate_ranking(
            [doc_id for doc_id, _ in ranked_results],
            relevant_docs,
            total_documents=ir_system.document_count,
            execution_time=execution_time,
            cutoff=top_k,
        )
        metrics_table = evaluation_report["metrics_table"]
        statistics_table = evaluation_report["statistics_table"]

        st.session_state.last_report = {
            "query": current_query,
            "query_terms": query_terms,
            "results_table": results_table,
            "query_tf_table": query_tf_df,
            "query_idf_table": query_idf_df,
            "evaluation_table": metrics_table,
            "statistics_table": statistics_table,
            "details_table": pd.DataFrame(),
            "retrieved_docs": evaluation_report["retrieved_docs"],
            "relevant_docs": evaluation_report["relevant_docs"],
            "precision": evaluation_report["precision"],
            "recall": evaluation_report["recall"],
            "f1": evaluation_report["f1"],
            "execution_time": execution_time,
            "top_k": top_k,
        }

        try:
            # trace the exact DataFrame passed into formatter
            console_df = st.session_state.last_report.get("results_table", pd.DataFrame())
            print("last_report['results_table'] columns:", console_df.columns.tolist())
            print("last_report['results_table'] head:\n", console_df.head().to_string(index=False))
            InformationRetrievalSystem.format_console_output(st.session_state.last_report)
        except Exception as e:
            import traceback
            traceback.print_exc()
            st.error(f"Error formatting console output: {e}")

report = st.session_state.last_report


# ------------------------------
# Query processing visualization
# ------------------------------
if report["query"]:
    st.subheader("Query Processing")
    qcol1, qcol2 = st.columns(2)
    with qcol1:
        st.markdown("<div style='color:#1f77b4; font-weight:600;'>Original Query</div>", unsafe_allow_html=True)
        st.write(report["query"])
    with qcol2:
        st.markdown("<div style='color:#2ca02c; font-weight:600;'>Processed Query Tokens</div>", unsafe_allow_html=True)
        st.code(str(report["query_terms"] if report["query_terms"] else []), language="python")


# ------------------------------
# Performance statistics cards
# ------------------------------
st.subheader("Performance Statistics")
stats_col1, stats_col2, stats_col3, stats_col4 = st.columns(4)
stats_col1.metric("Number of Documents", ir_system.document_count)
stats_col2.metric("Vocabulary Size", ir_system.vocabulary_size)
stats_col3.metric("Query Execution Time (s)", f"{report['execution_time']:.4f}")
stats_col4.metric("Matched Documents", len(report["retrieved_docs"]))


# ------------------------------
# TF and IDF tables
# ------------------------------
st.subheader("TF Values")
if report["query_tf_table"].empty:
    st.info("Run a search to see TF values for the query terms.")
else:
    with st.expander("Show TF Table", expanded=True):
        st.dataframe(report["query_tf_table"], use_container_width=True, hide_index=True)

st.subheader("IDF Values")
if report["query_idf_table"].empty:
    st.info("Run a search to see IDF values for the query terms.")
else:
    with st.expander("Show IDF Table", expanded=True):
        st.dataframe(report["query_idf_table"], use_container_width=True, hide_index=True)


# ------------------------------
# Evaluation metrics table
# ------------------------------
st.subheader("Evaluation Metrics")
evaluation_table = build_metric_table(
    report["precision"],
    report["recall"],
    report["f1"],
)
st.dataframe(evaluation_table, use_container_width=True, hide_index=True)

st.subheader("Retrieval Statistics")
st.dataframe(report["statistics_table"], use_container_width=True, hide_index=True)


# ------------------------------
# Ranked results and chart
# ------------------------------
if report["query"] and report["results_table"].empty:
    st.warning("No matching documents found for this query.")
elif not report["results_table"].empty:
    st.subheader(f"Results for: {report['query']}")

    left_col, right_col = st.columns([2, 1])
    with left_col:
        st.dataframe(report["results_table"], use_container_width=True, hide_index=True)
    with right_col:
        results_df = InformationRetrievalSystem.normalize_results_schema(report["results_table"].copy())
        required_cols = ["Rank", "Document", "Similarity Score"]
        missing_cols = [c for c in required_cols if c not in results_df.columns]
        if missing_cols:
            print("Missing columns:", missing_cols)
            st.error(f"Missing columns: {missing_cols}")
        else:
            chart_df = results_df[["Document", "Similarity Score"]].set_index("Document")
            st.bar_chart(chart_df)

if report["query"] and report["query"] not in RELEVANCE_JUDGMENTS:
    st.info("This query is not part of the fixed benchmark qrels, so standard Cranfield evaluation is only available for the predefined query set.")


# ------------------------------
# Optional benchmark summary
# ------------------------------
st.subheader("Search Performance Summary")
if st.button("Run Benchmark on All Standard Queries"):
    per_query_df, summary_df = benchmark_queries(
        ir_system,
        QUERIES,
        RELEVANCE_JUDGMENTS,
        cutoff=top_k,
    )
    st.session_state.benchmark_report = {
        "per_query": per_query_df,
        "summary": summary_df,
    }

benchmark_per_query = st.session_state.benchmark_report["per_query"]
benchmark_summary = st.session_state.benchmark_report["summary"]

if benchmark_summary.empty:
    st.info("Click the benchmark button to evaluate all predefined queries.")
else:
    st.dataframe(benchmark_summary, use_container_width=True, hide_index=True)
    with st.expander("Per-query benchmark results", expanded=False):
        st.dataframe(benchmark_per_query, use_container_width=True, hide_index=True)
