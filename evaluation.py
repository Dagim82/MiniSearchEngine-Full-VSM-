"""IR evaluation helpers for precision/recall-based dashboard reporting."""

import pandas as pd


def _as_ranked_docs(ranked_docs):
    """Normalize ranked documents into a simple list of document IDs."""
    if ranked_docs is None:
        return []

    if isinstance(ranked_docs, pd.DataFrame):
        # accept either standardized 'doc_id' or legacy 'Document'
        cols = [c.strip().lower() for c in ranked_docs.columns]
        if "doc_id" in cols:
            actual = ranked_docs.columns[cols.index("doc_id")]
            return ranked_docs[actual].tolist()
        if "document" in cols:
            actual = ranked_docs.columns[cols.index("document")]
            return ranked_docs[actual].tolist()
        return []

    normalized = []
    for item in ranked_docs:
        if isinstance(item, (tuple, list)) and item:
            normalized.append(item[0])
        else:
            normalized.append(item)
    return normalized


def precision_recall_f1(ranked_docs, relevant_docs, cutoff=None):
    """Compute Precision, Recall, and F1 for a ranked list.

    The cutoff controls how many retrieved documents are considered. When no
    cutoff is given, the full ranking is used.
    """
    docs = _as_ranked_docs(ranked_docs)
    if cutoff is not None:
        docs = docs[:cutoff]

    retrieved_set = set(docs)
    relevant_set = set(relevant_docs or [])

    true_positives = len(retrieved_set & relevant_set)
    precision = true_positives / len(docs) if docs else 0.0
    recall = true_positives / len(relevant_set) if relevant_set else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    return precision, recall, f1, true_positives


def precision_at_k(ranked_docs, relevant_docs, k):
    """Compute Precision@k for a ranked list."""
    docs = _as_ranked_docs(ranked_docs)[:k]
    if not docs:
        return 0.0

    relevant_set = set(relevant_docs or [])
    true_positives = len(set(docs) & relevant_set)
    return true_positives / len(docs)


def build_metric_table(
    precision,
    recall,
    f1,
):
    """Create a tidy two-column metrics table for display."""
    rows = [
        {"Metric": "Precision", "Score": round(precision, 4)},
        {"Metric": "Recall", "Score": round(recall, 4)},
        {"Metric": "F1-score", "Score": round(f1, 4)},
    ]

    return pd.DataFrame(rows)


def build_relevance_comparison_table(ranked_docs, relevant_docs, cutoff=None):
    """Show whether each retrieved document is relevant according to qrels."""
    docs = _as_ranked_docs(ranked_docs)
    if cutoff is not None:
        docs = docs[:cutoff]

    relevant_set = set(relevant_docs or [])
    return pd.DataFrame(
        [
            {"Retrieved Doc": doc_id, "Relevant?": "Yes" if doc_id in relevant_set else "No"}
            for doc_id in docs
        ]
    )


def evaluate_ranking(
    ranked_docs,
    relevant_docs,
    total_documents=None,
    execution_time=0.0,
    cutoff=None,
    k_values=(5, 10),
):
    """Evaluate a single ranked retrieval result against qrels.

    Returns a dictionary containing metric values and the small tables needed by
    the Streamlit interface and console output.
    """
    docs = _as_ranked_docs(ranked_docs)
    eval_cutoff = cutoff if cutoff is not None else len(docs)
    top_docs = docs[:eval_cutoff]

    precision, recall, f1, true_positives = precision_recall_f1(docs, relevant_docs, cutoff=eval_cutoff)
    metrics_table = build_metric_table(
        precision,
        recall,
        f1,
    )

    statistics_rows = [
        {"Metric": "Total documents", "Score": total_documents if total_documents is not None else 0},
        {"Metric": "Retrieved documents", "Score": len(top_docs)},
        {"Metric": "Relevant documents", "Score": len(set(relevant_docs or []))},
        {"Metric": "True positives", "Score": true_positives},
        {"Metric": "Query execution time (s)", "Score": round(execution_time, 4)},
    ]
    statistics_table = pd.DataFrame(statistics_rows)
    relevance_table = build_relevance_comparison_table(top_docs, relevant_docs, cutoff=eval_cutoff)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positives": true_positives,
        "retrieved_docs": top_docs,
        "relevant_docs": list(relevant_docs or []),
        "metrics_table": metrics_table,
        "statistics_table": statistics_table,
        "relevance_table": relevance_table,
    }


def benchmark_queries(ir_system, query_list, relevance_judgments, cutoff=10):
    """Evaluate the full fixed query set and produce per-query and summary tables."""
    rows = []
    for query in query_list:
        ranked_results, execution_time = ir_system.search(query)
        ranked_docs = [doc_id for doc_id, _ in ranked_results]
        qrels = relevance_judgments.get(query, [])
        report = evaluate_ranking(
            ranked_docs,
            qrels,
            total_documents=ir_system.document_count,
            execution_time=execution_time,
            cutoff=cutoff,
        )

        rows.append(
            {
                "Query": query,
                "Precision": round(report["precision"], 4),
                "Recall": round(report["recall"], 4),
                "F1-score": round(report["f1"], 4),
                "Execution Time (s)": round(execution_time, 4),
                "Relevant Docs": len(qrels),
            }
        )

    per_query_df = pd.DataFrame(rows)
    if per_query_df.empty:
        summary_df = pd.DataFrame()
    else:
        summary_df = pd.DataFrame(
            [
                {"Metric": "Mean Precision", "Score": round(per_query_df["Precision"].mean(), 4)},
                {"Metric": "Mean Recall", "Score": round(per_query_df["Recall"].mean(), 4)},
                {"Metric": "Mean F1-score", "Score": round(per_query_df["F1-score"].mean(), 4)},
            ]
        )

    return per_query_df, summary_df
