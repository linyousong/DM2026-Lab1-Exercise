import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server environment
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List, Optional
from sklearn.feature_extraction.text import CountVectorizer
from langchain_core.tools import tool


def make_tools(session):
    """
    Factory function returning tools for document-term matrix (DTM) construction and analysis.
    """

    @tool
    def build_dtm_tool(
        text_column: str = "text",
        ngram_range: List[int] = [1, 1],
        max_features: Optional[int] = None,
        min_df: float = 1,
    ) -> Dict[str, Any]:
        """
        Builds a global Document-Term Matrix (DTM) from session.dataframe using CountVectorizer.

        Fits CountVectorizer on session.dataframe[text_column] and stores:
          - session.feature_matrix: Scipy sparse matrix of term counts.
          - session.feature_names: List of vocabulary feature names.
          - session.artifacts["count_vectorizer"]: The fitted CountVectorizer object (required by cosine_similarity_tool).
        Also syncs labels on session if category_name column exists.
        Calculates matrix sparsity: non_zero, total_elements, and sparsity_pct.

        Args:
            text_column (str): The column in session.dataframe containing document text. Default is "text".
            ngram_range (List[int]): Range of n-gram sizes as [min_n, max_n]. Default is [1, 1]. Safe for Gemini.
            max_features (Optional[int]): Maximum number of top vocabulary features to keep. Default is None.
            min_df (float): Minimum document frequency for vocabulary filtering (int or float fraction). Default is 1.

        Returns:
            dict: A JSON-serializable result summary containing:
                - result_id (str): Unique execution identifier (e.g. "build_dtm_1").
                - n_documents (int): Number of rows (documents) in DTM.
                - n_terms (int): Number of columns (vocabulary features) in DTM.
                - non_zero (int): Total non-zero cell count (matrix.nnz).
                - total_elements (int): Total matrix cells (n_documents * n_terms).
                - sparsity_pct (float): Sparsity percentage: 100 * (1 - non_zero / total_elements).
                - feature_names (List[str]): List of extracted vocabulary feature names.
        """
        if session.dataframe is None:
            raise ValueError("No dataset loaded in session. Call load_dataset_tool first.")

        if text_column not in session.dataframe.columns:
            raise ValueError(f"Column '{text_column}' not found in session.dataframe.")

        # Convert list to tuple for CountVectorizer (avoiding fixed Tuple type hint for Gemini safety)
        ngram_tuple = (int(ngram_range[0]), int(ngram_range[1])) if len(ngram_range) >= 2 else (1, 1)

        # Ensure text column is filled with string representations (convert missing/NaN to empty string)
        texts = session.dataframe[text_column].fillna("").astype(str).tolist()

        vectorizer = CountVectorizer(
            ngram_range=ngram_tuple,
            max_features=max_features,
            min_df=min_df,
        )

        feature_matrix = vectorizer.fit_transform(texts)
        feature_names = vectorizer.get_feature_names_out().tolist()

        # Set session state attributes
        session.feature_matrix = feature_matrix
        session.feature_names = feature_names
        session.artifacts["count_vectorizer"] = vectorizer

        # Set labels on session if category_name or label column exists
        if "category_name" in session.dataframe.columns:
            session.set_labels(session.dataframe["category_name"].to_numpy())
        elif "label" in session.dataframe.columns:
            session.set_labels(session.dataframe["label"].to_numpy())

        # Compute sparsity metrics matching Master notebook calculations
        n_documents, n_terms = feature_matrix.shape
        non_zero = int(feature_matrix.nnz)
        total_elements = n_documents * n_terms
        sparsity_pct = 100.0 * (1.0 - non_zero / total_elements) if total_elements > 0 else 0.0

        summary = {
            "result_id": session.next_result_id("build_dtm"),
            "n_documents": n_documents,
            "n_terms": n_terms,
            "non_zero": non_zero,
            "total_elements": total_elements,
            "sparsity_pct": round(sparsity_pct, 4),
        }

        session.store_result(
            "build_dtm_tool",
            {
                "text_column": text_column,
                "ngram_range": ngram_range,
                "max_features": max_features,
                "min_df": min_df,
            },
            summary,
        )

        return summary

    @tool
    def term_frequency_tool() -> Dict[str, Any]:
        """
        Aggregates global term frequencies across all documents in session.feature_matrix.

        Sums occurrences of each term across all rows in the global document-term matrix (session.feature_matrix).
        The summary returns only the 20 most frequent terms with their total counts, plus overall totals,
        so the result stays small enough for the model context. The complete frequency table is stored in
        full_report (term, frequency; sorted descending) and in session.artifacts["term_frequencies"].
        If build_dtm_tool has not been run, returns a structured error summary dict instead of crashing.

        Returns:
            dict: A JSON-serializable result summary containing:
                - result_id (str): Unique execution identifier (e.g. "term_frequency_1").
                - status (str): "success" or "error".
                - top_terms (Dict[str, int]): The 20 most frequent terms mapped to their total occurrence counts.
                - total_terms (int): Total unique vocabulary size.
                - total_word_occurrences (int): Sum of all term frequencies across all documents.
        """
        if session.feature_matrix is None or session.feature_names is None:
            summary = {
                "result_id": session.next_result_id("term_frequency"),
                "status": "error",
                "error": "No document-term matrix found in session. Call build_dtm_tool first.",
                "top_terms": {},
                "total_terms": 0,
                "total_word_occurrences": 0,
            }
            session.store_result("term_frequency_tool", {}, summary)
            return summary

        # Sum columns of sparse matrix across all rows
        col_sums = np.asarray(session.feature_matrix.sum(axis=0)).flatten()
        term_frequencies = {
            str(term): int(count)
            for term, count in zip(session.feature_names, col_sums)
        }

        # Full table stays in session artifacts (not sent to the LLM)
        session.artifacts["term_frequencies"] = term_frequencies

        # Only the 20 most frequent terms go into the summary
        sorted_items = sorted(term_frequencies.items(), key=lambda kv: kv[1], reverse=True)
        top_terms = {term: count for term, count in sorted_items[:20]}

        summary = {
            "result_id": session.next_result_id("term_frequency"),
            "status": "success",
            "top_terms": top_terms,
            "total_terms": len(term_frequencies),
            "total_word_occurrences": int(col_sums.sum()),
        }

        session.store_result("term_frequency_tool", {}, summary)

        return summary

    @tool
    def dtm_heatmap_tool(
        n_terms: int = 20,
        n_documents: int = 20,
    ) -> Dict[str, Any]:
        """
        Visualizes a positional slice of the raw global Document-Term Matrix (DTM) as a heatmap.

        Extracts a raw positional slice session.feature_matrix[0:n_documents, 0:n_terms] taking terms
        in session.feature_names's existing order and documents in session.dataframe's existing row order
        without sorting or variance filtering. Generates a seaborn heatmap annotated with raw counts and sets
        the matplotlib figure on session.pending_figure.

        Args:
            n_terms (int): Maximum number of terms (columns) to slice from start of vocabulary. Default is 20.
            n_documents (int): Maximum number of documents (rows) to slice from start of dataset. Default is 20.

        Returns:
            dict: A JSON-serializable result summary containing:
                - result_id (str): Unique execution identifier (e.g. "dtm_heatmap_1").
                - status (str): "success" or "error".
                - dtm_slice (List[List[int]]): 2D nested list of term occurrence counts for sliced sub-matrix.
                - terms (List[str]): Sliced vocabulary feature names.
                - document_indices (List[int]): Sliced 0-based row indices.
                - n_terms (int): Actual number of terms included in slice.
                - n_documents (int): Actual number of documents included in slice.
        """
        if session.feature_matrix is None or session.feature_names is None:
            summary = {
                "result_id": session.next_result_id("dtm_heatmap"),
                "status": "error",
                "error": "No document-term matrix found in session. Call build_dtm_tool first.",
                "dtm_slice": [],
                "terms": [],
                "document_indices": [],
                "n_terms": 0,
                "n_documents": 0,
            }
            session.store_result("dtm_heatmap_tool", {"n_terms": n_terms, "n_documents": n_documents}, summary)
            return summary

        total_docs, total_terms = session.feature_matrix.shape

        # Take raw positional slice [0:n_documents, 0:n_terms]
        actual_docs = min(max(1, n_documents), total_docs)
        actual_terms = min(max(1, n_terms), total_terms)

        sliced_matrix = session.feature_matrix[:actual_docs, :actual_terms].toarray()
        dtm_slice = sliced_matrix.tolist()

        sliced_terms = session.feature_names[:actual_terms]
        doc_indices = list(range(actual_docs))

        # Generate annotated seaborn heatmap
        fig, ax = plt.subplots(figsize=(max(6, actual_terms * 0.8), max(4, actual_docs * 0.5)))
        sns.heatmap(
            sliced_matrix,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=sliced_terms,
            yticklabels=doc_indices,
            ax=ax,
            cbar=True,
        )
        ax.set_title(f"Raw DTM Heatmap ({actual_docs} Docs x {actual_terms} Terms)")
        ax.set_xlabel("Vocabulary Terms")
        ax.set_ylabel("Document Index")
        fig.tight_layout()

        # Set matplotlib Figure object on session.pending_figure
        session.pending_figure = fig
        plt.close(fig)

        summary = {
            "result_id": session.next_result_id("dtm_heatmap"),
            "status": "success",
            "dtm_slice": dtm_slice,
            "terms": sliced_terms,
            "document_indices": doc_indices,
            "n_terms": actual_terms,
            "n_documents": actual_docs,
        }

        session.store_result("dtm_heatmap_tool", {"n_terms": n_terms, "n_documents": n_documents}, summary)

        return summary

    return [build_dtm_tool, term_frequency_tool, dtm_heatmap_tool]
