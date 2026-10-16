import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server environment
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List, Union, Optional
from sklearn.metrics.pairwise import cosine_similarity
from langchain_core.tools import tool


def make_tools(session):
    """
    Factory function returning tools for dataset exploration and similarity analysis.
    """

    @tool
    def cosine_similarity_tool(
        doc1: Union[int, str] = 0,
        doc2: Union[int, str] = 1,
    ) -> Dict[str, Any]:
        """
        Computes cosine similarity between two documents in session.feature_matrix or raw text strings.

        Accepts document identifiers doc1 and doc2 as 0-based integer row positions (e.g. 0, 1) or raw text strings.
        If raw text is provided, transforms it into vector space using session.artifacts["count_vectorizer"].
        Evaluates pairwise cosine similarity, replacing NaN (e.g. from empty text rows with zero norm) with 0.0.

        Args:
            doc1 (Union[int, str]): Row index (int) or raw text string for 1st document. Default is 0.
            doc2 (Union[int, str]): Row index (int) or raw text string for 2nd document. Default is 1.

        Returns:
            dict: A JSON-serializable result summary containing:
                - result_id (str): Unique execution identifier (e.g. "cosine_similarity_1").
                - status (str): "success" or "error".
                - doc1 (Union[int, str]): 1st document identifier/text.
                - doc2 (Union[int, str]): 2nd document identifier/text.
                - similarity (float): Cosine similarity score rounded to 4 decimal places (0.0 to 1.0).
        """
        if session.feature_matrix is None or session.feature_names is None:
            summary = {
                "result_id": session.next_result_id("cosine_similarity"),
                "status": "error",
                "error": "No document-term matrix found in session. Call build_dtm_tool first.",
                "doc1": doc1,
                "doc2": doc2,
                "similarity": 0.0,
            }
            session.store_result("cosine_similarity_tool", {"doc1": doc1, "doc2": doc2}, summary)
            return summary

        vectorizer = session.artifacts.get("count_vectorizer")

        def _get_vector(doc_identifier):
            """Helper to extract a 1xV sparse row vector for an int row position or string text."""
            if isinstance(doc_identifier, int) or (isinstance(doc_identifier, str) and doc_identifier.isdigit()):
                idx = int(doc_identifier)
                if idx < 0 or idx >= session.feature_matrix.shape[0]:
                    raise ValueError(f"Document index {idx} out of range [0, {session.feature_matrix.shape[0]-1}].")
                return session.feature_matrix[idx : idx + 1]
            elif isinstance(doc_identifier, str):
                if vectorizer is None:
                    raise ValueError("No count_vectorizer artifact found in session to vectorize raw text.")
                return vectorizer.transform([doc_identifier])
            else:
                raise ValueError(f"Invalid document identifier type: {type(doc_identifier)}.")

        try:
            vec1 = _get_vector(doc1)
            vec2 = _get_vector(doc2)
        except Exception as e:
            summary = {
                "result_id": session.next_result_id("cosine_similarity"),
                "status": "error",
                "error": str(e),
                "doc1": doc1,
                "doc2": doc2,
                "similarity": 0.0,
            }
            session.store_result("cosine_similarity_tool", {"doc1": doc1, "doc2": doc2}, summary)
            return summary

        # Check for zero norms (empty text or zero-count vector) to avoid NaN
        norm1 = np.linalg.norm(vec1.toarray())
        norm2 = np.linalg.norm(vec2.toarray())

        if norm1 == 0 or norm2 == 0:
            sim_val = 0.0
        else:
            sim_matrix = cosine_similarity(vec1, vec2)
            sim_val = float(sim_matrix[0, 0])
            if np.isnan(sim_val):
                sim_val = 0.0

        similarity = float(round(sim_val, 4))

        summary = {
            "result_id": session.next_result_id("cosine_similarity"),
            "status": "success",
            "doc1": doc1,
            "doc2": doc2,
            "similarity": similarity,
        }

        session.store_result("cosine_similarity_tool", {"doc1": doc1, "doc2": doc2}, summary)

        return summary

    @tool
    def feature_correlation_matrix_tool(
        top_n: int = 20,
    ) -> Dict[str, Any]:
        """
        Computes feature-vs-feature Pearson correlation heatmap over the top terms by variance.

        Identifies the top top_n terms with highest population variance in session.feature_matrix.
        Computes the pairwise Pearson correlation matrix across those terms' count vectors.
        Generates an annotated seaborn heatmap on session.pending_figure and returns the correlation matrix.

        Args:
            top_n (int): Number of top-variance terms to select for correlation analysis. Default is 20.

        Returns:
            dict: A JSON-serializable result summary containing:
                - result_id (str): Unique execution identifier (e.g. "feature_correlation_1").
                - status (str): "success" or "error".
                - top_terms (List[str]): List of top-variance term names in descending order of variance.
                - correlation_matrix (List[List[float]]): Pairwise correlation matrix (rounded to 4 decimal places).
                - n_terms (int): Actual number of terms evaluated.
        """
        if session.feature_matrix is None or session.feature_names is None:
            summary = {
                "result_id": session.next_result_id("feature_correlation"),
                "status": "error",
                "error": "No document-term matrix found in session. Call build_dtm_tool first.",
                "top_terms": [],
                "correlation_matrix": [],
                "n_terms": 0,
            }
            session.store_result("feature_correlation_matrix_tool", {"top_n": top_n}, summary)
            return summary

        dense_matrix = session.feature_matrix.toarray()
        feature_names = np.array(session.feature_names)

        # 1. Compute population variance (ddof=0) for each feature
        variances = np.var(dense_matrix, axis=0, ddof=0)

        # 2. Select top top_n terms by variance (descending)
        sorted_indices = np.argsort(-variances)  # Negate for descending sort
        selected_indices = sorted_indices[: min(top_n, len(feature_names))]

        top_terms = feature_names[selected_indices].tolist()
        top_matrix = dense_matrix[:, selected_indices]

        # 3. Compute Pearson correlation matrix across selected top-variance terms
        # np.corrcoef expects variables in rows, so transpose top_matrix (columns=features)
        if top_matrix.shape[1] > 0:
            corr_mat = np.corrcoef(top_matrix, rowvar=False)
            # Handle potential 1x1 scalar return from np.corrcoef
            if np.ndim(corr_mat) == 0:
                corr_mat = np.array([[1.0]])
            corr_mat = np.nan_to_num(corr_mat, nan=0.0)
            corr_mat = np.round(corr_mat, 4)
        else:
            corr_mat = np.empty((0, 0))

        corr_list = corr_mat.tolist()

        # 4. Generate annotated heatmap and set on session.pending_figure
        if len(top_terms) > 0:
            fig, ax = plt.subplots(figsize=(max(6, len(top_terms) * 0.8), max(5, len(top_terms) * 0.7)))
            sns.heatmap(
                corr_mat,
                annot=True,
                fmt=".4f",
                cmap="coolwarm",
                vmin=-1.0,
                vmax=1.0,
                xticklabels=top_terms,
                yticklabels=top_terms,
                ax=ax,
            )
            ax.set_title(f"Feature Correlation Matrix (Top {len(top_terms)} Terms by Variance)")
            fig.tight_layout()

            session.pending_figure = fig
            plt.close(fig)

        summary = {
            "result_id": session.next_result_id("feature_correlation"),
            "status": "success",
            "top_terms": top_terms,
            "correlation_matrix": corr_list,
            "n_terms": len(top_terms),
        }

        session.store_result("feature_correlation_matrix_tool", {"top_n": top_n}, summary)

        return summary

    return [cosine_similarity_tool, feature_correlation_matrix_tool]
