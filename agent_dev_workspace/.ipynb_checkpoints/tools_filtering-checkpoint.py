import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from scipy.stats import pearsonr, spearmanr
from langchain_core.tools import tool


def make_tools(session):
    """
    Factory function returning tools for feature selection and term filtering.
    """

    @tool
    def variance_filter_tool(
        threshold: float = 0.15,
    ) -> Dict[str, Any]:
        """
        Filters out low-variance terms from the global Document-Term Matrix (session.feature_matrix).

        Calculates population variance (ddof=0) for each vocabulary term in session.feature_matrix.
        Identifies terms with variance >= threshold to keep, and terms with variance < threshold to remove.
        Creates a full_report DataFrame sorted in descending order of variance with term as 1st column
        and variance as 2nd column (matching visualize_result_tool requirements).

        Args:
            threshold (float): Minimum variance required to keep a feature. Default is 0.15.

        Returns:
            dict: A JSON-serializable result summary containing:
                - result_id (str): Unique execution identifier (e.g. "variance_filter_1").
                - status (str): "success" or "error".
                - threshold (float): Evaluated variance threshold.
                - total_terms (int): Initial vocabulary size.
                - kept_terms (List[str]): List of term names meeting or exceeding threshold.
                - removed_terms (List[str]): List of term names below threshold.
                - kept_count (int): Number of kept features.
                - removed_count (int): Number of removed features.
                - variances (Dict[str, float]): Mapping of term names to variance values.
        """
        if session.feature_matrix is None or session.feature_names is None:
            summary = {
                "result_id": session.next_result_id("variance_filter"),
                "status": "error",
                "error": "No document-term matrix found in session. Call build_dtm_tool first.",
                "threshold": threshold,
                "total_terms": 0,
                "kept_terms": [],
                "removed_terms": [],
                "kept_count": 0,
                "removed_count": 0,
                "variances": {},
            }
            session.store_result("variance_filter_tool", {"threshold": threshold}, summary)
            return summary

        # Convert sparse matrix to dense array for variance computation
        dense_matrix = session.feature_matrix.toarray()

        # Compute population variance (ddof=0) matching TEST_FIXTURE.md known answer #5
        variances = np.var(dense_matrix, axis=0, ddof=0)

        feature_names = session.feature_names
        var_dict = {str(term): float(var) for term, var in zip(feature_names, variances)}

        kept_terms = [term for term, var in var_dict.items() if var >= threshold]
        removed_terms = [term for term, var in var_dict.items() if var < threshold]

        # Construct full_report DataFrame: term as 1st column, variance as 2nd column, sorted descending
        report_data = [
            {"term": term, "variance": float(var)}
            for term, var in zip(feature_names, variances)
        ]
        full_report = pd.DataFrame(report_data).sort_values(by="variance", ascending=False).reset_index(drop=True)

        summary = {
            "result_id": session.next_result_id("variance_filter"),
            "status": "success",
            "threshold": threshold,
            "total_terms": len(feature_names),
            "kept_terms": kept_terms,
            "removed_terms": removed_terms,
            "kept_count": len(kept_terms),
            "removed_count": len(removed_terms),
            "variances": var_dict,
        }

        # Store result with full_report for visualize_result_tool compatibility
        session.store_result(
            "variance_filter_tool",
            {"threshold": threshold},
            summary,
            full_report=full_report,
        )

        return summary

    def _run_correlation_filter(
        tool_prefix: str,
        metric_name: str,
        corr_func,
        target_class: str,
        threshold: float,
    ) -> Dict[str, Any]:
        """Internal helper running One-vs-Rest correlation filtering (Pearson or Spearman)."""
        if session.feature_matrix is None or session.feature_names is None:
            summary = {
                "result_id": session.next_result_id(tool_prefix),
                "status": "error",
                "error": "No document-term matrix found in session. Call build_dtm_tool first.",
                "target_class": target_class,
                "threshold": threshold,
                "total_terms": 0,
                "kept_terms": [],
                "removed_terms": [],
                metric_name: {},
            }
            session.store_result(f"{tool_prefix}_tool", {"target_class": target_class, "threshold": threshold}, summary)
            return summary

        labels = None
        if session.dataframe is not None and "category_name" in session.dataframe.columns:
            labels = session.dataframe["category_name"].to_numpy()
        elif session.dataframe is not None and "label" in session.dataframe.columns:
            labels = session.dataframe["label"].to_numpy()
        elif session.labels is not None:
            labels = session.labels

        if labels is None:
            summary = {
                "result_id": session.next_result_id(tool_prefix),
                "status": "error",
                "error": "No category labels found in session. Load dataset with label_column specified.",
                "target_class": target_class,
                "threshold": threshold,
                "total_terms": 0,
                "kept_terms": [],
                "removed_terms": [],
                metric_name: {},
            }
            session.store_result(f"{tool_prefix}_tool", {"target_class": target_class, "threshold": threshold}, summary)
            return summary

        unique_classes = set(labels)
        if target_class not in unique_classes:
            summary = {
                "result_id": session.next_result_id(tool_prefix),
                "status": "error",
                "error": f"Target class '{target_class}' not found in dataset categories: {list(unique_classes)}.",
                "target_class": target_class,
                "threshold": threshold,
                "total_terms": 0,
                "kept_terms": [],
                "removed_terms": [],
                metric_name: {},
            }
            session.store_result(f"{tool_prefix}_tool", {"target_class": target_class, "threshold": threshold}, summary)
            return summary

        y = (labels == target_class).astype(float)
        dense_matrix = session.feature_matrix.toarray()

        correlations = {}
        report_rows = []

        for term, col in zip(session.feature_names, dense_matrix.T):
            if np.std(col) == 0 or np.std(y) == 0:
                r_val = 0.0
            else:
                r_val, _ = corr_func(col, y)
                if np.isnan(r_val):
                    r_val = 0.0

            r_val = float(round(r_val, 4))
            correlations[str(term)] = r_val
            report_rows.append({"term": str(term), metric_name: r_val})

        kept_terms = [term for term, r in correlations.items() if r >= threshold]
        removed_terms = [term for term, r in correlations.items() if r < threshold]

        full_report = pd.DataFrame(report_rows).sort_values(by=metric_name, ascending=False).reset_index(drop=True)

        summary = {
            "result_id": session.next_result_id(tool_prefix),
            "status": "success",
            "target_class": target_class,
            "threshold": threshold,
            "total_terms": len(session.feature_names),
            "kept_terms": kept_terms,
            "removed_terms": removed_terms,
            metric_name: correlations,
        }

        session.store_result(
            f"{tool_prefix}_tool",
            {"target_class": target_class, "threshold": threshold},
            summary,
            full_report=full_report,
        )

        return summary

    @tool
    def pearson_filter_tool(
        target_class: str = "catB",
        threshold: float = 0.5,
    ) -> Dict[str, Any]:
        """
        Computes Pearson correlation between each term's count in session.feature_matrix and a target class.

        Constructs a One-vs-Rest binary target vector (1 for target_class, 0 otherwise) and computes
        Pearson correlation coefficient r for each term. Replaces NaN correlations (from zero-variance terms) with 0.0.
        Creates a full_report DataFrame with term as 1st column and pearson_r as 2nd column, sorted descending by r.

        Args:
            target_class (str): The category name to correlate terms against (One-vs-Rest). Default is "catB".
            threshold (float): Minimum Pearson correlation r required to keep a term. Default is 0.5.

        Returns:
            dict: A JSON-serializable result summary containing:
                - result_id (str): Unique execution identifier (e.g. "pearson_filter_1").
                - status (str): "success" or "error".
                - target_class (str): Target category evaluated.
                - threshold (float): Minimum correlation threshold.
                - total_terms (int): Total unique vocabulary size evaluated.
                - kept_terms (List[str]): List of term names with pearson_r >= threshold.
                - removed_terms (List[str]): List of term names with pearson_r < threshold.
                - pearson_r (Dict[str, float]): Dict mapping term names to Pearson correlation r coefficients.
        """
        return _run_correlation_filter("pearson_filter", "pearson_r", pearsonr, target_class, threshold)

    @tool
    def spearman_filter_tool(
        target_class: str = "catB",
        threshold: float = 0.5,
    ) -> Dict[str, Any]:
        """
        Computes Spearman rank correlation between each term's count in session.feature_matrix and a target class.

        Constructs a One-vs-Rest binary target vector (1 for target_class, 0 otherwise) and computes
        Spearman rank correlation coefficient r for each term. Replaces NaN correlations with 0.0.
        Creates a full_report DataFrame with term as 1st column and spearman_r as 2nd column, sorted descending by r.

        Args:
            target_class (str): The category name to correlate terms against (One-vs-Rest). Default is "catB".
            threshold (float): Minimum Spearman rank correlation r required to keep a term. Default is 0.5.

        Returns:
            dict: A JSON-serializable result summary containing:
                - result_id (str): Unique execution identifier (e.g. "spearman_filter_1").
                - status (str): "success" or "error".
                - target_class (str): Target category evaluated.
                - threshold (float): Minimum correlation threshold.
                - total_terms (int): Total unique vocabulary size evaluated.
                - kept_terms (List[str]): List of term names with spearman_r >= threshold.
                - removed_terms (List[str]): List of term names with spearman_r < threshold.
                - spearman_r (Dict[str, float]): Dict mapping term names to Spearman correlation r coefficients.
        """
        return _run_correlation_filter("spearman_filter", "spearman_r", spearmanr, target_class, threshold)

    return [variance_filter_tool, pearson_filter_tool, spearman_filter_tool]
