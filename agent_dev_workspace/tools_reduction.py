import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server environment
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List, Optional, Literal
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
from langchain_core.tools import tool


def make_tools(session):
    """
    Factory function returning tools for dimensionality reduction and label binarization.
    """

    @tool
    def reduce_dimensions_tool(
        method: Literal["pca", "tsne", "umap"] = "pca",
        n_components: int = 2,
        perplexity: float = 30.0,
        n_neighbors: int = 15,
        random_state: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Reduces Document-Term Matrix (session.feature_matrix) dimensions to 2D coordinates using PCA, t-SNE, or UMAP.

        Applies PCA, t-SNE, or UMAP dimensionality reduction on session.feature_matrix.
        Builds a scatter plot of the 2D reduced coordinates colored by category and sets it on session.pending_figure.
        Stores reduced coordinates in session.artifacts["reduced_{method}"] and returns summary statistics.

        Args:
            method (Literal["pca", "tsne", "umap"]): Reduction algorithm to apply. Default is "pca".
            n_components (int): Target dimension size. Default is 2.
            perplexity (float): Perplexity parameter for t-SNE. Must be less than n_samples. Default is 30.0.
            n_neighbors (int): Number of neighbors parameter for UMAP. Must be less than n_samples. Default is 15.
            random_state (Optional[int]): Random seed for reproducible output. Default is None.

        Returns:
            dict: A JSON-serializable result summary containing:
                - result_id (str): Unique execution identifier (e.g. "reduce_dimensions_1").
                - status (str): "success" or "error".
                - method (str): The reduction method used ("pca", "tsne", or "umap").
                - coordinates (List[List[float]]): 2D reduced coordinates per document rounded to 4 decimal places.
                - explained_variance_ratio (Optional[List[float]]): Variance ratio for PCA (None for t-SNE and UMAP).
                - n_documents (int): Number of reduced documents.
        """
        if session.feature_matrix is None or session.feature_names is None:
            summary = {
                "result_id": session.next_result_id("reduce_dimensions"),
                "status": "error",
                "error": "No document-term matrix found in session. Call build_dtm_tool first.",
                "method": method,
                "coordinates": [],
                "explained_variance_ratio": None,
                "n_documents": 0,
            }
            session.store_result("reduce_dimensions_tool", {"method": method}, summary)
            return summary

        if method not in ["pca", "tsne", "umap"]:
            summary = {
                "result_id": session.next_result_id("reduce_dimensions"),
                "status": "error",
                "error": f"Method '{method}' is not supported. Supported methods are 'pca', 'tsne', and 'umap'.",
                "method": method,
                "coordinates": [],
                "explained_variance_ratio": None,
                "n_documents": 0,
            }
            session.store_result("reduce_dimensions_tool", {"method": method}, summary)
            return summary

        dense_matrix = session.feature_matrix.toarray()
        n_samples = dense_matrix.shape[0]

        if method == "tsne" and perplexity >= n_samples:
            summary = {
                "result_id": session.next_result_id("reduce_dimensions"),
                "status": "error",
                "error": f"perplexity must be less than n_samples ({n_samples}), got {perplexity}.",
                "method": method,
                "coordinates": [],
                "explained_variance_ratio": None,
                "n_documents": n_samples,
            }
            session.store_result("reduce_dimensions_tool", {"method": method, "perplexity": perplexity}, summary)
            return summary

        if method == "umap" and n_neighbors >= n_samples:
            summary = {
                "result_id": session.next_result_id("reduce_dimensions"),
                "status": "error",
                "error": f"n_neighbors must be less than n_samples ({n_samples}), got {n_neighbors}.",
                "method": method,
                "coordinates": [],
                "explained_variance_ratio": None,
                "n_documents": n_samples,
            }
            session.store_result("reduce_dimensions_tool", {"method": method, "n_neighbors": n_neighbors}, summary)
            return summary

        if method == "pca":
            model = PCA(n_components=n_components, random_state=random_state)
            coords = model.fit_transform(dense_matrix)
            explained_variance_ratio = [float(round(v, 4)) for v in model.explained_variance_ratio_]
        elif method == "tsne":
            model = TSNE(
                n_components=n_components,
                perplexity=perplexity,
                random_state=random_state,
            )
            coords = model.fit_transform(dense_matrix)
            explained_variance_ratio = None
        elif method == "umap":
            model = umap.UMAP(
                n_components=n_components,
                n_neighbors=n_neighbors,
                random_state=random_state,
            )
            coords = model.fit_transform(dense_matrix)
            explained_variance_ratio = None

        coords_rounded = np.round(coords, 4).tolist()

        # Store in session artifacts
        session.artifacts[f"reduced_{method}"] = coords_rounded

        # Get category labels for plot coloring
        category_labels = None
        if session.dataframe is not None and "category_name" in session.dataframe.columns:
            category_labels = session.dataframe["category_name"].tolist()
        elif session.dataframe is not None and "label" in session.dataframe.columns:
            category_labels = session.dataframe["label"].tolist()
        elif session.labels is not None:
            category_labels = list(session.labels)
        else:
            category_labels = ["Unlabeled"] * n_samples

        # Create scatter plot of 2D coordinates colored by category
        fig, ax = plt.subplots(figsize=(8, 6))
        plot_df = pd.DataFrame({
            "Component 1": coords[:, 0],
            "Component 2": coords[:, 1] if coords.shape[1] > 1 else np.zeros(n_samples),
            "Category": category_labels,
        })

        sns.scatterplot(
            data=plot_df,
            x="Component 1",
            y="Component 2",
            hue="Category",
            style="Category",
            s=100,
            ax=ax,
        )
        ax.set_title(f"2D Dimension Reduction ({method.upper()})")
        ax.grid(True, linestyle="--", alpha=0.5)
        fig.tight_layout()

        # REQUIRED PLOTTING CONTRACT: Set matplotlib Figure directly on session.pending_figure
        session.pending_figure = fig
        plt.close(fig)

        summary = {
            "result_id": session.next_result_id("reduce_dimensions"),
            "status": "success",
            "method": method,
            "coordinates": coords_rounded,
            "explained_variance_ratio": explained_variance_ratio,
            "n_documents": n_samples,
        }

        session.store_result(
            "reduce_dimensions_tool",
            {
                "method": method,
                "n_components": n_components,
                "perplexity": perplexity,
                "n_neighbors": n_neighbors,
                "random_state": random_state,
            },
            summary,
        )

        return summary

    @tool
    def binarize_labels_tool(
        column_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        One-hot encodes category labels into a binary indicator matrix (N x K), with one column per unique category.

        Reads target labels from session.dataframe or session.labels, performs one-hot binarization (N x K),
        stores the resulting binary matrix in session.artifacts["binarized_labels"], updates session.categories,
        and returns a summary of classes, shapes, and class counts.

        Args:
            column_name (Optional[str]): DataFrame column to binarize. If None, checks 'category_name', 'label',
                or 'category' in session.dataframe, falling back to session.labels.

        Returns:
            dict: Summary containing:
                - result_id (str): Unique result identifier (e.g. "binarize_labels_1").
                - status (str): "success" or "error".
                - n_documents (int): Number of documents encoded.
                - classes (List[str]): List of category column names in binary matrix order.
                - shape (List[int]): Dimensions of the binary indicator matrix [N, K].
                - class_counts (Dict[str, int]): Count of documents per category.
                - has_missing (bool): True if missing/null labels were present.
                - note (Optional[str]): Additional note if single-class dataset or missing values encountered.
        """
        if session.dataframe is None and session.labels is None:
            summary = {
                "result_id": session.next_result_id("binarize_labels"),
                "status": "error",
                "error": "No dataframe or labels found in session state. Load a dataset first.",
                "n_documents": 0,
                "classes": [],
                "shape": [0, 0],
                "class_counts": {},
                "has_missing": False,
                "note": None,
            }
            session.store_result("binarize_labels_tool", {"column_name": column_name}, summary)
            return summary

        labels_series = None

        if session.dataframe is not None:
            df = session.dataframe
            if column_name is not None and column_name in df.columns:
                labels_series = df[column_name]
            elif "category_name" in df.columns:
                labels_series = df["category_name"]
            elif "label" in df.columns:
                labels_series = df["label"]
            elif "category" in df.columns:
                labels_series = df["category"]

        if labels_series is None and session.labels is not None:
            labels_series = pd.Series(session.labels)

        if labels_series is None:
            summary = {
                "result_id": session.next_result_id("binarize_labels"),
                "status": "error",
                "error": f"Column '{column_name}' not found in dataframe and no default label column found.",
                "n_documents": 0,
                "classes": [],
                "shape": [0, 0],
                "class_counts": {},
                "has_missing": False,
                "note": None,
            }
            session.store_result("binarize_labels_tool", {"column_name": column_name}, summary)
            return summary

        n_documents = len(labels_series)
        has_missing = bool(labels_series.isna().any() or (labels_series == "").any())

        # Determine unique non-null classes in sorted order
        clean_series = labels_series.dropna().astype(str)
        clean_series = clean_series[clean_series != ""]

        unique_classes = sorted(list(clean_series.unique()))

        if len(unique_classes) == 0:
            summary = {
                "result_id": session.next_result_id("binarize_labels"),
                "status": "error",
                "error": "No valid non-null category labels found to binarize.",
                "n_documents": n_documents,
                "classes": [],
                "shape": [n_documents, 0],
                "class_counts": {},
                "has_missing": True,
                "note": "All label values were null or empty strings.",
            }
            session.store_result("binarize_labels_tool", {"column_name": column_name}, summary)
            return summary

        # Perform one-hot encoding (N x K), with one column per category
        # Using dummy_na=False ensures missing rows get encoded as [0, 0, ..., 0]
        binary_df = pd.get_dummies(labels_series.astype(str), dtype=int)

        # Ensure only columns for valid unique_classes exist and are ordered
        for cls in unique_classes:
            if cls not in binary_df.columns:
                binary_df[cls] = 0

        binary_df = binary_df[unique_classes]
        binary_matrix = binary_df.to_numpy()

        # Update session state
        session.artifacts["binarized_labels"] = binary_matrix
        session.categories = unique_classes

        class_counts = {cls: int((binary_df[cls] == 1).sum()) for cls in unique_classes}
        shape = [n_documents, len(unique_classes)]

        note = None
        if len(unique_classes) == 1:
            note = f"Single class dataset detected ('{unique_classes[0]}'). Binarized matrix shape is {shape}."
        elif has_missing:
            note = "Missing labels were encountered and encoded as all-zero rows."

        summary = {
            "result_id": session.next_result_id("binarize_labels"),
            "status": "success",
            "n_documents": n_documents,
            "classes": unique_classes,
            "shape": shape,
            "class_counts": class_counts,
            "has_missing": has_missing,
            "note": note,
        }

        session.store_result("binarize_labels_tool", {"column_name": column_name}, summary)
        return summary

    return [reduce_dimensions_tool, binarize_labels_tool]
