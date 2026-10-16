import os
import pandas as pd
import nltk
import matplotlib
matplotlib.use("Agg")  # Use non-interactive backend for server environment
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Any, List, Optional
from langchain_core.tools import tool

# Ensure punkt sentence tokenizer resources are available if needed
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)

try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab", quiet=True)


def make_tools(session):
    """
    Factory function returning tools for data inspection, cleaning, and tokenization.
    """

    @tool
    def tokenize_tool(
        text_column: str = "text",
        output_column: str = "unigrams",
    ) -> Dict[str, Any]:
        """
        Tokenizes text documents from session.dataframe into lowercased unigrams using NLTK.

        Reads the specified text column from session.dataframe, lowercases the text,
        and uses nltk.word_tokenize to split each document into a list of word tokens (unigrams).
        Empty or missing (NaN/None) text entries produce an empty token list [].
        Stores the resulting token lists in session.dataframe[output_column] and in
        session.artifacts[output_column], and records execution details in session.results_store.

        Args:
            text_column (str): The column name in session.dataframe containing raw text documents. Default is "text".
            output_column (str): The column/artifact name where tokenized lists will be stored. Default is "unigrams".

        Returns:
            dict: A JSON-serializable result summary containing:
                - result_id (str): Unique execution identifier (e.g. "tokenize_1").
                - n_documents (int): Total number of documents processed.
                - total_tokens (int): Total token count across all documents.
                - vocab_size (int): Number of unique unigram tokens discovered.
                - text_column (str): Name of the source text column.
                - output_column (str): Column name where tokens were stored.
        """
        if session.dataframe is None:
            raise ValueError("No dataset loaded in session. Call load_dataset_tool first.")

        if text_column not in session.dataframe.columns:
            raise ValueError(f"Column '{text_column}' not found in session.dataframe.")

        def tokenize_row(text):
            if pd.isna(text) or text is None:
                return []
            text_str = str(text).lower().strip()
            if not text_str:
                return []
            return nltk.word_tokenize(text_str)

        unigrams = session.dataframe[text_column].apply(tokenize_row).tolist()

        # Store results back in session dataframe and artifacts
        session.dataframe[output_column] = unigrams
        session.artifacts[output_column] = unigrams

        # Compute summary metrics
        n_docs = len(unigrams)
        all_tokens = [tok for doc in unigrams for tok in doc]
        total_tokens = len(all_tokens)
        vocab_size = len(set(all_tokens))

        summary = {
            "result_id": session.next_result_id("tokenize"),
            "n_documents": n_docs,
            "total_tokens": total_tokens,
            "vocab_size": vocab_size,
            "text_column": text_column,
            "output_column": output_column,
        }

        session.store_result("tokenize_tool", {"text_column": text_column, "output_column": output_column}, summary)

        return summary

    @tool
    def list_files_tool(
        subdirectory: str = ".",
    ) -> Dict[str, Any]:
        """
        Lists files and directories within the repository/workspace workspace root or a specified subdirectory.

        Scans the target subdirectory and returns sorted lists of subdirectories and file names.
        Includes security checks to ensure the path cannot traverse outside the workspace root.
        If the directory does not exist or traversal is attempted, returns an informative error summary dict.

        Args:
            subdirectory (str): Subdirectory relative to the repository workspace root. Default is "." (root).

        Returns:
            dict: A JSON-serializable summary dictionary containing:
                - result_id (str): Unique execution identifier (e.g. "list_files_1").
                - subdirectory (str): The requested target subdirectory path.
                - directories (List[str]): List of subdirectory names found.
                - files (List[str]): List of file names found.
                - status (str): "success" or "error".
                - error (str, optional): Error message if the directory was invalid or inaccessible.
        """
        root_path = Path(".").resolve()
        target_path = (root_path / subdirectory).resolve()

        # Path traversal protection check
        try:
            target_path.relative_to(root_path)
        except ValueError:
            summary = {
                "result_id": session.next_result_id("list_files"),
                "subdirectory": subdirectory,
                "directories": [],
                "files": [],
                "status": "error",
                "error": f"Access denied: Path '{subdirectory}' attempts to traverse outside workspace root.",
            }
            session.store_result("list_files_tool", {"subdirectory": subdirectory}, summary)
            return summary

        if not target_path.exists() or not target_path.is_dir():
            summary = {
                "result_id": session.next_result_id("list_files"),
                "subdirectory": subdirectory,
                "directories": [],
                "files": [],
                "status": "error",
                "error": f"Directory '{subdirectory}' does not exist or is not a directory.",
            }
            session.store_result("list_files_tool", {"subdirectory": subdirectory}, summary)
            return summary

        directories = []
        files = []

        for item in sorted(target_path.iterdir()):
            if item.is_dir():
                directories.append(item.name)
            elif item.is_file():
                files.append(item.name)

        summary = {
            "result_id": session.next_result_id("list_files"),
            "subdirectory": subdirectory,
            "directories": directories,
            "files": files,
            "status": "success",
        }

        session.store_result("list_files_tool", {"subdirectory": subdirectory}, summary)

        return summary

    @tool
    def check_missing_tool(
        text_column: str = "text",
    ) -> Dict[str, Any]:
        """
        Checks for missing or empty/whitespace-only text documents in session.dataframe.

        Scans the specified text column for NaN, None, empty strings, or whitespace-only entries.
        Reports the total document count, missing count, missing row indices, and percentage of missing entries.

        Args:
            text_column (str): The column in session.dataframe to check for missing values. Default is "text".

        Returns:
            dict: A JSON-serializable result summary containing:
                - result_id (str): Unique execution identifier (e.g. "check_missing_1").
                - n_documents (int): Total number of rows in the dataset.
                - missing_count (int): Number of missing/empty text rows detected.
                - missing_pct (float): Percentage of missing rows (0.0 to 100.0).
                - missing_indices (List[int]): List of 0-based DataFrame index positions that are missing/empty.
                - text_column (str): Name of the evaluated text column.
        """
        if session.dataframe is None:
            raise ValueError("No dataset loaded in session. Call load_dataset_tool first.")

        if text_column not in session.dataframe.columns:
            raise ValueError(f"Column '{text_column}' not found in session.dataframe.")

        series = session.dataframe[text_column]

        # Check for NaN/None OR empty/whitespace-only strings
        is_missing_mask = series.isna() | (series.astype(str).str.strip() == "")

        missing_indices = session.dataframe.index[is_missing_mask].tolist()
        missing_count = len(missing_indices)
        n_documents = len(session.dataframe)
        missing_pct = (missing_count / n_documents * 100.0) if n_documents > 0 else 0.0

        summary = {
            "result_id": session.next_result_id("check_missing"),
            "n_documents": n_documents,
            "missing_count": missing_count,
            "missing_pct": round(missing_pct, 4),
            "missing_indices": missing_indices,
            "text_column": text_column,
        }

        session.store_result("check_missing_tool", {"text_column": text_column}, summary)

        return summary

    @tool
    def check_duplicates_tool(
        subset: List[str] = ["text"],
        drop: bool = False,
    ) -> Dict[str, Any]:
        """
        Checks for (and optionally drops) duplicate documents in session.dataframe.

        Scans session.dataframe for duplicate rows evaluated on the specified column subset.
        When drop=True, directly modifies session.dataframe in-place by removing all duplicate rows
        using keep=False (purging all copies of a duplicated row, matching Master's
        X.drop_duplicates(keep=False, inplace=True) behavior). When drop=False, session.dataframe is left unchanged.

        Args:
            subset (List[str]): List of column names to evaluate for duplicates. Default is ["text"].
            drop (bool): Whether to drop detected duplicates in-place from session.dataframe. Default is False.

        Returns:
            dict: A JSON-serializable result summary containing:
                - result_id (str): Unique execution identifier (e.g. "check_duplicates_1").
                - initial_rows (int): Row count before duplicate processing.
                - duplicate_count (int): Number of duplicate rows detected (using pandas duplicated()).
                - rows_dropped (int): Total number of rows removed when drop=True.
                - remaining_rows (int): Final row count after duplicate processing.
                - dropped (bool): Whether session.dataframe was modified in-place to drop duplicate rows.
                - subset (List[str]): Evaluated column subset.
        """
        if session.dataframe is None:
            raise ValueError("No dataset loaded in session. Call load_dataset_tool first.")

        for col in subset:
            if col not in session.dataframe.columns:
                raise ValueError(f"Column '{col}' not found in session.dataframe.")

        initial_rows = len(session.dataframe)

        # Count duplicate entries (pandas default duplicated() flags extra copies)
        duplicate_mask = session.dataframe.duplicated(subset=subset)
        duplicate_count = int(duplicate_mask.sum())

        rows_dropped = 0
        if drop:
            # Use keep=False as required by spec (drops ALL copies of duplicated rows)
            before_drop = len(session.dataframe)
            session.dataframe.drop_duplicates(subset=subset, keep=False, inplace=True)
            # Reindex after dropping to maintain clean contiguous indices
            session.dataframe.reset_index(drop=True, inplace=True)
            after_drop = len(session.dataframe)
            rows_dropped = before_drop - after_drop

        remaining_rows = len(session.dataframe)

        summary = {
            "result_id": session.next_result_id("check_duplicates"),
            "initial_rows": initial_rows,
            "duplicate_count": duplicate_count,
            "rows_dropped": rows_dropped,
            "remaining_rows": remaining_rows,
            "dropped": drop,
            "subset": subset,
        }

        session.store_result("check_duplicates_tool", {"subset": subset, "drop": drop}, summary)

        return summary

    @tool
    def sample_data_tool(
        n: int = 4,
        random_state: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Subsamples rows from session.dataframe and updates session.dataframe in-place.

        Extracts a random sample of n rows from session.dataframe using pandas DataFrame.sample().
        Accepts random_state for reproducible output matching TEST_FIXTURE.md known answer #13.
        Updates session.dataframe in-place to contain the sampled subset and resets its index.

        Args:
            n (int): Number of rows to sample. Default is 4.
            random_state (Optional[int]): Random seed for reproducible sampling. Default is None.

        Returns:
            dict: A JSON-serializable result summary containing:
                - result_id (str): Unique execution identifier (e.g. "sample_data_1").
                - initial_rows (int): Original row count before sampling.
                - sampled_rows (int): Number of rows in the sampled subset.
                - sampled_indices (List[int]): Original 0-based DataFrame row indices that were sampled.
                - random_state (Optional[int]): The random seed used.
        """
        if session.dataframe is None:
            raise ValueError("No dataset loaded in session. Call load_dataset_tool first.")

        initial_rows = len(session.dataframe)
        actual_n = min(n, initial_rows)

        if actual_n <= 0:
            raise ValueError(f"Sample size n={n} must be greater than 0.")

        sampled_df = session.dataframe.sample(n=actual_n, random_state=random_state)
        sampled_indices = sampled_df.index.tolist()

        # Update session dataframe in-place and reset index
        session.dataframe = sampled_df.reset_index(drop=True)

        summary = {
            "result_id": session.next_result_id("sample_data"),
            "initial_rows": initial_rows,
            "sampled_rows": len(session.dataframe),
            "sampled_indices": sampled_indices,
            "random_state": random_state,
        }

        session.store_result("sample_data_tool", {"n": n, "random_state": random_state}, summary)

        return summary

    @tool
    def inspect_data_tool(
        n_rows: int = 5,
    ) -> Dict[str, Any]:
        """
        Peeks at the current session.dataframe and returns dataset shape, column information, and row previews.

        Returns dataset dimensions (shape), column names, data types, class/category distributions, and the
        first n_rows rows as dictionaries with any text values truncated to a maximum of 200 characters.
        If no dataset is loaded, returns a structured error summary instead of crashing.

        Args:
            n_rows (int): Number of top rows to preview. Default is 5.

        Returns:
            dict: A JSON-serializable result summary containing:
                - result_id (str): Unique execution identifier (e.g. "inspect_data_1").
                - status (str): "success" or "error".
                - shape (List[int]): Dataset dimensions [n_rows, n_cols].
                - columns (List[str]): List of column names in session.dataframe.
                - dtypes (Dict[str, str]): Mapping of column names to string data types.
                - category_counts (Optional[Dict[str, int]]): Value counts of "category_name" or label column if available.
                - head (List[Dict[str, Any]]): List of top n_rows row dicts with text strings truncated to 200 chars.
        """
        if session.dataframe is None:
            summary = {
                "result_id": session.next_result_id("inspect_data"),
                "status": "error",
                "error": "No dataset loaded in session. Call load_dataset_tool first.",
                "shape": [0, 0],
                "columns": [],
                "dtypes": {},
                "category_counts": None,
                "head": [],
            }
            session.store_result("inspect_data_tool", {"n_rows": n_rows}, summary)
            return summary

        df = session.dataframe
        shape = [int(df.shape[0]), int(df.shape[1])]
        columns = df.columns.tolist()
        dtypes = {col: str(dtype) for col, dtype in df.dtypes.items()}

        # Category distribution counts if category_name or label column exists
        category_counts = None
        if "category_name" in df.columns:
            category_counts = {str(k): int(v) for k, v in df["category_name"].value_counts().to_dict().items()}
        elif "label" in df.columns:
            category_counts = {str(k): int(v) for k, v in df["label"].value_counts().to_dict().items()}

        # Preview top n_rows rows with text values truncated to 200 characters
        top_df = df.head(max(0, n_rows)).copy()
        head_records = []

        for row in top_df.to_dict(orient="records"):
            formatted_row = {}
            for col, val in row.items():
                if pd.isna(val) or val is None:
                    formatted_row[col] = None
                elif isinstance(val, str):
                    formatted_row[col] = val[:200]
                elif isinstance(val, (list, tuple)):
                    formatted_row[col] = [str(x)[:200] if isinstance(x, str) else x for x in val]
                else:
                    formatted_row[col] = val
            head_records.append(formatted_row)

        summary = {
            "result_id": session.next_result_id("inspect_data"),
            "status": "success",
            "shape": shape,
            "columns": columns,
            "dtypes": dtypes,
            "category_counts": category_counts,
            "head": head_records,
        }

        session.store_result("inspect_data_tool", {"n_rows": n_rows}, summary)

        return summary

    @tool
    def describe_data_tool(
        text_column: str = "text",
        category_column: str = "category_name",
    ) -> Dict[str, Any]:
        """
        Computes text length statistics and generates a box plot grouped by category.

        Calculates character length text_length = df[text_column].apply(len) (treating missing/NaN as length 0).
        Computes standard .describe() summary statistics overall and broken down by category_column.
        Generates a seaborn box plot of text_length grouped by category_column and stores the matplotlib
        figure object on session.pending_figure before returning summary statistics.

        Args:
            text_column (str): The column containing document text. Default is "text".
            category_column (str): The column containing category labels. Default is "category_name".

        Returns:
            dict: A JSON-serializable result summary containing:
                - result_id (str): Unique execution identifier (e.g. "describe_data_1").
                - overall_stats (Dict[str, float]): Overall text length describe() stats.
                - by_category_stats (Dict[str, Dict[str, float]]): Describe() stats per category.
                - text_column (str): Name of evaluated text column.
                - category_column (str): Name of evaluated category column.
        """
        if session.dataframe is None:
            raise ValueError("No dataset loaded in session. Call load_dataset_tool first.")

        if text_column not in session.dataframe.columns:
            raise ValueError(f"Column '{text_column}' not found in session.dataframe.")

        df = session.dataframe.copy()

        # Compute text length in characters (missing or NaN evaluates to length 0)
        df["text_length"] = df[text_column].apply(lambda x: len(str(x)) if (pd.notna(x) and x is not None) else 0)

        # Compute overall stats using pandas describe()
        overall_describe = df["text_length"].describe()
        overall_stats = {str(k): float(v) for k, v in overall_describe.to_dict().items()}

        # Compute stats per category if category_column exists
        by_category_stats = {}
        has_category = category_column in df.columns

        if has_category:
            grouped = df.groupby(category_column)["text_length"]
            for cat, group in grouped:
                by_category_stats[str(cat)] = {str(k): float(v) for k, v in group.describe().to_dict().items()}

        # Generate box plot of text_length grouped by category
        fig, ax = plt.subplots(figsize=(8, 5))
        if has_category:
            sns.boxplot(data=df, x=category_column, y="text_length", ax=ax)
            ax.set_title(f"Text Length Distribution by {category_column}")
            ax.set_xlabel(category_column)
        else:
            sns.boxplot(data=df, y="text_length", ax=ax)
            ax.set_title("Text Length Distribution")

        ax.set_ylabel("Text Length (characters)")
        fig.tight_layout()

        # Set matplotlib Figure object directly on session.pending_figure
        session.pending_figure = fig
        plt.close(fig)

        summary = {
            "result_id": session.next_result_id("describe_data"),
            "overall_stats": overall_stats,
            "by_category_stats": by_category_stats,
            "text_column": text_column,
            "category_column": category_column,
        }

        session.store_result("describe_data_tool", {"text_column": text_column, "category_column": category_column}, summary)

        return summary

    return [
        tokenize_tool,
        list_files_tool,
        check_missing_tool,
        check_duplicates_tool,
        sample_data_tool,
        inspect_data_tool,
        describe_data_tool,
    ]
