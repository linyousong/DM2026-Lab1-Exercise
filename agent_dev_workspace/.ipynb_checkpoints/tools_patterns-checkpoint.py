import os
import tempfile
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Literal
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from langchain_core.tools import tool

# PAMI imports for transactional mining
try:
    from PAMI.extras.convert import DF2DB
    from PAMI.frequentPattern.basic import FPGrowth
    from PAMI.frequentPattern.topk import FAE
    from PAMI.frequentPattern.maximal import MaxFPGrowth
except ImportError:
    DF2DB = None
    FPGrowth = None
    FAE = None
    MaxFPGrowth = None


def make_tools(session):
    """
    Factory function returning tools for pattern mining.
    """

    @tool
    def mine_patterns_tool(
        category_name: str,
        filtering_method: Literal["variance", "tfidf", "term_frequency"] = "tfidf",
        algorithm: Literal["fpgrowth", "topk", "maxfpgrowth"] = "fpgrowth",
        min_sup: Optional[int] = 1,
        k: Optional[int] = 10,
    ) -> Dict[str, Any]:
        """
        Mines frequent itemsets/patterns for a specific category using PAMI algorithms.

        Filters session.dataframe to category_name, builds a fresh category-local DTM with English stop words removed,
        applies the selected vocabulary filter (variance, tfidf, or term_frequency), converts the surviving terms into a
        transactional database, and runs the requested PAMI mining algorithm (fpgrowth, topk, or maxfpgrowth).

        Args:
            category_name (str): Category label to filter by (e.g. "catA").
            filtering_method (Literal["variance", "tfidf", "term_frequency"]): Filter to apply to category vocabulary.
                Default is "tfidf".
            algorithm (Literal["fpgrowth", "topk", "maxfpgrowth"]): PAMI mining algorithm to run. Default is "fpgrowth".
            min_sup (Optional[int]): Minimum support threshold for fpgrowth and maxfpgrowth. Default is 1.
            k (Optional[int]): Number of top patterns to return for topk algorithm. Default is 10.

        Returns:
            dict: Result summary containing:
                - result_id (str): Unique result identifier (e.g. "mine_patterns_1").
                - status (str): "success" or "error".
                - category_name (str): The mined category.
                - filtering_method (str): Filtering method used.
                - algorithm (str): Mining algorithm used.
                - patterns (List[Dict[str, Any]]): List of pattern dicts with keys "pattern" (str/list) and "support" (int).
                - n_patterns (int): Number of patterns found.
                - note (Optional[str]): Informative note if 0 terms survived or 0 patterns were found.
        """
        result_id = session.next_result_id("mine_patterns")

        if session.dataframe is None:
            summary = {
                "result_id": result_id,
                "status": "error",
                "error": "No dataframe found in session state. Call load_dataset_tool first.",
                "category_name": category_name,
                "filtering_method": filtering_method,
                "algorithm": algorithm,
                "patterns": [],
                "n_patterns": 0,
                "note": None,
            }
            session.store_result("mine_patterns_tool", {"category_name": category_name}, summary)
            return summary

        df = session.dataframe

        # Standardize column lookup for category
        cat_col = None
        if "category_name" in df.columns:
            cat_col = "category_name"
        elif "label" in df.columns:
            cat_col = "label"
        elif "category" in df.columns:
            cat_col = "category"

        if cat_col is None:
            summary = {
                "result_id": result_id,
                "status": "error",
                "error": "No category/label column found in dataframe.",
                "category_name": category_name,
                "filtering_method": filtering_method,
                "algorithm": algorithm,
                "patterns": [],
                "n_patterns": 0,
                "note": None,
            }
            session.store_result("mine_patterns_tool", {"category_name": category_name}, summary)
            return summary

        # =========================================================================
        # STEP 1: Filter to category & build fresh category-local DTM
        # =========================================================================
        cat_df = df[df[cat_col].astype(str) == category_name]
        if len(cat_df) == 0:
            summary = {
                "result_id": result_id,
                "status": "error",
                "error": f"Category '{category_name}' not found in dataframe.",
                "category_name": category_name,
                "filtering_method": filtering_method,
                "algorithm": algorithm,
                "patterns": [],
                "n_patterns": 0,
                "note": None,
            }
            session.store_result("mine_patterns_tool", {"category_name": category_name}, summary)
            return summary

        # Get text column
        text_col = "text" if "text" in cat_df.columns else cat_df.columns[0]
        texts = cat_df[text_col].fillna("").tolist()

        # Build fresh local DTM with stop_words="english"
        vectorizer = CountVectorizer(stop_words="english")
        try:
            dtm_sparse = vectorizer.fit_transform(texts)
            feature_names = np.array(vectorizer.get_feature_names_out())
        except ValueError:
            # Handle case where texts have no words or only stopwords
            feature_names = np.array([])
            dtm_sparse = None

        if dtm_sparse is None or len(feature_names) == 0:
            summary = {
                "result_id": result_id,
                "status": "success",
                "category_name": category_name,
                "filtering_method": filtering_method,
                "algorithm": algorithm,
                "patterns": [],
                "n_patterns": 0,
                "note": "no terms survived stopword removal for this category",
            }
            artifact_key = f"patterns_{category_name}_{filtering_method}_{algorithm}"
            session.artifacts[artifact_key] = []
            session.store_result("mine_patterns_tool", {"category_name": category_name}, summary)
            return summary

        dtm_dense = dtm_sparse.toarray()

        # =========================================================================
        # STEP 2: Apply category vocabulary filtering
        # =========================================================================
        surviving_mask = np.zeros(len(feature_names), dtype=bool)

        if filtering_method == "variance":
            variances = np.var(dtm_dense, axis=0)
            p5 = np.percentile(variances, 5)
            p95 = np.percentile(variances, 95)
            surviving_mask = (variances > p5) & (variances < p95)
        elif filtering_method == "term_frequency":
            freqs = np.sum(dtm_dense, axis=0)
            p5 = np.percentile(freqs, 5)
            p95 = np.percentile(freqs, 95)
            surviving_mask = (freqs > p5) & (freqs < p95)
        elif filtering_method == "tfidf":
            tfidf_transformer = TfidfTransformer()
            tfidf_matrix = tfidf_transformer.fit_transform(dtm_dense).toarray()
            mean_tfidf = np.mean(tfidf_matrix, axis=0)
            p20 = np.percentile(mean_tfidf, 20)
            surviving_mask = mean_tfidf >= p20

        surviving_terms = feature_names[surviving_mask]
        surviving_dtm = dtm_dense[:, surviving_mask]

        # =========================================================================
        # STEP 3: Handle zero survived terms
        # =========================================================================
        if len(surviving_terms) == 0:
            summary = {
                "result_id": result_id,
                "status": "success",
                "category_name": category_name,
                "filtering_method": filtering_method,
                "algorithm": algorithm,
                "patterns": [],
                "n_patterns": 0,
                "note": "no terms survived filtering",
            }
            artifact_key = f"patterns_{category_name}_{filtering_method}_{algorithm}"
            session.artifacts[artifact_key] = []
            session.store_result("mine_patterns_tool", {"category_name": category_name}, summary)
            return summary

        # =========================================================================
        # STEP 4: Convert to transactional DB & Mine Patterns via PAMI
        # =========================================================================
        # Build a DataFrame with surviving terms
        surviving_df = pd.DataFrame(surviving_dtm, columns=surviving_terms)

        # Create temporary file for transactional DB
        with tempfile.NamedTemporaryFile(mode="w+", delete=False, suffix=".csv") as tmp_file:
            tmp_db_path = tmp_file.name

        try:
            # Convert DataFrame to Transactional Database using PAMI DF2DB
            df2db = DF2DB.DF2DB(surviving_df, threshold=1)
            df2db.convert2TransactionalDatabase(tmp_db_path)

            # Mine patterns using requested algorithm
            raw_patterns = {}
            effective_min_sup = min_sup if min_sup is not None else 1
            effective_k = k if k is not None else 10

            if algorithm == "fpgrowth":
                obj = FPGrowth.FPGrowth(iFile=tmp_db_path, minSup=effective_min_sup, sep="\t")
                obj.startMine()
                raw_patterns = obj.getPatterns()
            elif algorithm == "topk":
                obj = FAE.FAE(iFile=tmp_db_path, k=effective_k, sep="\t")
                obj.startMine()
                raw_patterns = obj.getPatterns()
            elif algorithm == "maxfpgrowth":
                obj = MaxFPGrowth.MaxFPGrowth(iFile=tmp_db_path, minSup=effective_min_sup, sep="\t")
                obj.startMine()
                raw_patterns = obj.getPatterns()
        finally:
            if os.path.exists(tmp_db_path):
                os.remove(tmp_db_path)

        # =========================================================================
        # STEP 5: Format patterns & Store in session state and artifacts
        # =========================================================================
        formatted_patterns = []
        if raw_patterns:
            for itemset, sup in raw_patterns.items():
                # Format pattern itemset as list or space-separated string
                if isinstance(itemset, str):
                    pattern_items = itemset.strip().split()
                else:
                    pattern_items = list(itemset)
                formatted_patterns.append({
                    "pattern": pattern_items,
                    "support": int(sup),
                })

        # Sort patterns by support descending
        formatted_patterns.sort(key=lambda x: x["support"], reverse=True)

        artifact_key = f"patterns_{category_name}_{filtering_method}_{algorithm}"
        session.artifacts[artifact_key] = formatted_patterns

        note = None
        if len(formatted_patterns) == 0:
            note = f"No patterns found by algorithm '{algorithm}' with current parameters."

        summary = {
            "result_id": result_id,
            "status": "success",
            "category_name": category_name,
            "filtering_method": filtering_method,
            "algorithm": algorithm,
            "patterns": formatted_patterns,
            "n_patterns": len(formatted_patterns),
            "note": note,
        }

        session.store_result(
            "mine_patterns_tool",
            {
                "category_name": category_name,
                "filtering_method": filtering_method,
                "algorithm": algorithm,
                "min_sup": min_sup,
                "k": k,
            },
            summary,
        )

        return summary

    return [mine_patterns_tool]
