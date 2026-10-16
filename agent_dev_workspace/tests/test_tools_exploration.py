import pytest
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_exploration import make_tools
from tools_dtm import make_tools as make_dtm_tools


def test_cosine_similarity_tool_known_answer_16():
    """
    Tests cosine_similarity_tool against TEST_FIXTURE.md known answer #16.
    Fixture file: agent_dev/sample_fixture.csv
    Prerequisite: must invoke load_dataset_tool then build_dtm_tool first.
    Expected pairwise cosine similarities:
      - Doc 0 vs 1 (both catA, near-identical text): 0.9847
      - Doc 0 vs 3 (catA vs catB, share only "always"): 0.0909
      - Doc 3 vs 4 (both catB, near-identical text): 0.9847
      - Doc 0 vs 7 (catA vs empty-text row): 0.0
    """
    session = SessionState()

    # 1. Load dataset using premade load_dataset_tool
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    # 2. Build DTM to populate session.feature_matrix and session.artifacts["count_vectorizer"]
    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    build_dtm_tool.invoke({"text_column": "text"})

    # 3. Get student cosine_similarity_tool from make_tools(session)
    exploration_tools = make_tools(session)
    cosine_similarity_tool = next(t for t in exploration_tools if t.name == "cosine_similarity_tool")

    # Pair 1: Doc 0 vs 1 (0.9847)
    res_0_1 = cosine_similarity_tool.invoke({"doc1": 0, "doc2": 1})
    assert res_0_1["status"] == "success"
    assert res_0_1["similarity"] == pytest.approx(0.9847, abs=1e-3)

    # Pair 2: Doc 0 vs 3 (0.0909)
    res_0_3 = cosine_similarity_tool.invoke({"doc1": 0, "doc2": 3})
    assert res_0_3["status"] == "success"
    assert res_0_3["similarity"] == pytest.approx(0.0909, abs=1e-3)

    # Pair 3: Doc 3 vs 4 (0.9847)
    res_3_4 = cosine_similarity_tool.invoke({"doc1": 3, "doc2": 4})
    assert res_3_4["status"] == "success"
    assert res_3_4["similarity"] == pytest.approx(0.9847, abs=1e-3)

    # Pair 4: Doc 0 vs 7 (empty-text row -> 0.0)
    res_0_7 = cosine_similarity_tool.invoke({"doc1": 0, "doc2": 7})
    assert res_0_7["status"] == "success"
    assert res_0_7["similarity"] == 0.0


def test_feature_correlation_matrix_tool_known_answer_19():
    """
    Tests feature_correlation_matrix_tool against TEST_FIXTURE.md known answer #19.
    Fixture file: agent_dev/sample_fixture.csv
    Prerequisite: must invoke load_dataset_tool then build_dtm_tool first.
    Expected top terms (sorted by variance descending):
      ['alpha', 'gamma', 'delta', 'beta', 'always']
    Expected correlation matrix key values:
      - Diagonal terms against self: 1.0000
      - alpha vs beta (0 vs 3): 0.8885
      - gamma vs delta (1 vs 2): 0.8307
      - session.pending_figure is set.
    """
    session = SessionState()

    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    build_dtm_tool.invoke({"text_column": "text"})

    exploration_tools = make_tools(session)
    feature_correlation_matrix_tool = next(t for t in exploration_tools if t.name == "feature_correlation_matrix_tool")

    # Invoke feature_correlation_matrix_tool with top_n=20
    result = feature_correlation_matrix_tool.invoke({"top_n": 20})

    # 1. Verify pending figure was set
    assert session.pending_figure is not None

    # 2. Verify top terms ordering by variance descending (Known Answer #19)
    expected_top_terms = ['alpha', 'gamma', 'delta', 'beta', 'always']
    assert result["status"] == "success"
    assert result["top_terms"] == expected_top_terms
    assert result["n_terms"] == 5

    corr_matrix = result["correlation_matrix"]

    # 3. Verify diagonal is 1.0000 for every term against itself
    for i in range(5):
        assert corr_matrix[i][i] == pytest.approx(1.0000, abs=1e-3)

    # 4. Verify alpha vs beta (alpha is idx 0, beta is idx 3 in top_terms): 0.8885
    idx_alpha = expected_top_terms.index("alpha")
    idx_beta = expected_top_terms.index("beta")
    assert corr_matrix[idx_alpha][idx_beta] == pytest.approx(0.8885, abs=1e-3)
    assert corr_matrix[idx_beta][idx_alpha] == pytest.approx(0.8885, abs=1e-3)

    # 5. Verify gamma vs delta (gamma is idx 1, delta is idx 2 in top_terms): 0.8307
    idx_gamma = expected_top_terms.index("gamma")
    idx_delta = expected_top_terms.index("delta")
    assert corr_matrix[idx_gamma][idx_delta] == pytest.approx(0.8307, abs=1e-3)
    assert corr_matrix[idx_delta][idx_gamma] == pytest.approx(0.8307, abs=1e-3)
