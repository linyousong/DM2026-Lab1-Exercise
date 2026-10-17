import pytest
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_dtm import make_tools


def test_build_dtm_tool_known_answer_4():
    """
    Tests build_dtm_tool against TEST_FIXTURE.md known answer #4.
    Fixture file: agent_dev/sample_fixture.csv
    Expected results:
      - Vocabulary (5 terms): ['alpha', 'always', 'beta', 'delta', 'gamma']
      - Sparsity: 21 non-zero cells out of 40 total (8 rows * 5 columns)
      - sparsity_pct = 100 * (1 - 21/40) = 47.5
      - Vectorizer is stored at session.artifacts["count_vectorizer"]
      - Labels and categories are set on session via session.set_labels(...)
    """
    session = SessionState()

    # Load premade load_dataset_tool
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    # Load sample fixture CSV
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    # Get student build_dtm_tool from make_tools(session)
    tools = make_tools(session)
    build_dtm_tool = next(t for t in tools if t.name == "build_dtm_tool")

    # Invoke build_dtm_tool
    result = build_dtm_tool.invoke({"text_column": "text"})

    # 1. Assert feature names and dimensions
    expected_vocab = ['alpha', 'always', 'beta', 'delta', 'gamma']
    assert result["n_terms"] == len(expected_vocab)
    assert result["n_documents"] == 8
    assert result["n_terms"] == 5

    # 2. Assert matrix sparsity metrics against Known Answer #4
    assert result["non_zero"] == 21
    assert result["total_elements"] == 40
    assert pytest.approx(result["sparsity_pct"], abs=1e-3) == 47.5

    # 3. Assert session attributes were populated
    assert session.feature_matrix is not None
    assert session.feature_matrix.shape == (8, 5)
    assert session.feature_names == expected_vocab

    # 4. Assert session.labels and session.categories got set
    assert session.labels is not None
    assert len(session.labels) == 8
    assert session.categories == ["catA", "catB"]

    # 5. Assert CountVectorizer object is stored in session.artifacts
    assert "count_vectorizer" in session.artifacts
    assert session.artifacts["count_vectorizer"] is not None
    assert hasattr(session.artifacts["count_vectorizer"], "get_feature_names_out")


def test_term_frequency_tool_known_answer_14():
    """
    Tests term_frequency_tool against TEST_FIXTURE.md known answer #14.
    Fixture file: agent_dev/sample_fixture.csv
    Prerequisite: must invoke load_dataset_tool then build_dtm_tool first.
    Expected term frequencies:
      - alpha: 6
      - always: 7
      - beta: 3
      - delta: 4
      - gamma: 7
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

    # 2. Get student tools from make_tools(session)
    tools = make_tools(session)
    build_dtm_tool = next(t for t in tools if t.name == "build_dtm_tool")
    term_frequency_tool = next(t for t in tools if t.name == "term_frequency_tool")

    # 3. Must run build_dtm_tool first to populate session.feature_matrix
    build_dtm_tool.invoke({"text_column": "text"})

    # 4. Invoke term_frequency_tool
    result = term_frequency_tool.invoke({})

    # Assertions based on TEST_FIXTURE.md known answer #14
    assert result["status"] == "success"
    assert result["total_terms"] == 5
    assert result["total_word_occurrences"] == 27  # 6 + 7 + 3 + 4 + 7

    expected_freqs = {
        "alpha": 6,
        "always": 7,
        "beta": 3,
        "delta": 4,
        "gamma": 7,
    }
    assert result["top_terms"] == expected_freqs
    assert session.artifacts["term_frequencies"] == expected_freqs


def test_dtm_heatmap_tool_known_answer_21():
    """
    Tests dtm_heatmap_tool against TEST_FIXTURE.md known answer #21.
    Fixture file: agent_dev/sample_fixture.csv
    Checks:
      1. Default call (n_terms=20, n_documents=20) returns full 8x5 matrix.
      2. Slice call (n_terms=2, n_documents=3) returns top-left 3x2 corner [[3, 1], [2, 1], [1, 1]].
      3. session.pending_figure is set.
    """
    session = SessionState()

    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    tools = make_tools(session)
    build_dtm_tool = next(t for t in tools if t.name == "build_dtm_tool")
    dtm_heatmap_tool = next(t for t in tools if t.name == "dtm_heatmap_tool")

    # Run build_dtm_tool first
    build_dtm_tool.invoke({"text_column": "text"})

    # 1. Full table check (n_terms=20, n_documents=20 defaults exceed data, returning full 8x5 matrix)
    result_full = dtm_heatmap_tool.invoke({"n_terms": 20, "n_documents": 20})

    assert session.pending_figure is not None
    assert result_full["status"] == "success"
    assert result_full["n_terms"] == 5
    assert result_full["n_documents"] == 8
    assert result_full["terms"] == ['alpha', 'always', 'beta', 'delta', 'gamma']

    expected_full_matrix = [
        [3, 1, 1, 0, 0],
        [2, 1, 1, 0, 0],
        [1, 1, 1, 0, 0],
        [0, 1, 0, 1, 3],
        [0, 1, 0, 1, 2],
        [0, 1, 0, 1, 1],
        [0, 1, 0, 1, 1],
        [0, 0, 0, 0, 0]
    ]
    assert result_full["dtm_slice"] == expected_full_matrix

    # 2. Slice check (n_terms=2, n_documents=3)
    result_slice = dtm_heatmap_tool.invoke({"n_terms": 2, "n_documents": 3})

    assert result_slice["status"] == "success"
    assert result_slice["n_terms"] == 2
    assert result_slice["n_documents"] == 3
    assert result_slice["terms"] == ['alpha', 'always']
    assert result_slice["dtm_slice"] == [
        [3, 1],
        [2, 1],
        [1, 1]
    ]
