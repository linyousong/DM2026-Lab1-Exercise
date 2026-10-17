import pytest
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_filtering import make_tools
from tools_dtm import make_tools as make_dtm_tools


def test_variance_filter_tool_known_answer_5():
    """
    Tests variance_filter_tool against TEST_FIXTURE.md known answer #5.
    Fixture file: agent_dev/sample_fixture.csv
    Prerequisite: must invoke load_dataset_tool then build_dtm_tool first.
    Expected variances:
      - alpha: 1.1875
      - always: 0.109375
      - beta: 0.234375
      - delta: 0.25
      - gamma: 1.109375
    At threshold=0.15:
      - kept_count: 4
      - removed_count: 1
    Verifies full_report formatting (1st col: term, 2nd col: variance, sorted descending) and per-term scores.
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

    # 2. Run build_dtm_tool to populate session.feature_matrix
    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    build_dtm_tool.invoke({"text_column": "text"})

    # 3. Get student variance_filter_tool from make_tools(session)
    filtering_tools = make_tools(session)
    variance_filter_tool = next(t for t in filtering_tools if t.name == "variance_filter_tool")

    # 4. Invoke variance_filter_tool with threshold=0.15
    result = variance_filter_tool.invoke({"threshold": 0.15})

    assert result["status"] == "success"
    assert result["total_terms"] == 5
    assert result["kept_count"] == 4
    assert result["removed_count"] == 1

    # Verify full_report stored in results_store for visualize_result_tool compatibility and per-term scores
    stored_result = session.results_store.get(result["result_id"])
    assert stored_result is not None
    full_report = stored_result.get("full_report")
    assert full_report is not None
    assert full_report.columns[0] == "term"
    assert full_report.columns[1] == "variance"

    # Verify sorted descending by variance
    variances_list = full_report["variance"].tolist()
    assert variances_list == sorted(variances_list, reverse=True)

    # Verify per-term variances against Known Answer #5 from full_report
    vars_dict = dict(zip(full_report["term"], full_report["variance"]))
    assert vars_dict["alpha"] == pytest.approx(1.1875, abs=1e-4)
    assert vars_dict["always"] == pytest.approx(0.109375, abs=1e-4)
    assert vars_dict["beta"] == pytest.approx(0.234375, abs=1e-4)
    assert vars_dict["delta"] == pytest.approx(0.25, abs=1e-4)
    assert vars_dict["gamma"] == pytest.approx(1.109375, abs=1e-4)

    # Verify kept/removed terms at threshold=0.15 from full_report
    kept_terms = set(full_report[full_report["variance"] >= 0.15]["term"])
    removed_terms = list(full_report[full_report["variance"] < 0.15]["term"])
    assert kept_terms == {"alpha", "beta", "delta", "gamma"}
    assert removed_terms == ["always"]


def test_pearson_filter_tool_known_answer_6():
    """
    Tests pearson_filter_tool against TEST_FIXTURE.md known answer #6.
    Fixture file: agent_dev/sample_fixture.csv
    Prerequisite: must invoke load_dataset_tool then build_dtm_tool first.
    Target class: "catB"
    Expected Pearson correlations:
      - alpha: -0.6882
      - always: 0.3780
      - beta: -0.7746
      - delta: 1.0000
      - gamma: 0.8307
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

    filtering_tools = make_tools(session)
    pearson_filter_tool = next(t for t in filtering_tools if t.name == "pearson_filter_tool")

    # Invoke pearson_filter_tool for catB
    result = pearson_filter_tool.invoke({"target_class": "catB", "threshold": 0.5})

    assert result["status"] == "success"
    assert result["target_class"] == "catB"
    assert result["total_terms"] == 5
    assert result["kept_count"] == 2
    assert result["removed_count"] == 3

    # Verify full_report column structure, sorting, and correlation coefficients
    stored_result = session.results_store.get(result["result_id"])
    assert stored_result is not None
    full_report = stored_result.get("full_report")
    assert full_report is not None
    assert full_report.columns[0] == "term"
    assert full_report.columns[1] == "pearson_r"
    r_list = full_report["pearson_r"].tolist()
    assert r_list == sorted(r_list, reverse=True)

    # Verify Pearson correlations against Known Answer #6 from full_report
    p_dict = dict(zip(full_report["term"], full_report["pearson_r"]))
    assert p_dict["alpha"] == pytest.approx(-0.6882, abs=1e-3)
    assert p_dict["always"] == pytest.approx(0.3780, abs=1e-3)
    assert p_dict["beta"] == pytest.approx(-0.7746, abs=1e-3)
    assert p_dict["delta"] == pytest.approx(1.0000, abs=1e-3)
    assert p_dict["gamma"] == pytest.approx(0.8307, abs=1e-3)

    # Verify kept terms at threshold=0.5 from full_report
    kept_terms = set(full_report[full_report["pearson_r"] >= 0.5]["term"])
    assert kept_terms == {"delta", "gamma"}


def test_spearman_filter_tool_known_answer_6():
    """
    Tests spearman_filter_tool against TEST_FIXTURE.md known answer #6.
    Fixture file: agent_dev/sample_fixture.csv
    Prerequisite: must invoke load_dataset_tool then build_dtm_tool first.
    Target class: "catB"
    Expected Spearman correlations:
      - alpha: -0.7500
      - always: 0.3780
      - beta: -0.7746
      - delta: 1.0000
      - gamma: 0.9363
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

    filtering_tools = make_tools(session)
    spearman_filter_tool = next(t for t in filtering_tools if t.name == "spearman_filter_tool")

    # Invoke spearman_filter_tool for catB
    result = spearman_filter_tool.invoke({"target_class": "catB", "threshold": 0.5})

    assert result["status"] == "success"
    assert result["target_class"] == "catB"
    assert result["total_terms"] == 5
    assert result["kept_count"] == 2
    assert result["removed_count"] == 3

    # Verify full_report column structure, sorting, and correlation coefficients
    stored_result = session.results_store.get(result["result_id"])
    assert stored_result is not None
    full_report = stored_result.get("full_report")
    assert full_report is not None
    assert full_report.columns[0] == "term"
    assert full_report.columns[1] == "spearman_r"
    r_list = full_report["spearman_r"].tolist()
    assert r_list == sorted(r_list, reverse=True)

    # Verify Spearman correlations against Known Answer #6 from full_report
    s_dict = dict(zip(full_report["term"], full_report["spearman_r"]))
    assert s_dict["alpha"] == pytest.approx(-0.7500, abs=1e-3)
    assert s_dict["always"] == pytest.approx(0.3780, abs=1e-3)
    assert s_dict["beta"] == pytest.approx(-0.7746, abs=1e-3)
    assert s_dict["delta"] == pytest.approx(1.0000, abs=1e-3)
    assert s_dict["gamma"] == pytest.approx(0.9363, abs=1e-3)

    # Verify kept terms at threshold=0.5 from full_report
    kept_terms = set(full_report[full_report["spearman_r"] >= 0.5]["term"])
    assert kept_terms == {"delta", "gamma"}
