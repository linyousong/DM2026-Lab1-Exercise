import json
import pytest
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_data import make_tools


def test_tokenize_tool_known_answer_10():
    """
    Tests tokenize_tool against TEST_FIXTURE.md known answer #10.
    Fixture file: agent_dev/sample_fixture.csv
    Expected output per document:
      Doc 0: ['always', 'alpha', 'alpha', 'alpha', 'beta']
      Doc 1: ['always', 'alpha', 'alpha', 'beta']
      Doc 2: ['always', 'alpha', 'beta']
      Doc 3: ['always', 'gamma', 'gamma', 'gamma', 'delta']
      Doc 4: ['always', 'gamma', 'gamma', 'delta']
      Doc 5: ['always', 'gamma', 'delta']
      Doc 6: ['always', 'gamma', 'delta']
      Doc 7: []
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

    # Get student tokenize_tool from make_tools(session)
    tools = make_tools(session)
    tokenize_tool = next(t for t in tools if t.name == "tokenize_tool")

    # Invoke tokenize_tool
    result = tokenize_tool.invoke({})

    # Verify summary result
    assert result["n_documents"] == 8
    assert result["text_column"] == "text"
    assert result["output_column"] == "unigrams"

    # Verify unigrams in session.dataframe against known answer #10
    expected_tokens = [
        ['always', 'alpha', 'alpha', 'alpha', 'beta'],
        ['always', 'alpha', 'alpha', 'beta'],
        ['always', 'alpha', 'beta'],
        ['always', 'gamma', 'gamma', 'gamma', 'delta'],
        ['always', 'gamma', 'gamma', 'delta'],
        ['always', 'gamma', 'delta'],
        ['always', 'gamma', 'delta'],
        []
    ]

    actual_tokens = session.dataframe["unigrams"].tolist()
    assert actual_tokens == expected_tokens


def test_list_files_tool_known_answer_11():
    """
    Tests list_files_tool against TEST_FIXTURE.md known answer #11.
    Target directory: "newdataset"
    Expected result: directories: [], files: ["Reddit-stock-sentiment.csv"]
    """
    session = SessionState()
    tools = make_tools(session)
    list_files_tool = next(t for t in tools if t.name == "list_files_tool")

    result = list_files_tool.invoke({"subdirectory": "newdataset"})

    assert result["status"] == "success"
    assert result["directories"] == []
    assert result["files"] == ["Reddit-stock-sentiment.csv"]


def test_check_missing_tool_known_answer_2():
    """
    Tests check_missing_tool against TEST_FIXTURE.md known answer #2.
    Fixture file: agent_dev/sample_fixture.csv
    Expected result: Exactly 1 missing row (row index 7, empty text).
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

    # Get student check_missing_tool from make_tools(session)
    tools = make_tools(session)
    check_missing_tool = next(t for t in tools if t.name == "check_missing_tool")

    # Invoke check_missing_tool
    result = check_missing_tool.invoke({"text_column": "text"})

    # Assertions based on TEST_FIXTURE.md known answer #2
    assert result["n_documents"] == 8
    assert result["missing_count"] == 1
    assert result["missing_indices"] == [7]
    assert pytest.approx(result["missing_pct"], abs=1e-3) == 12.5


def test_check_duplicates_tool_known_answer_3():
    """
    Tests check_duplicates_tool against TEST_FIXTURE.md known answer #3.
    Fixture file: agent_dev/sample_fixture.csv
    Expected result:
      - drop=False: 1 duplicate detected (row 6 duplicate of row 5), 8 remaining rows.
      - drop=True: keep=False drops BOTH row 5 and row 6 ("always gamma delta"), leaving exactly 6 rows.
    """
    # Test drop=False
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    tools = make_tools(session)
    check_duplicates_tool = next(t for t in tools if t.name == "check_duplicates_tool")

    result_check = check_duplicates_tool.invoke({"subset": ["text"], "drop": False})

    assert result_check["initial_rows"] == 8
    assert result_check["duplicate_count"] == 1
    assert result_check["rows_dropped"] == 0
    assert result_check["remaining_rows"] == 8

    # Test drop=True with fresh session state
    session_drop = SessionState()
    premade_tools_drop, _ = load_workspace_tools("premade_tools", session_drop)
    load_dataset_tool_drop = next(t for t in premade_tools_drop if t.name == "load_dataset_tool")

    load_dataset_tool_drop.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    tools_drop = make_tools(session_drop)
    check_duplicates_tool_drop = next(t for t in tools_drop if t.name == "check_duplicates_tool")

    result_drop = check_duplicates_tool_drop.invoke({"subset": ["text"], "drop": True})

    assert result_drop["initial_rows"] == 8
    assert result_drop["duplicate_count"] == 1
    assert result_drop["rows_dropped"] == 2
    assert result_drop["remaining_rows"] == 6

    # Verify that "always gamma delta" (rows 5 and 6) is completely gone from dataframe
    texts_remaining = session_drop.dataframe["text"].tolist()
    assert "always gamma delta" not in texts_remaining
    assert len(texts_remaining) == 6


def test_sample_data_tool_known_answer_13():
    """
    Tests sample_data_tool against TEST_FIXTURE.md known answer #13.
    Fixture file: agent_dev/sample_fixture.csv
    Call parameters: n=4, random_state=42
    Expected result: sampled_indices == [1, 5, 0, 7]
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

    # Get student sample_data_tool from make_tools(session)
    tools = make_tools(session)
    sample_data_tool = next(t for t in tools if t.name == "sample_data_tool")

    # Invoke sample_data_tool with n=4, random_state=42
    result = sample_data_tool.invoke({"n": 4, "random_state": 42})

    # Assertions based on TEST_FIXTURE.md known answer #13
    assert result["initial_rows"] == 8
    assert result["sampled_rows"] == 4
    assert result["sampled_indices"] == [1, 5, 0, 7]
    assert len(session.dataframe) == 4


def test_inspect_data_tool():
    """
    Tests inspect_data_tool (custom tool with no TEST_FIXTURE.md entry).
    Verifies:
      - Unloaded state: returns status="error".
      - Loaded state on sample_fixture.csv: shape[0] == 8, n_rows requested comes back,
        text length truncated to <= 200 chars, result is JSON serializable.
    """
    # 1. Unloaded session check
    session_unloaded = SessionState()
    tools_unloaded = make_tools(session_unloaded)
    inspect_data_tool_unloaded = next(t for t in tools_unloaded if t.name == "inspect_data_tool")

    result_unloaded = inspect_data_tool_unloaded.invoke({"n_rows": 5})
    assert result_unloaded["status"] == "error"
    assert result_unloaded["head"] == []

    # 2. Loaded session check on sample_fixture.csv
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    tools = make_tools(session)
    inspect_data_tool = next(t for t in tools if t.name == "inspect_data_tool")

    result = inspect_data_tool.invoke({"n_rows": 3})

    # Structural / Invariant assertions based on fixture facts and spec contract
    assert result["status"] == "success"
    assert result["shape"][0] == 8  # sample_fixture.csv has 8 rows
    assert len(result["head"]) == 3  # n_rows=3 requested

    # Check text truncation contract (all string values in head <= 200 chars)
    for row in result["head"]:
        for val in row.values():
            if isinstance(val, str):
                assert len(val) <= 200

    # Check JSON-serializability
    json_str = json.dumps(result)
    assert isinstance(json_str, str)


def test_describe_data_tool_known_answer_18():
    """
    Tests describe_data_tool against TEST_FIXTURE.md known answer #18.
    Fixture file: agent_dev/sample_fixture.csv
    Expected text lengths: [29, 23, 17, 30, 24, 18, 18, 0]
    Overall stats: count=8, mean=19.875, std=9.4330, min=0, 25%=17.75, 50%=20.5, 75%=25.25, max=30
    catA stats: count=4, mean=17.25, std=12.5, min=0, 25%=12.75, 50%=20.0, 75%=24.5, max=29
    catB stats: count=4, mean=22.5, std=5.7446, min=18, 25%=18.0, 50%=21.0, 75%=25.5, max=30
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
    describe_data_tool = next(t for t in tools if t.name == "describe_data_tool")

    # Invoke describe_data_tool
    result = describe_data_tool.invoke({
        "text_column": "text",
        "category_column": "category_name"
    })

    # 1. Verify session.pending_figure was set
    assert session.pending_figure is not None

    # 2. Verify Overall Stats against Known Answer #18
    overall = result["overall_stats"]
    assert overall["count"] == 8.0
    assert overall["mean"] == pytest.approx(19.875, abs=1e-3)
    assert overall["std"] == pytest.approx(9.4330, abs=1e-3)
    assert overall["min"] == 0.0
    assert overall["25%"] == pytest.approx(17.75, abs=1e-3)
    assert overall["50%"] == pytest.approx(20.5, abs=1e-3)
    assert overall["75%"] == pytest.approx(25.25, abs=1e-3)
    assert overall["max"] == 30.0

    # 3. Verify Per-Category Stats (catA)
    catA = result["by_category_stats"]["catA"]
    assert catA["count"] == 4.0
    assert catA["mean"] == pytest.approx(17.25, abs=1e-3)
    assert catA["std"] == pytest.approx(12.5, abs=1e-3)
    assert catA["min"] == 0.0
    assert catA["25%"] == pytest.approx(12.75, abs=1e-3)
    assert catA["50%"] == pytest.approx(20.0, abs=1e-3)
    assert catA["75%"] == pytest.approx(24.5, abs=1e-3)
    assert catA["max"] == 29.0

    # 4. Verify Per-Category Stats (catB)
    catB = result["by_category_stats"]["catB"]
    assert catB["count"] == 4.0
    assert catB["mean"] == pytest.approx(22.5, abs=1e-3)
    assert catB["std"] == pytest.approx(5.7446, abs=1e-3)
    assert catB["min"] == 18.0
    assert catB["25%"] == pytest.approx(18.0, abs=1e-3)
    assert catB["50%"] == pytest.approx(21.0, abs=1e-3)
    assert catB["75%"] == pytest.approx(25.5, abs=1e-3)
    assert catB["max"] == 30.0
