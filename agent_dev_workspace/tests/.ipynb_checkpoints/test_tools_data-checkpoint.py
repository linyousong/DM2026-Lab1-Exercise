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
