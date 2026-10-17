import numpy as np
import pytest
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_reduction import make_tools
from tools_dtm import make_tools as make_dtm_tools


def test_reduce_dimensions_pca_known_answer_7():
    """
    Tests reduce_dimensions_tool (PCA) against TEST_FIXTURE.md known answer #7.
    Fixture file: agent_dev/sample_fixture.csv
    Prerequisite: must invoke load_dataset_tool then build_dtm_tool first.
    Expected PCA 2D coordinates:
      Doc 0: [2.3669, 0.8871]
      Doc 1: [1.7039, 0.2642]
      Doc 2: [1.0408, -0.3587]
      Doc 3: [-2.0759, 1.0129]
      Doc 4: [-1.4554, 0.3405]
      Doc 5: [-0.8349, -0.3319]
      Doc 6: [-0.8349, -0.3319]
      Doc 7: [0.0894, -1.4822]
    Expected explained_variance_ratio: [0.7532, 0.1965]
    Also asserts session.pending_figure is set.
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

    reduction_tools = make_tools(session)
    reduce_dimensions_tool = next(t for t in reduction_tools if t.name == "reduce_dimensions_tool")

    result = reduce_dimensions_tool.invoke({"method": "pca", "n_components": 2})

    assert result["status"] == "success"
    assert result["method"] == "pca"
    assert result["n_documents"] == 8

    # Assert pending figure requirement
    assert session.pending_figure is not None
    assert hasattr(session.pending_figure, "savefig")

    # Expected PCA 2D coordinates from Known Answer #7
    expected_coords = np.array([
        [2.3669, 0.8871],
        [1.7039, 0.2642],
        [1.0408, -0.3587],
        [-2.0759, 1.0129],
        [-1.4554, 0.3405],
        [-0.8349, -0.3319],
        [-0.8349, -0.3319],
        [0.0894, -1.4822]
    ])

    actual_coords = np.array(result["coordinates"])

    assert actual_coords.shape == (8, 2)
    assert np.allclose(actual_coords, expected_coords, atol=1e-3) or np.allclose(np.abs(actual_coords), np.abs(expected_coords), atol=1e-3)

    # Expected explained_variance_ratio
    expected_exp_var = [0.7532, 0.1965]
    actual_exp_var = result["explained_variance_ratio"]
    assert np.allclose(actual_exp_var, expected_exp_var, atol=1e-3)
