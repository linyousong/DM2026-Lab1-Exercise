import pytest
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_patterns import make_tools


def test_mine_patterns_zero_survivors_known_answer_17():
    """
    Tests mine_patterns_tool zero-survivor handling against TEST_FIXTURE.md known answer #17.
    Fixture file: agent_dev/sample_fixture.csv

    Known answer #17:
      filtering_method="variance" and filtering_method="term_frequency" both keep
      zero terms for both categories ("catA" and "catB").
      mine_patterns_tool must return a clean "no terms survived filtering" result, not raise an exception.
    """
    session = SessionState()

    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    pattern_tools = make_tools(session)
    mine_patterns_tool = next(t for t in pattern_tools if t.name == "mine_patterns_tool")

    for cat in ["catA", "catB"]:
        for method in ["variance", "term_frequency"]:
            result = mine_patterns_tool.invoke({
                "category_name": cat,
                "filtering_method": method,
                "algorithm": "fpgrowth",
                "min_sup": 1
            })

            assert result["status"] == "success"
            assert result["category_name"] == cat
            assert result["filtering_method"] == method
            assert result["patterns"] == []
            assert result["n_patterns"] == 0
            assert result["note"] == "no terms survived filtering"

            artifact_key = f"patterns_{cat}_{method}_fpgrowth"
            assert artifact_key in session.artifacts
            assert session.artifacts[artifact_key] == []


def test_mine_patterns_tfidf_fpgrowth_known_answer_17():
    """
    Tests mine_patterns_tool (tfidf + fpgrowth) against TEST_FIXTURE.md known answer #17.
    Fixture file: agent_dev/sample_fixture.csv

    Known Answer #17:
      catA keeps `alpha` (support 3), found at min_sup=1..3 (0 patterns at min_sup=4).
      catB keeps `gamma` (support 4), found at min_sup=1..4.
    """
    session = SessionState()

    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    pattern_tools = make_tools(session)
    mine_patterns_tool = next(t for t in pattern_tools if t.name == "mine_patterns_tool")

    # --- Test catA ---
    res_catA = mine_patterns_tool.invoke({
        "category_name": "catA",
        "filtering_method": "tfidf",
        "algorithm": "fpgrowth",
        "min_sup": 1
    })

    assert res_catA["status"] == "success"
    assert res_catA["n_patterns"] == 1
    assert res_catA["patterns"][0]["pattern"] == ["alpha"]
    assert res_catA["patterns"][0]["support"] == 3

    # Check min_sup=4 for catA returns 0 patterns
    res_catA_sup4 = mine_patterns_tool.invoke({
        "category_name": "catA",
        "filtering_method": "tfidf",
        "algorithm": "fpgrowth",
        "min_sup": 4
    })
    assert res_catA_sup4["status"] == "success"
    assert res_catA_sup4["n_patterns"] == 0

    # --- Test catB ---
    res_catB = mine_patterns_tool.invoke({
        "category_name": "catB",
        "filtering_method": "tfidf",
        "algorithm": "fpgrowth",
        "min_sup": 1
    })

    assert res_catB["status"] == "success"
    assert res_catB["n_patterns"] == 1
    assert res_catB["patterns"][0]["pattern"] == ["gamma"]
    assert res_catB["patterns"][0]["support"] == 4


def test_mine_patterns_topk_and_maxfpgrowth_known_answer_20():
    """
    Tests mine_patterns_tool (topk and maxfpgrowth) against TEST_FIXTURE.md known answer #20.
    Fixture file: agent_dev/sample_fixture.csv

    Known Answer #20:
      algorithm="topk": at k=1, k=2, k=3, both categories return exactly one pattern:
        catA -> {alpha} support 3
        catB -> {gamma} support 4

      algorithm="maxfpgrowth": returns zero patterns for both categories at min_sup=1 and min_sup=4.
        This is real PAMI behavior on a single-surviving-term transactional DB, not a bug.
    """
    session = SessionState()

    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    pattern_tools = make_tools(session)
    mine_patterns_tool = next(t for t in pattern_tools if t.name == "mine_patterns_tool")

    # --- Test algorithm="topk" ---
    for k_val in [1, 2, 3]:
        # catA topk
        res_catA_topk = mine_patterns_tool.invoke({
            "category_name": "catA",
            "filtering_method": "tfidf",
            "algorithm": "topk",
            "k": k_val
        })
        assert res_catA_topk["status"] == "success"
        assert res_catA_topk["n_patterns"] == 1
        assert res_catA_topk["patterns"][0]["pattern"] == ["alpha"]
        assert res_catA_topk["patterns"][0]["support"] == 3

        # catB topk
        res_catB_topk = mine_patterns_tool.invoke({
            "category_name": "catB",
            "filtering_method": "tfidf",
            "algorithm": "topk",
            "k": k_val
        })
        assert res_catB_topk["status"] == "success"
        assert res_catB_topk["n_patterns"] == 1
        assert res_catB_topk["patterns"][0]["pattern"] == ["gamma"]
        assert res_catB_topk["patterns"][0]["support"] == 4

    # --- Test algorithm="maxfpgrowth" ---
    for cat in ["catA", "catB"]:
        for sup in [1, 4]:
            res_max = mine_patterns_tool.invoke({
                "category_name": cat,
                "filtering_method": "tfidf",
                "algorithm": "maxfpgrowth",
                "min_sup": sup
            })
            assert res_max["status"] == "success"
            assert res_max["n_patterns"] == 0
            assert res_max["patterns"] == []
