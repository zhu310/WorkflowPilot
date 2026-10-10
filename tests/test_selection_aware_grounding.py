from __future__ import annotations

from types import SimpleNamespace

from src import semantic_shadow


def graph() -> dict:
    return {
        "nodes": [
            {"id": "node_a", "type": "task", "label": "Review", "x": 20, "y": 20, "width": 160, "height": 80},
            {"id": "node_b", "type": "task", "label": "Risk", "x": 20, "y": 140, "width": 160, "height": 80},
            {"id": "anchor", "type": "task", "label": "Anchor", "x": 260, "y": 20, "width": 160, "height": 80},
            {"id": "manual", "type": "task", "label": "Manual Handling", "x": 520, "y": 20, "width": 160, "height": 80},
        ],
        "edges": [
            {"id": "edge_a", "source": "node_a", "target": "manual", "label": "failure", "branch_label": "failure"},
            {"id": "edge_b", "source": "node_b", "target": "manual", "label": "failure", "branch_label": "failure"},
        ],
    }


def selected_node_layout_goal() -> dict:
    return {
        "status": "resolved",
        "semantic_goals": [
            {
                "goal_category": "relation_goal",
                "text": "Move this node to the right of Anchor.",
                "objects": [
                    {"id": "moving", "type": "existing_reference", "text": "\u8fd9\u4e2a\u8282\u70b9", "role": "object"},
                    {"id": "anchor", "type": "existing_reference", "text": "Anchor", "role": "object"},
                ],
                "relation_intents": [{"source": "moving", "relation": "right_of", "target": "anchor"}],
            }
        ],
        "clarification": {"needed": False, "question": "", "reason": ""},
    }


def selected_edge_branch_goal() -> dict:
    return {
        "status": "resolved",
        "semantic_goals": [
            {
                "goal_category": "object_state_goal",
                "text": "Rename this branch to manual handling.",
                "objects": [
                    {"id": "branch", "type": "edge_reference", "text": "\u8fd9\u4e2a\u5206\u652f", "role": "object"}
                ],
                "object_desired_states": [
                    {"object": "branch", "state": "property_equals", "property": "display_label", "value": "\u4eba\u5de5\u5904\u7406"}
                ],
            }
        ],
        "clarification": {"needed": False, "question": "", "reason": ""},
    }


def mock_understanding(monkeypatch, spec: dict) -> None:
    monkeypatch.setattr(
        semantic_shadow,
        "_complete_json",
        lambda _system, _payload: SimpleNamespace(raw_response="", parsed_json=spec, parse_error=None, normalized_from_fence=False),
    )


def test_selected_node_deictic_reference_uses_unique_selection(monkeypatch) -> None:
    mock_understanding(monkeypatch, selected_node_layout_goal())

    trace = semantic_shadow.run_semantic_shadow(
        graph(),
        "\u628a\u8fd9\u4e2a\u8282\u70b9\u653e\u5230 Anchor \u53f3\u4fa7",
        {"selected_node_ids": ["node_a"], "edge_ids": []},
    )

    assert trace["stage"] == "success"
    provenance = trace["grounded_goal"]["grounding"]["provenance"]
    assert any(row["reference"] == "\u8fd9\u4e2a\u8282\u70b9" and row["resolved_id"] == "node_a" for row in provenance)
    assert trace["compiler_input"]["workflow_ir"]["properties"][0]["node_id"] == "node_a"


def test_selected_node_deictic_reference_multiple_selection_is_ambiguous(monkeypatch) -> None:
    mock_understanding(monkeypatch, selected_node_layout_goal())

    trace = semantic_shadow.run_semantic_shadow(
        graph(),
        "\u628a\u8fd9\u4e2a\u8282\u70b9\u653e\u5230 Anchor \u53f3\u4fa7",
        {"selected_node_ids": ["node_a", "node_b"], "edge_ids": []},
    )

    assert trace["failure_stage"] == "validation"
    assert trace["stage"] == "goal_ground"
    assert trace["lowering_result"] is None
    assert trace["validation"]["issues"][0]["kind"] == "ambiguous_node_reference"
    assert trace["validation"]["issues"][0]["candidates"] == ["node_a", "node_b"]
    result = semantic_shadow.semantic_trace_to_transaction_result(trace)
    assert result.status == "needs_clarification"


def test_selected_node_deictic_reference_without_selection_needs_clarification(monkeypatch) -> None:
    mock_understanding(monkeypatch, selected_node_layout_goal())

    trace = semantic_shadow.run_semantic_shadow(
        graph(),
        "\u628a\u8fd9\u4e2a\u8282\u70b9\u653e\u5230 Anchor \u53f3\u4fa7",
        {"node_ids": [], "edge_ids": []},
    )

    assert trace["failure_stage"] == "validation"
    assert trace["stage"] == "goal_ground"
    assert trace["lowering_result"] is None
    assert trace["validation"]["issues"][0]["kind"] == "missing_node_reference"
    result = semantic_shadow.semantic_trace_to_transaction_result(trace)
    assert result.status == "needs_clarification"


def test_selected_edge_deictic_reference_still_uses_unique_selection(monkeypatch) -> None:
    mock_understanding(monkeypatch, selected_edge_branch_goal())

    trace = semantic_shadow.run_semantic_shadow(
        graph(),
        "\u7ed9\u8fd9\u4e2a\u5206\u652f\u6539\u540d\u4e3a\u4eba\u5de5\u5904\u7406",
        {"node_ids": [], "selected_edge_ids": ["edge_a"]},
    )

    assert trace["stage"] == "success"
    provenance = trace["grounded_goal"]["grounding"]["provenance"]
    assert any(row["reference"] == "\u8fd9\u4e2a\u5206\u652f" and row["resolved_id"] == "edge_a" for row in provenance)
    assert trace["compiler_input"]["workflow_ir"]["edge_properties"] == [
        {"edge_id": "edge_a", "changes": {"branch_label": "\u4eba\u5de5\u5904\u7406"}}
    ]
