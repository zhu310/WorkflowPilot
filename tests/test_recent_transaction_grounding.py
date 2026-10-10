from __future__ import annotations

from types import SimpleNamespace

from src import semantic_shadow
from src.versioning import EditTransaction, transaction_diff


def graph() -> dict:
    return {
        "nodes": [
            {"id": "node_selected", "type": "task", "label": "Selected Node", "x": 20, "y": 20, "width": 160, "height": 80},
            {"id": "node_recent", "type": "task", "label": "Manual Review", "x": 20, "y": 140, "width": 160, "height": 80},
            {"id": "node_a", "type": "task", "label": "A", "x": 20, "y": 260, "width": 160, "height": 80},
            {"id": "node_b", "type": "task", "label": "B", "x": 20, "y": 380, "width": 160, "height": 80},
            {"id": "risk", "type": "task", "label": "Risk Control", "x": 300, "y": 140, "width": 160, "height": 80},
        ],
        "edges": [],
    }


def node_layout_goal(reference: str) -> dict:
    return {
        "status": "resolved",
        "semantic_goals": [
            {
                "goal_category": "relation_goal",
                "text": "Move this node to the right of Risk Control.",
                "objects": [
                    {"id": "moving", "type": "existing_reference", "text": reference, "role": "object"},
                    {"id": "risk", "type": "existing_reference", "text": "Risk Control", "role": "object"},
                ],
                "relation_intents": [{"source": "moving", "relation": "right_of", "target": "risk"}],
            }
        ],
        "clarification": {"needed": False, "question": "", "reason": ""},
    }


def selected_node_layout_goal() -> dict:
    return node_layout_goal("\u8fd9\u4e2a\u8282\u70b9")


def recent_diff(*node_ids: str, key: str = "affected_nodes") -> dict:
    diff = {
        "transaction_id": "tx1",
        "from_version_id": "v1",
        "to_version_id": "v2",
        "instruction": "previous edit",
        "mutations": [],
        "affected_nodes": list(node_ids),
        "affected_edges": [],
    }
    if key == "created_nodes":
        diff["created_nodes"] = [{"id": node_id, "label": node_id, "type": "node"} for node_id in node_ids]
    elif key == "changed_nodes":
        diff["changed_nodes"] = [{"id": node_id, "label": node_id, "type": "node"} for node_id in node_ids]
    return diff


def mock_understanding(monkeypatch) -> None:
    mock_understanding_with_goal(monkeypatch, selected_node_layout_goal())


def mock_understanding_with_goal(monkeypatch, goal: dict) -> None:
    monkeypatch.setattr(
        semantic_shadow,
        "_complete_json",
        lambda _system, _payload: SimpleNamespace(raw_response="", parsed_json=goal, parse_error=None, normalized_from_fence=False),
    )


def test_transaction_diff_records_object_summary_for_recent_grounding() -> None:
    before = {"nodes": [{"id": "risk", "type": "task", "label": "Risk Control"}], "edges": []}
    after = {
        "nodes": [
            {"id": "risk", "type": "task", "label": "Risk Review"},
            {"id": "node_recent", "type": "task", "label": "Manual Review"},
        ],
        "edges": [{"id": "e1", "source": "node_recent", "target": "risk", "label": ""}],
    }
    transaction = EditTransaction(
        transaction_id="tx1",
        parent_version_id="v1",
        instruction="add manual review",
        selection={"node_ids": [], "edge_ids": []},
        target=None,
        specification=None,
        plan=[],
        resulting_version_id="v2",
        status="success",
    )

    diff = transaction_diff(transaction, before, after)

    assert diff["mutations"]
    assert diff["created_nodes"] == [{"id": "node_recent", "label": "Manual Review", "type": "node", "node_type": "task"}]
    assert diff["changed_nodes"] == [{"id": "risk", "label": "Risk Review", "type": "node", "node_type": "task", "changed_properties": ["label"]}]
    assert diff["created_edges"] == [{"id": "e1", "type": "edge", "source": "node_recent", "target": "risk", "label": ""}]
    assert set(diff["affected_nodes"]) == {"node_recent", "risk"}
    assert diff["affected_edges"] == ["e1"]


def test_deictic_node_uses_unique_recent_created_node_when_no_selection(monkeypatch) -> None:
    mock_understanding(monkeypatch)

    trace = semantic_shadow.run_semantic_shadow(
        graph(),
        "\u628a\u8fd9\u4e2a\u8282\u70b9\u653e\u5230 Risk Control \u53f3\u4fa7",
        {"node_ids": [], "edge_ids": []},
        {"recent_transaction_diff": recent_diff("node_recent", key="created_nodes")},
    )

    assert trace["stage"] == "success"
    provenance = trace["grounded_goal"]["grounding"]["provenance"]
    assert any(
        row["reference"] == "\u8fd9\u4e2a\u8282\u70b9"
        and row["resolved_id"] == "node_recent"
        and row["resolution_type"] == "recent_transaction"
        and row["matched_transaction_id"] == "tx1"
        for row in provenance
    )
    assert trace["compiler_input"]["workflow_ir"]["properties"][0]["node_id"] == "node_recent"


def test_deictic_node_recent_multiple_affected_nodes_is_ambiguous(monkeypatch) -> None:
    mock_understanding(monkeypatch)

    trace = semantic_shadow.run_semantic_shadow(
        graph(),
        "\u628a\u8fd9\u4e2a\u8282\u70b9\u653e\u5230 Risk Control \u53f3\u4fa7",
        {"node_ids": [], "edge_ids": []},
        {"recent_transaction_diff": recent_diff("node_a", "node_b")},
    )

    assert trace["failure_stage"] == "validation"
    assert trace["stage"] == "goal_ground"
    assert trace["lowering_result"] is None
    assert trace["validation"]["issues"][0]["kind"] == "ambiguous_node_reference"
    assert trace["validation"]["issues"][0]["candidates"] == ["node_a", "node_b"]
    result = semantic_shadow.semantic_trace_to_transaction_result(trace)
    assert result.status == "needs_clarification"


def test_deictic_node_without_selection_or_recent_context_needs_clarification(monkeypatch) -> None:
    mock_understanding(monkeypatch)

    trace = semantic_shadow.run_semantic_shadow(
        graph(),
        "\u628a\u8fd9\u4e2a\u8282\u70b9\u653e\u5230 Risk Control \u53f3\u4fa7",
        {"node_ids": [], "edge_ids": []},
    )

    assert trace["failure_stage"] == "validation"
    assert trace["stage"] == "goal_ground"
    assert trace["lowering_result"] is None
    assert trace["validation"]["issues"][0]["kind"] == "missing_node_reference"
    result = semantic_shadow.semantic_trace_to_transaction_result(trace)
    assert result.status == "needs_clarification"


def test_deictic_node_selection_takes_priority_over_recent_transaction(monkeypatch) -> None:
    mock_understanding(monkeypatch)

    trace = semantic_shadow.run_semantic_shadow(
        graph(),
        "\u628a\u8fd9\u4e2a\u8282\u70b9\u653e\u5230 Risk Control \u53f3\u4fa7",
        {"selected_node_ids": ["node_selected"], "edge_ids": []},
        {"recent_transaction_diff": recent_diff("node_recent", key="created_nodes")},
    )

    assert trace["stage"] == "success"
    provenance = trace["grounded_goal"]["grounding"]["provenance"]
    assert any(
        row["reference"] == "\u8fd9\u4e2a\u8282\u70b9"
        and row["resolved_id"] == "node_selected"
        and row["resolution_type"] == "selection"
        for row in provenance
    )
    assert trace["compiler_input"]["workflow_ir"]["properties"][0]["node_id"] == "node_selected"


def test_explicit_recent_created_node_reference_uses_recent_transaction(monkeypatch) -> None:
    reference = "\u521a\u624d\u65b0\u589e\u7684\u90a3\u4e2a\u8282\u70b9"
    mock_understanding_with_goal(monkeypatch, node_layout_goal(reference))

    trace = semantic_shadow.run_semantic_shadow(
        graph(),
        "\u628a\u521a\u624d\u65b0\u589e\u7684\u90a3\u4e2a\u8282\u70b9\u653e\u5230 Risk Control \u53f3\u4fa7",
        {"node_ids": [], "edge_ids": []},
        {"recent_transaction_diff": recent_diff("node_recent", key="created_nodes")},
    )

    assert trace["stage"] == "success"
    provenance = trace["grounded_goal"]["grounding"]["provenance"]
    assert any(
        row["reference"] == reference
        and row["resolved_id"] == "node_recent"
        and row["resolution_type"] == "recent_transaction"
        and row["matched_transaction_id"] == "tx1"
        for row in provenance
    )
    assert trace["compiler_input"]["workflow_ir"]["properties"][0]["node_id"] == "node_recent"


def test_explicit_recent_created_node_reference_with_multiple_created_nodes_is_ambiguous(monkeypatch) -> None:
    reference = "\u4e0a\u4e00\u4e2a\u65b0\u589e\u8282\u70b9"
    mock_understanding_with_goal(monkeypatch, node_layout_goal(reference))

    trace = semantic_shadow.run_semantic_shadow(
        graph(),
        "\u628a\u4e0a\u4e00\u4e2a\u65b0\u589e\u8282\u70b9\u653e\u5230 Risk Control \u53f3\u4fa7",
        {"node_ids": [], "edge_ids": []},
        {"recent_transaction_diff": recent_diff("node_a", "node_b", key="created_nodes")},
    )

    assert trace["failure_stage"] == "validation"
    assert trace["validation"]["issues"][0]["kind"] == "ambiguous_node_reference"
    assert trace["validation"]["issues"][0]["candidates"] == ["node_a", "node_b"]
    assert trace["grounded_goal"]["grounding"]["issues"][0]["reason"] == "multiple nodes were affected by the recent transaction"


def test_explicit_recent_changed_node_reference_uses_changed_node(monkeypatch) -> None:
    reference = "\u521a\u624d\u4fee\u6539\u7684\u90a3\u4e2a\u8282\u70b9"
    mock_understanding_with_goal(monkeypatch, node_layout_goal(reference))

    trace = semantic_shadow.run_semantic_shadow(
        graph(),
        "\u628a\u521a\u624d\u4fee\u6539\u7684\u90a3\u4e2a\u8282\u70b9\u653e\u5230 Risk Control \u53f3\u4fa7",
        {"node_ids": [], "edge_ids": []},
        {"recent_transaction_diff": recent_diff("node_recent", key="changed_nodes")},
    )

    assert trace["stage"] == "success"
    provenance = trace["grounded_goal"]["grounding"]["provenance"]
    assert any(
        row["reference"] == reference
        and row["resolved_id"] == "node_recent"
        and row["resolution_type"] == "recent_transaction"
        for row in provenance
    )
