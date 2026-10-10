from __future__ import annotations

from types import SimpleNamespace

from src import semantic_shadow


def movement_goal(source_text: str = "\u5ba1\u6838", anchor_text: str = "\u98ce\u63a7") -> dict:
    return {
        "status": "resolved",
        "semantic_goals": [
            {
                "goal_category": "relation_goal",
                "text": "Move source after anchor.",
                "objects": [
                    {"id": "source", "type": "existing_reference", "text": source_text, "role": "object"},
                    {"id": "anchor", "type": "existing_reference", "text": anchor_text, "role": "object"},
                ],
                "relation_intents": [{"source": "source", "relation": "after", "target": "anchor"}],
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


def graph_with(*nodes: dict) -> dict:
    normalized_nodes = [
            {**node, "type": node.get("type", "task"), "x": node.get("x", index * 180), "y": node.get("y", 80), "width": 140, "height": 72}
            for index, node in enumerate(nodes)
    ]
    return {
        "nodes": normalized_nodes,
        "edges": [
            {"id": f"e_{normalized_nodes[index]['id']}_{normalized_nodes[index + 1]['id']}", "source": normalized_nodes[index]["id"], "target": normalized_nodes[index + 1]["id"], "label": ""}
            for index in range(len(normalized_nodes) - 1)
        ],
    }


def recent_diff(node_id: str) -> dict:
    return {
        "transaction_id": "tx1",
        "from_version_id": "v1",
        "to_version_id": "v2",
        "instruction": "previous edit",
        "mutations": [],
        "created_nodes": [{"id": node_id, "label": node_id, "type": "node"}],
        "affected_nodes": [node_id],
        "affected_edges": [],
    }


def test_semantic_reference_unique_node_match_runs_grounding(monkeypatch) -> None:
    mock_understanding(monkeypatch, movement_goal())
    graph = graph_with(
        {"id": "approval", "label": "\u5ba1\u6279\u8282\u70b9"},
        {"id": "risk", "label": "\u98ce\u9669\u63a7\u5236"},
    )

    trace = semantic_shadow.run_semantic_shadow(graph, "\u628a\u5ba1\u6838\u79fb\u52a8\u5230\u98ce\u63a7\u540e\u9762")

    assert trace["stage"] == "success"
    provenance = trace["grounded_goal"]["grounding"]["provenance"]
    assert any(
        row["reference"] == "\u5ba1\u6838"
        and row["resolved_id"] == "approval"
        and row["resolution_type"] == "semantic_similarity"
        for row in provenance
    )
    assert any(
        row["reference"] == "\u98ce\u63a7"
        and row["resolved_id"] == "risk"
        and row["resolution_type"] == "semantic_similarity"
        for row in provenance
    )


def test_semantic_reference_multiple_close_candidates_stops_for_clarification(monkeypatch) -> None:
    mock_understanding(monkeypatch, movement_goal())
    graph = graph_with(
        {"id": "approval", "label": "\u5ba1\u6279\u8282\u70b9"},
        {"id": "manual_review", "label": "\u4eba\u5de5\u590d\u6838"},
        {"id": "risk", "label": "\u98ce\u9669\u63a7\u5236"},
    )

    trace = semantic_shadow.run_semantic_shadow(graph, "\u628a\u5ba1\u6838\u79fb\u52a8\u5230\u98ce\u63a7\u540e\u9762")

    assert trace["failure_stage"] == "validation"
    assert trace["stage"] == "goal_ground"
    assert trace["lowering_result"] is None
    issues = trace["validation"]["issues"]
    assert issues[0]["kind"] == "ambiguous_node_reference"
    assert issues[0]["candidates"] == ["approval", "manual_review"]
    result = semantic_shadow.semantic_trace_to_transaction_result(trace)
    assert result.status == "needs_clarification"


def test_semantic_reference_does_not_override_selection(monkeypatch) -> None:
    mock_understanding(monkeypatch, movement_goal("\u8fd9\u4e2a\u8282\u70b9", "\u98ce\u63a7"))
    graph = graph_with(
        {"id": "selected", "label": "\u4eba\u5de5\u590d\u6838"},
        {"id": "approval", "label": "\u5ba1\u6279\u8282\u70b9"},
        {"id": "risk", "label": "\u98ce\u9669\u63a7\u5236"},
    )

    trace = semantic_shadow.run_semantic_shadow(
        graph,
        "\u628a\u8fd9\u4e2a\u8282\u70b9\u79fb\u52a8\u5230\u98ce\u63a7\u540e\u9762",
        {"selected_node_ids": ["selected"], "edge_ids": []},
    )

    assert trace["stage"] == "success"
    provenance = trace["grounded_goal"]["grounding"]["provenance"]
    assert any(
        row["reference"] == "\u8fd9\u4e2a\u8282\u70b9"
        and row["resolved_id"] == "selected"
        and row["resolution_type"] == "selection"
        for row in provenance
    )


def test_semantic_reference_does_not_override_recent_transaction(monkeypatch) -> None:
    mock_understanding(monkeypatch, movement_goal("\u521a\u624d\u90a3\u4e2a\u8282\u70b9", "\u98ce\u63a7"))
    graph = graph_with(
        {"id": "recent", "label": "\u4eba\u5de5\u5ba1\u6838"},
        {"id": "approval", "label": "\u5ba1\u6279\u8282\u70b9"},
        {"id": "risk", "label": "\u98ce\u9669\u63a7\u5236"},
    )

    trace = semantic_shadow.run_semantic_shadow(
        graph,
        "\u628a\u521a\u624d\u90a3\u4e2a\u8282\u70b9\u79fb\u52a8\u5230\u98ce\u63a7\u540e\u9762",
        {"node_ids": [], "edge_ids": []},
        {"recent_transaction_diff": recent_diff("recent")},
    )

    assert trace["stage"] == "success"
    provenance = trace["grounded_goal"]["grounding"]["provenance"]
    assert any(
        row["reference"] == "\u521a\u624d\u90a3\u4e2a\u8282\u70b9"
        and row["resolved_id"] == "recent"
        and row["resolution_type"] == "recent_transaction"
        for row in provenance
    )


def test_semantic_reference_low_confidence_missing_reference(monkeypatch) -> None:
    mock_understanding(monkeypatch, movement_goal())
    graph = graph_with(
        {"id": "payment", "label": "\u652f\u4ed8\u8282\u70b9"},
        {"id": "inventory", "label": "\u5e93\u5b58\u8282\u70b9"},
        {"id": "shipping", "label": "\u53d1\u8d27\u8282\u70b9"},
    )

    trace = semantic_shadow.run_semantic_shadow(graph, "\u628a\u5ba1\u6838\u79fb\u52a8\u5230\u98ce\u63a7\u540e\u9762")

    assert trace["failure_stage"] == "validation"
    assert trace["stage"] == "goal_ground"
    assert trace["lowering_result"] is None
    assert trace["validation"]["issues"][0]["kind"] == "missing_node_reference"
    result = semantic_shadow.semantic_trace_to_transaction_result(trace)
    assert result.status == "needs_clarification"
