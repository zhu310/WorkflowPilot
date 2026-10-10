from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from demo import server
from src import semantic_shadow


def graph() -> dict[str, Any]:
    return {
        "nodes": [
            {"id": "n_decision", "type": "decision", "label": "Decision"},
            {"id": "n_report", "type": "task", "label": "Report"},
            {"id": "n_manual", "type": "task", "label": "Manual"},
            {"id": "n_end", "type": "end", "label": "End"},
        ],
        "edges": [
            {"id": "e_decision_report", "source": "n_decision", "target": "n_report", "label": "success"},
            {"id": "e_decision_manual", "source": "n_decision", "target": "n_manual", "label": "failure"},
            {"id": "e_report_end", "source": "n_report", "target": "n_end", "label": ""},
        ],
    }


def route_existing_reference_goal(route_text: str = "this edge") -> dict[str, Any]:
    return {
        "status": "resolved",
        "semantic_goals": [
            {
                "id": "g1",
                "goal_category": "relation_goal",
                "text": "route selected edge to End",
                "objects": [
                    {"id": "route", "type": "existing_reference", "text": route_text, "role": "object"},
                    {"id": "target", "type": "existing_reference", "text": "End", "role": "object"},
                ],
                "relation_intents": [{"source": "route", "relation": "route_to", "target": "target"}],
            }
        ],
        "clarification": {"needed": False, "question": "", "reason": ""},
    }


def route_edge_reference_goal(route_text: str = "this branch") -> dict[str, Any]:
    spec = route_existing_reference_goal(route_text)
    spec["semantic_goals"][0]["objects"][0]["type"] = "edge_reference"
    return spec


def selected_node_layout_goal() -> dict[str, Any]:
    return {
        "status": "resolved",
        "semantic_goals": [
            {
                "id": "g1",
                "goal_category": "relation_goal",
                "text": "move this node right of End",
                "objects": [
                    {"id": "node", "type": "existing_reference", "text": "this node", "role": "object"},
                    {"id": "end", "type": "existing_reference", "text": "End", "role": "object"},
                ],
                "relation_intents": [{"source": "node", "relation": "right_of", "target": "end"}],
            }
        ],
        "clarification": {"needed": False, "question": "", "reason": ""},
    }


def mock_understanding(monkeypatch, spec: dict[str, Any]) -> None:
    monkeypatch.setattr(
        semantic_shadow,
        "_complete_json",
        lambda _system, _payload: SimpleNamespace(raw_response="", parsed_json=spec, parse_error=None, normalized_from_fence=False),
    )


def test_route_slot_existing_reference_recovers_unique_selected_edge(monkeypatch) -> None:
    mock_understanding(monkeypatch, route_existing_reference_goal())

    trace = semantic_shadow.run_semantic_shadow(
        graph(),
        "route this edge to End",
        {"node_ids": [], "edge_ids": ["e_decision_report"]},
    )

    assert trace["stage"] == "success"
    bindings = trace["grounded_goal"]["semantic_goals"][0]["bindings"]
    assert any(binding.get("role") == "route" and binding.get("binding_type") == "edge" and binding.get("id") == "e_decision_report" for binding in bindings)
    assert not any(issue.get("kind") == "missing_node_reference" for issue in trace["grounded_goal"]["grounding"]["issues"])


def test_route_slot_existing_reference_multiple_selected_edges_is_edge_ambiguous(monkeypatch) -> None:
    mock_understanding(monkeypatch, route_existing_reference_goal())

    trace = semantic_shadow.run_semantic_shadow(
        graph(),
        "route this edge to End",
        {"node_ids": [], "edge_ids": ["e_decision_report", "e_decision_manual"]},
    )

    assert trace["failure_stage"] == "validation"
    issues = trace["validation"]["issues"]
    assert issues[0]["kind"] == "ambiguous_edge_reference"
    assert issues[0]["candidates"] == ["e_decision_manual", "e_decision_report"]
    assert not any(issue.get("kind") == "missing_node_reference" for issue in issues)


def test_route_slot_existing_reference_without_selection_is_edge_clarification(monkeypatch) -> None:
    mock_understanding(monkeypatch, route_existing_reference_goal())

    trace = semantic_shadow.run_semantic_shadow(graph(), "route this edge to End", {"node_ids": [], "edge_ids": []})

    assert trace["failure_stage"] == "validation"
    issues = trace["validation"]["issues"]
    assert issues[0]["kind"] == "ambiguous_edge_reference"
    assert issues[0]["candidates"] == ["e_decision_manual", "e_decision_report", "e_report_end"]
    assert not any(issue.get("kind") == "missing_node_reference" for issue in issues)


def test_existing_edge_reference_path_still_uses_selected_edge(monkeypatch) -> None:
    mock_understanding(monkeypatch, route_edge_reference_goal())

    trace = semantic_shadow.run_semantic_shadow(
        graph(),
        "route this branch to End",
        {"node_ids": [], "edge_ids": ["e_decision_report"]},
    )

    assert trace["stage"] == "success"
    assert trace["compiler_input"]["workflow_ir"]["auxiliary_relations"] == [
        {"source": "n_decision", "relation_type": "success", "targets": ["n_end"]}
    ]


def test_non_route_node_reference_still_uses_node_selection(monkeypatch) -> None:
    mock_understanding(monkeypatch, selected_node_layout_goal())

    trace = semantic_shadow.run_semantic_shadow(
        graph(),
        "move this node right of End",
        {"node_ids": ["n_report"], "edge_ids": ["e_decision_report"]},
    )

    assert trace["stage"] == "success"
    assert any(
        row["reference"] == "this node" and row["resolved_id"] == "n_report" and row["resolution_type"] == "selection"
        for row in trace["grounded_goal"]["grounding"]["provenance"]
    )


def test_real_api_selected_edge_existing_reference_recovery(monkeypatch) -> None:
    server.SESSIONS.clear()
    monkeypatch.setenv("AI_EDIT_PIPELINE", "semantic")
    mock_understanding(monkeypatch, route_existing_reference_goal())
    created = server.create_session({"graph": graph()})

    response = server.run_edit(
        {
            "session_id": created["session_id"],
            "instruction": "route this edge to End",
            "selection": {"node_ids": [], "edge_ids": ["e_decision_report"]},
        }
    )

    assert response["status"] == "success"
    assert response["transaction"]["selection"] == {"node_ids": [], "edge_ids": ["e_decision_report"]}
    edges = {edge["id"]: edge for edge in response["graph"]["edges"]}
    assert edges["e_decision_report"]["target"] == "n_end"
    assert response["diff"]["changed_edges"] == [
        {"id": "e_decision_report", "changes": {"target": {"before": "n_report", "after": "n_end"}}}
    ]


def test_real_api_multiple_selected_edges_clarifies_with_edge_candidates(monkeypatch) -> None:
    server.SESSIONS.clear()
    monkeypatch.setenv("AI_EDIT_PIPELINE", "semantic")
    mock_understanding(monkeypatch, route_existing_reference_goal())
    created = server.create_session({"graph": graph()})

    response = server.run_edit(
        {
            "session_id": created["session_id"],
            "instruction": "route this edge to End",
            "selection": {"node_ids": [], "edge_ids": ["e_decision_report", "e_decision_manual"]},
        }
    )

    assert response["status"] == "needs_clarification"
    assert response["transaction"]["selection"]["edge_ids"] == ["e_decision_report", "e_decision_manual"]
    issues = response["specification"]["trace"]["validation"]["issues"]
    assert issues[0]["kind"] == "ambiguous_edge_reference"
    assert issues[0]["candidates"] == ["e_decision_manual", "e_decision_report"]


def test_real_api_no_selected_edge_clarifies_without_guessing(monkeypatch) -> None:
    server.SESSIONS.clear()
    monkeypatch.setenv("AI_EDIT_PIPELINE", "semantic")
    mock_understanding(monkeypatch, route_existing_reference_goal())
    created = server.create_session({"graph": graph()})

    response = server.run_edit(
        {
            "session_id": created["session_id"],
            "instruction": "route this edge to End",
            "selection": {"node_ids": [], "edge_ids": []},
        }
    )

    assert response["status"] == "needs_clarification"
    issues = response["specification"]["trace"]["validation"]["issues"]
    assert issues[0]["kind"] == "ambiguous_edge_reference"
    assert issues[0]["candidates"] == ["e_decision_manual", "e_decision_report", "e_report_end"]
    edges = {edge["id"]: edge for edge in response["graph"]["edges"]}
    assert edges["e_decision_report"]["target"] == "n_report"
