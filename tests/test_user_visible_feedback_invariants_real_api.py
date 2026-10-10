from __future__ import annotations

from typing import Any

from demo import server
from demo_api_helpers import assert_no_internal_visible_text, create_session, mock_understanding


def graph() -> dict[str, Any]:
    return {
        "nodes": [
            {"id": "decision", "type": "decision", "label": "Decision", "x": 100, "y": 100},
            {"id": "report", "type": "task", "label": "Report", "x": 300, "y": 60},
            {"id": "publish", "type": "task", "label": "Publish", "x": 500, "y": 60},
            {"id": "human", "type": "task", "label": "Human Review", "x": 300, "y": 240},
            {"id": "end", "type": "end", "label": "End", "x": 700, "y": 100},
        ],
        "edges": [
            {"id": "e_success", "source": "decision", "target": "report", "label": "success", "branch_label": "Success"},
            {"id": "e_failure", "source": "decision", "target": "human", "label": "failure", "branch_label": "Failure"},
            {"id": "e_report_publish", "source": "report", "target": "publish", "label": ""},
            {"id": "e_publish_end", "source": "publish", "target": "end", "label": ""},
        ],
    }


def visible_strings(value: Any, *, in_debug: bool = False):
    if isinstance(value, str):
        if not in_debug:
            yield value
    elif isinstance(value, list):
        for item in value:
            yield from visible_strings(item, in_debug=in_debug)
    elif isinstance(value, dict):
        for key, item in value.items():
            debug = in_debug or key in {
                "raw_partial_execution",
                "specification",
                "semantic_shadow",
                "transaction",
                "committed_version",
                "state",
                "mutation_authorization",
                "graph_context",
                "target",
                "steps",
                "final_graph",
                "graph",
                "diff",
                "transaction_diff",
            }
            yield from visible_strings(item, in_debug=debug)


def assert_visible_response_is_readable(response: dict[str, Any]) -> None:
    for text in visible_strings(response):
        assert_no_internal_visible_text(text)


def rename_spec() -> dict[str, Any]:
    return {
        "status": "resolved",
        "semantic_goals": [
            {
                "id": "g1",
                "goal_category": "object_state_goal",
                "text": "rename report",
                "objects": [{"id": "report_obj", "type": "existing_reference", "text": "Report"}],
                "object_desired_states": [
                    {"object": "report_obj", "state": "property_equals", "property": "display_label", "value": "Final Report"}
                ],
            }
        ],
        "clarification": {"needed": False, "question": "", "reason": ""},
    }


def insert_spec() -> dict[str, Any]:
    return {
        "status": "resolved",
        "semantic_goals": [
            {
                "id": "g1",
                "goal_category": "composite_goal",
                "text": "insert check before end",
                "objects": [
                    {"id": "check", "type": "new_entity", "label": "Exception Check"},
                    {"id": "end_obj", "type": "existing_reference", "text": "End"},
                ],
                "object_desired_states": [{"object": "check", "state": "exists"}],
                "relation_intents": [{"source": "check", "relation": "before", "target": "end_obj"}],
            }
        ],
        "clarification": {"needed": False, "question": "", "reason": ""},
    }


def route_spec(edge_text: str = "this branch", target_text: str = "Human Review") -> dict[str, Any]:
    return {
        "status": "resolved",
        "semantic_goals": [
            {
                "id": "g1",
                "goal_category": "relation_goal",
                "text": "route branch",
                "objects": [
                    {"id": "route", "type": "edge_reference", "text": edge_text},
                    {"id": "target", "type": "existing_reference", "text": target_text},
                ],
                "relation_intents": [{"source": "route", "relation": "route_to", "target": "target"}],
            }
        ],
        "clarification": {"needed": False, "question": "", "reason": ""},
    }


def unsupported_relation_spec() -> dict[str, Any]:
    return {
        "status": "resolved",
        "semantic_goals": [
            {
                "id": "g1",
                "goal_category": "relation_goal",
                "text": "connect report to human",
                "objects": [
                    {"id": "source", "type": "existing_reference", "text": "Report"},
                    {"id": "target", "type": "existing_reference", "text": "Human Review"},
                ],
                "relation_intents": [{"source": "source", "relation": "connected_to", "target": "target"}],
            }
        ],
        "clarification": {"needed": False, "question": "", "reason": ""},
    }


def contract_violation_spec() -> dict[str, Any]:
    return {
        "status": "resolved",
        "semantic_goals": [
            {
                "id": "g1",
                "goal_category": "object_state_goal",
                "text": "rename report",
                "objects": [{"id": "report_obj", "type": "existing_reference", "text": "Pre-resolved Report Node"}],
                "object_desired_states": [
                    {"object": "report_obj", "state": "property_equals", "property": "display_label", "value": "Final Report"}
                ],
            }
        ],
        "clarification": {"needed": False, "question": "", "reason": ""},
    }


def abstract_safe_stop_spec() -> dict[str, Any]:
    return {
        "status": "needs_clarification",
        "semantic_goals": [],
        "clarification": {"needed": True, "question": "请说明你希望优化的具体目标或范围。", "reason": "abstract request"},
    }


def run_with_spec(monkeypatch, spec: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    server.SESSIONS.clear()
    monkeypatch.setenv("AI_EDIT_PIPELINE", "semantic")
    mock_understanding(monkeypatch, spec)
    session_id = create_session(graph())
    return server.run_edit({"session_id": session_id, **payload})


def test_real_api_visible_feedback_invariants_for_success_and_clarification_paths(monkeypatch) -> None:
    cases = [
        (
            rename_spec(),
            {"instruction": "rename report", "selection": {"node_ids": [], "edge_ids": []}},
        ),
        (
            insert_spec(),
            {"instruction": "insert check before end", "selection": {"node_ids": [], "edge_ids": []}},
        ),
        (
            route_spec(),
            {"instruction": "route this branch to human", "selection": {"node_ids": [], "edge_ids": []}},
        ),
        (
            route_spec(),
            {"instruction": "route this branch to human", "selection": {"node_ids": [], "edge_ids": ["e_success"]}},
        ),
        (
            abstract_safe_stop_spec(),
            {"instruction": "optimize this workflow", "selection": {"node_ids": [], "edge_ids": []}},
        ),
    ]
    for spec, payload in cases:
        response = run_with_spec(monkeypatch, spec, payload)
        assert_visible_response_is_readable(response)


def test_real_api_ambiguous_edge_question_does_not_expose_raw_reason(monkeypatch) -> None:
    response = run_with_spec(
        monkeypatch,
        route_spec("this branch", "Human Review"),
        {"instruction": "route this branch to human", "selection": {"node_ids": [], "edge_ids": []}},
    )

    assert response["status"] in {"needs_clarification", "unsupported"}
    assert "multiple graph edges match reference" not in response["message"]
    assert "multiple graph edges match reference" not in response["user_feedback"]["message"]
    assert_visible_response_is_readable(response)


def test_real_api_missing_edge_question_does_not_expose_raw_reason(monkeypatch) -> None:
    response = run_with_spec(
        monkeypatch,
        route_spec("nonexistent branch", "Human Review"),
        {"instruction": "route nonexistent branch to human", "selection": {"node_ids": [], "edge_ids": []}},
    )

    assert response["status"] in {"needs_clarification", "unsupported"}
    assert "no graph edge matches reference" not in response["message"]
    assert "no graph edge matches reference" not in response["user_feedback"]["message"]
    assert response["user_feedback"]["incomplete"] or response["user_feedback"]["clarification"]["question"]
    assert_visible_response_is_readable(response)


def test_real_api_lowering_reason_does_not_enter_user_visible_fields(monkeypatch) -> None:
    response = run_with_spec(
        monkeypatch,
        unsupported_relation_spec(),
        {"instruction": "connect report to human", "selection": {"node_ids": [], "edge_ids": []}},
    )

    assert response["status"] in {"unsupported", "partial_success", "needs_clarification"}
    assert_visible_response_is_readable(response)


def test_real_api_semantic_contract_violation_message_is_debug_only(monkeypatch) -> None:
    response = run_with_spec(
        monkeypatch,
        contract_violation_spec(),
        {"instruction": "rename report", "selection": {"node_ids": [], "edge_ids": []}},
    )

    assert response["status"] in {"unsupported", "needs_clarification"}
    assert_visible_response_is_readable(response)
