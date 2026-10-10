from types import SimpleNamespace

from demo import server
from src import semantic_shadow


def mock_understanding(monkeypatch, spec: dict) -> None:
    monkeypatch.setattr(
        semantic_shadow,
        "_complete_json",
        lambda _system, _payload: SimpleNamespace(raw_response="", parsed_json=spec, parse_error=None, normalized_from_fence=False),
    )


def create_session(graph: dict) -> str:
    result = server.create_session({"graph": graph})
    assert result["status"] == "success"
    return result["session_id"]


def duplicate_review_graph() -> dict:
    return {
        "nodes": [
            {"id": "review_a", "type": "task", "label": "Duplicate Review"},
            {"id": "review_b", "type": "task", "label": "Duplicate Review"},
        ],
        "edges": [],
    }


def route_graph() -> dict:
    return {
        "nodes": [
            {"id": "decision", "type": "decision", "label": "Decision"},
            {"id": "reject_a", "type": "task", "label": "Reject A"},
            {"id": "reject_b", "type": "task", "label": "Reject B"},
            {"id": "manual", "type": "task", "label": "Manual Review"},
        ],
        "edges": [
            {"id": "edge_a", "source": "decision", "target": "reject_a", "label": "failure", "branch_label": "Failure"},
            {"id": "edge_b", "source": "decision", "target": "reject_b", "label": "failure", "branch_label": "Failure"},
        ],
    }


def test_node_candidate_selection_continues_pending_delete(monkeypatch) -> None:
    server.SESSIONS.clear()
    monkeypatch.setenv("AI_EDIT_PIPELINE", "semantic")
    mock_understanding(
        monkeypatch,
        {
            "status": "resolved",
            "semantic_goals": [
                {
                    "id": "g1",
                    "goal_category": "object_state_goal",
                    "text": "Delete Duplicate Review",
                    "objects": [{"id": "target", "type": "existing_reference", "text": "Duplicate Review"}],
                    "object_desired_states": [{"object": "target", "state": "absent"}],
                }
            ],
            "clarification": {"needed": False, "question": "", "reason": ""},
        },
    )
    session_id = create_session(duplicate_review_graph())

    first = server.run_edit({"session_id": session_id, "instruction": "Delete Duplicate Review"})
    assert first["status"] == "needs_clarification"
    assert first["pending_id"]
    assert len(first["user_feedback"]["clarification"]["candidates"][0]["candidates"]) == 2

    second = server.run_edit({"session_id": session_id, "instruction": "first"})
    assert second["status"] == "success"
    remaining = {node["id"] for node in second["graph"]["nodes"]}
    assert "review_a" not in remaining
    assert "review_b" in remaining


def test_edge_candidate_selection_continues_pending_route(monkeypatch) -> None:
    server.SESSIONS.clear()
    monkeypatch.setenv("AI_EDIT_PIPELINE", "semantic")
    mock_understanding(
        monkeypatch,
        {
            "status": "resolved",
            "semantic_goals": [
                {
                    "id": "g1",
                    "goal_category": "relation_goal",
                    "text": "Route Failure to Manual Review",
                    "objects": [
                        {"id": "route", "type": "edge_reference", "text": "Failure"},
                        {"id": "manual", "type": "existing_reference", "text": "Manual Review"},
                    ],
                    "relation_intents": [{"source": "route", "relation": "route_to", "target": "manual"}],
                }
            ],
            "clarification": {"needed": False, "question": "", "reason": ""},
        },
    )
    session_id = create_session(route_graph())

    first = server.run_edit({"session_id": session_id, "instruction": "Route Failure to Manual Review"})
    assert first["status"] == "needs_clarification"

    second = server.run_edit({"session_id": session_id, "instruction": "second"})
    assert second["status"] == "success"
    assert any(edge.get("source") == "decision" and edge.get("target") == "manual" for edge in second["graph"]["edges"])


def test_invalid_candidate_selection_stops_without_edit(monkeypatch) -> None:
    server.SESSIONS.clear()
    monkeypatch.setenv("AI_EDIT_PIPELINE", "semantic")
    mock_understanding(
        monkeypatch,
        {
            "status": "resolved",
            "semantic_goals": [
                {
                    "id": "g1",
                    "goal_category": "object_state_goal",
                    "text": "Delete Duplicate Review",
                    "objects": [{"id": "target", "type": "existing_reference", "text": "Duplicate Review"}],
                    "object_desired_states": [{"object": "target", "state": "absent"}],
                }
            ],
            "clarification": {"needed": False, "question": "", "reason": ""},
        },
    )
    session_id = create_session(duplicate_review_graph())
    server.run_edit({"session_id": session_id, "instruction": "Delete Duplicate Review"})

    result = server.run_edit({"session_id": session_id, "instruction": "third"})
    assert result["status"] == "needs_clarification"
    assert len(result["graph"]["nodes"]) == 2


def test_selection_without_pending_stops(monkeypatch) -> None:
    server.SESSIONS.clear()
    monkeypatch.setenv("AI_EDIT_PIPELINE", "semantic")
    session_id = create_session(duplicate_review_graph())

    result = server.run_edit({"session_id": session_id, "instruction": "first"})
    assert result["status"] == "needs_clarification"
    assert "没有待确认" in result["clarification_message"] or "no pending" in result["clarification_message"].lower()


def test_stale_pending_candidate_stops(monkeypatch) -> None:
    server.SESSIONS.clear()
    monkeypatch.setenv("AI_EDIT_PIPELINE", "semantic")
    mock_understanding(
        monkeypatch,
        {
            "status": "resolved",
            "semantic_goals": [
                {
                    "id": "g1",
                    "goal_category": "object_state_goal",
                    "text": "Delete Duplicate Review",
                    "objects": [{"id": "target", "type": "existing_reference", "text": "Duplicate Review"}],
                    "object_desired_states": [{"object": "target", "state": "absent"}],
                }
            ],
            "clarification": {"needed": False, "question": "", "reason": ""},
        },
    )
    session_id = create_session(duplicate_review_graph())
    server.run_edit({"session_id": session_id, "instruction": "Delete Duplicate Review"})
    state = server.SESSIONS[session_id]
    state.versions[state.current_version_id].graph_snapshot["nodes"] = [state.versions[state.current_version_id].graph_snapshot["nodes"][1]]

    result = server.run_edit({"session_id": session_id, "instruction": "first"})
    assert result["status"] == "needs_clarification"
    assert "已经不存在" in result["clarification_message"] or "no longer exists" in result["clarification_message"].lower()
