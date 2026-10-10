from types import SimpleNamespace

from demo import server
from src import semantic_shadow


def mock_understanding(monkeypatch, spec: dict) -> None:
    monkeypatch.setattr(
        semantic_shadow,
        "_complete_json",
        lambda _system, _payload: SimpleNamespace(
            raw_response="",
            parsed_json=spec,
            parse_error=None,
            normalized_from_fence=False,
        ),
    )


def graph() -> dict:
    return {
        "nodes": [
            {"id": "decision", "type": "decision", "label": "Decision"},
            {"id": "report", "type": "task", "label": "Report"},
            {"id": "human", "type": "task", "label": "Human Review"},
        ],
        "edges": [
            {
                "id": "e_success",
                "source": "decision",
                "target": "report",
                "label": "success",
                "branch_label": "Success",
            },
            {
                "id": "e_failure",
                "source": "decision",
                "target": "human",
                "label": "failure",
                "branch_label": "Failure",
            },
        ],
    }


def route_selected_branch_to_human_spec() -> dict:
    return {
        "status": "resolved",
        "semantic_goals": [
            {
                "id": "g1",
                "goal_category": "relation_goal",
                "text": "this branch routes to Human Review",
                "objects": [
                    {"id": "route", "type": "edge_reference", "text": "this branch"},
                    {"id": "human", "type": "existing_reference", "text": "Human Review"},
                ],
                "relation_intents": [{"source": "route", "relation": "route_to", "target": "human"}],
            }
        ],
        "clarification": {"needed": False, "question": "", "reason": ""},
    }


def create_session() -> str:
    result = server.create_session({"graph": graph()})
    assert result["status"] == "success"
    return result["session_id"]


def selected_success_payload(session_id: str, instruction: str) -> dict:
    return {
        "session_id": session_id,
        "instruction": instruction,
        "selection": {"node_ids": [], "edge_ids": ["e_success"]},
    }


def test_route_to_same_source_same_target_requires_confirmation(monkeypatch) -> None:
    server.SESSIONS.clear()
    monkeypatch.setenv("AI_EDIT_PIPELINE", "semantic")
    mock_understanding(monkeypatch, route_selected_branch_to_human_spec())
    session_id = create_session()

    result = server.run_edit(selected_success_payload(session_id, "this branch routes to Human Review"))

    assert result["status"] == "needs_clarification"
    assert result["pending_id"]
    assert result["diff"] == {
        "added_nodes": [],
        "removed_nodes": [],
        "changed_nodes": [],
        "added_edges": [],
        "removed_edges": [],
        "changed_edges": [],
    }
    assert "Decision" in result["user_feedback"]["message"]
    assert "Human Review" in result["user_feedback"]["message"]


def test_route_to_same_source_same_target_confirmation_executes(monkeypatch) -> None:
    server.SESSIONS.clear()
    monkeypatch.setenv("AI_EDIT_PIPELINE", "semantic")
    mock_understanding(monkeypatch, route_selected_branch_to_human_spec())
    session_id = create_session()

    first = server.run_edit(selected_success_payload(session_id, "this branch routes to Human Review"))
    assert first["status"] == "needs_clarification"

    second = server.run_edit({"session_id": session_id, "instruction": "yes", "selection": {"node_ids": [], "edge_ids": []}})

    assert second["status"] == "success"
    success = next(edge for edge in second["graph"]["edges"] if edge["id"] == "e_success")
    assert success["target"] == "human"
    assert second["state"]["pending_clarification"] is None


def test_route_to_same_source_same_target_cancel_does_not_edit(monkeypatch) -> None:
    server.SESSIONS.clear()
    monkeypatch.setenv("AI_EDIT_PIPELINE", "semantic")
    mock_understanding(monkeypatch, route_selected_branch_to_human_spec())
    session_id = create_session()

    server.run_edit(selected_success_payload(session_id, "this branch routes to Human Review"))
    cancelled = server.run_edit({"session_id": session_id, "instruction": "cancel", "selection": {"node_ids": [], "edge_ids": []}})

    assert cancelled["status"] == "cancelled"
    success = next(edge for edge in cancelled["graph"]["edges"] if edge["id"] == "e_success")
    assert success["target"] == "report"
    assert cancelled["state"]["pending_clarification"] is None
