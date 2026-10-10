from __future__ import annotations

from typing import Any

from demo import server
from demo_api_helpers import assert_no_internal_visible_text, create_session, mock_understanding


def graph() -> dict[str, Any]:
    nodes = [{"id": "decision", "type": "decision", "label": "Decision"}]
    edges = []
    for index in range(8):
        target = f"target_{index}"
        nodes.append({"id": target, "type": "task", "label": f"Target {index}"})
        edges.append(
            {
                "id": f"edge_{index}",
                "source": "decision",
                "target": target,
                "label": f"branch_{index}",
                "branch_label": f"Branch {index}",
            }
        )
    return {"nodes": nodes, "edges": edges}


def route_this_branch_spec() -> dict[str, Any]:
    return {
        "status": "resolved",
        "semantic_goals": [
            {
                "id": "g1",
                "goal_category": "relation_goal",
                "text": "route this branch",
                "objects": [
                    {"id": "route", "type": "edge_reference", "text": "this branch"},
                    {"id": "target", "type": "existing_reference", "text": "Target 0"},
                ],
                "relation_intents": [{"source": "route", "relation": "route_to", "target": "target"}],
            }
        ],
        "clarification": {"needed": False, "question": "", "reason": ""},
    }


def test_ambiguous_branch_user_message_does_not_mention_debug_details(monkeypatch) -> None:
    server.SESSIONS.clear()
    monkeypatch.setenv("AI_EDIT_PIPELINE", "semantic")
    mock_understanding(monkeypatch, route_this_branch_spec())
    session_id = create_session(graph())

    response = server.run_edit(
        {
            "session_id": session_id,
            "instruction": "route this branch to Target 0",
            "selection": {"node_ids": [], "edge_ids": []},
        }
    )

    message = str(response.get("message") or "")
    feedback_message = str(response.get("user_feedback", {}).get("message") or "")
    assert "Debug details" not in message
    assert "Debug details" not in feedback_message
    assert "还有" in message
    assert_no_internal_visible_text(message)
    assert_no_internal_visible_text(feedback_message)
