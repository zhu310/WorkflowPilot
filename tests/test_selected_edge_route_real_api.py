from __future__ import annotations

from typing import Any

from demo import server
from demo_api_helpers import assert_no_internal_visible_text, create_session, mock_understanding


def graph() -> dict[str, Any]:
    return {
        "nodes": [
            {"id": "report", "type": "task", "label": "Report", "x": 100, "y": 100},
            {"id": "publish", "type": "task", "label": "Publish", "x": 300, "y": 100},
            {"id": "end", "type": "end", "label": "End", "x": 500, "y": 100},
        ],
        "edges": [
            {"id": "edge_report_publish", "source": "report", "target": "publish", "label": ""},
            {"id": "edge_publish_end", "source": "publish", "target": "end", "label": ""},
        ],
    }


def route_selected_edge_to_end_spec() -> dict[str, Any]:
    return {
        "status": "resolved",
        "semantic_goals": [
            {
                "id": "g1",
                "goal_category": "relation_goal",
                "text": "this branch routes to End",
                "objects": [
                    {"id": "route", "type": "edge_reference", "text": "this branch"},
                    {"id": "target", "type": "existing_reference", "text": "End"},
                ],
                "relation_intents": [{"source": "route", "relation": "route_to", "target": "target"}],
            }
        ],
        "clarification": {"needed": False, "question": "", "reason": ""},
    }


def visible_texts(response: dict[str, Any]):
    feedback = response.get("user_feedback", {})
    yield str(response.get("message") or "")
    yield str(feedback.get("message") or "") if isinstance(feedback, dict) else ""
    if isinstance(feedback, dict):
        for row in feedback.get("executed", []) if isinstance(feedback.get("executed"), list) else []:
            if isinstance(row, dict):
                yield str(row.get("summary") or "")


def test_selected_edge_route_real_api_updates_selected_edge_and_feedback(monkeypatch) -> None:
    server.SESSIONS.clear()
    monkeypatch.setenv("AI_EDIT_PIPELINE", "semantic")
    mock_understanding(monkeypatch, route_selected_edge_to_end_spec())
    session_id = create_session(graph())

    response = server.run_edit(
        {
            "session_id": session_id,
            "instruction": "route this branch to end",
            "selection": {"node_ids": [], "edge_ids": ["edge_report_publish"]},
        }
    )

    assert response["status"] == "success"
    edges = {edge["id"]: edge for edge in response["graph"]["edges"]}
    assert edges["edge_report_publish"]["source"] == "report"
    assert edges["edge_report_publish"]["target"] == "end"
    assert edges["edge_publish_end"]["source"] == "publish"
    assert edges["edge_publish_end"]["target"] == "end"
    assert response["diff"]["added_edges"] == []
    assert response["diff"]["removed_edges"] == []
    assert response["diff"]["changed_edges"] == [
        {"id": "edge_report_publish", "changes": {"target": {"before": "publish", "after": "end"}}}
    ]
    for text in visible_texts(response):
        assert_no_internal_visible_text(text)
    assert "Report -> Publish" in response["message"]
    assert "End" in response["message"]
