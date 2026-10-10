from __future__ import annotations

from demo import server
from demo_api_helpers import assert_readable_response, create_session, mock_understanding


def graph() -> dict:
    return {
        "nodes": [
            {"id": "n_report", "type": "task", "label": "Generate Report", "x": 100, "y": 100},
            {"id": "n_publish", "type": "task", "label": "Publish Result", "x": 300, "y": 100},
            {"id": "n_end", "type": "end", "label": "End", "x": 500, "y": 100},
            {"id": "n_archive", "type": "task", "label": "Archive", "x": 500, "y": 240},
            {"id": "n_review", "type": "task", "label": "Review", "x": 500, "y": 360},
        ],
        "edges": [
            {"id": "e_report_publish", "source": "n_report", "target": "n_publish", "label": ""},
            {"id": "e_publish_end", "source": "n_publish", "target": "n_end", "label": ""},
            {"id": "e_report_archive", "source": "n_report", "target": "n_archive", "label": "archive"},
        ],
    }


def route_selected_edge_spec(target_text: str = "End") -> dict:
    return {
        "status": "resolved",
        "semantic_goals": [
            {
                "id": "g1",
                "goal_category": "relation_goal",
                "text": "route this branch",
                "objects": [
                    {"id": "route", "type": "edge_reference", "text": "this branch"},
                    {"id": "target", "type": "existing_reference", "text": target_text},
                ],
                "relation_intents": [{"source": "route", "relation": "route_to", "target": "target"}],
            }
        ],
        "clarification": {"needed": False, "question": "", "reason": ""},
    }


def test_selected_edge_route_to_updates_selected_edge_target(monkeypatch) -> None:
    server.SESSIONS.clear()
    monkeypatch.setenv("AI_EDIT_PIPELINE", "semantic")
    mock_understanding(monkeypatch, route_selected_edge_spec())
    session_id = create_session(graph())

    response = server.run_edit(
        {
            "session_id": session_id,
            "instruction": "route this branch to end",
            "selection": {"node_ids": [], "edge_ids": ["e_report_publish"]},
        }
    )

    assert response["status"] == "success"
    assert_readable_response(response)
    edges = {edge["id"]: edge for edge in response["graph"]["edges"]}
    assert edges["e_report_publish"]["source"] == "n_report"
    assert edges["e_report_publish"]["target"] == "n_end"
    assert edges["e_publish_end"]["source"] == "n_publish"
    assert edges["e_publish_end"]["target"] == "n_end"
    assert not any(
        edge["id"] != "e_publish_end" and edge["source"] == "n_publish" and edge["target"] == "n_end"
        for edge in response["graph"]["edges"]
    )
    assert response["diff"]["added_edges"] == []
    assert response["diff"]["removed_edges"] == []
    assert response["diff"]["changed_edges"] == [
        {"id": "e_report_publish", "changes": {"target": {"before": "n_publish", "after": "n_end"}}}
    ]


def test_selected_edge_route_to_regular_node_updates_selected_edge(monkeypatch) -> None:
    server.SESSIONS.clear()
    monkeypatch.setenv("AI_EDIT_PIPELINE", "semantic")
    mock_understanding(monkeypatch, route_selected_edge_spec("Review"))
    session_id = create_session(graph())

    response = server.run_edit(
        {
            "session_id": session_id,
            "instruction": "route this branch to review",
            "selection": {"node_ids": [], "edge_ids": ["e_report_publish"]},
        }
    )

    assert response["status"] == "success"
    edges = {edge["id"]: edge for edge in response["graph"]["edges"]}
    assert edges["e_report_publish"]["target"] == "n_review"
    assert response["diff"]["added_edges"] == []
    assert response["diff"]["removed_edges"] == []
    assert response["diff"]["changed_edges"][0]["id"] == "e_report_publish"


def test_route_this_branch_without_selection_stops_for_clarification(monkeypatch) -> None:
    server.SESSIONS.clear()
    monkeypatch.setenv("AI_EDIT_PIPELINE", "semantic")
    mock_understanding(monkeypatch, route_selected_edge_spec())
    session_id = create_session(graph())

    response = server.run_edit(
        {
            "session_id": session_id,
            "instruction": "route this branch to end",
            "selection": {"node_ids": [], "edge_ids": []},
        }
    )

    assert response["status"] in {"needs_clarification", "unsupported"}
    assert response["diff"] == {"added_nodes": [], "removed_nodes": [], "changed_nodes": [], "added_edges": [], "removed_edges": [], "changed_edges": []}
