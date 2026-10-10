from demo import server

from demo_api_helpers import assert_readable_response, create_session, demo_graph, mock_understanding


def route_selected_success_to_human_spec() -> dict:
    return {
        "status": "resolved",
        "semantic_goals": [
            {
                "id": "g1",
                "goal_category": "relation_goal",
                "text": "这个分支转人工复核",
                "objects": [
                    {"id": "route", "type": "edge_reference", "text": "这个分支"},
                    {"id": "human", "type": "existing_reference", "text": "人工复核"},
                ],
                "relation_intents": [{"source": "route", "relation": "route_to", "target": "human"}],
            }
        ],
        "clarification": {"needed": False, "question": "", "reason": ""},
    }


def payload(session_id: str, instruction: str, edge_ids: list[str] | None = None) -> dict:
    return {
        "session_id": session_id,
        "instruction": instruction,
        "selection": {"node_ids": [], "edge_ids": edge_ids or []},
    }


def test_same_target_route_confirmation_real_selected_edge(monkeypatch) -> None:
    server.SESSIONS.clear()
    monkeypatch.setenv("AI_EDIT_PIPELINE", "semantic")
    mock_understanding(monkeypatch, route_selected_success_to_human_spec())
    session_id = create_session(demo_graph())

    first = server.run_edit(payload(session_id, "这个分支转人工复核", ["e_decision_report"]))

    assert first["status"] == "needs_clarification"
    assert first["pending_id"]
    assert first["diff"] == {
        "added_nodes": [],
        "removed_nodes": [],
        "changed_nodes": [],
        "added_edges": [],
        "removed_edges": [],
        "changed_edges": [],
    }
    assert "结果判断" in first["message"]
    assert "人工复核" in first["message"]
    assert_readable_response(first)

    second = server.run_edit(payload(session_id, "确认"))
    assert second["status"] == "success"
    assert_readable_response(second)
    success_edge = next(edge for edge in second["graph"]["edges"] if edge["id"] == "e_decision_report")
    assert success_edge["target"] == "n_human"


def test_same_target_route_cancel_does_not_execute(monkeypatch) -> None:
    server.SESSIONS.clear()
    monkeypatch.setenv("AI_EDIT_PIPELINE", "semantic")
    mock_understanding(monkeypatch, route_selected_success_to_human_spec())
    session_id = create_session(demo_graph())

    first = server.run_edit(payload(session_id, "这个分支转人工复核", ["e_decision_report"]))
    assert first["status"] == "needs_clarification"

    cancelled = server.run_edit(payload(session_id, "取消"))
    assert cancelled["status"] == "cancelled"
    success_edge = next(edge for edge in cancelled["graph"]["edges"] if edge["id"] == "e_decision_report")
    assert success_edge["target"] == "n_report"
    assert cancelled["state"]["pending_clarification"] is None
