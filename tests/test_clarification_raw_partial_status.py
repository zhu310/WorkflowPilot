from demo import server

from demo_api_helpers import assert_readable_response, create_session, demo_graph, mock_understanding


def clarification_spec() -> dict:
    return {
        "status": "clarification",
        "semantic_goals": [],
        "clarification": {
            "needed": True,
            "question": "请说明具体要优化什么。",
            "reason": "abstract request",
        },
    }


def test_clarification_feedback_status_is_not_success_and_hides_raw_partial(monkeypatch) -> None:
    server.SESSIONS.clear()
    monkeypatch.setenv("AI_EDIT_PIPELINE", "semantic")
    mock_understanding(monkeypatch, clarification_spec())
    session_id = create_session(demo_graph())

    response = server.run_edit(
        {
            "session_id": session_id,
            "instruction": "优化一下这个流程",
            "selection": {"node_ids": [], "edge_ids": []},
        }
    )

    assert response["status"] == "needs_clarification"
    assert_readable_response(response)
    assert response["user_feedback"]["status"] == "needs_clarification"
    assert "raw_partial_execution" not in response["user_feedback"]
