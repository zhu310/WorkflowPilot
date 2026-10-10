from demo import server

from demo_api_helpers import assert_readable_response, create_session, demo_graph, mock_understanding


def rename_report_spec() -> dict:
    return {
        "status": "resolved",
        "semantic_goals": [
            {
                "id": "g1",
                "goal_category": "object_state_goal",
                "text": "把生成报告节点名称改成生成正式报告",
                "objects": [{"id": "report", "type": "existing_reference", "text": "生成报告"}],
                "object_desired_states": [
                    {"object": "report", "state": "property_equals", "property": "display_label", "value": "生成正式报告"}
                ],
            }
        ],
        "clarification": {"needed": False, "question": "", "reason": ""},
    }


def test_demo_api_message_and_user_feedback_are_readable(monkeypatch) -> None:
    server.SESSIONS.clear()
    monkeypatch.setenv("AI_EDIT_PIPELINE", "semantic")
    mock_understanding(monkeypatch, rename_report_spec())
    session_id = create_session(demo_graph())

    response = server.run_edit(
        {
            "session_id": session_id,
            "instruction": "把生成报告节点名称改成生成正式报告",
            "selection": {"node_ids": [], "edge_ids": []},
        }
    )

    assert response["status"] == "success"
    assert_readable_response(response)
    assert "生成正式报告" in response["message"]
    assert response["user_feedback"]["executed"]
