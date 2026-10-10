from demo import server

from demo_api_helpers import assert_readable_response, create_session, demo_graph, mock_understanding


def add_exception_before_end_spec() -> dict:
    return {
        "status": "resolved",
        "semantic_goals": [
            {
                "id": "g1",
                "goal_category": "composite_goal",
                "text": "在结束前增加异常处理节点",
                "objects": [
                    {"id": "exists_node", "type": "new_entity", "label": "异常处理节点"},
                    {"id": "inserted_node", "type": "new_entity", "label": "异常处理"},
                    {"id": "end", "type": "existing_reference", "text": "结束"},
                ],
                "object_desired_states": [{"object": "exists_node", "state": "exists"}],
                "relation_intents": [{"source": "exists_node", "relation": "before", "target": "end"}],
            }
        ],
        "clarification": {"needed": False, "question": "", "reason": ""},
    }


def test_no_duplicate_exception_node_for_real_demo_input(monkeypatch) -> None:
    server.SESSIONS.clear()
    monkeypatch.setenv("AI_EDIT_PIPELINE", "semantic")
    mock_understanding(monkeypatch, add_exception_before_end_spec())
    session_id = create_session(demo_graph())

    response = server.run_edit(
        {
            "session_id": session_id,
            "instruction": "在结束前增加异常处理节点",
            "selection": {"node_ids": [], "edge_ids": []},
        }
    )

    assert response["status"] == "success"
    assert_readable_response(response)
    added = response["diff"]["added_nodes"]
    assert len([node for node in added if "异常处理" in node.get("label", "")]) == 1
    all_exception_nodes = [node for node in response["graph"]["nodes"] if "异常处理" in node.get("label", "")]
    assert len(all_exception_nodes) == 1
    created_id = all_exception_nodes[0]["id"]
    assert any(edge.get("source") == created_id or edge.get("target") == created_id for edge in response["graph"]["edges"])
