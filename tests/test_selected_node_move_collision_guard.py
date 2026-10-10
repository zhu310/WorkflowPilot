from demo import server

from demo_api_helpers import assert_readable_response, create_session, demo_graph, mock_understanding


def move_selected_node_right_spec() -> dict:
    return {
        "status": "resolved",
        "semantic_goals": [
            {
                "id": "g1",
                "goal_category": "relation_goal",
                "text": "把这个节点移动到右边",
                "objects": [{"id": "selected", "type": "existing_reference", "text": "这个节点"}],
                "relation_intents": [{"source": "selected", "relation": "right_of", "target": "selected"}],
            }
        ],
        "clarification": {"needed": False, "question": "", "reason": ""},
    }


def test_selected_node_move_right_chooses_non_overlapping_position(monkeypatch) -> None:
    server.SESSIONS.clear()
    monkeypatch.setenv("AI_EDIT_PIPELINE", "semantic")
    mock_understanding(monkeypatch, move_selected_node_right_spec())
    session_id = create_session(demo_graph())

    response = server.run_edit(
        {
            "session_id": session_id,
            "instruction": "把这个节点移动到右边",
            "selection": {"node_ids": ["n_report"], "edge_ids": []},
        }
    )

    assert response["status"] == "success"
    report = next(node for node in response["graph"]["nodes"] if node["id"] == "n_report")
    publish = next(node for node in response["graph"]["nodes"] if node["id"] == "n_publish")
    assert report["x"] > publish["x"] + (publish["width"] + report["width"]) / 2
    assert publish["x"] == 1090
    assert_readable_response(response)
