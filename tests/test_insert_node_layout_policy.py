from __future__ import annotations

from demo import server
from demo_api_helpers import assert_readable_response, create_session, mock_understanding


def serial_graph() -> dict:
    return {
        "nodes": [
            {"id": "start", "type": "start", "label": "Start", "x": 100, "y": 200, "width": 96, "height": 96},
            {"id": "review", "type": "task", "label": "Review", "x": 320, "y": 200, "width": 140, "height": 72},
            {"id": "publish", "type": "task", "label": "Publish", "x": 540, "y": 200, "width": 140, "height": 72},
            {"id": "end", "type": "end", "label": "End", "x": 760, "y": 200, "width": 96, "height": 96},
        ],
        "edges": [
            {"id": "e_start_review", "source": "start", "target": "review", "label": ""},
            {"id": "e_review_publish", "source": "review", "target": "publish", "label": ""},
            {"id": "e_publish_end", "source": "publish", "target": "end", "label": ""},
        ],
    }


def insertion_spec(*, label: str, anchor_text: str, relation: str) -> dict:
    return {
        "status": "resolved",
        "semantic_goals": [
            {
                "id": "g1",
                "goal_category": "composite_goal",
                "text": f"insert {label} {relation} {anchor_text}",
                "objects": [
                    {"id": "new_node", "type": "new_entity", "label": label},
                    {"id": "anchor", "type": "existing_reference", "text": anchor_text},
                ],
                "object_desired_states": [{"object": "new_node", "state": "exists"}],
                "relation_intents": [{"source": "new_node", "relation": relation, "target": "anchor"}],
            }
        ],
        "clarification": {"needed": False, "question": "", "reason": ""},
    }


def rect(node: dict) -> tuple[float, float, float, float]:
    width = float(node.get("width", 140))
    height = float(node.get("height", 72))
    x = float(node.get("x", 0))
    y = float(node.get("y", 0))
    return x - width / 2, y - height / 2, x + width / 2, y + height / 2


def overlaps(left: dict, right: dict) -> bool:
    a = rect(left)
    b = rect(right)
    return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])


def run_insert(monkeypatch, spec: dict, instruction: str) -> tuple[dict, dict]:
    server.SESSIONS.clear()
    monkeypatch.setenv("AI_EDIT_PIPELINE", "semantic")
    mock_understanding(monkeypatch, spec)
    session_id = create_session(serial_graph())
    response = server.run_edit(
        {
            "session_id": session_id,
            "instruction": instruction,
            "selection": {"node_ids": [], "edge_ids": []},
        }
    )
    assert response["status"] == "success"
    assert_readable_response(response)
    added = response["diff"]["added_nodes"]
    assert len(added) == 1
    created = next(node for node in response["graph"]["nodes"] if node["id"] == added[0]["id"])
    return response, created


def assert_created_node_has_visible_non_overlapping_layout(response: dict, created: dict) -> None:
    assert created.get("x") != 0
    assert created.get("y") != 0
    for node in response["graph"]["nodes"]:
        if node["id"] != created["id"]:
            assert not overlaps(created, node)


def test_insert_before_anchor_assigns_visible_non_overlapping_layout(monkeypatch) -> None:
    response, created = run_insert(
        monkeypatch,
        insertion_spec(label="Quality Check", anchor_text="Publish", relation="before"),
        "add quality check before publish",
    )

    assert_created_node_has_visible_non_overlapping_layout(response, created)


def test_insert_after_anchor_assigns_visible_non_overlapping_layout(monkeypatch) -> None:
    response, created = run_insert(
        monkeypatch,
        insertion_spec(label="Archive", anchor_text="Publish", relation="after"),
        "add archive after publish",
    )

    assert_created_node_has_visible_non_overlapping_layout(response, created)


def test_insert_before_end_does_not_fall_back_to_origin(monkeypatch) -> None:
    response, created = run_insert(
        monkeypatch,
        insertion_spec(label="Exception Handling", anchor_text="End", relation="before"),
        "add exception handling before end",
    )

    assert_created_node_has_visible_non_overlapping_layout(response, created)
