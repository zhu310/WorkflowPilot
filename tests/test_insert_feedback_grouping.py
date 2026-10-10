from __future__ import annotations

from demo.server import graph_diff
from demo_api_helpers import assert_no_internal_visible_text
from src.product_feedback import build_user_feedback


def before_graph() -> dict:
    return {
        "nodes": [
            {"id": "report", "type": "task", "label": "Generate Report", "x": 100, "y": 100},
            {"id": "publish", "type": "task", "label": "Publish Result", "x": 300, "y": 100},
            {"id": "end", "type": "end", "label": "End", "x": 500, "y": 100},
        ],
        "edges": [
            {"id": "e_report_publish", "source": "report", "target": "publish", "label": ""},
            {"id": "e_publish_end", "source": "publish", "target": "end", "label": ""},
        ],
    }


def after_insert(*, edge_id: str, new_id: str, label: str) -> dict:
    graph = before_graph()
    old_edge = next(edge for edge in graph["edges"] if edge["id"] == edge_id)
    old_target = old_edge["target"]
    nodes = [*graph["nodes"], {"id": new_id, "type": "task", "label": label, "x": 400, "y": 100}]
    edges = []
    for edge in graph["edges"]:
        if edge["id"] == edge_id:
            edges.append({**edge, "target": new_id})
        else:
            edges.append(edge)
    edges.append({"id": f"e_{new_id}_{old_target}", "source": new_id, "target": old_target, "label": ""})
    return {"nodes": nodes, "edges": edges}


def feedback_message_for_insert(after: dict) -> str:
    diff = graph_diff(before_graph(), after)
    feedback = build_user_feedback(
        "success",
        {
            "partial_execution": {
                "subgoals": [
                    {"id": "g1:state:0", "description": "o1 property_equals", "status": "executed"},
                    {"id": "g1:relation:0", "description": "o1 before o2", "status": "executed"},
                ],
                "final_status": "success",
            },
            "trace": {"graph_diff": diff},
        },
        after,
    )
    return feedback["message"]


def assert_grouped_insert_feedback(message: str, inserted_label: str, source: str, target: str) -> None:
    assert inserted_label in message
    assert source in message
    assert target in message
    assert message.count(inserted_label) == 1
    assert "changed" not in message.lower()
    assert "branch" not in message.lower()
    assert_no_internal_visible_text(message)


def test_insert_before_end_feedback_is_grouped_as_one_user_action() -> None:
    message = feedback_message_for_insert(
        after_insert(edge_id="e_publish_end", new_id="quality_gate", label="Quality Gate")
    )

    assert_grouped_insert_feedback(message, "Quality Gate", "Publish Result", "End")


def test_insert_before_report_feedback_is_grouped_as_one_user_action() -> None:
    message = feedback_message_for_insert(
        after_insert(edge_id="e_report_publish", new_id="quality", label="Quality Check")
    )

    assert_grouped_insert_feedback(message, "Quality Check", "Generate Report", "Publish Result")


def test_insert_after_publish_feedback_is_grouped_as_one_user_action() -> None:
    message = feedback_message_for_insert(
        after_insert(edge_id="e_publish_end", new_id="archive", label="Archive")
    )

    assert_grouped_insert_feedback(message, "Archive", "Publish Result", "End")
