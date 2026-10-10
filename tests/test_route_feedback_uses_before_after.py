from __future__ import annotations

from demo.server import graph_diff
from demo_api_helpers import assert_no_internal_visible_text
from src.product_feedback import build_user_feedback


def graph_before(label: str = "") -> dict:
    return {
        "nodes": [
            {"id": "decision", "type": "decision", "label": "Decision"},
            {"id": "report", "type": "task", "label": "Generate Report"},
            {"id": "publish", "type": "task", "label": "Publish Result"},
            {"id": "human", "type": "task", "label": "Human Review"},
        ],
        "edges": [{"id": "e_route", "source": "decision", "target": "report", "label": label}],
    }


def graph_after(label: str = "") -> dict:
    graph = graph_before(label)
    graph["edges"] = [{**graph["edges"][0], "target": "human"}]
    return graph


def feedback_message(label: str = "") -> str:
    after = graph_after(label)
    feedback = build_user_feedback(
        "success",
        {
            "partial_execution": {
                "subgoals": [{"id": "g1:relation:0", "description": "route route_to human", "status": "executed"}],
                "final_status": "success",
            },
            "trace": {"graph_diff": graph_diff(graph_before(label), after)},
        },
        after,
    )
    return feedback["message"]


def test_unlabeled_edge_route_feedback_uses_old_and_new_target() -> None:
    message = feedback_message("")

    assert "Generate Report" in message
    assert "Human Review" in message
    assert "Decision -> Generate Report" in message
    assert "Decision -> Human Review" not in message
    assert "\u8fde\u7ebf" in message
    assert_no_internal_visible_text(message)


def test_labeled_edge_route_feedback_uses_branch_label_and_new_target() -> None:
    message = feedback_message("success")

    assert "success" in message
    assert "\u5206\u652f" in message
    assert "Human Review" in message
    assert "route_to" not in message
    assert "branch:" not in message
    assert_no_internal_visible_text(message)


def test_same_target_confirmation_feedback_does_not_leak_route_to() -> None:
    feedback = build_user_feedback(
        "needs_clarification",
        {
            "trace": {
                "grounded_goal": {
                    "grounding": {
                        "issues": [
                            {
                                "kind": "route_same_source_target_confirmation",
                                "source": "decision",
                                "target": "human",
                                "relation_type": "success",
                                "conflicting_branch_label": "failure",
                            }
                        ]
                    }
                }
            }
        },
        graph_after("success"),
    )
    message = feedback["message"]

    assert "\u9700\u8981\u4f60\u786e\u8ba4" in message
    assert "route_to" not in message
    assert_no_internal_visible_text(message)
