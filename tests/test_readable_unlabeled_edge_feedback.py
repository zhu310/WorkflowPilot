from __future__ import annotations

from demo_api_helpers import assert_no_internal_visible_text
from src.product_feedback import build_user_feedback


def graph() -> dict:
    return {
        "nodes": [
            {"id": "decision", "type": "decision", "label": "Decision"},
            {"id": "report", "type": "task", "label": "Generate Report"},
            {"id": "human", "type": "task", "label": "Human Review"},
            {"id": "publish", "type": "task", "label": "Publish Result"},
            {"id": "end", "type": "end", "label": "End"},
        ],
        "edges": [
            {"id": "e_report_publish", "source": "report", "target": "publish", "label": ""},
            {"id": "e_publish_end", "source": "publish", "target": "end", "label": ""},
            {"id": "e_human_publish", "source": "human", "target": "publish", "label": ""},
            {"id": "e_decision_report", "source": "decision", "target": "report", "label": "success"},
            {"id": "e_decision_human", "source": "decision", "target": "human", "label": "failure"},
        ],
    }


def ambiguous_edge_spec() -> dict:
    return {
        "validation": {
            "issues": [
                {
                    "kind": "ambiguous_edge_reference",
                    "reference": "this branch",
                    "location": "semantic_goals[0].objects[0]",
                    "candidates": [
                        "e_report_publish",
                        "e_publish_end",
                        "e_human_publish",
                        "e_decision_report",
                        "e_decision_human",
                    ],
                    "message": "multiple graph edges match reference",
                }
            ]
        }
    }


def test_unlabeled_edges_use_source_and_target_labels_in_candidates() -> None:
    feedback = build_user_feedback("needs_clarification", ambiguous_edge_spec(), graph())
    rows = feedback["clarification"]["candidates"][0]["candidates"]
    labels = {row["id"]: row["label"] for row in rows}

    assert labels["e_report_publish"] == "\u8fde\u7ebf\uff1aGenerate Report -> Publish Result"
    assert labels["e_publish_end"] == "\u8fde\u7ebf\uff1aPublish Result -> End"
    assert labels["e_human_publish"] == "\u8fde\u7ebf\uff1aHuman Review -> Publish Result"


def test_labeled_edges_sort_before_unlabeled_edges_and_keep_labels() -> None:
    feedback = build_user_feedback("needs_clarification", ambiguous_edge_spec(), graph())
    rows = feedback["clarification"]["candidates"][0]["candidates"]

    assert rows[0]["id"] == "e_decision_human"
    assert rows[1]["id"] == "e_decision_report"
    assert "failure \u5206\u652f\uff1aDecision -> Human Review" == rows[0]["label"]
    assert "success \u5206\u652f\uff1aDecision -> Generate Report" == rows[1]["label"]


def test_ambiguous_edge_message_hides_internal_ids_and_raw_reason() -> None:
    feedback = build_user_feedback("needs_clarification", ambiguous_edge_spec(), graph())
    message = feedback["message"]

    assert "multiple graph edges match reference" not in message
    for edge_id in ("e_report_publish", "e_publish_end", "e_human_publish"):
        assert edge_id not in message
    assert "\u8fde\u7ebf\uff1aGenerate Report -> Publish Result" in message
    assert_no_internal_visible_text(message)
