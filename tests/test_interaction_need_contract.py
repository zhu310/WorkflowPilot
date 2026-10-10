from __future__ import annotations

from src.product_feedback import build_user_feedback


def graph() -> dict:
    return {
        "nodes": [
            {"id": "approve", "type": "decision", "label": "Approve"},
            {"id": "review", "type": "task", "label": "Review"},
            {"id": "archive", "type": "task", "label": "Archive"},
            {"id": "finish", "type": "end", "label": "Finish"},
        ],
        "edges": [
            {"id": "edge_yes", "source": "approve", "target": "review", "branch_label": "yes", "label": "yes"},
            {"id": "edge_no", "source": "approve", "target": "archive", "branch_label": "no", "label": "no"},
            {"id": "edge_timeout", "source": "approve", "target": "finish", "branch_label": "timeout", "label": "timeout"},
        ],
    }


def test_missing_subject_slot_preserves_known_target() -> None:
    feedback = build_user_feedback(
        "needs_clarification",
        {
            "validation": {"issues": [{"kind": "missing_route_edge_binding"}]},
            "semantic_goal": {
                "normalized": {
                    "semantic_goals_v2": [
                        {
                            "objects": [{"id": "target", "type": "existing_reference", "reference": "Finish"}],
                            "relation_intents": [{"source": "route", "relation": "route_to", "target": "target"}],
                        }
                    ]
                }
            },
        },
        graph(),
    )

    assert feedback["interaction"]["type"] == "provide_missing_slot"
    assert feedback["interaction"]["slot"] == "subject_edge"
    assert feedback["interaction"]["known_target"] == "Finish"
    assert "Finish" in feedback["message"]
    assert "未完成" not in feedback["message"]


def test_missing_target_slot_preserves_known_selected_edge() -> None:
    feedback = build_user_feedback(
        "needs_clarification",
        {
            "validation": {"issues": [{"kind": "missing_route_target_binding"}]},
            "trace": {"selection": {"node_ids": [], "edge_ids": ["edge_yes"]}},
            "semantic_goal": {
                "normalized": {
                    "semantic_goals_v2": [
                        {
                            "objects": [{"id": "route", "type": "edge_reference", "reference": "selected edge"}],
                            "relation_intents": [{"source": "route", "relation": "route_to", "target": "target"}],
                        }
                    ]
                }
            },
        },
        graph(),
    )

    assert feedback["interaction"]["type"] == "provide_missing_slot"
    assert feedback["interaction"]["slot"] == "route_target"
    assert "Approve -> Review" in feedback["interaction"]["known_subject"]
    assert "哪" in feedback["message"]
    assert "不确定要修改的对象" not in feedback["message"]


def test_ambiguous_interaction_is_one_prompt_plus_real_candidates() -> None:
    feedback = build_user_feedback(
        "needs_clarification",
        {
            "validation": {
                "issues": [
                    {
                        "kind": "ambiguous_edge_reference",
                        "reference": "that connection",
                        "candidates": ["edge_yes", "edge_no"],
                        "reason": "multiple graph edges match reference",
                    }
                ]
            }
        },
        graph(),
    )

    assert feedback["interaction"]["type"] == "choose_candidate"
    assert feedback["message"].count("候选") == 0
    assert "未完成" not in feedback["message"]
    assert "edge_yes" not in feedback["message"]
    assert "edge_no" not in feedback["message"]
    assert "1." in feedback["message"]
    assert "2." in feedback["message"]


def test_confirmation_interaction_does_not_use_failure_copy() -> None:
    feedback = build_user_feedback(
        "needs_clarification",
        {
            "validation": {
                "issues": [
                    {
                        "kind": "route_same_source_target_confirmation",
                        "source": "approve",
                        "target": "review",
                        "relation_type": "yes",
                        "conflicting_branch_label": "no",
                    }
                ]
            }
        },
        graph(),
    )

    assert feedback["interaction"]["type"] == "confirm_change"
    assert "Approve" in feedback["message"]
    assert "Review" in feedback["message"]
    assert "未完成" not in feedback["message"]
    assert "不能安全执行" not in feedback["message"]


def test_unknown_failure_does_not_blame_user_ambiguity() -> None:
    feedback = build_user_feedback("execution_error", None, graph(), error="RuntimeError: executor exploded")

    assert feedback["interaction"]["type"] == "retry_after_internal_failure"
    assert "没有成功" in feedback["message"]
    assert "不确定" not in feedback["message"]
    assert "更明确的修改对象" not in feedback["message"]
