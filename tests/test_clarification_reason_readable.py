from __future__ import annotations

from demo_api_helpers import assert_no_internal_visible_text
from src.product_feedback import build_user_feedback


def graph() -> dict:
    return {
        "nodes": [
            {"id": "a", "type": "task", "label": "A"},
            {"id": "b", "type": "task", "label": "B"},
            {"id": "c", "type": "task", "label": "C"},
        ],
        "edges": [
            {"id": "e_a_b", "source": "a", "target": "b", "label": ""},
            {"id": "e_a_c", "source": "a", "target": "c", "label": ""},
        ],
    }


def assert_feedback_readable(feedback: dict) -> None:
    assert_no_internal_visible_text(feedback["message"])
    assert_no_internal_visible_text(feedback["clarification"]["question"])
    for group in feedback["clarification"]["candidates"]:
        assert_no_internal_visible_text(group.get("reason", ""))
        for candidate in group["candidates"]:
            assert_no_internal_visible_text(candidate.get("label", ""))
            assert_no_internal_visible_text(candidate.get("reason", ""))


def test_multiple_graph_edges_reason_is_readable() -> None:
    feedback = build_user_feedback(
        "needs_clarification",
        {"validation": {"issues": [{"kind": "ambiguous_edge_reference", "reference": "branch", "candidates": ["e_a_b", "e_a_c"], "message": "multiple graph edges match reference"}]}},
        graph(),
    )

    assert "\u54ea\u4e00\u6761\u5206\u652f" in feedback["clarification"]["candidates"][0]["reason"]
    assert_feedback_readable(feedback)


def test_unique_binding_reason_is_readable_in_incomplete() -> None:
    feedback = build_user_feedback(
        "needs_clarification",
        {
            "partial_execution": {
                "subgoals": [
                    {
                        "id": "g1:relation:0",
                        "description": "o1 route_to o2",
                        "status": "ambiguous",
                        "reason": "grounding did not produce a unique executable binding",
                    }
                ]
            }
        },
        graph(),
    )

    assert "\u552f\u4e00\u786e\u5b9a" in feedback["incomplete"][0]["reason"]
    assert_no_internal_visible_text(feedback["message"])


def test_supported_semantic_relation_lowering_reason_is_readable() -> None:
    feedback = build_user_feedback(
        "unsupported",
        {
            "partial_execution": {
                "subgoals": [
                    {
                        "id": "g1:relation:0",
                        "description": "o1 route_to o2",
                        "status": "unsupported",
                        "reason": "supported semantic relation could not be lowered",
                    }
                ]
            }
        },
        graph(),
    )

    assert "\u8fdb\u4e00\u6b65\u786e\u8ba4" in feedback["incomplete"][0]["reason"]
    assert_no_internal_visible_text(feedback["message"])


def test_missing_selection_for_this_branch_is_readable() -> None:
    feedback = build_user_feedback(
        "needs_clarification",
        {"validation": {"issues": [{"kind": "missing_edge_reference", "reference": "\u8fd9\u4e2a\u5206\u652f", "candidates": [], "message": "no graph edge matches reference"}]}},
        graph(),
    )

    assert "\u8fd9\u4e2a\u5206\u652f" in feedback["message"]
    assert_feedback_readable(feedback)


def test_missing_edge_raw_reason_is_mapped_without_candidates() -> None:
    feedback = build_user_feedback(
        "needs_clarification",
        {"validation": {"issues": [{"kind": "missing_edge_reference", "reference": "this branch", "candidates": [], "message": "no graph edge matches reference"}]}},
        graph(),
        clarification_message="no graph edge matches reference",
    )

    assert "no graph edge matches reference" not in feedback["message"]
    assert "\u5206\u652f" in feedback["clarification"]["question"]
    assert feedback["status"] == "needs_clarification"
    assert_feedback_readable(feedback)


def test_ambiguous_issue_with_no_real_candidates_does_not_render_empty_candidates() -> None:
    feedback = build_user_feedback(
        "needs_clarification",
        {"validation": {"issues": [{"kind": "ambiguous_edge_reference", "reference": "this branch", "candidates": ["missing_edge"], "message": "multiple graph edges match reference"}]}},
        graph(),
        clarification_message="multiple graph edges match reference",
    )

    assert feedback["clarification"]["candidates"] == []
    assert "\u5019\u9009" not in feedback["message"]
    assert "\u4ece\u5019\u9009" not in feedback["clarification"]["question"]
    assert_feedback_readable(feedback)


def test_unknown_technical_reason_uses_safe_fallback_not_raw_text() -> None:
    raw_reason = "executor failed: workflow_ir.edge_properties[0] could not bind e_a_b"
    feedback = build_user_feedback(
        "unsupported",
        {
            "partial_execution": {
                "subgoals": [
                    {
                        "id": "g1:relation:0",
                        "description": "o1 route_to o2",
                        "status": "unsupported",
                        "reason": raw_reason,
                    }
                ]
            }
        },
        graph(),
        error=raw_reason,
    )

    assert raw_reason not in feedback["message"]
    assert raw_reason not in feedback["incomplete"][0]["reason"]
    assert feedback["incomplete"]
    assert_no_internal_visible_text(feedback["message"])
    assert_no_internal_visible_text(feedback["incomplete"][0]["reason"])


def test_execution_failed_reason_is_user_readable() -> None:
    feedback = build_user_feedback(
        "execution_error",
        None,
        graph(),
        error="execution failed: deterministic executor raised RuntimeError",
    )

    assert "\u6267\u884c" in feedback["message"]
    assert_no_internal_visible_text(feedback["message"])


def test_abstract_safe_stop_reason_is_readable() -> None:
    feedback = build_user_feedback(
        "unsupported",
        {
            "partial_execution": {
                "subgoals": [
                    {
                        "id": "g1:relation:0",
                        "description": "abstract optimization",
                        "status": "unsupported",
                        "reason": "unsupported abstract request",
                    }
                ]
            }
        },
        graph(),
    )

    assert_no_internal_visible_text(feedback["message"])
