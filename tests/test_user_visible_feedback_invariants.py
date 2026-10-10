from __future__ import annotations

from demo.server import graph_diff
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
            {"id": "e_b_c", "source": "b", "target": "c", "label": ""},
        ],
    }


def visible_strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, list):
        for item in value:
            yield from visible_strings(item)
    elif isinstance(value, dict):
        for key, item in value.items():
            if key in {"raw_partial_execution", "id", "location"}:
                continue
            yield from visible_strings(item)


def test_success_feedback_visible_fields_do_not_leak_internal_contract_terms() -> None:
    before = graph()
    after = {
        "nodes": [*before["nodes"], {"id": "new_check", "type": "task", "label": "Check"}],
        "edges": [
            {"id": "e_a_b", "source": "a", "target": "new_check", "label": ""},
            {"id": "e_b_c", "source": "b", "target": "c", "label": ""},
            {"id": "e_new_check_b", "source": "new_check", "target": "b", "label": ""},
        ],
    }
    feedback = build_user_feedback(
        "success",
        {
            "partial_execution": {
                "subgoals": [
                    {"id": "g1:state:0", "description": "o1 property_equals", "status": "executed"},
                    {"id": "g1:relation:0", "description": "o1 route_to o2", "status": "executed"},
                ],
                "final_status": "success",
            },
            "trace": {"graph_diff": graph_diff(before, after)},
        },
        after,
    )

    assert "raw_partial_execution" not in feedback
    for text in visible_strings(feedback):
        assert_no_internal_visible_text(text)


def test_clarification_feedback_visible_fields_do_not_leak_internal_contract_terms() -> None:
    feedback = build_user_feedback(
        "needs_clarification",
        {
            "validation": {
                "issues": [
                    {
                        "kind": "ambiguous_edge_reference",
                        "reference": "this branch",
                        "candidates": ["e_a_b", "e_b_c"],
                        "message": "multiple graph edges match reference",
                    }
                ]
            }
        },
        graph(),
    )

    for text in visible_strings(feedback):
        assert_no_internal_visible_text(text)


def test_partial_success_unknown_technical_reason_is_not_visible() -> None:
    feedback = build_user_feedback(
        "partial_success",
        {
            "partial_execution": {
                "subgoals": [
                    {"id": "g1:state:0", "description": "o1 exists", "status": "executed"},
                    {
                        "id": "g1:relation:0",
                        "description": "o1 connected_to o2",
                        "status": "unsupported",
                        "reason": "relation intent was not lowered",
                    },
                ],
                "final_status": "partial_success",
            },
            "trace": {"graph_diff": {"added_nodes": [{"id": "x", "label": "Check"}]}},
        },
        graph(),
    )

    assert feedback["incomplete"]
    assert feedback["incomplete"][0]["reason"] != "relation intent was not lowered"
    for text in visible_strings(feedback):
        assert_no_internal_visible_text(text)


def test_semantic_contract_violation_message_is_debug_only() -> None:
    feedback = build_user_feedback(
        "unsupported",
        {
            "validation": {
                "issues": [
                    {
                        "kind": "semantic_goal_contract_violation",
                        "message": "object references must preserve phrases from the user instruction; Semantic Understanding must not pre-resolve graph objects",
                    }
                ]
            }
        },
        graph(),
        error="object references must preserve phrases from the user instruction; Semantic Understanding must not pre-resolve graph objects",
    )

    assert feedback["message"]
    for text in visible_strings(feedback):
        assert_no_internal_visible_text(text)


def test_execution_exception_text_is_not_visible() -> None:
    feedback = build_user_feedback(
        "execution_error",
        {"validation": {"issues": [{"kind": "execution_error", "message": "ValueError: planner exploded"}]}},
        graph(),
        error="ValueError: planner exploded at workflow_ir.properties[0]",
    )

    for text in visible_strings(feedback):
        assert_no_internal_visible_text(text)


def test_unknown_issue_kind_uses_safe_fallback() -> None:
    feedback = build_user_feedback(
        "needs_clarification",
        {
            "validation": {
                "issues": [
                    {
                        "kind": "brand_new_internal_issue",
                        "message": "{'workflow_ir': {'properties': [{'node_id': 'n1'}]}}",
                        "reason": "lowering diagnostic did not produce executable binding",
                    }
                ]
            }
        },
        graph(),
    )

    assert feedback["clarification"]["question"]
    for text in visible_strings(feedback):
        assert_no_internal_visible_text(text)
