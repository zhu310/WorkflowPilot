from src.product_feedback import build_user_feedback


INTERNAL_TOKENS = ("o1", "o2", "property_equals", "route_to", "grounding", "executable binding")


def graph() -> dict:
    return {
        "nodes": [
            {"id": "decision", "type": "decision", "label": "Decision"},
            {"id": "report", "type": "task", "label": "Report"},
            {"id": "human", "type": "task", "label": "Human Review"},
            {"id": "review", "type": "task", "label": "Review"},
        ],
        "edges": [
            {"id": "e_success", "source": "decision", "target": "report", "label": "success", "branch_label": "Success"},
            {"id": "e_failure", "source": "decision", "target": "human", "label": "failure", "branch_label": "Failure"},
        ],
    }


def assert_no_internal_tokens(message: str) -> None:
    lowered = message.lower()
    for token in INTERNAL_TOKENS:
        assert token not in lowered


def test_feedback_uses_graph_diff_copy_for_node_rename() -> None:
    spec = {
        "partial_execution": {
            "subgoals": [
                {"id": "g1:state:0", "description": "o1 property_equals Manual Review", "status": "executed"}
            ],
            "final_status": "success",
        },
        "trace": {
            "graph_diff": {
                "added_nodes": [],
                "removed_nodes": [],
                "changed_nodes": [{"id": "review", "changes": {"label": {"before": "Review", "after": "Manual Review"}}}],
                "added_edges": [],
                "removed_edges": [],
                "changed_edges": [],
            }
        },
    }

    feedback = build_user_feedback("success", spec, graph())

    assert feedback["executed"]
    assert "Manual Review" in feedback["message"]
    assert_no_internal_tokens(feedback["message"])


def test_feedback_uses_graph_diff_copy_for_route_change() -> None:
    spec = {
        "partial_execution": {
            "subgoals": [{"id": "g1:relation:0", "description": "route route_to human", "status": "executed"}],
            "final_status": "success",
        },
        "trace": {
            "graph_diff": {
                "added_nodes": [],
                "removed_nodes": [],
                "changed_nodes": [],
                "added_edges": [],
                "removed_edges": [],
                "changed_edges": [{"id": "e_success", "changes": {"target": {"before": "report", "after": "human"}}}],
            }
        },
    }

    feedback = build_user_feedback("success", spec, graph())

    assert "Human Review" in feedback["message"]
    assert_no_internal_tokens(feedback["message"])


def test_route_same_target_confirmation_feedback_is_user_readable() -> None:
    spec = {
        "validation": {
            "issues": [
                {
                    "kind": "route_same_source_target_confirmation",
                    "source": "decision",
                    "target": "human",
                    "relation_type": "success",
                    "conflicting_branch_label": "Failure",
                }
            ]
        }
    }

    feedback = build_user_feedback(
        "needs_clarification",
        spec,
        graph(),
        clarification_message="another branch from the same source already routes to the requested target",
    )

    assert "Decision" in feedback["message"]
    assert "Human Review" in feedback["message"]
    assert "Failure" in feedback["message"]
    assert "another branch from the same source" not in feedback["message"]
    assert_no_internal_tokens(feedback["message"])


def test_partial_feedback_rewrites_internal_incomplete_reason() -> None:
    spec = {
        "partial_execution": {
            "subgoals": [
                {"id": "g1:relation:0", "description": "route_to target", "status": "executed"},
                {
                    "id": "g1:state:0",
                    "description": "o2 property_equals branch",
                    "status": "unsupported",
                    "reason": "grounding did not produce a unique executable binding",
                },
            ],
            "final_status": "partial_success",
        },
        "trace": {
            "graph_diff": {
                "added_nodes": [],
                "removed_nodes": [],
                "changed_nodes": [],
                "added_edges": [],
                "removed_edges": [],
                "changed_edges": [{"id": "e_success", "changes": {"target": {"before": "report", "after": "human"}}}],
            }
        },
    }

    feedback = build_user_feedback("partial_success", spec, graph())

    assert feedback["executed"]
    assert feedback["incomplete"]
    assert_no_internal_tokens(feedback["message"])
