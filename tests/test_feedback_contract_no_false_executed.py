from __future__ import annotations

from src.product_feedback import build_user_feedback


def spec_with_partial(graph_diff: dict | None, final_status: str = "partial_success") -> dict:
    return {
        "partial_execution": {
            "subgoals": [
                {"id": "g1:relation:0", "description": "route exception to manual", "status": "executed"},
                {"id": "g1:state:0", "description": "remove stale edge", "status": "unsupported", "reason": "edge absence is unsupported"},
            ],
            "executed_count": 1,
            "incomplete_count": 1,
            "final_status": final_status,
        },
        "trace": {"graph_diff": graph_diff} if graph_diff is not None else {},
    }


def test_unsupported_with_no_graph_diff_does_not_show_executed_subgoals() -> None:
    feedback = build_user_feedback(
        "unsupported",
        spec_with_partial({"added_nodes": [], "removed_nodes": [], "changed_nodes": [], "added_edges": [], "removed_edges": [], "changed_edges": []}, "unsupported"),
        {"nodes": [], "edges": []},
    )

    assert feedback["executed"] == []
    assert feedback["incomplete"]
    assert "route exception to manual" not in [row["summary"] for row in feedback["executed"]]


def test_partial_success_with_real_graph_diff_shows_executed_and_incomplete() -> None:
    feedback = build_user_feedback(
        "partial_success",
        spec_with_partial({"added_nodes": [{"id": "new_1"}], "removed_nodes": [], "changed_nodes": [], "added_edges": [], "removed_edges": [], "changed_edges": []}),
        {"nodes": [{"id": "new_1"}], "edges": []},
    )

    assert feedback["executed"]
    assert feedback["executed"][0]["summary"] != "route exception to manual"
    assert feedback["incomplete"][0]["summary"] == "remove stale edge"
    assert feedback["incomplete"][0]["status"] == "unsupported"
    assert feedback["incomplete"][0]["reason"] != "edge absence is unsupported"


def test_needs_clarification_with_no_graph_diff_does_not_show_executed_subgoals() -> None:
    feedback = build_user_feedback(
        "needs_clarification",
        spec_with_partial({"added_nodes": [], "removed_nodes": [], "changed_nodes": [], "added_edges": [], "removed_edges": [], "changed_edges": []}, "needs_clarification"),
        {"nodes": [], "edges": []},
        clarification_message="Please choose a target.",
    )

    assert feedback["executed"] == []
    assert feedback["clarification"]["question"] == "Please choose a target."
