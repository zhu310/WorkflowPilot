from types import SimpleNamespace

from src.product_feedback import build_user_feedback, extract_user_candidates
from src import semantic_shadow


def mock_understanding(monkeypatch, spec: dict) -> None:
    monkeypatch.setattr(
        semantic_shadow,
        "_complete_json",
        lambda _system, _payload: SimpleNamespace(raw_response="", parsed_json=spec, parse_error=None, normalized_from_fence=False),
    )


def graph() -> dict:
    return {
        "nodes": [
            {"id": "review_a", "type": "task", "label": "Duplicate Review"},
            {"id": "review_b", "type": "task", "label": "Duplicate Review"},
        ],
        "edges": [],
    }


def test_feedback_exposes_ambiguous_candidates(monkeypatch) -> None:
    mock_understanding(
        monkeypatch,
        {
            "status": "resolved",
            "semantic_goals": [
                {
                    "id": "g1",
                    "goal_category": "object_state_goal",
                    "text": "Delete Duplicate Review",
                    "objects": [{"id": "target", "type": "existing_reference", "text": "Duplicate Review"}],
                    "object_desired_states": [{"object": "target", "state": "absent"}],
                }
            ],
            "clarification": {"needed": False, "question": "", "reason": ""},
        },
    )
    trace = semantic_shadow.run_semantic_shadow(graph(), "Delete Duplicate Review")
    result = semantic_shadow.semantic_trace_to_transaction_result(trace)

    feedback = build_user_feedback(result.status, result.specification, graph(), result.clarification_message, result.error)

    assert result.status == "needs_clarification"
    assert "候选" in feedback["message"]
    assert feedback["clarification"]["candidates"][0]["candidates"][0]["id"] == "review_a"
    assert extract_user_candidates(result.specification, graph())[0]["type"] == "node"


def test_feedback_summarizes_partial_success(monkeypatch) -> None:
    mock_understanding(
        monkeypatch,
        {
            "status": "resolved",
            "semantic_goals": [
                {
                    "id": "g1",
                    "goal_category": "composite_goal",
                    "text": "Add Notice and connect it to End",
                    "objects": [
                        {"id": "notice", "type": "new_entity", "label": "Notice"},
                        {"id": "end_obj", "type": "existing_reference", "text": "End"},
                    ],
                    "object_desired_states": [{"object": "notice", "state": "exists"}],
                    "relation_intents": [{"source": "notice", "relation": "connected_to", "target": "end_obj"}],
                }
            ],
            "clarification": {"needed": False, "question": "", "reason": ""},
        },
    )
    base = {"nodes": [{"id": "end", "type": "end", "label": "End"}], "edges": []}
    trace = semantic_shadow.run_semantic_shadow(base, "Add Notice and connect it to End")
    result = semantic_shadow.semantic_trace_to_transaction_result(trace)

    feedback = build_user_feedback(result.status, result.specification, trace["final_shadow_graph"], result.clarification_message, result.error)

    assert result.status == "partial_success"
    assert feedback["executed"]
    assert feedback["incomplete"]
    assert "已完成" in feedback["message"]
    assert "未完成" in feedback["message"]
