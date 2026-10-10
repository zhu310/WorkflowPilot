from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

from src import semantic_shadow


def mock_understanding(monkeypatch, spec: dict) -> None:
    monkeypatch.setattr(
        semantic_shadow,
        "_complete_json",
        lambda _system, _payload: SimpleNamespace(raw_response="", parsed_json=deepcopy(spec), parse_error=None, normalized_from_fence=False),
    )


def graph() -> dict:
    return {
        "nodes": [
            {"id": "start", "type": "start", "label": "Start", "x": 0, "y": 0, "width": 160, "height": 80},
            {"id": "review", "type": "task", "label": "Review", "x": 220, "y": 0, "width": 160, "height": 80},
            {"id": "end", "type": "end", "label": "End", "x": 440, "y": 0, "width": 160, "height": 80},
        ],
        "edges": [
            {"id": "e_start_review", "source": "start", "target": "review", "label": ""},
            {"id": "e_review_end", "source": "review", "target": "end", "label": ""},
        ],
    }


def rename_spec() -> dict:
    return {
        "status": "resolved",
        "semantic_goals": [
            {
                "id": "g1",
                "goal_category": "object_state_goal",
                "objects": [{"id": "target", "type": "existing_reference", "text": "Review"}],
                "object_desired_states": [
                    {"object": "target", "state": "property_equals", "property": "display_label", "value": "Manual Review"}
                ],
            }
        ],
        "clarification": {"needed": False, "question": "", "reason": ""},
    }


def production_facing_projection(trace: dict) -> dict:
    return {
        "stage": trace.get("stage"),
        "failure_stage": trace.get("failure_stage"),
        "validation": trace.get("validation"),
        "lowering_result": trace.get("lowering_result"),
        "compiler_input": trace.get("compiler_input"),
        "final_shadow_graph": trace.get("final_shadow_graph"),
        "graph_diff": trace.get("graph_diff"),
        "partial_execution": trace.get("partial_execution"),
    }


def test_canonical_shadow_on_off_does_not_change_production_result(monkeypatch) -> None:
    mock_understanding(monkeypatch, rename_spec())
    monkeypatch.delenv("SEMANTIC_CANONICAL_SHADOW", raising=False)
    off = semantic_shadow.run_semantic_shadow(graph(), "Rename Review to Manual Review")

    mock_understanding(monkeypatch, rename_spec())
    monkeypatch.setenv("SEMANTIC_CANONICAL_SHADOW", "1")
    on = semantic_shadow.run_semantic_shadow(graph(), "Rename Review to Manual Review")

    assert "canonical_shadow" not in off
    assert on["canonical_shadow"]["ok"] is True
    assert on["canonical_shadow"]["goal_state"]["desired_facts"][0]["predicate"] == "property.equals"
    assert production_facing_projection(on) == production_facing_projection(off)


def test_canonical_shadow_failure_is_fail_open(monkeypatch) -> None:
    mock_understanding(monkeypatch, rename_spec())
    monkeypatch.setenv("SEMANTIC_CANONICAL_SHADOW", "1")

    def boom(_normalized_goal, *, instruction=None):
        raise RuntimeError("adapter failed")

    monkeypatch.setattr(semantic_shadow, "build_canonical_shadow_artifact", boom)

    trace = semantic_shadow.run_semantic_shadow(graph(), "Rename Review to Manual Review")

    assert trace["stage"] == "success"
    assert trace["canonical_shadow"]["ok"] is False
    assert trace["canonical_shadow"]["failure_stage"] == "canonical_shadow"
    review = next(node for node in trace["final_shadow_graph"]["nodes"] if node["id"] == "review")
    assert review["label"] == "Manual Review"
