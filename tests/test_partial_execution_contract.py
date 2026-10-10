from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

from src import semantic_shadow


def graph() -> dict:
    return {
        "nodes": [
            {"id": "start", "type": "start", "label": "Start", "x": 0, "y": 80, "width": 120, "height": 60},
            {"id": "approval", "type": "task", "label": "\u5ba1\u6279\u8282\u70b9", "x": 180, "y": 80, "width": 140, "height": 72},
            {"id": "risk", "type": "task", "label": "\u98ce\u9669\u63a7\u5236", "x": 360, "y": 80, "width": 140, "height": 72},
            {"id": "manual", "type": "task", "label": "\u4eba\u5de5\u5904\u7406", "x": 540, "y": 120, "width": 140, "height": 72},
            {"id": "end", "type": "end", "label": "End", "x": 720, "y": 80, "width": 120, "height": 60},
        ],
        "edges": [
            {"id": "e_start_approval", "source": "start", "target": "approval", "label": ""},
            {"id": "e_approval_risk", "source": "approval", "target": "risk", "label": ""},
            {"id": "e_risk_end", "source": "risk", "target": "end", "label": "success", "branch_label": "\u6210\u529f"},
            {"id": "e_risk_manual_failure", "source": "risk", "target": "manual", "label": "failure", "branch_label": "\u5931\u8d25"},
        ],
    }


def run_with_goal(monkeypatch, semantic_goal: dict, before: dict | None = None):
    monkeypatch.setattr(
        semantic_shadow,
        "_complete_json",
        lambda _system, _payload: SimpleNamespace(raw_response="", parsed_json=deepcopy(semantic_goal), parse_error=None, normalized_from_fence=False),
    )
    instruction = "\u589e\u52a0\u4eba\u5de5\u5ba1\u6838\u8282\u70b9\uff0c\u628a\u5ba1\u6279\u8282\u70b9\u653e\u5230\u98ce\u9669\u63a7\u5236\u540e\u9762\uff0c\u628a\u5931\u8d25\u5206\u652f\u6539\u6210\u4eba\u5de5\u5904\u7406\uff0c\u5e76\u8fde\u63a5\u5230 End"
    return semantic_shadow.run_semantic_transaction(before or graph(), instruction)


def v2_goal(goals: list[dict]) -> dict:
    return {
        "status": "resolved",
        "semantic_goals": goals,
        "clarification": {"needed": False, "question": "", "reason": ""},
    }


def partial(result):
    return result.specification["partial_execution"]


def statuses(result) -> dict[str, str]:
    return {row["id"]: row["status"] for row in partial(result)["subgoals"]}


def test_route_to_new_target_executes_when_exists_dependency_is_present(monkeypatch) -> None:
    semantic_goal = v2_goal(
        [
            {
                "id": "g1",
                "goal_category": "composite_goal",
                "objects": [
                    {"id": "review", "type": "new_entity", "label": "\u4eba\u5de5\u5ba1\u6838", "role": "object"},
                    {"id": "failure", "type": "edge_reference", "text": "\u5931\u8d25\u5206\u652f", "role": "route"},
                ],
                "object_desired_states": [{"object": "review", "state": "exists"}],
                "relation_intents": [{"source": "failure", "relation": "route_to", "target": "review"}],
            }
        ]
    )

    result = run_with_goal(monkeypatch, semantic_goal)

    assert result.status == "success"
    assert result.final_graph is not None
    assert statuses(result)["g1:state:0"] == "executed"
    assert statuses(result)["g1:relation:0"] == "executed"
    assert any(node.get("label") == "\u4eba\u5de5\u5ba1\u6838" for node in result.final_graph["nodes"])


def test_independent_supported_goal_can_partial_execute(monkeypatch) -> None:
    semantic_goal = v2_goal(
        [
            {
                "id": "g1",
                "goal_category": "composite_goal",
                "objects": [
                    {"id": "review", "type": "new_entity", "label": "\u4eba\u5de5\u5ba1\u6838", "role": "object"},
                    {"id": "end", "type": "existing_reference", "text": "End", "role": "object"},
                ],
                "object_desired_states": [{"object": "review", "state": "exists"}],
                "relation_intents": [{"source": "review", "relation": "connected_to", "target": "end"}],
            }
        ]
    )

    result = run_with_goal(monkeypatch, semantic_goal)

    assert result.status == "partial_success"
    assert result.final_graph is not None
    assert statuses(result)["g1:state:0"] == "executed"
    assert statuses(result)["g1:relation:0"] == "unsupported"
    assert "Not completed" in (result.clarification_message or "")


def test_all_supported_subgoals_return_success(monkeypatch) -> None:
    semantic_goal = v2_goal(
        [
            {
                "id": "g1",
                "goal_category": "composite_goal",
                "objects": [
                    {"id": "approval", "type": "existing_reference", "text": "\u5ba1\u6279\u8282\u70b9", "role": "object"},
                    {"id": "risk", "type": "existing_reference", "text": "\u98ce\u9669\u63a7\u5236", "role": "anchor"},
                    {"id": "failure", "type": "edge_reference", "text": "\u5931\u8d25\u5206\u652f", "role": "branch"},
                ],
                "object_desired_states": [
                    {"object": "failure", "state": "property_equals", "property": "display_label", "value": "\u4eba\u5de5\u5904\u7406"}
                ],
                "relation_intents": [{"source": "approval", "relation": "after", "target": "risk"}],
            }
        ]
    )

    result = run_with_goal(monkeypatch, semantic_goal)

    assert result.status == "success"
    assert set(statuses(result).values()) == {"executed"}


def test_parallel_with_executes_for_simple_direct_serial_chain(monkeypatch) -> None:
    semantic_goal = v2_goal(
        [
            {
                "id": "g1",
                "goal_category": "relation_goal",
                "objects": [
                    {"id": "risk", "type": "existing_reference", "text": "\u98ce\u9669\u63a7\u5236", "role": "object"},
                    {"id": "approval", "type": "existing_reference", "text": "\u5ba1\u6279\u8282\u70b9", "role": "object"},
                ],
                "relation_intents": [{"source": "risk", "relation": "parallel_with", "target": "approval"}],
            }
        ]
    )

    result = run_with_goal(monkeypatch, semantic_goal)

    assert result.status == "success"
    assert result.final_graph is not None
    assert statuses(result)["g1:relation:0"] == "executed"


def test_ambiguous_grounding_does_not_execute(monkeypatch) -> None:
    before = graph()
    before["edges"].append({"id": "e_other_failure", "source": "approval", "target": "manual", "label": "failure", "branch_label": "\u5931\u8d25"})
    semantic_goal = v2_goal(
        [
            {
                "id": "g1",
                "goal_category": "object_state_goal",
                "objects": [{"id": "failure", "type": "edge_reference", "text": "\u5931\u8d25\u5206\u652f", "role": "branch"}],
                "object_desired_states": [
                    {"object": "failure", "state": "property_equals", "property": "display_label", "value": "\u4eba\u5de5\u5904\u7406"}
                ],
            }
        ]
    )

    result = run_with_goal(monkeypatch, semantic_goal, before)

    assert result.status == "needs_clarification"
    assert result.final_graph is None
    assert statuses(result)["g1:state:0"] == "ambiguous"
