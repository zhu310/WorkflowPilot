from __future__ import annotations

from types import SimpleNamespace

from src import semantic_shadow
from src.goal_lowering import lower_grounded_goal_to_workflow_ir
from src.semantic_goal import ground_semantic_goal_spec, validate_semantic_goal_spec


def route_graph() -> dict:
    return {
        "nodes": [
            {"id": "decision", "type": "decision", "label": "判断", "x": 0, "y": 0},
            {"id": "approve", "type": "task", "label": "通过处理", "x": 180, "y": -80},
            {"id": "reject", "type": "task", "label": "拒绝处理", "x": 180, "y": 80},
            {"id": "manual_review", "type": "human_review", "label": "人工审核", "x": 360, "y": 80},
        ],
        "edges": [
            {"id": "e_success", "source": "decision", "target": "approve", "label": "success"},
            {"id": "e_failure", "source": "decision", "target": "reject", "label": "failure"},
        ],
    }


def route_goal() -> dict:
    return {
        "status": "resolved",
        "semantic_goals": [
            {
                "kind": "route",
                "text": "失败情况转人工审核",
                "objects": [
                    {"type": "edge_reference", "text": "失败情况", "role": "route"},
                    {"type": "existing_reference", "text": "人工审核", "role": "target"},
                ],
                "relation": "route_to",
                "explicit_constraints": [],
            }
        ],
        "clarification": {"needed": False, "question": "", "reason": ""},
    }


def edge_set(graph: dict) -> set[tuple[str, str, str]]:
    return {
        (edge["source"], edge["target"], edge.get("label", ""))
        for edge in graph.get("edges", [])
        if isinstance(edge, dict)
    }


def test_route_goal_lowers_to_auxiliary_relations() -> None:
    grounded = ground_semantic_goal_spec(route_goal(), route_graph(), {"node_ids": [], "edge_ids": []})

    lowered = lower_grounded_goal_to_workflow_ir(grounded, route_graph())

    assert grounded["grounding"]["issues"] == []
    bindings = grounded["semantic_goals"][0]["bindings"]
    assert any(binding.get("role") == "route" and binding.get("binding_type") == "edge" and binding.get("id") == "e_failure" for binding in bindings)
    assert any(binding.get("role") == "target" and binding.get("binding_type") == "node" and binding.get("id") == "manual_review" for binding in bindings)
    assert lowered["ok"] is True
    assert lowered["semantic_spec"]["workflow_ir"] == {
        "control_flow_regions": [],
        "auxiliary_relations": [{"source": "decision", "relation_type": "failure", "targets": ["manual_review"]}],
        "absent_entities": [],
        "properties": [],
        "edge_properties": [],
    }


def test_semantic_shadow_route_target_runs_existing_backend(monkeypatch) -> None:
    monkeypatch.setattr(
        semantic_shadow,
        "_complete_json",
        lambda _system, _payload: SimpleNamespace(raw_response="", parsed_json=route_goal(), parse_error=None, normalized_from_fence=False),
    )

    trace = semantic_shadow.run_semantic_shadow(route_graph(), "失败情况转人工审核")

    assert trace["stage"] == "success"
    assert trace["grounded_goal"]["grounding"]["issues"] == []
    assert trace["compiler_input"]["workflow_ir"]["auxiliary_relations"] == [
        {"source": "decision", "relation_type": "failure", "targets": ["manual_review"]}
    ]
    final_edges = edge_set(trace["final_shadow_graph"])
    assert ("decision", "manual_review", "failure") in final_edges
    assert ("decision", "reject", "failure") not in final_edges
    assert ("decision", "approve", "success") in final_edges


def test_route_ambiguous_edge_reference_stops_before_lowering(monkeypatch) -> None:
    graph = route_graph()
    graph["edges"].append({"id": "e_failure_2", "source": "approve", "target": "manual_review", "label": "failure"})
    monkeypatch.setattr(
        semantic_shadow,
        "_complete_json",
        lambda _system, _payload: SimpleNamespace(raw_response="", parsed_json=route_goal(), parse_error=None, normalized_from_fence=False),
    )

    trace = semantic_shadow.run_semantic_shadow(graph, "失败情况转人工审核")

    assert trace["failure_stage"] == "validation"
    assert trace["validation"]["issues"][0]["kind"] == "ambiguous_edge_reference"
    assert trace["validation"]["issues"][0]["candidates"] == ["e_failure", "e_failure_2"]
    assert trace["lowering_result"] is None


def test_route_missing_target_stops_before_lowering(monkeypatch) -> None:
    graph = route_graph()
    graph["nodes"] = [node for node in graph["nodes"] if node["id"] != "manual_review"]
    monkeypatch.setattr(
        semantic_shadow,
        "_complete_json",
        lambda _system, _payload: SimpleNamespace(raw_response="", parsed_json=route_goal(), parse_error=None, normalized_from_fence=False),
    )

    trace = semantic_shadow.run_semantic_shadow(graph, "失败情况转人工审核")

    assert trace["failure_stage"] == "validation"
    assert trace["validation"]["issues"][0]["kind"] == "missing_node_reference"
    assert trace["lowering_result"] is None


def test_route_contract_rejects_compiler_facing_fields() -> None:
    result = validate_semantic_goal_spec(
        {
            "status": "resolved",
            "semantic_goals": [
                {
                    "kind": "route",
                    "edge_id": "e_failure",
                    "auxiliary_relations": [{"source": "decision", "relation_type": "failure", "targets": ["manual_review"]}],
                    "required_relations": [{"source": "decision", "target": "manual_review", "label": "failure"}],
                    "objects": [
                        {"type": "edge_reference", "text": "失败情况", "role": "route"},
                        {"type": "existing_reference", "text": "人工审核", "role": "target"},
                    ],
                    "relation": "route_to",
                }
            ],
        }
    )

    assert result["valid"] is False
    assert "semantic_goal_contract_violation" in {issue["kind"] for issue in result["issues"]}
