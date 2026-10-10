from __future__ import annotations

from types import SimpleNamespace

from src import semantic_shadow
from src.goal_lowering import lower_grounded_goal_to_workflow_ir
from src.semantic_goal import ground_semantic_goal_spec


def route_graph() -> dict:
    return {
        "nodes": [
            {"id": "decision", "type": "decision", "label": "Decision"},
            {"id": "exception", "type": "task", "label": "Exception"},
            {"id": "manual", "type": "task", "label": "Manual Handling"},
        ],
        "edges": [
            {"id": "e_exception", "source": "decision", "target": "exception", "label": "exception", "branch_label": "exception"}
        ],
    }


def route_to_new_entity_without_exists(label: str = "Manual Handling") -> dict:
    return {
        "status": "resolved",
        "semantic_goals": [
            {
                "id": "g1",
                "goal_category": "relation_goal",
                "text": "this branch routes to manual handling",
                "objects": [
                    {"id": "e", "type": "edge_reference", "text": "this branch"},
                    {"id": "m", "type": "new_entity", "label": label},
                ],
                "relation_intents": [{"source": "e", "relation": "route_to", "target": "m"}],
            }
        ],
        "clarification": {"needed": False, "question": "", "reason": ""},
    }


def test_selected_edge_route_to_existing_target_succeeds_even_if_llm_outputs_new_entity(monkeypatch) -> None:
    monkeypatch.setattr(
        semantic_shadow,
        "_complete_json",
        lambda _system, _payload: SimpleNamespace(
            raw_response="",
            parsed_json=route_to_new_entity_without_exists(),
            parse_error=None,
            normalized_from_fence=False,
        ),
    )

    trace = semantic_shadow.run_semantic_shadow(
        route_graph(),
        "this branch routes to manual handling",
        {"node_ids": [], "edge_ids": ["e_exception"]},
    )

    assert trace["stage"] == "success"
    assert trace["compiler_input"]["workflow_ir"]["control_flow_regions"] == []
    assert trace["compiler_input"]["workflow_ir"]["auxiliary_relations"] == [
        {"source": "decision", "relation_type": "exception", "targets": ["manual"]}
    ]
    assert not any(node.get("label") == "Manual Handling" and node.get("id") != "manual" for node in trace["final_shadow_graph"]["nodes"])


def test_route_target_new_entity_without_exists_and_missing_graph_target_clarifies(monkeypatch) -> None:
    graph = route_graph()
    graph["nodes"] = [node for node in graph["nodes"] if node.get("id") != "manual"]
    monkeypatch.setattr(
        semantic_shadow,
        "_complete_json",
        lambda _system, _payload: SimpleNamespace(
            raw_response="",
            parsed_json=route_to_new_entity_without_exists(),
            parse_error=None,
            normalized_from_fence=False,
        ),
    )

    trace = semantic_shadow.run_semantic_shadow(graph, "this branch routes to manual handling", {"node_ids": [], "edge_ids": ["e_exception"]})
    result = semantic_shadow.semantic_trace_to_transaction_result(trace)

    assert result.status == "needs_clarification"
    assert trace["final_shadow_graph"] is None
    assert trace["validation"]["issues"][0]["kind"] == "missing_node_reference"


def test_route_target_new_entity_without_exists_and_multiple_graph_targets_clarifies(monkeypatch) -> None:
    graph = route_graph()
    graph["nodes"].append({"id": "manual_2", "type": "task", "label": "Manual Handling"})
    monkeypatch.setattr(
        semantic_shadow,
        "_complete_json",
        lambda _system, _payload: SimpleNamespace(
            raw_response="",
            parsed_json=route_to_new_entity_without_exists(),
            parse_error=None,
            normalized_from_fence=False,
        ),
    )

    trace = semantic_shadow.run_semantic_shadow(graph, "this branch routes to manual handling", {"node_ids": [], "edge_ids": ["e_exception"]})
    result = semantic_shadow.semantic_trace_to_transaction_result(trace)

    assert result.status == "needs_clarification"
    assert trace["final_shadow_graph"] is None
    assert trace["validation"]["issues"][0]["kind"] == "ambiguous_node_reference"


def test_explicit_create_plus_route_still_lowers_to_new_target() -> None:
    spec = {
        "status": "resolved",
        "semantic_goals": [
            {
                "id": "g1",
                "goal_category": "composite_goal",
                "text": "create review and route exception to it",
                "objects": [
                    {"id": "e", "type": "edge_reference", "text": "exception"},
                    {"id": "m", "type": "new_entity", "label": "Manual Review"},
                ],
                "object_desired_states": [{"object": "m", "state": "exists"}],
                "relation_intents": [{"source": "e", "relation": "route_to", "target": "m"}],
            }
        ],
    }
    grounded = ground_semantic_goal_spec(spec, route_graph(), {"node_ids": [], "edge_ids": []})
    lowered = lower_grounded_goal_to_workflow_ir(grounded, route_graph())

    assert lowered["ok"] is True
    ir = lowered["semantic_spec"]["workflow_ir"]
    new_ref = ir["control_flow_regions"][0]["structure"]["ref"]
    assert ir["auxiliary_relations"] == [{"source": "decision", "relation_type": "exception", "targets": [new_ref]}]
