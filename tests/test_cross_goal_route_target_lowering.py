from __future__ import annotations

from types import SimpleNamespace

from src import semantic_shadow
from src.goal_lowering import lower_grounded_goal_to_workflow_ir
from src.semantic_goal import ground_semantic_goal_spec, validate_grounded_goal


def graph() -> dict:
    return {
        "nodes": [
            {"id": "decision", "type": "decision", "label": "缁撴灉鍒ゆ柇", "x": 0, "y": 0},
            {"id": "manual", "type": "task", "label": "浜哄伐澶勭悊", "x": 220, "y": 80},
            {"id": "exception", "type": "task", "label": "寮傚父澶勭悊", "x": 220, "y": 180},
        ],
        "edges": [
            {"id": "e_failure", "source": "decision", "target": "manual", "label": "failure", "branch_label": "澶辫触"},
            {"id": "e_exception", "source": "decision", "target": "exception", "label": "exception", "branch_label": "寮傚父"},
        ],
    }


def split_exists_and_route_goal() -> dict:
    return {
        "status": "resolved",
        "semantic_goals": [
            {
                "id": "g1",
                "goal_category": "object_state_goal",
                "text": "澧炲姞浜哄伐瀹℃牳鑺傜偣",
                "objects": [{"id": "n", "type": "new_entity", "label": "浜哄伐瀹℃牳鑺傜偣"}],
                "object_desired_states": [{"object": "n", "state": "exists"}],
            },
            {
                "id": "g2",
                "goal_category": "relation_goal",
                "text": "route failure branch to manual review",
                "objects": [
                    {"id": "e", "type": "edge_reference", "text": "澶辫触鍒嗘敮"},
                    {"id": "n", "type": "new_entity", "label": "浜哄伐瀹℃牳"},
                ],
                "relation_intents": [{"source": "e", "relation": "route_to", "target": "n"}],
            },
        ],
        "clarification": {"needed": False, "question": "", "reason": ""},
    }


def test_split_exists_and_route_reuse_same_new_entity_ref() -> None:
    grounded = ground_semantic_goal_spec(split_exists_and_route_goal(), graph(), {"node_ids": [], "edge_ids": []})
    lowered = lower_grounded_goal_to_workflow_ir(grounded, graph())

    assert validate_grounded_goal(grounded)["valid"] is True
    assert lowered["ok"] is True
    ir = lowered["semantic_spec"]["workflow_ir"]
    created_ref = ir["control_flow_regions"][0]["structure"]["ref"]
    created_node = ir["entities"][0]
    assert isinstance(created_node["x"], (int, float))
    assert isinstance(created_node["y"], (int, float))
    created_node.pop("x")
    created_node.pop("y")
    assert ir["entities"] == [{"ref": created_ref, "node_type": "task", "label": "浜哄伐瀹℃牳鑺傜偣"}]
    assert ir["control_flow_regions"] == [{"relation_type": "success", "structure": {"type": "node", "ref": created_ref}}]
    assert ir["auxiliary_relations"] == [{"source": "decision", "relation_type": "failure", "targets": [created_ref]}]
    assert lowered["partial_execution"]["final_status"] == "success"
    assert lowered["partial_execution"]["executed_count"] == 2


def test_semantic_shadow_split_exists_and_route_runs_existing_backend(monkeypatch) -> None:
    monkeypatch.setattr(
        semantic_shadow,
        "_complete_json",
        lambda _system, _payload: SimpleNamespace(
            raw_response="",
            parsed_json=split_exists_and_route_goal(),
            parse_error=None,
            normalized_from_fence=False,
        ),
    )

    spec = split_exists_and_route_goal()
    source = f"{spec['semantic_goals'][0]['objects'][0]['label']} {spec['semantic_goals'][1]['objects'][0]['text']}"
    trace = semantic_shadow.run_semantic_shadow(graph(), source)

    assert trace["stage"] == "success"
    final_nodes = trace["final_shadow_graph"]["nodes"]
    created = [node for node in final_nodes if node.get("label") == "浜哄伐瀹℃牳鑺傜偣"]
    assert len(created) == 1
    assert any(
        edge.get("source") == "decision" and edge.get("target") == created[0]["id"] and edge.get("label") == "failure"
        for edge in trace["final_shadow_graph"]["edges"]
    )


def test_two_routes_share_one_created_target() -> None:
    spec = {
        "status": "resolved",
        "semantic_goals": [
            {
                "id": "g1",
                "goal_category": "object_state_goal",
                "text": "澧炲姞閫氱煡鑺傜偣",
                "objects": [{"id": "n", "type": "new_entity", "label": "閫氱煡"}],
                "object_desired_states": [{"object": "n", "state": "exists"}],
            },
            {
                "id": "g2",
                "goal_category": "relation_goal",
                "text": "澶辫触鍜屽紓甯搁兘璧伴€氱煡",
                "objects": [
                    {"id": "f", "type": "edge_reference", "text": "澶辫触"},
                    {"id": "x", "type": "edge_reference", "text": "寮傚父"},
                    {"id": "n", "type": "new_entity", "label": "閫氱煡"},
                ],
                "relation_intents": [
                    {"source": "f", "relation": "route_to", "target": "n"},
                    {"source": "x", "relation": "route_to", "target": "n"},
                ],
            },
        ],
    }
    grounded = ground_semantic_goal_spec(spec, graph(), {"node_ids": [], "edge_ids": []})
    lowered = lower_grounded_goal_to_workflow_ir(grounded, graph())

    assert lowered["ok"] is True
    ir = lowered["semantic_spec"]["workflow_ir"]
    created_ref = ir["control_flow_regions"][0]["structure"]["ref"]
    assert len(ir["control_flow_regions"]) == 1
    assert {
        (row["source"], row["relation_type"], tuple(row["targets"]))
        for row in ir["auxiliary_relations"]
    } == {
        ("decision", "failure", (created_ref,)),
        ("decision", "exception", (created_ref,)),
    }


def test_existing_target_is_reused_without_duplicate_creation() -> None:
    test_graph = graph()
    test_graph["nodes"].append({"id": "manual_review", "type": "task", "label": "浜哄伐瀹℃牳", "x": 440, "y": 80})
    spec = {
        "status": "resolved",
        "semantic_goals": [
            {
                "id": "g1",
                "goal_category": "object_state_goal",
                "text": "浜哄伐瀹℃牳瀛樺湪",
                "objects": [{"id": "n", "type": "existing_reference", "text": "浜哄伐瀹℃牳"}],
                "object_desired_states": [{"object": "n", "state": "exists"}],
            },
            {
                "id": "g2",
                "goal_category": "relation_goal",
                "text": "route failure branch to manual review",
                "objects": [
                    {"id": "e", "type": "edge_reference", "text": "澶辫触鍒嗘敮"},
                    {"id": "n", "type": "existing_reference", "text": "浜哄伐瀹℃牳"},
                ],
                "relation_intents": [{"source": "e", "relation": "route_to", "target": "n"}],
            },
        ],
    }
    grounded = ground_semantic_goal_spec(spec, test_graph, {"node_ids": [], "edge_ids": []})
    lowered = lower_grounded_goal_to_workflow_ir(grounded, test_graph)

    assert lowered["ok"] is True
    ir = lowered["semantic_spec"]["workflow_ir"]
    assert ir["control_flow_regions"] == []
    assert ir["auxiliary_relations"] == [{"source": "decision", "relation_type": "failure", "targets": ["manual_review"]}]


def test_ambiguous_existing_route_target_stops_before_lowering() -> None:
    test_graph = {
        "nodes": [
            {"id": "decision", "type": "decision", "label": "Decision", "x": 0, "y": 0},
            {"id": "reject", "type": "task", "label": "Rejected", "x": 220, "y": 80},
            {"id": "manual_review_a", "type": "task", "label": "Manual Review", "x": 440, "y": 80},
            {"id": "manual_review_b", "type": "task", "label": "Manual Review", "x": 440, "y": 180},
        ],
        "edges": [
            {"id": "e_failure", "source": "decision", "target": "reject", "label": "failure", "branch_label": "failure"}
        ],
    }
    spec = {
        "status": "resolved",
        "semantic_goals": [
            {
                "id": "g1",
                "goal_category": "relation_goal",
                "text": "failure route to Manual Review",
                "objects": [
                    {"id": "e", "type": "edge_reference", "text": "failure"},
                    {"id": "n", "type": "existing_reference", "text": "Manual Review"},
                ],
                "relation_intents": [{"source": "e", "relation": "route_to", "target": "n"}],
            }
        ],
    }
    grounded = ground_semantic_goal_spec(spec, test_graph, {"node_ids": [], "edge_ids": []})
    validation = validate_grounded_goal(grounded)

    assert validation["lowering_ready"] is False
    assert any(issue["kind"] == "ambiguous_node_reference" for issue in validation["issues"])


def test_new_route_target_without_exists_is_not_auto_created() -> None:
    spec = {
        "status": "resolved",
        "semantic_goals": [
            {
                "id": "g1",
                "goal_category": "relation_goal",
                "text": "route failure branch to manual review",
                "objects": [
                    {"id": "e", "type": "edge_reference", "text": "澶辫触鍒嗘敮"},
                    {"id": "n", "type": "new_entity", "label": "Legal Review"},
                ],
                "relation_intents": [{"source": "e", "relation": "route_to", "target": "n"}],
            }
        ],
    }
    grounded = ground_semantic_goal_spec(spec, graph(), {"node_ids": [], "edge_ids": []})
    lowered = lower_grounded_goal_to_workflow_ir(grounded, graph())

    assert validate_grounded_goal(grounded)["valid"] is True
    assert lowered["ok"] is False
    assert lowered["issues"][0]["kind"] == "missing_node_reference"
    assert lowered["partial_execution"]["final_status"] == "unsupported"
