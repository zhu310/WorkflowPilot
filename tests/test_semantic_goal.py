from __future__ import annotations

from types import SimpleNamespace

from src import semantic_shadow
from src.semantic_goal import ground_semantic_goal_spec, validate_grounded_goal, validate_semantic_goal_spec


def contains_subset(rows: list[dict], expected: dict) -> bool:
    return any(all(row.get(key) == value for key, value in expected.items()) for row in rows)


def graph() -> dict:
    return {
        "nodes": [
            {"id": "parse", "type": "task", "label": "Parse Task", "x": 100, "y": 80},
            {"id": "knowledge", "type": "task", "label": "Knowledge Retrieval", "x": 20, "y": 20},
            {"id": "submit_a", "type": "task", "label": "Submit", "x": 0, "y": 0},
            {"id": "submit_b", "type": "task", "label": "Submit", "x": 0, "y": 100},
        ],
        "edges": [],
    }


def layout_goal() -> dict:
    return {
        "status": "resolved",
        "semantic_goals": [
            {
                "kind": "layout",
                "text": "Put Knowledge Retrieval to the right of Parse Task.",
                "objects": [
                    {"reference": "Knowledge Retrieval", "role": "target"},
                    {"reference": "Parse Task", "role": "anchor"},
                ],
                "relation": "right_of",
                "explicit_constraints": ["preserve_topology"],
            }
        ],
        "clarification": {"needed": False, "question": "", "reason": ""},
    }


def test_semantic_goal_rejects_workflow_ir_fields() -> None:
    result = validate_semantic_goal_spec(
        {
            "status": "resolved",
            "workflow_ir": {"properties": []},
            "semantic_goals": [
                {
                    "kind": "layout",
                    "objects": [{"reference": "A", "role": "target"}],
                    "relation": "right_of",
                }
            ],
        }
    )

    assert result["valid"] is False
    assert "semantic_goal_contract_violation" in {issue["kind"] for issue in result["issues"]}


def test_layout_goal_grounding_binds_target_and_anchor() -> None:
    grounded = ground_semantic_goal_spec(layout_goal(), graph(), {"node_ids": [], "edge_ids": []})

    assert grounded["grounding"]["issues"] == []
    bindings = grounded["semantic_goals"][0]["bindings"]
    assert contains_subset(bindings, {"type": "existing_reference", "reference": "Knowledge Retrieval", "role": "target", "binding_type": "node", "id": "knowledge"})
    assert contains_subset(bindings, {"type": "existing_reference", "reference": "Parse Task", "role": "anchor", "binding_type": "node", "id": "parse"})
    assert {row["resolved_id"] for row in grounded["grounding"]["provenance"]} == {"knowledge", "parse"}


def test_grounding_reports_ambiguous_candidates() -> None:
    grounded = ground_semantic_goal_spec(
        {
            "status": "resolved",
            "semantic_goals": [
                {
                    "kind": "layout",
                    "objects": [
                        {"reference": "Submit", "role": "target"},
                        {"reference": "Parse Task", "role": "anchor"},
                    ],
                    "relation": "right_of",
                }
            ],
        },
        graph(),
        {"node_ids": [], "edge_ids": []},
    )

    issues = grounded["grounding"]["issues"]
    assert issues[0]["kind"] == "ambiguous_node_reference"
    assert issues[0]["candidates"] == ["submit_a", "submit_b"]


def test_existing_reference_object_grounds_to_node() -> None:
    grounded = ground_semantic_goal_spec(
        {
            "status": "resolved",
            "semantic_goals": [
                {
                    "kind": "ordering",
                    "subject": {"type": "new_entity", "label": "Shipping Confirmation"},
                    "anchor": {"type": "existing_reference", "text": "Parse Task"},
                    "relation": "before",
                }
            ],
        },
        graph(),
        {"node_ids": [], "edge_ids": []},
    )

    bindings = grounded["semantic_goals"][0]["bindings"]
    assert contains_subset(bindings, {"type": "existing_reference", "reference": "Parse Task", "role": "anchor", "binding_type": "node", "id": "parse"})


def test_new_entity_does_not_enter_node_matching() -> None:
    grounded = ground_semantic_goal_spec(
        {
            "status": "resolved",
            "semantic_goals": [
                {
                    "kind": "ordering",
                    "subject": {"type": "new_entity", "label": "Outbound Confirmation"},
                    "anchor": {"type": "existing_reference", "text": "Parse Task"},
                    "relation": "before",
                }
            ],
        },
        graph(),
        {"node_ids": [], "edge_ids": []},
    )

    assert grounded["grounding"]["issues"] == []
    subject = next(binding for binding in grounded["semantic_goals"][0]["bindings"] if binding["role"] == "subject")
    assert subject == {"type": "new_entity", "label": "Outbound Confirmation", "role": "subject", "needs_creation": True}
    assert any(row["resolution_type"] == "new_entity" for row in grounded["grounding"]["provenance"])


def test_ordering_new_entity_and_existing_anchor_passes_goal_validation() -> None:
    spec = {
        "status": "resolved",
        "semantic_goals": [
            {
                "kind": "ordering",
                "subject": {"type": "new_entity", "label": "Outbound Confirmation"},
                "anchor": {"type": "existing_reference", "text": "Parse Task"},
                "relation": "before",
            }
        ],
    }

    semantic_validation = validate_semantic_goal_spec(spec)
    grounded = ground_semantic_goal_spec(spec, graph(), {"node_ids": [], "edge_ids": []})
    grounded_validation = validate_grounded_goal(grounded)

    assert semantic_validation["ready_for_grounding"] is True
    assert grounded["grounding"]["issues"] == []
    assert grounded_validation["valid"] is True
    assert grounded_validation["lowering_ready"] is True
    assert grounded_validation["issues"] == []


def test_v2_layout_relation_maps_to_v1_compat_goal() -> None:
    spec = {
        "status": "resolved",
        "semantic_goals": [
            {
                "goal_category": "relation_goal",
                "text": "Put Knowledge Retrieval to the right of Parse Task.",
                "objects": [
                    {"id": "knowledge_obj", "type": "existing_reference", "text": "Knowledge Retrieval", "role": "object"},
                    {"id": "parse_obj", "type": "existing_reference", "text": "Parse Task", "role": "object"},
                ],
                "relation_intents": [
                    {"source": "knowledge_obj", "relation": "right_of", "target": "parse_obj"}
                ],
            }
        ],
    }

    semantic_validation = validate_semantic_goal_spec(spec, source_instruction="Put Knowledge Retrieval to the right of Parse Task.")
    grounded = ground_semantic_goal_spec(spec, graph(), {"node_ids": [], "edge_ids": []})

    assert semantic_validation["ready_for_grounding"] is True
    assert semantic_validation["normalized"]["semantic_goals_v2"][0]["relation_intents"][0]["relation"] == "right_of"
    assert semantic_validation["normalized"]["semantic_goals"][0]["kind"] == "layout"
    assert grounded["grounding"]["issues"] == []


def test_v2_ordering_exists_before_maps_to_v1_compat_goal() -> None:
    spec = {
        "status": "resolved",
        "semantic_goals": [
            {
                "goal_category": "composite_goal",
                "text": "Add Outbound Confirmation before Parse Task.",
                "objects": [
                    {"id": "confirm", "type": "new_entity", "label": "Outbound Confirmation", "role": "object"},
                    {"id": "parse", "type": "existing_reference", "text": "Parse Task", "role": "object"},
                ],
                "object_desired_states": [{"object": "confirm", "state": "exists"}],
                "relation_intents": [{"source": "confirm", "relation": "before", "target": "parse"}],
            }
        ],
    }

    semantic_validation = validate_semantic_goal_spec(spec, source_instruction="Add Outbound Confirmation before Parse Task.")
    grounded = ground_semantic_goal_spec(spec, graph(), {"node_ids": [], "edge_ids": []})

    assert semantic_validation["ready_for_grounding"] is True
    assert semantic_validation["normalized"]["semantic_goals"][0]["kind"] == "ordering"
    assert grounded["grounding"]["issues"] == []
    subject = next(binding for binding in grounded["semantic_goals"][0]["bindings"] if binding["role"] == "subject")
    assert subject["needs_creation"] is True


def test_v2_absent_object_state_is_expressible_and_ready_for_grounding() -> None:
    spec = {
        "status": "resolved",
        "semantic_goals": [
            {
                "goal_category": "object_state_goal",
                "text": "Delete inventory check.",
                "objects": [
                    {"id": "inventory", "type": "existing_reference", "text": "Inventory Check", "role": "object"}
                ],
                "object_desired_states": [{"object": "inventory", "state": "absent"}],
            }
        ],
    }

    semantic_validation = validate_semantic_goal_spec(spec, source_instruction="Delete Inventory Check.")

    assert semantic_validation["valid"] is True
    assert semantic_validation["ready_for_grounding"] is True
    assert semantic_validation["issues"] == []
    assert semantic_validation["normalized"]["semantic_goals_v2"][0]["object_desired_states"][0]["state"] == "absent"


def test_v2_forbids_operation_and_graph_binding_fields() -> None:
    result = validate_semantic_goal_spec(
        {
            "status": "resolved",
            "semantic_goals": [
                {
                    "goal_category": "object_state_goal",
                    "objects": [{"id": "x", "type": "existing_reference", "text": "A", "role": "object", "node_id": "n1"}],
                    "object_desired_states": [{"object": "x", "state": "exists", "create_node": True}],
                }
            ],
        },
        source_instruction="A",
    )

    assert result["valid"] is False
    assert "semantic_goal_contract_violation" in {issue["kind"] for issue in result["issues"]}


def clarification_spec(question: str, reason: str, goals: list[dict] | None = None) -> dict:
    return {
        "status": "clarification",
        "semantic_goals": goals or [],
        "clarification": {"needed": True, "question": question, "reason": reason},
    }


def assert_safe_stop(monkeypatch, spec: dict, instruction: str) -> dict:
    monkeypatch.setattr(
        semantic_shadow,
        "_complete_json",
        lambda _system, _payload: SimpleNamespace(raw_response="", parsed_json=spec, parse_error=None, normalized_from_fence=False),
    )

    trace = semantic_shadow.run_semantic_shadow(graph(), instruction)
    result = semantic_shadow.semantic_trace_to_transaction_result(trace)

    assert trace["failure_stage"] == "validation"
    assert trace["lowering_result"] is None
    assert trace["compiler_input"] is None
    assert trace["planner"] is None
    assert trace["executor"] is None
    assert result.status == "needs_clarification"
    assert {issue["kind"] for issue in trace["validation"]["issues"]} >= {"clarification_needed"}
    return trace


def test_abstract_optimize_safe_stops_without_forcing_executable_goal(monkeypatch) -> None:
    trace = assert_safe_stop(
        monkeypatch,
        clarification_spec("What concrete change should I make?", "abstract optimization is not executable"),
        "优化一下流程",
    )

    assert trace["semantic_goal"]["normalized"]["semantic_goals"] == []


def test_abstract_reduce_wait_with_incomplete_placeholder_safe_stops(monkeypatch) -> None:
    trace = assert_safe_stop(
        monkeypatch,
        clarification_spec(
            "Which concrete waiting point should change?",
            "high-level waiting reduction needs a concrete edit target",
            [
                {
                    "kind": "layout",
                    "text": "减少等待，同时保留异常处理",
                    "objects": [],
                    "relation": "",
                    "explicit_constraints": ["preserve_exception_handling"],
                }
            ],
        ),
        "减少等待，同时保留异常处理",
    )

    assert "unsupported_layout_relation" not in {issue["kind"] for issue in trace["validation"]["issues"]}


def test_scope_missing_selection_safe_stops_without_empty_layout_error(monkeypatch) -> None:
    trace = assert_safe_stop(
        monkeypatch,
        clarification_spec(
            "Which area or selected nodes should I optimize?",
            "scope is missing and no selection was provided",
            [
                {
                    "kind": "layout",
                    "text": "只优化当前区域",
                    "objects": [],
                    "relation": "",
                    "explicit_constraints": ["preserve_topology"],
                }
            ],
        ),
        "只优化当前区域",
    )

    assert "unsupported_layout_relation" not in {issue["kind"] for issue in trace["validation"]["issues"]}


def test_clarification_preserves_partial_route_goal_with_unresolved_source() -> None:
    spec = {
        "status": "clarification",
        "semantic_goals": [
            {
                "id": "g1",
                "goal_category": "relation_goal",
                "objects": [
                    {"id": "source", "type": "unresolved_reference", "role": "source"},
                    {"id": "target", "type": "existing_reference", "text": "Human Review", "role": "target"},
                ],
                "relation_intents": [{"source": "source", "relation": "route_to", "target": "target"}],
            }
        ],
        "clarification": {"needed": True, "question": "Which route should go to Human Review?", "reason": "missing route"},
    }

    result = validate_semantic_goal_spec(spec, source_instruction="route to Human Review")

    assert result["valid"] is True
    assert result["ready_for_grounding"] is False
    assert result["normalized"]["semantic_goals"] == []
    goal = result["normalized"]["semantic_goals_v2"][0]
    assert goal["objects"][0]["type"] == "unresolved_reference"
    assert goal["objects"][1]["reference"] == "Human Review"
    assert goal["relation_intents"] == [{"source": "source", "relation": "route_to", "target": "target"}]
    issue_kinds = {issue["kind"] for issue in result["issues"]}
    assert {"clarification_needed", "missing_semantic_role"} <= issue_kinds


def test_clarification_preserves_known_object_with_unresolved_transformation() -> None:
    spec = {
        "status": "clarification",
        "semantic_goals": [
            {
                "id": "g1",
                "goal_category": "",
                "objects": [
                    {"id": "o1", "type": "edge_reference", "text": "this edge", "role": "object"},
                ],
                "unresolved_dimensions": [
                    {"dimension": "transformation", "object": "o1"},
                ],
            }
        ],
        "clarification": {"needed": True, "question": "How should this edge change?", "reason": "missing transformation"},
    }

    result = validate_semantic_goal_spec(spec, source_instruction="change this edge")

    assert result["valid"] is True
    assert result["ready_for_grounding"] is False
    goal = result["normalized"]["semantic_goals_v2"][0]
    assert goal["goal_category"] == ""
    assert goal["objects"] == [{"id": "o1", "type": "edge_reference", "reference": "this edge", "role": "object"}]
    assert goal["object_desired_states"] == []
    assert goal["relation_intents"] == []
    assert goal["unresolved_dimensions"] == [{"dimension": "transformation", "object": "o1"}]
    issue_kinds = {issue["kind"] for issue in result["issues"]}
    assert {"clarification_needed", "missing_semantic_dimension"} <= issue_kinds


def test_complete_v2_goal_requires_committed_category() -> None:
    spec = {
        "status": "resolved",
        "semantic_goals": [
            {
                "id": "g1",
                "objects": [{"id": "o1", "type": "existing_reference", "text": "Report", "role": "object"}],
                "object_desired_states": [{"object": "o1", "state": "property_equals", "property": "display_label", "value": "Final Report"}],
            }
        ],
        "clarification": {"needed": False, "question": "", "reason": ""},
    }

    result = validate_semantic_goal_spec(spec, source_instruction="rename Report to Final Report")

    assert result["valid"] is True
    assert result["ready_for_grounding"] is False
    assert "missing_goal_category" in {issue["kind"] for issue in result["issues"]}
