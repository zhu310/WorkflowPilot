from __future__ import annotations

from copy import deepcopy

from src.workflow_ir_core import ground_understand_semantic_spec, validate_grounded_semantic_spec


def graph() -> dict:
    return {
        "nodes": [
            {"id": "n_x", "type": "task", "label": "X", "x": 0, "y": 0},
            {"id": "n_a", "type": "task", "label": "A", "x": 100, "y": 0},
            {"id": "n_b", "type": "task", "label": "B", "x": 200, "y": 0},
            {"id": "n_c", "type": "task", "label": "C", "x": 300, "y": 0},
            {"id": "n_d", "type": "task", "label": "D", "x": 400, "y": 0},
        ],
        "edges": [
            {"id": "e_x_a", "source": "n_x", "target": "n_a", "label": ""},
            {"id": "e_a_b", "source": "n_a", "target": "n_b", "label": ""},
            {"id": "e_b_c", "source": "n_b", "target": "n_c", "label": ""},
            {"id": "e_a_d_success", "source": "n_a", "target": "n_d", "label": "success", "branch_label": "old"},
        ],
    }


def validate_after_ground(spec: dict, current_graph: dict | None = None) -> tuple[dict, dict]:
    current_graph = current_graph or graph()
    grounded = ground_understand_semantic_spec(spec, current_graph)
    return grounded, validate_grounded_semantic_spec(grounded, current_graph)


def issue_kinds(result: dict) -> set[str]:
    return {row["kind"] for row in result["issues"]}


def test_resolved_fully_grounded_sequence_is_ready_to_attempt_compile() -> None:
    _, result = validate_after_ground(
        {
            "status": "resolved",
            "workflow_ir": {
                "control_flow_regions": [
                    {
                        "relation_type": "",
                        "structure": {"type": "sequence", "items": [{"type": "node", "id": "A"}, {"type": "node", "id": "B"}]},
                    }
                ]
            },
        }
    )

    assert result == {"valid": True, "deterministic_ready": True, "issues": []}


def test_resolved_nested_parallel_is_ready_to_attempt_compile() -> None:
    _, result = validate_after_ground(
        {
            "status": "resolved",
            "workflow_ir": {
                "control_flow_regions": [
                    {
                        "relation_type": "",
                        "structure": {
                            "type": "sequence",
                            "items": [
                                {"type": "parallel", "join_mode": "all", "branches": [{"type": "node", "id": "A"}, {"type": "node", "id": "B"}]},
                                {"type": "node", "id": "C"},
                            ],
                        },
                    }
                ]
            },
        }
    )

    assert result["valid"] is True
    assert result["deterministic_ready"] is True


def test_resolved_with_ambiguous_grounding_issue_is_not_ready() -> None:
    current_graph = graph()
    current_graph["nodes"].append({"id": "n_a_2", "type": "task", "label": "A", "x": 500, "y": 0})

    _, result = validate_after_ground({"status": "resolved", "workflow_ir": {"properties": [{"node_id": "A", "changes": {"label": "A+"}}]}}, current_graph)

    assert result["valid"] is False
    assert result["deterministic_ready"] is False
    assert "grounding_issues_present" in issue_kinds(result)


def test_resolved_with_missing_reference_is_not_ready() -> None:
    _, result = validate_after_ground({"status": "resolved", "workflow_ir": {"properties": [{"node_id": "Missing", "changes": {"label": "M"}}]}})

    assert result["valid"] is False
    assert result["deterministic_ready"] is False
    assert "grounding_issues_present" in issue_kinds(result)


def test_partial_is_valid_but_not_deterministic_ready_and_keeps_grounded_ir() -> None:
    grounded, result = validate_after_ground(
        {
            "status": "partial",
            "workflow_ir": {
                "control_flow_regions": [
                    {"relation_type": "", "structure": {"type": "parallel", "branches": [{"type": "node", "id": "A"}, {"type": "node", "id": "B"}]}}
                ]
            },
            "semantic_goals": [{"goal_type": "merge_target", "description": "merge target still needs local completion"}],
        }
    )

    assert grounded["workflow_ir"]["control_flow_regions"][0]["structure"]["branches"] == [{"type": "node", "id": "n_a"}, {"type": "node", "id": "n_b"}]
    assert result["valid"] is True
    assert result["deterministic_ready"] is False
    assert {"partial_spec", "unresolved_semantic_goals"} <= issue_kinds(result)


def test_clarification_is_not_ready_and_does_not_enter_execution() -> None:
    grounded, result = validate_after_ground(
        {
            "status": "clarification",
            "workflow_ir": {"properties": [{"node_id": "A", "changes": {"label": "A+"}}]},
            "clarification": {"needed": True, "question": "Which goal?", "reason": "multiple possible goals"},
        }
    )

    assert grounded["workflow_ir"]["properties"] == []
    assert result["valid"] is True
    assert result["deterministic_ready"] is False
    assert "clarification_needed" in issue_kinds(result)


def test_invalid_ir_schema_is_invalid_and_not_ready() -> None:
    grounded = {
        "status": "resolved",
        "workflow_ir": {"control_flow_regions": {}, "auxiliary_relations": [], "absent_entities": [], "properties": [], "edge_properties": []},
        "semantic_goals": [],
        "clarification": {"needed": False, "question": "", "reason": ""},
        "grounding": {"issues": [], "provenance": []},
    }

    result = validate_grounded_semantic_spec(grounded, graph())

    assert result["valid"] is False
    assert result["deterministic_ready"] is False
    assert "invalid_workflow_ir" in issue_kinds(result)


def test_illegal_relation_type_is_not_ready() -> None:
    _, result = validate_after_ground(
        {
            "status": "resolved",
            "workflow_ir": {
                "control_flow_regions": [
                    {"relation_type": "after", "structure": {"type": "sequence", "items": [{"type": "node", "id": "A"}, {"type": "node", "id": "B"}]}}
                ]
            },
        }
    )

    assert result["valid"] is False
    assert result["deterministic_ready"] is False
    assert "grounding_issues_present" in issue_kinds(result)


def test_invalid_property_target_is_rejected_by_ir_validation() -> None:
    grounded = {
        "status": "resolved",
        "workflow_ir": {"control_flow_regions": [], "auxiliary_relations": [], "absent_entities": [], "properties": [{"node_id": "missing", "changes": {"label": "M"}}], "edge_properties": []},
        "semantic_goals": [],
        "clarification": {"needed": False, "question": "", "reason": ""},
        "grounding": {"issues": [], "provenance": []},
    }

    result = validate_grounded_semantic_spec(grounded, graph())

    assert result["valid"] is False
    assert result["deterministic_ready"] is False
    assert "invalid_workflow_ir" in issue_kinds(result)


def test_invalid_edge_property_target_is_rejected_by_ir_validation() -> None:
    grounded = {
        "status": "resolved",
        "workflow_ir": {"control_flow_regions": [], "auxiliary_relations": [], "absent_entities": [], "properties": [], "edge_properties": [{"edge_id": "missing_edge", "changes": {"branch_label": "x"}}]},
        "semantic_goals": [],
        "clarification": {"needed": False, "question": "", "reason": ""},
        "grounding": {"issues": [], "provenance": []},
    }

    result = validate_grounded_semantic_spec(grounded, graph())

    assert result["valid"] is False
    assert result["deterministic_ready"] is False
    assert "invalid_workflow_ir" in issue_kinds(result)


def test_unresolved_semantic_goals_block_ready_without_making_spec_invalid() -> None:
    _, result = validate_after_ground(
        {
            "status": "resolved",
            "workflow_ir": {"properties": [{"node_id": "A", "changes": {"label": "A+"}}]},
            "semantic_goals": [{"goal_type": "placement", "description": "new node placement still needs completion"}],
        }
    )

    assert result["valid"] is True
    assert result["deterministic_ready"] is False
    assert "unresolved_semantic_goals" in issue_kinds(result)


def test_llm_deterministic_ready_field_is_not_authoritative() -> None:
    grounded, _ = validate_after_ground({"status": "resolved", "workflow_ir": {"properties": [{"node_id": "A", "changes": {"label": "A+"}}]}})
    grounded["deterministic_ready"] = True

    result = validate_grounded_semantic_spec(grounded, graph())

    assert result["valid"] is False
    assert result["deterministic_ready"] is False
    assert "llm_deterministic_ready_not_allowed" in issue_kinds(result)


def test_validation_does_not_do_topology_completion() -> None:
    grounded, result = validate_after_ground(
        {
            "status": "resolved",
            "workflow_ir": {
                "control_flow_regions": [
                    {
                        "relation_type": "",
                        "structure": {
                            "type": "sequence",
                            "items": [
                                {"type": "parallel", "branches": [{"type": "node", "id": "A"}, {"type": "node", "id": "B"}]},
                                {"type": "node", "id": "C"},
                            ],
                        },
                    }
                ]
            },
        }
    )

    snapshot = deepcopy(grounded)
    result_text = str(result)

    assert grounded == snapshot
    assert "n_x" not in result_text
    assert "required_relations" not in result_text
    assert "forbidden_relations" not in result_text


def test_validation_does_not_read_conversation_history_or_reinterpret_user_intent() -> None:
    grounded, _ = validate_after_ground({"status": "resolved", "workflow_ir": {"properties": [{"node_id": "A", "changes": {"label": "A+"}}]}})

    result_without_history = validate_grounded_semantic_spec(grounded, graph())
    result_with_unrelated_context = validate_grounded_semantic_spec(grounded, graph())

    assert result_with_unrelated_context == result_without_history


def test_status_resolved_with_grounding_issue_cannot_be_ready() -> None:
    grounded = {
        "status": "resolved",
        "workflow_ir": {"control_flow_regions": [], "auxiliary_relations": [], "absent_entities": [], "properties": [], "edge_properties": []},
        "semantic_goals": [],
        "clarification": {"needed": False, "question": "", "reason": ""},
        "grounding": {"issues": [{"kind": "ambiguous_node_reference", "location": "workflow_ir.properties[0].node_id"}], "provenance": []},
    }

    result = validate_grounded_semantic_spec(grounded, graph())

    assert result["valid"] is False
    assert result["deterministic_ready"] is False
    assert "grounding_issues_present" in issue_kinds(result)


def test_machine_checkable_validation_does_not_claim_natural_language_completeness() -> None:
    _, result = validate_after_ground({"status": "resolved", "workflow_ir": {"properties": [{"node_id": "A", "changes": {"label": "A+"}}]}})

    assert result["valid"] is True
    assert result["deterministic_ready"] is True
    assert "semantic_equivalence" not in result
    assert all("semantic_equivalence" not in row["kind"] for row in result["issues"])
