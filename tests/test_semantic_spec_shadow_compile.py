from __future__ import annotations

import pytest

import src.workflow_ir_core as core
from src.workflow_ir_core import (
    compile_grounded_semantic_spec,
    compile_ir,
    constraints_from_compiled,
    ground_understand_semantic_spec,
    validate_grounded_semantic_spec,
    validate_ir,
)


def graph() -> dict:
    return {
        "nodes": [
            {"id": "n_x", "type": "task", "label": "X", "x": 0, "y": 0},
            {"id": "n_a", "type": "task", "label": "A", "x": 100, "y": 0},
            {"id": "n_b", "type": "task", "label": "B", "x": 200, "y": 0},
            {"id": "n_c", "type": "task", "label": "C", "x": 300, "y": 0},
            {"id": "n_d", "type": "task", "label": "D", "x": 400, "y": 0},
            {"id": "n_review", "type": "human_review", "label": "Review", "x": 500, "y": 0},
        ],
        "edges": [
            {"id": "e_x_a", "source": "n_x", "target": "n_a", "label": ""},
            {"id": "e_a_b", "source": "n_a", "target": "n_b", "label": ""},
            {"id": "e_b_c", "source": "n_b", "target": "n_c", "label": ""},
            {"id": "e_c_d", "source": "n_c", "target": "n_d", "label": ""},
            {"id": "e_a_review_success", "source": "n_a", "target": "n_review", "label": "success", "branch_label": "old", "condition": {"expression": "score > 0"}},
            {"id": "e_b_review_failure", "source": "n_b", "target": "n_review", "label": "failure"},
        ],
    }


def shadow(spec: dict, current_graph: dict | None = None) -> tuple[dict, dict]:
    current_graph = current_graph or graph()
    grounded = ground_understand_semantic_spec(spec, current_graph)
    return grounded, compile_grounded_semantic_spec(grounded, current_graph)


def relations(rows: list[dict]) -> set[tuple[str, str, str]]:
    return {(row["source"], row["target"], row.get("label", "")) for row in rows}


def shared_new_graph() -> dict:
    return {
        "nodes": [
            {"id": "A", "type": "task", "label": "A"},
            {"id": "B", "type": "task", "label": "B"},
            {"id": "C", "type": "task", "label": "C"},
            {"id": "Z", "type": "task", "label": "Z"},
            {"id": "Y", "type": "task", "label": "Y"},
        ],
        "edges": [{"id": "e_z_y", "source": "Z", "target": "Y", "label": ""}],
    }


def shared_new_contract() -> dict:
    return {
        "allowed_existing_node_ids": ["A", "B", "C", "Z", "Y"],
        "allowed_structural_relation_types": [""],
        "allowed_auxiliary_relation_types": [""],
        "allowed_property_targets": ["A", "B", "C", "Z", "Y"],
        "allowed_edge_property_targets": ["e_z_y"],
        "allow_absent_entities": True,
        "allow_new_entities": True,
    }


def shared_new_ir() -> dict:
    occurrence = {"type": "node", "ref": "new_x"}
    return {
        "entities": [{"ref": "new_x", "node_type": "task", "label": "X"}],
        "control_flow_regions": [
            {"relation_type": "", "structure": {"type": "sequence", "items": [{"type": "node", "id": "A"}, occurrence]}},
            {"relation_type": "", "structure": {"type": "sequence", "items": [{"type": "node", "id": "B"}, occurrence]}},
            {"relation_type": "", "structure": {"type": "sequence", "items": [occurrence, {"type": "node", "id": "C"}]}},
        ],
    }


def test_entity_registry_allows_shared_new_ref_in_multiple_structural_occurrences() -> None:
    ir = shared_new_ir()
    valid, error = validate_ir(ir, shared_new_graph(), shared_new_contract())

    assert (valid, error) == (True, None)

    compiled = compile_ir(ir, shared_new_graph())
    constraints = constraints_from_compiled(shared_new_graph(), compiled)

    assert compiled["entities"] == [{"ref": "new_x", "type": "task", "label": "X"}]
    assert relations(constraints["required_relations"]) == {
        ("A", "new_x", ""),
        ("B", "new_x", ""),
        ("new_x", "C", ""),
    }
    assert constraints["forbidden_relations"] == []


def test_undeclared_structural_ref_is_rejected() -> None:
    ir = shared_new_ir()
    ir.pop("entities")

    assert validate_ir(ir, shared_new_graph(), shared_new_contract()) == (False, "undeclared new ref")


def test_duplicate_entity_declaration_is_rejected() -> None:
    ir = shared_new_ir()
    ir["entities"].append({"ref": "new_x", "node_type": "task", "label": "X"})

    assert validate_ir(ir, shared_new_graph(), shared_new_contract()) == (False, "duplicate new ref")


def test_conflicting_entity_declaration_metadata_is_rejected() -> None:
    ir = shared_new_ir()
    ir["control_flow_regions"][0]["structure"]["items"][1] = {
        "type": "node",
        "ref": "new_x",
        "node_type": "task",
        "label": "Different X",
    }

    assert validate_ir(ir, shared_new_graph(), shared_new_contract()) == (False, "conflicting new ref metadata")


def test_legacy_inline_new_node_declaration_still_valid() -> None:
    ir = {
        "control_flow_regions": [
            {
                "relation_type": "",
                "structure": {
                    "type": "sequence",
                    "items": [
                        {"type": "node", "id": "A"},
                        {"type": "node", "ref": "new_x", "node_type": "task", "label": "X"},
                    ],
                },
            }
        ]
    }

    assert validate_ir(ir, shared_new_graph(), shared_new_contract()) == (True, None)
    assert compile_ir(ir, shared_new_graph())["entities"] == [{"ref": "new_x", "type": "task", "label": "X"}]


def test_auxiliary_relation_can_target_registry_declared_new_ref() -> None:
    ir = {
        "entities": [{"ref": "new_x", "node_type": "task", "label": "X"}],
        "auxiliary_relations": [{"source": "A", "relation_type": "", "targets": ["new_x"]}],
    }

    assert validate_ir(ir, shared_new_graph(), shared_new_contract()) == (True, None)
    constraints = constraints_from_compiled(shared_new_graph(), compile_ir(ir, shared_new_graph()))
    assert relations(constraints["required_relations"]) == {("A", "new_x", "")}


def test_sequence_clear_path_compiles_to_constraints() -> None:
    _, result = shadow(
        {
            "status": "resolved",
            "workflow_ir": {
                "control_flow_regions": [
                    {"relation_type": "", "structure": {"type": "sequence", "items": [{"type": "node", "id": "B"}, {"type": "node", "id": "A"}]}}
                ]
            },
        }
    )

    assert result["ok"] is True
    assert result["stage"] == "constraints"
    assert ("n_b", "n_a", "") in relations(result["constraints"]["required_relations"])
    assert ("n_a", "n_b", "") in relations(result["constraints"]["forbidden_relations"])


def test_simple_parallel_uses_existing_compiler_boundary_inference() -> None:
    _, result = shadow(
        {
            "status": "resolved",
            "workflow_ir": {
                "control_flow_regions": [
                    {"relation_type": "", "structure": {"type": "parallel", "join_mode": "all", "branches": [{"type": "node", "id": "A"}, {"type": "node", "id": "B"}]}}
                ]
            },
        }
    )

    assert result["ok"] is True
    required = relations(result["constraints"]["required_relations"])
    forbidden = relations(result["constraints"]["forbidden_relations"])
    assert {("n_x", "n_a", ""), ("n_x", "n_b", ""), ("n_a", "n_c", ""), ("n_b", "n_c", "")} <= required
    assert ("n_a", "n_b", "") in forbidden
    assert result["constraints"]["properties"] == [{"node_id": "n_c", "changes": {"join": {"mode": "all"}}}]


def test_parallel_plus_merge_uses_structural_ir_without_edge_patch_adapter() -> None:
    _, result = shadow(
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

    assert result["ok"] is True
    required = relations(result["constraints"]["required_relations"])
    assert {("n_a", "n_c", ""), ("n_b", "n_c", "")} <= required
    assert ("n_a", "n_b", "") in relations(result["constraints"]["forbidden_relations"])
    assert result["constraints"]["properties"] == [{"node_id": "n_c", "changes": {"join": {"mode": "all"}}}]


def test_mixed_topology_compiles_directly() -> None:
    _, result = shadow(
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
                                {"type": "node", "id": "D"},
                            ],
                        },
                    }
                ]
            },
        }
    )

    assert result["ok"] is True
    required = relations(result["constraints"]["required_relations"])
    assert {("n_a", "n_c", ""), ("n_b", "n_c", ""), ("n_c", "n_d", "")} <= required


def test_node_property_compiles_without_obligation_or_skeleton() -> None:
    _, result = shadow({"status": "resolved", "workflow_ir": {"properties": [{"node_id": "A", "changes": {"label": "A+"}}]}})

    assert result["ok"] is True
    assert result["compiled"]["properties"] == [{"node_id": "n_a", "changes": {"label": "A+"}}]
    assert result["constraints"]["properties"] == [{"node_id": "n_a", "changes": {"label": "A+"}}]


def test_edge_property_compiles_to_constraints_without_relation_type_pollution() -> None:
    _, result = shadow(
        {
            "status": "resolved",
            "workflow_ir": {
                "edge_properties": [
                    {
                        "source": "A",
                        "target": "Review",
                        "relation_type": "success",
                        "changes": {"branch_label": "approved", "condition": {"expression": "score >= 90"}},
                    }
                ]
            },
        }
    )

    assert result["ok"] is True
    assert result["constraints"]["edge_properties"] == [
        {"edge_id": "e_a_review_success", "changes": {"branch_label": "approved", "condition": {"expression": "score >= 90"}}}
    ]
    assert result["constraints"]["required_relations"] == []


def test_relation_metadata_auxiliary_relations_compile_to_constraints() -> None:
    _, result = shadow(
        {
            "status": "resolved",
            "workflow_ir": {"auxiliary_relations": [{"source": "A", "relation_type": "success", "targets": ["Review"]}]},
        }
    )

    assert result["ok"] is True
    assert ("n_a", "n_review", "success") in relations(result["constraints"]["required_relations"])
    assert all(row.get("label") == "success" for row in result["constraints"]["required_relations"])


def test_auxiliary_relation_can_target_declared_new_ref_and_inherit_metadata() -> None:
    _, result = shadow(
        {
            "status": "resolved",
            "workflow_ir": {
                "control_flow_regions": [
                    {
                        "relation_type": "",
                        "structure": {
                            "type": "sequence",
                            "items": [
                                {"type": "node", "ref": "confirm", "node_type": "task", "label": "Confirm"},
                                {"type": "node", "id": "Review"},
                            ],
                        },
                    }
                ],
                "auxiliary_relations": [{"source": "A", "relation_type": "success", "targets": ["confirm"]}],
            },
        }
    )

    assert result["ok"] is True
    required = result["constraints"]["required_relations"]
    assert {"source": "n_a", "target": "confirm", "label": "success", "id": "e_a_review_success", "branch_label": "old", "condition": {"expression": "score > 0"}} in required
    assert {"source": "confirm", "target": "n_review", "label": ""} in required
    assert {"source": "n_a", "target": "n_review", "label": "success"} in result["constraints"]["forbidden_relations"]


def test_auxiliary_relation_unknown_new_ref_is_rejected() -> None:
    grounded, result = shadow(
        {
            "status": "resolved",
            "workflow_ir": {"auxiliary_relations": [{"source": "A", "relation_type": "success", "targets": ["missing_ref"]}]},
        }
    )

    assert result["ok"] is False
    assert result["stage"] == "validation"
    assert result["validation"]["deterministic_ready"] is False
    assert grounded["grounding"]["issues"][0]["kind"] == "missing_node_reference"


def test_absent_entity_compiles_to_constraints() -> None:
    _, result = shadow({"status": "resolved", "workflow_ir": {"absent_entities": [{"node_id": "Review"}]}})

    assert result["ok"] is True
    assert result["constraints"]["absent_entities"] == [{"node_id": "n_review"}]
    forbidden = relations(result["constraints"]["forbidden_relations"])
    assert ("n_a", "n_review", "success") in forbidden
    assert ("n_b", "n_review", "failure") in forbidden
    assert "n_review" not in result["constraints"]["preserve"]["node_ids"]


def test_not_ready_validation_result_refuses_to_call_compile(monkeypatch: pytest.MonkeyPatch) -> None:
    grounded = ground_understand_semantic_spec(
        {
            "status": "partial",
            "workflow_ir": {"properties": [{"node_id": "A", "changes": {"label": "A+"}}]},
            "semantic_goals": [{"goal_type": "placement", "description": "still unresolved"}],
        },
        graph(),
    )

    def fail_compile(*_args: object, **_kwargs: object) -> dict:
        raise AssertionError("compile_ir should not be called")

    monkeypatch.setattr(core, "compile_ir", fail_compile)
    result = compile_grounded_semantic_spec(grounded, graph())

    assert result["ok"] is False
    assert result["stage"] == "validation"
    assert result["compiled"] is None
    assert result["constraints"] is None


def test_ambiguous_boundary_blocks_deterministic_readiness() -> None:
    current_graph = graph()
    current_graph["nodes"].append({"id": "n_y", "type": "task", "label": "Y", "x": -100, "y": 100})
    current_graph["edges"].append({"id": "e_y_a", "source": "n_y", "target": "n_a", "label": ""})
    grounded = ground_understand_semantic_spec(
        {
            "status": "resolved",
            "workflow_ir": {
                "control_flow_regions": [
                    {"relation_type": "", "structure": {"type": "parallel", "join_mode": "all", "branches": [{"type": "node", "id": "A"}, {"type": "node", "id": "B"}]}}
                ]
            },
        },
        current_graph,
    )
    validation = validate_grounded_semantic_spec(grounded, current_graph)

    result = compile_grounded_semantic_spec(grounded, current_graph, validation)

    assert validation["valid"] is True
    assert validation["deterministic_ready"] is False
    assert validation["issues"][0]["kind"] == "ambiguous_structural_boundary"
    assert result["ok"] is False
    assert result["stage"] == "validation"
    assert result["issues"][0]["kind"] == "ambiguous_structural_boundary"


def test_shadow_path_does_not_call_old_unit_obligation_skeleton_or_ir_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    forbidden_names = [
        "requirement_local_facts",
        "form_spec_units",
        "requirement_conservation",
        "explicit_obligations",
        "required_obligations",
        "skeleton_from_obligations",
        "llm_requirements",
        "structural_participant_coverage",
        "obligation_coverage",
    ]

    def fail_old_layer(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("old intermediate layer should not be called")

    for name in forbidden_names:
        monkeypatch.setattr(core, name, fail_old_layer)

    _, result = shadow({"status": "resolved", "workflow_ir": {"properties": [{"node_id": "A", "changes": {"label": "A+"}}]}})

    assert result["ok"] is True
    assert result["stage"] == "constraints"


def test_shadow_path_does_not_call_planner_or_executor() -> None:
    _, result = shadow({"status": "resolved", "workflow_ir": {"properties": [{"node_id": "A", "changes": {"label": "A+"}}]}})

    assert result["ok"] is True
    assert "plan" not in result
    assert "mutation" not in result
    assert "version" not in result
