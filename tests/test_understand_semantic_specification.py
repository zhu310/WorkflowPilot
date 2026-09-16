from __future__ import annotations

from src.workflow_ir_core import constraints_from_compiled, normalize_understand_semantic_spec, validate_understand_semantic_spec


def graph() -> dict:
    return {
        "nodes": [
            {"id": "a", "type": "task", "label": "A", "x": 0, "y": 0},
            {"id": "b", "type": "task", "label": "B", "x": 160, "y": 0},
            {"id": "c", "type": "task", "label": "C", "x": 320, "y": 0},
            {"id": "d", "type": "task", "label": "D", "x": 480, "y": 0},
        ],
        "edges": [
            {"id": "e_ab", "source": "a", "target": "b", "label": ""},
            {"id": "e_ac", "source": "a", "target": "c", "label": ""},
            {"id": "e_ac_failure", "source": "a", "target": "c", "label": "failure"},
            {"id": "e_cb_retry", "source": "c", "target": "b", "label": "retry"},
            {
                "id": "e_ad_success",
                "source": "a",
                "target": "d",
                "label": "success",
                "branch_label": "old branch",
                "condition": {"expression": "score > 0.8"},
            },
        ],
    }


def assert_valid(spec: dict) -> dict:
    valid, error = validate_understand_semantic_spec(spec, graph())
    assert valid is True, error
    return normalize_understand_semantic_spec(spec)


def test_resolved_topology_reuses_workflow_ir_composition() -> None:
    normalized = assert_valid(
        {
            "status": "resolved",
            "workflow_ir": {
                "control_flow_regions": [
                    {
                        "relation_type": "",
                        "structure": {
                            "type": "sequence",
                            "items": [
                                {
                                    "type": "parallel",
                                    "join_mode": "all",
                                    "branches": [{"type": "node", "id": "a"}, {"type": "node", "id": "b"}],
                                },
                                {"type": "node", "id": "c"},
                            ],
                        },
                    }
                ]
            },
        }
    )

    structure = normalized["workflow_ir"]["control_flow_regions"][0]["structure"]
    assert normalized["status"] == "resolved"
    assert structure["items"][0]["type"] == "parallel"
    assert structure["items"][1] == {"type": "node", "id": "c"}


def test_canonical_sequence_json_is_preserved() -> None:
    normalized = assert_valid(
        {
            "status": "resolved",
            "workflow_ir": {
                "control_flow_regions": [
                    {
                        "relation_type": "",
                        "structure": {"type": "sequence", "items": [{"type": "node", "id": "b"}, {"type": "node", "id": "a"}]},
                    }
                ]
            },
        }
    )

    assert normalized["workflow_ir"]["control_flow_regions"] == [
        {"relation_type": "", "structure": {"type": "sequence", "items": [{"type": "node", "id": "b"}, {"type": "node", "id": "a"}]}}
    ]


def test_canonical_parallel_json_is_preserved() -> None:
    normalized = assert_valid(
        {
            "status": "resolved",
            "workflow_ir": {
                "control_flow_regions": [
                    {
                        "relation_type": "",
                        "structure": {
                            "type": "parallel",
                            "join_mode": "all",
                            "branches": [{"type": "node", "id": "a"}, {"type": "node", "id": "b"}],
                        },
                    }
                ]
            },
        }
    )

    assert normalized["workflow_ir"]["control_flow_regions"][0]["structure"] == {
        "type": "parallel",
        "branches": [{"type": "node", "id": "a"}, {"type": "node", "id": "b"}],
        "join_mode": "all",
    }


def test_malformed_non_empty_control_flow_region_is_not_silently_dropped() -> None:
    spec = {
        "status": "resolved",
        "workflow_ir": {
            "control_flow_regions": [
                {"type": "sequence", "nodes": [{"type": "node", "id": "a"}, {"type": "node", "id": "b"}]}
            ]
        },
    }

    normalized = normalize_understand_semantic_spec(spec)
    valid, error = validate_understand_semantic_spec(spec, graph())

    assert normalized["workflow_ir"]["control_flow_regions"] == []
    assert {issue["kind"] for issue in normalized["normalization_issues"]} == {
        "invalid_control_flow_region_shape",
        "semantic_spec_normalization_loss",
    }
    assert valid is False
    assert error == "semantic spec normalization loss"


def test_resolved_property_edit_does_not_require_topology_structure() -> None:
    normalized = assert_valid(
        {
            "status": "resolved",
            "workflow_ir": {
                "properties": [{"node_id": "a", "changes": {"label": "Review request"}}],
            },
        }
    )

    assert normalized["workflow_ir"]["control_flow_regions"] == []
    assert normalized["workflow_ir"]["properties"] == [{"node_id": "a", "changes": {"label": "Review request"}}]


def test_resolved_relation_metadata_edit_reuses_existing_workflow_ir_fields() -> None:
    normalized = assert_valid(
        {
            "status": "resolved",
            "workflow_ir": {
                "auxiliary_relations": [
                    {"source": "a", "relation_type": "failure", "targets": ["c"]},
                    {"source": "c", "relation_type": "retry", "targets": ["b"]},
                    {"source": "a", "relation_type": "success", "targets": ["d"]},
                ],
                "edge_properties": [
                    {
                        "edge_id": "e_ad_success",
                        "changes": {
                            "branch_label": "high score",
                            "condition": {"expression": "score >= 90", "language": "python"},
                        },
                    }
                ],
            },
        }
    )

    assert {row["relation_type"] for row in normalized["workflow_ir"]["auxiliary_relations"]} == {
        "failure",
        "retry",
        "success",
    }
    assert normalized["workflow_ir"]["edge_properties"][0]["changes"]["branch_label"] == "high score"


def test_partial_request_can_keep_semantic_goal_without_forcing_ir() -> None:
    normalized = assert_valid(
        {
            "status": "partial",
            "semantic_goals": [
                {"goal_type": "parallelize", "description": "make A and B parallel, but local entry and merge still need context"}
            ],
        }
    )

    assert normalized["workflow_ir"] == {
        "control_flow_regions": [],
        "auxiliary_relations": [],
        "absent_entities": [],
        "properties": [],
        "edge_properties": [],
    }
    assert normalized["semantic_goals"][0]["description"].startswith("make A and B")


def test_clarification_can_stop_without_structural_generation() -> None:
    normalized = assert_valid(
        {
            "status": "clarification",
            "semantic_goals": [
                {"goal_type": "optimization", "description": "optimize the whole workflow has multiple possible goals"}
            ],
            "clarification": {
                "needed": True,
                "question": "Which optimization goal matters most?",
                "reason": "multiple reasonable business goals",
            },
        }
    )

    assert normalized["status"] == "clarification"
    assert normalized["workflow_ir"] == {
        "control_flow_regions": [],
        "auxiliary_relations": [],
        "absent_entities": [],
        "properties": [],
        "edge_properties": [],
    }
    assert normalized["clarification"]["needed"] is True


def test_status_resolved_does_not_mean_deterministic_ready() -> None:
    normalized = assert_valid(
        {
            "status": "resolved",
            "workflow_ir": {"properties": [{"node_id": "a", "changes": {"label": "A+"}}]},
        }
    )

    assert normalized["status"] == "resolved"
    assert "deterministic_ready" not in normalized


def test_complete_topology_edge_patch_fields_are_forbidden() -> None:
    spec = {
        "status": "resolved",
        "workflow_ir": {
            "control_flow_regions": [
                {
                    "relation_type": "",
                    "structure": {"type": "parallel", "branches": [{"type": "node", "id": "a"}, {"type": "node", "id": "b"}]},
                }
            ]
        },
        "required_edges": [{"source": "a", "target": "c"}],
        "forbidden_edges": [{"source": "a", "target": "b"}],
    }

    valid, error = validate_understand_semantic_spec(spec, graph())

    assert valid is False
    assert error == "complete topology edge patch fields are not allowed in understand semantic specification"


def test_explicit_single_relation_deletion_is_current_workflow_ir_gap() -> None:
    current_graph = graph()
    workflow_ir = {
        "control_flow_regions": [],
        "properties": [],
        "edge_properties": [],
        "absent_entities": [],
        "auxiliary_relations": [{"source": "a", "relation_type": "", "targets": ["c"]}],
    }

    constraints = constraints_from_compiled(
        current_graph,
        {
            "entities": [],
            "control_flow_regions": [],
            "properties": [],
            "edge_properties": [],
            "absent_entities": [],
            "auxiliary_relations": workflow_ir["auxiliary_relations"],
        },
    )

    assert constraints["forbidden_relations"] == [{"source": "a", "target": "b", "label": ""}]
    assert constraints["required_relations"] == [{"source": "a", "target": "c", "label": ""}]
