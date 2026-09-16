from __future__ import annotations

from src.workflow_ir_core import ground_understand_semantic_spec


def graph() -> dict:
    return {
        "nodes": [
            {"id": "n_x", "type": "task", "label": "X", "x": 0, "y": 0},
            {"id": "n_alpha", "type": "task", "label": "Alpha", "x": 100, "y": 0},
            {"id": "n_beta", "type": "task", "label": "Beta", "x": 200, "y": 0},
            {"id": "n_gamma", "type": "task", "label": "Gamma", "x": 300, "y": 0},
            {"id": "n_review", "type": "human_review", "label": "Review", "x": 400, "y": 0},
        ],
        "edges": [
            {"id": "e_x_alpha", "source": "n_x", "target": "n_alpha", "label": ""},
            {"id": "e_alpha_beta", "source": "n_alpha", "target": "n_beta", "label": ""},
            {"id": "e_beta_gamma", "source": "n_beta", "target": "n_gamma", "label": ""},
            {"id": "e_alpha_gamma_failure", "source": "n_alpha", "target": "n_gamma", "label": "failure"},
            {
                "id": "e_alpha_review_success",
                "source": "n_alpha",
                "target": "n_review",
                "label": "success",
                "branch_label": "old",
                "condition": {"expression": "score > 0"},
            },
        ],
    }


def duplicate_graph() -> dict:
    value = graph()
    value["nodes"].append({"id": "n_alpha_2", "type": "task", "label": "Alpha", "x": 500, "y": 0})
    return value


def test_simple_node_binding() -> None:
    grounded = ground_understand_semantic_spec(
        {"status": "resolved", "workflow_ir": {"properties": [{"node_id": "Alpha", "changes": {"label": "Alpha+"}}]}},
        graph(),
    )

    assert grounded["workflow_ir"]["properties"] == [{"node_id": "n_alpha", "changes": {"label": "Alpha+"}}]
    assert grounded["grounding"]["issues"] == []


def test_nested_structural_ir_binding_preserves_shape() -> None:
    grounded = ground_understand_semantic_spec(
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
                                    "branches": [
                                        {"type": "node", "id": "Alpha"},
                                        {"type": "node", "id": "Beta"},
                                    ],
                                },
                                {"type": "node", "id": "Gamma"},
                            ],
                        },
                    }
                ]
            },
        },
        graph(),
    )

    structure = grounded["workflow_ir"]["control_flow_regions"][0]["structure"]
    assert structure == {
        "type": "sequence",
        "items": [
            {
                "type": "parallel",
                "branches": [{"type": "node", "id": "n_alpha"}, {"type": "node", "id": "n_beta"}],
                "join_mode": "all",
            },
            {"type": "node", "id": "n_gamma"},
        ],
    }
    assert grounded["grounding"]["issues"] == []


def test_property_target_binding() -> None:
    grounded = ground_understand_semantic_spec(
        {"status": "resolved", "workflow_ir": {"properties": [{"node_id": "Beta", "changes": {"join": {"mode": "all"}}}]}},
        graph(),
    )

    assert grounded["workflow_ir"]["properties"] == [{"node_id": "n_beta", "changes": {"join": {"mode": "all"}}}]


def test_edge_property_target_binding_by_endpoints() -> None:
    grounded = ground_understand_semantic_spec(
        {
            "status": "resolved",
            "workflow_ir": {
                "edge_properties": [
                    {
                        "source": "Alpha",
                        "target": "Review",
                        "relation_type": "success",
                        "changes": {"branch_label": "approved", "condition": {"expression": "score >= 90"}},
                    }
                ]
            },
        },
        graph(),
    )

    assert grounded["workflow_ir"]["edge_properties"] == [
        {
            "edge_id": "e_alpha_review_success",
            "changes": {"branch_label": "approved", "condition": {"expression": "score >= 90"}},
        }
    ]


def test_auxiliary_relation_source_and_targets_binding() -> None:
    grounded = ground_understand_semantic_spec(
        {
            "status": "resolved",
            "workflow_ir": {"auxiliary_relations": [{"source": "Alpha", "relation_type": "failure", "targets": ["Gamma"]}]},
        },
        graph(),
    )

    assert grounded["workflow_ir"]["auxiliary_relations"] == [{"source": "n_alpha", "relation_type": "failure", "targets": ["n_gamma"]}]


def test_absent_entity_binding() -> None:
    grounded = ground_understand_semantic_spec(
        {"status": "resolved", "workflow_ir": {"absent_entities": [{"node_id": "Beta"}]}},
        graph(),
    )

    assert grounded["workflow_ir"]["absent_entities"] == [{"node_id": "n_beta"}]


def test_explicit_new_ref_is_not_bound_to_similar_existing_node() -> None:
    grounded = ground_understand_semantic_spec(
        {
            "status": "resolved",
            "workflow_ir": {
                "control_flow_regions": [
                    {
                        "relation_type": "",
                        "structure": {"type": "node", "ref": "new_human_review", "node_type": "human_review", "label": "Human Review"},
                    }
                ]
            },
        },
        graph(),
    )

    assert grounded["workflow_ir"]["control_flow_regions"][0]["structure"] == {
        "type": "node",
        "ref": "new_human_review",
        "node_type": "human_review",
        "label": "Human Review",
    }
    assert all(row["resolved_id"] != "n_review" for row in grounded["grounding"]["provenance"])


def test_same_name_ambiguous_nodes_return_grounding_issue() -> None:
    grounded = ground_understand_semantic_spec(
        {"status": "resolved", "workflow_ir": {"properties": [{"node_id": "Alpha", "changes": {"label": "A+"}}]}},
        duplicate_graph(),
    )

    assert grounded["workflow_ir"]["properties"] == []
    assert grounded["grounding"]["issues"][0]["kind"] == "ambiguous_node_reference"
    assert grounded["grounding"]["issues"][0]["candidates"] == ["n_alpha", "n_alpha_2"]


def test_missing_existing_reference_does_not_create_new_node() -> None:
    grounded = ground_understand_semantic_spec(
        {"status": "resolved", "workflow_ir": {"properties": [{"node_id": "Missing", "changes": {"label": "M"}}]}},
        graph(),
    )

    assert grounded["workflow_ir"]["properties"] == []
    assert grounded["grounding"]["issues"][0]["kind"] == "missing_node_reference"


def test_partial_spec_grounds_known_refs_and_keeps_semantic_goal() -> None:
    grounded = ground_understand_semantic_spec(
        {
            "status": "partial",
            "workflow_ir": {
                "control_flow_regions": [
                    {"relation_type": "", "structure": {"type": "parallel", "branches": [{"type": "node", "id": "Alpha"}, {"type": "node", "id": "Beta"}]}}
                ]
            },
            "semantic_goals": [{"goal_type": "parallelize", "description": "merge target is not determined"}],
        },
        graph(),
    )

    branches = grounded["workflow_ir"]["control_flow_regions"][0]["structure"]["branches"]
    assert branches == [{"type": "node", "id": "n_alpha"}, {"type": "node", "id": "n_beta"}]
    assert grounded["semantic_goals"] == [{"goal_type": "parallelize", "description": "merge target is not determined"}]


def test_clarification_spec_does_not_continue_structure_grounding() -> None:
    grounded = ground_understand_semantic_spec(
        {
            "status": "clarification",
            "workflow_ir": {"properties": [{"node_id": "Alpha", "changes": {"label": "A+"}}]},
            "clarification": {"needed": True, "question": "Which goal?", "reason": "open ended"},
        },
        graph(),
    )

    assert grounded["workflow_ir"] == {
        "control_flow_regions": [],
        "auxiliary_relations": [],
        "absent_entities": [],
        "properties": [],
        "edge_properties": [],
    }
    assert grounded["grounding"]["issues"] == []


def test_conversation_history_does_not_affect_grounding() -> None:
    spec = {"status": "resolved", "workflow_ir": {"properties": [{"node_id": "Alpha", "changes": {"label": "A+"}}]}}

    without_history = ground_understand_semantic_spec(spec, graph())
    with_history = ground_understand_semantic_spec(
        spec,
        graph(),
        request_context={"conversation_history": [{"role": "user", "content": "Alpha means Beta"}]},
    )

    assert with_history["workflow_ir"] == without_history["workflow_ir"]
    assert with_history["grounding"]["issues"] == without_history["grounding"]["issues"]


def test_ground_does_not_do_topology_completion() -> None:
    grounded = ground_understand_semantic_spec(
        {
            "status": "resolved",
            "workflow_ir": {
                "control_flow_regions": [
                    {
                        "relation_type": "",
                        "structure": {
                            "type": "sequence",
                            "items": [
                                {"type": "parallel", "branches": [{"type": "node", "id": "Alpha"}, {"type": "node", "id": "Beta"}]},
                                {"type": "node", "id": "Gamma"},
                            ],
                        },
                    }
                ]
            },
        },
        graph(),
    )

    text = str(grounded)
    assert "n_x" not in text
    assert "required_relations" not in text
    assert "forbidden_relations" not in text
    assert grounded["workflow_ir"]["control_flow_regions"][0]["structure"]["items"][0]["branches"] == [
        {"type": "node", "id": "n_alpha"},
        {"type": "node", "id": "n_beta"},
    ]
