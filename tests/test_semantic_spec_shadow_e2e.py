from __future__ import annotations

from src.constraint_planner import plan_constraints
from src.mvp22 import execute_deterministic
from src.workflow_ir_core import compile_grounded_semantic_spec, ground_understand_semantic_spec


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


def edge_set(value: dict) -> set[tuple[str, str, str]]:
    return {(edge["source"], edge["target"], edge.get("label", "")) for edge in value["edges"]}


def node_by_id(value: dict, node_id: str) -> dict:
    return next(node for node in value["nodes"] if node["id"] == node_id)


def edge_by_id(value: dict, edge_id: str) -> dict:
    return next(edge for edge in value["edges"] if edge["id"] == edge_id)


def execute_shadow(spec: dict, current_graph: dict | None = None) -> dict:
    current_graph = current_graph or graph()
    grounded = ground_understand_semantic_spec(spec, current_graph)
    shadow = compile_grounded_semantic_spec(grounded, current_graph)
    assert shadow["validation"]["valid"] is True
    assert shadow["validation"]["deterministic_ready"] is True
    assert shadow["constraints"] is not None
    planned = plan_constraints(current_graph, shadow["constraints"])
    assert planned["supported"] is True
    executed = execute_deterministic(
        current_graph,
        {"core_node_ids": [], "core_edge_ids": []},
        {"incoming": [], "outgoing": []},
        planned["steps"],
        planned["mutation_authorization"],
    )
    assert executed.error is None
    assert executed.graph is not None
    return {"grounded": grounded, "shadow": shadow, "planned": planned, "final_graph": executed.graph}


def test_sequence_reorder_final_graph_reconnects_boundary() -> None:
    result = execute_shadow(
        {
            "status": "resolved",
            "workflow_ir": {
                "control_flow_regions": [
                    {"relation_type": "", "structure": {"type": "sequence", "items": [{"type": "node", "id": "B"}, {"type": "node", "id": "A"}]}}
                ]
            },
        }
    )

    final_edges = edge_set(result["final_graph"])

    assert ("n_b", "n_a", "") in final_edges
    assert ("n_a", "n_b", "") not in final_edges
    assert ("n_x", "n_b", "") in final_edges
    assert ("n_a", "n_c", "") in final_edges
    assert ("n_x", "n_a", "") not in final_edges
    assert ("n_b", "n_c", "") not in final_edges


def test_standalone_parallel_final_graph_satisfies_parallel_semantics() -> None:
    result = execute_shadow(
        {
            "status": "resolved",
            "workflow_ir": {
                "control_flow_regions": [
                    {"relation_type": "", "structure": {"type": "parallel", "join_mode": "all", "branches": [{"type": "node", "id": "A"}, {"type": "node", "id": "B"}]}}
                ]
            },
        }
    )

    final_edges = edge_set(result["final_graph"])

    assert {("n_x", "n_a", ""), ("n_x", "n_b", ""), ("n_a", "n_c", ""), ("n_b", "n_c", "")} <= final_edges
    assert ("n_a", "n_b", "") not in final_edges
    assert node_by_id(result["final_graph"], "n_c")["join"] == {"mode": "all"}


def test_nested_parallel_merge_final_graph_reconnects_upstream_fanout() -> None:
    result = execute_shadow(
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

    final_edges = edge_set(result["final_graph"])

    assert {("n_x", "n_a", ""), ("n_x", "n_b", ""), ("n_a", "n_c", ""), ("n_b", "n_c", "")} <= final_edges
    assert ("n_a", "n_b", "") not in final_edges
    assert node_by_id(result["final_graph"], "n_c")["join"] == {"mode": "all"}


def test_mixed_topology_final_graph_reconnects_upstream_fanout() -> None:
    result = execute_shadow(
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

    final_edges = edge_set(result["final_graph"])

    assert {("n_x", "n_a", ""), ("n_x", "n_b", ""), ("n_a", "n_c", ""), ("n_b", "n_c", ""), ("n_c", "n_d", "")} <= final_edges
    assert ("n_a", "n_b", "") not in final_edges
    assert node_by_id(result["final_graph"], "n_c")["join"] == {"mode": "all"}


def test_topology_edit_preserves_metadata_relations() -> None:
    result = execute_shadow(
        {
            "status": "resolved",
            "workflow_ir": {
                "control_flow_regions": [
                    {"relation_type": "", "structure": {"type": "parallel", "join_mode": "all", "branches": [{"type": "node", "id": "A"}, {"type": "node", "id": "B"}]}}
                ]
            },
        }
    )

    final_edges = edge_set(result["final_graph"])

    assert ("n_a", "n_review", "success") in final_edges
    assert ("n_b", "n_review", "failure") in final_edges


def test_ambiguous_boundary_stops_at_validation() -> None:
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
    shadow = compile_grounded_semantic_spec(grounded, current_graph)

    assert shadow["ok"] is False
    assert shadow["stage"] == "validation"
    assert shadow["issues"][0]["kind"] == "ambiguous_structural_boundary"


def test_node_property_final_graph_matches_spec() -> None:
    result = execute_shadow({"status": "resolved", "workflow_ir": {"properties": [{"node_id": "A", "changes": {"label": "A+"}}]}})

    assert node_by_id(result["final_graph"], "n_a")["label"] == "A+"


def test_edge_property_final_graph_matches_spec() -> None:
    result = execute_shadow(
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

    edge = edge_by_id(result["final_graph"], "e_a_review_success")
    assert edge["branch_label"] == "approved"
    assert edge["condition"] == {"expression": "score >= 90"}
    assert edge["label"] == "success"


def test_auxiliary_relation_metadata_final_graph_matches_spec() -> None:
    result = execute_shadow(
        {
            "status": "resolved",
            "workflow_ir": {"auxiliary_relations": [{"source": "A", "relation_type": "success", "targets": ["Review"]}]},
        }
    )

    assert ("n_a", "n_review", "success") in edge_set(result["final_graph"])


def test_absent_entity_final_graph_matches_spec() -> None:
    result = execute_shadow({"status": "resolved", "workflow_ir": {"absent_entities": [{"node_id": "Review"}]}})

    assert "n_review" not in {node["id"] for node in result["final_graph"]["nodes"]}
    assert all(edge["source"] != "n_review" and edge["target"] != "n_review" for edge in result["final_graph"]["edges"])


def test_new_entity_after_parallel_uses_one_concrete_identity_for_topology_and_join() -> None:
    current_graph = {
        "nodes": [
            {"id": "submit", "type": "start", "label": "提交", "x": 0, "y": 0},
            {"id": "audit", "type": "task", "label": "审计", "x": 100, "y": -60},
            {"id": "risk", "type": "task", "label": "风控", "x": 100, "y": 60},
            {"id": "publish", "type": "end", "label": "发布", "x": 300, "y": 0, "join": {"mode": "all"}},
        ],
        "edges": [
            {"id": "e_submit_audit", "source": "submit", "target": "audit", "label": ""},
            {"id": "e_submit_risk", "source": "submit", "target": "risk", "label": ""},
            {"id": "e_audit_publish", "source": "audit", "target": "publish", "label": ""},
            {"id": "e_risk_publish", "source": "risk", "target": "publish", "label": ""},
        ],
    }
    result = execute_shadow(
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
                                    "branches": [{"type": "node", "id": "audit"}, {"type": "node", "id": "risk"}],
                                },
                                {"type": "node", "ref": "compliance_confirm", "node_type": "task", "label": "合规确认"},
                                {"type": "node", "id": "publish"},
                            ],
                        },
                    }
                ]
            },
        },
        current_graph,
    )

    new_node_id = result["planned"]["entity_ids"]["compliance_confirm"]
    final_edges = edge_set(result["final_graph"])

    assert new_node_id != "compliance_confirm"
    assert {("audit", new_node_id, ""), ("risk", new_node_id, ""), (new_node_id, "publish", "")} <= final_edges
    assert ("audit", "publish", "") not in final_edges
    assert ("risk", "publish", "") not in final_edges
    assert node_by_id(result["final_graph"], new_node_id)["join"] == {"mode": "all"}
    assert all("compliance_confirm" not in str(step) for step in result["planned"]["steps"])


def test_relation_aware_insert_new_ref_preserves_success_metadata_on_rewritten_edge() -> None:
    result = execute_shadow(
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

    new_node_id = result["planned"]["entity_ids"]["confirm"]
    final_edges = edge_set(result["final_graph"])

    assert ("n_a", new_node_id, "success") in final_edges
    assert (new_node_id, "n_review", "") in final_edges
    assert ("n_a", "n_review", "success") not in final_edges
    rewritten = next(edge for edge in result["final_graph"]["edges"] if edge["source"] == "n_a" and edge["target"] == new_node_id)
    assert rewritten["label"] == "success"
    assert rewritten["branch_label"] == "old"
    assert rewritten["condition"] == {"expression": "score > 0"}
    continuation = next(edge for edge in result["final_graph"]["edges"] if edge["source"] == new_node_id and edge["target"] == "n_review")
    assert continuation.get("label", "") == ""
    assert "branch_label" not in continuation
    assert "condition" not in continuation
    assert all("confirm" not in str(step) for step in result["planned"]["steps"])


def test_relation_aware_incoming_boundary_inherits_metadata_for_new_node_before_anchor() -> None:
    current_graph = {
        "nodes": [
            {"id": "risk", "type": "task", "label": "Risk", "x": 0, "y": 0},
            {"id": "ship", "type": "task", "label": "Ship", "x": 100, "y": 0},
            {"id": "notify", "type": "task", "label": "Notify", "x": 200, "y": 0},
        ],
        "edges": [
            {
                "id": "e_risk_ship_success",
                "source": "risk",
                "target": "ship",
                "label": "success",
                "branch_label": "ok",
                "condition": {"expression": "risk == low"},
            },
            {"id": "e_ship_notify", "source": "ship", "target": "notify", "label": ""},
        ],
    }
    result = execute_shadow(
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
                                {"type": "node", "id": "ship"},
                            ],
                        },
                    }
                ]
            },
        },
        current_graph,
    )

    new_node_id = result["planned"]["entity_ids"]["confirm"]
    final_edges = edge_set(result["final_graph"])

    assert ("risk", new_node_id, "success") in final_edges
    assert (new_node_id, "ship", "") in final_edges
    assert ("ship", "notify", "") in final_edges
    assert ("risk", "ship", "success") not in final_edges
    rewritten = next(edge for edge in result["final_graph"]["edges"] if edge["source"] == "risk" and edge["target"] == new_node_id)
    assert rewritten["branch_label"] == "ok"
    assert rewritten["condition"] == {"expression": "risk == low"}
    continuation = next(edge for edge in result["final_graph"]["edges"] if edge["source"] == new_node_id and edge["target"] == "ship")
    assert continuation.get("label", "") == ""
    assert "branch_label" not in continuation
    assert "condition" not in continuation


def test_relation_aware_internal_split_inherits_metadata_for_new_node_between_existing_anchors() -> None:
    result = execute_shadow(
        {
            "status": "resolved",
            "workflow_ir": {
                "control_flow_regions": [
                    {
                        "relation_type": "",
                        "structure": {
                            "type": "sequence",
                            "items": [
                                {"type": "node", "id": "A"},
                                {"type": "node", "ref": "confirm", "node_type": "task", "label": "Confirm"},
                                {"type": "node", "id": "Review"},
                            ],
                        },
                    }
                ]
            },
        }
    )

    new_node_id = result["planned"]["entity_ids"]["confirm"]
    final_edges = edge_set(result["final_graph"])

    assert ("n_a", new_node_id, "success") in final_edges
    assert ("n_a", new_node_id, "") not in final_edges
    assert (new_node_id, "n_review", "") in final_edges
    assert ("n_a", "n_review", "success") not in final_edges
    rewritten = next(edge for edge in result["final_graph"]["edges"] if edge["source"] == "n_a" and edge["target"] == new_node_id)
    assert rewritten["branch_label"] == "old"
    assert rewritten["condition"] == {"expression": "score > 0"}


def test_relation_aware_outgoing_boundary_inherits_metadata_for_new_node_after_anchor() -> None:
    current_graph = {
        "nodes": [
            {"id": "risk", "type": "task", "label": "Risk", "x": 0, "y": 0},
            {"id": "ship", "type": "task", "label": "Ship", "x": 100, "y": 0},
            {"id": "notify", "type": "task", "label": "Notify", "x": 200, "y": 0},
        ],
        "edges": [
            {
                "id": "e_risk_ship_success",
                "source": "risk",
                "target": "ship",
                "label": "success",
                "branch_label": "ok",
                "condition": {"expression": "risk == low"},
            },
            {"id": "e_ship_notify", "source": "ship", "target": "notify", "label": ""},
        ],
    }
    result = execute_shadow(
        {
            "status": "resolved",
            "workflow_ir": {
                "control_flow_regions": [
                    {
                        "relation_type": "",
                        "structure": {
                            "type": "sequence",
                            "items": [
                                {"type": "node", "id": "risk"},
                                {"type": "node", "ref": "confirm", "node_type": "task", "label": "Confirm"},
                            ],
                        },
                    }
                ]
            },
        },
        current_graph,
    )

    new_node_id = result["planned"]["entity_ids"]["confirm"]
    final_edges = edge_set(result["final_graph"])

    assert ("risk", new_node_id, "success") in final_edges
    assert (new_node_id, "ship", "") in final_edges
    assert ("risk", "ship", "success") not in final_edges
    rewritten = next(edge for edge in result["final_graph"]["edges"] if edge["source"] == "risk" and edge["target"] == new_node_id)
    assert rewritten["branch_label"] == "ok"
    assert rewritten["condition"] == {"expression": "risk == low"}


def test_relation_aware_incoming_boundary_ambiguity_blocks_compile() -> None:
    current_graph = graph()
    current_graph["nodes"].append({"id": "n_y", "type": "task", "label": "Y", "x": 0, "y": 100})
    current_graph["edges"].append({"id": "e_y_review_retry", "source": "n_y", "target": "n_review", "label": "retry"})
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
                                {"type": "node", "ref": "confirm", "node_type": "task", "label": "Confirm"},
                                {"type": "node", "id": "Review"},
                            ],
                        },
                    }
                ]
            },
        },
        current_graph,
    )
    shadow = compile_grounded_semantic_spec(grounded, current_graph)

    assert shadow["ok"] is False
    assert shadow["stage"] == "validation"
    assert shadow["issues"][0]["kind"] == "ambiguous_structural_boundary"


def placement_graph() -> dict:
    return {
        "nodes": [
            {"id": "p", "type": "task", "label": "P", "x": 0, "y": 0},
            {"id": "a", "type": "task", "label": "A", "x": 100, "y": 0},
            {"id": "s", "type": "task", "label": "S", "x": 200, "y": 0},
            {"id": "x", "type": "task", "label": "X", "x": 100, "y": 100},
        ],
        "edges": [
            {"id": "e_p_a", "source": "p", "target": "a", "label": ""},
            {"id": "e_a_s", "source": "a", "target": "s", "label": ""},
        ],
    }


def test_insert_new_node_before_anchor_final_graph() -> None:
    result = execute_shadow(
        {
            "status": "resolved",
            "workflow_ir": {
                "control_flow_regions": [
                    {
                        "relation_type": "",
                        "structure": {
                            "type": "sequence",
                            "items": [
                                {"type": "node", "id": "p"},
                                {"type": "node", "ref": "inserted", "node_type": "task", "label": "Inserted"},
                                {"type": "node", "id": "a"},
                                {"type": "node", "id": "s"},
                            ],
                        },
                    }
                ]
            },
        },
        placement_graph(),
    )

    new_node_id = result["planned"]["entity_ids"]["inserted"]
    final_edges = edge_set(result["final_graph"])

    assert {("p", new_node_id, ""), (new_node_id, "a", ""), ("a", "s", "")} <= final_edges
    assert ("p", "a", "") not in final_edges


def test_insert_new_node_after_anchor_final_graph() -> None:
    result = execute_shadow(
        {
            "status": "resolved",
            "workflow_ir": {
                "control_flow_regions": [
                    {
                        "relation_type": "",
                        "structure": {
                            "type": "sequence",
                            "items": [
                                {"type": "node", "id": "p"},
                                {"type": "node", "id": "a"},
                                {"type": "node", "ref": "inserted", "node_type": "task", "label": "Inserted"},
                                {"type": "node", "id": "s"},
                            ],
                        },
                    }
                ]
            },
        },
        placement_graph(),
    )

    new_node_id = result["planned"]["entity_ids"]["inserted"]
    final_edges = edge_set(result["final_graph"])

    assert {("p", "a", ""), ("a", new_node_id, ""), (new_node_id, "s", "")} <= final_edges
    assert ("a", "s", "") not in final_edges


def test_relocate_existing_node_before_anchor_final_graph() -> None:
    result = execute_shadow(
        {
            "status": "resolved",
            "workflow_ir": {
                "control_flow_regions": [
                    {
                        "relation_type": "",
                        "structure": {
                            "type": "sequence",
                            "items": [
                                {"type": "node", "id": "p"},
                                {"type": "node", "id": "x"},
                                {"type": "node", "id": "a"},
                                {"type": "node", "id": "s"},
                            ],
                        },
                    }
                ]
            },
        },
        placement_graph(),
    )

    final_edges = edge_set(result["final_graph"])

    assert {("p", "x", ""), ("x", "a", ""), ("a", "s", "")} <= final_edges
    assert ("p", "a", "") not in final_edges


def test_relocate_existing_node_after_anchor_final_graph() -> None:
    result = execute_shadow(
        {
            "status": "resolved",
            "workflow_ir": {
                "control_flow_regions": [
                    {
                        "relation_type": "",
                        "structure": {
                            "type": "sequence",
                            "items": [
                                {"type": "node", "id": "p"},
                                {"type": "node", "id": "a"},
                                {"type": "node", "id": "x"},
                                {"type": "node", "id": "s"},
                            ],
                        },
                    }
                ]
            },
        },
        placement_graph(),
    )

    final_edges = edge_set(result["final_graph"])

    assert {("p", "a", ""), ("a", "x", ""), ("x", "s", "")} <= final_edges
    assert ("a", "s", "") not in final_edges
