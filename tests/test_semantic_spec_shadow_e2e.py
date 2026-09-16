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


def test_ambiguous_boundary_stops_at_compile() -> None:
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
    assert shadow["stage"] == "compile"
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
