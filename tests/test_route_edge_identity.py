from __future__ import annotations

from src.goal_lowering import lower_grounded_goal_to_workflow_ir
from src.semantic_goal import ground_semantic_goal_spec, validate_grounded_goal
from src.semantic_shadow import run_semantic_shadow


def branch_graph() -> dict:
    return {
        "nodes": [
            {"id": "check", "type": "task", "label": "Risk Check"},
            {"id": "approve", "type": "task", "label": "Approve"},
            {"id": "reject", "type": "task", "label": "Reject"},
        ],
        "edges": [
            {"id": "e_check_approve", "source": "check", "target": "approve", "label": "success", "branch_label": "Success"},
            {"id": "e_check_reject", "source": "check", "target": "approve", "label": "failure", "branch_label": "Failure"},
        ],
    }


def route_goal(source_text: str, target_text: str = "Reject", source_type: str = "existing_reference") -> dict:
    return {
        "status": "resolved",
        "semantic_goals": [
            {
                "id": "g1",
                "goal_category": "relation_goal",
                "text": "When Risk Check fails, route to Reject.",
                "objects": [
                    {"id": "route", "type": source_type, "text": source_text, "role": "source"},
                    {"id": "target", "type": "existing_reference", "text": target_text, "role": "target"},
                ],
                "relation_intents": [{"source": "route", "relation": "route_to", "target": "target"}],
            }
        ],
    }


def test_route_goal_text_resolves_failure_relation_without_guessing_success() -> None:
    graph = branch_graph()
    graph["edges"] = [edge for edge in graph["edges"] if edge["id"] != "e_check_reject"]

    grounded = ground_semantic_goal_spec(route_goal("Risk Check"), graph, {"node_ids": [], "edge_ids": []})
    assert validate_grounded_goal(grounded)["lowering_ready"] is True

    lowered = lower_grounded_goal_to_workflow_ir(grounded, graph)
    assert lowered["ok"] is True
    assert lowered["semantic_spec"]["workflow_ir"]["auxiliary_relations"] == [
        {"source": "check", "relation_type": "failure", "targets": ["reject"]}
    ]


def test_full_edge_reference_resolves_exact_failure_sibling() -> None:
    grounded = ground_semantic_goal_spec(route_goal("Risk Check fails", source_type="edge_reference"), branch_graph(), {"node_ids": [], "edge_ids": []})

    route_binding = grounded["semantic_goals"][0]["bindings"][0]
    assert route_binding["id"] == "e_check_reject"
    assert route_binding["graph_facts"]["label"] == "failure"


def test_node_phrase_without_edge_local_evidence_does_not_guess_sibling() -> None:
    spec = route_goal("Risk Check")
    spec["semantic_goals"][0]["text"] = "Route Risk Check to Reject."

    grounded = ground_semantic_goal_spec(spec, branch_graph(), {"node_ids": [], "edge_ids": []})
    validation = validate_grounded_goal(grounded)

    assert validation["lowering_ready"] is False
    assert any(issue["kind"] == "ambiguous_edge_reference" for issue in validation["issues"])


def test_conflicting_evidence_does_not_resolve_success_edge() -> None:
    graph = branch_graph()
    graph["edges"] = [edge for edge in graph["edges"] if edge["id"] == "e_check_approve"]

    grounded = ground_semantic_goal_spec(route_goal("Risk Check fails", source_type="edge_reference"), graph, {"node_ids": [], "edge_ids": []})
    binding = grounded["semantic_goals"][0]["bindings"][0]

    assert binding.get("id") != "e_check_approve"
    assert binding["graph_facts"]["source"] == "check"
    assert binding["graph_facts"]["label"] == "failure"


def test_selected_exact_edge_route_still_wins_for_deictic_reference() -> None:
    grounded = ground_semantic_goal_spec(route_goal("this branch", source_type="edge_reference"), branch_graph(), {"node_ids": [], "edge_ids": ["e_check_approve"]})

    route_binding = next(binding for binding in grounded["semantic_goals"][0]["bindings"] if binding["role"] == "route")
    assert route_binding["id"] == "e_check_approve"
    assert route_binding["graph_facts"]["label"] == "success"


def test_same_source_target_siblings_keep_identity() -> None:
    graph = branch_graph()
    graph["edges"][1]["target"] = "approve"

    grounded = ground_semantic_goal_spec(route_goal("Failure", source_type="edge_reference"), graph, {"node_ids": [], "edge_ids": []})

    route_binding = next(binding for binding in grounded["semantic_goals"][0]["bindings"] if binding["role"] == "route")
    assert route_binding["id"] == "e_check_reject"
    assert route_binding["graph_facts"]["label"] == "failure"


def test_parallel_node_level_route_sources_still_stop_before_execution() -> None:
    graph = {
        "nodes": [
            {"id": "x", "type": "task", "label": "Intake"},
            {"id": "a", "type": "task", "label": "Validate"},
            {"id": "b", "type": "task", "label": "Enrich"},
            {"id": "c", "type": "task", "label": "Review"},
        ],
        "edges": [
            {"id": "e_x_a", "source": "x", "target": "a", "label": ""},
            {"id": "e_a_b", "source": "a", "target": "b", "label": ""},
            {"id": "e_b_c", "source": "b", "target": "c", "label": ""},
        ],
    }
    spec = {
        "status": "resolved",
        "semantic_goals": [
            {
                "id": "g1",
                "goal_category": "relation_goal",
                "text": "Merge Validate and Enrich into Review",
                "objects": [
                    {"id": "v", "type": "existing_reference", "text": "Validate", "role": "source"},
                    {"id": "r", "type": "existing_reference", "text": "Review", "role": "target"},
                ],
                "relation_intents": [{"source": "v", "relation": "route_to", "target": "r"}],
            }
        ],
    }

    trace = run_semantic_shadow(graph, "Merge Validate and Enrich into Review", request_context={"semantic_goal_override": spec})

    assert trace["failure_stage"] == "validation"
    assert trace["final_shadow_graph"] is None
