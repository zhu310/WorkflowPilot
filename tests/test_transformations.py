from __future__ import annotations

from copy import deepcopy

from src import mvp23
from src.mvp22 import execute_deterministic
from src.transformations import PARALLELIZE_SELECTED_CHAIN_TO_JOIN, compile_parallelize_selected_chain_to_join


def graph_for(branch_ids: list[str], join_id: str, join_type: str = "task") -> dict:
    nodes = [{"id": "start", "type": "start", "label": "Start", "x": 0, "y": 0}]
    nodes.extend({"id": node_id, "type": "task", "label": node_id, "x": 100 * (index + 1), "y": 0} for index, node_id in enumerate(branch_ids))
    nodes.extend(
        [
            {"id": join_id, "type": join_type, "label": join_id, "x": 100 * (len(branch_ids) + 1), "y": 0},
            {"id": "end", "type": "end", "label": "End", "x": 100 * (len(branch_ids) + 2), "y": 0},
        ]
    )
    pairs = [("start", branch_ids[0])]
    pairs.extend(zip(branch_ids, branch_ids[1:]))
    pairs.extend([(branch_ids[-1], join_id), (join_id, "end")])
    return {"nodes": nodes, "edges": [{"id": f"e{index}", "source": source, "target": target, "label": ""} for index, (source, target) in enumerate(pairs, 1)]}


def compile_and_execute(graph: dict, branch_ids: list[str], join_id: str) -> tuple[dict, dict]:
    scope = {"core_node_ids": [*branch_ids, join_id], "core_edge_ids": [edge["id"] for edge in graph["edges"] if edge["source"] in branch_ids or edge["target"] in branch_ids]}
    boundary = {"incoming": [{"external_node_id": "start", "internal_node_id": branch_ids[0], "existing_edge_id": "e1"}], "outgoing": [{"external_node_id": "end", "internal_node_id": join_id, "existing_edge_id": graph["edges"][-1]["id"]}]}
    operation = {"op": PARALLELIZE_SELECTED_CHAIN_TO_JOIN, "branch_node_ids": branch_ids, "join_node_id": join_id, "join_mode": "all"}
    compiled = compile_parallelize_selected_chain_to_join(graph, operation, scope, boundary)
    result = execute_deterministic(graph, scope, boundary, compiled["steps"])
    assert result.error is None
    assert result.graph is not None
    return compiled, result.graph


def pairs(graph: dict) -> set[tuple[str, str, str]]:
    return {(edge["source"], edge["target"], edge.get("label", "")) for edge in graph["edges"]}


def test_two_node_parallel_uses_minimum_diff() -> None:
    graph = graph_for(["material", "risk"], "approval")
    compiled, output = compile_and_execute(graph, ["material", "risk"], "approval")
    assert len(compiled["steps"]) == 4
    assert pairs(output) == {("start", "material", ""), ("start", "risk", ""), ("material", "approval", ""), ("risk", "approval", ""), ("approval", "end", "")}
    assert next(node for node in output["nodes"] if node["id"] == "approval")["join"] == {"mode": "all"}


def test_three_node_parallel_preserves_external_graph() -> None:
    graph = graph_for(["identity_check", "material_check", "risk_check"], "approval")
    compiled, output = compile_and_execute(graph, ["identity_check", "material_check", "risk_check"], "approval")
    assert len(compiled["steps"]) == 7
    assert ("start", "identity_check", "") in pairs(output)
    assert ("start", "material_check", "") in pairs(output)
    assert ("start", "risk_check", "") in pairs(output)
    assert ("approval", "end", "") in pairs(output)
    assert next(edge for edge in output["edges"] if edge["id"] == "e5") == next(edge for edge in graph["edges"] if edge["id"] == "e5")


def test_human_review_join_is_generic() -> None:
    graph = graph_for(["document_check", "policy_check"], "human_review", "human_review")
    _, output = compile_and_execute(graph, ["document_check", "policy_check"], "human_review")
    assert next(node for node in output["nodes"] if node["id"] == "human_review")["join"] == {"mode": "all"}
    assert ("document_check", "human_review", "") in pairs(output)
    assert ("policy_check", "human_review", "") in pairs(output)


def test_existing_correct_edges_are_not_recreated_or_duplicated() -> None:
    graph = graph_for(["material", "risk"], "approval")
    compiled, output = compile_and_execute(graph, ["material", "risk"], "approval")
    assert {step["edge_id"] for step in compiled["steps"] if step["op"] == "remove_edge"} == {"e2"}
    assert ("risk", "approval", "") not in {(step.get("source"), step.get("target"), step.get("label", "")) for step in compiled["steps"] if step["op"] == "add_edge"}
    output_pairs = [(edge["source"], edge["target"], edge.get("label", "")) for edge in output["edges"]]
    assert len(output_pairs) == len(set(output_pairs))


def test_ambiguous_topology_is_unsupported() -> None:
    graph = graph_for(["material", "risk"], "approval")
    ambiguous = deepcopy(graph)
    ambiguous["edges"].append({"id": "e_ambiguous", "source": "risk", "target": "material", "label": ""})
    operation = {"op": PARALLELIZE_SELECTED_CHAIN_TO_JOIN, "branch_node_ids": ["material", "risk"], "join_node_id": "approval", "join_mode": "all"}
    result = compile_parallelize_selected_chain_to_join(
        ambiguous,
        operation,
        {"core_node_ids": ["material", "risk", "approval"], "core_edge_ids": ["e2", "e3", "e_ambiguous"]},
        {"incoming": [], "outgoing": []},
    )
    assert result["supported"] is False
    assert result["reason"].startswith("unsupported_transformation:")


def test_unsupported_topology_returns_unsupported_without_atomic_fallback(monkeypatch) -> None:
    graph = graph_for(["material", "risk"], "approval")
    graph["nodes"].append({"id": "manual", "type": "human_review", "label": "manual", "x": 500, "y": 80})
    graph["edges"].append({"id": "e_side", "source": "material", "target": "manual", "label": ""})
    interpretation = {
        "status": "clear",
        "target": {"node_ids": ["material", "risk"], "edge_ids": [], "join_node_id": "approval"},
        "intent": {"kind": "parallelize_selected_chain"},
        "policy": {},
    }
    specification = {"status": "ready", "constraints": {"entities": [], "properties": [], "required_relations": [], "forbidden_relations": []}, "directives": [{"kind": "parallelize_selected_chain", "branches": ["material", "risk"], "upstream": "start", "join": "missing_join", "completion": "all"}]}
    monkeypatch.setattr(mvp23, "run_edit_interpretation", lambda *args, **kwargs: mvp23.JsonRun("", interpretation, None, False))
    monkeypatch.setattr(mvp23, "finalize_edit_specification", lambda *args, **kwargs: mvp23.JsonRun("", specification, None, False))

    result = mvp23.run_transaction(graph, 1, {"node_ids": ["material", "risk"], "edge_ids": []}, "parallelize the checks", allowed_transformations=["set_join"])
    assert result.status == "planning_error"
    assert result.planning_path == "constraint"
    assert result.final_graph is None
