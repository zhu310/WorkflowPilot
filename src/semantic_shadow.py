from __future__ import annotations

from copy import deepcopy
from typing import Any

from .constraint_planner import plan_constraints
from .mvp22 import execute_deterministic
from .workflow_ir_core import (
    compile_grounded_semantic_spec,
    graph_is_valid,
    ground_understand_semantic_spec,
    normalize_request_context,
    normalize_understand_semantic_spec,
)
from .workflow_ir_runtime import UNDERSTAND_SEMANTIC_SPECIFICATION_SYSTEM, _complete_json


def graph_diff(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    before_nodes = {node["id"]: node for node in before.get("nodes", []) if isinstance(node, dict) and isinstance(node.get("id"), str)}
    after_nodes = {node["id"]: node for node in after.get("nodes", []) if isinstance(node, dict) and isinstance(node.get("id"), str)}
    before_edges = {edge["id"]: edge for edge in before.get("edges", []) if isinstance(edge, dict) and isinstance(edge.get("id"), str)}
    after_edges = {edge["id"]: edge for edge in after.get("edges", []) if isinstance(edge, dict) and isinstance(edge.get("id"), str)}
    changed_nodes = []
    changed_edges = []
    for node_id in sorted(set(before_nodes) & set(after_nodes)):
        changes = {
            key: {"before": before_nodes[node_id].get(key), "after": after_nodes[node_id].get(key)}
            for key in sorted(set(before_nodes[node_id]) | set(after_nodes[node_id]))
            if before_nodes[node_id].get(key) != after_nodes[node_id].get(key)
        }
        if changes:
            changed_nodes.append({"id": node_id, "changes": changes})
    for edge_id in sorted(set(before_edges) & set(after_edges)):
        changes = {
            key: {"before": before_edges[edge_id].get(key), "after": after_edges[edge_id].get(key)}
            for key in sorted(set(before_edges[edge_id]) | set(after_edges[edge_id]))
            if before_edges[edge_id].get(key) != after_edges[edge_id].get(key)
        }
        if changes:
            changed_edges.append({"id": edge_id, "changes": changes})
    return {
        "added_nodes": [after_nodes[node_id] for node_id in sorted(set(after_nodes) - set(before_nodes))],
        "removed_nodes": [before_nodes[node_id] for node_id in sorted(set(before_nodes) - set(after_nodes))],
        "changed_nodes": changed_nodes,
        "added_edges": [after_edges[edge_id] for edge_id in sorted(set(after_edges) - set(before_edges))],
        "removed_edges": [before_edges[edge_id] for edge_id in sorted(set(before_edges) - set(after_edges))],
        "changed_edges": changed_edges,
    }


def _graph_summary(graph: dict[str, Any]) -> dict[str, Any]:
    return {
        "node_count": len(graph.get("nodes", [])) if isinstance(graph.get("nodes"), list) else 0,
        "edge_count": len(graph.get("edges", [])) if isinstance(graph.get("edges"), list) else 0,
        "node_ids": [node.get("id") for node in graph.get("nodes", []) if isinstance(node, dict)],
        "edge_ids": [edge.get("id") for edge in graph.get("edges", []) if isinstance(edge, dict)],
    }


def run_semantic_shadow(
    graph: dict[str, Any],
    instruction: str,
    selection: dict[str, Any] | None = None,
    request_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    before = deepcopy(graph)
    normalized_selection = selection if isinstance(selection, dict) else {"node_ids": [], "edge_ids": []}
    normalized_context = normalize_request_context(normalized_selection, request_context)
    trace: dict[str, Any] = {
        "enabled": True,
        "stage": "start",
        "instruction": instruction,
        "selection": normalized_selection,
        "request_context": normalized_context,
        "before_graph": before,
        "graph_context": _graph_summary(before),
        "semantic_spec": None,
        "grounded_spec": None,
        "validation": None,
        "compiler_input": None,
        "compiled": None,
        "constraints": None,
        "planner": None,
        "executor": None,
        "final_shadow_graph": None,
        "graph_diff": None,
    }
    if not graph_is_valid(before):
        trace.update({"stage": "input", "failure_stage": "input", "failure_reason": "invalid graph"})
        return trace

    payload = {
        "instruction": instruction,
        "graph": before,
        "selection": normalized_selection,
        "request_context": normalized_context,
        "conversation_history": normalized_context.get("conversation_history", []),
    }
    trace["stage"] = "understand"
    understand_run = _complete_json(UNDERSTAND_SEMANTIC_SPECIFICATION_SYSTEM, payload)
    trace["semantic_spec"] = {
        "raw_response": understand_run.raw_response,
        "parsed": understand_run.parsed_json,
        "parse_error": understand_run.parse_error,
        "normalized_from_fence": understand_run.normalized_from_fence,
    }
    if understand_run.parsed_json is None:
        trace.update({"failure_stage": "understand", "failure_reason": understand_run.parse_error or "semantic spec parse failed"})
        return trace

    normalized_spec = normalize_understand_semantic_spec(understand_run.parsed_json)
    trace["semantic_spec"]["normalized"] = normalized_spec

    trace["stage"] = "ground"
    grounded = ground_understand_semantic_spec(normalized_spec, before, normalized_selection)
    trace["grounded_spec"] = grounded

    trace["stage"] = "validation"
    shadow = compile_grounded_semantic_spec(grounded, before)
    trace["validation"] = shadow.get("validation")
    trace["compiler_input"] = {"workflow_ir": grounded.get("workflow_ir")}
    if shadow.get("stage") == "validation":
        trace.update({"failure_stage": "validation", "failure_reason": shadow.get("issues", [])})
        return trace

    trace["stage"] = "compile"
    trace["compiled"] = shadow.get("compiled")
    trace["constraints"] = shadow.get("constraints")
    if shadow.get("stage") == "compile":
        trace.update({"failure_stage": "compile", "failure_reason": shadow.get("issues", [])})
        return trace

    trace["stage"] = "planner"
    constraints = shadow.get("constraints")
    planner = plan_constraints(before, constraints if isinstance(constraints, dict) else {})
    trace["planner"] = planner
    if not planner.get("supported"):
        trace.update({"failure_stage": "planner", "failure_reason": planner.get("reason")})
        return trace

    trace["stage"] = "executor"
    execution = execute_deterministic(
        before,
        {"core_node_ids": [], "core_edge_ids": []},
        {"incoming": [], "outgoing": []},
        planner.get("steps", []),
        planner.get("mutation_authorization"),
    )
    trace["executor"] = {
        "steps": planner.get("steps", []),
        "mutation_authorization": planner.get("mutation_authorization", []),
        "error": execution.error,
        "applied_steps": execution.applied_steps,
    }
    if execution.graph is None or execution.error:
        trace.update({"failure_stage": "executor", "failure_reason": execution.error})
        return trace
    trace["final_shadow_graph"] = execution.graph
    trace["graph_diff"] = graph_diff(before, execution.graph)
    trace["stage"] = "success"
    return trace
