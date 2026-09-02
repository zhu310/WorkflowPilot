from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any


@dataclass
class NormalizedJson:
    payload: dict[str, Any] | None
    parse_error: str | None
    normalized_text: str | None


@dataclass
class ExecutorResult:
    graph: dict[str, Any] | None
    error: str | None
    applied_steps: int


def parse_json_with_fence(raw_text: str) -> NormalizedJson:
    try:
        parsed = json.loads(raw_text)
        if not isinstance(parsed, dict):
            raise TypeError("top-level JSON is not an object")
        return NormalizedJson(payload=parsed, parse_error=None, normalized_text=raw_text)
    except (json.JSONDecodeError, TypeError):
        pass

    match = re.fullmatch(r"\s*```(?:json)?\s*(.*?)\s*```\s*", raw_text, flags=re.DOTALL | re.IGNORECASE)
    if not match:
        return NormalizedJson(payload=None, parse_error="JSON parse failed and no single fenced JSON block found", normalized_text=None)

    normalized = match.group(1)
    try:
        parsed = json.loads(normalized)
        if not isinstance(parsed, dict):
            raise TypeError("top-level JSON is not an object")
        return NormalizedJson(payload=parsed, parse_error=None, normalized_text=normalized)
    except (json.JSONDecodeError, TypeError) as error:
        return NormalizedJson(payload=None, parse_error=f"{type(error).__name__}: {error}", normalized_text=normalized)


def _node_map(graph: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {node["id"]: node for node in graph.get("nodes", [])}


def _edge_map(graph: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {edge["id"]: edge for edge in graph.get("edges", [])}


def _boundary_external_nodes(boundary: dict[str, Any]) -> set[str]:
    external: set[str] = set()
    for row in boundary.get("incoming", []):
        external.add(row["external_node_id"])
    for row in boundary.get("outgoing", []):
        external.add(row["external_node_id"])
    return external


def _boundary_edge_ids(boundary: dict[str, Any]) -> set[str]:
    edge_ids: set[str] = set()
    for row in boundary.get("incoming", []):
        if row.get("existing_edge_id"):
            edge_ids.add(row["existing_edge_id"])
    for row in boundary.get("outgoing", []):
        if row.get("existing_edge_id"):
            edge_ids.add(row["existing_edge_id"])
    return edge_ids


def _is_legal_edge_connection(source: str, target: str, core_nodes: set[str], boundary: dict[str, Any]) -> bool:
    external_nodes = _boundary_external_nodes(boundary)
    both_core = source in core_nodes and target in core_nodes
    inbound_boundary = source in external_nodes and target in core_nodes
    outbound_boundary = source in core_nodes and target in external_nodes
    return both_core or inbound_boundary or outbound_boundary


def _next_edge_id(edges: list[dict[str, Any]]) -> str:
    existing = {edge["id"] for edge in edges}
    index = 1
    while f"e_new_{index}" in existing:
        index += 1
    return f"e_new_{index}"


def _copy_graph(graph: dict[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(graph, ensure_ascii=False))


def apply_step(
    graph: dict[str, Any],
    resolved_scope: dict[str, Any],
    boundary: dict[str, Any],
    step: dict[str, Any],
    authorized: bool = False,
) -> tuple[dict[str, Any] | None, str | None]:
    next_graph = _copy_graph(graph)
    nodes = next_graph["nodes"]
    edges = next_graph["edges"]
    node_map = _node_map(next_graph)
    edge_map = _edge_map(next_graph)
    core_nodes = set(resolved_scope.get("core_node_ids", []))
    core_edges = set(resolved_scope.get("core_edge_ids", []))
    boundary_edge_ids = _boundary_edge_ids(boundary)

    op = step.get("op")
    if op == "move_node":
        node_id = step.get("node_id")
        if node_id not in node_map or (not authorized and node_id not in core_nodes):
            return None, f"operation rejected: move_node target {node_id!r} is outside resolved scope or missing"
        node_map[node_id]["x"] = step.get("x", node_map[node_id].get("x"))
        node_map[node_id]["y"] = step.get("y", node_map[node_id].get("y"))
        return next_graph, None

    if op == "update_node":
        node_id = step.get("node_id")
        if node_id not in node_map or (not authorized and node_id not in core_nodes):
            return None, f"operation rejected: update_node target {node_id!r} is outside resolved scope or missing"
        changes = step.get("changes", {})
        if not isinstance(changes, dict):
            return None, "operation rejected: update_node changes is not an object"
        for key, value in changes.items():
            node_map[node_id][key] = value
        return next_graph, None

    if op == "set_join":
        node_id = step.get("node_id")
        if node_id not in node_map or (not authorized and node_id not in core_nodes):
            return None, f"operation rejected: set_join target {node_id!r} is outside resolved scope or missing"
        node_map[node_id]["join"] = step.get("join")
        return next_graph, None

    if op == "add_node":
        node = step.get("node")
        if not isinstance(node, dict) or "id" not in node:
            return None, "operation rejected: add_node requires node with id"
        node_id = node["id"]
        if node_id in node_map:
            return None, f"operation rejected: add_node id {node_id!r} already exists"
        nodes.append(node)
        return next_graph, None

    if op == "remove_node":
        node_id = step.get("node_id")
        if node_id not in node_map or (not authorized and node_id not in core_nodes):
            return None, f"operation rejected: remove_node target {node_id!r} is outside resolved scope or missing"
        if any(edge.get("source") == node_id or edge.get("target") == node_id for edge in edges):
            return None, f"operation rejected: remove_node target {node_id!r} still has incident edges"
        next_graph["nodes"] = [node for node in nodes if node.get("id") != node_id]
        return next_graph, None

    if op == "remove_edge":
        edge_id = step.get("edge_id")
        if edge_id not in edge_map:
            return None, f"operation rejected: remove_edge target {edge_id!r} is missing"
        if not authorized and edge_id not in core_edges and edge_id not in boundary_edge_ids:
            return None, f"operation rejected: remove_edge target {edge_id!r} is outside resolved scope/boundary"
        next_graph["edges"] = [edge for edge in edges if edge["id"] != edge_id]
        return next_graph, None

    if op == "add_edge":
        source = step.get("source")
        target = step.get("target")
        if source not in node_map or target not in node_map:
            return None, f"operation rejected: add_edge endpoints {source!r}->{target!r} are missing"
        if not authorized and not _is_legal_edge_connection(source, target, core_nodes, boundary):
            return None, f"operation rejected: add_edge {source!r}->{target!r} is outside resolved scope/boundary"
        edge_id = step.get("edge_id") or _next_edge_id(edges)
        if edge_id in edge_map:
            return None, f"operation rejected: add_edge id {edge_id!r} already exists"
        edge = {
            key: value
            for key, value in step.items()
            if key not in {"op", "edge_id"}
        }
        edge["id"] = edge_id
        edge["source"] = source
        edge["target"] = target
        edge["label"] = step.get("label", "")
        edges.append(edge)
        return next_graph, None

    if op == "update_edge":
        edge_id = step.get("edge_id")
        if edge_id not in edge_map:
            return None, f"operation rejected: update_edge target {edge_id!r} is missing"
        if not authorized and edge_id not in core_edges and edge_id not in boundary_edge_ids:
            return None, f"operation rejected: update_edge target {edge_id!r} is outside resolved scope/boundary"
        changes = step.get("changes", {})
        if not isinstance(changes, dict):
            return None, "operation rejected: update_edge changes is not an object"
        for key, value in changes.items():
            edge_map[edge_id][key] = value
        return next_graph, None

    return None, f"operation rejected: unsupported op {op!r}"


def execute_deterministic(
    graph: dict[str, Any],
    resolved_scope: dict[str, Any],
    boundary: dict[str, Any],
    steps: list[dict[str, Any]],
    mutation_authorization: list[dict[str, Any]] | None = None,
) -> ExecutorResult:
    if mutation_authorization is not None:
        if len(mutation_authorization) != len(steps):
            return ExecutorResult(graph=None, error="mutation authorization does not cover every step", applied_steps=0)
        seen_authorized_steps: set[str] = set()
        for index, row in enumerate(mutation_authorization, start=1):
            if not isinstance(row, dict) or row.get("step") != steps[index - 1] or not row.get("constraint_rule"):
                return ExecutorResult(graph=None, error=f"step_{index}: mutation is not authorized by the plan", applied_steps=0)
            marker = json.dumps({"step": row.get("step"), "constraint_rule": row.get("constraint_rule"), "constraint": row.get("constraint")}, ensure_ascii=False, sort_keys=True)
            if marker in seen_authorized_steps:
                return ExecutorResult(graph=None, error=f"step_{index}: duplicate authorized mutation is rejected", applied_steps=0)
            seen_authorized_steps.add(marker)
    current = _copy_graph(graph)
    for index, step in enumerate(steps, start=1):
        current, error = apply_step(current, resolved_scope, boundary, step, authorized=mutation_authorization is not None)
        if error:
            return ExecutorResult(graph=None, error=f"step_{index}: {error}", applied_steps=index - 1)
        assert current is not None
    return ExecutorResult(graph=current, error=None, applied_steps=len(steps))
