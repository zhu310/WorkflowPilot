from __future__ import annotations

from typing import Any


def _ids(items: list[dict[str, Any]]) -> list[str]:
    return [item.get("id") for item in items]


def evaluate(initial: dict[str, Any], output: dict[str, Any] | None, rules: dict[str, Any], parse_error: str | None) -> dict[str, Any]:
    """Test-only deterministic observations. It never changes the model result."""
    checks: dict[str, bool] = {}
    details: list[str] = []
    if parse_error:
        return {"passed": False, "failure_category": "invalid_json", "checks": {"json_parsed": False}, "details": [parse_error]}

    assert output is not None
    nodes_are_list = isinstance(output.get("nodes"), list)
    edges_are_list = isinstance(output.get("edges"), list)
    nodes = output.get("nodes") if nodes_are_list else []
    edges = output.get("edges") if edges_are_list else []
    checks["nodes_is_array"] = nodes_are_list
    checks["edges_is_array"] = edges_are_list
    node_ids, edge_ids = _ids(nodes), _ids(edges)
    node_id_set = set(node_ids)
    checks["node_ids_unique"] = len(node_ids) == len(node_id_set) and None not in node_id_set
    checks["edge_ids_unique"] = len(edge_ids) == len(set(edge_ids)) and None not in set(edge_ids)
    checks["edges_reference_existing_nodes"] = all(edge.get("source") in node_id_set and edge.get("target") in node_id_set for edge in edges)

    output_by_id = {node.get("id"): node for node in nodes}
    initial_by_id = {node.get("id"): node for node in initial["nodes"]}
    protected = rules.get("unchanged_node_ids", [])
    checks["unrelated_nodes_preserved"] = all(output_by_id.get(node_id) == initial_by_id.get(node_id) for node_id in protected)
    output_edges_by_id = {edge.get("id"): edge for edge in edges}
    initial_edges_by_id = {edge.get("id"): edge for edge in initial["edges"]}
    protected_edges = rules.get("unchanged_edge_ids", [])
    checks["unrelated_edges_preserved"] = all(
        output_edges_by_id.get(edge_id) == initial_edges_by_id.get(edge_id) for edge_id in protected_edges
    )
    checks["node_count"] = len(nodes) == len(initial["nodes"]) + rules.get("node_delta", 0)
    checks["edge_count"] = len(edges) == len(initial["edges"]) + rules.get("edge_delta", 0)

    for expected in rules.get("expected_nodes", []):
        found = next((node for node in nodes if all(node.get(k) == v for k, v in expected.items())), None)
        checks[f"expected_node:{expected.get('id', expected.get('label', 'unknown'))}"] = found is not None
    edge_pairs = {(edge.get("source"), edge.get("target"), edge.get("label", "")) for edge in edges}
    for expected in rules.get("expected_edges", []):
        key = (expected["source"], expected["target"], expected.get("label", ""))
        checks[f"expected_edge:{key}"] = key in edge_pairs
    for forbidden in rules.get("forbidden_edges", []):
        key = (forbidden["source"], forbidden["target"], forbidden.get("label", ""))
        checks[f"forbidden_edge:{key}"] = key not in edge_pairs
    for node_id in rules.get("absent_node_ids", []):
        checks[f"absent_node:{node_id}"] = node_id not in output_by_id
    for path in rules.get("required_paths", []):
        resolved: list[str] = []
        for step in path:
            if isinstance(step, str):
                resolved.append(step)
                continue
            matches = [node_id for node_id, node in output_by_id.items() if all(node.get(k) == v for k, v in step.items())]
            if len(matches) != 1:
                resolved = []
                break
            resolved.append(matches[0])
        checks[f"required_path:{path}"] = bool(resolved) and all(
            any(edge.get("source") == source and edge.get("target") == target for edge in edges)
            for source, target in zip(resolved, resolved[1:])
        )

    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        details.extend(failed)
    return {
        "passed": not failed,
        "failure_category": None if not failed else "acceptance_failed",
        "checks": checks,
        "details": details,
        "observed_node_delta": len(nodes) - len(initial["nodes"]),
        "observed_edge_delta": len(edges) - len(initial["edges"]),
    }
