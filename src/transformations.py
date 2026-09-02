from __future__ import annotations

from typing import Any


PARALLELIZE_SELECTED_CHAIN_TO_JOIN = "parallelize_selected_chain_to_join"


def _edge_pairs(graph: dict[str, Any]) -> set[tuple[str, str, str]]:
    return {
        (edge.get("source", ""), edge.get("target", ""), edge.get("label", ""))
        for edge in graph.get("edges", [])
    }


def _chain_from_graph(graph: dict[str, Any], branch_ids: list[str]) -> tuple[list[str] | None, str | None]:
    branches = set(branch_ids)
    internal = [edge for edge in graph.get("edges", []) if edge.get("source") in branches and edge.get("target") in branches]
    if len(internal) != len(branches) - 1:
        return None, "branch nodes do not have exactly n-1 internal edges"

    outgoing = {node_id: [] for node_id in branches}
    incoming = {node_id: [] for node_id in branches}
    for edge in internal:
        if edge.get("label", "") != "":
            return None, "branch chain contains a labeled edge"
        outgoing[edge["source"]].append(edge["target"])
        incoming[edge["target"]].append(edge["source"])
    heads = [node_id for node_id in branches if not incoming[node_id]]
    tails = [node_id for node_id in branches if not outgoing[node_id]]
    if (
        len(heads) != 1
        or len(tails) != 1
        or any(len(values) > 1 for values in outgoing.values())
        or any(len(values) > 1 for values in incoming.values())
    ):
        return None, "branch topology is not one unambiguous serial chain"

    ordered = [heads[0]]
    while outgoing[ordered[-1]]:
        ordered.append(outgoing[ordered[-1]][0])
    if len(ordered) != len(branches):
        return None, "branch chain is disconnected or cyclic"
    return ordered, None


def has_unambiguous_serial_chain(graph: dict[str, Any], node_ids: list[str]) -> bool:
    """A conservative routing precheck; compilation still performs all safety checks."""
    if len(node_ids) < 2 or len(node_ids) != len(set(node_ids)):
        return False
    chain, _ = _chain_from_graph(graph, node_ids)
    return chain is not None


def compile_parallelize_selected_chain_to_join(
    graph: dict[str, Any],
    semantic_operation: dict[str, Any],
    resolved_scope: dict[str, Any],
    boundary: dict[str, Any],
) -> dict[str, Any]:
    """Compile one safe serial-chain fan-out/fan-in transformation into atomic steps."""
    branch_ids = semantic_operation.get("branch_node_ids")
    join_id = semantic_operation.get("join_node_id")
    if semantic_operation.get("op") != PARALLELIZE_SELECTED_CHAIN_TO_JOIN:
        return {"supported": False, "reason": "unsupported_transformation: unexpected op", "steps": []}
    if not isinstance(branch_ids, list) or len(branch_ids) < 2 or len(branch_ids) != len(set(branch_ids)):
        return {
            "supported": False,
            "reason": "unsupported_transformation: branch_node_ids must be unique and contain at least two nodes",
            "steps": [],
        }
    if semantic_operation.get("join_mode") != "all":
        return {"supported": False, "reason": "unsupported_transformation: join_mode must be all", "steps": []}

    node_ids = {node.get("id") for node in graph.get("nodes", [])}
    if any(node_id not in node_ids for node_id in branch_ids) or join_id not in node_ids or join_id in branch_ids:
        return {"supported": False, "reason": "unsupported_transformation: referenced node is missing or ambiguous", "steps": []}
    core_nodes = set(resolved_scope.get("core_node_ids", []))
    if not set(branch_ids).issubset(core_nodes) or join_id not in core_nodes:
        return {"supported": False, "reason": "unsupported_transformation: branch or join is outside resolved scope", "steps": []}

    chain, chain_error = _chain_from_graph(graph, branch_ids)
    if chain_error:
        return {"supported": False, "reason": f"unsupported_transformation: {chain_error}", "steps": []}
    assert chain is not None

    branches = set(branch_ids)
    head, tail = chain[0], chain[-1]
    external_in = [edge for edge in graph.get("edges", []) if edge.get("target") == head and edge.get("source") not in branches]
    join_edges = [edge for edge in graph.get("edges", []) if edge.get("source") == tail and edge.get("target") == join_id]
    if len(external_in) != 1 or len(join_edges) != 1:
        return {"supported": False, "reason": "unsupported_transformation: upstream or join interface is not unique", "steps": []}
    upstream = external_in[0]
    if upstream.get("label", "") != "" or join_edges[0].get("label", "") != "":
        return {"supported": False, "reason": "unsupported_transformation: labeled interface edge is not supported", "steps": []}

    # Reject side connections on branch nodes rather than guessing how they should be preserved.
    allowed_external_pairs = {(upstream["source"], head), (tail, join_id)}
    for edge in graph.get("edges", []):
        source, target = edge.get("source"), edge.get("target")
        if source in branches and target not in branches and (source, target) not in allowed_external_pairs:
            return {"supported": False, "reason": "unsupported_transformation: branch has an ambiguous outgoing side connection", "steps": []}
        if target in branches and source not in branches and (source, target) not in allowed_external_pairs:
            return {"supported": False, "reason": "unsupported_transformation: branch has an ambiguous incoming side connection", "steps": []}

    # Derive the minimum diff from the graph. Existing correct interface edges are retained.
    current_pairs = _edge_pairs(graph)
    desired_pairs = {(upstream["source"], branch_id, "") for branch_id in chain}
    desired_pairs.update((branch_id, join_id, "") for branch_id in chain)
    serial_edges = [edge for edge in graph.get("edges", []) if edge.get("source") in branches and edge.get("target") in branches]
    steps: list[dict[str, Any]] = [
        {"op": "remove_edge", "edge_id": edge["id"], "instruction": "remove obsolete serial branch edge"}
        for edge in serial_edges
    ]
    for source, target, label in sorted(desired_pairs):
        if (source, target, label) not in current_pairs:
            steps.append(
                {
                    "op": "add_edge",
                    "source": source,
                    "target": target,
                    "label": label,
                    "instruction": "add required parallel topology edge",
                }
            )
    join_node = next(node for node in graph.get("nodes", []) if node.get("id") == join_id)
    if join_node.get("join") != {"mode": "all"}:
        steps.append(
            {
                "op": "set_join",
                "node_id": join_id,
                "join": {"mode": "all"},
                "instruction": "configure all-branches join",
            }
        )
    return {
        "supported": True,
        "reason": None,
        "steps": steps,
        "derived": {"ordered_chain": chain, "upstream_node_id": upstream["source"], "join_node_id": join_id},
        "scope_received": resolved_scope,
        "boundary_received": boundary,
    }
