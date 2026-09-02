from __future__ import annotations

"""Declarative workflow graph planning.

The planner deliberately owns graph construction details.  Callers provide a
validated desired state; they never provide executable add/remove scripts.
"""

from copy import deepcopy
from typing import Any


def _nodes(graph: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {node["id"]: node for node in graph.get("nodes", [])}


def _edge_pairs(graph: dict[str, Any]) -> set[tuple[str, str, str]]:
    return {(edge.get("source"), edge.get("target"), edge.get("label", "")) for edge in graph.get("edges", [])}


def _edge_map(graph: dict[str, Any]) -> dict[tuple[str, str, str], dict[str, Any]]:
    return {(edge.get("source"), edge.get("target"), edge.get("label", "")): edge for edge in graph.get("edges", [])}


def _adjacency(graph: dict[str, Any]) -> tuple[dict[str, list[dict[str, Any]]], dict[str, list[dict[str, Any]]]]:
    outgoing: dict[str, list[dict[str, Any]]] = {}
    incoming: dict[str, list[dict[str, Any]]] = {}
    for edge in graph.get("edges", []):
        source = edge.get("source")
        target = edge.get("target")
        if isinstance(source, str):
            outgoing.setdefault(source, []).append(edge)
        if isinstance(target, str):
            incoming.setdefault(target, []).append(edge)
    return outgoing, incoming


def extract_graph_context(graph: dict[str, Any], target: dict[str, Any], selection: dict[str, Any]) -> dict[str, Any]:
    """Return read-only local graph facts for the target; this grants no edit permission."""
    nodes = _nodes(graph)
    target_nodes = set(target.get("node_ids", []))
    target_nodes.update(value for value in (target.get("anchor_node_id"), target.get("moving_node_id"), target.get("join_node_id")) if value)
    selected_edges = set(selection.get("edge_ids", [])) | set(target.get("edge_ids", []))
    edges = graph.get("edges", [])
    outgoing, incoming = _adjacency(graph)
    related_ids: set[str] = set(selected_edges)
    for node_id in target_nodes:
        related_ids.update(edge.get("id") for edge in outgoing.get(node_id, []) if edge.get("id"))
        related_ids.update(edge.get("id") for edge in incoming.get(node_id, []) if edge.get("id"))
    related = [edge for edge in edges if edge.get("id") in related_ids]
    context_ids = target_nodes | {endpoint for edge in related for endpoint in (edge.get("source"), edge.get("target")) if endpoint}
    return {
        "target_nodes": [deepcopy(nodes[node_id]) for node_id in sorted(context_ids) if node_id in nodes],
        "selected_edges": [deepcopy(edge) for edge in edges if edge.get("id") in selected_edges],
        "incident_edges": [deepcopy(edge) for edge in related],
        "facts": {
            "predecessors": {node_id: [edge["source"] for edge in incoming.get(node_id, [])] for node_id in sorted(target_nodes)},
            "successors": {node_id: [edge["target"] for edge in outgoing.get(node_id, [])] for node_id in sorted(target_nodes)},
        },
    }


def _selected_edge(graph: dict[str, Any], selection: dict[str, Any]) -> dict[str, Any] | None:
    edge_ids = selection.get("edge_ids", [])
    if len(edge_ids) != 1:
        return None
    return next((edge for edge in graph.get("edges", []) if edge.get("id") == edge_ids[0]), None)


def _serial_chain(graph: dict[str, Any], branch_ids: list[str]) -> tuple[list[str] | None, str | None]:
    branches = set(branch_ids)
    internal = [edge for edge in graph.get("edges", []) if edge.get("source") in branches and edge.get("target") in branches]
    if len(branches) < 2 or len(internal) != len(branches) - 1:
        return None, "selected nodes are not one serial chain"
    incoming = {node_id: [] for node_id in branches}
    outgoing = {node_id: [] for node_id in branches}
    for edge in internal:
        if edge.get("label", ""):
            return None, "labeled serial-chain edges are unsupported"
        incoming[edge["target"]].append(edge["source"])
        outgoing[edge["source"]].append(edge["target"])
    heads = [node_id for node_id in branches if not incoming[node_id]]
    tails = [node_id for node_id in branches if not outgoing[node_id]]
    if len(heads) != 1 or len(tails) != 1 or any(len(items) > 1 for items in incoming.values()) or any(len(items) > 1 for items in outgoing.values()):
        return None, "selected nodes are not an unambiguous serial chain"
    ordered = [heads[0]]
    while outgoing[ordered[-1]]:
        ordered.append(outgoing[ordered[-1]][0])
    return (ordered, None) if len(ordered) == len(branches) else (None, "serial chain is cyclic")


def _parallel_upstream_from_constraints(constraints: dict[str, Any], branches: list[str], join: str) -> str | None:
    branch_set = set(branches)
    counts: dict[str, set[str]] = {}
    for row in constraints.get("required_relations", []):
        relation = _relation(row)
        if relation is None:
            continue
        source, target, label = relation
        if label or target not in branch_set or source in branch_set or source == join:
            continue
        counts.setdefault(source, set()).add(target)
    matches = [source for source, covered in counts.items() if covered == branch_set]
    return matches[0] if len(matches) == 1 else None


def expand_specification(graph: dict[str, Any], selection: dict[str, Any], specification: dict[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
    """Expand a small number of declarative topology directives into generic constraints."""
    constraints = deepcopy(specification.get("constraints", {}))
    constraints.setdefault("entities", [])
    constraints.setdefault("absent_entities", [])
    constraints.setdefault("properties", [])
    constraints.setdefault("edge_properties", [])
    constraints.setdefault("required_relations", [])
    constraints.setdefault("forbidden_relations", [])
    constraints.setdefault("preserve", specification.get("preserve", {"outside_specification": True}))
    if any(not isinstance(constraints[key], list) for key in ("entities", "absent_entities", "properties", "edge_properties", "required_relations", "forbidden_relations")):
        return None, "invalid constraint schema: entities, absent_entities, properties, edge_properties, and relations must be arrays"
    if not isinstance(specification.get("directives", []), list):
        return None, "invalid specification schema: directives must be an array"
    for directive in specification.get("directives", []):
        kind = directive.get("kind")
        if kind == "insert_on_selected_edge":
            edge, entity = _selected_edge(graph, selection), directive.get("new_entity", {})
            if not edge or not entity.get("ref") or not entity.get("type") or not entity.get("label"):
                return None, "insert_on_selected_edge is underspecified"
            constraints["entities"].append(entity)
            constraints["required_relations"].extend([
                {"source": edge["source"], "target": entity["ref"], "label": edge.get("label", "")},
                {"source": entity["ref"], "target": edge["target"], "label": edge.get("label", "")},
            ])
            constraints["forbidden_relations"].append({"source": edge["source"], "target": edge["target"], "label": edge.get("label", "")})
        elif kind == "redirect_selected_edge":
            edge, target = _selected_edge(graph, selection), directive.get("target_node_id")
            if not edge or not target:
                return None, "redirect_selected_edge is underspecified"
            constraints["required_relations"].append({"source": edge["source"], "target": target, "label": edge.get("label", "")})
            constraints["forbidden_relations"].append({"source": edge["source"], "target": edge["target"], "label": edge.get("label", "")})
        elif kind == "parallelize_selected_chain":
            branches = directive.get("branches", directive.get("branch_node_ids", selection.get("node_ids", [])))
            upstream_node_id = directive.get("upstream")
            join = directive.get("join", directive.get("join_node_id"))
            completion = directive.get("completion", directive.get("join_mode"))
            valid_branch_list = isinstance(branches, list) and all(isinstance(node_id, str) and node_id for node_id in branches)
            chain, error = _serial_chain(graph, branches) if valid_branch_list else (None, "invalid branch nodes")
            if error or not upstream_node_id or not join or completion != "all":
                return None, error or "parallelize_selected_chain is underspecified"
            # The directive is the authoritative statement for the local
            # upstream/branch/join topology. Discard any direct relation fully
            # inside that owned region so redundant or contradictory repeats
            # cannot conflict with the canonical expansion, while external
            # retry/failure routing facts remain untouched.
            canonical_required = {
                (upstream_node_id, branch, "") for branch in chain
            } | {(branch, join, "") for branch in chain}
            canonical_forbidden = {(chain[index], chain[index + 1], "") for index in range(len(chain) - 1)}
            owned_nodes = set(chain) | {upstream_node_id, join}
            for key in ("required_relations", "forbidden_relations"):
                constraints[key] = [
                    row for row in constraints[key]
                    if row.get("source") not in owned_nodes or row.get("target") not in owned_nodes
                ]
            constraints["properties"] = [item for item in constraints["properties"] if item.get("node_id") != join or "join" not in item.get("changes", {})]
            constraints["required_relations"].extend({"source": upstream_node_id, "target": branch, "label": ""} for branch in chain)
            constraints["required_relations"].extend({"source": branch, "target": join, "label": ""} for branch in chain)
            constraints["forbidden_relations"].extend({"source": chain[index], "target": chain[index + 1], "label": ""} for index in range(len(chain) - 1))
            constraints["properties"].append({"node_id": join, "changes": {"join": {"mode": "all"}}})
        elif kind == "relocate_after":
            moving, anchor = directive.get("moving_node_id"), directive.get("anchor_node_id")
            if not directive.get("reconnect_old_location") or not moving or not anchor:
                return None, "relocate_after lacks explicit reconnection policy"
            pred = [edge for edge in graph.get("edges", []) if edge.get("target") == moving]
            succ = [edge for edge in graph.get("edges", []) if edge.get("source") == moving]
            anchor_succ = [edge for edge in graph.get("edges", []) if edge.get("source") == anchor]
            if len(pred) != 1 or len(succ) != 1 or len(anchor_succ) != 1:
                return None, "relocate_after requires unique predecessor and successors"
            constraints["required_relations"].extend([
                {"source": pred[0]["source"], "target": succ[0]["target"], "label": pred[0].get("label", "")},
                {"source": anchor, "target": moving, "label": ""},
                {"source": moving, "target": anchor_succ[0]["target"], "label": ""},
            ])
            constraints["forbidden_relations"].extend([
                {"source": pred[0]["source"], "target": moving, "label": pred[0].get("label", "")},
                {"source": moving, "target": succ[0]["target"], "label": succ[0].get("label", "")},
                {"source": anchor, "target": anchor_succ[0]["target"], "label": anchor_succ[0].get("label", "")},
            ])
        elif kind:
            return None, f"unsupported directive: {kind}"
    return constraints, None


def _relation(row: dict[str, Any], refs: dict[str, str] | None = None) -> tuple[str, str, str] | None:
    source, target = row.get("source"), row.get("target")
    if refs:
        source, target = refs.get(source, source), refs.get(target, target)
    if not isinstance(source, str) or not isinstance(target, str) or not source or not target:
        return None
    return source, target, row.get("label", "")


def _edge_fields_from_relation(row: dict[str, Any], refs: dict[str, str] | None = None) -> dict[str, Any]:
    source, target = row.get("source"), row.get("target")
    if refs:
        source, target = refs.get(source, source), refs.get(target, target)
    fields = {
        key: deepcopy(value)
        for key, value in row.items()
        if key not in {"source", "target", "label"}
    }
    if isinstance(source, str):
        fields["source"] = source
    if isinstance(target, str):
        fields["target"] = target
    fields["label"] = row.get("label", "")
    if isinstance(fields.get("id"), str) and "edge_id" not in fields:
        fields["edge_id"] = fields["id"]
    return fields


def plan_constraints(graph: dict[str, Any], constraints: dict[str, Any]) -> dict[str, Any]:
    """Validate constraints, calculate the minimal diff, and attach provenance per mutation."""
    node_map = _nodes(graph)
    refs: dict[str, str] = {}
    steps: list[dict[str, Any]] = []
    authorization: list[dict[str, Any]] = []
    preserve = constraints.get("preserve", {})
    preserved_nodes = set(preserve.get("node_ids", [])) if isinstance(preserve, dict) else set()
    preserved_edges = set(preserve.get("edge_ids", [])) if isinstance(preserve, dict) else set()
    for entity in constraints.get("entities", []):
        if not isinstance(entity, dict) or not all(entity.get(key) for key in ("ref", "type", "label")) or entity["ref"] in refs or entity["ref"] in node_map:
            return {"supported": False, "reason": "invalid entity constraint", "steps": [], "mutation_authorization": []}
        index = 1
        node_id = f"new_{index}"
        while node_id in node_map or node_id in refs.values():
            index += 1
            node_id = f"new_{index}"
        refs[entity["ref"]] = node_id
        node = {"id": node_id, "type": entity["type"], "label": entity["label"], "x": entity.get("x", 0), "y": entity.get("y", 0)}
        step = {"op": "add_node", "node": node}
        steps.append(step)
        authorization.append({"step": step, "constraint_rule": "entity_exists", "constraint": entity})
    absent_entities: list[dict[str, Any]] = []
    absent_node_ids: set[str] = set()
    for entity in constraints.get("absent_entities", []):
        node_id = entity.get("node_id") if isinstance(entity, dict) else None
        if not isinstance(node_id, str) or not node_id or node_id in absent_node_ids or node_id in refs.values() or node_id not in node_map:
            return {"supported": False, "reason": "invalid absent entity constraint", "steps": [], "mutation_authorization": []}
        absent_entities.append(entity)
        absent_node_ids.add(node_id)
    if preserved_nodes & absent_node_ids:
        return {"supported": False, "reason": "preserve constraint blocks node deletion", "steps": [], "mutation_authorization": []}
    required = {_relation(row, refs) for row in constraints.get("required_relations", [])}
    forbidden = {_relation(row, refs) for row in constraints.get("forbidden_relations", [])}
    if None in required or None in forbidden:
        return {"supported": False, "reason": "invalid relation constraint", "steps": [], "mutation_authorization": []}
    if required & forbidden:
        return {"supported": False, "reason": "relation is both required and forbidden", "steps": [], "mutation_authorization": []}
    all_nodes = set(node_map) | set(refs.values())
    if any(source not in all_nodes or target not in all_nodes for source, target, _ in required | forbidden):
        return {"supported": False, "reason": "relation endpoint is missing", "steps": [], "mutation_authorization": []}
    if any(source in absent_node_ids or target in absent_node_ids for source, target, _ in required):
        return {"supported": False, "reason": "required relation references a deleted node", "steps": [], "mutation_authorization": []}
    edge_map = _edge_map(graph)
    current = _edge_pairs(graph)
    for relation in sorted(forbidden):
        if relation in current:
            if edge_map[relation]["id"] in preserved_edges:
                return {"supported": False, "reason": "preserve constraint blocks edge removal", "steps": [], "mutation_authorization": []}
            step = {"op": "remove_edge", "edge_id": edge_map[relation]["id"]}
            steps.append(step)
            authorization.append({"step": step, "constraint_rule": "relation_forbidden", "constraint": relation})
    removed_relations = {row["step"]["edge_id"] for row in authorization if row["step"]["op"] == "remove_edge"}
    for entity in absent_entities:
        node_id = entity["node_id"]
        incident = [edge for edge in graph.get("edges", []) if edge.get("source") == node_id or edge.get("target") == node_id]
        for edge in incident:
            edge_id = edge.get("id")
            if edge_id in removed_relations:
                continue
            if edge_id in preserved_edges:
                return {"supported": False, "reason": "preserve constraint blocks incident edge removal", "steps": [], "mutation_authorization": []}
            step = {"op": "remove_edge", "edge_id": edge_id}
            steps.append(step)
            authorization.append({"step": step, "constraint_rule": "entity_absent_incident_relation", "constraint": entity})
            if edge_id:
                removed_relations.add(edge_id)
        step = {"op": "remove_node", "node_id": node_id}
        steps.append(step)
        authorization.append({"step": step, "constraint_rule": "entity_absent", "constraint": entity})
    required_rows = {
        relation: _edge_fields_from_relation(row, refs)
        for row in constraints.get("required_relations", [])
        if (relation := _relation(row, refs)) is not None
    }
    for source, target, label in sorted(required):
        if (source, target, label) not in current:
            step = {"op": "add_edge", **required_rows.get((source, target, label), {"source": source, "target": target, "label": label})}
            steps.append(step)
            authorization.append({"step": step, "constraint_rule": "relation_required", "constraint": required_rows.get((source, target, label), (source, target, label))})
    for property_constraint in constraints.get("properties", []):
        node_id, changes = property_constraint.get("node_id"), property_constraint.get("changes")
        if node_id not in node_map or not isinstance(changes, dict) or not changes:
            return {"supported": False, "reason": "invalid property constraint", "steps": [], "mutation_authorization": []}
        delta = {key: value for key, value in changes.items() if node_map[node_id].get(key) != value}
        if not delta:
            continue
        if set(delta) == {"join"}:
            step = {"op": "set_join", "node_id": node_id, "join": delta["join"]}
        elif set(delta).issubset({"x", "y"}):
            step = {"op": "move_node", "node_id": node_id, "x": delta.get("x", node_map[node_id].get("x")), "y": delta.get("y", node_map[node_id].get("y"))}
        else:
            step = {"op": "update_node", "node_id": node_id, "changes": delta}
        steps.append(step)
        authorization.append({"step": step, "constraint_rule": "property_equals", "constraint": property_constraint})
    edge_id_map = {edge.get("id"): edge for edge in graph.get("edges", [])}
    for property_constraint in constraints.get("edge_properties", []):
        edge_id, changes = property_constraint.get("edge_id"), property_constraint.get("changes")
        if edge_id not in edge_id_map or not isinstance(changes, dict) or not changes:
            return {"supported": False, "reason": "invalid edge property constraint", "steps": [], "mutation_authorization": []}
        delta = {key: value for key, value in changes.items() if edge_id_map[edge_id].get(key) != value}
        if delta:
            step = {"op": "update_edge", "edge_id": edge_id, "changes": delta}
            steps.append(step)
            authorization.append({"step": step, "constraint_rule": "edge_property_equals", "constraint": property_constraint})
    return {"supported": True, "reason": None, "steps": steps, "mutation_authorization": authorization, "constraints": constraints, "entity_ids": refs}
