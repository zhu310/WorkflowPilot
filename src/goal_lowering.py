from __future__ import annotations

import hashlib
import re
from copy import deepcopy
from typing import Any

from .semantic_reference import resolve_semantic_node_reference
from .workflow_ir_core import DEFAULT_NODE_HEIGHT, DEFAULT_NODE_WIDTH, normalize_edge_changes, normalize_ir


DEFAULT_LAYOUT_SPACING = 80
SAFE_NODE_DISPLAY_PROPERTIES = {"label", "name", "display_label"}
OUTER_QUOTE_PAIRS = {
    '"': '"',
    "'": "'",
    "\u201c": "\u201d",
    "\u2018": "\u2019",
    "\u300c": "\u300d",
    "\u300e": "\u300f",
}


def _strip_outer_reference_quotes(value: Any) -> str:
    text = str(value or "").strip()
    if len(text) >= 2:
        closing = OUTER_QUOTE_PAIRS.get(text[0])
        if closing and text[-1] == closing:
            inner = text[1:-1].strip()
            if inner:
                return inner
    return text


def _nodes_by_id(graph_data: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        node["id"]: node
        for node in graph_data.get("nodes", [])
        if isinstance(node, dict) and isinstance(node.get("id"), str)
    }


def _binding_by_role(goal: dict[str, Any], role: str) -> dict[str, Any] | None:
    for binding in goal.get("bindings", []):
        if isinstance(binding, dict) and binding.get("role") == role:
            return binding
    return None


def _branch_property_change(goal: dict[str, Any]) -> dict[str, str] | None:
    prop = goal.get("property") if isinstance(goal.get("property"), dict) else {}
    name = prop.get("name")
    value = prop.get("value")
    if name not in {"label", "branch_label"} or not isinstance(value, str) or not value.strip():
        return None
    # Workflow IR's executable edge display-name field is branch_label;
    # edge.label is the structural relation type and is intentionally not changed here.
    return {"branch_label": value.strip()}


def _new_ref_from_label(label: str) -> str:
    label = _strip_outer_reference_quotes(label)
    compact = re.sub(r"[^a-zA-Z0-9_]+", "_", label.strip().lower()).strip("_")
    if compact:
        return f"new_{compact}"
    digest = hashlib.sha1(label.strip().encode("utf-8")).hexdigest()[:10]
    return f"new_{digest}"


def _new_ref_from_identity(identity_key: str, binding: dict[str, Any]) -> str:
    if identity_key.startswith("v2:"):
        compact = re.sub(r"[^a-zA-Z0-9_]+", "_", identity_key[3:].strip().lower()).strip("_")
        if compact:
            return f"new_{compact}"
    if binding.get("role") == "subject" and identity_key.startswith("legacy_label:"):
        return "new_subject"
    label = _strip_outer_reference_quotes(binding.get("label")) if isinstance(binding.get("label"), str) else identity_key
    return _new_ref_from_label(str(label))


def _new_entity_identity_key(binding: dict[str, Any]) -> str:
    v2_object_id = binding.get("v2_object_id")
    if isinstance(v2_object_id, str) and v2_object_id.strip():
        return f"v2:{v2_object_id.strip()}"
    existing_ref = binding.get("ref")
    if isinstance(existing_ref, str) and existing_ref.strip():
        return f"ref:{existing_ref.strip()}"
    return f"legacy_label:{_canonical_new_entity_label(binding.get('label'))}"


def _workflow_node_from_binding(
    binding: dict[str, Any],
    ref_override: str | None = None,
    *,
    identity_key: str | None = None,
) -> dict[str, Any] | None:
    if binding.get("binding_type") == "node" and isinstance(binding.get("id"), str):
        return {"type": "node", "id": binding["id"]}
    if binding.get("needs_creation") is True and isinstance(binding.get("label"), str) and binding["label"].strip():
        label = _strip_outer_reference_quotes(binding["label"])
        if isinstance(ref_override, str) and ref_override:
            return {"type": "node", "ref": ref_override, "node_type": "task", "label": label}
        key = identity_key or _new_entity_identity_key(binding)
        return {"type": "node", "ref": _new_ref_from_identity(key, binding), "node_type": "task", "label": label}
    return None


def _same_new_entity_key(label: Any) -> str:
    return _canonical_new_entity_label(label)


def _register_created_node(registry: dict[str, dict[str, Any]], binding: dict[str, Any], node: dict[str, Any]) -> None:
    if binding.get("needs_creation") is not True or not isinstance(binding.get("label"), str):
        return
    key = _new_entity_identity_key(binding)
    if key and isinstance(node.get("ref"), str):
        registry.setdefault(key, node)


def _node_for_new_entity(
    binding: dict[str, Any],
    registry: dict[str, dict[str, Any]],
    *,
    prefer_existing_registry: bool = True,
) -> dict[str, Any] | None:
    if binding.get("needs_creation") is not True or not isinstance(binding.get("label"), str):
        return _workflow_node_from_binding(binding)
    key = _new_entity_identity_key(binding)
    if prefer_existing_registry and key in registry:
        return registry[key]
    ref_override = registry[key]["ref"] if key in registry and isinstance(registry[key].get("ref"), str) else None
    node = _workflow_node_from_binding(binding, ref_override=ref_override, identity_key=key)
    if node is not None:
        _register_created_node(registry, binding, node)
    return node


def _entity_declaration_from_node(node: dict[str, Any]) -> dict[str, Any] | None:
    ref = node.get("ref")
    node_type = node.get("node_type", node.get("type"))
    label = node.get("label")
    if not (isinstance(ref, str) and ref and isinstance(node_type, str) and node_type and isinstance(label, str) and label):
        return None
    declaration: dict[str, Any] = {"ref": ref, "node_type": node_type, "label": label}
    for key in ("x", "y", "width", "height"):
        if isinstance(node.get(key), (int, float)):
            declaration[key] = node[key]
    return declaration


def _declare_new_entity(workflow_ir: dict[str, Any], node: dict[str, Any]) -> None:
    declaration = _entity_declaration_from_node(node)
    if declaration is None:
        return
    entities = workflow_ir.setdefault("entities", [])
    for existing in entities:
        if isinstance(existing, dict) and existing.get("ref") == declaration["ref"]:
            same_identity = (
                existing.get("node_type", existing.get("type")) == declaration["node_type"]
                and existing.get("label") == declaration["label"]
            )
            if same_identity:
                for key in ("x", "y", "width", "height"):
                    if key not in existing and key in declaration:
                        existing[key] = declaration[key]
                return
            # Preserve the conflict for validate_ir instead of silently choosing
            # one declaration for the same transaction-local ref.
            break
    entities.append(declaration)


def _structural_occurrence(node: dict[str, Any]) -> dict[str, Any]:
    if isinstance(node.get("id"), str):
        return {"type": "node", "id": node["id"]}
    if isinstance(node.get("ref"), str):
        return {"type": "node", "ref": node["ref"]}
    return dict(node)


def _declare_and_occurrence(workflow_ir: dict[str, Any], node: dict[str, Any]) -> dict[str, Any]:
    if isinstance(node.get("ref"), str):
        _declare_new_entity(workflow_ir, node)
    return _structural_occurrence(node)


def _incoming_branch_nodes(graph_data: dict[str, Any], anchor_id: str, relation_type: str) -> list[dict[str, str]]:
    sources = sorted(
        {
            edge["source"]
            for edge in graph_data.get("edges", [])
            if isinstance(edge, dict)
            and edge.get("target") == anchor_id
            and edge.get("label", "") == relation_type
            and isinstance(edge.get("source"), str)
        }
    )
    return [{"type": "node", "id": source} for source in sources]


def _preserve_join_requested(goal: dict[str, Any]) -> bool:
    constraints = goal.get("explicit_constraints", [])
    if not isinstance(constraints, list):
        return False
    normalized = " ".join(str(item).strip().lower() for item in constraints if isinstance(item, str))
    return "preserve" in normalized and ("join" in normalized or "merge" in normalized)


def _node_number(node: dict[str, Any], key: str, default: float) -> float:
    value = node.get(key)
    return float(value) if isinstance(value, (int, float)) else default


def _rect_for_node(node: dict[str, Any], x: float | None = None, y: float | None = None) -> tuple[float, float, float, float]:
    width = _node_number(node, "width", float(DEFAULT_NODE_WIDTH))
    height = _node_number(node, "height", float(DEFAULT_NODE_HEIGHT))
    center_x = _node_number(node, "x", 0.0) if x is None else x
    center_y = _node_number(node, "y", 0.0) if y is None else y
    return center_x - width / 2, center_y - height / 2, center_x + width / 2, center_y + height / 2


def _rects_overlap(left: tuple[float, float, float, float], right: tuple[float, float, float, float]) -> bool:
    return not (
        left[2] <= right[0]
        or right[2] <= left[0]
        or left[3] <= right[1]
        or right[3] <= left[1]
    )


def _layout_collision(
    graph_data: dict[str, Any],
    target_id: str,
    target_node: dict[str, Any],
    x: float,
    y: float,
) -> dict[str, Any] | None:
    proposed = _rect_for_node(target_node, x, y)
    for node in graph_data.get("nodes", []):
        if not isinstance(node, dict) or node.get("id") == target_id or not isinstance(node.get("id"), str):
            continue
        if _rects_overlap(proposed, _rect_for_node(node)):
            return {
                "kind": "layout_collision",
                "message": "layout change would overlap another node",
                "severity": "blocking",
                "location": "semantic_goals.layout",
                "target": target_id,
                "conflicting_node": node["id"],
                "proposed": {"x": x, "y": y},
            }
    return None


def _first_non_colliding_layout(
    graph_data: dict[str, Any],
    target_id: str,
    target_node: dict[str, Any],
    x: float,
    y: float,
) -> tuple[float, float] | None:
    step = _node_number(target_node, "width", float(DEFAULT_NODE_WIDTH)) + DEFAULT_LAYOUT_SPACING
    candidate_x = x
    for _ in range(8):
        if _layout_collision(graph_data, target_id, target_node, candidate_x, y) is None:
            return candidate_x, y
        candidate_x += step
    return None


def _first_non_colliding_from_candidates(
    graph_data: dict[str, Any],
    target_id: str,
    target_node: dict[str, Any],
    candidates: list[tuple[float, float]],
) -> tuple[float, float]:
    for x, y in candidates:
        if _layout_collision(graph_data, target_id, target_node, x, y) is None:
            return x, y
    if candidates:
        return _first_non_colliding_layout(graph_data, target_id, target_node, candidates[0][0], candidates[0][1]) or candidates[0]
    return _default_new_node_position(graph_data, target_node)


def _default_new_node_position(graph_data: dict[str, Any], node: dict[str, Any]) -> tuple[float, float]:
    nodes = [row for row in graph_data.get("nodes", []) if isinstance(row, dict)]
    if not nodes:
        return 160.0, 160.0
    max_x = max(_node_number(row, "x", 0.0) for row in nodes)
    avg_y = sum(_node_number(row, "y", 0.0) for row in nodes) / len(nodes)
    step = _node_number(node, "width", float(DEFAULT_NODE_WIDTH)) + DEFAULT_LAYOUT_SPACING
    candidates = [(max_x + step * column, avg_y + 120.0 * row) for column in range(1, 6) for row in (0, 1, -1, 2, -2)]
    return _first_non_colliding_from_candidates(graph_data, str(node.get("ref") or "new_node"), node, candidates)


def _edge_split_candidates(
    graph_data: dict[str, Any],
    source_id: str,
    target_id: str,
) -> list[tuple[float, float]]:
    nodes = _nodes_by_id(graph_data)
    source = nodes.get(source_id)
    target = nodes.get(target_id)
    if not source or not target:
        return []
    source_x = _node_number(source, "x", 0.0)
    source_y = _node_number(source, "y", 0.0)
    target_x = _node_number(target, "x", 0.0)
    target_y = _node_number(target, "y", 0.0)
    mid_x = (source_x + target_x) / 2
    mid_y = (source_y + target_y) / 2
    return [
        (mid_x, mid_y),
        (mid_x, mid_y + 120.0),
        (mid_x, mid_y - 120.0),
        (mid_x + 120.0, mid_y),
        (mid_x - 120.0, mid_y),
        (mid_x + 120.0, mid_y + 120.0),
        (mid_x + 120.0, mid_y - 120.0),
    ]


def _ordering_layout_candidates(graph_data: dict[str, Any], anchor_id: str, relation: str) -> list[tuple[float, float]]:
    nodes = _nodes_by_id(graph_data)
    anchor = nodes.get(anchor_id)
    if not anchor:
        return []
    if relation == "before":
        incoming = [
            edge
            for edge in graph_data.get("edges", [])
            if isinstance(edge, dict) and edge.get("target") == anchor_id and isinstance(edge.get("source"), str)
        ]
        if len(incoming) == 1:
            return _edge_split_candidates(graph_data, incoming[0]["source"], anchor_id)
        anchor_x = _node_number(anchor, "x", 0.0)
        anchor_y = _node_number(anchor, "y", 0.0)
        step = _node_number(anchor, "width", float(DEFAULT_NODE_WIDTH)) + DEFAULT_LAYOUT_SPACING
        return [(anchor_x - step, anchor_y), (anchor_x - step, anchor_y + 120.0), (anchor_x - step, anchor_y - 120.0)]
    outgoing = [
        edge
        for edge in graph_data.get("edges", [])
        if isinstance(edge, dict) and edge.get("source") == anchor_id and isinstance(edge.get("target"), str)
    ]
    if len(outgoing) == 1:
        return _edge_split_candidates(graph_data, anchor_id, outgoing[0]["target"])
    anchor_x = _node_number(anchor, "x", 0.0)
    anchor_y = _node_number(anchor, "y", 0.0)
    step = _node_number(anchor, "width", float(DEFAULT_NODE_WIDTH)) + DEFAULT_LAYOUT_SPACING
    return [(anchor_x + step, anchor_y), (anchor_x + step, anchor_y + 120.0), (anchor_x + step, anchor_y - 120.0)]


def _ensure_new_node_layout(
    graph_data: dict[str, Any],
    node: dict[str, Any],
    *,
    candidates: list[tuple[float, float]] | None = None,
) -> None:
    if "ref" not in node or isinstance(node.get("x"), (int, float)) or isinstance(node.get("y"), (int, float)):
        return
    x, y = _first_non_colliding_from_candidates(
        graph_data,
        str(node.get("ref") or "new_node"),
        node,
        candidates or [_default_new_node_position(graph_data, node)],
    )
    node["x"] = x
    node["y"] = y


def _empty_workflow_ir() -> dict[str, Any]:
    return {
        "control_flow_regions": [],
        "auxiliary_relations": [],
        "properties": [],
        "edge_properties": [],
        "absent_entities": [],
        "preserve": {"node_ids": [], "edge_ids": []},
    }


def _creation_relation_type(graph_data: dict[str, Any]) -> str:
    return "" if any(isinstance(edge, dict) and edge.get("label", "") == "" for edge in graph_data.get("edges", [])) else "success"


def _append_creation_region(workflow_ir: dict[str, Any], node: dict[str, Any], graph_data: dict[str, Any]) -> None:
    ref = node.get("ref")
    if not isinstance(ref, str):
        return
    _declare_new_entity(workflow_ir, node)
    for region in workflow_ir.get("control_flow_regions", []):
        structure = region.get("structure") if isinstance(region, dict) else None
        if isinstance(structure, dict) and structure.get("type") == "node" and structure.get("ref") == ref:
            return
        if isinstance(structure, dict) and structure.get("type") == "sequence":
            for item in structure.get("items", []):
                if isinstance(item, dict) and item.get("type") == "node" and item.get("ref") == ref:
                    return
    workflow_ir["control_flow_regions"].append({"relation_type": _creation_relation_type(graph_data), "structure": _structural_occurrence(node)})


def _sequence_node_key(node: dict[str, Any]) -> tuple[str, str] | None:
    if not isinstance(node, dict) or node.get("type") != "node":
        return None
    if isinstance(node.get("id"), str) and node["id"]:
        return ("id", node["id"])
    if isinstance(node.get("ref"), str) and node["ref"]:
        return ("ref", node["ref"])
    return None


def _node_only_sequence_items(region: dict[str, Any]) -> list[dict[str, Any]] | None:
    if not isinstance(region, dict) or not isinstance(region.get("relation_type"), str):
        return None
    structure = region.get("structure")
    if not isinstance(structure, dict) or structure.get("type") != "sequence":
        return None
    items = structure.get("items", [])
    if not isinstance(items, list) or len(items) < 2:
        return None
    if any(_sequence_node_key(item) is None for item in items if isinstance(item, dict)) or not all(isinstance(item, dict) for item in items):
        return None
    return items


def _normalize_contiguous_ordering_sequences(workflow_ir: dict[str, Any]) -> None:
    regions = workflow_ir.get("control_flow_regions")
    if not isinstance(regions, list) or len(regions) < 2:
        return

    eligible_by_relation: dict[str, list[tuple[int, list[dict[str, Any]]]]] = {}
    for index, region in enumerate(regions):
        if isinstance(region.get("lowering_scope_id"), str) and region["lowering_scope_id"]:
            continue
        items = _node_only_sequence_items(region)
        if items is None:
            continue
        eligible_by_relation.setdefault(region["relation_type"], []).append((index, items))

    replacements: dict[int, dict[str, Any]] = {}
    remove_indexes: set[int] = set()
    for relation_type, entries in eligible_by_relation.items():
        if len(entries) < 2:
            continue
        node_by_key: dict[tuple[str, str], dict[str, Any]] = {}
        outgoing: dict[tuple[str, str], set[tuple[str, str]]] = {}
        incoming: dict[tuple[str, str], set[tuple[str, str]]] = {}
        edge_indexes: dict[tuple[tuple[str, str], tuple[str, str]], set[int]] = {}
        for index, items in entries:
            for item in items:
                key = _sequence_node_key(item)
                if key is not None:
                    node_by_key.setdefault(key, item)
            for left, right in zip(items, items[1:]):
                left_key = _sequence_node_key(left)
                right_key = _sequence_node_key(right)
                if left_key is None or right_key is None or left_key == right_key:
                    continue
                outgoing.setdefault(left_key, set()).add(right_key)
                incoming.setdefault(right_key, set()).add(left_key)
                edge_indexes.setdefault((left_key, right_key), set()).add(index)

        all_keys = set(node_by_key)
        visited: set[tuple[str, str]] = set()
        for start in sorted(all_keys):
            if start in visited or incoming.get(start):
                continue
            chain = [start]
            current = start
            local_seen = {start}
            while True:
                next_nodes = outgoing.get(current, set())
                if len(next_nodes) != 1:
                    break
                next_key = next(iter(next_nodes))
                if len(incoming.get(next_key, set())) != 1 or next_key in local_seen:
                    break
                chain.append(next_key)
                local_seen.add(next_key)
                current = next_key
            if len(chain) < 3:
                visited.update(chain)
                continue
            if any(len(outgoing.get(key, set())) > 1 or len(incoming.get(key, set())) > 1 for key in chain):
                visited.update(chain)
                continue
            component_edges = {(chain[index], chain[index + 1]) for index in range(len(chain) - 1)}
            participating_indexes = set().union(*(edge_indexes.get(edge, set()) for edge in component_edges))
            if len(participating_indexes) < 2:
                visited.update(chain)
                continue
            merged_region = {
                "relation_type": relation_type,
                "structure": {
                    "type": "sequence",
                    "items": [deepcopy(node_by_key[key]) for key in chain],
                },
            }
            first_index = min(participating_indexes)
            replacements[first_index] = merged_region
            remove_indexes.update(participating_indexes - {first_index})
            visited.update(chain)

    if not replacements and not remove_indexes:
        return
    workflow_ir["control_flow_regions"] = [
        replacements.get(index, region)
        for index, region in enumerate(regions)
        if index not in remove_indexes
    ]


def _parallel_branch_keys(structure: dict[str, Any]) -> tuple[tuple[str, str], ...] | None:
    if structure.get("type") != "parallel" or structure.get("join_mode") != "all":
        return None
    branches = structure.get("branches", [])
    if not isinstance(branches, list) or len(branches) < 2:
        return None
    keys: list[tuple[str, str]] = []
    for branch in branches:
        key = _sequence_node_key(branch) if isinstance(branch, dict) else None
        if key is None:
            return None
        keys.append(key)
    if len(set(keys)) != len(keys):
        return None
    return tuple(keys)


def _normalize_parallel_common_continuations(workflow_ir: dict[str, Any]) -> None:
    regions = workflow_ir.get("control_flow_regions")
    if not isinstance(regions, list) or len(regions) < 2:
        return

    sequence_edges: dict[str, dict[tuple[tuple[str, str], tuple[str, str]], tuple[int, list[dict[str, Any]]]]] = {}
    for index, region in enumerate(regions):
        items = _node_only_sequence_items(region)
        if items is None or len(items) != 2:
            continue
        source_key = _sequence_node_key(items[0])
        target_key = _sequence_node_key(items[1])
        if source_key is None or target_key is None or source_key == target_key:
            continue
        sequence_edges.setdefault(region["relation_type"], {})[(source_key, target_key)] = (index, items)

    replacements: dict[int, dict[str, Any]] = {}
    remove_indexes: set[int] = set()
    for index, region in enumerate(regions):
        if index in remove_indexes:
            continue
        structure = region.get("structure") if isinstance(region, dict) else None
        if not isinstance(structure, dict):
            continue
        branch_keys = _parallel_branch_keys(structure)
        if branch_keys is None:
            continue
        relation_type = region.get("relation_type")
        edges_for_relation = sequence_edges.get(relation_type, {})
        targets: list[tuple[str, str]] = []
        participating_indexes: set[int] = set()
        continuation_node: dict[str, Any] | None = None
        complete = True
        for branch_key in branch_keys:
            matches = [
                (target_key, edge_index, items)
                for (source_key, target_key), (edge_index, items) in edges_for_relation.items()
                if source_key == branch_key
            ]
            if len(matches) != 1:
                complete = False
                break
            target_key, edge_index, items = matches[0]
            targets.append(target_key)
            participating_indexes.add(edge_index)
            if continuation_node is None:
                continuation_node = items[1]
        if not complete or continuation_node is None:
            continue
        if len(set(targets)) != 1:
            continue
        merged_region = {
            "relation_type": relation_type,
            "structure": {
                "type": "sequence",
                "items": [deepcopy(structure), deepcopy(continuation_node)],
            },
        }
        replacements[index] = merged_region
        remove_indexes.update(participating_indexes)

    if not replacements and not remove_indexes:
        return
    workflow_ir["control_flow_regions"] = [
        replacements.get(index, region)
        for index, region in enumerate(regions)
        if index not in remove_indexes
    ]


def _plain_relation_edges(graph_data: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        edge
        for edge in graph_data.get("edges", [])
        if isinstance(edge, dict)
        and edge.get("label", "") == ""
        and not edge.get("branch_label")
        and not edge.get("condition")
        and isinstance(edge.get("source"), str)
        and isinstance(edge.get("target"), str)
    ]


def _graph_plain_chain_for_nodes(graph_data: dict[str, Any], node_ids: set[str]) -> list[str] | None:
    if len(node_ids) < 3:
        return None
    internal_edges = [
        edge
        for edge in _plain_relation_edges(graph_data)
        if edge["source"] in node_ids and edge["target"] in node_ids
    ]
    if len(internal_edges) != len(node_ids) - 1:
        return None
    incoming: dict[str, list[str]] = {node_id: [] for node_id in node_ids}
    outgoing: dict[str, list[str]] = {node_id: [] for node_id in node_ids}
    for edge in internal_edges:
        incoming[edge["target"]].append(edge["source"])
        outgoing[edge["source"]].append(edge["target"])
    if any(len(values) > 1 for values in incoming.values()) or any(len(values) > 1 for values in outgoing.values()):
        return None
    heads = [node_id for node_id in node_ids if not incoming[node_id]]
    tails = [node_id for node_id in node_ids if not outgoing[node_id]]
    if len(heads) != 1 or len(tails) != 1:
        return None
    ordered = [heads[0]]
    while outgoing[ordered[-1]]:
        ordered.append(outgoing[ordered[-1]][0])
    return ordered if set(ordered) == node_ids and len(ordered) == len(node_ids) else None


def _coherent_precedence_sequence_items(
    relation_fact: dict[str, Any],
    graph_data: dict[str, Any],
    node_for_relation_binding: Any,
) -> list[dict[str, Any]] | None:
    if relation_fact.get("kind") != "precedence":
        return None
    members = relation_fact.get("members")
    if not isinstance(members, list) or len(members) < 2:
        return None

    node_items: dict[str, dict[str, Any]] = {}
    outgoing: dict[str, set[str]] = {}
    incoming: dict[str, set[str]] = {}
    for member in members:
        if not isinstance(member, dict):
            return None
        source_node = node_for_relation_binding(member.get("source_binding"))
        target_node = node_for_relation_binding(member.get("target_binding"))
        if not (
            isinstance(source_node, dict)
            and isinstance(source_node.get("id"), str)
            and isinstance(target_node, dict)
            and isinstance(target_node.get("id"), str)
        ):
            return None
        source_id = source_node["id"]
        target_id = target_node["id"]
        if source_id == target_id:
            return None
        node_items.setdefault(source_id, source_node)
        node_items.setdefault(target_id, target_node)
        outgoing.setdefault(source_id, set()).add(target_id)
        incoming.setdefault(target_id, set()).add(source_id)

    node_ids = set(node_items)
    if len(node_ids) < 3 or sum(len(values) for values in outgoing.values()) != len(node_ids) - 1:
        return None
    if any(len(values) > 1 for values in outgoing.values()) or any(len(values) > 1 for values in incoming.values()):
        return None
    starts = [node_id for node_id in node_ids if node_id not in incoming]
    ends = [node_id for node_id in node_ids if node_id not in outgoing]
    if len(starts) != 1 or len(ends) != 1:
        return None
    target_order = [starts[0]]
    while target_order[-1] in outgoing:
        next_nodes = outgoing[target_order[-1]]
        if len(next_nodes) != 1:
            return None
        next_node = next(iter(next_nodes))
        if next_node in target_order:
            return None
        target_order.append(next_node)
    if set(target_order) != node_ids or len(target_order) != len(node_ids):
        return None

    current_order = _graph_plain_chain_for_nodes(graph_data, node_ids)
    if current_order is None or current_order == target_order:
        return None
    return [deepcopy(node_items[node_id]) for node_id in target_order]


def _node_property_change(state: dict[str, Any]) -> dict[str, str] | None:
    prop = str(state.get("property", "")).strip()
    value = state.get("value")
    if prop not in SAFE_NODE_DISPLAY_PROPERTIES or not isinstance(value, str) or not value.strip():
        return None
    return {"label": value.strip()}


def _edge_property_change_from_state(state: dict[str, Any], current_edge: dict[str, Any] | None = None) -> dict[str, Any] | None:
    prop = str(state.get("property", "")).strip()
    value = state.get("value")
    if not isinstance(value, str) or not value.strip():
        return None
    if prop in {"label", "branch_label", "display_label"}:
        return {"branch_label": value.strip()}
    if prop == "condition.expression":
        return normalize_edge_changes({"condition": {"expression": value.strip()}}, current_edge)
    if prop == "condition":
        return normalize_edge_changes({"condition_expression": value.strip()}, current_edge)
    return None


def _semantic_spec_from_ir(workflow_ir: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": "resolved",
        "workflow_ir": normalize_ir(workflow_ir),
        "semantic_goals": [],
        "clarification": {"needed": False, "question": "", "reason": ""},
    }


def _v2_object_by_id(goal: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        item["id"]: item
        for item in goal.get("objects", [])
        if isinstance(item, dict) and isinstance(item.get("id"), str)
    }


def _workflow_ir_has_work(workflow_ir: dict[str, Any]) -> bool:
    return any(bool(workflow_ir.get(key)) for key in ("entities", "control_flow_regions", "auxiliary_relations", "properties", "edge_properties", "absent_entities"))


def _preserved_state_ids_for_object(grounded_goal: dict[str, Any], object_id: Any) -> list[str]:
    object_id = str(object_id or "").strip()
    if not object_id:
        return []
    result: list[str] = []
    for goal in grounded_goal.get("semantic_goals_v2", []) if isinstance(grounded_goal.get("semantic_goals_v2"), list) else []:
        if not isinstance(goal, dict):
            continue
        for state_index, state in enumerate(goal.get("object_desired_states", []) if isinstance(goal.get("object_desired_states"), list) else []):
            if isinstance(state, dict) and state.get("object") == object_id and state.get("state") == "preserved":
                result.append(f"{goal.get('id', 'g')}:state:{state_index}")
    return result


def _subgoal_description(kind: str, goal: dict[str, Any], row: dict[str, Any]) -> str:
    if kind == "state":
        return f"{row.get('object')} {row.get('state')}"
    return f"{row.get('source')} {row.get('relation')} {row.get('target')}"


def _has_exists_state_for_object(grounded_goal: dict[str, Any], object_id: Any) -> bool:
    object_id = str(object_id or "").strip()
    if not object_id:
        return False
    for goal in grounded_goal.get("semantic_goals_v2", []) if isinstance(grounded_goal.get("semantic_goals_v2"), list) else []:
        if not isinstance(goal, dict):
            continue
        for state in goal.get("object_desired_states", []) if isinstance(goal.get("object_desired_states"), list) else []:
            if isinstance(state, dict) and state.get("object") == object_id and state.get("state") == "exists":
                return True
    return False


def _exists_state_ids_for_object(grounded_goal: dict[str, Any], object_id: Any) -> list[str]:
    object_id = str(object_id or "").strip()
    if not object_id:
        return []
    result: list[str] = []
    for goal in grounded_goal.get("semantic_goals_v2", []) if isinstance(grounded_goal.get("semantic_goals_v2"), list) else []:
        if not isinstance(goal, dict):
            continue
        for state_index, state in enumerate(goal.get("object_desired_states", []) if isinstance(goal.get("object_desired_states"), list) else []):
            if isinstance(state, dict) and state.get("object") == object_id and state.get("state") == "exists":
                result.append(f"{goal.get('id', 'g')}:state:{state_index}")
    return result


def _canonical_new_entity_label(label: Any) -> str:
    normalized = re.sub(r"\s+", " ", _strip_outer_reference_quotes(label).lower())
    for suffix in ("节点", "步骤", "环节", "node", "step", "task"):
        if normalized.endswith(suffix):
            normalized = normalized[: -len(suffix)].strip()
            break
    return normalized


def _new_entity_exists_state_ids_for_label(grounded_goal: dict[str, Any], label: Any) -> list[str]:
    needle = _canonical_new_entity_label(label)
    if not needle:
        return []
    result: list[str] = []
    for goal in grounded_goal.get("semantic_goals_v2", []) if isinstance(grounded_goal.get("semantic_goals_v2"), list) else []:
        if not isinstance(goal, dict):
            continue
        objects = _v2_object_by_id(goal)
        for state_index, state in enumerate(goal.get("object_desired_states", []) if isinstance(goal.get("object_desired_states"), list) else []):
            if not isinstance(state, dict) or state.get("state") != "exists":
                continue
            item = objects.get(str(state.get("object", "")).strip())
            if isinstance(item, dict) and item.get("type") == "new_entity" and _canonical_new_entity_label(item.get("label")) == needle:
                result.append(f"{goal.get('id', 'g')}:state:{state_index}")
    return result


def _has_exists_state_for_new_entity_label(grounded_goal: dict[str, Any], label: Any) -> bool:
    return bool(_new_entity_exists_state_ids_for_label(grounded_goal, label))


def _new_entity_exists_binding_for_label(grounded_goal: dict[str, Any], label: Any) -> dict[str, Any] | None:
    needle = _canonical_new_entity_label(label)
    if not needle:
        return None
    for goal in grounded_goal.get("semantic_goals", []) if isinstance(grounded_goal.get("semantic_goals"), list) else []:
        if not isinstance(goal, dict) or goal.get("kind") != "object_state":
            continue
        state = goal.get("object_state") if isinstance(goal.get("object_state"), dict) else {}
        if state.get("state") != "exists":
            continue
        binding = _binding_by_role(goal, "object")
        if isinstance(binding, dict) and binding.get("needs_creation") is True and _canonical_new_entity_label(binding.get("label")) == needle:
            return binding
    return None


def _resolve_route_target_existing_node(graph_data: dict[str, Any], label: Any) -> tuple[str | None, list[str], dict[str, Any]]:
    phrase = str(label or "").strip()
    if not phrase:
        return None, [], {"resolution_type": "missing"}
    node_id, ambiguous, metadata = resolve_semantic_node_reference(graph_data, phrase)
    return node_id, ambiguous, metadata


def _route_same_target_conflict(
    graph_data: dict[str, Any],
    *,
    current_edge_id: Any,
    source: str,
    relation_type: str,
    target_id: str,
) -> dict[str, Any] | None:
    if not isinstance(current_edge_id, str):
        return None
    current_label = relation_type
    for edge in graph_data.get("edges", []):
        if not isinstance(edge, dict) or edge.get("id") == current_edge_id:
            continue
        if edge.get("source") != source or edge.get("target") != target_id:
            continue
        other_label = edge.get("branch_label") or edge.get("label", "")
        if other_label == current_label:
            continue
        return {
            "kind": "route_same_source_target_confirmation",
            "message": "another branch from the same source already routes to the requested target",
            "severity": "blocking",
            "location": "semantic_goals.route",
            "edge_id": current_edge_id,
            "source": source,
            "target": target_id,
            "relation_type": relation_type,
            "conflicting_edge_id": edge.get("id"),
            "conflicting_relation_type": edge.get("label", ""),
            "conflicting_branch_label": edge.get("branch_label", edge.get("label", "")),
        }
    return None


def _unsupported_state_reason(state: dict[str, Any], item: dict[str, Any] | None) -> str:
    state_name = state.get("state")
    if state_name == "exists":
        return "exists state was not lowered"
    if state_name == "absent":
        return "object absence could not be lowered"
    if state_name == "preserved":
        return "preserve-only state is a semantic constraint and is not executable by itself"
    if state_name == "property_equals":
        return "this object property is not supported by current lowering"
    return "object desired state is not supported by current lowering"


def _build_partial_contract(
    grounded_goal: dict[str, Any],
    executed_subgoal_ids: set[str] | None = None,
    skipped_subgoal_ids: set[str] | None = None,
    lowering_issue_subgoal_ids: set[str] | None = None,
) -> dict[str, Any]:
    executed_subgoal_ids = executed_subgoal_ids or set()
    skipped_subgoal_ids = skipped_subgoal_ids or set()
    lowering_issue_subgoal_ids = lowering_issue_subgoal_ids or set()
    v2_goals = grounded_goal.get("semantic_goals_v2", []) if isinstance(grounded_goal.get("semantic_goals_v2"), list) else []
    v1_goals = grounded_goal.get("semantic_goals", []) if isinstance(grounded_goal.get("semantic_goals"), list) else []
    grounding_issues = grounded_goal.get("grounding", {}).get("issues", []) if isinstance(grounded_goal.get("grounding"), dict) else []
    subgoals: list[dict[str, Any]] = []
    unsupported_exists_objects: set[str] = set()

    v1_by_subgoal = {
        row.get("source_subgoal_id"): row
        for row in v1_goals
        if isinstance(row, dict) and isinstance(row.get("source_subgoal_id"), str)
    }
    issue_locations = [
        str(issue.get("location", ""))
        for issue in grounding_issues
        if isinstance(issue, dict)
    ]

    def grounding_status_for_subgoal(source_subgoal_id: str) -> str | None:
        v1_goal = v1_by_subgoal.get(source_subgoal_id)
        if not isinstance(v1_goal, dict):
            return None
        try:
            index = v1_goals.index(v1_goal)
        except ValueError:
            return None
        prefix = f"semantic_goals[{index}]"
        relevant = [
            issue
            for issue in grounding_issues
            if isinstance(issue, dict) and str(issue.get("location", "")).startswith(prefix)
        ]
        if any(str(issue.get("kind", "")).startswith("ambiguous_") for issue in relevant):
            return "ambiguous"
        if any(str(issue.get("kind", "")).startswith("missing_") for issue in relevant):
            return "missing"
        return None

    for goal in v2_goals:
        if not isinstance(goal, dict):
            continue
        objects = _v2_object_by_id(goal)
        for state_index, state in enumerate(goal.get("object_desired_states", []) if isinstance(goal.get("object_desired_states"), list) else []):
            if not isinstance(state, dict):
                continue
            subgoal_id = f"{goal.get('id', 'g')}:state:{state_index}"
            object_id = str(state.get("object", "")).strip()
            item = objects.get(object_id)
            status = "executed" if subgoal_id in executed_subgoal_ids else "unsupported"
            reason = ""
            grounding_status = grounding_status_for_subgoal(subgoal_id)
            if grounding_status:
                status = grounding_status
                reason = "grounding did not produce a unique executable binding"
            elif subgoal_id in lowering_issue_subgoal_ids:
                status = "unsupported"
                reason = "supported semantic state could not be lowered"
            elif status != "executed":
                reason = _unsupported_state_reason(state, item)
            if state.get("state") == "exists" and status == "unsupported" and item and item.get("type") == "new_entity":
                unsupported_exists_objects.add(object_id)
            subgoals.append(
                {
                    "id": subgoal_id,
                    "type": "object_desired_state",
                    "status": status,
                    "object": object_id,
                    "state": state.get("state"),
                    "description": _subgoal_description("state", goal, state),
                    "reason": reason,
                }
            )
        for relation_index, relation in enumerate(goal.get("relation_intents", []) if isinstance(goal.get("relation_intents"), list) else []):
            if not isinstance(relation, dict):
                continue
            subgoal_id = f"{goal.get('id', 'g')}:relation:{relation_index}"
            relation_type = relation.get("relation")
            source = str(relation.get("source", "")).strip()
            target = str(relation.get("target", "")).strip()
            status = "executed" if subgoal_id in executed_subgoal_ids else "unsupported"
            reason = ""
            dependency_objects = unsupported_exists_objects & {source, target}
            ordering_creates_subject = relation_type in {"before", "after"} and source in unsupported_exists_objects
            if status != "executed" and (subgoal_id in skipped_subgoal_ids or (dependency_objects and not ordering_creates_subject)):
                status = "skipped_due_to_dependency"
                reason = "depends on an object existence goal that was not completed"
            elif subgoal_id in lowering_issue_subgoal_ids:
                status = "unsupported"
                reason = "supported semantic relation could not be lowered"
            elif status != "executed":
                grounding_status = grounding_status_for_subgoal(subgoal_id)
                if grounding_status:
                    status = grounding_status
                    reason = "grounding did not produce a unique executable binding"
                elif relation_type not in {"before", "after", "right_of", "route_to", "parallel_with"}:
                    reason = f"relation intent {relation_type} is not supported by current lowering"
                else:
                    reason = "relation intent was not lowered"
            subgoals.append(
                {
                    "id": subgoal_id,
                    "type": "relation_intent",
                    "status": status,
                    "source": source,
                    "relation": relation_type,
                    "target": target,
                    "description": _subgoal_description("relation", goal, relation),
                    "reason": reason,
                }
            )

    executed_count = len([row for row in subgoals if row.get("status") == "executed"])
    incomplete = [row for row in subgoals if row.get("status") != "executed"]
    if not subgoals and (
        grounding_issues
        or grounded_goal.get("status") == "clarification"
        or (isinstance(grounded_goal.get("clarification"), dict) and grounded_goal["clarification"].get("needed") is True)
    ):
        final_status = "needs_clarification"
    elif not subgoals:
        final_status = "success"
    elif not incomplete:
        final_status = "success"
    elif executed_count:
        final_status = "partial_success"
    elif any(row.get("status") in {"ambiguous", "missing", "skipped_due_to_dependency"} for row in subgoals):
        final_status = "needs_clarification"
    else:
        final_status = "unsupported"
    return {
        "subgoals": subgoals,
        "executed_count": executed_count,
        "incomplete_count": len(incomplete),
        "final_status": final_status,
    }


def partial_execution_contract_for_grounded_goal(grounded_goal: dict[str, Any]) -> dict[str, Any]:
    return _build_partial_contract(grounded_goal)


def _parallel_nodes_are_safe_chain(graph_data: dict[str, Any], left: str, right: str) -> bool:
    if left == right:
        return False
    blank_edges = [
        edge
        for edge in graph_data.get("edges", [])
        if isinstance(edge, dict)
        and isinstance(edge.get("source"), str)
        and isinstance(edge.get("target"), str)
        and edge.get("label", "") == ""
    ]
    return any(
        (edge.get("source") == left and edge.get("target") == right)
        or (edge.get("source") == right and edge.get("target") == left)
        for edge in blank_edges
    )


def lower_grounded_goal_to_workflow_ir(grounded_goal: dict[str, Any], graph_data: dict[str, Any]) -> dict[str, Any]:
    """Lower a grounded semantic goal into the existing Workflow IR schema.

    This function intentionally emits only Workflow IR fields. It does not emit
    planner constraints, edge diffs, mutation steps, or compiler boundary data.
    """
    nodes = _nodes_by_id(graph_data)
    workflow_ir = _empty_workflow_ir()
    preserve_constraints = grounded_goal.get("preserve_constraints") if isinstance(grounded_goal.get("preserve_constraints"), dict) else {}
    workflow_ir["preserve"] = {
        "node_ids": sorted({node_id for node_id in preserve_constraints.get("node_ids", []) if isinstance(node_id, str) and node_id}),
        "edge_ids": sorted({edge_id for edge_id in preserve_constraints.get("edge_ids", []) if isinstance(edge_id, str) and edge_id}),
        "abstract": [
            deepcopy(row)
            for row in preserve_constraints.get("abstract", [])
            if isinstance(row, dict)
        ],
    }
    issues: list[dict[str, Any]] = []
    executed_subgoal_ids: set[str] = set()
    lowering_issue_subgoal_ids: set[str] = set()
    skipped_subgoal_ids: set[str] = set()
    created_new_entities: dict[str, dict[str, Any]] = {}
    allow_route_same_target = grounded_goal.get("_allow_route_same_target_confirmation") is True

    for goal in grounded_goal.get("semantic_goals", []) if isinstance(grounded_goal.get("semantic_goals"), list) else []:
        if not isinstance(goal, dict) or goal.get("kind") != "object_state":
            continue
        state = goal.get("object_state") if isinstance(goal.get("object_state"), dict) else {}
        if state.get("state") != "preserved":
            continue
        binding = _binding_by_role(goal, "object")
        if not isinstance(binding, dict):
            continue
        if binding.get("binding_type") == "edge" and isinstance(binding.get("id"), str):
            workflow_ir["preserve"]["edge_ids"] = sorted({*workflow_ir["preserve"].get("edge_ids", []), binding["id"]})
            for subgoal_id in _preserved_state_ids_for_object(grounded_goal, binding.get("v2_object_id")):
                executed_subgoal_ids.add(subgoal_id)
        elif binding.get("binding_type") == "node" and isinstance(binding.get("id"), str):
            workflow_ir["preserve"]["node_ids"] = sorted({*workflow_ir["preserve"].get("node_ids", []), binding["id"]})
            for subgoal_id in _preserved_state_ids_for_object(grounded_goal, binding.get("v2_object_id")):
                executed_subgoal_ids.add(subgoal_id)

    pre_contract = _build_partial_contract(grounded_goal, executed_subgoal_ids=executed_subgoal_ids)

    def node_for_relation_binding(binding: Any) -> dict[str, Any] | None:
        if not isinstance(binding, dict):
            return None
        row = dict(binding)
        ref_id = row.get("ref_id")
        if isinstance(ref_id, str) and ref_id and not isinstance(row.get("v2_object_id"), str):
            row["v2_object_id"] = ref_id
        return _node_for_new_entity(row, created_new_entities)

    def append_relation_fact_fragments(relation_fact: dict[str, Any], relation_index: int) -> bool:
        relation_id = relation_fact.get("relation_id")
        kind = relation_fact.get("kind")
        members = relation_fact.get("members")
        if not (isinstance(relation_id, str) and relation_id and kind in {"precedence", "connection"} and isinstance(members, list)):
            issues.append(
                {
                    "kind": "unsupported_relation_fact_lowering",
                    "message": "RelationFact lowering requires a managed relation_id, supported kind, and members",
                    "severity": "blocking",
                    "location": f"relation_facts[{relation_index}]",
                }
            )
            return False
        lowered_any = False
        scope_id = f"relation_fact:{relation_id}"
        coherent_items = _coherent_precedence_sequence_items(relation_fact, graph_data, node_for_relation_binding)
        if coherent_items is not None:
            workflow_ir["control_flow_regions"].append(
                {
                    "relation_type": "",
                    "lowering_scope_id": scope_id,
                    "structure": {"type": "sequence", "items": coherent_items},
                }
            )
            return True
        for member_index, member in enumerate(members):
            if not isinstance(member, dict):
                continue
            source_node = node_for_relation_binding(member.get("source_binding"))
            target_node = node_for_relation_binding(member.get("target_binding"))
            if source_node is None or target_node is None:
                issues.append(
                    {
                        "kind": "missing_relation_fact_lowering_binding",
                        "message": "RelationFact member lowering requires node or creatable new entity endpoints",
                        "severity": "blocking",
                        "location": f"relation_facts[{relation_index}].members[{member_index}]",
                    }
                )
                continue
            source_item = _declare_and_occurrence(workflow_ir, source_node)
            target_item = _declare_and_occurrence(workflow_ir, target_node)
            if kind == "connection":
                source_id = source_item.get("id") if isinstance(source_item.get("id"), str) else source_item.get("ref")
                target_id = target_item.get("id") if isinstance(target_item.get("id"), str) else target_item.get("ref")
                if isinstance(source_id, str) and isinstance(target_id, str):
                    workflow_ir["auxiliary_relations"].append({"source": source_id, "relation_type": "", "targets": [target_id]})
                    lowered_any = True
                continue
            workflow_ir["control_flow_regions"].append(
                {
                    "relation_type": "",
                    "lowering_scope_id": scope_id,
                    "structure": {"type": "sequence", "items": [source_item, target_item]},
                }
            )
            lowered_any = True
        return lowered_any

    for relation_index, relation_fact in enumerate(
        grounded_goal.get("relation_facts", []) if isinstance(grounded_goal.get("relation_facts"), list) else []
    ):
        if isinstance(relation_fact, dict):
            if append_relation_fact_fragments(relation_fact, relation_index):
                for suppressed in grounded_goal.get("suppressed_legacy_relation_goals", []) if isinstance(grounded_goal.get("suppressed_legacy_relation_goals"), list) else []:
                    source_subgoal_id = suppressed.get("source_subgoal_id") if isinstance(suppressed, dict) else None
                    if isinstance(source_subgoal_id, str) and source_subgoal_id:
                        executed_subgoal_ids.add(source_subgoal_id)

    for index, goal in enumerate(grounded_goal.get("semantic_goals", []) if isinstance(grounded_goal.get("semantic_goals"), list) else []):
        if not isinstance(goal, dict):
            continue
        source_subgoal_id = goal.get("source_subgoal_id") if isinstance(goal.get("source_subgoal_id"), str) else ""
        if source_subgoal_id in skipped_subgoal_ids:
            continue
        if goal.get("kind") == "object_state":
            binding = _binding_by_role(goal, "object")
            state = goal.get("object_state") if isinstance(goal.get("object_state"), dict) else {}
            state_name = state.get("state")
            if state_name == "exists":
                node = _node_for_new_entity(binding, created_new_entities) if isinstance(binding, dict) else None
                if isinstance(binding, dict) and binding.get("binding_type") == "node":
                    if source_subgoal_id:
                        executed_subgoal_ids.add(source_subgoal_id)
                    continue
                if node is None or "ref" not in node:
                    issues.append(
                        {
                            "kind": "missing_exists_lowering_binding",
                            "message": "exists lowering requires a new entity label or existing node binding",
                            "severity": "blocking",
                            "location": f"semantic_goals[{index}].bindings",
                        }
                    )
                    if source_subgoal_id:
                        lowering_issue_subgoal_ids.add(source_subgoal_id)
                    continue
                _ensure_new_node_layout(graph_data, node)
                _append_creation_region(workflow_ir, node, graph_data)
                if source_subgoal_id:
                    executed_subgoal_ids.add(source_subgoal_id)
                continue
            if state_name == "absent":
                if isinstance(binding, dict) and binding.get("binding_type") == "node" and isinstance(binding.get("id"), str):
                    workflow_ir["absent_entities"].append({"node_id": binding["id"]})
                    if source_subgoal_id:
                        executed_subgoal_ids.add(source_subgoal_id)
                    continue
                issues.append(
                    {
                        "kind": "unsupported_absent_lowering_target",
                        "message": "absent lowering currently supports existing nodes only",
                        "severity": "blocking",
                        "location": f"semantic_goals[{index}].bindings",
                    }
                )
                if source_subgoal_id:
                    lowering_issue_subgoal_ids.add(source_subgoal_id)
                continue
            if state_name == "preserved":
                if source_subgoal_id in executed_subgoal_ids:
                    continue
                issues.append(
                    {
                        "kind": "unsupported_object_state_lowering",
                        "message": "preserve state requires a grounded node or edge binding",
                        "severity": "blocking",
                        "location": f"semantic_goals[{index}].object_state",
                    }
                )
                if source_subgoal_id:
                    lowering_issue_subgoal_ids.add(source_subgoal_id)
                continue
            if state_name == "property_equals":
                if isinstance(binding, dict) and binding.get("binding_type") == "node" and isinstance(binding.get("id"), str):
                    changes = _node_property_change(state)
                    if changes is not None:
                        workflow_ir["properties"].append({"node_id": binding["id"], "changes": changes})
                        if source_subgoal_id:
                            executed_subgoal_ids.add(source_subgoal_id)
                        continue
                if isinstance(binding, dict) and binding.get("binding_type") == "edge" and isinstance(binding.get("id"), str):
                    current_edge = next(
                        (
                            edge
                            for edge in graph_data.get("edges", [])
                            if isinstance(edge, dict) and edge.get("id") == binding["id"]
                        ),
                        None,
                    )
                    changes = _edge_property_change_from_state(state, current_edge)
                    if changes is not None:
                        workflow_ir["edge_properties"].append({"edge_id": binding["id"], "changes": changes})
                        if source_subgoal_id:
                            executed_subgoal_ids.add(source_subgoal_id)
                        continue
                issues.append(
                    {
                        "kind": "unsupported_property_lowering",
                        "message": "property_equals lowering only supports node labels and branch labels",
                        "severity": "blocking",
                        "location": f"semantic_goals[{index}].object_state",
                    }
                )
                if source_subgoal_id:
                    lowering_issue_subgoal_ids.add(source_subgoal_id)
                continue
            issues.append(
                {
                    "kind": "unsupported_object_state_lowering",
                    "message": "this object desired state cannot be lowered to current Workflow IR",
                    "severity": "blocking",
                    "location": f"semantic_goals[{index}].object_state",
                }
            )
            if source_subgoal_id:
                lowering_issue_subgoal_ids.add(source_subgoal_id)
            continue
        if goal.get("kind") == "route":
            route_binding = _binding_by_role(goal, "route")
            target_binding = _binding_by_role(goal, "target")
            route_facts = route_binding.get("graph_facts", {}) if isinstance(route_binding, dict) else {}
            source = route_facts.get("source")
            relation_type = route_facts.get("label", "")
            target_id = target_binding.get("id") if isinstance(target_binding, dict) and target_binding.get("binding_type") == "node" else None
            if target_id is None and isinstance(target_binding, dict) and target_binding.get("needs_creation") is True:
                has_target_exists_state = _has_exists_state_for_object(grounded_goal, target_binding.get("v2_object_id"))
                if has_target_exists_state:
                    target_node = _node_for_new_entity(target_binding, created_new_entities)
                    if target_node is not None and isinstance(target_node.get("ref"), str):
                        route_candidates: list[tuple[float, float]] = []
                        route_target = route_facts.get("target")
                        if isinstance(route_facts.get("source"), str) and isinstance(route_target, str):
                            route_candidates = _edge_split_candidates(graph_data, route_facts["source"], route_target)
                        _ensure_new_node_layout(graph_data, target_node, candidates=route_candidates)
                        _append_creation_region(workflow_ir, target_node, graph_data)
                        target_id = target_node["ref"]
                else:
                    resolved_id, ambiguous, metadata = _resolve_route_target_existing_node(graph_data, target_binding.get("label"))
                    if isinstance(resolved_id, str):
                        target_id = resolved_id
                    elif ambiguous:
                        issues.append(
                            {
                                "kind": "ambiguous_node_reference",
                                "message": "route target matches multiple existing nodes; choose one target or request creating a new node",
                                "severity": "blocking",
                                "location": f"semantic_goals[{index}].bindings",
                                "reference": target_binding.get("label", ""),
                                "candidates": ambiguous,
                                "metadata": metadata,
                            }
                        )
                        if source_subgoal_id:
                            lowering_issue_subgoal_ids.add(source_subgoal_id)
                        continue
                    else:
                        issues.append(
                            {
                                "kind": "missing_node_reference",
                                "message": "route target does not match an existing node; ask whether to create a new target node",
                                "severity": "blocking",
                                "location": f"semantic_goals[{index}].bindings",
                                "reference": target_binding.get("label", ""),
                                "candidates": [],
                                "metadata": metadata,
                            }
                        )
                        if source_subgoal_id:
                            lowering_issue_subgoal_ids.add(source_subgoal_id)
                        continue
            if not (isinstance(source, str) and isinstance(relation_type, str) and isinstance(target_id, str)):
                issues.append(
                    {
                        "kind": "missing_route_lowering_binding",
                        "message": "route lowering requires a grounded route edge and a target node or creatable target",
                        "severity": "blocking",
                        "location": f"semantic_goals[{index}].bindings",
                    }
                )
                if source_subgoal_id:
                    lowering_issue_subgoal_ids.add(source_subgoal_id)
                continue
            route_edge_id = route_binding.get("id") if isinstance(route_binding, dict) else None
            conflict = None if allow_route_same_target else _route_same_target_conflict(
                graph_data,
                current_edge_id=route_edge_id,
                source=source,
                relation_type=relation_type,
                target_id=target_id,
            )
            if conflict is not None:
                issues.append(conflict)
                if source_subgoal_id:
                    lowering_issue_subgoal_ids.add(source_subgoal_id)
                continue
            workflow_ir["auxiliary_relations"].append(
                {
                    "source": source,
                    "relation_type": relation_type,
                    "targets": [target_id],
                }
            )
            if source_subgoal_id:
                executed_subgoal_ids.add(source_subgoal_id)
                if isinstance(target_binding, dict) and target_binding.get("needs_creation") is True:
                    for row in pre_contract.get("subgoals", []):
                        if row.get("type") == "object_desired_state" and row.get("object") == target_binding.get("v2_object_id"):
                            executed_subgoal_ids.add(str(row.get("id")))
                    for subgoal_id in _exists_state_ids_for_object(grounded_goal, target_binding.get("v2_object_id")):
                        executed_subgoal_ids.add(subgoal_id)
            continue

        if goal.get("kind") == "connection":
            source_binding = _binding_by_role(goal, "source")
            target_binding = _binding_by_role(goal, "target")
            source_node = _node_for_new_entity(source_binding, created_new_entities) if isinstance(source_binding, dict) else None
            target_node = _node_for_new_entity(target_binding, created_new_entities) if isinstance(target_binding, dict) else None
            source_id = source_node.get("id") if isinstance(source_node, dict) and isinstance(source_node.get("id"), str) else None
            target_id = target_node.get("id") if isinstance(target_node, dict) and isinstance(target_node.get("id"), str) else None
            if not (isinstance(source_id, str) and isinstance(target_id, str)):
                issues.append(
                    {
                        "kind": "missing_connection_lowering_binding",
                        "message": "connection lowering requires grounded existing source and target nodes",
                        "severity": "blocking",
                        "location": f"semantic_goals[{index}].bindings",
                    }
                )
                if source_subgoal_id:
                    lowering_issue_subgoal_ids.add(source_subgoal_id)
                continue
            workflow_ir["auxiliary_relations"].append(
                {
                    "source": source_id,
                    "relation_type": "",
                    "targets": [target_id],
                }
            )
            if source_subgoal_id:
                executed_subgoal_ids.add(source_subgoal_id)
            continue

        if goal.get("kind") == "branch_property":
            edge_binding = _binding_by_role(goal, "branch")
            edge_id = edge_binding.get("id") if isinstance(edge_binding, dict) and edge_binding.get("binding_type") == "edge" else None
            changes = _branch_property_change(goal)
            if not isinstance(edge_id, str) or changes is None:
                issues.append(
                    {
                        "kind": "missing_branch_property_lowering_binding",
                        "message": "branch_property lowering requires a grounded edge and supported property change",
                        "severity": "blocking",
                        "location": f"semantic_goals[{index}].bindings",
                    }
                )
                if source_subgoal_id:
                    lowering_issue_subgoal_ids.add(source_subgoal_id)
                continue
            workflow_ir["edge_properties"].append({"edge_id": edge_id, "changes": changes})
            if source_subgoal_id:
                executed_subgoal_ids.add(source_subgoal_id)
            continue

        if goal.get("kind") == "ordering" and goal.get("relation") in {"before", "after"}:
            subject_binding = _binding_by_role(goal, "subject")
            anchor_binding = _binding_by_role(goal, "anchor")
            subject_node = _node_for_new_entity(subject_binding, created_new_entities) if isinstance(subject_binding, dict) else None
            anchor_node = _node_for_new_entity(anchor_binding, created_new_entities) if isinstance(anchor_binding, dict) else None
            if subject_node is None or anchor_node is None:
                issues.append(
                    {
                        "kind": "missing_ordering_lowering_binding",
                        "message": "ordering lowering requires node or creatable new entity participants",
                        "severity": "blocking",
                        "location": f"semantic_goals[{index}].bindings",
                    }
                )
                if source_subgoal_id:
                    lowering_issue_subgoal_ids.add(source_subgoal_id)
                continue
            if isinstance(anchor_node.get("id"), str):
                _ensure_new_node_layout(
                    graph_data,
                    subject_node,
                    candidates=_ordering_layout_candidates(graph_data, anchor_node["id"], goal.get("relation")),
                )
            subject_item = _declare_and_occurrence(workflow_ir, subject_node)
            anchor_item = _declare_and_occurrence(workflow_ir, anchor_node)
            items = [subject_item, anchor_item] if goal.get("relation") == "before" else [anchor_item, subject_item]
            anchor_facts = anchor_binding.get("graph_facts", {}) if isinstance(anchor_binding, dict) else {}
            if (
                goal.get("relation") == "after"
                and isinstance(anchor_node.get("id"), str)
                and (anchor_facts.get("is_join_anchor") is True or _preserve_join_requested(goal))
            ):
                branches = _incoming_branch_nodes(graph_data, anchor_node["id"], "")
                if len(branches) > 1:
                    items = [{"type": "parallel", "join_mode": "all", "branches": branches}, anchor_item, subject_item]
            workflow_ir["control_flow_regions"].append(
                {
                    "relation_type": "",
                    "structure": {"type": "sequence", "items": items},
                }
            )
            if source_subgoal_id:
                executed_subgoal_ids.add(source_subgoal_id)
                for participant_binding in (subject_binding, anchor_binding):
                    if isinstance(participant_binding, dict) and participant_binding.get("needs_creation") is True:
                        for row in pre_contract.get("subgoals", []):
                            if row.get("type") == "object_desired_state" and row.get("object") == participant_binding.get("v2_object_id"):
                                executed_subgoal_ids.add(str(row.get("id")))
            continue

        if goal.get("kind") == "parallel" and goal.get("relation") == "parallel_with":
            left_binding = _binding_by_role(goal, "left")
            right_binding = _binding_by_role(goal, "right")
            left_id = left_binding.get("id") if isinstance(left_binding, dict) and left_binding.get("binding_type") == "node" else None
            right_id = right_binding.get("id") if isinstance(right_binding, dict) and right_binding.get("binding_type") == "node" else None
            if not (isinstance(left_id, str) and isinstance(right_id, str) and _parallel_nodes_are_safe_chain(graph_data, left_id, right_id)):
                issues.append(
                    {
                        "kind": "unsupported_parallel_boundary",
                        "message": "parallel_with lowering requires two existing nodes with a simple direct serial relation",
                        "severity": "blocking",
                        "location": f"semantic_goals[{index}].bindings",
                    }
                )
                if source_subgoal_id:
                    lowering_issue_subgoal_ids.add(source_subgoal_id)
                continue
            workflow_ir["control_flow_regions"].append(
                {
                    "relation_type": "",
                    "structure": {
                        "type": "parallel",
                        "join_mode": "all",
                        "branches": [{"type": "node", "id": left_id}, {"type": "node", "id": right_id}],
                    },
                }
            )
            if source_subgoal_id:
                executed_subgoal_ids.add(source_subgoal_id)
            continue

        if goal.get("kind") != "layout" or goal.get("relation") != "right_of":
            issues.append(
                {
                    "kind": "unsupported_lowering_goal",
                    "message": "this grounded semantic goal cannot be lowered by current Goal Lowering",
                    "severity": "blocking",
                    "location": f"semantic_goals[{index}]",
                }
            )
            if source_subgoal_id:
                lowering_issue_subgoal_ids.add(source_subgoal_id)
            continue
        target_binding = _binding_by_role(goal, "target")
        anchor_binding = _binding_by_role(goal, "anchor")
        target_id = target_binding.get("id") if isinstance(target_binding, dict) else None
        anchor_id = anchor_binding.get("id") if isinstance(anchor_binding, dict) else None
        target_node = nodes.get(target_id) if isinstance(target_id, str) else None
        anchor_node = nodes.get(anchor_id) if isinstance(anchor_id, str) else None
        if not (isinstance(target_binding, dict) and target_binding.get("binding_type") == "node"):
            target_node = None
        if not (isinstance(anchor_binding, dict) and anchor_binding.get("binding_type") == "node"):
            anchor_node = None
        if target_node is None or anchor_node is None:
            issues.append(
                {
                    "kind": "missing_layout_lowering_binding",
                    "message": "layout right_of lowering requires grounded target and anchor nodes",
                    "severity": "blocking",
                    "location": f"semantic_goals[{index}].bindings",
                }
            )
            if source_subgoal_id:
                lowering_issue_subgoal_ids.add(source_subgoal_id)
            continue
        anchor_x = _node_number(anchor_node, "x", 0.0)
        anchor_y = _node_number(anchor_node, "y", 0.0)
        anchor_width = _node_number(anchor_node, "width", float(DEFAULT_NODE_WIDTH))
        target_height = _node_number(target_node, "height", float(DEFAULT_NODE_HEIGHT))
        anchor_height = _node_number(anchor_node, "height", float(DEFAULT_NODE_HEIGHT))
        next_x = anchor_x + anchor_width + DEFAULT_LAYOUT_SPACING
        next_y = anchor_y + (anchor_height - target_height) / 2
        resolved_position = _first_non_colliding_layout(graph_data, target_id, target_node, next_x, next_y)
        if resolved_position is None:
            issues.append(
                _layout_collision(graph_data, target_id, target_node, next_x, next_y)
                or {
                    "kind": "layout_collision",
                    "message": "layout change would overlap another node",
                    "severity": "blocking",
                    "location": "semantic_goals.layout",
                    "target": target_id,
                }
            )
            if source_subgoal_id:
                lowering_issue_subgoal_ids.add(source_subgoal_id)
            continue
        next_x, next_y = resolved_position
        workflow_ir["properties"].append(
            {
                "node_id": target_id,
                "changes": {
                    "x": next_x,
                    "y": next_y,
                },
            }
        )
        if source_subgoal_id:
            executed_subgoal_ids.add(source_subgoal_id)

    _normalize_parallel_common_continuations(workflow_ir)
    _normalize_contiguous_ordering_sequences(workflow_ir)
    partial_execution = _build_partial_contract(grounded_goal, executed_subgoal_ids, skipped_subgoal_ids, lowering_issue_subgoal_ids)
    if issues:
        return {
            "ok": False,
            "stage": "lowering",
            "issues": issues,
            "partial_execution": partial_execution,
            "semantic_spec": {
                "status": "partial",
                "workflow_ir": _empty_workflow_ir(),
                "semantic_goals": [
                    {"goal_type": "lowering", "description": "grounded semantic goal could not be lowered to Workflow IR"}
                ],
                "clarification": {"needed": False, "question": "", "reason": ""},
            },
        }
    if not _workflow_ir_has_work(workflow_ir) and partial_execution.get("final_status") != "success":
        return {
            "ok": False,
            "stage": "lowering",
            "issues": [
                {
                    "kind": "no_executable_subgoals",
                    "message": "no semantic subgoal can be safely lowered to Workflow IR",
                    "severity": "blocking",
                    "location": "semantic_goals",
                }
            ],
            "partial_execution": partial_execution,
            "semantic_spec": _semantic_spec_from_ir(deepcopy(workflow_ir)),
        }
    return {
        "ok": True,
        "stage": "workflow_ir",
        "issues": [],
        "partial_execution": partial_execution,
        "semantic_spec": _semantic_spec_from_ir(deepcopy(workflow_ir)),
    }
