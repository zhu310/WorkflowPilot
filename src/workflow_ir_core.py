from __future__ import annotations

import json
import re
from collections import defaultdict
from copy import deepcopy
from typing import Any


NON_EDGE_RELATION_TOKENS = (
    "after",
    "before",
    "wait for",
    "wait for all incoming",
    "all incoming",
)

STRUCTURE_TYPES = {"node", "sequence", "parallel", "choice"}
SCOPE_CONTAINER_KINDS = ("stage", "group", "subprocess")
SCOPE_CONTAINER_COLLECTION_KEYS = {
    "stage": "stages",
    "group": "groups",
    "subprocess": "subprocesses",
}
DEFAULT_NODE_WIDTH = 160
DEFAULT_NODE_HEIGHT = 80


def empty_constraints() -> dict[str, Any]:
    return {
        "entities": [],
        "absent_entities": [],
        "properties": [],
        "edge_properties": [],
        "required_relations": [],
        "forbidden_relations": [],
        "preserve": {},
    }


def normalize_properties(rows: list[Any]) -> list[dict[str, Any]]:
    normalized_rows: list[dict[str, Any]] = []
    for row in rows:
        if not (isinstance(row, dict) and isinstance(row.get("changes"), dict)):
            continue
        changes = dict(row["changes"])
        canonical_join = None
        if changes.get("wait_for_all_incoming") is True or changes.get("waitForAllIncoming") is True:
            canonical_join = {"mode": "all"}
        join_value = changes.get("join")
        if isinstance(join_value, dict) and join_value.get("mode") in {"all", "any"}:
            canonical_join = {"mode": join_value["mode"]}
        changes.pop("wait_for_all_incoming", None)
        changes.pop("waitForAllIncoming", None)
        if canonical_join is not None:
            changes["join"] = canonical_join
        normalized_rows.append({**row, "changes": changes})
    return normalized_rows


def graph_is_valid(graph_data: dict[str, Any] | None) -> bool:
    if not isinstance(graph_data, dict):
        return False
    nodes = graph_data.get("nodes")
    edges = graph_data.get("edges")
    if not isinstance(nodes, list) or not isinstance(edges, list):
        return False
    node_ids = [node.get("id") for node in nodes if isinstance(node, dict)]
    edge_ids = [edge.get("id") for edge in edges if isinstance(edge, dict)]
    if len(node_ids) != len(nodes) or len(edge_ids) != len(edges):
        return False
    if len(node_ids) != len(set(node_ids)) or len(edge_ids) != len(set(edge_ids)):
        return False
    known_nodes = set(node_ids)
    return all(
        isinstance(edge, dict)
        and edge.get("source") in known_nodes
        and edge.get("target") in known_nodes
        for edge in edges
    )


def graph_node_ids(graph_data: dict[str, Any]) -> set[str]:
    return {node["id"] for node in graph_data.get("nodes", []) if isinstance(node, dict) and isinstance(node.get("id"), str)}


def graph_labels(graph_data: dict[str, Any]) -> set[str]:
    return {edge.get("label", "") for edge in graph_data.get("edges", []) if isinstance(edge, dict)}


def selection_edge_ids(selection: dict[str, Any] | None) -> list[str]:
    if not isinstance(selection, dict):
        return []
    value = selection.get("edge_ids", [])
    return [edge_id for edge_id in value if isinstance(edge_id, str)] if isinstance(value, list) else []


def normalize_request_context(selection: dict[str, Any] | None, request_context: dict[str, Any] | None) -> dict[str, Any]:
    selection = selection if isinstance(selection, dict) else {}
    request_context = request_context if isinstance(request_context, dict) else {}
    visible_area = request_context.get("visible_area")
    if not isinstance(visible_area, dict) and isinstance(selection.get("visible_area"), dict):
        visible_area = selection.get("visible_area")
    recent_transaction_diff = request_context.get("recent_transaction_diff")
    if not isinstance(recent_transaction_diff, dict):
        recent_transaction_diff = None
    return {
        "selected_node_ids": _selection_node_ids(selection),
        "selected_edge_ids": selection_edge_ids(selection),
        "visible_area": visible_area if isinstance(visible_area, dict) else None,
        "recent_transaction_diff": recent_transaction_diff,
    }


def goal_allows_entity_creation(texts: list[str]) -> bool:
    lowered = " ".join(text.lower() for text in texts)
    return any(token in lowered for token in ("add ", "create ", "insert ", "new ", "append ", "introduce "))


def _actual_relation_labels(graph_data: dict[str, Any]) -> set[str]:
    return {edge.get("label", "") for edge in graph_data.get("edges", []) if isinstance(edge, dict)}


def sanitize_relation_type(raw_value: Any, graph_data: dict[str, Any]) -> str | None:
    if not isinstance(raw_value, str):
        return None
    relation_type = raw_value.strip()
    if relation_type == "":
        return ""
    lowered = relation_type.lower()
    if any(token in lowered for token in NON_EDGE_RELATION_TOKENS):
        return None
    return relation_type if relation_type in _actual_relation_labels(graph_data) else None


def _node_matches(graph_data: dict[str, Any], raw: str | None) -> list[str]:
    if not raw:
        return []
    if raw == "$selected_node":
        return []
    lowered = str(raw).lower()
    exact = [node["id"] for node in graph_data.get("nodes", []) if node.get("id") == raw]
    if exact:
        return exact
    matches: list[str] = []
    for node in graph_data.get("nodes", []):
        if str(node.get("label", "")).lower() == lowered:
            matches.append(node["id"])
    return matches


def match_node(graph_data: dict[str, Any], raw: str | None) -> str | None:
    matches = _node_matches(graph_data, raw)
    return matches[0] if len(matches) == 1 else None


def explicit_mentions(text: str, graph_data: dict[str, Any]) -> set[str]:
    lowered = text.lower()
    mentioned: set[str] = set()
    for node in graph_data.get("nodes", []):
        node_id = str(node.get("id", "")).lower()
        label = str(node.get("label", "")).lower()
        node_id_pattern = rf"(?<![a-z0-9_]){re.escape(node_id)}(?![a-z0-9_])"
        label_pattern = rf"(?<![a-z0-9_]){re.escape(label)}(?![a-z0-9_])"
        if re.search(node_id_pattern, lowered) or re.search(label_pattern, lowered):
            mentioned.add(node["id"])
    return mentioned


def _selection_node_ids(selection: dict[str, Any] | None) -> list[str]:
    if not isinstance(selection, dict):
        return []
    value = selection.get("node_ids", [])
    return [node_id for node_id in value if isinstance(node_id, str)] if isinstance(value, list) else []


def _selected_edge_id(selection: dict[str, Any] | None) -> str | None:
    if not isinstance(selection, dict):
        return None
    value = selection.get("edge_ids", [])
    edge_ids = [edge_id for edge_id in value if isinstance(edge_id, str)] if isinstance(value, list) else []
    return edge_ids[0] if len(edge_ids) == 1 else None


def _normalize_visible_area(raw: Any) -> dict[str, float] | None:
    if not isinstance(raw, dict):
        return None
    if all(isinstance(raw.get(key), (int, float)) for key in ("x", "y", "width", "height")):
        x = float(raw["x"])
        y = float(raw["y"])
        width = float(raw["width"])
        height = float(raw["height"])
        if width <= 0 or height <= 0:
            return None
        return {"left": x, "top": y, "right": x + width, "bottom": y + height}
    if all(isinstance(raw.get(key), (int, float)) for key in ("left", "top", "right", "bottom")):
        left = float(raw["left"])
        top = float(raw["top"])
        right = float(raw["right"])
        bottom = float(raw["bottom"])
        if right <= left or bottom <= top:
            return None
        return {"left": left, "top": top, "right": right, "bottom": bottom}
    return None


def _node_box(node: dict[str, Any]) -> tuple[float, float, float, float] | None:
    if not isinstance(node, dict):
        return None
    x = node.get("x")
    y = node.get("y")
    if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
        return None
    width = node.get("width", DEFAULT_NODE_WIDTH)
    height = node.get("height", DEFAULT_NODE_HEIGHT)
    if not isinstance(width, (int, float)) or not isinstance(height, (int, float)):
        return None
    return float(x), float(y), float(x) + float(width), float(y) + float(height)


def visible_node_ids(graph_data: dict[str, Any], visible_area: dict[str, Any] | None) -> list[str]:
    rect = _normalize_visible_area(visible_area)
    if rect is None:
        return []
    visible: list[str] = []
    for node in graph_data.get("nodes", []):
        box = _node_box(node)
        if box is None or not isinstance(node.get("id"), str):
            continue
        left, top, right, bottom = box
        intersects = right > rect["left"] and left < rect["right"] and bottom > rect["top"] and top < rect["bottom"]
        if intersects:
            visible.append(node["id"])
    return sorted(visible)


def _container_aliases(container: dict[str, Any]) -> set[str]:
    return {
        str(value).strip().lower()
        for value in (container.get("id"), container.get("name"), container.get("title"), container.get("label"))
        if isinstance(value, str) and str(value).strip()
    }


def _container_member_ids(container: dict[str, Any], graph_data: dict[str, Any], kind: str) -> list[str]:
    explicit_keys = ("node_ids", "member_node_ids", "members", "nodes")
    member_ids: set[str] = set()
    for key in explicit_keys:
        value = container.get(key)
        if isinstance(value, list):
            for item in value:
                if isinstance(item, str):
                    member_ids.add(item)
                elif isinstance(item, dict) and isinstance(item.get("id"), str):
                    member_ids.add(item["id"])
    container_id = container.get("id")
    aliases = _container_aliases(container)
    for node in graph_data.get("nodes", []):
        node_id = node.get("id")
        if not isinstance(node_id, str):
            continue
        for key in (f"{kind}_id", kind):
            value = node.get(key)
            if isinstance(value, str) and ((isinstance(container_id, str) and value == container_id) or value.lower() in aliases):
                member_ids.add(node_id)
        multi_key = f"{kind}_ids"
        values = node.get(multi_key)
        if isinstance(values, list):
            normalized_values = {value for value in values if isinstance(value, str)}
            if (isinstance(container_id, str) and container_id in normalized_values) or aliases & {value.lower() for value in normalized_values}:
                member_ids.add(node_id)
    return sorted(member_ids)


def scope_containers(graph_data: dict[str, Any]) -> list[dict[str, Any]]:
    containers: list[dict[str, Any]] = []
    for kind in SCOPE_CONTAINER_KINDS:
        collection = graph_data.get(SCOPE_CONTAINER_COLLECTION_KEYS[kind], [])
        if not isinstance(collection, list):
            continue
        for index, item in enumerate(collection, start=1):
            if not isinstance(item, dict):
                continue
            container_id = item.get("id")
            if not isinstance(container_id, str) or not container_id:
                container_id = f"{kind}_{index}"
            names = sorted(
                {
                    value.strip()
                    for value in (item.get("id"), item.get("name"), item.get("title"), item.get("label"))
                    if isinstance(value, str) and value.strip()
                }
            )
            containers.append(
                {
                    "kind": kind,
                    "id": container_id,
                    "names": names,
                    "node_ids": _container_member_ids({**item, "id": container_id}, graph_data, kind),
                }
            )
    return containers


def _resolve_scope_container(graph_data: dict[str, Any], kind: str, raw_name: str | None) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    if not isinstance(raw_name, str) or not raw_name.strip():
        return None, []
    name = raw_name.strip().lower()
    matches = [
        container
        for container in scope_containers(graph_data)
        if container["kind"] == kind and any(alias.lower() == name for alias in container.get("names", []))
    ]
    if len(matches) == 1:
        return matches[0], []
    return None, matches


def _infer_scope_from_text(graph_data: dict[str, Any], requirement: dict[str, Any]) -> tuple[str | None, str | None]:
    text = str(requirement.get("text", ""))
    lowered = text.lower()
    if any(phrase in lowered for phrase in ("whole canvas", "entire workflow", "all nodes", "every node", "整张画布", "整体流程", "所有节点")):
        return "global", None
    if any(phrase in lowered for phrase in ("currently visible", "visible nodes", "in view", "current view", "当前可见", "看到的")):
        return "visible_area", None
    if any(
        phrase in lowered
        for phrase in (
            "selected nodes",
            "selected steps",
            "selected objects",
            "selected items",
            "选中的节点",
            "选中的步骤",
            "选中的对象",
        )
    ):
        return "selected_only", None
    for kind in SCOPE_CONTAINER_KINDS:
        if kind == "stage" and "stage" not in lowered and "阶段" not in lowered:
            continue
        if kind == "group" and "group" not in lowered and "分组" not in lowered:
            continue
        if kind == "subprocess" and "subprocess" not in lowered and "子流程" not in lowered:
            continue
        matches = [
            container
            for container in scope_containers(graph_data)
            if container.get("kind") == kind
            and any(isinstance(alias, str) and alias.lower() in lowered for alias in container.get("names", []))
        ]
        if len(matches) == 1:
            container = matches[0]
            names = [alias for alias in container.get("names", []) if isinstance(alias, str) and alias.strip()]
            if names:
                return kind, names[0]
    return None, None


def _parse_coordinate_offset(text: str) -> tuple[float, float] | None:
    lowered = text.lower()
    if not any(token in lowered for token in ("pixel", "px", "left", "right", "up", "down", "坐标", "像素", "左移", "右移", "上移", "下移")):
        return None
    dx = 0.0
    dy = 0.0
    matched = False
    patterns = (
        (r"right\s+(\d+(?:\.\d+)?)", 1.0, "x"),
        (r"left\s+(\d+(?:\.\d+)?)", -1.0, "x"),
        (r"up\s+(\d+(?:\.\d+)?)", -1.0, "y"),
        (r"down\s+(\d+(?:\.\d+)?)", 1.0, "y"),
        (r"(\d+(?:\.\d+)?)\s*(?:pixels?|px)\s+to\s+the\s+right", 1.0, "x"),
        (r"(\d+(?:\.\d+)?)\s*(?:pixels?|px)\s+to\s+the\s+left", -1.0, "x"),
        (r"(\d+(?:\.\d+)?)\s*(?:pixels?|px)\s+up", -1.0, "y"),
        (r"(\d+(?:\.\d+)?)\s*(?:pixels?|px)\s+down", 1.0, "y"),
        (r"右移\s*(\d+(?:\.\d+)?)", 1.0, "x"),
        (r"左移\s*(\d+(?:\.\d+)?)", -1.0, "x"),
        (r"上移\s*(\d+(?:\.\d+)?)", -1.0, "y"),
        (r"下移\s*(\d+(?:\.\d+)?)", 1.0, "y"),
        (r"向右\s*(\d+(?:\.\d+)?)\s*像素", 1.0, "x"),
        (r"向左\s*(\d+(?:\.\d+)?)\s*像素", -1.0, "x"),
        (r"向上\s*(\d+(?:\.\d+)?)\s*像素", -1.0, "y"),
        (r"向下\s*(\d+(?:\.\d+)?)\s*像素", 1.0, "y"),
    )
    for pattern, sign, axis in patterns:
        match = re.search(pattern, lowered)
        if not match:
            continue
        matched = True
        value = float(match.group(1)) * sign
        if axis == "x":
            dx = value
        else:
            dy = value
    if not matched:
        return None
    return dx, dy


def _resolve_scope_policy(
    graph_data: dict[str, Any],
    requirement: dict[str, Any],
    selection: dict[str, Any] | None,
    request_context: dict[str, Any] | None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    desired_state = requirement.get("desired_state", {}) if isinstance(requirement.get("desired_state"), dict) else {}
    raw_scope = requirement.get("scope", {}) if isinstance(requirement.get("scope"), dict) else {}
    normalized_context = normalize_request_context(selection, request_context)
    grounded_refs: dict[str, Any] = {}
    clarification = requirement.get("clarification", {}) if isinstance(requirement.get("clarification"), dict) else {}
    scope_policy = requirement.get("scope_policy", {}) if isinstance(requirement.get("scope_policy"), dict) else {}
    kind = raw_scope.get("kind") if isinstance(raw_scope.get("kind"), str) else None
    raw_name = raw_scope.get("name") if isinstance(raw_scope.get("name"), str) else None
    inferred_kind, inferred_name = _infer_scope_from_text(graph_data, requirement)
    if inferred_kind in {"stage", "group", "subprocess"} and kind in {None, "global"}:
        kind = inferred_kind
        raw_name = inferred_name
    elif kind is None:
        kind = inferred_kind
    if raw_name is None:
        raw_name = inferred_name
    if desired_state.get("selected_only") is True or requirement.get("selected_only") is True:
        kind = "selected_only"
    if kind == "selected_only":
        node_ids = normalize_request_context(selection, request_context)["selected_node_ids"]
        grounded_refs["resolved_scope_node_ids"] = node_ids
        return {**scope_policy, "kind": "selected_only", "selected_only": True, "node_ids": node_ids}, grounded_refs, clarification
    if kind == "visible_area":
        visible_area = normalized_context.get("visible_area")
        if not isinstance(visible_area, dict):
            clarification = {"needed": True, "question": "Please provide the current visible area.", "reason": "missing visible area"}
            return scope_policy, grounded_refs, clarification
        node_ids = visible_node_ids(graph_data, visible_area)
        grounded_refs["resolved_visible_node_ids"] = node_ids
        return {**scope_policy, "kind": "visible_area", "node_ids": node_ids, "visible_area": _normalize_visible_area(visible_area)}, grounded_refs, clarification
    if kind in {"stage", "group", "subprocess"}:
        container, ambiguous = _resolve_scope_container(graph_data, kind, raw_name)
        if ambiguous:
            candidate_ids = [row["id"] for row in ambiguous]
            grounded_refs["ambiguous_scope_container_ids"] = candidate_ids
            clarification = {
                "needed": True,
                "question": f"Which {kind} do you want to edit? Candidates: {', '.join(candidate_ids)}.",
                "reason": f"ambiguous {kind} scope",
            }
            return scope_policy, grounded_refs, clarification
        if container is None:
            clarification = {
                "needed": True,
                "question": f"Which {kind} do you want to edit?",
                "reason": f"unresolved {kind} scope",
            }
            return scope_policy, grounded_refs, clarification
        grounded_refs["scope_container_id"] = container["id"]
        grounded_refs["resolved_scope_node_ids"] = container["node_ids"]
        return {
            **scope_policy,
            "kind": kind,
            "container_id": container["id"],
            "node_ids": container["node_ids"],
        }, grounded_refs, clarification
    if kind == "global":
        node_ids = sorted(graph_node_ids(graph_data))
        grounded_refs["resolved_scope_node_ids"] = node_ids
        return {**scope_policy, "kind": "global", "node_ids": node_ids}, grounded_refs, clarification
    return scope_policy, grounded_refs, clarification


def _resolve_node_reference(
    graph_data: dict[str, Any],
    raw: Any,
    selection: dict[str, Any] | None,
) -> tuple[str | None, list[str]]:
    if not isinstance(raw, str) or not raw.strip():
        return None, []
    if raw == "$selected_node":
        selected = _selection_node_ids(selection)
        return (selected[0], []) if len(selected) == 1 else (None, selected)
    matches = _node_matches(graph_data, raw)
    return (matches[0], []) if len(matches) == 1 else (None, matches)


def _scope_node_filter(scope_policy: dict[str, Any]) -> set[str]:
    return {
        node_id
        for node_id in scope_policy.get("node_ids", [])
        if isinstance(node_id, str)
    }


def _apply_scope_to_ambiguous_references(grounded_refs: dict[str, Any], scope_policy: dict[str, Any]) -> None:
    scope_nodes = _scope_node_filter(scope_policy)
    if not scope_nodes:
        return
    ambiguous_bindings = (
        ("ambiguous_source_node_ids", "source_node_id"),
        ("ambiguous_target_node_ids", "target_node_id"),
        ("ambiguous_selected_node_ids", "selected_node_id"),
        ("ambiguous_node_ids", "source_node_id"),
    )
    for ambiguous_key, resolved_key in ambiguous_bindings:
        candidates = [
            node_id
            for node_id in grounded_refs.get(ambiguous_key, [])
            if isinstance(node_id, str) and node_id in scope_nodes
        ]
        if len(candidates) == 1:
            grounded_refs[resolved_key] = candidates[0]
            grounded_refs.pop(ambiguous_key, None)
        elif candidates:
            grounded_refs[ambiguous_key] = sorted(candidates)


def _has_history_reference_signal(text: str) -> bool:
    lowered = text.lower()
    restore_terms = ("undo", "revert", "restore", "撤销", "还原", "恢复")
    temporal_terms = ("previous", "last", "recent", "earlier", "original", "back to", "之前", "刚才", "上一", "上次", "原来")
    mutation_terms = ("move", "moved", "rename", "renamed", "change", "changed", "edit", "edited", "移动", "挪动", "改名", "修改")
    has_restore_action = any(term in lowered for term in restore_terms)
    has_history_reference = any(term in lowered for term in temporal_terms)
    has_mutation_reference = any(term in lowered for term in mutation_terms)
    if not (has_restore_action and (has_history_reference or has_mutation_reference)):
        return False
    historical_terms = (
        "undo",
        "revert",
        "restore",
        "previous",
        "last",
        "original",
        "back to",
        "刚才",
        "上一",
        "上次",
        "恢复",
        "撤销",
        "还原",
        "原来",
    )
    return any(term in lowered for term in historical_terms)


def _history_property_mutations(request_context: dict[str, Any] | None) -> list[dict[str, Any]]:
    context = request_context if isinstance(request_context, dict) else {}
    diff = context.get("recent_transaction_diff")
    if not isinstance(diff, dict):
        return []
    mutations = diff.get("mutations", [])
    return [
        mutation
        for mutation in mutations
        if isinstance(mutation, dict)
        and mutation.get("kind") == "property"
        and isinstance(mutation.get("target_entity_id"), str)
        and isinstance(mutation.get("property"), str)
    ]


def _node_current_property(graph_data: dict[str, Any], node_id: str, property_name: str) -> Any:
    node = next((row for row in graph_data.get("nodes", []) if isinstance(row, dict) and row.get("id") == node_id), None)
    return node.get(property_name) if isinstance(node, dict) else None


def _resolve_history_reference(
    graph_data: dict[str, Any],
    requirement: dict[str, Any],
    grounded_refs: dict[str, Any],
    request_context: dict[str, Any] | None,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    text = str(requirement.get("text", ""))
    if not _has_history_reference_signal(text):
        return [], {}, {}
    diff = request_context.get("recent_transaction_diff") if isinstance(request_context, dict) else None
    if not isinstance(diff, dict):
        return [], {}, {
            "needed": True,
            "question": "Which previous edit should I restore?",
            "reason": "missing recent transaction diff",
        }
    selected_or_source_ids = {
        node_id
        for node_id in {
            grounded_refs.get("selected_node_id"),
            grounded_refs.get("source_node_id"),
        }
        if isinstance(node_id, str)
    }
    target_ids_from_refs = {
        node_id
        for node_id in {
            grounded_refs.get("target_node_id"),
        }
        if isinstance(node_id, str)
    }
    mentioned_ids = grounded_refs.get("mentioned_node_ids", [])
    fallback_target_ids = {node_id for node_id in mentioned_ids if isinstance(node_id, str)} if isinstance(mentioned_ids, list) else set()
    target_ids = selected_or_source_ids or target_ids_from_refs or fallback_target_ids
    if not target_ids:
        return [], {}, {
            "needed": True,
            "question": "Which edited node should I restore?",
            "reason": "missing historical target reference",
        }
    property_candidates = _history_property_mutations(request_context)
    if any(word in text.lower() for word in ("move", "position", "x", "y", "位置", "移动", "挪", "坐标")):
        property_candidates = [
            mutation for mutation in property_candidates if mutation.get("property") in {"x", "y"}
        ]
    matches = [
        mutation
        for mutation in property_candidates
        if mutation.get("target_entity_id") in target_ids
        and _node_current_property(graph_data, mutation["target_entity_id"], mutation["property"]) == mutation.get("after")
    ]
    stale_matches = [
        mutation
        for mutation in property_candidates
        if mutation.get("target_entity_id") in target_ids
        and _node_current_property(graph_data, mutation["target_entity_id"], mutation["property"]) != mutation.get("after")
    ]
    if not matches and stale_matches:
        return [], {}, {
            "needed": True,
            "question": "The referenced edit is no longer the current state. Should I still restore its earlier value?",
            "reason": "historical mutation no longer matches current graph",
        }
    if not matches:
        return [], {}, {
            "needed": True,
            "question": "Which previous edit should I restore?",
            "reason": "no matching historical mutation",
        }
    matched_node_ids = {mutation["target_entity_id"] for mutation in matches}
    if len(matched_node_ids) != 1:
        return [], {}, {
            "needed": True,
            "question": "Which edited node should I restore?",
            "reason": "ambiguous historical target reference",
        }
    by_property: dict[str, dict[str, Any]] = {}
    for mutation in matches:
        property_name = mutation["property"]
        if property_name in by_property and by_property[property_name].get("mutation_id") != mutation.get("mutation_id"):
            return [], {}, {
                "needed": True,
                "question": "Which previous property change should I restore?",
                "reason": "ambiguous historical mutation reference",
            }
        by_property[property_name] = mutation
    node_id = next(iter(matched_node_ids))
    changes = {property_name: mutation.get("before") for property_name, mutation in by_property.items()}
    history_refs = {
        "history_reference_used": True,
        "matched_transaction_id": diff.get("transaction_id"),
        "matched_mutation_ids": sorted(
            mutation.get("mutation_id")
            for mutation in by_property.values()
            if isinstance(mutation.get("mutation_id"), str)
        ),
    }
    if len(history_refs["matched_mutation_ids"]) == 1:
        history_refs["matched_mutation_id"] = history_refs["matched_mutation_ids"][0]
    property_row = {
        "node_id": node_id,
        "target_entity_ref": node_id,
        "source_requirement_id": requirement.get("id"),
        "derivations": {property_name: "historical_mutation_reference" for property_name in changes},
        "changes": changes,
    }
    return [property_row], history_refs, {}


def _selection_cardinality_hints(text: str, selection: dict[str, Any] | None) -> tuple[str | None, list[str]]:
    lowered = text.lower()
    selected = _selection_node_ids(selection)
    if not selected:
        return None, []
    if "selected nodes" in lowered or "selected steps" in lowered or "selected objects" in lowered or "selected items" in lowered:
        return None, []
    if "selected node" in lowered or "selected step" in lowered or "selected object" in lowered or "selected item" in lowered:
        return ((selected[0], []) if len(selected) == 1 else (None, selected))
    return None, []


def _deterministic_new_entities_for_requirement(requirement: dict[str, Any]) -> list[dict[str, str]]:
    text = str(requirement.get("text", ""))
    lowered = text.lower()
    if not any(token in lowered for token in ("add ", "create ", "insert ", "new ", "append ", "introduce ")):
        return []
    named_match = re.search(
        r"\b(?:add|create|insert|introduce)\b.+?\bnamed\s+([A-Za-z0-9 _-]+?)(?:\bbetween\b|\bafter\b|\bbefore\b|\bon\b|[.;]|$)",
        text,
        re.IGNORECASE,
    )
    if not named_match:
        return []
    label = named_match.group(1).strip().strip(".")
    prefix = text[: named_match.start(1)].lower()
    if "human review" in prefix:
        node_type = "human_review"
    elif "end" in prefix:
        node_type = "end"
    elif "start" in prefix:
        node_type = "start"
    else:
        node_type = "task"
    return [{"label": label, "node_type": node_type}]


def _contains_word(text: str, word: str) -> bool:
    return re.search(rf"\b{re.escape(word)}\b", text.lower()) is not None


def _requirement_property_signals(requirement: dict[str, Any]) -> dict[str, bool]:
    text = str(requirement.get("text", "")).lower()
    return {
        "rename": any(token in text for token in ("rename", "relabel", "label ", "rename the selected", "改名", "重命名")),
        "join": any(token in text for token in ("wait", "join", "incoming", "all incoming", "等待", "汇合", "所有输入")),
        "move": any(
            (_contains_word(text, token) if token in {"move", "shift", "left", "right", "up", "down", "pixel", "position"} else token in text)
            for token in ("move", "shift", "left", "right", "up", "down", "pixel", "position", "坐标", "移动", "左移", "右移", "上移", "下移", "像素")
        ),
    }


def _is_existing_property_change_authorized(
    requirement: dict[str, Any],
    grounded_refs: dict[str, Any],
    selection: dict[str, Any] | None,
    node_id: str,
    property_name: str,
    value: Any,
) -> tuple[bool, str]:
    signals = _requirement_property_signals(requirement)
    creates_entity = bool(_deterministic_new_entities_for_requirement(requirement))
    explicit_targets = {
        grounded_refs.get("source_node_id"),
        grounded_refs.get("target_node_id"),
        grounded_refs.get("selected_node_id"),
        *grounded_refs.get("mentioned_node_ids", []),
        *_selection_node_ids(selection),
    }
    explicit_targets = {target for target in explicit_targets if isinstance(target, str)}
    scope_policy = requirement.get("scope_policy", {}) if isinstance(requirement.get("scope_policy"), dict) else {}
    scope_targets = {node_id for node_id in scope_policy.get("node_ids", []) if isinstance(node_id, str)}
    new_entity_labels = {row["label"] for row in _deterministic_new_entities_for_requirement(requirement)}
    if property_name in {"x", "y"}:
        return signals["move"] and node_id in (explicit_targets | scope_targets), "grounded_position_offset"
    if property_name == "join":
        return signals["join"], "grounded_explicit_join"
    if property_name == "label":
        if creates_entity and isinstance(value, str) and value in new_entity_labels and not signals["rename"]:
            return False, "rejected_new_entity_label_on_existing_node"
        if creates_entity and not signals["rename"]:
            return False, "rejected_existing_label_without_explicit_rename"
        return node_id in (explicit_targets | scope_targets), "grounded_explicit_label"
    if creates_entity:
        return False, "rejected_existing_property_on_entity_creation_requirement"
    return node_id in (explicit_targets | scope_targets), "grounded_explicit_property"


def _resolve_grounded_properties(
    graph_data: dict[str, Any],
    requirement: dict[str, Any],
    grounded_refs: dict[str, Any],
    selection: dict[str, Any] | None,
    request_context: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    desired_state = requirement.get("desired_state", {}) if isinstance(requirement.get("desired_state"), dict) else {}
    scope_policy = requirement.get("scope_policy", {}) if isinstance(requirement.get("scope_policy"), dict) else {}
    scope_node_ids = [node_id for node_id in scope_policy.get("node_ids", []) if isinstance(node_id, str)]
    properties: list[dict[str, Any]] = []
    for row in desired_state.get("properties", []):
        if not (isinstance(row, dict) and isinstance(row.get("changes"), dict)):
            continue
        node_ids: list[str] = []
        ambiguous: list[str] = []
        if isinstance(row.get("node_id"), str):
            node_id, ambiguous = _resolve_node_reference(graph_data, row["node_id"], selection)
            if node_id:
                node_ids = [node_id]
        elif row.get("target") == "source_node":
            node_id = grounded_refs.get("source_node_id") if isinstance(grounded_refs.get("source_node_id"), str) else None
            if node_id:
                node_ids = [node_id]
        elif row.get("target") == "target_node":
            node_id = grounded_refs.get("target_node_id") if isinstance(grounded_refs.get("target_node_id"), str) else None
            if node_id:
                node_ids = [node_id]
        elif row.get("target") == "selected_node":
            if scope_policy.get("selected_only") is True and len(scope_node_ids) > 1:
                node_ids = scope_node_ids
            else:
                node_id, ambiguous = _resolve_node_reference(graph_data, "$selected_node", selection)
                if node_id:
                    node_ids = [node_id]
        elif row.get("target") == "scope_nodes":
            node_ids = scope_node_ids
        if ambiguous:
            grounded_refs.setdefault("ambiguous_node_ids", sorted(ambiguous))
            continue
        for node_id in node_ids:
            current = next((node for node in graph_data.get("nodes", []) if node.get("id") == node_id), None)
            changes = normalize_changes(row["changes"])
            if isinstance(changes.get("label"), str) and not changes["label"].strip():
                changes.pop("label", None)
            if isinstance(current, dict) and changes.get("label") == current.get("label"):
                changes.pop("label", None)
            authorized_changes: dict[str, Any] = {}
            derivations: dict[str, str] = {}
            for key, value in changes.items():
                allowed, derivation = _is_existing_property_change_authorized(
                    requirement,
                    grounded_refs,
                    selection,
                    node_id,
                    key,
                    value,
                )
                if allowed:
                    authorized_changes[key] = value
                    derivations[key] = derivation
            if authorized_changes:
                properties.append(
                    {
                        "node_id": node_id,
                        "target_entity_ref": node_id,
                        "source_requirement_id": requirement.get("id"),
                        "derivations": derivations,
                        "changes": authorized_changes,
                    }
                )
    position_rows = [row for row in desired_state.get("position_offsets", []) if isinstance(row, dict)]
    if not position_rows:
        inferred_offset = _parse_coordinate_offset(str(requirement.get("text", "")))
        if inferred_offset is not None:
            dx, dy = inferred_offset
            if scope_node_ids:
                position_rows = [{"target": "scope_nodes", "dx": dx, "dy": dy}]
            elif isinstance(grounded_refs.get("source_node_id"), str) and not isinstance(grounded_refs.get("target_node_id"), str):
                position_rows = [{"target": "source_node", "dx": dx, "dy": dy}]
            elif isinstance(grounded_refs.get("selected_node_id"), str):
                position_rows = [{"target": "selected_node", "dx": dx, "dy": dy}]
            elif isinstance(grounded_refs.get("mentioned_node_ids"), list) and len(grounded_refs["mentioned_node_ids"]) == 1:
                position_rows = [{"node_id": grounded_refs["mentioned_node_ids"][0], "dx": dx, "dy": dy}]
    for row in position_rows:
        if not isinstance(row, dict):
            continue
        node_ids: list[str] = []
        ambiguous = []
        if isinstance(row.get("node_id"), str):
            node_id, ambiguous = _resolve_node_reference(graph_data, row["node_id"], selection)
            if node_id:
                node_ids = [node_id]
        elif row.get("target") == "source_node":
            node_id = grounded_refs.get("source_node_id") if isinstance(grounded_refs.get("source_node_id"), str) else None
            if node_id:
                node_ids = [node_id]
        elif row.get("target") == "target_node":
            node_id = grounded_refs.get("target_node_id") if isinstance(grounded_refs.get("target_node_id"), str) else None
            if node_id:
                node_ids = [node_id]
        elif row.get("target") == "selected_node":
            if scope_policy.get("selected_only") is True and len(scope_node_ids) > 1:
                node_ids = scope_node_ids
            else:
                node_id, ambiguous = _resolve_node_reference(graph_data, "$selected_node", selection)
                if node_id:
                    node_ids = [node_id]
        elif row.get("target") == "scope_nodes":
            node_ids = scope_node_ids
        if ambiguous:
            grounded_refs.setdefault("ambiguous_node_ids", sorted(ambiguous))
            continue
        for node_id in node_ids:
            current = next((node for node in graph_data.get("nodes", []) if node.get("id") == node_id), None)
            if not isinstance(current, dict):
                continue
            dx = row.get("dx", 0)
            dy = row.get("dy", 0)
            changes: dict[str, Any] = {}
            if isinstance(dx, (int, float)):
                changes["x"] = current.get("x", 0) + dx
            if isinstance(dy, (int, float)):
                changes["y"] = current.get("y", 0) + dy
            authorized_changes: dict[str, Any] = {}
            derivations: dict[str, str] = {}
            for key, value in changes.items():
                allowed, derivation = _is_existing_property_change_authorized(
                    requirement,
                    grounded_refs,
                    selection,
                    node_id,
                    key,
                    value,
                )
                if allowed:
                    authorized_changes[key] = value
                    derivations[key] = derivation
            if authorized_changes:
                properties.append(
                    {
                        "node_id": node_id,
                        "target_entity_ref": node_id,
                        "source_requirement_id": requirement.get("id"),
                        "derivations": derivations,
                        "changes": authorized_changes,
                    }
                )
    if not any(isinstance(row.get("changes"), dict) and row["changes"] for row in properties):
        inferred_offset = _parse_coordinate_offset(str(requirement.get("text", "")))
        if inferred_offset is not None and scope_node_ids:
            dx, dy = inferred_offset
            for node_id in scope_node_ids:
                current = next((node for node in graph_data.get("nodes", []) if node.get("id") == node_id), None)
                if not isinstance(current, dict):
                    continue
                changes: dict[str, Any] = {}
                if isinstance(dx, (int, float)):
                    changes["x"] = current.get("x", 0) + dx
                if isinstance(dy, (int, float)):
                    changes["y"] = current.get("y", 0) + dy
                authorized_changes: dict[str, Any] = {}
                derivations: dict[str, str] = {}
                for key, value in changes.items():
                    allowed, derivation = _is_existing_property_change_authorized(
                        requirement,
                        grounded_refs,
                        selection,
                        node_id,
                        key,
                        value,
                    )
                    if allowed:
                        authorized_changes[key] = value
                        derivations[key] = derivation
                if authorized_changes:
                    properties.append(
                        {
                            "node_id": node_id,
                            "target_entity_ref": node_id,
                            "source_requirement_id": requirement.get("id"),
                            "derivations": derivations,
                            "changes": authorized_changes,
                        }
                    )
    deduped = {
        (row["node_id"], json.dumps(row["changes"], ensure_ascii=False, sort_keys=True)): row
        for row in properties
        if row.get("changes")
    }
    return list(deduped.values())


def _resolve_grounded_relations(
    graph_data: dict[str, Any],
    requirement: dict[str, Any],
    grounded_refs: dict[str, Any],
    selection: dict[str, Any] | None,
) -> list[dict[str, str]]:
    desired_state = requirement.get("desired_state", {}) if isinstance(requirement.get("desired_state"), dict) else {}
    if _requires_structural_region(requirement, grounded_refs) and not _requires_explicit_relation(requirement, grounded_refs):
        return []
    text = str(requirement.get("text", "")).lower()
    explicit_relation_tokens = (
        "redirect",
        "restart",
        "retry",
        "exception",
        "failure relation",
        "success relation",
        "relation from",
    )
    if desired_state.get("explicit_relations") and not any(token in text for token in explicit_relation_tokens):
        return []
    rows: list[dict[str, str]] = []
    for row in desired_state.get("explicit_relations", []):
        if not isinstance(row, dict):
            continue
        source = None
        target = None
        source_ambiguous: list[str] = []
        target_ambiguous: list[str] = []
        if isinstance(row.get("source_node"), str):
            source, source_ambiguous = _resolve_node_reference(graph_data, row["source_node"], selection)
        elif row.get("source") == "source_node":
            source = grounded_refs.get("source_node_id") if isinstance(grounded_refs.get("source_node_id"), str) else None
        if isinstance(row.get("target_node"), str):
            target, target_ambiguous = _resolve_node_reference(graph_data, row["target_node"], selection)
        elif row.get("target") == "target_node":
            target = grounded_refs.get("target_node_id") if isinstance(grounded_refs.get("target_node_id"), str) else None
        relation_type = sanitize_relation_type(row.get("relation_type"), graph_data)
        if source_ambiguous or target_ambiguous:
            grounded_refs.setdefault("ambiguous_node_ids", sorted(set(source_ambiguous + target_ambiguous)))
            continue
        if source and target and relation_type is not None:
            rows.append({"source": source, "relation_type": relation_type, "target": target})
    return rows


def ground_requirement(
    graph_data: dict[str, Any],
    requirement: dict[str, Any],
    selection: dict[str, Any] | None = None,
    request_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    refs = requirement.get("references", {}) if isinstance(requirement.get("references"), dict) else {}
    grounded_refs: dict[str, Any] = {}
    source_node_id, source_ambiguous = _resolve_node_reference(graph_data, refs.get("source_node"), selection)
    target_node_id, target_ambiguous = _resolve_node_reference(graph_data, refs.get("target_node"), selection)
    if source_ambiguous:
        grounded_refs["ambiguous_source_node_ids"] = sorted(source_ambiguous)
    if target_ambiguous:
        grounded_refs["ambiguous_target_node_ids"] = sorted(target_ambiguous)
    relation_type = sanitize_relation_type(refs.get("relation_type"), graph_data)
    selected_edge = refs.get("selected_edge") if isinstance(refs.get("selected_edge"), dict) else None
    selected_edge_id = _selected_edge_id(selection)
    if selected_edge:
        src, src_ambiguous = _resolve_node_reference(graph_data, selected_edge.get("source"), selection)
        tgt, tgt_ambiguous = _resolve_node_reference(graph_data, selected_edge.get("target"), selection)
        selected_relation_type = sanitize_relation_type(selected_edge.get("relation_type"), graph_data)
        label = selected_relation_type if selected_relation_type is not None else relation_type
        candidates = [
            edge
            for edge in graph_data.get("edges", [])
            if (src is None or edge.get("source") == src)
            and (tgt is None or edge.get("target") == tgt)
            and (label is None or edge.get("label", "") == label)
        ]
        if len(candidates) == 1:
            edge = candidates[0]
            selected_edge_id = edge["id"]
            source_node_id = edge["source"]
            target_node_id = edge["target"]
            relation_type = edge.get("label", "")
        if src_ambiguous:
            grounded_refs["ambiguous_source_node_ids"] = sorted(src_ambiguous)
        if tgt_ambiguous:
            grounded_refs["ambiguous_target_node_ids"] = sorted(tgt_ambiguous)
    elif selected_edge_id:
        selected = next((edge for edge in graph_data.get("edges", []) if edge.get("id") == selected_edge_id), None)
        if isinstance(selected, dict):
            grounded_refs["selected_edge_id"] = selected_edge_id
            source_node_id = source_node_id or selected.get("source")
            target_node_id = target_node_id or selected.get("target")
            if relation_type is None:
                relation_type = selected.get("label", "")
    if source_node_id is None and target_node_id is None:
        inferred_selected_id, selected_ambiguous = _selection_cardinality_hints(str(requirement.get("text", "")), selection)
        if inferred_selected_id:
            source_node_id = inferred_selected_id
            grounded_refs["selected_node_id"] = inferred_selected_id
        elif selected_ambiguous:
            grounded_refs["ambiguous_selected_node_ids"] = sorted(selected_ambiguous)
    if source_node_id:
        grounded_refs["source_node_id"] = source_node_id
    if target_node_id:
        grounded_refs["target_node_id"] = target_node_id
    if relation_type is not None:
        grounded_refs["relation_type"] = relation_type
    if selected_edge_id:
        grounded_refs["selected_edge_id"] = selected_edge_id

    text = str(requirement.get("text", ""))
    mentions = sorted(explicit_mentions(text, graph_data))
    if mentions:
        grounded_refs["mentioned_node_ids"] = mentions
    lowered = text.lower()

    if "former predecessor of" in lowered and " to " in lowered and len(mentions) >= 2:
        anchor = mentions[0]
        moving = next((node_id for node_id in mentions if node_id != anchor), None)
        if moving:
            predecessors = [edge for edge in graph_data.get("edges", []) if edge.get("target") == moving]
            if len(predecessors) == 1:
                grounded_refs["source_node_id"] = predecessors[0]["source"]
                grounded_refs["target_node_id"] = anchor
                grounded_refs["relation_type"] = predecessors[0].get("label", "")
                grounded_refs["selected_edge_id"] = predecessors[0]["id"]
    if " after " in lowered and len(mentions) >= 2:
        anchor = next((node_id for node_id in mentions if node_id.lower() in lowered.split(" after ", 1)[1]), None)
        moving = next((node_id for node_id in mentions if node_id != anchor), None)
        if anchor and moving:
            grounded_refs.setdefault("source_node_id", anchor)
            grounded_refs.setdefault("target_node_id", moving)
            if "relation_type" not in grounded_refs or grounded_refs.get("relation_type") == "":
                outgoing_labels = {edge.get("label", "") for edge in graph_data.get("edges", []) if edge.get("source") == anchor}
                if len(outgoing_labels) == 1:
                    grounded_refs["relation_type"] = next(iter(outgoing_labels))
    if " before " in lowered and len(mentions) >= 2:
        target = next((node_id for node_id in mentions if node_id.lower() in lowered.split(" before ", 1)[1]), None)
        source = next((node_id for node_id in mentions if node_id != target), None)
        if source and target:
            grounded_refs.setdefault("source_node_id", source)
            grounded_refs.setdefault("target_node_id", target)
            if "relation_type" not in grounded_refs or grounded_refs.get("relation_type") == "":
                incoming_labels = {edge.get("label", "") for edge in graph_data.get("edges", []) if edge.get("target") == target}
                if len(incoming_labels) == 1:
                    grounded_refs["relation_type"] = next(iter(incoming_labels))
    if "relation_type" not in grounded_refs:
        grounded_source = grounded_refs.get("source_node_id")
        grounded_target = grounded_refs.get("target_node_id")
        if isinstance(grounded_source, str) and isinstance(grounded_target, str):
            matching_edges = [
                edge
                for edge in graph_data.get("edges", [])
                if edge.get("source") == grounded_source and edge.get("target") == grounded_target
            ]
            if len(matching_edges) == 1:
                grounded_refs["relation_type"] = matching_edges[0].get("label", "")
    scope_policy, scope_refs, scope_clarification = _resolve_scope_policy(graph_data, requirement, selection, request_context)
    grounded_refs.update(scope_refs)
    _apply_scope_to_ambiguous_references(grounded_refs, scope_policy)
    if scope_policy.get("selected_only") is True and len(scope_policy.get("node_ids", [])) > 1:
        for key in ("ambiguous_selected_node_ids", "ambiguous_source_node_ids", "ambiguous_target_node_ids", "ambiguous_node_ids"):
            grounded_refs.pop(key, None)
    requirement = {**requirement, "scope_policy": scope_policy}
    history_properties, history_refs, history_clarification = _resolve_history_reference(
        graph_data,
        requirement,
        grounded_refs,
        request_context,
    )
    grounded_refs.update(history_refs)
    properties = (
        history_properties
        if history_refs.get("history_reference_used") is True
        else _resolve_grounded_properties(graph_data, requirement, grounded_refs, selection, request_context)
    )
    grounded_desired_state = {
        "abstract_goals": _normalize_abstract_goals(requirement),
        "properties": properties,
        "explicit_relations": _resolve_grounded_relations(graph_data, requirement, grounded_refs, selection),
        "absent_entities": [],
        "clarification": (
            history_clarification
            if history_clarification.get("needed")
            else scope_clarification
            if scope_clarification.get("needed")
            else requirement.get("clarification", {})
        ),
        "scope_policy": scope_policy,
    }
    desired_state = requirement.get("desired_state", {}) if isinstance(requirement.get("desired_state"), dict) else {}
    for row in desired_state.get("absent_entities", []):
        node_id = None
        ambiguous: list[str] = []
        if isinstance(row, dict) and isinstance(row.get("node_id"), str):
            node_id, ambiguous = _resolve_node_reference(graph_data, row["node_id"], selection)
        elif isinstance(row, str):
            node_id, ambiguous = _resolve_node_reference(graph_data, row, selection)
        if ambiguous:
            grounded_refs.setdefault("ambiguous_node_ids", sorted(ambiguous))
            continue
        if node_id:
            grounded_desired_state["absent_entities"].append({"node_id": node_id})
    if desired_state.get("selected_only") is True:
        grounded_desired_state["scope_policy"] = {"kind": "selected_only", "selected_only": True, "node_ids": _selection_node_ids(selection)}
    if requirement.get("selected_only") is True:
        grounded_desired_state["scope_policy"] = {"kind": "selected_only", "selected_only": True, "node_ids": _selection_node_ids(selection)}
    return {"id": requirement.get("id"), "text": text, "references": grounded_refs, "grounded_desired_state": grounded_desired_state}


def requirement_local_facts(graph_data: dict[str, Any], requirement: dict[str, Any]) -> dict[str, Any]:
    refs = requirement.get("references", {}) if isinstance(requirement.get("references"), dict) else {}
    desired_state = requirement.get("grounded_desired_state", {}) if isinstance(requirement.get("grounded_desired_state"), dict) else {}
    scope_policy = desired_state.get("scope_policy", {}) if isinstance(desired_state.get("scope_policy"), dict) else {}
    mentioned = set(refs.get("mentioned_node_ids", []))
    core_node_ids = {value for key, value in refs.items() if key.endswith("_node_id") and isinstance(value, str)}
    core_node_ids |= {row.get("node_id") for row in desired_state.get("properties", []) if isinstance(row, dict) and isinstance(row.get("node_id"), str)}
    core_node_ids |= {node_id for node_id in scope_policy.get("node_ids", []) if isinstance(node_id, str)}
    core_node_ids |= mentioned
    edge_ids: set[str] = set()
    relation_keys: set[tuple[str, str, str]] = set()
    context_node_ids: set[str] = set(core_node_ids)
    for edge in graph_data.get("edges", []):
        if edge.get("id") == refs.get("selected_edge_id"):
            edge_ids.add(edge["id"])
            relation_keys.add((edge["source"], edge["target"], edge.get("label", "")))
            context_node_ids.update([edge["source"], edge["target"]])
        if edge.get("source") in core_node_ids or edge.get("target") in core_node_ids:
            context_node_ids.update([edge["source"], edge["target"]])
            if edge.get("source") in core_node_ids and edge.get("target") in core_node_ids:
                relation_keys.add((edge["source"], edge["target"], edge.get("label", "")))
                edge_ids.add(edge["id"])
    relation_type = refs.get("relation_type")
    if isinstance(relation_type, str):
        for edge in graph_data.get("edges", []):
            if edge.get("label", "") == relation_type and (
                not core_node_ids or edge.get("source") in core_node_ids or edge.get("target") in core_node_ids
            ):
                relation_keys.add((edge["source"], edge["target"], edge.get("label", "")))
                edge_ids.add(edge["id"])
                context_node_ids.update([edge["source"], edge["target"]])
    semantic_relation_types: set[str] = set()
    if isinstance(relation_type, str):
        semantic_relation_types.add(relation_type)
    elif refs.get("selected_edge_id"):
        selected = next((edge for edge in graph_data.get("edges", []) if edge.get("id") == refs["selected_edge_id"]), None)
        if selected is not None:
            semantic_relation_types.add(selected.get("label", ""))
    if not semantic_relation_types:
        semantic_relation_types = {key[2] for key in relation_keys}
    property_only = not relation_keys and not any(
        token in requirement.get("text", "").lower()
        for token in ("route", "before", "after", "between", "relation", "redirect", "retry", "success", "failure")
    )
    return {
        "node_ids": sorted(core_node_ids),
        "context_node_ids": sorted(context_node_ids),
        "edge_ids": sorted(edge_ids),
        "relation_keys": [list(item) for item in sorted(relation_keys)],
        "relation_types": sorted(semantic_relation_types),
        "property_only": property_only,
    }


def _requirements_overlap_or_adjacent(left: dict[str, Any], right: dict[str, Any]) -> bool:
    left_nodes = set(left["local_facts"]["context_node_ids"])
    right_nodes = set(right["local_facts"]["context_node_ids"])
    if left_nodes & right_nodes:
        return True
    left_core = set(left["local_facts"]["node_ids"])
    right_core = set(right["local_facts"]["node_ids"])
    return bool(
        left_core & set(right["local_facts"]["context_node_ids"])
        or right_core & set(left["local_facts"]["context_node_ids"])
    )


def _relation_semantics_compatible(left: dict[str, Any], right: dict[str, Any]) -> bool:
    left_types = set(left["local_facts"]["relation_types"])
    right_types = set(right["local_facts"]["relation_types"])
    if left_types and right_types:
        return bool(left_types & right_types)
    return True


def _property_depends_on_relation(prop_req: dict[str, Any], other_req: dict[str, Any]) -> bool:
    prop_text = prop_req["text"].lower()
    if not any(token in prop_text for token in ("wait", "join", "incoming", "outgoing", "all incoming", "all outgoing")):
        return False
    prop_nodes = set(prop_req["local_facts"]["node_ids"])
    other_nodes = set(other_req["local_facts"]["context_node_ids"])
    if not prop_nodes or not other_nodes:
        return False
    return prop_nodes <= other_nodes and any(
        token in other_req["text"].lower()
        for token in ("between", "before", "after", "route", "relation", "redirect")
    )


def _requirements_structurally_dependent(left: dict[str, Any], right: dict[str, Any]) -> bool:
    dependency_tokens = (
        "reconnect",
        "former predecessor",
        "former successor",
        "old predecessor",
        "old successor",
        "original predecessor",
        "original successor",
    )
    topology_tokens = ("between", "before", "after", "route", "sequence", "parallel", "insert", "add ", "create ", "new ")
    left_text = left["text"].lower()
    right_text = right["text"].lower()
    if not (any(token in left_text for token in dependency_tokens) or any(token in right_text for token in dependency_tokens)):
        return False
    if not (any(token in left_text for token in topology_tokens) or any(token in right_text for token in topology_tokens)):
        return False
    left_nodes = set(left["local_facts"]["context_node_ids"])
    right_nodes = set(right["local_facts"]["context_node_ids"])
    return bool(left_nodes & right_nodes)


def form_spec_units(
    grounded_requirements: list[dict[str, Any]],
    grounded_references: list[dict[str, Any]],
    local_graph_facts: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    items = []
    for requirement, references, facts in zip(grounded_requirements, grounded_references, local_graph_facts, strict=True):
        items.append(
            {
                "id": requirement["id"],
                "requirement": deepcopy(requirement),
                "text": requirement["text"],
                "references": references,
                "local_facts": facts,
            }
        )
    parent = {item["id"]: item["id"] for item in items}

    def find(req_id: str) -> str:
        while parent[req_id] != req_id:
            parent[req_id] = parent[parent[req_id]]
            req_id = parent[req_id]
        return req_id

    def union(left_id: str, right_id: str) -> None:
        left_root, right_root = find(left_id), find(right_id)
        if left_root != right_root:
            parent[right_root] = left_root

    for index, left in enumerate(items):
        for right in items[index + 1 :]:
            if left["local_facts"]["property_only"] and right["local_facts"]["property_only"]:
                continue
            if left["local_facts"]["property_only"] and _property_depends_on_relation(left, right):
                union(left["id"], right["id"])
                continue
            if right["local_facts"]["property_only"] and _property_depends_on_relation(right, left):
                union(left["id"], right["id"])
                continue
            if _requirements_structurally_dependent(left, right):
                union(left["id"], right["id"])
                continue
            if _relation_semantics_compatible(left, right) and _requirements_overlap_or_adjacent(left, right):
                union(left["id"], right["id"])

    groups: dict[str, list[dict[str, Any]]] = {}
    for item in items:
        groups.setdefault(find(item["id"]), []).append(item)
    units: list[dict[str, Any]] = []
    for index, members in enumerate(sorted(groups.values(), key=lambda rows: sorted(row["id"] for row in rows)), start=1):
        core_node_ids = sorted({node_id for member in members for node_id in member["local_facts"]["node_ids"]})
        context_node_ids = sorted({node_id for member in members for node_id in member["local_facts"]["context_node_ids"]})
        edge_ids = sorted({edge_id for member in members for edge_id in member["local_facts"]["edge_ids"]})
        relation_types = sorted({label for member in members for label in member["local_facts"]["relation_types"]})
        unit_kind = "property" if all(member["local_facts"]["property_only"] for member in members) else "topology"
        units.append(
            {
                "unit_id": f"u{index}",
                "requirement_ids": [member["id"] for member in members],
                "requirements": [deepcopy(member["requirement"]) for member in members],
                "references": [member["references"] for member in members],
                "local_graph_facts": {
                    "node_ids": core_node_ids,
                    "context_node_ids": context_node_ids,
                    "edge_ids": edge_ids,
                    "relation_types": relation_types,
                },
                "signature": {"kind": unit_kind, "nodes": core_node_ids, "relation_types": relation_types},
            }
        )
    return units


def requirement_conservation(grounded_requirements: list[dict[str, Any]], units: list[dict[str, Any]]) -> dict[str, Any]:
    requirement_ids = [requirement["id"] for requirement in grounded_requirements if isinstance(requirement.get("id"), str)]
    owners: dict[str, list[str]] = {requirement_id: [] for requirement_id in requirement_ids}
    for unit in units:
        unit_id = unit.get("unit_id")
        for requirement_id in unit.get("requirement_ids", []):
            if requirement_id in owners and isinstance(unit_id, str):
                owners[requirement_id].append(unit_id)
    owner_mapping = {requirement_id: owner_ids[0] for requirement_id, owner_ids in owners.items() if len(owner_ids) == 1}
    missing = sorted(requirement_id for requirement_id, owner_ids in owners.items() if not owner_ids)
    duplicate = {requirement_id: owner_ids for requirement_id, owner_ids in owners.items() if len(owner_ids) > 1}
    return {
        "grounded_requirement_ids": requirement_ids,
        "owner_mapping": owner_mapping,
        "missing_owners": missing,
        "duplicate_owners": duplicate,
        "valid": not missing and not duplicate,
    }


def graph_context_for_unit(graph_data: dict[str, Any], unit: dict[str, Any]) -> dict[str, Any]:
    node_ids = set(unit.get("local_graph_facts", {}).get("context_node_ids", []))
    edge_ids = set(unit.get("local_graph_facts", {}).get("edge_ids", []))
    nodes = [deepcopy(node) for node in graph_data.get("nodes", []) if node.get("id") in node_ids]
    edges = [
        deepcopy(edge)
        for edge in graph_data.get("edges", [])
        if edge.get("id") in edge_ids or edge.get("source") in node_ids or edge.get("target") in node_ids
    ]
    return {"nodes": nodes, "edges": edges}


def _property_targets_from_requirements(requirements: list[dict[str, Any]], references: list[dict[str, Any]]) -> set[str]:
    allowed: set[str] = set()
    for requirement, refs in zip(requirements, references, strict=True):
        desired_state = requirement.get("grounded_desired_state", {}) if isinstance(requirement.get("grounded_desired_state"), dict) else {}
        mentioned = set(refs.get("mentioned_node_ids", [])) if isinstance(refs, dict) else set()
        for row in desired_state.get("properties", []):
            if isinstance(row, dict) and isinstance(row.get("node_id"), str):
                allowed.add(row["node_id"])
        if not allowed:
            text = requirement.get("text", "").lower()
            if any(token in text for token in ("wait", "join", "incoming", "all incoming", "rename")):
                allowed.update(mentioned)
                if isinstance(refs, dict) and isinstance(refs.get("target_node_id"), str):
                    allowed.add(refs["target_node_id"])
                if isinstance(refs, dict) and isinstance(refs.get("source_node_id"), str):
                    allowed.add(refs["source_node_id"])
        scope_policy = desired_state.get("scope_policy", {}) if isinstance(desired_state.get("scope_policy"), dict) else {}
        if _is_abstract_requirement(requirement, refs):
            allowed.update(node_id for node_id in scope_policy.get("node_ids", []) if isinstance(node_id, str))
    return allowed


def _requires_structural_region(requirement: dict[str, Any], _ref: dict[str, Any]) -> bool:
    text = requirement.get("text", "").lower()
    topology_tokens = ("after", "before", "between", "parallel", "branch", "sequence", "route", "insert", "add ", "create ", "new ")
    has_topology_signal = any(token in text for token in topology_tokens)
    if any(token in text for token in ("rename", "wait", "join", "incoming")) and not has_topology_signal:
        return False
    return has_topology_signal


def _requires_explicit_relation(requirement: dict[str, Any], ref: dict[str, Any]) -> bool:
    desired_state = requirement.get("grounded_desired_state", {}) if isinstance(requirement.get("grounded_desired_state"), dict) else {}
    if desired_state.get("explicit_relations"):
        return True
    text = requirement.get("text", "").lower()
    source = ref.get("source_node_id") if isinstance(ref, dict) else None
    target = ref.get("target_node_id") if isinstance(ref, dict) else None
    relation_type = ref.get("relation_type") if isinstance(ref, dict) else None
    if not (isinstance(source, str) and isinstance(target, str) and isinstance(relation_type, str)):
        return False
    return any(
        token in text
        for token in (
            "redirect",
            "restart",
            "retry",
            "exception",
            "failure relation",
            "success relation",
            "relation from",
        )
    )


def edge_exists(graph_data: dict[str, Any], source: str, target: str, label: str) -> bool:
    return any(
        edge.get("source") == source and edge.get("target") == target and edge.get("label", "") == label
        for edge in graph_data.get("edges", [])
    )


def _edge_by_id(graph_data: dict[str, Any], edge_id: str | None) -> dict[str, Any] | None:
    if not isinstance(edge_id, str) or not edge_id:
        return None
    return next((edge for edge in graph_data.get("edges", []) if isinstance(edge, dict) and edge.get("id") == edge_id), None)


def _relation_rows_fully_cover_existing_relation_mutation(
    graph_data: dict[str, Any],
    requirement: dict[str, Any],
    refs: dict[str, Any],
    explicit_rows: list[dict[str, Any]],
) -> bool:
    """True when a requirement is already a complete deterministic edge target state."""
    if not explicit_rows:
        return False
    source = refs.get("source_node_id")
    old_target = refs.get("target_node_id")
    relation_type = refs.get("relation_type")
    if not (isinstance(source, str) and isinstance(old_target, str) and isinstance(relation_type, str)):
        return False
    selected_edge = _edge_by_id(graph_data, refs.get("selected_edge_id"))
    if selected_edge is not None:
        if selected_edge.get("source") != source or selected_edge.get("target") != old_target or selected_edge.get("label", "") != relation_type:
            return False
    elif not edge_exists(graph_data, source, old_target, relation_type):
        return False
    rows_for_group = [
        row
        for row in explicit_rows
        if row.get("source") == source and row.get("relation_type") == relation_type and isinstance(row.get("target"), str)
    ]
    if not rows_for_group:
        return False
    return any(row["target"] != old_target for row in rows_for_group)


def explicit_relation_rows_for_requirement(requirement: dict[str, Any], refs: dict[str, Any], graph_data: dict[str, Any]) -> list[dict[str, str]]:
    text = str(requirement.get("text", "")).lower()
    mentioned = {node_id for node_id in refs.get("mentioned_node_ids", []) if isinstance(node_id, str)}
    source = refs.get("source_node_id") if isinstance(refs.get("source_node_id"), str) else None
    target = refs.get("target_node_id") if isinstance(refs.get("target_node_id"), str) else None
    relation_type = refs.get("relation_type") if isinstance(refs.get("relation_type"), str) else None
    rows: list[dict[str, str]] = []
    scope_policy = requirement.get("grounded_desired_state", {}) if isinstance(requirement.get("grounded_desired_state"), dict) else {}
    scope_policy = scope_policy.get("scope_policy", {}) if isinstance(scope_policy.get("scope_policy"), dict) else {}
    scope_nodes = _scope_node_filter(scope_policy)
    explicit_relation_tokens = (
        "redirect",
        "restart",
        "retry",
        "exception",
        "failure relation",
        "success relation",
        "relation from",
        "reconnect",
        "former predecessor",
        "old predecessor",
        "original predecessor",
    )
    all_relation_target_request = (
        isinstance(target, str)
        and isinstance(relation_type, str)
        and any(token in text for token in ("every ", "all "))
        and any(token in text for token in ("relation", "edge", "route"))
        and any(token in text for token in (" to ", "toward ", "towards "))
    )
    if all_relation_target_request:
        candidate_edges = [
            edge
            for edge in graph_data.get("edges", [])
            if isinstance(edge, dict)
            and edge.get("label", "") == relation_type
            and isinstance(edge.get("source"), str)
            and (not scope_nodes or edge["source"] in scope_nodes)
        ]
        for edge in candidate_edges:
            rows.append({"source": edge["source"], "relation_type": relation_type, "target": target})
    elif "restart at" in text and {"manual", "start"} <= mentioned and not edge_exists(graph_data, "manual", "start", ""):
        rows.append({"source": "manual", "relation_type": "", "target": "start"})
    elif any(token in text for token in ("former predecessor", "old predecessor", "original predecessor")):
        anchors = {node_id for node_id in (source, target) if isinstance(node_id, str)}
        candidate_nodes = sorted(node_id for node_id in mentioned if node_id not in anchors)
        for candidate in candidate_nodes:
            incoming = [
                edge
                for edge in graph_data.get("edges", [])
                if edge.get("target") == candidate and isinstance(edge.get("source"), str)
            ]
            if not incoming:
                continue
            if anchors and not any(edge.get("source") == candidate and edge.get("target") in anchors for edge in graph_data.get("edges", [])):
                continue
            for anchor in sorted(anchors):
                for edge in incoming:
                    label = edge.get("label", "")
                    predecessor = edge["source"]
                    if predecessor != anchor and not edge_exists(graph_data, predecessor, anchor, label):
                        rows.append({"source": predecessor, "relation_type": label, "target": anchor})
    elif any(token in text for token in explicit_relation_tokens) and isinstance(relation_type, str) and source:
        resolved_target = target
        alternate_targets = sorted(node_id for node_id in mentioned if node_id not in {source, target})
        if (
            len(alternate_targets) == 1
            and resolved_target
            and any(token in text for token in (" instead of ", " to ", " toward ", " towards "))
            and edge_exists(graph_data, source, resolved_target, relation_type)
            and not edge_exists(graph_data, source, alternate_targets[0], relation_type)
        ):
            resolved_target = alternate_targets[0]
        elif resolved_target is None and len(alternate_targets) == 1:
            resolved_target = alternate_targets[0]
        if resolved_target and not edge_exists(graph_data, source, resolved_target, relation_type):
            rows.append({"source": source, "relation_type": relation_type, "target": resolved_target})
    return rows


def _normalize_abstract_goals(requirement: dict[str, Any]) -> list[dict[str, str]]:
    desired_state = requirement.get("desired_state", {}) if isinstance(requirement.get("desired_state"), dict) else {}
    rows = []
    for row in desired_state.get("abstract_goals", []):
        if not isinstance(row, dict):
            continue
        goal_type = row.get("goal_type")
        description = row.get("description", requirement.get("text", ""))
        if isinstance(goal_type, str) and goal_type:
            rows.append({"goal_type": goal_type, "description": str(description)})
    return rows


def _has_delete_intent(text: str) -> bool:
    lowered = text.lower()
    if any(
        phrase in lowered
        for phrase in (
            "do not remove",
            "don't remove",
            "dont remove",
            "do not delete",
            "don't delete",
            "dont delete",
            "without removing",
            "keep all",
            "preserve all",
        )
    ):
        return False
    return any(token in lowered for token in ("delete", "remove"))


def _has_concrete_edit_signal(requirement: dict[str, Any], refs: dict[str, Any]) -> bool:
    desired_state = requirement.get("grounded_desired_state", {}) if isinstance(requirement.get("grounded_desired_state"), dict) else {}
    if desired_state.get("properties") or desired_state.get("explicit_relations") or desired_state.get("absent_entities"):
        return True
    text = str(requirement.get("text", "")).lower()
    concrete_tokens = (
        "rename",
        "relabel",
        "label ",
        "redirect",
        "retry",
        "exception",
        "failure relation",
        "success relation",
        "after",
        "before",
        "between",
        "parallel",
        "branch",
        "sequence",
        "insert",
        "add ",
        "create ",
        "new ",
        "wait",
        "join",
        "incoming",
    )
    if any(token in text for token in concrete_tokens):
        return True
    if any(_contains_word(text, token) for token in ("move", "shift", "pixel", "position")):
        return True
    if _has_delete_intent(text):
        return True
    return _requires_explicit_relation(requirement, refs) or _requires_structural_region(requirement, refs)


def _is_abstract_requirement(requirement: dict[str, Any], refs: dict[str, Any]) -> bool:
    desired_state = requirement.get("grounded_desired_state", {}) if isinstance(requirement.get("grounded_desired_state"), dict) else {}
    if "grounded_desired_state" not in requirement and isinstance(requirement.get("desired_state"), dict):
        desired_state = requirement["desired_state"]
    clarification = desired_state.get("clarification", {}) if isinstance(desired_state.get("clarification"), dict) else {}
    if clarification.get("needed") is True:
        return False
    return any(isinstance(row, dict) for row in desired_state.get("abstract_goals", []))


def build_unit_contract(graph_data: dict[str, Any], unit: dict[str, Any]) -> dict[str, Any]:
    requirements = unit.get("requirements", [])
    references = unit.get("references", [])
    texts = [item.get("text", "") for item in requirements]
    relation_types = {
        label
        for label in unit.get("local_graph_facts", {}).get("relation_types", [])
        if isinstance(label, str)
    }
    structural_relation_types: set[str] = set()
    auxiliary_groups: set[tuple[str, str]] = set()
    allow_absent_entities = False
    for requirement, ref in zip(requirements, references, strict=True):
        if not isinstance(ref, dict):
            continue
        source = ref.get("source_node_id")
        relation_type = ref.get("relation_type")
        if _requires_structural_region(requirement, ref):
            if isinstance(relation_type, str):
                structural_relation_types.add(relation_type)
            else:
                structural_relation_types.update(relation_types)
        elif _is_abstract_requirement(requirement, ref):
            scope_policy = requirement.get("grounded_desired_state", {}) if isinstance(requirement.get("grounded_desired_state"), dict) else {}
            scope_policy = scope_policy.get("scope_policy", {}) if isinstance(scope_policy.get("scope_policy"), dict) else {}
            if scope_policy.get("kind") in {"stage", "group", "subprocess", "global"}:
                if "" in relation_types:
                    structural_relation_types.add("")
                else:
                    structural_relation_types.update(relation_types)
        if _requires_explicit_relation(requirement, ref) and isinstance(source, str) and isinstance(relation_type, str):
            auxiliary_groups.add((source, relation_type))
        for row in explicit_relation_rows_for_requirement(requirement, ref, graph_data):
            auxiliary_groups.add((row["source"], row["relation_type"]))
        desired_state = requirement.get("grounded_desired_state", {}) if isinstance(requirement.get("grounded_desired_state"), dict) else {}
        if desired_state.get("absent_entities") or any(token in requirement.get("text", "").lower() for token in ("delete", "remove")):
            allow_absent_entities = True
    auxiliary_relation_types = sorted({label for _source, label in auxiliary_groups})
    property_targets = sorted(_property_targets_from_requirements(requirements, references))
    return {
        "allowed_structural_relation_types": sorted(structural_relation_types),
        "allowed_auxiliary_relation_types": auxiliary_relation_types,
        "allowed_auxiliary_relation_groups": [
            {"source": source, "relation_type": relation_type}
            for source, relation_type in sorted(auxiliary_groups)
        ],
        "allowed_existing_node_ids": sorted(
            {node["id"] for node in graph_context_for_unit(graph_data, unit).get("nodes", []) if isinstance(node.get("id"), str)}
        ),
        "allow_new_entities": goal_allows_entity_creation(texts),
        "allowed_property_targets": property_targets,
        "allow_absent_entities": allow_absent_entities,
    }


def _abstract_preserve_constraints(scope_node_ids: list[str], graph_data: dict[str, Any]) -> dict[str, Any]:
    scope_set = {node_id for node_id in scope_node_ids if isinstance(node_id, str)}
    labeled_edges = []
    for edge in graph_data.get("edges", []):
        if not isinstance(edge, dict):
            continue
        source = edge.get("source")
        target = edge.get("target")
        relation_type = edge.get("label", "")
        if (
            isinstance(source, str)
            and isinstance(target, str)
            and isinstance(relation_type, str)
            and relation_type
            and source in scope_set
            and target in scope_set
        ):
            labeled_edges.append({"source": source, "target": target, "relation_type": relation_type})
    return {
        "preserve_node_ids": sorted(scope_set),
        "preserve_labeled_relations": sorted(labeled_edges, key=lambda row: (row["source"], row["relation_type"], row["target"])),
    }


def explicit_obligations(requirements: list[dict[str, Any]], graph_data: dict[str, Any]) -> dict[str, Any]:
    obligations = {"auxiliary": [], "properties": [], "entity_create": False, "absent_entities": []}
    for requirement in requirements:
        text = str(requirement.get("text", "")).lower()
        original_text = str(requirement.get("text", ""))
        refs = requirement.get("references", {}) if isinstance(requirement.get("references"), dict) else {}
        desired_state = requirement.get("grounded_desired_state", {}) if isinstance(requirement.get("grounded_desired_state"), dict) else {}
        mentioned = {node_id for node_id in refs.get("mentioned_node_ids", []) if isinstance(node_id, str)}
        source = refs.get("source_node_id") if isinstance(refs.get("source_node_id"), str) else None
        target = refs.get("target_node_id") if isinstance(refs.get("target_node_id"), str) else None

        if any(token in text for token in ("add ", "create ", "insert ", "new ", "append ", "introduce ")):
            obligations["entity_create"] = True
        if desired_state.get("absent_entities"):
            obligations["absent_entities"].extend(desired_state["absent_entities"])
        elif any(token in text for token in ("delete ", "remove ")):
            for node_id in sorted(mentioned | ({source} if source else set()) | ({target} if target else set())):
                obligations["absent_entities"].append({"node_id": node_id})
        if desired_state.get("properties"):
            for row in desired_state["properties"]:
                if not (isinstance(row, dict) and isinstance(row.get("node_id"), str) and isinstance(row.get("changes"), dict)):
                    continue
                changes = normalize_changes(row["changes"])
                if not changes:
                    continue
                if changes.get("join") == {"mode": "all"}:
                    obligations["properties"].append(
                        {
                            "node_id": row["node_id"],
                            "target_entity_ref": row.get("target_entity_ref", row["node_id"]),
                            "property": "join",
                            "kind": "join_all",
                            "value": {"mode": "all"},
                            "source_requirement_id": row.get("source_requirement_id", requirement.get("id")),
                            "derivation": (row.get("derivations", {}) or {}).get("join", "grounded_explicit_join"),
                        }
                    )
                for key, value in changes.items():
                    if key == "join":
                        continue
                    obligations["properties"].append(
                        {
                            "node_id": row["node_id"],
                            "target_entity_ref": row.get("target_entity_ref", row["node_id"]),
                            "property": key,
                            "kind": key,
                            "value": value,
                            "source_requirement_id": row.get("source_requirement_id", requirement.get("id")),
                            "derivation": (row.get("derivations", {}) or {}).get(key, "grounded_explicit_property"),
                        }
                    )
        elif any(token in text for token in ("wait", "join", "incoming", "all incoming")):
            property_target = source or target or next(iter(mentioned), None)
            if property_target:
                obligations["properties"].append(
                    {
                        "node_id": property_target,
                        "target_entity_ref": property_target,
                        "property": "join",
                        "kind": "join_all",
                        "value": {"mode": "all"},
                        "source_requirement_id": requirement.get("id"),
                        "derivation": "textual_explicit_join",
                    }
                )
        if not desired_state.get("properties") and text.startswith("rename ") and " to " in text:
            property_target = source or next(iter(mentioned), None)
            new_label = original_text.split(" to ", 1)[1].strip().rstrip(".")
            if property_target and new_label:
                obligations["properties"].append(
                    {
                        "node_id": property_target,
                        "target_entity_ref": property_target,
                        "property": "label",
                        "kind": "label",
                        "value": new_label,
                        "source_requirement_id": requirement.get("id"),
                        "derivation": "textual_explicit_label",
                    }
                )
        if desired_state.get("explicit_relations"):
            obligations["auxiliary"].extend(desired_state["explicit_relations"])
        else:
            obligations["auxiliary"].extend(explicit_relation_rows_for_requirement(requirement, refs, graph_data))

    obligations["auxiliary"] = list(
        {(row["source"], row["relation_type"], row["target"]): row for row in obligations["auxiliary"]}.values()
    )
    obligations["properties"] = list(
        {
            (
                row["node_id"],
                row["kind"],
                json.dumps(row.get("value"), ensure_ascii=False, sort_keys=True),
                row.get("source_requirement_id"),
                row.get("derivation"),
            ): row
            for row in obligations["properties"]
        }.values()
    )
    obligations["absent_entities"] = list({row["node_id"]: row for row in obligations["absent_entities"]}.values())
    return obligations


def required_obligations(
    unit: dict[str, Any],
    graph_data: dict[str, Any],
    unit_contract: dict[str, Any],
    global_obligations: dict[str, Any],
) -> dict[str, Any]:
    obligations = {
        "auxiliary": [],
        "properties": [],
        "entity_create": False,
        "absent_entities": [],
        "structural_participants": [],
        "abstract_goals": [],
    }
    allowed_aux_groups = {
        (row["source"], row["relation_type"])
        for row in unit_contract.get("allowed_auxiliary_relation_groups", [])
        if isinstance(row, dict) and isinstance(row.get("source"), str) and isinstance(row.get("relation_type"), str)
    }
    for requirement, refs in zip(unit.get("requirements", []), unit.get("references", []), strict=True):
        text = str(requirement.get("text", "")).lower()
        desired_state = requirement.get("grounded_desired_state", {}) if isinstance(requirement.get("grounded_desired_state"), dict) else {}
        mentioned = {node_id for node_id in refs.get("mentioned_node_ids", []) if isinstance(node_id, str)}
        resolved_scope_nodes = {node_id for node_id in refs.get("resolved_scope_node_ids", []) if isinstance(node_id, str)}
        source = refs.get("source_node_id") if isinstance(refs.get("source_node_id"), str) else None
        target = refs.get("target_node_id") if isinstance(refs.get("target_node_id"), str) else None

        if any(token in text for token in ("add ", "create ", "insert ", "new ", "append ", "introduce ")):
            obligations["entity_create"] = True
        if desired_state.get("absent_entities"):
            obligations["absent_entities"].extend(desired_state["absent_entities"])
        elif any(token in text for token in ("delete ", "remove ")):
            for node_id in sorted(mentioned | ({source} if source else set()) | ({target} if target else set())):
                obligations["absent_entities"].append({"node_id": node_id})

        base_participants = {node_id for node_id in mentioned | ({source} if source else set()) | ({target} if target else set()) if node_id}
        explicit_rows = desired_state.get("explicit_relations", []) or explicit_relation_rows_for_requirement(requirement, refs, graph_data)
        explicit_participants = {
            node_id
            for row in explicit_rows
            for node_id in (row.get("source"), row.get("target"))
            if isinstance(node_id, str)
        }
        explicit_relation_fully_covers_requirement = (
            (bool(explicit_rows) and bool(base_participants) and base_participants <= explicit_participants)
            or _relation_rows_fully_cover_existing_relation_mutation(graph_data, requirement, refs, explicit_rows)
        )
        if (
            base_participants
            and unit_contract.get("allowed_structural_relation_types")
            and _requires_structural_region(requirement, refs)
            and not explicit_relation_fully_covers_requirement
        ):
            relation_types = unit_contract.get("allowed_structural_relation_types", [])
            ref_relation_type = refs.get("relation_type") if isinstance(refs.get("relation_type"), str) else None
            obligation_relation_types = [ref_relation_type] if ref_relation_type in relation_types else relation_types
            for allowed_type in obligation_relation_types:
                relation_participants = set(base_participants)
                for row in explicit_rows:
                    if row.get("relation_type") != allowed_type:
                        continue
                    if isinstance(row.get("source"), str):
                        relation_participants.add(row["source"])
                    if isinstance(row.get("target"), str):
                        relation_participants.add(row["target"])
                participants = sorted(relation_participants)
                obligations["structural_participants"].append({"relation_type": allowed_type, "node_ids": participants})
        elif _is_abstract_requirement(requirement, refs):
            scope_policy = desired_state.get("scope_policy", {}) if isinstance(desired_state.get("scope_policy"), dict) else {}
            scope_node_ids = [node_id for node_id in scope_policy.get("node_ids", []) if isinstance(node_id, str)]
            if scope_node_ids:
                preserve_constraints = _abstract_preserve_constraints(scope_node_ids, graph_data)
                abstract_goals = desired_state.get("abstract_goals", [])
                for goal in abstract_goals if isinstance(abstract_goals, list) else []:
                    if not isinstance(goal, dict):
                        continue
                    obligations["abstract_goals"].append(
                        {
                            "requirement_id": requirement.get("id"),
                            "goal_type": goal.get("goal_type"),
                            "goal_text": requirement.get("text", ""),
                            "description": goal.get("description", ""),
                            "scope_kind": scope_policy.get("kind"),
                            "scope_node_ids": scope_node_ids,
                            "allowed_structural_relation_types": list(unit_contract.get("allowed_structural_relation_types", [])),
                            "allowed_property_targets": [node_id for node_id in unit_contract.get("allowed_property_targets", []) if node_id in scope_node_ids],
                            "preserve_constraints": preserve_constraints,
                        }
                    )
        obligations["auxiliary"].extend(explicit_rows)

    property_targets = set(unit_contract.get("allowed_property_targets", []))
    for row in global_obligations["properties"]:
        if row["node_id"] in property_targets:
            obligations["properties"].append(row)
    for row in global_obligations["auxiliary"]:
        if (row["source"], row["relation_type"]) in allowed_aux_groups:
            obligations["auxiliary"].append(row)
    if global_obligations["entity_create"] and unit_contract.get("allow_new_entities"):
        obligations["entity_create"] = True
    if unit_contract.get("allow_absent_entities"):
        visible_ids = set(unit_contract.get("allowed_existing_node_ids", []))
        for row in global_obligations["absent_entities"]:
            if row["node_id"] in visible_ids:
                obligations["absent_entities"].append(row)

    obligations["auxiliary"] = list(
        {(row["source"], row["relation_type"], row["target"]): row for row in obligations["auxiliary"]}.values()
    )
    obligations["properties"] = list(
        {
            (
                row["node_id"],
                row["kind"],
                json.dumps(row.get("value"), ensure_ascii=False, sort_keys=True),
                row.get("source_requirement_id"),
                row.get("derivation"),
            ): row
            for row in obligations["properties"]
        }.values()
    )
    obligations["absent_entities"] = list({row["node_id"]: row for row in obligations["absent_entities"]}.values())
    structural_unique: dict[tuple[str, tuple[str, ...]], dict[str, Any]] = {}
    for row in obligations["structural_participants"]:
        structural_unique[(row["relation_type"], tuple(row["node_ids"]))] = row
    obligations["structural_participants"] = list(structural_unique.values())
    abstract_unique: dict[str, dict[str, Any]] = {}
    for row in obligations["abstract_goals"]:
        if isinstance(row.get("requirement_id"), str):
            abstract_unique[row["requirement_id"]] = row
    obligations["abstract_goals"] = list(abstract_unique.values())
    return obligations


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return slug or "new_node"


def unique_ref(base: str, existing_ids: set[str], used_refs: set[str]) -> str:
    candidate = base
    index = 2
    while candidate in existing_ids or candidate in used_refs:
        candidate = f"{base}_{index}"
        index += 1
    used_refs.add(candidate)
    return candidate


def deterministic_new_entities(unit: dict[str, Any], graph_data: dict[str, Any]) -> list[dict[str, str]]:
    existing_ids = {node["id"] for node in graph_data.get("nodes", []) if isinstance(node.get("id"), str)}
    used_refs: set[str] = set()
    entities: list[dict[str, str]] = []
    for requirement in unit.get("requirements", []):
        for row in _deterministic_new_entities_for_requirement(requirement):
            ref = unique_ref(slugify(row["label"]), existing_ids, used_refs)
            entities.append({"ref": ref, "node_type": row["node_type"], "label": row["label"]})
    return entities


def dedupe_entities(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return list({(row["ref"], row["node_type"], row["label"]): row for row in rows}.values())


def group_explicit_relations(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], set[str]] = defaultdict(set)
    for row in rows:
        grouped[(row["source"], row["relation_type"])].add(row["target"])
    return [
        {"source": source, "relation_type": relation_type, "targets": sorted(targets)}
        for (source, relation_type), targets in sorted(grouped.items())
    ]


def with_entity_specs(obligations: dict[str, Any], unit: dict[str, Any], graph_data: dict[str, Any]) -> dict[str, Any]:
    enriched = dict(obligations)
    entity_specs = list(enriched.get("entity_specs", []))
    if enriched.get("entity_create"):
        entity_specs.extend(deterministic_new_entities(unit, graph_data))
    enriched["entity_specs"] = dedupe_entities(entity_specs)
    return enriched


def skeleton_from_obligations(
    unit: dict[str, Any],
    graph_data: dict[str, Any],
    obligations: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, str]]]:
    deterministic_entities = dedupe_entities(obligations.get("entity_specs", []) or deterministic_new_entities(unit, graph_data))
    properties: list[dict[str, Any]] = []
    for row in obligations["properties"]:
        if row.get("property") == "join" or row["kind"] == "join_all":
            properties.append({"node_id": row["node_id"], "changes": {"join": {"mode": "all"}}})
        else:
            property_name = row.get("property", row["kind"])
            properties.append({"node_id": row["node_id"], "changes": {property_name: row["value"]}})
    skeleton = {
        "entities": deterministic_entities,
        "absent_entities": [{"node_id": row["node_id"]} for row in obligations["absent_entities"]],
        "properties": properties,
        "structural_regions": [],
        "explicit_relations": group_explicit_relations(obligations["auxiliary"]),
    }
    structural_slots = [
        {
            "relation_type": row["relation_type"],
            "required_existing_node_ids": list(row["node_ids"]),
            "allowed_new_refs": [entity["ref"] for entity in deterministic_entities],
        }
        for row in obligations["structural_participants"]
    ]
    abstract_slots = [
        {
            "requirement_id": row["requirement_id"],
            "goal_type": row.get("goal_type"),
            "goal_text": row["goal_text"],
            "description": row.get("description", ""),
            "scope_kind": row.get("scope_kind"),
            "scope_node_ids": list(row.get("scope_node_ids", [])),
            "allowed_structural_relation_types": list(row.get("allowed_structural_relation_types", [])),
            "allowed_property_targets": list(row.get("allowed_property_targets", [])),
            "preserve_constraints": deepcopy(row.get("preserve_constraints", {})) if isinstance(row.get("preserve_constraints"), dict) else {},
        }
        for row in obligations.get("abstract_goals", [])
    ]
    return skeleton, structural_slots, abstract_slots, deterministic_entities


def obligation_key(kind: str, row: dict[str, Any] | None = None) -> str:
    payload = json.dumps(row or {}, ensure_ascii=False, sort_keys=True)
    return f"{kind}:{payload}"


def obligation_realization(
    obligations: dict[str, Any],
    skeleton: dict[str, Any],
    structural_slots: list[dict[str, Any]],
    abstract_slots: list[dict[str, Any]],
    owner_unit_id: str,
) -> dict[str, Any]:
    realization_map: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in obligations.get("properties", []):
        realization_map[obligation_key("property", row)].append({"owner_unit_id": owner_unit_id, "realization_mode": "deterministic_fact"})
    for row in obligations.get("auxiliary", []):
        realization_map[obligation_key("explicit_relation", row)].append({"owner_unit_id": owner_unit_id, "realization_mode": "deterministic_fact"})
    for row in obligations.get("absent_entities", []):
        realization_map[obligation_key("absent_entity", row)].append({"owner_unit_id": owner_unit_id, "realization_mode": "deterministic_fact"})
    for row in obligations.get("entity_specs", []):
        realization_map[obligation_key("entity_new", row)].append({"owner_unit_id": owner_unit_id, "realization_mode": "deterministic_fact"})
    for row in obligations.get("structural_participants", []):
        realization_map[obligation_key("structural_participants", row)].append({"owner_unit_id": owner_unit_id, "realization_mode": "semantic_slot"})
    for row in obligations.get("abstract_goals", []):
        realization_map[obligation_key("abstract_goal", row)].append({"owner_unit_id": owner_unit_id, "realization_mode": "validation_contract"})

    expected_keys = {
        *(obligation_key("property", row) for row in obligations.get("properties", [])),
        *(obligation_key("explicit_relation", row) for row in obligations.get("auxiliary", [])),
        *(obligation_key("absent_entity", row) for row in obligations.get("absent_entities", [])),
        *(obligation_key("entity_new", row) for row in obligations.get("entity_specs", [])),
        *(obligation_key("structural_participants", row) for row in obligations.get("structural_participants", [])),
        *(obligation_key("abstract_goal", row) for row in obligations.get("abstract_goals", [])),
    }
    orphan_obligations = sorted(key for key in expected_keys if key not in realization_map)
    duplicate_realizations = {key: value for key, value in realization_map.items() if len(value) != 1}
    return {
        "owner_unit_id": owner_unit_id,
        "realizations": {key: value[0] for key, value in realization_map.items() if len(value) == 1},
        "orphan_obligations": orphan_obligations,
        "duplicate_realizations": duplicate_realizations,
        "obligation_realization_complete": not orphan_obligations and not duplicate_realizations,
        "deterministic_facts_count": len(skeleton["entities"]) + len(skeleton["absent_entities"]) + len(skeleton["properties"]) + len(skeleton["explicit_relations"]),
        "semantic_slots_count": len(structural_slots),
    }


def generation_contract(
    unit_contract: dict[str, Any],
    structural_slots: list[dict[str, Any]],
    abstract_slots: list[dict[str, Any]],
    deterministic_entities: list[dict[str, str]],
) -> dict[str, Any]:
    if not structural_slots and not abstract_slots:
        return {
            "allowed_structural_relation_types": [],
            "allowed_auxiliary_relation_types": [],
            "allowed_auxiliary_relation_groups": [],
            "allowed_existing_node_ids": [],
            "allow_new_entities": False,
            "allowed_property_targets": [],
            "allow_absent_entities": False,
        }
    allowed_structural = sorted(
        {
            *[slot["relation_type"] for slot in structural_slots],
            *[relation_type for slot in abstract_slots for relation_type in slot.get("allowed_structural_relation_types", [])],
        }
    )
    locked_existing_ids = sorted(
        {
            *[node_id for slot in structural_slots for node_id in slot["required_existing_node_ids"]],
            *[node_id for slot in abstract_slots for node_id in slot.get("scope_node_ids", [])],
        }
    )
    allowed_property_targets = sorted(
        {
            *[node_id for slot in abstract_slots for node_id in slot.get("allowed_property_targets", [])],
        }
    )
    return {
        "allowed_structural_relation_types": allowed_structural,
        "allowed_auxiliary_relation_types": [],
        "allowed_auxiliary_relation_groups": [],
        "allowed_existing_node_ids": [node_id for node_id in unit_contract.get("allowed_existing_node_ids", []) if node_id in locked_existing_ids],
        "allow_new_entities": bool(deterministic_entities),
        "allowed_property_targets": allowed_property_targets,
        "allow_absent_entities": False,
        "locked_new_entities": deterministic_entities,
        "locked_structural_slots": structural_slots,
        "locked_abstract_slots": abstract_slots,
    }


def normalize_changes(changes: dict[str, Any]) -> dict[str, Any]:
    data = dict(changes)
    if data.get("wait_for_all_incoming") is True or data.get("waitForAllIncoming") is True:
        data["join"] = {"mode": "all"}
    join_value = data.get("join")
    if isinstance(join_value, dict) and join_value.get("mode") in {"all", "any"}:
        data["join"] = {"mode": join_value["mode"]}
    data.pop("wait_for_all_incoming", None)
    data.pop("waitForAllIncoming", None)
    if data.get("join") == {}:
        data.pop("join", None)
    return data


def normalize_ir(value: dict[str, Any]) -> dict[str, Any]:
    def normalize_structure(structure: dict[str, Any]) -> dict[str, Any]:
        row: dict[str, Any] = {"type": structure["type"]}
        if "id" in structure:
            row["id"] = structure["id"]
        if "ref" in structure:
            row["ref"] = structure["ref"]
        if "node_type" in structure:
            row["node_type"] = structure["node_type"]
        if "label" in structure:
            row["label"] = structure["label"]
        if structure["type"] == "sequence":
            row["items"] = [normalize_structure(item) for item in structure.get("items", [])]
        if structure["type"] in {"parallel", "choice"}:
            row["branches"] = [normalize_structure(item) for item in structure.get("branches", [])]
        if structure["type"] == "parallel" and structure.get("join_mode") in {"all", "any"}:
            row["join_mode"] = structure["join_mode"]
        return row

    return {
        "control_flow_regions": [
            {"relation_type": row["relation_type"], "structure": normalize_structure(row["structure"])}
            for row in value.get("control_flow_regions", [])
            if isinstance(row, dict) and isinstance(row.get("relation_type"), str) and isinstance(row.get("structure"), dict)
        ],
        "auxiliary_relations": sorted(
            [
                {"source": row["source"], "relation_type": row["relation_type"], "targets": sorted(row["targets"])}
                for row in value.get("auxiliary_relations", [])
                if isinstance(row, dict)
                and isinstance(row.get("source"), str)
                and isinstance(row.get("relation_type"), str)
                and isinstance(row.get("targets"), list)
            ],
            key=lambda row: (row["source"], row["relation_type"], tuple(row["targets"])),
        ),
        "absent_entities": sorted(
            [
                {"node_id": row["node_id"]}
                for row in value.get("absent_entities", [])
                if isinstance(row, dict) and isinstance(row.get("node_id"), str)
            ],
            key=lambda row: row["node_id"],
        ),
        "properties": sorted(
            [
                {"node_id": row["node_id"], "changes": normalize_changes(row["changes"])}
                for row in value.get("properties", [])
                if isinstance(row, dict)
                and isinstance(row.get("node_id"), str)
                and isinstance(row.get("changes"), dict)
                and normalize_changes(row["changes"])
            ],
            key=lambda row: row["node_id"],
        ),
    }


def canonicalize_schema_variants(value: dict[str, Any] | None) -> tuple[dict[str, Any] | None, bool]:
    def canonicalize_structure(structure: Any) -> Any:
        if not isinstance(structure, dict):
            return structure
        row = deepcopy(structure)
        kind = row.get("type")
        if kind == "sequence":
            if "children" in row and "items" not in row and isinstance(row["children"], list):
                row["items"] = row.pop("children")
            row["items"] = [canonicalize_structure(child) for child in row.get("items", [])]
            return row
        if kind in {"parallel", "choice"}:
            if "children" in row and "branches" not in row and isinstance(row["children"], list):
                row["branches"] = row.pop("children")
            if "items" in row and "branches" not in row and isinstance(row["items"], list):
                row["branches"] = row.pop("items")
            row["branches"] = [canonicalize_structure(child) for child in row.get("branches", [])]
            return row
        return row

    if not isinstance(value, dict):
        return value, False
    changed = False
    row = deepcopy(value)
    normalized_regions = []
    for region in row.get("control_flow_regions", []):
        if not isinstance(region, dict):
            normalized_regions.append(region)
            continue
        before = json.dumps(region.get("structure"), ensure_ascii=False, sort_keys=True)
        structure = canonicalize_structure(region.get("structure"))
        after = json.dumps(structure, ensure_ascii=False, sort_keys=True)
        changed = changed or before != after
        normalized_regions.append({**region, "structure": structure})
    row["control_flow_regions"] = normalized_regions
    return row, changed


def is_schema_structural_error(error: str | None) -> bool:
    if not isinstance(error, str):
        return False
    lowered = error.lower()
    return any(
        token in lowered
        for token in (
            "invalid workflow ir",
            "top-level fields must be arrays",
            "invalid structure node",
            "must contain children",
            "join_mode invalid",
            "invalid control_flow_region",
            "node must use exactly one of id or ref",
            "new node missing",
            "duplicate new ref",
            "json parse",
            "parse_error",
        )
    )


def validate_ir(value: dict[str, Any] | None, graph_data: dict[str, Any], unit_contract: dict[str, Any]) -> tuple[bool, str | None]:
    if not isinstance(value, dict):
        return False, "invalid workflow ir"
    if not all(isinstance(value.get(key, []), list) for key in ("control_flow_regions", "auxiliary_relations", "absent_entities", "properties")):
        return False, "workflow ir top-level fields must be arrays"
    known_nodes = graph_node_ids(graph_data)
    known_labels = graph_labels(graph_data)
    seen_refs: set[str] = set()
    visible_node_ids = set(unit_contract.get("allowed_existing_node_ids", []))
    allowed_structural = set(unit_contract.get("allowed_structural_relation_types", []))
    allowed_auxiliary = set(unit_contract.get("allowed_auxiliary_relation_types", []))
    allowed_auxiliary_groups = {
        (row["source"], row["relation_type"])
        for row in unit_contract.get("allowed_auxiliary_relation_groups", [])
        if isinstance(row, dict) and isinstance(row.get("source"), str) and isinstance(row.get("relation_type"), str)
    }
    allow_new_entities = bool(unit_contract.get("allow_new_entities"))
    allowed_property_targets = set(unit_contract.get("allowed_property_targets", []))
    allow_absent_entities = bool(unit_contract.get("allow_absent_entities"))

    def validate_structure(structure: Any) -> tuple[bool, str | None]:
        if not isinstance(structure, dict) or structure.get("type") not in STRUCTURE_TYPES:
            return False, "invalid structure node"
        kind = structure["type"]
        if kind == "node":
            node_id = structure.get("id")
            ref = structure.get("ref")
            has_id = isinstance(node_id, str) and bool(node_id)
            has_ref = isinstance(ref, str) and bool(ref)
            if has_id == has_ref:
                return False, "node must use exactly one of id or ref"
            if has_id:
                if node_id not in known_nodes:
                    return False, "node id missing from graph"
                if node_id not in visible_node_ids:
                    return False, "node id outside unit graph context"
            else:
                if not allow_new_entities:
                    return False, "new entity is not allowed by unit contract"
                if ref in seen_refs:
                    return False, "duplicate new ref"
                if not isinstance(structure.get("node_type"), str) or not structure["node_type"]:
                    return False, "new node missing node_type"
                if not isinstance(structure.get("label"), str) or not structure["label"]:
                    return False, "new node missing label"
                seen_refs.add(ref)
            return True, None
        children = structure.get("items" if kind == "sequence" else "branches")
        if not isinstance(children, list) or not children:
            return False, f"{kind} must contain children"
        if kind == "parallel" and structure.get("join_mode") not in {None, "all", "any"}:
            return False, "parallel join_mode invalid"
        for child in children:
            ok, error = validate_structure(child)
            if not ok:
                return False, error
        return True, None

    for region in value["control_flow_regions"]:
        if not (isinstance(region, dict) and isinstance(region.get("relation_type"), str) and isinstance(region.get("structure"), dict)):
            return False, "invalid control_flow_region"
        if region["relation_type"] not in known_labels:
            return False, "control_flow relation_type is not a real graph label"
        if region["relation_type"] not in allowed_structural:
            return False, "control_flow relation_type is not allowed by unit contract"
        ok, error = validate_structure(region["structure"])
        if not ok:
            return False, error
    for row in value["auxiliary_relations"]:
        if not (
            isinstance(row, dict)
            and isinstance(row.get("source"), str)
            and row["source"] in known_nodes
            and isinstance(row.get("relation_type"), str)
            and row["relation_type"] in known_labels
            and isinstance(row.get("targets"), list)
            and all(isinstance(target, str) and target in known_nodes for target in row["targets"])
        ):
            return False, "invalid auxiliary relation"
        if row["relation_type"] not in allowed_auxiliary:
            return False, "auxiliary relation_type is not allowed by unit contract"
        if allowed_auxiliary_groups and (row["source"], row["relation_type"]) not in allowed_auxiliary_groups:
            return False, "auxiliary relation group is not allowed by unit contract"
    for row in value["properties"]:
        if not (
            isinstance(row, dict)
            and isinstance(row.get("node_id"), str)
            and row["node_id"] in known_nodes
            and isinstance(row.get("changes"), dict)
            and normalize_changes(row["changes"])
        ):
            return False, "invalid property row"
        if row["node_id"] not in allowed_property_targets:
            return False, "property target is not allowed by unit contract"
    for row in value["absent_entities"]:
        if not (isinstance(row, dict) and isinstance(row.get("node_id"), str) and row["node_id"] in known_nodes):
            return False, "invalid absent entity row"
        if not allow_absent_entities:
            return False, "absent_entities are not allowed by unit contract"
    if not allowed_structural and value["control_flow_regions"]:
        return False, "control_flow_regions are not allowed by unit contract"
    if not allowed_auxiliary and value["auxiliary_relations"]:
        return False, "auxiliary_relations are not allowed by unit contract"
    return True, None


def ir_nodes(structure: dict[str, Any]) -> list[dict[str, Any]]:
    if structure.get("type") == "node":
        return [structure]
    children = structure.get("items" if structure.get("type") == "sequence" else "branches", [])
    rows: list[dict[str, Any]] = []
    for child in children:
        rows.extend(ir_nodes(child))
    return rows


def project_generated_ir(raw_ir: dict[str, Any] | None, generation_contract: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(raw_ir, dict):
        return {"control_flow_regions": [], "auxiliary_relations": [], "absent_entities": [], "properties": []}
    allowed_structural = set(generation_contract.get("allowed_structural_relation_types", []))
    allowed_auxiliary = set(generation_contract.get("allowed_auxiliary_relation_types", []))
    allowed_existing = set(generation_contract.get("allowed_existing_node_ids", []))
    allowed_property_targets = set(generation_contract.get("allowed_property_targets", []))
    allowed_auxiliary_groups = {
        (row["source"], row["relation_type"])
        for row in generation_contract.get("allowed_auxiliary_relation_groups", [])
        if isinstance(row, dict) and isinstance(row.get("source"), str) and isinstance(row.get("relation_type"), str)
    }
    locked_new_entities = {
        row["ref"]: {"node_type": row["node_type"], "label": row["label"]}
        for row in generation_contract.get("locked_new_entities", [])
        if isinstance(row, dict) and isinstance(row.get("ref"), str)
    }
    projected_regions: list[dict[str, Any]] = []

    def structure_allowed(structure: Any) -> bool:
        if not isinstance(structure, dict) or structure.get("type") not in STRUCTURE_TYPES:
            return False
        if structure["type"] == "node":
            if isinstance(structure.get("id"), str):
                return structure["id"] in allowed_existing
            if isinstance(structure.get("ref"), str):
                locked = locked_new_entities.get(structure["ref"])
                return locked is not None and structure.get("node_type") == locked["node_type"] and structure.get("label") == locked["label"]
            return False
        children = structure.get("items" if structure["type"] == "sequence" else "branches")
        return isinstance(children, list) and bool(children) and all(structure_allowed(child) for child in children)

    for row in raw_ir.get("control_flow_regions", []):
        if not (isinstance(row, dict) and row.get("relation_type") in allowed_structural and structure_allowed(row.get("structure"))):
            continue
        projected_regions.append({"relation_type": row["relation_type"], "structure": row["structure"]})

    projected_properties = []
    for row in raw_ir.get("properties", []):
        if not (
            isinstance(row, dict)
            and isinstance(row.get("node_id"), str)
            and row["node_id"] in allowed_property_targets
            and isinstance(row.get("changes"), dict)
            and normalize_changes(row["changes"])
        ):
            continue
        projected_properties.append({"node_id": row["node_id"], "changes": row["changes"]})

    projected_auxiliary = []
    for row in raw_ir.get("auxiliary_relations", []):
        if not (
            isinstance(row, dict)
            and isinstance(row.get("source"), str)
            and row["source"] in allowed_existing
            and isinstance(row.get("relation_type"), str)
            and row["relation_type"] in allowed_auxiliary
            and (not allowed_auxiliary_groups or (row["source"], row["relation_type"]) in allowed_auxiliary_groups)
            and isinstance(row.get("targets"), list)
            and all(isinstance(target, str) and target in allowed_existing for target in row["targets"])
        ):
            continue
        projected_auxiliary.append({"source": row["source"], "relation_type": row["relation_type"], "targets": row["targets"]})

    return normalize_ir(
        {
            "control_flow_regions": projected_regions,
            "auxiliary_relations": projected_auxiliary,
            "absent_entities": [],
            "properties": projected_properties,
        }
    )


def merge_skeleton_and_generated(skeleton: dict[str, Any], generated_ir: dict[str, Any]) -> dict[str, Any]:
    return normalize_ir(
        {
            "control_flow_regions": list(generated_ir.get("control_flow_regions", [])),
            "auxiliary_relations": list(skeleton.get("explicit_relations", [])) + list(generated_ir.get("auxiliary_relations", [])),
            "absent_entities": list(skeleton.get("absent_entities", [])),
            "properties": list(skeleton.get("properties", [])) + list(generated_ir.get("properties", [])),
        }
    )


def structural_participant_coverage(ir: dict[str, Any] | None, structural_slots: list[dict[str, Any]]) -> bool:
    if not structural_slots:
        return True
    if not isinstance(ir, dict):
        return False
    regions_by_relation_type: dict[str, list[set[str]]] = defaultdict(list)
    new_refs_by_relation_type: dict[str, list[set[str]]] = defaultdict(list)
    for region in ir.get("control_flow_regions", []):
        if not (isinstance(region, dict) and isinstance(region.get("relation_type"), str) and isinstance(region.get("structure"), dict)):
            continue
        existing_ids = {row["id"] for row in ir_nodes(region["structure"]) if isinstance(row, dict) and isinstance(row.get("id"), str)}
        new_refs = {row["ref"] for row in ir_nodes(region["structure"]) if isinstance(row, dict) and isinstance(row.get("ref"), str)}
        regions_by_relation_type[region["relation_type"]].append(existing_ids)
        new_refs_by_relation_type[region["relation_type"]].append(new_refs)
    for slot in structural_slots:
        relation_type = slot["relation_type"]
        required_existing = set(slot["required_existing_node_ids"])
        allowed_new_refs = set(slot["allowed_new_refs"])
        matched = False
        for existing_ids, new_refs in zip(
            regions_by_relation_type.get(relation_type, []),
            new_refs_by_relation_type.get(relation_type, []),
            strict=False,
        ):
            # A single structural region may deterministically realize multiple
            # locked participant slots for the same relation_type. Treat each
            # slot as covered when its required existing participants are
            # present, without requiring an exact one-slot-one-region split.
            if required_existing <= existing_ids and new_refs <= allowed_new_refs:
                matched = True
                break
        if not matched:
            return False
    return True


def llm_requirements(
    unit: dict[str, Any],
    structural_slots: list[dict[str, Any]],
    abstract_slots: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not structural_slots and not abstract_slots:
        return [], []
    slot_relation_types = {slot["relation_type"] for slot in structural_slots}
    abstract_requirement_ids = {
        slot["requirement_id"]
        for slot in abstract_slots
        if isinstance(slot, dict) and isinstance(slot.get("requirement_id"), str)
    }
    requirements: list[dict[str, Any]] = []
    references: list[dict[str, Any]] = []
    for requirement, ref in zip(unit.get("requirements", []), unit.get("references", []), strict=True):
        if requirement.get("id") in abstract_requirement_ids:
            requirements.append({"id": requirement.get("id"), "text": requirement.get("text", "")})
            references.append(ref)
            continue
        relation_type = ref.get("relation_type") if isinstance(ref, dict) else None
        if relation_type in slot_relation_types or not isinstance(relation_type, str):
            text = str(requirement.get("text", "")).lower()
            if not _requires_structural_region(requirement, ref) and any(
                token in text
                for token in (
                    "rename",
                    "wait",
                    "join",
                    "incoming",
                    "redirect",
                    "send ",
                    "restart",
                    "retry",
                    "exception",
                    "failure relation",
                    "relation from",
                    "reconnect",
                    "former predecessor",
                    "former successor",
                    "old predecessor",
                    "old successor",
                    "original predecessor",
                    "original successor",
                )
            ):
                continue
            requirements.append({"id": requirement.get("id"), "text": requirement.get("text", "")})
            references.append(ref)
    return requirements, references


def obligation_coverage(ir: dict[str, Any] | None, obligations: dict[str, Any]) -> tuple[bool, list[str]]:
    if not isinstance(ir, dict):
        return False, ["no final ir"]
    errors: list[str] = []
    aux_lookup = {
        (row.get("source"), row.get("relation_type"), target)
        for row in ir.get("auxiliary_relations", [])
        if isinstance(row, dict)
        for target in row.get("targets", [])
        if isinstance(target, str)
    }
    for row in obligations["auxiliary"]:
        key = (row["source"], row["relation_type"], row["target"])
        if key not in aux_lookup:
            errors.append(f"missing auxiliary obligation {key}")

    property_lookup = {
        (row.get("node_id"), json.dumps(row.get("changes", {}), sort_keys=True, ensure_ascii=False))
        for row in ir.get("properties", [])
        if isinstance(row, dict)
    }
    for row in obligations["properties"]:
        if row["kind"] == "join_all":
            key = (row["node_id"], json.dumps({"join": {"mode": "all"}}, sort_keys=True, ensure_ascii=False))
            if key not in property_lookup:
                errors.append(f"missing property obligation {row['node_id']}.join.mode=all")
        else:
            key = (row["node_id"], json.dumps({row["kind"]: row["value"]}, sort_keys=True, ensure_ascii=False))
            if key not in property_lookup:
                errors.append(f"missing property obligation {row['node_id']}.{row['kind']}={row['value']}")

    if obligations["entity_create"]:
        has_new_ref = any(
            "ref" in node
            for region in ir.get("control_flow_regions", [])
            if isinstance(region, dict)
            for node in ir_nodes(region.get("structure", {}))
            if isinstance(node, dict)
        )
        if not has_new_ref and not obligations.get("entity_specs"):
            errors.append("missing entity creation obligation")
        if obligations.get("entity_specs"):
            expected_refs = {row["ref"] for row in obligations["entity_specs"]}
            actual_refs = {
                node["ref"]
                for region in ir.get("control_flow_regions", [])
                if isinstance(region, dict)
                for node in ir_nodes(region.get("structure", {}))
                if isinstance(node, dict) and isinstance(node.get("ref"), str)
            }
            if expected_refs and not expected_refs <= actual_refs:
                errors.append(f"missing entity creation obligation {sorted(expected_refs - actual_refs)}")

    absent_ids = {row.get("node_id") for row in ir.get("absent_entities", []) if isinstance(row, dict) and isinstance(row.get("node_id"), str)}
    for row in obligations["absent_entities"]:
        if row["node_id"] not in absent_ids:
            errors.append(f"missing absent entity obligation {row['node_id']}")

    regions_by_relation_type: dict[str, list[set[str]]] = {}
    for region in ir.get("control_flow_regions", []):
        if not (isinstance(region, dict) and isinstance(region.get("relation_type"), str) and isinstance(region.get("structure"), dict)):
            continue
        regions_by_relation_type.setdefault(region["relation_type"], []).append(
            {node["id"] for node in ir_nodes(region["structure"]) if isinstance(node, dict) and isinstance(node.get("id"), str)}
        )
    for row in obligations["structural_participants"]:
        relation_type = row["relation_type"]
        required_nodes = set(row["node_ids"])
        if not any(required_nodes <= actual_nodes for actual_nodes in regions_by_relation_type.get(relation_type, [])):
            errors.append(f"missing structural participants {(relation_type, tuple(row['node_ids']))}")
    return not errors, errors


def compile_ir(ir: dict[str, Any], _graph_data: dict[str, Any]) -> dict[str, Any]:
    entities: dict[str, dict[str, Any]] = {}
    properties: dict[str, dict[str, Any]] = {}

    def add_property(node_id: str, changes: dict[str, Any]) -> None:
        current = properties.get(node_id, {})
        current.update(normalize_changes(changes))
        properties[node_id] = current

    def structure_nodes(structure: dict[str, Any]) -> set[str]:
        if structure["type"] == "node":
            return {structure["id"]} if "id" in structure else set()
        children = structure.get("items" if structure["type"] == "sequence" else "branches", [])
        result: set[str] = set()
        for child in children:
            result.update(structure_nodes(child))
        return result

    def compile_structure(structure: dict[str, Any], relation_type: str) -> dict[str, Any]:
        kind = structure["type"]
        if kind == "node":
            if "ref" in structure:
                entities.setdefault(
                    structure["ref"],
                    {"ref": structure["ref"], "type": structure["node_type"], "label": structure["label"]},
                )
                node_key = structure["ref"]
            else:
                node_key = structure["id"]
            return {"entries": [node_key], "exits": [node_key], "edges": set(), "join_mode": None}
        if kind == "sequence":
            parts = [compile_structure(child, relation_type) for child in structure.get("items", [])]
            edges: set[tuple[str, str, str]] = set()
            for part in parts:
                edges |= part["edges"]
            for left, right in zip(parts, parts[1:]):
                for source in left["exits"]:
                    for target in right["entries"]:
                        edges.add((source, target, relation_type))
                if left.get("join_mode") and len(right["entries"]) == 1 and isinstance(right["entries"][0], str):
                    add_property(right["entries"][0], {"join": {"mode": left["join_mode"]}})
            return {"entries": parts[0]["entries"], "exits": parts[-1]["exits"], "edges": edges, "join_mode": None}
        branches = [compile_structure(child, relation_type) for child in structure.get("branches", [])]
        edges: set[tuple[str, str, str]] = set()
        entries: list[str] = []
        exits: list[str] = []
        for branch in branches:
            edges |= branch["edges"]
            entries.extend(branch["entries"])
            exits.extend(branch["exits"])
        return {"entries": entries, "exits": exits, "edges": edges, "join_mode": structure.get("join_mode")}

    control_flow_regions: list[dict[str, Any]] = []
    for region in ir.get("control_flow_regions", []):
        compiled = compile_structure(region["structure"], region["relation_type"])
        control_flow_regions.append(
            {
                "relation_type": region["relation_type"],
                "edges": sorted(compiled["edges"]),
                "node_ids": sorted(structure_nodes(region["structure"])),
                "entries": compiled["entries"],
                "exits": compiled["exits"],
            }
        )
    for row in ir.get("properties", []):
        add_property(row["node_id"], row["changes"])
    return {
        "entities": sorted(entities.values(), key=lambda row: row["ref"]),
        "control_flow_regions": control_flow_regions,
        "auxiliary_relations": list(ir.get("auxiliary_relations", [])),
        "absent_entities": list(ir.get("absent_entities", [])),
        "properties": [{"node_id": node_id, "changes": changes} for node_id, changes in sorted(properties.items()) if changes],
    }


def _structural_property_authorizations(final_ir: dict[str, Any]) -> list[dict[str, Any]]:
    authorized: list[dict[str, Any]] = []

    def compile_structure(structure: dict[str, Any]) -> dict[str, Any]:
        kind = structure["type"]
        if kind == "node":
            if "id" in structure:
                entry = {"kind": "id", "value": structure["id"]}
            else:
                entry = {"kind": "ref", "value": structure["ref"]}
            return {"entries": [entry], "exits": [entry], "join_mode": None}
        if kind == "sequence":
            parts = [compile_structure(child) for child in structure.get("items", [])]
            for left, right in zip(parts, parts[1:]):
                if left.get("join_mode") and len(right["entries"]) == 1 and right["entries"][0]["kind"] == "id":
                    authorized.append(
                        {
                            "node_id": right["entries"][0]["value"],
                            "property": "join",
                            "value": {"mode": left["join_mode"]},
                            "derivation": "structural_parallel_join",
                        }
                    )
            return {"entries": parts[0]["entries"], "exits": parts[-1]["exits"], "join_mode": None}
        branches = [compile_structure(child) for child in structure.get("branches", [])]
        entries: list[dict[str, str]] = []
        exits: list[dict[str, str]] = []
        for branch in branches:
            entries.extend(branch["entries"])
            exits.extend(branch["exits"])
        return {"entries": entries, "exits": exits, "join_mode": structure.get("join_mode")}

    for region in final_ir.get("control_flow_regions", []):
        if isinstance(region, dict) and isinstance(region.get("structure"), dict):
            compile_structure(region["structure"])
    return authorized


def validate_compiled_property_provenance(
    graph_data: dict[str, Any],
    final_ir: dict[str, Any],
    compiled: dict[str, Any],
    obligations: dict[str, Any],
) -> tuple[bool, list[str]]:
    existing_nodes = graph_node_ids(graph_data)
    explicit_authorized = {
        (
            row["node_id"],
            "join" if row.get("property") == "join" or row.get("kind") == "join_all" else row.get("property", row["kind"]),
            json.dumps(
                {"mode": "all"} if row.get("property") == "join" or row.get("kind") == "join_all" else row["value"],
                ensure_ascii=False,
                sort_keys=True,
            ),
        )
        for row in obligations.get("properties", [])
        if isinstance(row, dict) and isinstance(row.get("node_id"), str)
    }
    derived_authorized = {
        (
            row["node_id"],
            row["property"],
            json.dumps(row["value"], ensure_ascii=False, sort_keys=True),
        )
        for row in _structural_property_authorizations(final_ir)
        if isinstance(row.get("node_id"), str)
    }
    allowed = explicit_authorized | derived_authorized
    errors: list[str] = []
    for row in compiled.get("properties", []):
        if not (isinstance(row, dict) and isinstance(row.get("node_id"), str) and isinstance(row.get("changes"), dict)):
            continue
        if row["node_id"] not in existing_nodes:
            continue
        normalized_changes = normalize_changes(row["changes"])
        for property_name, value in normalized_changes.items():
            lookup_name = "join" if property_name == "join" else property_name
            lookup_value = {"mode": "all"} if property_name == "join" and value == {"mode": "all"} else value
            key = (row["node_id"], lookup_name, json.dumps(lookup_value, ensure_ascii=False, sort_keys=True))
            if key not in allowed:
                errors.append(
                    f"unauthorized property mutation {row['node_id']} {json.dumps({property_name: value}, ensure_ascii=False, sort_keys=True)}"
                )
    return not errors, errors


def graph_satisfies_obligations(graph_data: dict[str, Any], obligations: dict[str, Any]) -> bool:
    nodes = {
        node["id"]: node
        for node in graph_data.get("nodes", [])
        if isinstance(node, dict) and isinstance(node.get("id"), str)
    }
    edges = {
        (edge.get("source"), edge.get("target"), edge.get("label", ""))
        for edge in graph_data.get("edges", [])
        if isinstance(edge, dict)
    }
    for row in obligations.get("properties", []):
        node = nodes.get(row["node_id"])
        if not isinstance(node, dict):
            return False
        property_name = row.get("property", row.get("kind"))
        if property_name == "join" or row.get("kind") == "join_all":
            if normalize_changes({"join": node.get("join")}).get("join") != {"mode": "all"}:
                return False
            continue
        if node.get(property_name) != row["value"]:
            return False
    for row in obligations.get("auxiliary", []):
        if (row["source"], row["target"], row["relation_type"]) not in edges:
            return False
    for row in obligations.get("absent_entities", []):
        if row["node_id"] in nodes:
            return False
    if obligations.get("entity_specs") or obligations.get("structural_participants") or obligations.get("abstract_goals"):
        return False
    return True


def requirement_needs_clarification(requirement: dict[str, Any]) -> tuple[bool, str | None, str | None]:
    desired_state = requirement.get("grounded_desired_state", {}) if isinstance(requirement.get("grounded_desired_state"), dict) else {}
    clarification = desired_state.get("clarification", {}) if isinstance(desired_state.get("clarification"), dict) else {}
    if clarification.get("needed") is True:
        question = clarification.get("question") if isinstance(clarification.get("question"), str) else ""
        reason = clarification.get("reason") if isinstance(clarification.get("reason"), str) else "clarification required"
        return True, question or "Please clarify the unresolved edit target.", reason
    refs = requirement.get("references", {}) if isinstance(requirement.get("references"), dict) else {}
    ambiguous = sorted(
        {
            *refs.get("ambiguous_source_node_ids", []),
            *refs.get("ambiguous_target_node_ids", []),
            *refs.get("ambiguous_selected_node_ids", []),
            *refs.get("ambiguous_node_ids", []),
        }
    )
    if ambiguous:
        return True, f"Which node do you want to edit? Candidates: {', '.join(ambiguous)}.", "ambiguous node reference"
    abstract_goals = desired_state.get("abstract_goals", []) if isinstance(desired_state.get("abstract_goals"), list) else []
    if any(isinstance(row, dict) and row.get("goal_type") in {None, "", "other"} for row in abstract_goals):
        return (
            True,
            "What concrete success criterion should I use for this high-level goal?",
            "underspecified abstract goal",
        )
    return False, None, None


def selection_conflict(requirements: list[dict[str, Any]], selection: dict[str, Any] | None) -> tuple[bool, str | None]:
    selected = set(_selection_node_ids(selection))
    if not selected:
        return False, None
    selected_only = any(
        isinstance(req.get("grounded_desired_state"), dict)
        and isinstance(req["grounded_desired_state"].get("scope_policy"), dict)
        and req["grounded_desired_state"]["scope_policy"].get("selected_only") is True
        for req in requirements
    )
    if not selected_only:
        return False, None
    for req in requirements:
        refs = req.get("references", {}) if isinstance(req.get("references"), dict) else {}
        touched = {
            refs.get("source_node_id"),
            refs.get("target_node_id"),
            refs.get("selected_node_id"),
            *(refs.get("mentioned_node_ids", []) if isinstance(refs.get("mentioned_node_ids"), list) else []),
        }
        touched = {node_id for node_id in touched if isinstance(node_id, str)}
        if touched and not touched <= selected:
            return True, "The instruction conflicts with the selected-only scope. Should I also modify the unselected node(s)?"
    return False, None


def constraints_from_compiled(graph_data: dict[str, Any], compiled: dict[str, Any]) -> dict[str, Any]:
    constraints = empty_constraints()
    constraints["entities"] = list(compiled["entities"])
    constraints["absent_entities"] = list(compiled.get("absent_entities", []))
    nodes_by_id = {
        node["id"]: node
        for node in graph_data.get("nodes", [])
        if isinstance(node, dict) and isinstance(node.get("id"), str)
    }
    property_rows: list[dict[str, Any]] = []
    for row in compiled.get("properties", []):
        if not (isinstance(row, dict) and isinstance(row.get("node_id"), str) and isinstance(row.get("changes"), dict)):
            continue
        node = nodes_by_id.get(row["node_id"], {})
        changes = {
            key: value
            for key, value in normalize_changes(row["changes"]).items()
            if not isinstance(node, dict) or node.get(key) != value
        }
        if changes:
            property_rows.append({"node_id": row["node_id"], "changes": changes})
    constraints["properties"] = property_rows
    current_node_ids = graph_node_ids(graph_data)
    target_edges: set[tuple[str, str, str]] = set()
    target_edge_rows: dict[tuple[str, str, str], dict[str, Any]] = {}
    owned_current_edges: set[tuple[str, str, str]] = set()
    owned_components: dict[str, list[set[str]]] = {}
    for region in compiled.get("control_flow_regions", []):
        relation_type = region["relation_type"]
        region_nodes = set(region.get("node_ids", [])) | {
            endpoint
            for edge in region.get("edges", [])
            for endpoint in (edge[0], edge[1])
            if endpoint not in current_node_ids
        }
        components = owned_components.setdefault(relation_type, [])
        overlaps = [component for component in components if component & region_nodes]
        if overlaps:
            merged_nodes = set(region_nodes)
            for component in overlaps:
                merged_nodes.update(component)
                components.remove(component)
            components.append(merged_nodes)
        else:
            components.append(set(region_nodes))
        for edge in region.get("edges", []):
            target_edges.add(edge)
            target_edge_rows.setdefault(edge, {"source": edge[0], "target": edge[1], "label": edge[2]})
    for relation_type, components in owned_components.items():
        for component in components:
            for edge in graph_data.get("edges", []):
                relation = (edge["source"], edge["target"], edge.get("label", ""))
                if (
                    edge.get("label", "") == relation_type
                    and edge.get("source") in component
                    and edge.get("target") in component
                ):
                    owned_current_edges.add(relation)
    aux_by_group: dict[tuple[str, str], list[str]] = {}
    for row in compiled.get("auxiliary_relations", []):
        if isinstance(row, dict):
            aux_by_group[(row["source"], row["relation_type"])] = list(row["targets"])
    for (source, label), targets in aux_by_group.items():
        current_group_edges = [
            edge
            for edge in graph_data.get("edges", [])
            if isinstance(edge, dict) and edge.get("source") == source and edge.get("label", "") == label
        ]
        replaced_edges = [edge for edge in current_group_edges if edge.get("target") not in targets]
        for target in targets:
            relation = (source, target, label)
            target_edges.add(relation)
            row = target_edge_rows.setdefault(relation, {"source": source, "target": target, "label": label})
            if relation not in {
                (edge.get("source"), edge.get("target"), edge.get("label", ""))
                for edge in current_group_edges
            } and len(replaced_edges) == 1:
                for key, value in replaced_edges[0].items():
                    if key not in {"source", "target", "label"}:
                        row.setdefault(key, deepcopy(value))
    for relation in sorted(target_edges):
        source, target, label = relation
        constraints["required_relations"].append(target_edge_rows.get(relation, {"source": source, "target": target, "label": label}))

    absent_nodes = {row["node_id"] for row in compiled.get("absent_entities", []) if isinstance(row, dict) and isinstance(row.get("node_id"), str)}
    aux_current_edges = {
        (edge["source"], edge["target"], edge.get("label", ""))
        for edge in graph_data.get("edges", [])
        if (edge.get("source"), edge.get("label", "")) in aux_by_group
    }
    for edge in graph_data.get("edges", []):
        if edge.get("source") in absent_nodes or edge.get("target") in absent_nodes:
            owned_current_edges.add((edge["source"], edge["target"], edge.get("label", "")))
    for relation in sorted(owned_current_edges | aux_current_edges):
        if relation not in target_edges:
            constraints["forbidden_relations"].append({"source": relation[0], "target": relation[1], "label": relation[2]})
    constraints["preserve"] = {
        "node_ids": sorted(current_node_ids - absent_nodes),
        "edge_ids": [],
        "outside_specification": True,
    }
    return constraints


def merge_constraints(graph_data: dict[str, Any], unit_constraints: list[dict[str, Any]]) -> tuple[dict[str, Any] | None, str | None]:
    merged = empty_constraints()
    entity_rows: dict[str, dict[str, Any]] = {}
    property_rows: dict[str, dict[str, Any]] = {}
    absent_entities: set[str] = set()
    required: dict[tuple[str, str, str], dict[str, Any]] = {}
    forbidden: set[tuple[str, str, str]] = set()

    for constraints in unit_constraints:
        for row in constraints.get("entities", []):
            ref = row["ref"]
            if ref in entity_rows and entity_rows[ref] != row:
                return None, "conflicting entity refs across units"
            entity_rows[ref] = row
        for row in constraints.get("properties", []):
            current = property_rows.setdefault(row["node_id"], {})
            for key, value in row["changes"].items():
                if key in current and current[key] != value:
                    return None, "conflicting node properties across units"
                current[key] = value
        for row in constraints.get("absent_entities", []):
            if isinstance(row, dict) and isinstance(row.get("node_id"), str):
                absent_entities.add(row["node_id"])
        for row in constraints.get("required_relations", []):
            key = (row["source"], row["target"], row.get("label", ""))
            if key in required and required[key] != row:
                return None, "conflicting required relation metadata"
            required[key] = row
        for row in constraints.get("forbidden_relations", []):
            forbidden.add((row["source"], row["target"], row.get("label", "")))

    if set(required) & forbidden:
        return None, "conflicting required and forbidden relations"

    merged["entities"] = [entity_rows[key] for key in sorted(entity_rows)]
    merged["absent_entities"] = [{"node_id": node_id} for node_id in sorted(absent_entities)]
    merged["properties"] = [{"node_id": node_id, "changes": changes} for node_id, changes in sorted(property_rows.items()) if changes]
    merged["required_relations"] = [required[key] for key in sorted(required)]
    merged["forbidden_relations"] = [{"source": s, "target": t, "label": l} for s, t, l in sorted(forbidden)]
    merged["preserve"] = {
        "node_ids": sorted(node["id"] for node in graph_data.get("nodes", []) if node.get("id") not in absent_entities),
        "edge_ids": [],
        "outside_specification": True,
    }
    return merged, None
