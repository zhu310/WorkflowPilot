from __future__ import annotations

from copy import deepcopy
from typing import Any

from .canonical_goal_state import normalize_relation_fact
from .semantic_reference import (
    resolve_semantic_edge_reference,
    resolve_semantic_node_reference,
    resolve_strict_semantic_edge_reference,
    resolve_strict_semantic_node_reference,
    semantic_text_score,
)
from .workflow_ir_core import _resolve_node_reference


FORBIDDEN_SEMANTIC_GOAL_FIELDS = {
    "workflow_ir",
    "control_flow_regions",
    "auxiliary_relations",
    "properties",
    "edge_properties",
    "absent_entities",
    "required_relations",
    "forbidden_relations",
    "node_id",
    "edge_id",
    "x",
    "y",
    "steps",
    "mutation_steps",
    "planner_steps",
    "executor_steps",
    "graph_diff",
    "add_node",
    "remove_node",
    "add_edge",
    "remove_edge",
    "replace_edge",
    "rewire",
    "resolved_id",
    "candidate_id",
    "incoming_edges",
    "outgoing_edges",
    "owned_edges",
    "boundary",
    "branches",
    "parallel_nodes",
    "parallel_group",
}

SUPPORTED_GOAL_KINDS = {"layout", "ordering", "branch_property", "route", "connection", "object_state", "parallel"}
SUPPORTED_GOAL_CATEGORIES = {"object_state_goal", "relation_goal", "constraint_goal", "composite_goal"}
SUPPORTED_OBJECT_STATES = {"exists", "absent", "preserved", "property_equals"}
SUPPORTED_RELATION_INTENTS = {"before", "after", "right_of", "route_to", "parallel_with", "connected_to", "joins_before"}
SUPPORTED_LAYOUT_RELATIONS = {"right_of"}
SUPPORTED_ORDERING_RELATIONS = {"before", "after"}
SUPPORTED_BRANCH_PROPERTY_NAMES = {"label", "branch_label"}
SUPPORTED_ROUTE_RELATIONS = {"route_to"}
SUPPORTED_UNRESOLVED_DIMENSIONS = {"transformation", "reference", "parameter", "value", "constraint"}
DEICTIC_NODE_REFERENCES = {
    "$selected_node",
    "this node",
    "selected node",
    "current node",
    "this step",
    "selected step",
    "current step",
    "previous node",
    "last node",
    "recent node",
    "that node",
    "\u8fd9\u4e2a\u8282\u70b9",
    "\u5f53\u524d\u8282\u70b9",
    "\u9009\u4e2d\u7684\u8282\u70b9",
    "\u8be5\u8282\u70b9",
    "\u521a\u624d\u90a3\u4e2a\u8282\u70b9",
    "\u521a\u624d\u8bf4\u7684\u90a3\u4e2a\u8282\u70b9",
    "\u4e0a\u6b21\u90a3\u4e2a\u8282\u70b9",
    "\u8fd9\u4e2a\u6b65\u9aa4",
    "\u5f53\u524d\u6b65\u9aa4",
    "\u9009\u4e2d\u7684\u6b65\u9aa4",
    "\u8be5\u6b65\u9aa4",
    "\u521a\u624d\u90a3\u4e2a\u6b65\u9aa4",
    "\u521a\u624d\u8bf4\u7684\u90a3\u4e2a\u6b65\u9aa4",
    "\u4e0a\u6b21\u90a3\u4e2a\u6b65\u9aa4",
}
DEICTIC_EDGE_REFERENCES = {
    "$selected_edge",
    "this branch",
    "selected branch",
    "current branch",
    "this edge",
    "selected edge",
    "current edge",
    "这个分支",
    "当前分支",
    "该分支",
    "这条分支",
    "这个边",
    "当前边",
    "该边",
    "杩欎釜鍒嗘敮",
    "璇ュ垎鏀?",
}
RECENT_REFERENCE_TERMS = {
    "just",
    "recent",
    "recently",
    "previous",
    "last",
    "earlier",
    "newly",
    "previous edit",
    "last edit",
    "\u521a\u624d",
    "\u521a\u521a",
    "\u4e0a\u6b21",
    "\u4e0a\u4e00\u8f6e",
    "\u524d\u4e00\u8f6e",
    "\u4e4b\u524d",
    "\u6700\u8fd1",
    "\u4e0a\u4e00\u4e2a",
}
RECENT_ACTION_TERMS = {
    "created",
    "added",
    "newly added",
    "newly created",
    "modified",
    "changed",
    "updated",
    "\u65b0\u589e",
    "\u65b0\u5efa",
    "\u6dfb\u52a0",
    "\u521b\u5efa",
    "\u4fee\u6539",
    "\u6539\u8fc7",
    "\u53d8\u66f4",
}
REFERENCE_POINTER_TERMS = {
    "that",
    "that one",
    "the one",
    "\u90a3\u4e2a",
    "\u8fd9\u4e2a",
    "\u8be5",
    "\u4e0a\u4e00\u4e2a",
}
NODE_OBJECT_TERMS = {
    "node",
    "step",
    "object",
    "\u8282\u70b9",
    "\u6b65\u9aa4",
    "\u5bf9\u8c61",
}
EDGE_RELATION_ALIASES = {
    "success": {"success", "successful", "succeeded", "ok", "approved", "通过", "成功"},
    "failure": {"failure", "failed", "fail", "error", "rejected", "拒绝", "失败"},
    "retry": {"retry", "again", "重试"},
    "exception": {"exception", "异常"},
}


def _normalized_text(value: Any) -> str:
    return str(value or "").strip().lower()


def _phrase_in_source(phrase: Any, source_instruction: str | None) -> bool:
    if not isinstance(source_instruction, str) or not source_instruction.strip():
        return True
    needle = _normalized_text(phrase)
    haystack = _normalized_text(source_instruction)
    return bool(needle and needle in haystack)


def _canonical_kind(raw: Any) -> str:
    kind = str(raw or "").strip()
    return "ordering" if kind == "order" else kind


def _contains_forbidden_field(value: Any, path: str = "") -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    if isinstance(value, dict):
        for key, child in value.items():
            location = f"{path}.{key}" if path else key
            if key in FORBIDDEN_SEMANTIC_GOAL_FIELDS:
                issues.append(
                    {
                        "kind": "semantic_goal_contract_violation",
                        "message": f"{key} is a Workflow IR, graph binding, or mutation field and is not allowed in Semantic Goal",
                        "severity": "error",
                        "location": location,
                    }
                )
            issues.extend(_contains_forbidden_field(child, location))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            issues.extend(_contains_forbidden_field(child, f"{path}[{index}]"))
    return issues


def _normalize_object(item: Any, role: str | None = None) -> dict[str, Any] | None:
    if not isinstance(item, dict):
        return None
    object_type = str(item.get("type", "")).strip()
    item_role = str(role or item.get("role", "")).strip()
    if object_type == "unresolved_reference":
        if not item_role:
            return None
        result = {"type": "unresolved_reference", "role": item_role}
        if item.get("v2_object_id") is not None:
            result["v2_object_id"] = item.get("v2_object_id")
        return result
    if object_type == "edge_reference":
        reference = str(item.get("text", item.get("reference", ""))).strip()
        if not reference:
            return None
        result = {"type": "edge_reference", "reference": reference, "role": item_role or "branch"}
        if item.get("v2_object_id") is not None:
            result["v2_object_id"] = item.get("v2_object_id")
        return result
    if object_type == "new_entity":
        label = str(item.get("label", "")).strip()
        if not label or not item_role:
            return None
        result = {"type": "new_entity", "label": label, "role": item_role}
        if item.get("v2_object_id") is not None:
            result["v2_object_id"] = item.get("v2_object_id")
        return result
    # Backward-compatible first vertical slice shape:
    # {"reference": "...", "role": "..."} means existing_reference.
    reference = str(item.get("text", item.get("reference", ""))).strip()
    if not reference or not item_role:
        return None
    result = {"type": "existing_reference", "reference": reference, "role": item_role}
    if item.get("v2_object_id") is not None:
        result["v2_object_id"] = item.get("v2_object_id")
    return result


def _object_phrase(item: dict[str, Any]) -> str:
    return str(item.get("reference", item.get("label", item.get("text", ""))) or "").strip()


def _normalize_v2_object(item: Any, index: int) -> dict[str, Any] | None:
    if not isinstance(item, dict):
        return None
    object_type = str(item.get("type", "")).strip()
    role = str(item.get("role", "")).strip()
    object_id = str(item.get("id", item.get("object_id", ""))).strip() or f"o{index + 1}"
    if object_type == "unresolved_reference":
        return {"id": object_id, "type": "unresolved_reference", "role": role or "object"}
    if object_type == "edge_reference":
        reference = str(item.get("text", item.get("reference", ""))).strip()
        return {"id": object_id, "type": "edge_reference", "reference": reference, "role": role or "object"} if reference else None
    if object_type == "new_entity":
        label = str(item.get("label", item.get("text", ""))).strip()
        return {"id": object_id, "type": "new_entity", "label": label, "role": role or "object"} if label else None
    reference = str(item.get("text", item.get("reference", ""))).strip()
    return {"id": object_id, "type": "existing_reference", "reference": reference, "role": role or "object"} if reference else None


def _normalize_v2_state(row: Any) -> dict[str, Any] | None:
    if not isinstance(row, dict):
        return None
    object_ref = str(row.get("object", row.get("object_ref", row.get("target", "")))).strip()
    state = str(row.get("state", row.get("type", ""))).strip()
    if not object_ref or not state:
        return None
    normalized = {"object": object_ref, "state": state}
    if state == "property_equals":
        normalized["property"] = str(row.get("property", row.get("name", ""))).strip()
        value = row.get("value", "")
        normalized["value"] = value if isinstance(value, str) else ""
    return normalized


def _normalize_v2_relation(row: Any) -> dict[str, Any] | None:
    if not isinstance(row, dict):
        return None
    source = str(row.get("source", row.get("from", row.get("subject", "")))).strip()
    target = str(row.get("target", row.get("to", row.get("anchor", "")))).strip()
    relation = str(row.get("relation", row.get("type", ""))).strip()
    if not (source and target and relation):
        return None
    return {"source": source, "relation": relation, "target": target}


def _normalize_v2_relation_fact_member(row: Any) -> dict[str, Any] | None:
    if not isinstance(row, dict):
        return None
    source = str(row.get("source", row.get("source_ref", ""))).strip()
    target = str(row.get("target", row.get("target_ref", ""))).strip()
    if not (source and target):
        return None
    properties = row.get("properties")
    provenance_refs = row.get("provenance_refs")
    result: dict[str, Any] = {
        "source": source,
        "target": target,
        "properties": deepcopy(properties) if isinstance(properties, dict) else {},
    }
    if isinstance(provenance_refs, list):
        result["provenance_refs"] = [item for item in provenance_refs if isinstance(item, str) and item.strip()]
    return result


def _normalize_v2_relation_fact(row: Any) -> dict[str, Any] | None:
    if not isinstance(row, dict):
        return None
    kind = str(row.get("kind") or "").strip()
    members = row.get("members") if isinstance(row.get("members"), list) else []
    normalized_members = [
        normalized
        for item in members
        if (normalized := _normalize_v2_relation_fact_member(item)) is not None
    ]
    if not kind or not normalized_members:
        return None
    group_semantics = row.get("group_semantics")
    provenance_refs = row.get("provenance_refs")
    result: dict[str, Any] = {
        "kind": kind,
        "members": normalized_members,
        "group_semantics": deepcopy(group_semantics) if isinstance(group_semantics, dict) else {},
    }
    if isinstance(provenance_refs, list):
        result["provenance_refs"] = [item for item in provenance_refs if isinstance(item, str) and item.strip()]
    return result


def _normalize_v2_constraint(row: Any) -> dict[str, Any] | None:
    if isinstance(row, str):
        text = row.strip()
        return {"type": "preserve", "target": text, "text": text, "constraint_source": "legacy_explicit_constraint"} if text else None
    if not isinstance(row, dict):
        return None
    constraint_type = str(row.get("type", row.get("kind", ""))).strip()
    target = str(row.get("target", row.get("object", ""))).strip()
    text = str(row.get("text", row.get("description", ""))).strip()
    if not (constraint_type or target or text):
        return None
    result = {"type": constraint_type or "preserve", "target": target, "text": text or target}
    if isinstance(row.get("constraint_source"), str) and row["constraint_source"].strip():
        result["constraint_source"] = row["constraint_source"].strip()
    return result


def _normalize_v2_unresolved(row: Any) -> dict[str, Any] | None:
    if isinstance(row, str):
        dimension = row.strip()
        return {"dimension": dimension} if dimension else None
    if not isinstance(row, dict):
        return None
    dimension = str(row.get("dimension", row.get("type", ""))).strip()
    if not dimension:
        return None
    result = {"dimension": dimension}
    role = str(row.get("role", "")).strip()
    if role:
        result["role"] = role
    object_ref = str(row.get("object", row.get("object_ref", ""))).strip()
    if object_ref:
        result["object"] = object_ref
    text = str(row.get("text", row.get("description", ""))).strip()
    if text:
        result["text"] = text
    return result


def _normalize_v2_goal(row: Any, index: int) -> dict[str, Any] | None:
    if not isinstance(row, dict):
        return None
    if row.get("kind") and not any(
        key in row
        for key in (
            "goal_category",
            "object_desired_states",
            "object_states",
            "relation_intents",
            "relation_facts",
            "semantic_constraints",
            "unresolved_dimensions",
            "unresolved",
        )
    ):
        return None
    if not any(
        key in row
        for key in (
            "goal_category",
            "objects",
            "object_desired_states",
            "object_states",
            "relation_intents",
            "relation_facts",
            "semantic_constraints",
            "unresolved_dimensions",
            "unresolved",
        )
    ):
        return None
    raw_objects = row.get("objects", [])
    raw_states = row.get("object_desired_states", row.get("object_states", []))
    raw_relations = row.get("relation_intents", row.get("relations", []))
    raw_relation_facts = row.get("relation_facts", [])
    raw_constraints = row.get("semantic_constraints", row.get("explicit_constraints", row.get("constraints", [])))
    raw_unresolved = row.get("unresolved_dimensions", row.get("unresolved", []))
    if not isinstance(raw_objects, list):
        raw_objects = []
    if not isinstance(raw_states, list):
        raw_states = []
    if not isinstance(raw_relations, list):
        raw_relations = []
    if not isinstance(raw_relation_facts, list):
        raw_relation_facts = []
    if not isinstance(raw_constraints, list):
        raw_constraints = []
    if isinstance(raw_unresolved, dict):
        raw_unresolved = [raw_unresolved]
    elif isinstance(raw_unresolved, str):
        raw_unresolved = [raw_unresolved]
    elif not isinstance(raw_unresolved, list):
        raw_unresolved = []
    return {
        "id": str(row.get("id") or f"g{index + 1}"),
        "goal_category": str(row.get("goal_category", "")).strip(),
        "text": str(row.get("text", row.get("description", ""))).strip(),
        "objects": [
            normalized
            for object_index, item in enumerate(raw_objects)
            if (normalized := _normalize_v2_object(item, object_index)) is not None
        ],
        "object_desired_states": [
            normalized
            for item in raw_states
            if (normalized := _normalize_v2_state(item)) is not None
        ],
        "relation_intents": [
            normalized
            for item in raw_relations
            if (normalized := _normalize_v2_relation(item)) is not None
        ],
        "relation_facts": [
            normalized
            for item in raw_relation_facts
            if (normalized := _normalize_v2_relation_fact(item)) is not None
        ],
        "semantic_constraints": [
            normalized
            for item in raw_constraints
            if (normalized := _normalize_v2_constraint(item)) is not None
        ],
        "unresolved_dimensions": [
            normalized
            for item in raw_unresolved
            if (normalized := _normalize_v2_unresolved(item)) is not None
        ],
    }


def _partial_v2_goal_issues(goal: dict[str, Any], index: int) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    object_by_id = {
        item.get("id"): item
        for item in goal.get("objects", [])
        if isinstance(item, dict) and isinstance(item.get("id"), str)
    }
    for object_index, item in enumerate(goal.get("objects", [])):
        if isinstance(item, dict) and item.get("type") == "unresolved_reference":
            issues.append(
                {
                    "kind": "missing_semantic_role",
                    "message": "semantic role is unresolved",
                    "severity": "blocking",
                    "location": f"semantic_goals_v2[{index}].objects[{object_index}]",
                    "role": str(item.get("role") or "object"),
                }
            )
    for unresolved_index, unresolved in enumerate(goal.get("unresolved_dimensions", [])):
        if not isinstance(unresolved, dict):
            continue
        dimension = str(unresolved.get("dimension") or "").strip()
        issues.append(
            {
                "kind": "missing_semantic_dimension",
                "message": "semantic dimension is unresolved",
                "severity": "blocking",
                "location": f"semantic_goals_v2[{index}].unresolved_dimensions[{unresolved_index}]",
                "dimension": dimension or "unknown",
                "role": str(unresolved.get("role") or ""),
                "object": str(unresolved.get("object") or ""),
            }
        )
    for relation_index, relation in enumerate(goal.get("relation_intents", [])):
        source = object_by_id.get(relation.get("source"))
        target = object_by_id.get(relation.get("target"))
        if isinstance(source, dict) and source.get("type") == "unresolved_reference":
            issues.append(
                {
                    "kind": "missing_semantic_role",
                    "message": "relation source is unresolved",
                    "severity": "blocking",
                    "location": f"semantic_goals_v2[{index}].relation_intents[{relation_index}].source",
                    "role": str(source.get("role") or "source"),
                    "relation": relation.get("relation"),
                }
            )
        if isinstance(target, dict) and target.get("type") == "unresolved_reference":
            issues.append(
                {
                    "kind": "missing_semantic_role",
                    "message": "relation target is unresolved",
                    "severity": "blocking",
                    "location": f"semantic_goals_v2[{index}].relation_intents[{relation_index}].target",
                    "role": str(target.get("role") or "target"),
                    "relation": relation.get("relation"),
                }
            )
    for relation_fact_index, relation_fact in enumerate(goal.get("relation_facts", [])):
        if not isinstance(relation_fact, dict):
            continue
        for member_index, member in enumerate(relation_fact.get("members", []) if isinstance(relation_fact.get("members"), list) else []):
            source = object_by_id.get(member.get("source")) if isinstance(member, dict) else None
            target = object_by_id.get(member.get("target")) if isinstance(member, dict) else None
            if isinstance(source, dict) and source.get("type") == "unresolved_reference":
                issues.append(
                    {
                        "kind": "missing_semantic_role",
                        "message": "grouped relation source is unresolved",
                        "severity": "blocking",
                        "location": f"semantic_goals_v2[{index}].relation_facts[{relation_fact_index}].members[{member_index}].source",
                        "role": str(source.get("role") or "source"),
                        "relation_kind": relation_fact.get("kind"),
                    }
                )
            if isinstance(target, dict) and target.get("type") == "unresolved_reference":
                issues.append(
                    {
                        "kind": "missing_semantic_role",
                        "message": "grouped relation target is unresolved",
                        "severity": "blocking",
                        "location": f"semantic_goals_v2[{index}].relation_facts[{relation_fact_index}].members[{member_index}].target",
                        "role": str(target.get("role") or "target"),
                        "relation_kind": relation_fact.get("kind"),
                    }
                )
    return issues


def _v2_object_lookup(objects: list[dict[str, Any]], ref: Any) -> dict[str, Any] | None:
    needle = str(ref or "").strip()
    if not needle:
        return None
    for item in objects:
        values = {str(item.get("id", "")).strip(), str(item.get("reference", "")).strip(), str(item.get("label", "")).strip()}
        if needle in values:
            return item
    return None


def _v2_object_as_v1(item: dict[str, Any], role: str) -> dict[str, Any]:
    if item.get("type") == "unresolved_reference":
        return {"type": "unresolved_reference", "role": role, "v2_object_id": item.get("id")}
    if item.get("type") == "new_entity":
        return {"type": "new_entity", "label": str(item.get("label", "")).strip(), "role": role, "v2_object_id": item.get("id")}
    if item.get("type") == "edge_reference":
        return {"type": "edge_reference", "reference": _object_phrase(item), "role": role, "v2_object_id": item.get("id")}
    return {"type": "existing_reference", "reference": _object_phrase(item), "role": role, "v2_object_id": item.get("id")}


def _v2_constraint_texts(goal: dict[str, Any]) -> list[str]:
    rows = goal.get("semantic_constraints", [])
    if not isinstance(rows, list):
        return []
    result = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        text = " ".join(
            item
            for item in [
                str(row.get("type", "")).strip(),
                str(row.get("target", "")).strip(),
                str(row.get("text", "")).strip(),
            ]
            if item
        )
        if text:
            result.append(text)
    return result


def _v2_goal_to_v1_goals(goal: dict[str, Any]) -> list[dict[str, Any]]:
    objects = goal.get("objects", []) if isinstance(goal.get("objects"), list) else []
    states = goal.get("object_desired_states", []) if isinstance(goal.get("object_desired_states"), list) else []
    relations = goal.get("relation_intents", []) if isinstance(goal.get("relation_intents"), list) else []
    constraints = _v2_constraint_texts(goal)
    generated: list[dict[str, Any]] = []
    for relation_index, relation in enumerate(relations):
        source = _v2_object_lookup(objects, relation.get("source"))
        target = _v2_object_lookup(objects, relation.get("target"))
        relation_type = relation.get("relation")
        if not (source and target and isinstance(relation_type, str)):
            continue
        if source.get("type") == "unresolved_reference" or target.get("type") == "unresolved_reference":
            continue
        source_subgoal_id = f"{goal.get('id', 'g')}:relation:{relation_index}"
        if relation_type == "right_of":
            generated.append(
                {
                    "id": f"{goal.get('id', 'g')}_layout_{len(generated) + 1}",
                    "source_subgoal_id": source_subgoal_id,
                    "kind": "layout",
                    "text": goal.get("text", ""),
                    "objects": [_v2_object_as_v1(source, "target"), _v2_object_as_v1(target, "anchor")],
                    "relation": "right_of",
                    "explicit_constraints": constraints,
                }
            )
        elif relation_type in {"before", "after", "joins_before"}:
            executable_relation = "before" if relation_type == "joins_before" else relation_type
            generated.append(
                {
                    "id": f"{goal.get('id', 'g')}_ordering_{len(generated) + 1}",
                    "source_subgoal_id": source_subgoal_id,
                    "kind": "ordering",
                    "text": goal.get("text", ""),
                    "objects": [_v2_object_as_v1(source, "subject"), _v2_object_as_v1(target, "anchor")],
                    "relation": executable_relation,
                    "explicit_constraints": constraints,
                }
            )
        elif relation_type == "route_to":
            generated.append(
                {
                    "id": f"{goal.get('id', 'g')}_route_{len(generated) + 1}",
                    "source_subgoal_id": source_subgoal_id,
                    "kind": "route",
                    "text": goal.get("text", ""),
                    "objects": [_v2_object_as_v1(source, "route"), _v2_object_as_v1(target, "target")],
                    "relation": "route_to",
                    "explicit_constraints": constraints,
                }
            )
        elif relation_type == "connected_to":
            generated.append(
                {
                    "id": f"{goal.get('id', 'g')}_connection_{len(generated) + 1}",
                    "source_subgoal_id": source_subgoal_id,
                    "kind": "connection",
                    "text": goal.get("text", ""),
                    "objects": [_v2_object_as_v1(source, "source"), _v2_object_as_v1(target, "target")],
                    "relation": "connected_to",
                    "explicit_constraints": constraints,
                }
            )
        elif relation_type == "parallel_with":
            generated.append(
                {
                    "id": f"{goal.get('id', 'g')}_parallel_{len(generated) + 1}",
                    "source_subgoal_id": source_subgoal_id,
                    "kind": "parallel",
                    "text": goal.get("text", ""),
                    "objects": [_v2_object_as_v1(source, "left"), _v2_object_as_v1(target, "right")],
                    "relation": "parallel_with",
                    "explicit_constraints": constraints,
                }
            )
    for state_index, state in enumerate(states):
        if not isinstance(state, dict):
            continue
        item = _v2_object_lookup(objects, state.get("object"))
        if not item:
            continue
        if item.get("type") == "unresolved_reference":
            continue
        state_name = state.get("state")
        if state_name in {"exists", "absent", "property_equals", "preserved"}:
            generated.append(
                {
                    "id": f"{goal.get('id', 'g')}_object_state_{len(generated) + 1}",
                    "source_subgoal_id": f"{goal.get('id', 'g')}:state:{state_index}",
                    "kind": "object_state",
                    "text": goal.get("text", ""),
                    "objects": [_v2_object_as_v1(item, "object")],
                    "relation": "",
                    "object_state": deepcopy(state),
                    "explicit_constraints": constraints,
                }
            )
            continue
        if state_name != "property_equals":
            continue
        prop = str(state.get("property", "")).strip()
        if item.get("type") != "edge_reference" or prop not in {"display_label", "label", "branch_label"}:
            continue
        generated.append(
            {
                "id": f"{goal.get('id', 'g')}_branch_property_{len(generated) + 1}",
                "source_subgoal_id": f"{goal.get('id', 'g')}:state:{state_index}",
                "kind": "branch_property",
                "text": goal.get("text", ""),
                "objects": [_v2_object_as_v1(item, "branch")],
                "relation": "",
                "property": {"name": "label", "value": state.get("value", "")},
                "explicit_constraints": constraints,
            }
        )
    return generated


def _v1_goal_to_v2_goal(goal: dict[str, Any], index: int) -> dict[str, Any]:
    objects = [
        {**item, "id": str(item.get("role") or f"o{object_index + 1}")}
        for object_index, item in enumerate(goal.get("objects", []))
        if isinstance(item, dict)
    ]
    states: list[dict[str, Any]] = []
    relations: list[dict[str, Any]] = []
    category = "composite_goal"
    if goal.get("kind") == "layout":
        category = "relation_goal"
        target = next((item for item in objects if item.get("role") == "target"), None)
        anchor = next((item for item in objects if item.get("role") == "anchor"), None)
        if target and anchor:
            relations.append({"source": target["id"], "relation": goal.get("relation", ""), "target": anchor["id"]})
    elif goal.get("kind") == "ordering":
        subject = next((item for item in objects if item.get("role") == "subject"), None)
        anchor = next((item for item in objects if item.get("role") == "anchor"), None)
        if subject and subject.get("type") == "new_entity":
            states.append({"object": subject["id"], "state": "exists"})
        if subject and anchor:
            relations.append({"source": subject["id"], "relation": goal.get("relation", ""), "target": anchor["id"]})
    elif goal.get("kind") == "branch_property":
        category = "object_state_goal"
        branch = next((item for item in objects if item.get("role") == "branch"), None)
        prop = goal.get("property") if isinstance(goal.get("property"), dict) else {}
        if branch:
            states.append({"object": branch["id"], "state": "property_equals", "property": "display_label", "value": prop.get("value", "")})
    elif goal.get("kind") == "route":
        category = "relation_goal"
        route = next((item for item in objects if item.get("role") == "route"), None)
        target = next((item for item in objects if item.get("role") == "target"), None)
        if route and target:
            relations.append({"source": route["id"], "relation": goal.get("relation", ""), "target": target["id"]})
    return {
        "id": str(goal.get("id") or f"g{index + 1}"),
        "goal_category": category,
        "text": goal.get("text", ""),
        "objects": objects,
        "object_desired_states": states,
        "relation_intents": relations,
        "semantic_constraints": [
            {"type": "preserve", "target": str(item).strip(), "text": str(item).strip(), "constraint_source": "legacy_explicit_constraint"}
            for item in goal.get("explicit_constraints", [])
            if isinstance(item, str) and item.strip()
        ],
    }


def _normalize_goal(row: Any, index: int) -> dict[str, Any] | None:
    if not isinstance(row, dict):
        return None
    kind = _canonical_kind(row.get("kind"))
    text = str(row.get("text", row.get("description", ""))).strip()
    relation = str(row.get("relation", "")).strip()
    value = row.get("value", "")
    objects = row.get("objects", [])
    explicit_constraints = row.get("explicit_constraints", row.get("constraints", []))
    property_value = row.get("property")
    if not isinstance(objects, list):
        objects = []
    normalized_objects = []
    for object_index, item in enumerate(objects):
        normalized_object = _normalize_object(item)
        if normalized_object is not None:
            normalized_objects.append(normalized_object)
    subject = _normalize_object(row.get("subject"), "subject")
    anchor = _normalize_object(row.get("anchor"), "anchor")
    branch_object = _normalize_object(row.get("object"), "branch")
    if subject is not None:
        normalized_objects.append(subject)
    if anchor is not None:
        normalized_objects.append(anchor)
    if branch_object is not None:
        normalized_objects.append(branch_object)
    if not isinstance(explicit_constraints, list):
        explicit_constraints = []
    normalized_property = {}
    if isinstance(property_value, dict):
        name = str(property_value.get("name", "")).strip()
        prop_target_value = property_value.get("value", "")
        normalized_property = {"name": name, "value": prop_target_value if isinstance(prop_target_value, str) else ""}
    elif any(key in row for key in ("property_name", "property_value")):
        name = str(row.get("property_name", "")).strip()
        prop_target_value = row.get("property_value", "")
        normalized_property = {"name": name, "value": prop_target_value if isinstance(prop_target_value, str) else ""}
    if not kind:
        return None
    normalized_goal = {
        "id": str(row.get("id") or f"g{index + 1}"),
        "kind": kind,
        "text": text,
        "objects": normalized_objects,
        "relation": relation,
        "value": value if isinstance(value, str) else "",
        "explicit_constraints": [
            str(item).strip()
            for item in explicit_constraints
            if isinstance(item, str) and str(item).strip()
        ],
    }
    if isinstance(row.get("source_subgoal_id"), str):
        normalized_goal["source_subgoal_id"] = row["source_subgoal_id"]
    if kind == "object_state" and isinstance(row.get("object_state"), dict):
        normalized_goal["object_state"] = deepcopy(row["object_state"])
    if normalized_property:
        normalized_goal["property"] = normalized_property
    return normalized_goal


def normalize_semantic_goal_spec(value: dict[str, Any] | None) -> dict[str, Any]:
    value = value if isinstance(value, dict) else {}
    raw_goals = value.get("semantic_goals")
    if not isinstance(raw_goals, list):
        raw_goals = value.get("goals")
    if not isinstance(raw_goals, list):
        raw_goals = [value] if value.get("kind") or value.get("goal_category") else []
    raw_v2_goals = value.get("semantic_goals_v2")
    if not isinstance(raw_v2_goals, list):
        raw_v2_goals = None
    v2_goals = [
        goal
        for index, row in enumerate(raw_v2_goals if raw_v2_goals is not None else raw_goals)
        if (goal := _normalize_v2_goal(row, index)) is not None
    ]
    goals = [
        goal
        for index, row in enumerate(raw_goals)
        if (goal := _normalize_goal(row, index)) is not None
    ]
    if v2_goals and not goals:
        converted_goals = []
        for goal in v2_goals:
            converted_goals.extend(_v2_goal_to_v1_goals(goal))
        goals = converted_goals
    elif goals and not v2_goals:
        v2_goals = [_v1_goal_to_v2_goal(goal, index) for index, goal in enumerate(goals)]
    clarification = value.get("clarification") if isinstance(value.get("clarification"), dict) else {}
    status = value.get("status") if value.get("status") in {"resolved", "partial", "clarification"} else "resolved"
    return {
        "status": status,
        "semantic_goals": goals,
        "semantic_goals_v2": v2_goals,
        "clarification": {
            "needed": bool(clarification.get("needed")),
            "question": str(clarification.get("question", "")),
            "reason": str(clarification.get("reason", "")),
        },
    }


def validate_semantic_goal_spec(value: dict[str, Any] | None, source_instruction: str | None = None) -> dict[str, Any]:
    issues = _contains_forbidden_field(value)
    normalized = normalize_semantic_goal_spec(value)
    clarification_requested = normalized["clarification"]["needed"] is True
    v2_goals = normalized.get("semantic_goals_v2", []) if isinstance(normalized.get("semantic_goals_v2"), list) else []
    if clarification_requested:
        issues.append(
            {
                "kind": "clarification_needed",
                "message": str(normalized["clarification"].get("reason") or "clarification needed"),
                "severity": "blocking",
                "location": "clarification",
            }
        )
    if not normalized["semantic_goals"] and not v2_goals and not normalized["clarification"]["needed"]:
        issues.append(
            {
                "kind": "missing_semantic_goal",
                "message": "semantic goal spec must contain at least one semantic goal or a clarification",
                "severity": "error",
                "location": "semantic_goals",
            }
        )
    for index, goal in enumerate(v2_goals):
        category = str(goal.get("goal_category") or "").strip()
        unresolved_dimensions = goal.get("unresolved_dimensions") if isinstance(goal.get("unresolved_dimensions"), list) else []
        if not category and not clarification_requested:
            issues.append(
                {
                    "kind": "missing_goal_category",
                    "message": "complete semantic goals must commit a goal category",
                    "severity": "blocking",
                    "location": f"semantic_goals_v2[{index}].goal_category",
                }
            )
        elif category and category not in SUPPORTED_GOAL_CATEGORIES and not clarification_requested:
            issues.append(
                {
                    "kind": "unsupported_goal_category",
                    "message": f"unsupported semantic goal category: {goal.get('goal_category')}",
                    "severity": "blocking",
                    "location": f"semantic_goals_v2[{index}].goal_category",
                }
            )
        for unresolved_index, unresolved in enumerate(unresolved_dimensions):
            dimension = str(unresolved.get("dimension") or "").strip() if isinstance(unresolved, dict) else ""
            if dimension not in SUPPORTED_UNRESOLVED_DIMENSIONS:
                issues.append(
                    {
                        "kind": "unsupported_unresolved_dimension",
                        "message": f"unsupported unresolved semantic dimension: {dimension}",
                        "severity": "blocking",
                        "location": f"semantic_goals_v2[{index}].unresolved_dimensions[{unresolved_index}].dimension",
                    }
                )
        issues.extend(_partial_v2_goal_issues(goal, index))
        if clarification_requested:
            continue
        object_ids = {item.get("id") for item in goal.get("objects", []) if isinstance(item, dict)}
        for object_index, item in enumerate(goal.get("objects", [])):
            phrase = item.get("label") if item.get("type") == "new_entity" else item.get("reference")
            if phrase and not _phrase_in_source(phrase, source_instruction):
                issues.append(
                    {
                        "kind": "semantic_goal_contract_violation",
                        "message": "object references must preserve phrases from the user instruction; Semantic Understanding must not pre-resolve graph objects",
                        "severity": "blocking",
                        "location": f"semantic_goals_v2[{index}].objects[{object_index}]",
                    }
                )
        for state_index, state in enumerate(goal.get("object_desired_states", [])):
            if state.get("state") not in SUPPORTED_OBJECT_STATES:
                issues.append(
                    {
                        "kind": "unsupported_object_desired_state",
                        "message": f"unsupported object desired state: {state.get('state')}",
                        "severity": "blocking",
                        "location": f"semantic_goals_v2[{index}].object_desired_states[{state_index}].state",
                    }
                )
            if state.get("object") not in object_ids:
                issues.append(
                    {
                        "kind": "unknown_state_object",
                        "message": f"object desired state references an unknown object: {state.get('object')}",
                        "severity": "blocking",
                        "location": f"semantic_goals_v2[{index}].object_desired_states[{state_index}].object",
                    }
                )
        for relation_index, relation in enumerate(goal.get("relation_intents", [])):
            if relation.get("relation") not in SUPPORTED_RELATION_INTENTS:
                issues.append(
                    {
                        "kind": "unsupported_relation_intent",
                        "message": f"unsupported relation intent: {relation.get('relation')}",
                        "severity": "blocking",
                        "location": f"semantic_goals_v2[{index}].relation_intents[{relation_index}].relation",
                    }
                )
            if relation.get("source") not in object_ids or relation.get("target") not in object_ids:
                issues.append(
                    {
                        "kind": "unknown_relation_object",
                        "message": "relation intent source and target must reference Semantic Goal objects",
                        "severity": "blocking",
                        "location": f"semantic_goals_v2[{index}].relation_intents[{relation_index}]",
                    }
                )
        for relation_fact_index, relation_fact in enumerate(goal.get("relation_facts", [])):
            if relation_fact.get("kind") not in {"connection", "precedence"}:
                issues.append(
                    {
                        "kind": "unsupported_grouped_relation_kind",
                        "message": f"unsupported grouped relation kind: {relation_fact.get('kind')}",
                        "severity": "blocking",
                        "location": f"semantic_goals_v2[{index}].relation_facts[{relation_fact_index}].kind",
                    }
                )
            for member_index, member in enumerate(relation_fact.get("members", [])):
                if member.get("source") not in object_ids or member.get("target") not in object_ids:
                    issues.append(
                        {
                            "kind": "unknown_grouped_relation_object",
                            "message": "grouped relation member source and target must reference Semantic Goal objects",
                            "severity": "blocking",
                            "location": f"semantic_goals_v2[{index}].relation_facts[{relation_fact_index}].members[{member_index}]",
                        }
                    )
                if not isinstance(member.get("properties"), dict):
                    issues.append(
                        {
                            "kind": "invalid_grouped_relation_member_properties",
                            "message": "grouped relation member properties must be an object",
                            "severity": "blocking",
                            "location": f"semantic_goals_v2[{index}].relation_facts[{relation_fact_index}].members[{member_index}].properties",
                        }
                    )
    for index, goal in enumerate(normalized["semantic_goals"]):
        unsupported_kind = goal["kind"] not in SUPPORTED_GOAL_KINDS
        if unsupported_kind and not clarification_requested:
            issues.append(
                {
                    "kind": "unsupported_semantic_goal_kind",
                    "message": f"unsupported semantic goal kind: {goal['kind']}",
                    "severity": "blocking",
                    "location": f"semantic_goals[{index}].kind",
                }
            )
        if clarification_requested:
            continue
        if goal["kind"] == "layout" and goal["relation"] not in SUPPORTED_LAYOUT_RELATIONS:
            issues.append(
                {
                    "kind": "unsupported_layout_relation",
                    "message": f"unsupported layout relation: {goal['relation']}",
                    "severity": "blocking",
                    "location": f"semantic_goals[{index}].relation",
                }
            )
        if goal["kind"] == "ordering" and goal["relation"] not in SUPPORTED_ORDERING_RELATIONS:
            issues.append(
                {
                    "kind": "unsupported_ordering_relation",
                    "message": f"unsupported ordering relation: {goal['relation']}",
                    "severity": "blocking",
                    "location": f"semantic_goals[{index}].relation",
                }
            )
        if goal["kind"] == "route" and goal["relation"] not in SUPPORTED_ROUTE_RELATIONS:
            issues.append(
                {
                    "kind": "unsupported_route_relation",
                    "message": f"unsupported route relation: {goal['relation']}",
                    "severity": "blocking",
                    "location": f"semantic_goals[{index}].relation",
                }
            )
        if goal["kind"] == "connection" and goal["relation"] != "connected_to":
            issues.append(
                {
                    "kind": "unsupported_connection_relation",
                    "message": f"unsupported connection relation: {goal['relation']}",
                    "severity": "blocking",
                    "location": f"semantic_goals[{index}].relation",
                }
            )
        if goal["kind"] == "parallel" and goal["relation"] != "parallel_with":
            issues.append(
                {
                    "kind": "unsupported_parallel_relation",
                    "message": f"unsupported parallel relation: {goal['relation']}",
                    "severity": "blocking",
                    "location": f"semantic_goals[{index}].relation",
                }
            )
        if goal["kind"] == "branch_property":
            prop = goal.get("property") if isinstance(goal.get("property"), dict) else {}
            if prop.get("name") not in SUPPORTED_BRANCH_PROPERTY_NAMES:
                issues.append(
                    {
                        "kind": "unsupported_branch_property",
                        "message": f"unsupported branch property: {prop.get('name')}",
                        "severity": "blocking",
                        "location": f"semantic_goals[{index}].property.name",
                    }
                )
            if not isinstance(prop.get("value"), str) or not prop.get("value").strip():
                issues.append(
                    {
                        "kind": "invalid_branch_property_value",
                        "message": "branch property value must be a non-empty string",
                        "severity": "error",
                        "location": f"semantic_goals[{index}].property.value",
                    }
                )
        for object_index, item in enumerate(goal.get("objects", [])):
            if item.get("type") == "existing_reference" and not _phrase_in_source(item.get("reference"), source_instruction):
                issues.append(
                    {
                        "kind": "semantic_goal_contract_violation",
                        "message": "existing_reference must preserve an object phrase from the user instruction; Semantic Understanding must not pre-resolve graph objects",
                        "severity": "blocking",
                        "location": f"semantic_goals[{index}].objects[{object_index}].reference",
                    }
                )
            if item.get("type") == "edge_reference" and not _phrase_in_source(item.get("reference"), source_instruction):
                issues.append(
                    {
                        "kind": "semantic_goal_contract_violation",
                        "message": "edge_reference must preserve an object phrase from the user instruction; Semantic Understanding must not pre-resolve graph objects",
                        "severity": "blocking",
                        "location": f"semantic_goals[{index}].objects[{object_index}].reference",
                    }
                )
            if item.get("type") == "new_entity" and not _phrase_in_source(item.get("label"), source_instruction):
                issues.append(
                    {
                        "kind": "semantic_goal_contract_violation",
                        "message": "new_entity label must preserve the requested business step phrase from the user instruction",
                        "severity": "blocking",
                        "location": f"semantic_goals[{index}].objects[{object_index}].label",
                    }
                )
            if item.get("type") == "new_entity" and not item.get("label"):
                issues.append(
                    {
                        "kind": "invalid_new_entity",
                        "message": "new_entity requires a non-empty label",
                        "severity": "error",
                        "location": f"semantic_goals[{index}].objects[{object_index}]",
                    }
                )
            if item.get("type") == "existing_reference" and not item.get("reference"):
                issues.append(
                    {
                        "kind": "invalid_existing_reference",
                        "message": "existing_reference requires non-empty text/reference",
                        "severity": "error",
                        "location": f"semantic_goals[{index}].objects[{object_index}]",
                    }
                )
            if item.get("type") == "edge_reference" and not item.get("reference"):
                issues.append(
                    {
                        "kind": "invalid_edge_reference",
                        "message": "edge_reference requires non-empty text/reference",
                        "severity": "error",
                        "location": f"semantic_goals[{index}].objects[{object_index}]",
                    }
                )
    hard_error = any(issue["severity"] == "error" for issue in issues)
    blocker = any(issue["severity"] in {"error", "blocking"} for issue in issues)
    return {"valid": not hard_error, "ready_for_grounding": not blocker, "issues": issues, "normalized": normalized}


def _add_issue(issues: list[dict[str, Any]], kind: str, location: str, reference: Any, candidates: list[str], reason: str) -> None:
    issues.append(
        {
            "kind": kind,
            "location": location,
            "reference": reference,
            "candidates": sorted(candidates),
            "reason": reason,
        }
    )


def _add_provenance(provenance: list[dict[str, Any]], location: str, reference: Any, resolved_id: str, resolution_type: str) -> None:
    provenance.append(
        {
            "location": location,
            "reference": reference,
            "resolved_id": resolved_id,
            "resolution_type": resolution_type,
        }
    )


def _node_flow_facts(graph_data: dict[str, Any], node_id: str) -> dict[str, Any]:
    incoming = [
        edge
        for edge in graph_data.get("edges", [])
        if isinstance(edge, dict) and edge.get("target") == node_id and isinstance(edge.get("source"), str)
    ]
    outgoing = [
        edge
        for edge in graph_data.get("edges", [])
        if isinstance(edge, dict) and edge.get("source") == node_id and isinstance(edge.get("target"), str)
    ]
    node = next(
        (
            row
            for row in graph_data.get("nodes", [])
            if isinstance(row, dict) and row.get("id") == node_id
        ),
        {},
    )
    has_join_property = isinstance(node, dict) and isinstance(node.get("join"), dict)
    return {
        "incoming_count": len(incoming),
        "outgoing_count": len(outgoing),
        "incoming_sources": sorted({edge["source"] for edge in incoming}),
        "outgoing_targets": sorted({edge["target"] for edge in outgoing}),
        "has_join_property": has_join_property,
        "is_fan_in": len({edge["source"] for edge in incoming}) > 1,
        "is_join_anchor": has_join_property or len({edge["source"] for edge in incoming}) > 1,
    }


def _selection_ids(selection: dict[str, Any] | None, canonical_key: str, context_key: str) -> list[str]:
    if not isinstance(selection, dict):
        return []
    values: list[str] = []
    for key in (canonical_key, context_key):
        raw_value = selection.get(key)
        if isinstance(raw_value, list):
            values.extend(item for item in raw_value if isinstance(item, str))
    return sorted(set(values))


def _selected_node_ids(selection: dict[str, Any] | None) -> list[str]:
    return _selection_ids(selection, "node_ids", "selected_node_ids")


def _selected_edge_ids(selection: dict[str, Any] | None) -> list[str]:
    return _selection_ids(selection, "edge_ids", "selected_edge_ids")


def _candidate_binding(
    request_context: dict[str, Any] | None,
    location: str,
    reference: Any,
    binding_type: str,
) -> str | None:
    if not isinstance(request_context, dict):
        return None
    bindings = request_context.get("candidate_bindings")
    if not isinstance(bindings, dict):
        return None
    by_location = bindings.get("by_location")
    if isinstance(by_location, dict) and isinstance(by_location.get(location), str):
        return by_location[location]
    by_reference = bindings.get("by_reference")
    if isinstance(by_reference, dict) and isinstance(reference, str) and isinstance(by_reference.get(reference), str):
        return by_reference[reference]
    typed_key = "edges" if binding_type == "edge" else "nodes"
    typed = bindings.get(typed_key)
    if isinstance(typed, dict) and isinstance(reference, str) and isinstance(typed.get(reference), str):
        return typed[reference]
    return None


def _graph_has_node(graph_data: dict[str, Any], node_id: str) -> bool:
    return any(isinstance(node, dict) and node.get("id") == node_id for node in graph_data.get("nodes", []))


def _graph_has_edge(graph_data: dict[str, Any], edge_id: str) -> bool:
    return any(isinstance(edge, dict) and edge.get("id") == edge_id for edge in graph_data.get("edges", []))


def _is_deictic_node_reference(raw: Any) -> bool:
    if not isinstance(raw, str) or not raw.strip():
        return False
    return raw.strip() in DEICTIC_NODE_REFERENCES or raw.strip().lower() in DEICTIC_NODE_REFERENCES


def _contains_reference_term(text: str, compact_text: str, terms: set[str]) -> bool:
    for term in terms:
        lowered = term.lower()
        compact = lowered.replace(" ", "")
        if lowered in text or compact in compact_text:
            return True
    return False


def _is_recent_transaction_node_reference(raw: Any) -> bool:
    if not isinstance(raw, str) or not raw.strip():
        return False
    lowered = raw.strip().lower()
    compact = lowered.replace(" ", "")
    has_node_object = _contains_reference_term(lowered, compact, NODE_OBJECT_TERMS)
    if not has_node_object:
        return False
    has_recent = _contains_reference_term(lowered, compact, RECENT_REFERENCE_TERMS)
    if has_recent:
        return True
    has_pointer = _contains_reference_term(lowered, compact, REFERENCE_POINTER_TERMS)
    has_recent_action = _contains_reference_term(lowered, compact, RECENT_ACTION_TERMS)
    return has_pointer and has_recent_action


def _node_ids_from_summary(rows: Any, known_node_ids: set[str]) -> list[str]:
    if not isinstance(rows, list):
        return []
    result: list[str] = []
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("id"), str):
            continue
        if row["id"] in known_node_ids:
            result.append(row["id"])
    return sorted(set(result))


def _node_ids_from_values(values: Any, known_node_ids: set[str]) -> list[str]:
    if not isinstance(values, list):
        return []
    return sorted({item for item in values if isinstance(item, str) and item in known_node_ids})


def _recent_transaction_node_candidates(
    graph_data: dict[str, Any],
    request_context: dict[str, Any] | None,
) -> tuple[list[str], str | None]:
    context = request_context if isinstance(request_context, dict) else {}
    diff = context.get("recent_transaction_diff")
    if not isinstance(diff, dict):
        return [], None
    known_node_ids = {
        node["id"]
        for node in graph_data.get("nodes", [])
        if isinstance(node, dict) and isinstance(node.get("id"), str)
    }
    for key in ("created_nodes", "changed_nodes"):
        candidates = _node_ids_from_summary(diff.get(key), known_node_ids)
        if candidates:
            return candidates, str(diff.get("transaction_id", "")) or None
    candidates = _node_ids_from_values(diff.get("affected_nodes"), known_node_ids)
    return candidates, str(diff.get("transaction_id", "")) or None


def _conversation_history_node_candidates(
    graph_data: dict[str, Any],
    request_context: dict[str, Any] | None,
) -> list[str]:
    context = request_context if isinstance(request_context, dict) else {}
    history = context.get("conversation_history")
    if not isinstance(history, list):
        return []
    text_parts: list[str] = []
    for row in history:
        if isinstance(row, dict):
            content = row.get("content", row.get("text", ""))
            if isinstance(content, str):
                text_parts.append(content.lower())
        elif isinstance(row, str):
            text_parts.append(row.lower())
    history_text = "\n".join(text_parts)
    if not history_text:
        return []
    candidates = []
    for node in graph_data.get("nodes", []):
        if not isinstance(node, dict) or not isinstance(node.get("id"), str):
            continue
        values = [str(node.get("id", "")).lower(), str(node.get("label", "")).lower()]
        if any(value and value in history_text for value in values):
            candidates.append(node["id"])
    return sorted(set(candidates))


def _resolve_semantic_node_reference(
    graph_data: dict[str, Any],
    raw: Any,
    selection: dict[str, Any] | None,
    request_context: dict[str, Any] | None = None,
) -> tuple[str | None, list[str], dict[str, Any]]:
    def resolve_recent_transaction() -> tuple[str | None, list[str], dict[str, Any]]:
        recent, transaction_id = _recent_transaction_node_candidates(graph_data, request_context)
        metadata = {"resolution_type": "recent_transaction"}
        if transaction_id:
            metadata["matched_transaction_id"] = transaction_id
        if len(recent) == 1:
            return recent[0], [], metadata
        if recent:
            return None, recent, metadata
        return None, [], metadata

    if _is_deictic_node_reference(raw):
        selected = _selected_node_ids(selection)
        if len(selected) == 1:
            return selected[0], [], {"resolution_type": "selection"}
        if selected:
            return None, selected, {"resolution_type": "selection"}
        recent_id, recent_candidates, recent_metadata = resolve_recent_transaction()
        if recent_id is not None or recent_candidates:
            return recent_id, recent_candidates, recent_metadata
        history_candidates = _conversation_history_node_candidates(graph_data, request_context)
        if len(history_candidates) == 1:
            return history_candidates[0], [], {"resolution_type": "conversation_history"}
        if history_candidates:
            return None, history_candidates, {"resolution_type": "conversation_history"}
        return None, [], {"resolution_type": "deictic"}
    if _is_recent_transaction_node_reference(raw):
        recent_id, recent_candidates, recent_metadata = resolve_recent_transaction()
        if recent_id is not None or recent_candidates:
            return recent_id, recent_candidates, recent_metadata
    node_id, ambiguous = _resolve_node_reference(graph_data, raw, selection)
    if node_id is not None or ambiguous or not isinstance(raw, str) or not raw.strip() or raw == "$selected_node":
        return node_id, ambiguous, {"resolution_type": "node"}
    lowered = raw.strip().lower()
    matches = []
    for node in graph_data.get("nodes", []):
        if not isinstance(node, dict) or not isinstance(node.get("id"), str):
            continue
        node_id_value = str(node.get("id", "")).lower()
        label_value = str(node.get("label", "")).lower()
        if lowered in node_id_value or lowered in label_value:
            matches.append(node["id"])
    matches = sorted(set(matches))
    if len(matches) == 1:
        return matches[0], [], {"resolution_type": "node"}
    if matches:
        return None, matches, {"resolution_type": "node"}
    semantic_node_id, semantic_ambiguous, semantic_resolution = resolve_semantic_node_reference(graph_data, raw)
    return semantic_node_id, semantic_ambiguous, semantic_resolution


def _edge_relation_terms(reference: str) -> set[str]:
    lowered = reference.strip().lower()
    terms = {lowered}
    compact = lowered.replace(" branch", "").replace("分支", "").strip()
    if compact:
        terms.add(compact)
    for canonical, aliases in EDGE_RELATION_ALIASES.items():
        if lowered == canonical or compact == canonical or any(alias in lowered for alias in aliases):
            terms.add(canonical)
            terms |= aliases
    return {term for term in terms if term}


def _edge_matches_reference(edge: dict[str, Any], reference: str) -> bool:
    terms = _edge_relation_terms(reference)
    values = {
        str(edge.get("id", "")).strip().lower(),
        str(edge.get("label", "")).strip().lower(),
        str(edge.get("branch_label", "")).strip().lower(),
    }
    condition = edge.get("condition")
    if isinstance(condition, dict) and isinstance(condition.get("expression"), str):
        values.add(condition["expression"].strip().lower())
    values = {value for value in values if value}
    if terms & values:
        return True
    return any(term in value or value in term for term in terms for value in values)


def _canonical_relation_evidence(raw: str | None) -> set[str]:
    lowered = str(raw or "").strip().lower()
    if not lowered:
        return set()
    compact = lowered.replace(" branch", "").replace("鍒嗘敮", "").strip()
    result: set[str] = set()
    for canonical, aliases in EDGE_RELATION_ALIASES.items():
        if lowered == canonical or compact == canonical or any(alias in lowered for alias in aliases):
            result.add(canonical)
    return result


def _edge_conflicts_relation_evidence(graph_data: dict[str, Any], edge_id: str, raw: str | None, evidence_text: str | None = None) -> bool:
    relation_candidates = _canonical_relation_evidence(raw) | _canonical_relation_evidence(evidence_text)
    if len(relation_candidates) != 1:
        return False
    edge = next((row for row in graph_data.get("edges", []) if isinstance(row, dict) and row.get("id") == edge_id), None)
    if not isinstance(edge, dict):
        return False
    edge_label = str(edge.get("label", "")).strip()
    return bool(edge_label and edge_label != next(iter(relation_candidates)))


def _edge_endpoint_reference_resolution(graph_data: dict[str, Any], raw: str | None) -> tuple[str | None, list[str], dict[str, Any]]:
    reference = str(raw or "").strip()
    if not reference:
        return None, [], {"resolution_type": "edge_endpoint_evidence"}
    nodes = {
        node["id"]: node
        for node in graph_data.get("nodes", [])
        if isinstance(node, dict) and isinstance(node.get("id"), str)
    }

    def node_score(node_id: Any) -> float:
        if not isinstance(node_id, str):
            return 0.0
        node = nodes.get(node_id)
        texts = [node_id]
        if isinstance(node, dict):
            texts.extend(str(node.get(key, "")) for key in ("label", "type", "semantic_type", "description"))
        return max((semantic_text_score(reference, text) for text in texts if str(text or "").strip()), default=0.0)

    def edge_local_score(edge: dict[str, Any]) -> float:
        texts: list[str] = []
        for key in ("id", "label", "branch_label", "type", "description"):
            value = edge.get(key)
            if isinstance(value, str) and value.strip():
                texts.append(value)
        condition = edge.get("condition")
        if isinstance(condition, dict) and isinstance(condition.get("expression"), str):
            texts.append(condition["expression"])
        return max((semantic_text_score(reference, text) for text in texts), default=0.0)

    candidates: list[tuple[tuple[int, int, float, float, float], str, dict[str, Any]]] = []
    for edge in graph_data.get("edges", []):
        if not (isinstance(edge, dict) and isinstance(edge.get("id"), str)):
            continue
        source_score = node_score(edge.get("source"))
        target_score = node_score(edge.get("target"))
        local_score = edge_local_score(edge)
        source_match = source_score >= 0.90
        target_match = target_score >= 0.90
        local_match = local_score >= 0.90 or bool(_canonical_relation_evidence(reference) & {
            str(edge.get("label", "")).strip().lower(),
            str(edge.get("branch_label", "")).strip().lower(),
        })
        endpoint_count = int(source_match) + int(target_match)
        if endpoint_count < 2 and not (endpoint_count == 1 and local_match):
            continue
        rank = (endpoint_count, int(local_match), source_score + target_score + local_score, source_score, target_score)
        candidates.append(
            (
                rank,
                edge["id"],
                {
                    "source_score": round(source_score, 4),
                    "target_score": round(target_score, 4),
                    "edge_score": round(local_score, 4),
                    "endpoint_count": endpoint_count,
                    "edge_local_match": local_match,
                },
            )
        )
    if not candidates:
        return None, [], {"resolution_type": "edge_endpoint_evidence", "semantic_candidates": []}
    candidates.sort(key=lambda row: (-row[0][0], -row[0][1], -row[0][2], -row[0][3], -row[0][4], row[1]))
    best_rank = candidates[0][0]
    best = [row for row in candidates if row[0] == best_rank]
    metadata = {
        "resolution_type": "edge_endpoint_evidence",
        "semantic_candidates": [
            {"id": edge_id, **evidence}
            for _rank, edge_id, evidence in candidates[:5]
        ],
    }
    if len(best) == 1:
        return best[0][1], [], metadata
    return None, sorted(edge_id for _rank, edge_id, _evidence in best), metadata


def _resolve_route_relation_reference(
    graph_data: dict[str, Any],
    raw: str | None,
    evidence_text: str | None,
    source_hints: list[str] | None = None,
) -> tuple[dict[str, Any] | None, list[str], dict[str, Any]]:
    relation_candidates = _canonical_relation_evidence(raw) | _canonical_relation_evidence(evidence_text)
    if len(relation_candidates) != 1:
        return None, [], {"resolution_type": "route_relation_evidence"}
    relation_type = next(iter(relation_candidates))
    relation_edges = [
        edge
        for edge in graph_data.get("edges", [])
        if isinstance(edge, dict)
        and edge.get("label", "") == relation_type
        and isinstance(edge.get("source"), str)
        and isinstance(edge.get("id"), str)
    ]
    relation_sources = sorted({edge["source"] for edge in relation_edges})
    node_id, ambiguous_nodes, node_resolution = _resolve_semantic_node_reference(graph_data, raw, None, None)
    if node_id is None and not ambiguous_nodes and len(relation_sources) == 1:
        node_id = relation_sources[0]
        node_resolution = {"resolution_type": "route_relation_existing_source"}
    valid_hints = [node_id for node_id in (source_hints or []) if isinstance(node_id, str)]
    if node_id is None and not ambiguous_nodes and len(set(valid_hints)) == 1:
        node_id = valid_hints[0]
        node_resolution = {"resolution_type": "route_relation_semantic_source_hint"}
    if node_id is None:
        return None, ambiguous_nodes, {"resolution_type": "route_relation_evidence", "relation_type": relation_type, "node_resolution": node_resolution}

    matching_edges = [edge for edge in relation_edges if edge.get("source") == node_id]
    if len(matching_edges) > 1:
        return None, sorted(edge["id"] for edge in matching_edges), {
            "resolution_type": "route_relation_evidence",
            "relation_type": relation_type,
            "source": node_id,
        }
    graph_facts = {"source": node_id, "target": None, "label": relation_type, "branch_label": relation_type, "has_condition": False}
    if len(matching_edges) == 1:
        edge = matching_edges[0]
        graph_facts = _edge_facts(graph_data, edge["id"])
        return (
            {"id": edge["id"], "graph_facts": graph_facts},
            [],
            {"resolution_type": "route_relation_evidence", "relation_type": relation_type, "source": node_id},
        )
    return (
        {"id": None, "graph_facts": graph_facts},
        [],
        {"resolution_type": "route_relation_evidence", "relation_type": relation_type, "source": node_id, "virtual_relation": True},
    )


def _resolve_edge_reference(
    graph_data: dict[str, Any],
    raw: str | None,
    selection: dict[str, Any] | None,
    evidence_text: str | None = None,
) -> tuple[str | None, list[str], dict[str, Any]]:
    if not isinstance(raw, str) or not raw.strip():
        return None, [], {"resolution_type": "edge"}
    selected_edge_ids = _selected_edge_ids(selection)
    selected = [edge_id for edge_id in selected_edge_ids if isinstance(edge_id, str)]
    if raw.strip().lower() in DEICTIC_EDGE_REFERENCES or raw.strip() in DEICTIC_EDGE_REFERENCES:
        if len(selected) == 1:
            return selected[0], [], {"resolution_type": "selection"}
        if selected:
            return None, sorted(selected), {"resolution_type": "selection"}
        graph_edges = sorted(
            edge["id"]
            for edge in graph_data.get("edges", [])
            if isinstance(edge, dict) and isinstance(edge.get("id"), str)
        )
        return None, graph_edges, {"resolution_type": "deictic"}
    if raw.strip() in {"$selected_edge", "this branch", "selected branch", "this edge", "selected edge", "这个分支", "该分支"}:
        return (selected[0], [], {"resolution_type": "selection"}) if len(selected) == 1 else (None, sorted(selected), {"resolution_type": "selection"})
    endpoint_edge_id, endpoint_ambiguous, endpoint_resolution = _edge_endpoint_reference_resolution(graph_data, raw)
    if endpoint_edge_id is not None:
        return endpoint_edge_id, [], endpoint_resolution
    if endpoint_ambiguous:
        return None, endpoint_ambiguous, endpoint_resolution
    strict_edge_id, strict_ambiguous, strict_resolution = resolve_strict_semantic_edge_reference(graph_data, raw)
    if strict_edge_id is not None:
        return strict_edge_id, [], strict_resolution
    if strict_ambiguous:
        return None, strict_ambiguous, strict_resolution
    matches = [
        edge["id"]
        for edge in graph_data.get("edges", [])
        if isinstance(edge, dict) and isinstance(edge.get("id"), str) and _edge_matches_reference(edge, raw)
    ]
    matches = sorted(set(matches))
    if len(matches) == 1:
        return matches[0], [], {"resolution_type": "edge"}
    if matches:
        return None, matches, {"resolution_type": "edge"}
    semantic_edge_id, semantic_ambiguous, semantic_resolution = resolve_semantic_edge_reference(graph_data, raw)
    if isinstance(semantic_edge_id, str) and _edge_conflicts_relation_evidence(graph_data, semantic_edge_id, raw, evidence_text):
        semantic_edge_id = None
    if semantic_edge_id is None and not semantic_ambiguous and selected:
        if len(selected) == 1:
            return selected[0], [], {"resolution_type": "selection"}
        return None, sorted(selected), {"resolution_type": "selection"}
    return semantic_edge_id, semantic_ambiguous, semantic_resolution


def _should_recover_route_slot_as_edge(goal: dict[str, Any], item: dict[str, Any]) -> bool:
    return (
        goal.get("kind") == "route"
        and item.get("type") == "existing_reference"
        and str(item.get("role") or "").strip() == "route"
    )


def _edge_facts(graph_data: dict[str, Any], edge_id: str) -> dict[str, Any]:
    edge = next((row for row in graph_data.get("edges", []) if isinstance(row, dict) and row.get("id") == edge_id), {})
    return {
        "source": edge.get("source"),
        "target": edge.get("target"),
        "label": edge.get("label", ""),
        "branch_label": edge.get("branch_label", ""),
        "has_condition": isinstance(edge.get("condition"), dict),
    }


def _ground_preserve_constraints(
    normalized: dict[str, Any],
    graph_data: dict[str, Any],
    issues: list[dict[str, Any]],
    provenance: list[dict[str, Any]],
) -> dict[str, Any]:
    preserve: dict[str, Any] = {"node_ids": [], "edge_ids": [], "abstract": []}
    seen_nodes: set[str] = set()
    seen_edges: set[str] = set()
    for goal_index, goal in enumerate(normalized.get("semantic_goals_v2", []) if isinstance(normalized.get("semantic_goals_v2"), list) else []):
        if not isinstance(goal, dict):
            continue
        for constraint_index, constraint in enumerate(goal.get("semantic_constraints", []) if isinstance(goal.get("semantic_constraints"), list) else []):
            if not isinstance(constraint, dict) or str(constraint.get("type") or "").strip() != "preserve":
                continue
            if constraint.get("constraint_source") == "legacy_explicit_constraint":
                continue
            target = str(constraint.get("target") or constraint.get("text") or "").strip()
            location = f"semantic_goals_v2[{goal_index}].semantic_constraints[{constraint_index}]"
            if not target:
                continue
            edge_id, edge_ambiguous, edge_resolution = resolve_strict_semantic_edge_reference(graph_data, target)
            if isinstance(edge_id, str) and edge_id:
                if edge_id not in seen_edges:
                    seen_edges.add(edge_id)
                    preserve["edge_ids"].append(edge_id)
                    _add_provenance(provenance, location, target, edge_id, str(edge_resolution.get("resolution_type") or "edge"))
                continue
            if edge_ambiguous:
                _add_issue(issues, "ambiguous_preserve_reference", location, target, sorted(edge_ambiguous), "preserve constraint matches multiple graph edges")
                continue
            node_id, node_ambiguous, node_resolution = resolve_strict_semantic_node_reference(graph_data, target)
            if isinstance(node_id, str) and node_id:
                if node_id not in seen_nodes:
                    seen_nodes.add(node_id)
                    preserve["node_ids"].append(node_id)
                    _add_provenance(provenance, location, target, node_id, str(node_resolution.get("resolution_type") or "node"))
                continue
            ambiguous = sorted(set(edge_ambiguous or []) | set(node_ambiguous or []))
            if ambiguous:
                _add_issue(issues, "ambiguous_preserve_reference", location, target, ambiguous, "preserve constraint matches multiple graph objects")
            else:
                preserve["abstract"].append({"target": target, "text": str(constraint.get("text") or target).strip(), "location": location})
    preserve["node_ids"].sort()
    preserve["edge_ids"].sort()
    return preserve


def _recover_deleted_middle_node_direct_connections(
    grounded_goals: list[dict[str, Any]],
    graph_data: dict[str, Any],
    selection: dict[str, Any] | None,
    request_context: dict[str, Any] | None,
    issues: list[dict[str, Any]],
    provenance: list[dict[str, Any]],
) -> None:
    absent_node_ids: set[str] = set()
    for goal in grounded_goals:
        if goal.get("kind") != "object_state":
            continue
        state = goal.get("object_state") if isinstance(goal.get("object_state"), dict) else {}
        if state.get("state") != "absent":
            continue
        binding = next(
            (
                item
                for item in goal.get("bindings", [])
                if isinstance(item, dict)
                and item.get("role") == "object"
                and item.get("binding_type") == "node"
                and isinstance(item.get("id"), str)
            ),
            None,
        )
        if binding is not None:
            absent_node_ids.add(binding["id"])
    if not absent_node_ids:
        return

    def object_by_role(goal: dict[str, Any], role: str) -> dict[str, Any] | None:
        return next((item for item in goal.get("objects", []) if isinstance(item, dict) and item.get("role") == role), None)

    def binding_by_role(goal: dict[str, Any], role: str) -> dict[str, Any] | None:
        return next((item for item in goal.get("bindings", []) if isinstance(item, dict) and item.get("role") == role), None)

    def direct_unconditional_edge(source: str, target: str) -> dict[str, Any] | None:
        return next(
            (
                edge
                for edge in graph_data.get("edges", [])
                if isinstance(edge, dict)
                and edge.get("source") == source
                and edge.get("target") == target
                and edge.get("label", "") == ""
                and not edge.get("branch_label")
                and not edge.get("condition")
            ),
            None,
        )

    for goal_index, goal in enumerate(grounded_goals):
        if goal.get("kind") != "route" or goal.get("relation") != "route_to":
            continue
        route_object = object_by_role(goal, "route")
        if not (isinstance(route_object, dict) and route_object.get("type") != "edge_reference"):
            continue
        target_binding = binding_by_role(goal, "target")
        target_id = target_binding.get("id") if isinstance(target_binding, dict) and target_binding.get("binding_type") == "node" else None
        if not isinstance(target_id, str):
            continue
        source_id, source_ambiguous, source_resolution = _resolve_semantic_node_reference(
            graph_data,
            route_object.get("reference"),
            selection,
            request_context,
        )
        if not isinstance(source_id, str) or source_ambiguous:
            continue
        middle_nodes = [
            node_id
            for node_id in sorted(absent_node_ids)
            if direct_unconditional_edge(source_id, node_id) is not None
            and direct_unconditional_edge(node_id, target_id) is not None
        ]
        if len(middle_nodes) != 1:
            continue

        route_location = f"semantic_goals[{goal_index}].objects[0]"
        goal["kind"] = "connection"
        goal["relation"] = "connected_to"
        for item in goal.get("objects", []):
            if isinstance(item, dict) and item.get("role") == "route":
                item["role"] = "source"
        source_graph_facts = _node_flow_facts(graph_data, source_id)
        source_binding = {**route_object, "role": "source", "binding_type": "node", "id": source_id, "graph_facts": source_graph_facts}
        goal["bindings"] = [
            item
            for item in goal.get("bindings", [])
            if not (isinstance(item, dict) and item.get("role") == "route")
        ]
        goal["bindings"].append(source_binding)
        goal.pop("candidates", None)
        issues[:] = [
            issue
            for issue in issues
            if issue.get("location") != route_location
        ]
        _add_provenance(provenance, route_location, route_object.get("reference"), source_id, str(source_resolution.get("resolution_type") or "node"))
        provenance[-1]["graph_facts"] = source_graph_facts
        provenance[-1]["direct_connection_via_absent_node"] = middle_nodes[0]


def ground_semantic_goal_spec(
    value: dict[str, Any],
    graph_data: dict[str, Any],
    selection: dict[str, Any] | None = None,
    request_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    normalized = normalize_semantic_goal_spec(value)
    issues: list[dict[str, Any]] = []
    provenance: list[dict[str, Any]] = []
    grounded_goals: list[dict[str, Any]] = []
    route_source_hints: list[str] = []

    for goal_index, goal in enumerate(normalized["semantic_goals"]):
        grounded_goal = deepcopy(goal)
        bindings: list[dict[str, Any]] = []
        candidates: list[dict[str, Any]] = []
        for object_index, item in enumerate(goal.get("objects", [])):
            location = f"semantic_goals[{goal_index}].objects[{object_index}]"
            if item.get("type") == "new_entity":
                binding = {**item, "needs_creation": True}
                bindings.append(binding)
                _add_provenance(provenance, location, item.get("label"), item.get("label"), "new_entity")
                continue
            if item.get("type") == "edge_reference":
                confirmed_edge_id = _candidate_binding(request_context, location, item.get("reference"), "edge")
                if confirmed_edge_id:
                    if _graph_has_edge(graph_data, confirmed_edge_id):
                        graph_facts = _edge_facts(graph_data, confirmed_edge_id)
                        binding = {**item, "binding_type": "edge", "id": confirmed_edge_id, "graph_facts": graph_facts}
                        bindings.append(binding)
                        _add_provenance(provenance, location, item.get("reference"), confirmed_edge_id, "user_clarification")
                        provenance[-1]["graph_facts"] = graph_facts
                    else:
                        _add_issue(issues, "stale_edge_reference", location, item.get("reference"), [confirmed_edge_id], "selected edge candidate no longer exists")
                    continue
                edge_id, ambiguous, resolution = _resolve_edge_reference(graph_data, item.get("reference"), selection, goal.get("text"))
                if edge_id is not None:
                    graph_facts = _edge_facts(graph_data, edge_id)
                    binding = {**item, "binding_type": "edge", "id": edge_id, "graph_facts": graph_facts}
                    bindings.append(binding)
                    _add_provenance(provenance, location, item.get("reference"), edge_id, str(resolution.get("resolution_type") or "edge"))
                    provenance[-1]["graph_facts"] = graph_facts
                elif ambiguous:
                    route_relation = None
                    relation_ambiguous: list[str] = []
                    relation_resolution: dict[str, Any] = {}
                    if _should_recover_route_slot_as_edge(goal, {**item, "type": "existing_reference"}):
                        route_relation, relation_ambiguous, relation_resolution = _resolve_route_relation_reference(
                            graph_data,
                            item.get("reference"),
                            goal.get("text"),
                            route_source_hints,
                        )
                    if route_relation is not None:
                        graph_facts = route_relation["graph_facts"]
                        binding = {**item, "binding_type": "edge", "graph_facts": graph_facts}
                        if isinstance(route_relation.get("id"), str):
                            binding["id"] = route_relation["id"]
                        bindings.append(binding)
                        _add_provenance(
                            provenance,
                            location,
                            item.get("reference"),
                            route_relation.get("id") or f"{graph_facts.get('source')}:{graph_facts.get('label')}",
                            str(relation_resolution.get("resolution_type") or "route_relation_evidence"),
                        )
                        provenance[-1]["graph_facts"] = graph_facts
                    elif relation_ambiguous:
                        candidate = {**item, "binding_type": "edge", "candidates": relation_ambiguous}
                        candidates.append(candidate)
                        _add_issue(issues, "ambiguous_edge_reference", location, item.get("reference"), relation_ambiguous, "multiple graph edges match reference")
                    else:
                        candidate = {**item, "binding_type": "edge", "candidates": ambiguous}
                        candidates.append(candidate)
                        _add_issue(issues, "ambiguous_edge_reference", location, item.get("reference"), ambiguous, "multiple graph edges match reference")
                else:
                    route_relation = None
                    relation_ambiguous: list[str] = []
                    relation_resolution: dict[str, Any] = {}
                    if _should_recover_route_slot_as_edge(goal, {**item, "type": "existing_reference"}):
                        route_relation, relation_ambiguous, relation_resolution = _resolve_route_relation_reference(
                            graph_data,
                            item.get("reference"),
                            goal.get("text"),
                            route_source_hints,
                        )
                    if route_relation is not None:
                        graph_facts = route_relation["graph_facts"]
                        binding = {**item, "binding_type": "edge", "graph_facts": graph_facts}
                        if isinstance(route_relation.get("id"), str):
                            binding["id"] = route_relation["id"]
                        bindings.append(binding)
                        _add_provenance(
                            provenance,
                            location,
                            item.get("reference"),
                            route_relation.get("id") or f"{graph_facts.get('source')}:{graph_facts.get('label')}",
                            str(relation_resolution.get("resolution_type") or "route_relation_evidence"),
                        )
                        provenance[-1]["graph_facts"] = graph_facts
                    elif relation_ambiguous:
                        candidate = {**item, "binding_type": "edge", "candidates": relation_ambiguous}
                        candidates.append(candidate)
                        _add_issue(issues, "ambiguous_edge_reference", location, item.get("reference"), relation_ambiguous, "multiple graph edges match reference")
                    else:
                        _add_issue(issues, "missing_edge_reference", location, item.get("reference"), [], "no graph edge matches reference")
                continue
            if _should_recover_route_slot_as_edge(goal, item):
                route_relation, relation_ambiguous, relation_resolution = _resolve_route_relation_reference(
                    graph_data,
                    item.get("reference"),
                    goal.get("text"),
                    route_source_hints,
                )
                edge_id, ambiguous, resolution = (
                    (None, relation_ambiguous, relation_resolution)
                    if route_relation is None and relation_ambiguous
                    else _resolve_edge_reference(graph_data, item.get("reference"), selection, goal.get("text"))
                )
                if route_relation is not None:
                    graph_facts = route_relation["graph_facts"]
                    binding = {**item, "binding_type": "edge", "graph_facts": graph_facts}
                    if isinstance(route_relation.get("id"), str):
                        binding["id"] = route_relation["id"]
                    bindings.append(binding)
                    _add_provenance(
                        provenance,
                        location,
                        item.get("reference"),
                        route_relation.get("id") or f"{graph_facts.get('source')}:{graph_facts.get('label')}",
                        str(relation_resolution.get("resolution_type") or "route_relation_evidence"),
                    )
                    provenance[-1]["graph_facts"] = graph_facts
                elif edge_id is not None:
                    graph_facts = _edge_facts(graph_data, edge_id)
                    binding = {**item, "binding_type": "edge", "id": edge_id, "graph_facts": graph_facts}
                    bindings.append(binding)
                    _add_provenance(provenance, location, item.get("reference"), edge_id, str(resolution.get("resolution_type") or "edge"))
                    provenance[-1]["graph_facts"] = graph_facts
                elif ambiguous:
                    candidate = {**item, "type": "edge_reference", "binding_type": "edge", "candidates": ambiguous}
                    candidates.append(candidate)
                    _add_issue(issues, "ambiguous_edge_reference", location, item.get("reference"), ambiguous, "multiple graph edges match reference")
                else:
                    _add_issue(issues, "missing_edge_reference", location, item.get("reference"), [], "no graph edge matches reference")
                continue
            confirmed_node_id = _candidate_binding(request_context, location, item.get("reference"), "node")
            if confirmed_node_id:
                if _graph_has_node(graph_data, confirmed_node_id):
                    graph_facts = _node_flow_facts(graph_data, confirmed_node_id)
                    binding = {**item, "binding_type": "node", "id": confirmed_node_id, "graph_facts": graph_facts}
                    bindings.append(binding)
                    _add_provenance(provenance, location, item.get("reference"), confirmed_node_id, "user_clarification")
                    provenance[-1]["graph_facts"] = graph_facts
                else:
                    _add_issue(issues, "stale_node_reference", location, item.get("reference"), [confirmed_node_id], "selected node candidate no longer exists")
                continue
            node_id, ambiguous, resolution = _resolve_semantic_node_reference(graph_data, item.get("reference"), selection, request_context)
            if node_id is not None:
                graph_facts = _node_flow_facts(graph_data, node_id)
                binding = {**item, "binding_type": "node", "id": node_id, "graph_facts": graph_facts}
                bindings.append(binding)
                _add_provenance(provenance, location, item.get("reference"), node_id, str(resolution.get("resolution_type") or "node"))
                provenance[-1]["graph_facts"] = graph_facts
                if isinstance(resolution.get("matched_transaction_id"), str):
                    provenance[-1]["matched_transaction_id"] = resolution["matched_transaction_id"]
            elif ambiguous:
                candidate = {**item, "binding_type": "node", "candidates": ambiguous}
                candidates.append(candidate)
                reason = "multiple graph nodes match reference"
                if resolution.get("resolution_type") == "recent_transaction":
                    reason = "multiple nodes were affected by the recent transaction"
                elif resolution.get("resolution_type") == "selection":
                    reason = "multiple selected nodes match reference"
                _add_issue(issues, "ambiguous_node_reference", location, item.get("reference"), ambiguous, reason)
            else:
                _add_issue(issues, "missing_node_reference", location, item.get("reference"), [], "no graph node matches reference")
        grounded_goal["bindings"] = bindings
        if candidates:
            grounded_goal["candidates"] = candidates
        grounded_goals.append(grounded_goal)
        for binding in bindings:
            if (
                isinstance(binding, dict)
                and binding.get("binding_type") == "node"
                and isinstance(binding.get("id"), str)
                and binding.get("role") not in {"target", "anchor"}
            ):
                route_source_hints.append(binding["id"])

    _recover_deleted_middle_node_direct_connections(grounded_goals, graph_data, selection, request_context, issues, provenance)
    preserve_constraints = _ground_preserve_constraints(normalized, graph_data, issues, provenance)
    return {
        "status": normalized["status"],
        "semantic_goals": grounded_goals,
        "semantic_goals_v2": deepcopy(normalized.get("semantic_goals_v2", [])),
        "clarification": normalized["clarification"],
        "preserve_constraints": preserve_constraints,
        "grounding": {"issues": issues, "provenance": provenance},
    }


def _reference_text(reference: dict[str, Any]) -> str:
    mention = reference.get("mention") if isinstance(reference.get("mention"), dict) else {}
    return str(mention.get("text") or "").strip()


def _creatable_reference_ids(goal_state: dict[str, Any]) -> set[str]:
    result: set[str] = set()
    for fact in goal_state.get("desired_facts", []) if isinstance(goal_state.get("desired_facts"), list) else []:
        if not isinstance(fact, dict) or fact.get("predicate") != "entity.exists":
            continue
        operands = fact.get("operands") if isinstance(fact.get("operands"), dict) else {}
        subject = operands.get("subject") if isinstance(operands.get("subject"), dict) else {}
        ref_id = subject.get("ref_id")
        if isinstance(ref_id, str) and ref_id:
            result.add(ref_id)
    return result


def ground_canonical_relation_facts(
    goal_state: dict[str, Any],
    graph_data: dict[str, Any],
    selection: dict[str, Any] | None = None,
    request_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Ground canonical RelationFacts without flattening or changing production semantics."""
    issues: list[dict[str, Any]] = []
    provenance: list[dict[str, Any]] = []
    references = {
        ref.get("ref_id"): ref
        for ref in goal_state.get("references", [])
        if isinstance(ref, dict) and isinstance(ref.get("ref_id"), str)
    }
    creatable_refs = _creatable_reference_ids(goal_state)

    def ground_ref(ref_id: str, location: str) -> dict[str, Any] | None:
        reference = references.get(ref_id)
        if not isinstance(reference, dict):
            _add_issue(issues, "missing_semantic_reference", location, ref_id, [], "relation member references an unknown semantic ref")
            return None
        entity_type = str(reference.get("entity_type") or "unknown")
        text = _reference_text(reference)
        if ref_id in creatable_refs:
            return {"ref_id": ref_id, "binding_type": "new_entity", "needs_creation": True, "label": text}
        if entity_type != "node":
            _add_issue(issues, "unsupported_relation_fact_endpoint_type", location, ref_id, [], "grouped RelationFact grounding currently supports node endpoints")
            return None
        node_id, ambiguous, resolution = _resolve_semantic_node_reference(graph_data, text, selection, request_context)
        if node_id is not None:
            graph_facts = _node_flow_facts(graph_data, node_id)
            _add_provenance(provenance, location, text, node_id, str(resolution.get("resolution_type") or "node"))
            provenance[-1]["graph_facts"] = graph_facts
            return {"ref_id": ref_id, "binding_type": "node", "id": node_id, "graph_facts": graph_facts}
        if ambiguous:
            reason = "multiple graph nodes match reference"
            if resolution.get("resolution_type") == "recent_transaction":
                reason = "multiple nodes were affected by the recent transaction"
            elif resolution.get("resolution_type") == "selection":
                reason = "multiple selected nodes match reference"
            _add_issue(issues, "ambiguous_node_reference", location, text, ambiguous, reason)
            return None
        _add_issue(issues, "missing_node_reference", location, text, [], "no graph node matches reference")
        return None

    grounded_relation_facts: list[dict[str, Any]] = []
    for relation_index, relation_fact in enumerate(goal_state.get("relation_facts", []) if isinstance(goal_state.get("relation_facts"), list) else []):
        normalized = normalize_relation_fact(relation_fact)
        grounded_members: list[dict[str, Any]] = []
        for member_index, member in enumerate(normalized["members"]):
            source_location = f"relation_facts[{relation_index}].members[{member_index}].source_ref"
            target_location = f"relation_facts[{relation_index}].members[{member_index}].target_ref"
            source_binding = ground_ref(member["source_ref"], source_location)
            target_binding = ground_ref(member["target_ref"], target_location)
            grounded_member = deepcopy(member)
            if source_binding is not None:
                grounded_member["source_binding"] = source_binding
            if target_binding is not None:
                grounded_member["target_binding"] = target_binding
            grounded_members.append(grounded_member)
        grounded = deepcopy(normalized)
        grounded["members"] = grounded_members
        grounded_relation_facts.append(grounded)

    return {"relation_facts": grounded_relation_facts, "grounding": {"issues": issues, "provenance": provenance}}


def validate_grounded_goal(value: dict[str, Any]) -> dict[str, Any]:
    issues: list[dict[str, Any]] = []
    if not isinstance(value, dict):
        return {
            "valid": False,
            "lowering_ready": False,
            "issues": [{"kind": "invalid_grounded_goal", "message": "grounded goal must be an object", "severity": "error", "location": ""}],
        }
    grounding = value.get("grounding") if isinstance(value.get("grounding"), dict) else {}
    grounding_issues = grounding.get("issues", [])
    if isinstance(grounding_issues, list):
        for issue in grounding_issues:
            if isinstance(issue, dict):
                issues.append(
                    {
                        "kind": str(issue.get("kind", "grounding_issue")),
                        "message": str(issue.get("reason", "grounding issue")),
                        "severity": "blocking",
                        "location": str(issue.get("location", "grounding")),
                        "candidates": issue.get("candidates", []),
                    }
                )
    else:
        issues.append({"kind": "invalid_grounding_issues", "message": "grounding.issues must be an array", "severity": "error", "location": "grounding.issues"})
    clarification = value.get("clarification") if isinstance(value.get("clarification"), dict) else {}
    if clarification.get("needed"):
        issues.append({"kind": "clarification_needed", "message": str(clarification.get("reason", "clarification needed")), "severity": "blocking", "location": "clarification"})

    def is_valid_ordering_node_participant(binding: Any) -> bool:
        return isinstance(binding, dict) and (
            binding.get("binding_type") == "node" or binding.get("needs_creation") is True
        )

    for index, goal in enumerate(value.get("semantic_goals", []) if isinstance(value.get("semantic_goals"), list) else []):
        if goal.get("kind") == "ordering":
            roles = {binding.get("role"): binding for binding in goal.get("bindings", []) if isinstance(binding, dict)}
            subject = roles.get("subject")
            anchor = roles.get("anchor")
            if not is_valid_ordering_node_participant(subject):
                issues.append({"kind": "missing_ordering_subject", "message": "ordering requires a subject binding or new entity", "severity": "blocking", "location": f"semantic_goals[{index}].bindings"})
            if not is_valid_ordering_node_participant(anchor):
                issues.append({"kind": "missing_ordering_anchor", "message": "ordering requires an anchor binding or new entity", "severity": "blocking", "location": f"semantic_goals[{index}].bindings"})
            continue
        if goal.get("kind") == "branch_property":
            edge_binding = next(
                (
                    binding
                    for binding in goal.get("bindings", [])
                    if isinstance(binding, dict) and binding.get("role") == "branch" and binding.get("binding_type") == "edge"
                ),
                None,
            )
            prop = goal.get("property") if isinstance(goal.get("property"), dict) else {}
            if not isinstance(edge_binding, dict):
                issues.append({"kind": "missing_branch_edge_binding", "message": "branch_property requires an edge binding", "severity": "blocking", "location": f"semantic_goals[{index}].bindings"})
            if prop.get("name") not in SUPPORTED_BRANCH_PROPERTY_NAMES or not isinstance(prop.get("value"), str) or not prop.get("value").strip():
                issues.append({"kind": "invalid_branch_property", "message": "branch_property requires supported property name and non-empty value", "severity": "blocking", "location": f"semantic_goals[{index}].property"})
            continue
        if goal.get("kind") == "route":
            roles = {binding.get("role"): binding for binding in goal.get("bindings", []) if isinstance(binding, dict)}
            route = roles.get("route")
            target = roles.get("target")
            if not (isinstance(route, dict) and route.get("binding_type") == "edge"):
                issues.append({"kind": "missing_route_edge_binding", "message": "route requires an existing edge binding", "severity": "blocking", "location": f"semantic_goals[{index}].bindings"})
            if not (
                isinstance(target, dict)
                and (target.get("binding_type") == "node" or target.get("needs_creation") is True)
            ):
                issues.append({"kind": "missing_route_target_binding", "message": "route requires an existing target node binding", "severity": "blocking", "location": f"semantic_goals[{index}].bindings"})
            continue
        if goal.get("kind") == "connection":
            roles = {binding.get("role"): binding for binding in goal.get("bindings", []) if isinstance(binding, dict)}
            source = roles.get("source")
            target = roles.get("target")
            if not is_valid_ordering_node_participant(source):
                issues.append({"kind": "missing_connection_source_binding", "message": "connection requires a source node binding or new entity", "severity": "blocking", "location": f"semantic_goals[{index}].bindings"})
            if not is_valid_ordering_node_participant(target):
                issues.append({"kind": "missing_connection_target_binding", "message": "connection requires a target node binding or new entity", "severity": "blocking", "location": f"semantic_goals[{index}].bindings"})
            continue
        if goal.get("kind") == "object_state":
            binding = next((item for item in goal.get("bindings", []) if isinstance(item, dict) and item.get("role") == "object"), None)
            state = goal.get("object_state") if isinstance(goal.get("object_state"), dict) else {}
            state_name = state.get("state")
            if state_name not in SUPPORTED_OBJECT_STATES:
                issues.append({"kind": "invalid_object_state", "message": "object_state requires a supported desired state", "severity": "blocking", "location": f"semantic_goals[{index}].object_state"})
            if not isinstance(binding, dict):
                issues.append({"kind": "missing_object_state_binding", "message": "object_state requires an object binding or new entity", "severity": "blocking", "location": f"semantic_goals[{index}].bindings"})
                continue
            if state_name == "exists" and not (binding.get("needs_creation") is True or binding.get("binding_type") == "node"):
                issues.append({"kind": "missing_exists_binding", "message": "exists state requires a new entity or existing node", "severity": "blocking", "location": f"semantic_goals[{index}].bindings"})
            if state_name == "absent" and binding.get("binding_type") not in {"node", "edge"}:
                issues.append({"kind": "missing_absent_binding", "message": "absent state requires an existing node or edge binding", "severity": "blocking", "location": f"semantic_goals[{index}].bindings"})
            if state_name == "property_equals":
                prop = state.get("property")
                value = state.get("value")
                binding_type = binding.get("binding_type")
                node_properties = {"label", "name", "display_label"}
                edge_properties = {"label", "display_label", "branch_label", "condition.expression", "condition"}
                property_allowed = (
                    (binding_type == "node" and prop in node_properties)
                    or (binding_type == "edge" and prop in edge_properties)
                )
                if not property_allowed or not isinstance(value, str) or not value.strip():
                    issues.append({"kind": "invalid_property_equals", "message": "property_equals requires a safe display property and non-empty value", "severity": "blocking", "location": f"semantic_goals[{index}].object_state"})
                if binding_type not in {"node", "edge"}:
                    issues.append({"kind": "missing_property_binding", "message": "property_equals requires an existing node or edge binding", "severity": "blocking", "location": f"semantic_goals[{index}].bindings"})
            continue
        if goal.get("kind") == "parallel":
            roles = {binding.get("role"): binding for binding in goal.get("bindings", []) if isinstance(binding, dict)}
            left = roles.get("left")
            right = roles.get("right")
            if not (isinstance(left, dict) and left.get("binding_type") == "node" and isinstance(right, dict) and right.get("binding_type") == "node"):
                issues.append({"kind": "missing_parallel_binding", "message": "parallel_with requires two existing node bindings", "severity": "blocking", "location": f"semantic_goals[{index}].bindings"})
            continue
        if goal.get("kind") != "layout" or goal.get("relation") != "right_of":
            issues.append({"kind": "unsupported_grounded_goal", "message": "this grounded goal kind is not supported by current lowering", "severity": "blocking", "location": f"semantic_goals[{index}]"})
            continue
        roles = {
            binding.get("role")
            for binding in goal.get("bindings", [])
            if isinstance(binding, dict) and binding.get("binding_type") == "node"
        }
        if not {"target", "anchor"} <= roles:
            issues.append({"kind": "missing_layout_binding", "message": "layout right_of requires target and anchor bindings", "severity": "blocking", "location": f"semantic_goals[{index}].bindings"})
    hard_error = any(issue["severity"] == "error" for issue in issues)
    blocker = any(issue["severity"] in {"error", "blocking"} for issue in issues)
    return {"valid": not hard_error, "lowering_ready": not blocker, "issues": issues}
