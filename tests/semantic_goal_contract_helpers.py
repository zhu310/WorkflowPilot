from __future__ import annotations

from typing import Any


FORBIDDEN_ID_KEYS = {
    "node_id",
    "edge_id",
    "source_id",
    "target_id",
    "selected_node_id",
    "selected_edge_id",
}

FORBIDDEN_WORKFLOW_KEYS = {
    "workflow_ir",
    "control_flow_regions",
    "auxiliary_relations",
    "properties",
    "edge_properties",
    "absent_entities",
    "operations",
    "mutations",
    "graph_diff",
    "add_node",
    "delete_node",
    "remove_node",
    "update_node",
    "add_edge",
    "delete_edge",
    "remove_edge",
    "replace_edge",
    "planner_steps",
    "executor_steps",
    "required_relations",
    "forbidden_relations",
}

DISCRIMINATOR_KEYS = {"kind", "type", "action", "operation", "op", "goal_kind"}

FORBIDDEN_OPERATION_VALUES = {
    "insert",
    "delete",
    "rename",
    "move",
    "route",
    "reroute",
    "add_node",
    "delete_node",
    "update_node",
    "add_edge",
    "delete_edge",
    "replace",
    "reconnect",
    "bypass",
}

ALLOWED_DISCRIMINATOR_VALUES = {
    "existing_reference",
    "new_entity",
    "edge_reference",
    "preserve",
    "object_state_goal",
    "relation_goal",
    "constraint_goal",
    "composite_goal",
}


def _walk(value: Any, path: str = "") -> list[tuple[str, Any]]:
    rows: list[tuple[str, Any]] = [(path, value)]
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = f"{path}.{key}" if path else str(key)
            rows.extend(_walk(child, child_path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            child_path = f"{path}[{index}]"
            rows.extend(_walk(child, child_path))
    return rows


def semantic_goal_contract_violations(value: dict[str, Any]) -> list[dict[str, str]]:
    violations: list[dict[str, str]] = []
    for path, current in _walk(value):
        if not isinstance(current, dict):
            continue
        for key, child in current.items():
            normalized_key = str(key).strip().lower()
            child_path = f"{path}.{key}" if path else str(key)
            if normalized_key in FORBIDDEN_ID_KEYS:
                violations.append({"kind": "forbidden_id_key", "path": child_path, "value": normalized_key})
            if normalized_key in FORBIDDEN_WORKFLOW_KEYS:
                violations.append({"kind": "forbidden_workflow_key", "path": child_path, "value": normalized_key})
            if normalized_key in DISCRIMINATOR_KEYS and isinstance(child, str):
                normalized_value = child.strip().lower()
                if normalized_value in FORBIDDEN_OPERATION_VALUES and normalized_value not in ALLOWED_DISCRIMINATOR_VALUES:
                    violations.append({"kind": "operation_kind_regression", "path": child_path, "value": normalized_value})
    return violations
