from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from src.mvp23 import run_transaction
from src.runner import save_json
from src.workflow_ir_core import graph_is_valid


CATALOG_TYPES = [
    "start",
    "end",
    "decision",
    "parallel_split",
    "parallel_join",
    "subprocess",
    "loop",
    "wait",
    "task",
    "ai_agent",
    "tool",
    "human_review",
    "notification",
    "checkpoint",
    "data_source",
    "document",
    "storage",
    "variable",
    "resource",
    "note",
    "group",
    "milestone",
    "error_handler",
]

STRUCTURAL_TYPES = {"decision", "parallel_split", "parallel_join", "subprocess", "loop", "wait"}


@dataclass(frozen=True)
class CatalogCase:
    name: str
    graph: dict[str, Any]
    instruction: str
    selection: dict[str, Any]
    evaluator: Callable[[dict[str, Any], Any], tuple[bool, dict[str, Any]]]


def catalog_graph(include_unknown: bool = False) -> dict[str, Any]:
    nodes = [
        {"id": "start", "type": "start", "label": "Start", "x": 0, "y": 0, "metadata": {"role": "entry"}},
        {"id": "decision", "type": "decision", "label": "Eligibility Decision", "x": 180, "y": 0, "metadata": {"rule": "eligibility"}},
        {"id": "parallel_split", "type": "parallel_split", "label": "Parallel Split", "x": 360, "y": 0, "metadata": {"fanout": 2}},
        {"id": "task", "type": "task", "label": "Process Request", "x": 540, "y": -100, "metadata": {"sla": "normal"}},
        {"id": "ai_agent", "type": "ai_agent", "label": "AI Screening", "x": 540, "y": 0, "metadata": {"model": "reviewer"}},
        {"id": "tool", "type": "tool", "label": "External Tool", "x": 540, "y": 100, "metadata": {"tool_name": "registry_check"}},
        {"id": "parallel_join", "type": "parallel_join", "label": "Parallel Join", "x": 760, "y": 0, "join": {"mode": "all"}, "metadata": {"join_policy": "all"}},
        {"id": "human_review", "type": "human_review", "label": "Manual Review", "x": 960, "y": 0, "metadata": {"owner": "ops"}},
        {"id": "notification", "type": "notification", "label": "Notify Applicant", "x": 1160, "y": -80, "metadata": {"channel": "email"}},
        {"id": "checkpoint", "type": "checkpoint", "label": "Checkpoint", "x": 1160, "y": 80, "metadata": {"checkpoint": "pre_archive"}},
        {"id": "subprocess", "type": "subprocess", "label": "Subprocess Call", "x": 1360, "y": 0, "metadata": {"subprocess_ref": "sp_review"}},
        {"id": "loop", "type": "loop", "label": "Retry Loop", "x": 1540, "y": 0, "metadata": {"max_iterations": 3}},
        {"id": "wait", "type": "wait", "label": "Wait for SLA", "x": 1720, "y": 0, "metadata": {"event": "sla_elapsed"}},
        {"id": "data_source", "type": "data_source", "label": "Case Database", "x": 360, "y": -220, "metadata": {"source": "cases"}},
        {"id": "document", "type": "document", "label": "Policy Document", "x": 540, "y": -220, "metadata": {"version": "2026.1"}},
        {"id": "storage", "type": "storage", "label": "Evidence Store", "x": 760, "y": -220, "metadata": {"bucket": "evidence"}},
        {"id": "variable", "type": "variable", "label": "Risk Score", "x": 960, "y": -220, "metadata": {"data_type": "number"}},
        {"id": "resource", "type": "resource", "label": "Reviewer Pool", "x": 1160, "y": -220, "metadata": {"capacity": 5}},
        {"id": "note", "type": "note", "label": "Operational Note", "x": 360, "y": 240, "metadata": {"text": "Reference only"}},
        {"id": "group", "type": "group", "label": "Review Group Marker", "x": 540, "y": 240, "metadata": {"group_ref": "group_review"}},
        {"id": "milestone", "type": "milestone", "label": "Ready Milestone", "x": 760, "y": 240, "metadata": {"milestone": "ready"}},
        {"id": "error_handler", "type": "error_handler", "label": "Error Handler", "x": 960, "y": 240, "metadata": {"handles": "failure"}},
        {"id": "end", "type": "end", "label": "End", "x": 1900, "y": 0, "metadata": {"role": "exit"}},
    ]
    if include_unknown:
        nodes.append({"id": "custom_audit", "type": "custom_audit", "label": "Custom Audit", "x": 1360, "y": 240, "metadata": {"vendor": "custom"}})
    edges = [
        {"id": "e1", "source": "start", "target": "decision", "label": ""},
        {"id": "e2", "source": "decision", "target": "parallel_split", "label": "success", "condition": "eligible == true"},
        {"id": "e3", "source": "decision", "target": "error_handler", "label": "failure", "condition": "eligible == false"},
        {"id": "e4", "source": "parallel_split", "target": "task", "label": ""},
        {"id": "e5", "source": "parallel_split", "target": "ai_agent", "label": ""},
        {"id": "e6", "source": "parallel_split", "target": "tool", "label": ""},
        {"id": "e7", "source": "task", "target": "parallel_join", "label": ""},
        {"id": "e8", "source": "ai_agent", "target": "parallel_join", "label": ""},
        {"id": "e9", "source": "tool", "target": "parallel_join", "label": ""},
        {"id": "e10", "source": "parallel_join", "target": "human_review", "label": ""},
        {"id": "e11", "source": "human_review", "target": "notification", "label": "success"},
        {"id": "e12", "source": "human_review", "target": "checkpoint", "label": "failure"},
        {"id": "e13", "source": "checkpoint", "target": "loop", "label": "retry"},
        {"id": "e14", "source": "loop", "target": "wait", "label": ""},
        {"id": "e15", "source": "wait", "target": "subprocess", "label": ""},
        {"id": "e16", "source": "subprocess", "target": "end", "label": ""},
        {"id": "e17", "source": "data_source", "target": "task", "label": "data"},
        {"id": "e18", "source": "document", "target": "ai_agent", "label": "context"},
        {"id": "e19", "source": "tool", "target": "storage", "label": "writes"},
        {"id": "e20", "source": "resource", "target": "human_review", "label": "assignment"},
        {"id": "e21", "source": "note", "target": "group", "label": "annotation"},
        {"id": "e22", "source": "milestone", "target": "end", "label": "marker"},
    ]
    groups = [{"id": "group_review", "label": "Review Group", "node_ids": ["task", "ai_agent", "tool", "parallel_join", "human_review"]}]
    return {"nodes": nodes, "edges": edges, "groups": groups}


def nodes_by_id(graph: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    if not isinstance(graph, dict):
        return {}
    return {node["id"]: node for node in graph.get("nodes", []) if isinstance(node, dict) and isinstance(node.get("id"), str)}


def edges_by_id(graph: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    if not isinstance(graph, dict):
        return {}
    return {edge["id"]: edge for edge in graph.get("edges", []) if isinstance(edge, dict) and isinstance(edge.get("id"), str)}


def graph_unchanged_except_nodes(before: dict[str, Any], after: dict[str, Any] | None, allowed_node_ids: set[str]) -> bool:
    if not isinstance(after, dict):
        return False
    before_nodes = nodes_by_id(before)
    after_nodes = nodes_by_id(after)
    if set(before_nodes) != set(after_nodes):
        return False
    for node_id in set(before_nodes) - allowed_node_ids:
        if before_nodes[node_id] != after_nodes[node_id]:
            return False
    return before.get("edges", []) == after.get("edges", []) and before.get("groups", []) == after.get("groups", [])


def catalog_types_present(graph: dict[str, Any]) -> bool:
    present = {node.get("type") for node in graph.get("nodes", []) if isinstance(node, dict)}
    return set(CATALOG_TYPES) <= present


def catalog_types_preserved(before: dict[str, Any], after: dict[str, Any] | None) -> bool:
    if not isinstance(after, dict):
        return True
    before_nodes = nodes_by_id(before)
    after_nodes = nodes_by_id(after)
    if set(before_nodes) != set(after_nodes):
        return False
    return all(before_nodes[node_id].get("type") == after_nodes[node_id].get("type") for node_id in before_nodes)


def type_metadata_preserved(before: dict[str, Any], after: dict[str, Any] | None, ignored_node_ids: set[str] | None = None) -> bool:
    if not isinstance(after, dict):
        return True
    ignored = ignored_node_ids or set()
    before_nodes = nodes_by_id(before)
    after_nodes = nodes_by_id(after)
    for node_id in set(before_nodes) - ignored:
        before_node = before_nodes[node_id]
        after_node = after_nodes.get(node_id, {})
        for key, value in before_node.items():
            if key in {"label", "x", "y"}:
                continue
            if after_node.get(key) != value:
                return False
    return True


def structural_relations_preserved(before: dict[str, Any], after: dict[str, Any] | None) -> bool:
    if not isinstance(after, dict):
        return True
    before_nodes = nodes_by_id(before)
    structural_ids = {node_id for node_id, node in before_nodes.items() if node.get("type") in STRUCTURAL_TYPES}
    before_edges = edges_by_id(before)
    after_edges = edges_by_id(after)
    for edge_id, edge in before_edges.items():
        if edge.get("source") in structural_ids or edge.get("target") in structural_ids:
            if after_edges.get(edge_id) != edge:
                return False
    return True


def changed_node_ids(before: dict[str, Any], after: dict[str, Any] | None) -> list[str]:
    if not isinstance(after, dict):
        return []
    before_nodes = nodes_by_id(before)
    after_nodes = nodes_by_id(after)
    return sorted(node_id for node_id in set(before_nodes) | set(after_nodes) if before_nodes.get(node_id) != after_nodes.get(node_id))


def changed_edge_ids(before: dict[str, Any], after: dict[str, Any] | None) -> list[str]:
    if not isinstance(after, dict):
        return []
    before_edges = edges_by_id(before)
    after_edges = edges_by_id(after)
    return sorted(edge_id for edge_id in set(before_edges) | set(after_edges) if before_edges.get(edge_id) != after_edges.get(edge_id))


def graph_unchanged(before: dict[str, Any], after: dict[str, Any] | None) -> bool:
    return isinstance(after, dict) and before == after


def first_deviation_stage(result: Any, correct_execution: bool) -> str | None:
    if correct_execution:
        return None
    specification = result.specification if isinstance(result.specification, dict) else {}
    if specification.get("stop_layer"):
        return specification["stop_layer"]
    return {
        "needs_clarification": "Ground",
        "unsupported": "Obligation",
        "interpretation_error": "Understand",
        "planning_error": "Planner",
        "execution_error": "Executor",
        "success": "Evaluator",
    }.get(result.status)


def evaluate_catalog_preservation(before: dict[str, Any], result: Any) -> tuple[bool, dict[str, Any]]:
    after = result.final_graph
    nodes = nodes_by_id(after)
    correct = (
        result.status == "success"
        and nodes.get("notification", {}).get("label") == "Applicant Notification"
        and graph_unchanged_except_nodes(before, after, {"notification"})
        and catalog_types_preserved(before, after)
        and type_metadata_preserved(before, after, {"notification"})
        and structural_relations_preserved(before, after)
    )
    return correct, {}


def evaluate_rename(node_id: str, expected_label: str) -> Callable[[dict[str, Any], Any], tuple[bool, dict[str, Any]]]:
    def evaluator(before: dict[str, Any], result: Any) -> tuple[bool, dict[str, Any]]:
        after = result.final_graph
        nodes = nodes_by_id(after)
        correct = (
            result.status == "success"
            and nodes.get(node_id, {}).get("label") == expected_label
            and graph_unchanged_except_nodes(before, after, {node_id})
            and catalog_types_preserved(before, after)
            and type_metadata_preserved(before, after, {node_id})
            and structural_relations_preserved(before, after)
        )
        return correct, {}

    return evaluator


def evaluate_move(node_id: str, dx: int) -> Callable[[dict[str, Any], Any], tuple[bool, dict[str, Any]]]:
    def evaluator(before: dict[str, Any], result: Any) -> tuple[bool, dict[str, Any]]:
        after = result.final_graph
        before_node = nodes_by_id(before).get(node_id, {})
        after_node = nodes_by_id(after).get(node_id, {})
        correct = (
            result.status == "success"
            and after_node.get("x") == before_node.get("x") + dx
            and after_node.get("y") == before_node.get("y")
            and graph_unchanged_except_nodes(before, after, {node_id})
            and catalog_types_preserved(before, after)
            and type_metadata_preserved(before, after, {node_id})
            and structural_relations_preserved(before, after)
        )
        return correct, {}

    return evaluator


def evaluate_structural_preservation(before: dict[str, Any], result: Any) -> tuple[bool, dict[str, Any]]:
    after = result.final_graph
    nodes = nodes_by_id(after)
    correct = (
        result.status == "success"
        and nodes.get("resource", {}).get("label") == "Reviewer Capacity Pool"
        and graph_unchanged_except_nodes(before, after, {"resource"})
        and catalog_types_preserved(before, after)
        and type_metadata_preserved(before, after, {"resource"})
        and structural_relations_preserved(before, after)
    )
    return correct, {}


def evaluate_unknown_type_preservation(before: dict[str, Any], result: Any) -> tuple[bool, dict[str, Any]]:
    after = result.final_graph
    nodes = nodes_by_id(after)
    contract = "unknown node types are accepted and preserved" if "custom_audit" in nodes_by_id(before) else "unknown node types absent"
    correct = (
        result.status == "success"
        and nodes.get("task", {}).get("label") == "Reviewed Task"
        and nodes.get("custom_audit", {}) == nodes_by_id(before).get("custom_audit", {})
        and graph_unchanged_except_nodes(before, after, {"task"})
        and catalog_types_preserved(before, after)
        and type_metadata_preserved(before, after, {"task"})
        and structural_relations_preserved(before, after)
    )
    return correct, {"unknown_type_contract": contract}


def transaction_payload(result: Any) -> dict[str, Any]:
    return {
        "status": result.status,
        "error": result.error,
        "planning_path": result.planning_path,
        "steps": result.steps,
        "final_graph": result.final_graph,
        "target": result.target,
        "graph_context": result.graph_context,
        "specification": result.specification,
        "mutation_authorization": result.mutation_authorization,
    }


def run_case(root: Path, case: CatalogCase) -> dict[str, Any]:
    graph = deepcopy(case.graph)
    case_dir = root / case.name
    case_dir.mkdir(parents=True, exist_ok=True)
    result = run_transaction(deepcopy(graph), 1, deepcopy(case.selection), case.instruction)
    correct_execution, details = case.evaluator(graph, result)
    final_graph = result.final_graph
    invalid_graph = isinstance(final_graph, dict) and not graph_is_valid(final_graph)
    false_success = result.status == "success" and graph_unchanged(graph, final_graph) and not correct_execution
    unauthorized_change = bool(result.error and "authorized" in str(result.error).lower())
    unintended_change = result.status == "success" and not correct_execution and not graph_unchanged(graph, final_graph)
    row = {
        "case": case.name,
        "instruction": case.instruction,
        "selection": case.selection,
        "status": result.status,
        "catalog_types_present": catalog_types_present(graph),
        "catalog_types_preserved": catalog_types_preserved(graph, final_graph),
        "type_metadata_preserved": type_metadata_preserved(graph, final_graph, set(changed_node_ids(graph, final_graph))),
        "structural_relations_preserved": structural_relations_preserved(graph, final_graph),
        "correct_execution": bool(correct_execution and not invalid_graph and not unauthorized_change and not unintended_change),
        "false_success": false_success,
        "invalid_graph": invalid_graph,
        "unauthorized_change": unauthorized_change,
        "unintended_change": unintended_change,
        "first_deviation_stage": first_deviation_stage(result, correct_execution),
        "changed_nodes": changed_node_ids(graph, final_graph),
        "changed_edges": changed_edge_ids(graph, final_graph),
        "details": details,
    }
    save_json(case_dir / "input_graph.json", graph)
    save_json(case_dir / "transaction_result.json", transaction_payload(result))
    save_json(case_dir / "record.json", row)
    return row


def main() -> None:
    graph = catalog_graph()
    unknown_graph = catalog_graph(include_unknown=True)
    cases = [
        CatalogCase(
            "N1_catalog_ingestion_preservation",
            graph,
            "Rename notification to Applicant Notification.",
            {"node_ids": [], "edge_ids": []},
            evaluate_catalog_preservation,
        ),
        CatalogCase(
            "N2_task_rename",
            graph,
            "Rename task to Reviewed Task.",
            {"node_ids": [], "edge_ids": []},
            evaluate_rename("task", "Reviewed Task"),
        ),
        CatalogCase(
            "N3_human_review_move",
            graph,
            "Move human_review 80 pixels to the right.",
            {"node_ids": [], "edge_ids": []},
            evaluate_move("human_review", 80),
        ),
        CatalogCase(
            "N4_tool_property_edit",
            graph,
            "Rename tool to Registry Verification Tool.",
            {"node_ids": [], "edge_ids": []},
            evaluate_rename("tool", "Registry Verification Tool"),
        ),
        CatalogCase(
            "N5_note_rename",
            graph,
            "Rename note to Reviewer Guidance Note.",
            {"node_ids": [], "edge_ids": []},
            evaluate_rename("note", "Reviewer Guidance Note"),
        ),
        CatalogCase(
            "N6_structural_types_preservation",
            graph,
            "Rename resource to Reviewer Capacity Pool.",
            {"node_ids": [], "edge_ids": []},
            evaluate_structural_preservation,
        ),
        CatalogCase(
            "N7_unknown_type_preservation",
            unknown_graph,
            "Rename task to Reviewed Task.",
            {"node_ids": [], "edge_ids": []},
            evaluate_unknown_type_preservation,
        ),
    ]
    root = Path("experiments/runs") / f"stage3-node-type-catalog-acceptance-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    root.mkdir(parents=True, exist_ok=False)
    rows = [run_case(root, case) for case in cases]
    summary = {
        "root": str(root),
        "catalog_types": CATALOG_TYPES,
        "fixture": {
            "node_count": len(graph["nodes"]),
            "edge_count": len(graph["edges"]),
            "unknown_type_case_node_count": len(unknown_graph["nodes"]),
        },
        "totals": {
            "total": len(rows),
            "catalog_types_present": sum(bool(row["catalog_types_present"]) for row in rows),
            "catalog_types_preserved": sum(bool(row["catalog_types_preserved"]) for row in rows),
            "type_metadata_preserved": sum(bool(row["type_metadata_preserved"]) for row in rows),
            "structural_relations_preserved": sum(bool(row["structural_relations_preserved"]) for row in rows),
            "correct_execution": sum(bool(row["correct_execution"]) for row in rows),
            "false_success": sum(bool(row["false_success"]) for row in rows),
            "invalid_graph": sum(bool(row["invalid_graph"]) for row in rows),
            "unauthorized_change": sum(bool(row["unauthorized_change"]) for row in rows),
            "unintended_change": sum(bool(row["unintended_change"]) for row in rows),
        },
        "first_deviation_stages": {
            row["case"]: row["first_deviation_stage"]
            for row in rows
            if not row["correct_execution"]
        },
        "cases": rows,
    }
    save_json(root / "summary.json", summary)
    print(json.dumps({"root": str(root), "totals": summary["totals"], "first_deviation_stages": summary["first_deviation_stages"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
