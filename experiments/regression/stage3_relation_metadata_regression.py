from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from src.mvp23 import run_transaction
from src.runner import save_json
from src.workflow_ir_core import graph_is_valid


@dataclass(frozen=True)
class RelationCase:
    name: str
    edge_id: str | None
    instruction: str
    source: str
    old_target: str
    new_target: str
    label: str
    expect_clarification: bool = False


def fixture_graph() -> dict[str, Any]:
    nodes = [
        {"id": "src_unlabeled", "type": "task", "label": "Unlabeled Source", "x": 0, "y": 0},
        {"id": "src_success", "type": "decision", "label": "Success Source", "x": 0, "y": 100},
        {"id": "src_failure", "type": "decision", "label": "Failure Source", "x": 0, "y": 200},
        {"id": "src_retry", "type": "task", "label": "Retry Source", "x": 0, "y": 300},
        {"id": "src_exception", "type": "task", "label": "Exception Source", "x": 0, "y": 400},
        {"id": "old_unlabeled", "type": "task", "label": "Old Unlabeled", "x": 220, "y": 0},
        {"id": "old_success", "type": "task", "label": "Old Success", "x": 220, "y": 100},
        {"id": "old_failure", "type": "task", "label": "Old Failure", "x": 220, "y": 200},
        {"id": "old_retry", "type": "task", "label": "Old Retry", "x": 220, "y": 300},
        {"id": "old_exception", "type": "task", "label": "Old Exception", "x": 220, "y": 400},
        {"id": "new_target", "type": "human_review", "label": "Manual Review", "x": 460, "y": 200},
        {"id": "sibling", "type": "task", "label": "Sibling", "x": 460, "y": 20},
    ]
    edges = [
        {"id": "e_unlabeled", "source": "src_unlabeled", "target": "old_unlabeled", "label": "", "metadata": {"priority": "normal"}},
        {"id": "e_success", "source": "src_success", "target": "old_success", "label": "success", "condition": {"expression": "ok"}, "branch_label": "ok_branch"},
        {"id": "e_failure", "source": "src_failure", "target": "old_failure", "label": "failure", "condition": {"expression": "failed"}, "branch_label": "failed_branch"},
        {"id": "e_retry", "source": "src_retry", "target": "old_retry", "label": "retry", "metadata": {"max": 3}},
        {"id": "e_exception", "source": "src_exception", "target": "old_exception", "label": "exception", "metadata": {"handler": "default"}},
        {"id": "e_sibling_success", "source": "src_success", "target": "sibling", "label": "failure", "condition": {"expression": "not_ok"}, "branch_label": "not_ok_branch"},
    ]
    return {"nodes": nodes, "edges": edges}


def edges_by_id(graph: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    if not isinstance(graph, dict):
        return {}
    return {edge["id"]: edge for edge in graph.get("edges", []) if isinstance(edge, dict) and isinstance(edge.get("id"), str)}


def nodes_by_id(graph: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    if not isinstance(graph, dict):
        return {}
    return {node["id"]: node for node in graph.get("nodes", []) if isinstance(node, dict) and isinstance(node.get("id"), str)}


def matching_edge(graph: dict[str, Any] | None, source: str, target: str, label: str) -> dict[str, Any] | None:
    if not isinstance(graph, dict):
        return None
    matches = [
        edge
        for edge in graph.get("edges", [])
        if edge.get("source") == source and edge.get("target") == target and edge.get("label", "") == label
    ]
    return matches[0] if len(matches) == 1 else None


def graph_unchanged(before: dict[str, Any], after: dict[str, Any] | None) -> bool:
    return isinstance(after, dict) and before == after


def edge_payload_without_endpoint(edge: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in edge.items() if key not in {"source", "target", "label"}}


def evaluate_case(graph: dict[str, Any], case: RelationCase, result: Any) -> tuple[bool, bool, dict[str, Any]]:
    if case.expect_clarification:
        ok = result.status == "needs_clarification" and (result.final_graph is None or graph_unchanged(graph, result.final_graph))
        return False, ok, {"graph_unchanged": result.final_graph is None or graph_unchanged(graph, result.final_graph)}
    after = result.final_graph
    old_edge = edges_by_id(graph)[case.edge_id or ""]
    redirected = matching_edge(after, case.source, case.new_target, case.label)
    old_gone = matching_edge(after, case.source, case.old_target, case.label) is None
    metadata_preserved = bool(redirected) and edge_payload_without_endpoint(redirected) == edge_payload_without_endpoint(old_edge)
    sibling_preserved = edges_by_id(after).get("e_sibling_success") == edges_by_id(graph).get("e_sibling_success") if isinstance(after, dict) else False
    nodes_preserved = nodes_by_id(after) == nodes_by_id(graph)
    correct = result.status == "success" and old_gone and metadata_preserved and sibling_preserved and nodes_preserved
    return correct, False, {
        "redirected_edge": redirected,
        "metadata_preserved": metadata_preserved,
        "old_edge_payload": edge_payload_without_endpoint(old_edge),
        "sibling_preserved": sibling_preserved,
    }


def first_deviation_stage(result: Any, correct_execution: bool, correct_clarification: bool) -> str | None:
    if correct_execution or correct_clarification:
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


def run_case(root: Path, case: RelationCase) -> dict[str, Any]:
    graph = fixture_graph()
    selection = {"node_ids": [], "edge_ids": [case.edge_id] if case.edge_id else []}
    result = run_transaction(deepcopy(graph), 1, deepcopy(selection), case.instruction)
    correct_execution, correct_clarification, details = evaluate_case(graph, case, result)
    final_graph = result.final_graph
    invalid_graph = isinstance(final_graph, dict) and not graph_is_valid(final_graph)
    false_success = result.status == "success" and graph_unchanged(graph, final_graph) and not correct_execution
    unintended_change = result.status == "success" and not correct_execution and not graph_unchanged(graph, final_graph)
    unauthorized_change = bool(result.error and "authorized" in str(result.error).lower())
    row = {
        "case": case.name,
        "instruction": case.instruction,
        "selection": selection,
        "status": result.status,
        "metadata_preserved": bool(details.get("metadata_preserved") or correct_clarification),
        "correct_execution": bool(correct_execution and not invalid_graph and not unauthorized_change and not unintended_change),
        "correct_clarification": bool(correct_clarification),
        "false_success": false_success,
        "invalid_graph": invalid_graph,
        "unauthorized_change": unauthorized_change,
        "unintended_change": unintended_change,
        "first_deviation_stage": first_deviation_stage(result, correct_execution, correct_clarification),
        "details": details,
    }
    case_dir = root / case.name
    case_dir.mkdir(parents=True, exist_ok=True)
    save_json(case_dir / "input_graph.json", graph)
    save_json(case_dir / "transaction_result.json", transaction_payload(result))
    save_json(case_dir / "record.json", row)
    return row


def main() -> None:
    cases = [
        RelationCase("R1_unlabeled_redirect", "e_unlabeled", "Redirect the selected edge to Manual Review.", "src_unlabeled", "old_unlabeled", "new_target", ""),
        RelationCase("R2_success_redirect", "e_success", "Redirect the selected success relation to Manual Review.", "src_success", "old_success", "new_target", "success"),
        RelationCase("R3_failure_redirect", "e_failure", "Redirect the selected failure relation to Manual Review.", "src_failure", "old_failure", "new_target", "failure"),
        RelationCase("R4_retry_redirect", "e_retry", "Redirect the selected retry relation to Manual Review.", "src_retry", "old_retry", "new_target", "retry"),
        RelationCase("R5_exception_redirect", "e_exception", "Redirect the selected exception relation to Manual Review.", "src_exception", "old_exception", "new_target", "exception"),
        RelationCase("R6_mixed_metadata_edge", "e_success", "Redirect the selected branch to Manual Review.", "src_success", "old_success", "new_target", "success"),
        RelationCase("R7_ambiguous_edge_reference", None, "Redirect this edge to Manual Review.", "", "", "", "", expect_clarification=True),
    ]
    root = Path("experiments/runs") / f"stage3-relation-metadata-regression-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    root.mkdir(parents=True, exist_ok=False)
    rows = [run_case(root, case) for case in cases]
    summary = {
        "root": str(root),
        "totals": {
            "total": len(rows),
            "metadata_preserved": sum(bool(row["metadata_preserved"]) for row in rows),
            "correct_execution": sum(bool(row["correct_execution"]) for row in rows),
            "correct_clarification": sum(bool(row["correct_clarification"]) for row in rows),
            "false_success": sum(bool(row["false_success"]) for row in rows),
            "invalid_graph": sum(bool(row["invalid_graph"]) for row in rows),
            "unauthorized_change": sum(bool(row["unauthorized_change"]) for row in rows),
            "unintended_change": sum(bool(row["unintended_change"]) for row in rows),
        },
        "first_deviation_stages": {
            row["case"]: row["first_deviation_stage"]
            for row in rows
            if not row["correct_execution"] and not row["correct_clarification"]
        },
        "cases": rows,
    }
    save_json(root / "summary.json", summary)
    print(json.dumps({"root": str(root), "totals": summary["totals"], "first_deviation_stages": summary["first_deviation_stages"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
