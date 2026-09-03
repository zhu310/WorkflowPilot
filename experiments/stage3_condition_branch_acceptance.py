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


@dataclass(frozen=True)
class ConditionBranchCase:
    name: str
    instruction: str
    selection: dict[str, Any]
    evaluator: Callable[[dict[str, Any], Any], tuple[bool, bool, dict[str, Any]]]


def fixture_graph() -> dict[str, Any]:
    return {
        "nodes": [
            {"id": "start", "type": "start", "label": "Start", "x": 0, "y": 0, "width": 120, "height": 60},
            {"id": "collect", "type": "task", "label": "Collect Application", "x": 180, "y": 0, "width": 150, "height": 60},
            {"id": "decision", "type": "decision", "label": "Eligibility Decision", "x": 390, "y": 0, "width": 150, "height": 70},
            {"id": "auto_approve", "type": "task", "label": "Auto Approve", "x": 620, "y": -120, "width": 140, "height": 60},
            {"id": "manual_review", "type": "human_review", "label": "Manual Review", "x": 620, "y": 0, "width": 140, "height": 60},
            {"id": "compliance_review", "type": "human_review", "label": "Compliance Review", "x": 620, "y": 120, "width": 160, "height": 60},
            {"id": "audit_review", "type": "human_review", "label": "Audit Review", "x": 850, "y": -120, "width": 140, "height": 60},
            {"id": "retry_queue", "type": "task", "label": "Retry Queue", "x": 850, "y": 120, "width": 140, "height": 60},
            {"id": "archive", "type": "task", "label": "Archive", "x": 1060, "y": 0, "width": 120, "height": 60},
            {"id": "end", "type": "end", "label": "End", "x": 1240, "y": 0, "width": 120, "height": 60},
        ],
        "edges": [
            {"id": "e_start_collect", "source": "start", "target": "collect", "label": ""},
            {"id": "e_collect_decision", "source": "collect", "target": "decision", "label": ""},
            {
                "id": "e_decision_auto",
                "source": "decision",
                "target": "auto_approve",
                "label": "success",
                "condition": {"expression": "amount <= 1000", "language": "cel"},
                "branch_label": "low_amount",
            },
            {
                "id": "e_decision_manual",
                "source": "decision",
                "target": "manual_review",
                "label": "failure",
                "condition": {"expression": "amount > 1000", "language": "cel"},
                "branch_label": "high_amount",
            },
            {
                "id": "e_decision_compliance",
                "source": "decision",
                "target": "compliance_review",
                "label": "exception",
                "condition": {"expression": "country in restricted", "language": "cel"},
                "branch_label": "restricted_country",
            },
            {
                "id": "e_decision_audit",
                "source": "decision",
                "target": "audit_review",
                "label": "",
                "condition": {"expression": "requires_audit == true", "language": "cel"},
                "branch_label": "audit_required",
            },
            {"id": "e_auto_archive", "source": "auto_approve", "target": "archive", "label": ""},
            {"id": "e_manual_archive", "source": "manual_review", "target": "archive", "label": ""},
            {"id": "e_compliance_retry", "source": "compliance_review", "target": "retry_queue", "label": "retry"},
            {"id": "e_retry_decision", "source": "retry_queue", "target": "decision", "label": "retry"},
            {"id": "e_audit_archive", "source": "audit_review", "target": "archive", "label": ""},
            {"id": "e_archive_end", "source": "archive", "target": "end", "label": ""},
        ],
    }


def nodes_by_id(graph: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    if not isinstance(graph, dict):
        return {}
    return {node["id"]: node for node in graph.get("nodes", []) if isinstance(node, dict) and isinstance(node.get("id"), str)}


def edges_by_id(graph: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    if not isinstance(graph, dict):
        return {}
    return {edge["id"]: edge for edge in graph.get("edges", []) if isinstance(edge, dict) and isinstance(edge.get("id"), str)}


def edge_key(edge: dict[str, Any]) -> tuple[str, str, str]:
    return str(edge.get("source")), str(edge.get("target")), str(edge.get("label", ""))


def full_edge_key(edge: dict[str, Any]) -> str:
    return json.dumps(edge, ensure_ascii=False, sort_keys=True)


def edge_set(graph: dict[str, Any] | None) -> set[tuple[str, str, str]]:
    if not isinstance(graph, dict):
        return set()
    return {edge_key(edge) for edge in graph.get("edges", []) if isinstance(edge, dict)}


def conditional_edge_ids(graph: dict[str, Any]) -> set[str]:
    return {
        edge["id"]
        for edge in graph.get("edges", [])
        if isinstance(edge, dict) and isinstance(edge.get("id"), str) and ("condition" in edge or "branch_label" in edge)
    }


def condition_and_branch_preserved(
    before: dict[str, Any],
    after: dict[str, Any] | None,
    ignored_edge_ids: set[str] | None = None,
) -> bool:
    if after is None:
        return True
    if not isinstance(after, dict):
        return False
    ignored = ignored_edge_ids or set()
    before_edges = edges_by_id(before)
    after_edges = edges_by_id(after)
    for edge_id in conditional_edge_ids(before) - ignored:
        if before_edges.get(edge_id) != after_edges.get(edge_id):
            return False
    return True


def graph_unchanged(before: dict[str, Any], after: dict[str, Any] | None) -> bool:
    return isinstance(after, dict) and before == after


def changed_nodes(before: dict[str, Any], after: dict[str, Any] | None) -> list[str]:
    if not isinstance(after, dict):
        return []
    before_nodes = nodes_by_id(before)
    after_nodes = nodes_by_id(after)
    return sorted(node_id for node_id in set(before_nodes) | set(after_nodes) if before_nodes.get(node_id) != after_nodes.get(node_id))


def changed_edges(before: dict[str, Any], after: dict[str, Any] | None) -> dict[str, list[Any]]:
    if not isinstance(after, dict):
        return {"added_relation_keys": [], "removed_relation_keys": [], "changed_existing_edge_ids": []}
    before_edges = edges_by_id(before)
    after_edges = edges_by_id(after)
    changed_existing = sorted(edge_id for edge_id in set(before_edges) & set(after_edges) if before_edges[edge_id] != after_edges[edge_id])
    return {
        "added_relation_keys": [list(row) for row in sorted(edge_set(after) - edge_set(before))],
        "removed_relation_keys": [list(row) for row in sorted(edge_set(before) - edge_set(after))],
        "changed_existing_edge_ids": changed_existing,
    }


def edge_matching(graph: dict[str, Any] | None, source: str, target: str, label: str) -> list[dict[str, Any]]:
    if not isinstance(graph, dict):
        return []
    return [
        edge
        for edge in graph.get("edges", [])
        if isinstance(edge, dict)
        and edge.get("source") == source
        and edge.get("target") == target
        and edge.get("label", "") == label
    ]


def sibling_branches_unchanged(before: dict[str, Any], after: dict[str, Any] | None, ignored_edge_id: str) -> bool:
    if not isinstance(after, dict):
        return False
    before_edges = edges_by_id(before)
    after_edges = edges_by_id(after)
    for edge in before.get("edges", []):
        if edge.get("source") != "decision" or edge.get("id") == ignored_edge_id:
            continue
        if before_edges.get(edge["id"]) != after_edges.get(edge["id"]):
            return False
    return True


def clarification_ok(result: Any) -> bool:
    specification = result.specification if isinstance(result.specification, dict) else {}
    clarification = specification.get("clarification", {}) if isinstance(specification, dict) else {}
    return result.status == "needs_clarification" and bool(clarification.get("question"))


def first_deviation_stage(result: Any, correct_execution: bool, correct_clarification: bool) -> str | None:
    if correct_execution or correct_clarification:
        return None
    specification = result.specification if isinstance(result.specification, dict) else {}
    if isinstance(specification, dict) and specification.get("stop_layer"):
        return specification["stop_layer"]
    return {
        "needs_clarification": "Ground",
        "unsupported": "Obligation",
        "interpretation_error": "Understand",
        "planning_error": "Planner",
        "execution_error": "Executor",
        "success": "Evaluator",
    }.get(result.status)


def evaluate_condition_preservation(before: dict[str, Any], result: Any) -> tuple[bool, bool, dict[str, Any]]:
    after = result.final_graph
    nodes = nodes_by_id(after)
    correct = (
        result.status == "success"
        and isinstance(after, dict)
        and nodes.get("archive", {}).get("label") == "Completed Archive"
        and changed_nodes(before, after) == ["archive"]
        and edge_set(before) == edge_set(after)
        and condition_and_branch_preserved(before, after)
    )
    return correct, False, {"changed_nodes": changed_nodes(before, after), "changed_edges": changed_edges(before, after)}


def evaluate_labeled_branch_redirect(before: dict[str, Any], result: Any) -> tuple[bool, bool, dict[str, Any]]:
    after = result.final_graph
    original = edges_by_id(before)["e_decision_auto"]
    redirected = edge_matching(after, "decision", "audit_review", "success")
    redirect_condition_preserved = bool(redirected) and redirected[0].get("condition") == original.get("condition") and redirected[0].get("branch_label") == original.get("branch_label")
    correct = (
        result.status == "success"
        and isinstance(after, dict)
        and not edge_matching(after, "decision", "auto_approve", "success")
        and len(redirected) == 1
        and redirect_condition_preserved
        and sibling_branches_unchanged(before, after, "e_decision_auto")
        and nodes_by_id(before) == nodes_by_id(after)
    )
    return correct, False, {
        "redirected_edge": redirected[0] if redirected else None,
        "expected_preserved_condition": original.get("condition"),
        "expected_preserved_branch_label": original.get("branch_label"),
        "redirect_condition_preserved": redirect_condition_preserved,
        "sibling_branches_unchanged": sibling_branches_unchanged(before, after, "e_decision_auto"),
        "changed_edges": changed_edges(before, after),
    }


def evaluate_ambiguous_branch_reference(before: dict[str, Any], result: Any) -> tuple[bool, bool, dict[str, Any]]:
    correct_clarification = clarification_ok(result) and (result.final_graph is None or graph_unchanged(before, result.final_graph))
    return False, correct_clarification, {"graph_unchanged": result.final_graph is None or graph_unchanged(before, result.final_graph)}


def evaluate_unrelated_topology_edit(before: dict[str, Any], result: Any) -> tuple[bool, bool, dict[str, Any]]:
    after = result.final_graph
    expected_edge_keys = {
        ("start", "collect", ""),
        ("collect", "decision", ""),
        ("decision", "auto_approve", "success"),
        ("decision", "manual_review", "failure"),
        ("decision", "compliance_review", "exception"),
        ("decision", "audit_review", ""),
        ("auto_approve", "archive", ""),
        ("manual_review", "archive", ""),
        ("compliance_review", "retry_queue", "retry"),
        ("retry_queue", "decision", "retry"),
        ("audit_review", "archive", ""),
        ("archive", "end", ""),
    }
    actual_edge_keys = edge_set(after)
    nodes = nodes_by_id(after)
    correct = (
        result.status == "success"
        and isinstance(after, dict)
        and nodes.get("collect", {}).get("x") == 300
        and nodes.get("collect", {}).get("y") == 0
        and expected_edge_keys == actual_edge_keys
        and condition_and_branch_preserved(before, after)
    )
    return correct, False, {"changed_nodes": changed_nodes(before, after), "changed_edges": changed_edges(before, after)}


def evaluate_mixed_branch_types(before: dict[str, Any], result: Any) -> tuple[bool, bool, dict[str, Any]]:
    after = result.final_graph
    original_failure = edges_by_id(before)["e_decision_manual"]
    redirected_failure = edge_matching(after, "decision", "audit_review", "failure")
    success_unchanged = edges_by_id(after).get("e_decision_auto") == edges_by_id(before).get("e_decision_auto") if isinstance(after, dict) else False
    exception_unchanged = edges_by_id(after).get("e_decision_compliance") == edges_by_id(before).get("e_decision_compliance") if isinstance(after, dict) else False
    unlabeled_unchanged = edges_by_id(after).get("e_decision_audit") == edges_by_id(before).get("e_decision_audit") if isinstance(after, dict) else False
    failure_condition_preserved = bool(redirected_failure) and redirected_failure[0].get("condition") == original_failure.get("condition") and redirected_failure[0].get("branch_label") == original_failure.get("branch_label")
    correct = (
        result.status == "success"
        and isinstance(after, dict)
        and len(redirected_failure) == 1
        and not edge_matching(after, "decision", "manual_review", "failure")
        and failure_condition_preserved
        and success_unchanged
        and exception_unchanged
        and unlabeled_unchanged
        and nodes_by_id(before) == nodes_by_id(after)
    )
    return correct, False, {
        "redirected_failure": redirected_failure[0] if redirected_failure else None,
        "failure_condition_preserved": failure_condition_preserved,
        "success_unchanged": success_unchanged,
        "exception_unchanged": exception_unchanged,
        "unlabeled_unchanged": unlabeled_unchanged,
        "changed_edges": changed_edges(before, after),
    }


def evaluate_active_branch_label_edit(before: dict[str, Any], result: Any) -> tuple[bool, bool, dict[str, Any]]:
    after = result.final_graph
    before_edges = edges_by_id(before)
    after_edges = edges_by_id(after)
    target_before = before_edges.get("e_decision_auto", {})
    target_after = after_edges.get("e_decision_auto", {})
    sibling_unchanged = sibling_branches_unchanged(before, after, "e_decision_auto")
    target_identity_preserved = all(target_after.get(key) == target_before.get(key) for key in ("id", "source", "target", "label"))
    condition_preserved = target_after.get("condition") == target_before.get("condition")
    correct = (
        result.status == "success"
        and isinstance(after, dict)
        and target_after.get("branch_label") == "approved"
        and target_identity_preserved
        and condition_preserved
        and sibling_unchanged
        and nodes_by_id(before) == nodes_by_id(after)
    )
    return correct, False, {
        "target_edge": target_after,
        "target_identity_preserved": target_identity_preserved,
        "condition_metadata_preserved": condition_preserved,
        "branch_label_metadata_preserved": correct,
        "sibling_branches_unchanged": sibling_unchanged,
        "changed_edges": changed_edges(before, after),
    }


def evaluate_active_condition_expression_edit(before: dict[str, Any], result: Any) -> tuple[bool, bool, dict[str, Any]]:
    after = result.final_graph
    before_edges = edges_by_id(before)
    after_edges = edges_by_id(after)
    target_before = before_edges.get("e_decision_auto", {})
    target_after = after_edges.get("e_decision_auto", {})
    sibling_unchanged = sibling_branches_unchanged(before, after, "e_decision_auto")
    target_identity_preserved = all(target_after.get(key) == target_before.get(key) for key in ("id", "source", "target", "label"))
    branch_label_preserved = target_after.get("branch_label") == target_before.get("branch_label")
    expected_condition = {"expression": "amount <= 5000", "language": "cel"}
    correct = (
        result.status == "success"
        and isinstance(after, dict)
        and target_after.get("condition") == expected_condition
        and target_identity_preserved
        and branch_label_preserved
        and sibling_unchanged
        and nodes_by_id(before) == nodes_by_id(after)
    )
    return correct, False, {
        "target_edge": target_after,
        "expected_condition": expected_condition,
        "target_identity_preserved": target_identity_preserved,
        "condition_metadata_preserved": correct,
        "branch_label_metadata_preserved": branch_label_preserved,
        "sibling_branches_unchanged": sibling_unchanged,
        "changed_edges": changed_edges(before, after),
    }


def evaluate_ambiguous_condition_edit(before: dict[str, Any], result: Any) -> tuple[bool, bool, dict[str, Any]]:
    correct_clarification = clarification_ok(result) and (result.final_graph is None or graph_unchanged(before, result.final_graph))
    return False, correct_clarification, {"graph_unchanged": result.final_graph is None or graph_unchanged(before, result.final_graph)}


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


def run_case(root: Path, case: ConditionBranchCase) -> dict[str, Any]:
    graph = fixture_graph()
    case_dir = root / case.name
    case_dir.mkdir(parents=True, exist_ok=True)
    result = run_transaction(deepcopy(graph), 1, deepcopy(case.selection), case.instruction)
    correct_execution, correct_clarification, details = case.evaluator(graph, result)
    final_graph = result.final_graph
    condition_preserved = condition_and_branch_preserved(graph, final_graph) or bool(
        details.get("redirect_condition_preserved") or details.get("failure_condition_preserved") or details.get("condition_metadata_preserved")
    )
    branch_label_preserved = condition_and_branch_preserved(graph, final_graph) or bool(
        details.get("redirect_condition_preserved") or details.get("failure_condition_preserved") or details.get("branch_label_metadata_preserved")
    )
    invalid_graph = isinstance(final_graph, dict) and not graph_is_valid(final_graph)
    false_success = result.status == "success" and graph_unchanged(graph, final_graph) and not correct_execution
    unauthorized_change = bool(result.error and "authorized" in str(result.error).lower())
    unintended_change = result.status == "success" and not correct_execution and not graph_unchanged(graph, final_graph)
    row = {
        "case": case.name,
        "instruction": case.instruction,
        "selection": case.selection,
        "status": result.status,
        "condition_preserved": condition_preserved,
        "branch_label_preserved": branch_label_preserved,
        "correct_execution": correct_execution,
        "correct_clarification": correct_clarification,
        "false_success": false_success,
        "invalid_graph": invalid_graph,
        "unauthorized_change": unauthorized_change,
        "unintended_change": unintended_change,
        "first_deviation_stage": first_deviation_stage(result, correct_execution, correct_clarification),
        "changed_nodes": changed_nodes(graph, final_graph),
        "changed_edges": changed_edges(graph, final_graph),
        "details": details,
    }
    save_json(case_dir / "input_graph.json", graph)
    save_json(case_dir / "transaction_result.json", transaction_payload(result))
    save_json(case_dir / "record.json", row)
    return row


def main() -> None:
    cases = [
        ConditionBranchCase(
            "C1_condition_preservation_unrelated_rename",
            "Rename archive to Completed Archive.",
            {"node_ids": [], "edge_ids": []},
            evaluate_condition_preservation,
        ),
        ConditionBranchCase(
            "C2_labeled_condition_branch_redirect",
            "Redirect the selected success branch to Audit Review.",
            {"node_ids": [], "edge_ids": ["e_decision_auto"]},
            evaluate_labeled_branch_redirect,
        ),
        ConditionBranchCase(
            "C3_ambiguous_branch_reference",
            "Redirect this branch to Audit Review.",
            {"node_ids": [], "edge_ids": []},
            evaluate_ambiguous_branch_reference,
        ),
        ConditionBranchCase(
            "C4_unrelated_topology_move",
            "Move collect 120 pixels to the right.",
            {"node_ids": [], "edge_ids": []},
            evaluate_unrelated_topology_edit,
        ),
        ConditionBranchCase(
            "C5_mixed_labeled_condition_branches",
            "Redirect the failure branch from Eligibility Decision to Audit Review.",
            {"node_ids": [], "edge_ids": []},
            evaluate_mixed_branch_types,
        ),
        ConditionBranchCase(
            "C6_branch_label_active_edit",
            "Rename selected branch label to approved.",
            {"node_ids": [], "edge_ids": ["e_decision_auto"]},
            evaluate_active_branch_label_edit,
        ),
        ConditionBranchCase(
            "C7_condition_expression_explicit_replacement",
            "Replace selected branch condition expression with amount <= 5000.",
            {"node_ids": [], "edge_ids": ["e_decision_auto"]},
            evaluate_active_condition_expression_edit,
        ),
        ConditionBranchCase(
            "C8_ambiguous_condition_edit_clarification",
            "Replace the condition expression from Eligibility Decision with amount <= 5000.",
            {"node_ids": [], "edge_ids": []},
            evaluate_ambiguous_condition_edit,
        ),
    ]
    root = Path("experiments/runs") / f"stage3-condition-branch-acceptance-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    root.mkdir(parents=True, exist_ok=False)
    rows = [run_case(root, case) for case in cases]
    summary = {
        "root": str(root),
        "fixture": {
            "node_count": len(fixture_graph()["nodes"]),
            "edge_count": len(fixture_graph()["edges"]),
            "conditional_branch_edge_ids": sorted(conditional_edge_ids(fixture_graph())),
        },
        "totals": {
            "total": len(rows),
            "condition_preserved": sum(bool(row["condition_preserved"]) for row in rows),
            "branch_label_preserved": sum(bool(row["branch_label_preserved"]) for row in rows),
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
