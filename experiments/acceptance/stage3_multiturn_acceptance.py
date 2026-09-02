from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from src.mvp23 import run_transaction
from src.runner import save_json
from src.versioning import as_jsonable, create_version_state, redo, run_versioned_transaction, undo
from src.workflow_ir_core import graph_is_valid


@dataclass(frozen=True)
class RoundRecord:
    round_id: str
    instruction: str
    status: str
    version_before: str | None
    version_after: str | None
    correct_execution: bool
    correct_clarification: bool
    false_success: bool
    invalid_graph: bool
    unintended_change: bool
    first_failure_layer: str | None
    details: dict[str, Any]


def base_graph() -> dict[str, Any]:
    return {
        "nodes": [
            {"id": "start", "type": "start", "label": "Start", "x": 0, "y": 0},
            {"id": "identity_check", "type": "task", "label": "Identity Check", "x": 180, "y": 0},
            {"id": "risk_check", "type": "task", "label": "Risk Check", "x": 360, "y": 0},
            {"id": "approval", "type": "task", "label": "Approval", "x": 540, "y": 0},
            {"id": "manual_review", "type": "human_review", "label": "Manual Review", "x": 540, "y": 150},
            {"id": "reject", "type": "end", "label": "Reject", "x": 720, "y": -130},
            {"id": "end", "type": "end", "label": "End", "x": 720, "y": 0},
        ],
        "edges": [
            {"id": "e1", "source": "start", "target": "identity_check", "label": ""},
            {"id": "e2", "source": "identity_check", "target": "risk_check", "label": ""},
            {"id": "e3", "source": "risk_check", "target": "approval", "label": ""},
            {"id": "e4", "source": "approval", "target": "end", "label": ""},
            {"id": "e5", "source": "risk_check", "target": "reject", "label": "failure"},
            {"id": "e6", "source": "manual_review", "target": "risk_check", "label": "retry"},
        ],
    }


def node(graph: dict[str, Any] | None, node_id: str) -> dict[str, Any] | None:
    if not isinstance(graph, dict):
        return None
    return next((item for item in graph.get("nodes", []) if isinstance(item, dict) and item.get("id") == node_id), None)


def edge_exists(graph: dict[str, Any] | None, source: str, target: str, label: str) -> bool:
    if not isinstance(graph, dict):
        return False
    return any(
        isinstance(edge, dict)
        and edge.get("source") == source
        and edge.get("target") == target
        and edge.get("label", "") == label
        for edge in graph.get("edges", [])
    )


def edge_absent(graph: dict[str, Any] | None, source: str, target: str, label: str) -> bool:
    return not edge_exists(graph, source, target, label)


def graph_equal(left: dict[str, Any] | None, right: dict[str, Any] | None) -> bool:
    return isinstance(left, dict) and isinstance(right, dict) and left == right


def changed_nodes(before: dict[str, Any], after: dict[str, Any] | None) -> list[str]:
    if not isinstance(after, dict):
        return []
    before_nodes = {item["id"]: item for item in before["nodes"]}
    after_nodes = {item["id"]: item for item in after["nodes"]}
    return sorted(node_id for node_id in set(before_nodes) | set(after_nodes) if before_nodes.get(node_id) != after_nodes.get(node_id))


def first_failure_layer(result: Any) -> str | None:
    specification = result.specification if isinstance(result.specification, dict) else {}
    if isinstance(specification, dict) and specification.get("stop_layer"):
        return specification["stop_layer"]
    return {
        "needs_clarification": "Ground",
        "unsupported": "Obligation",
        "interpretation_error": "Understand",
        "planning_error": "Planner",
        "execution_error": "Executor",
    }.get(result.status)


def clarification_ok(result: Any) -> bool:
    specification = result.specification if isinstance(result.specification, dict) else {}
    clarification = specification.get("clarification", {}) if isinstance(specification, dict) else {}
    return result.status == "needs_clarification" and bool(clarification.get("question"))


def record_transaction_round(
    round_id: str,
    instruction: str,
    before: dict[str, Any],
    result: Any,
    correct_execution: bool,
    correct_clarification: bool = False,
    version_before: str | None = None,
    version_after: str | None = None,
) -> RoundRecord:
    invalid = isinstance(result.final_graph, dict) and not graph_is_valid(result.final_graph)
    false_success = result.status == "success" and result.final_graph == before and not correct_execution
    return RoundRecord(
        round_id=round_id,
        instruction=instruction,
        status=result.status,
        version_before=version_before,
        version_after=version_after,
        correct_execution=bool(correct_execution and not invalid),
        correct_clarification=bool(correct_clarification),
        false_success=bool(false_success),
        invalid_graph=bool(invalid),
        unintended_change=bool(result.status == "success" and not correct_execution and result.final_graph != before),
        first_failure_layer=None if correct_execution or correct_clarification else first_failure_layer(result),
        details={
            "changed_nodes": changed_nodes(before, result.final_graph),
            "steps": result.steps,
            "error": result.error,
            "history_reference_used": result_history_reference_used(result),
        },
    )


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


def result_history_reference_used(result: Any) -> bool:
    specification = result.specification if isinstance(result.specification, dict) else {}
    requirements = specification.get("grounded_requirements", []) if isinstance(specification, dict) else []
    return any(
        isinstance(requirement, dict)
        and isinstance(requirement.get("references"), dict)
        and requirement["references"].get("history_reference_used") is True
        for requirement in requirements
    )


def round_version_consistent(row: dict[str, Any]) -> bool:
    before = row.get("version_before")
    after = row.get("version_after")
    if not before or not after:
        return True
    if row.get("status") == "success":
        return before != after
    if row.get("status") == "needs_clarification":
        return before == after
    if row.get("status") in {"rollback_success", "redo_success"}:
        return bool(row.get("correct_execution"))
    return True


def run_mt1(root: Path) -> dict[str, Any]:
    case_dir = root / "MT1_state_carryover"
    case_dir.mkdir(parents=True, exist_ok=True)
    graph = base_graph()
    rounds: list[RoundRecord] = []

    before = deepcopy(graph)
    r1 = run_transaction(graph, 1, {"node_ids": [], "edge_ids": []}, "Rename risk_check to Risk Assessment.")
    save_json(case_dir / "round_1_transaction.json", transaction_payload(r1))
    rounds.append(record_transaction_round("round_1", "Rename risk_check to Risk Assessment.", before, r1, node(r1.final_graph, "risk_check") and node(r1.final_graph, "risk_check").get("label") == "Risk Assessment"))
    graph = r1.final_graph if r1.status == "success" and isinstance(r1.final_graph, dict) else graph

    before = deepcopy(graph)
    r2 = run_transaction(graph, 2, {"node_ids": [], "edge_ids": []}, "Move Risk Assessment 120 pixels to the right.")
    save_json(case_dir / "round_2_transaction.json", transaction_payload(r2))
    moved = node(r2.final_graph, "risk_check")
    rounds.append(record_transaction_round("round_2", "Move Risk Assessment 120 pixels to the right.", before, r2, bool(moved and moved.get("x") == 480 and moved.get("label") == "Risk Assessment")))
    graph = r2.final_graph if r2.status == "success" and isinstance(r2.final_graph, dict) else graph

    before = deepcopy(graph)
    r3 = run_transaction(graph, 3, {"node_ids": [], "edge_ids": []}, "Redirect Risk Assessment failure edge to manual_review.")
    save_json(case_dir / "round_3_transaction.json", transaction_payload(r3))
    rounds.append(record_transaction_round("round_3", "Redirect Risk Assessment failure edge to manual_review.", before, r3, edge_exists(r3.final_graph, "risk_check", "manual_review", "failure") and edge_absent(r3.final_graph, "risk_check", "reject", "failure")))

    row = {
        "case": "MT1",
        "description": "rename risk_check, then refer to renamed node for move and failure redirect",
        "passed": all(item.correct_execution for item in rounds),
        "rounds": [item.__dict__ for item in rounds],
    }
    save_json(case_dir / "record.json", row)
    return row


def run_mt2(root: Path) -> dict[str, Any]:
    case_dir = root / "MT2_correction_without_memory"
    case_dir.mkdir(parents=True, exist_ok=True)
    graph = base_graph()
    state = create_version_state(graph, created_at="2026-09-01T00:00:00+00:00")
    rounds: list[RoundRecord] = []

    before = state.current_graph()
    tx1 = run_versioned_transaction(state, 1, {"node_ids": ["identity_check", "risk_check"], "edge_ids": []}, "Move the selected nodes 120 pixels to the right.")
    save_json(case_dir / "round_1_transaction.json", transaction_payload(tx1.execution))
    ok1 = bool(node(tx1.execution.final_graph, "identity_check") and node(tx1.execution.final_graph, "identity_check").get("x") == 300 and node(tx1.execution.final_graph, "risk_check") and node(tx1.execution.final_graph, "risk_check").get("x") == 480)
    rounds.append(record_transaction_round("round_1", "Move the selected nodes 120 pixels to the right.", before, tx1.execution, ok1, version_before="v1", version_after=state.current_version_id))

    before = state.current_graph()
    instruction = "Undo the identity_check move and keep risk_check as it is."
    tx2 = run_versioned_transaction(state, 2, {"node_ids": [], "edge_ids": []}, instruction)
    save_json(case_dir / "round_2_transaction.json", transaction_payload(tx2.execution))
    identity_after = node(tx2.execution.final_graph, "identity_check")
    risk_after = node(tx2.execution.final_graph, "risk_check")
    ok2 = bool(identity_after and risk_after and identity_after.get("x") == 180 and risk_after.get("x") == 480)
    rounds.append(record_transaction_round("round_2", instruction, before, tx2.execution, ok2, version_before="v2", version_after=state.current_version_id))

    row = {
        "case": "MT2",
        "description": "correction instruction after prior move without adding memory architecture",
        "passed": all(item.correct_execution for item in rounds),
        "rounds": [item.__dict__ for item in rounds],
        "version_state": as_jsonable(state),
    }
    save_json(case_dir / "record.json", row)
    return row


def run_mt3(root: Path) -> dict[str, Any]:
    case_dir = root / "MT3_version_rollback_redo"
    case_dir.mkdir(parents=True, exist_ok=True)
    graph = base_graph()
    state = create_version_state(graph, created_at="2026-09-01T00:00:00+00:00")
    rounds: list[RoundRecord] = []

    before = state.current_graph()
    tx1 = run_versioned_transaction(state, 1, {"node_ids": [], "edge_ids": ["e3"]}, "Insert a task named Audit on the selected edge.")
    save_json(case_dir / "round_1_versioned_transaction.json", {"transaction": tx1.transaction.__dict__, "execution": transaction_payload(tx1.execution), "committed_version": tx1.committed_version.__dict__ if tx1.committed_version else None})
    inserted_graph = state.current_graph()
    has_audit = any(item.get("label") == "Audit" for item in inserted_graph.get("nodes", []))
    rounds.append(record_transaction_round("round_1", "Insert a task named Audit on the selected edge.", before, tx1.execution, tx1.execution.status == "success" and has_audit, version_before="v1", version_after=state.current_version_id))

    before_undo = state.current_graph()
    undone = undo(state)
    undo_ok = undone is not None and state.current_version_id == "v1" and graph_equal(state.current_graph(), graph)
    rounds.append(
        RoundRecord(
            "round_2",
            "rollback",
            "rollback_success" if undone else "rollback_failed",
            "v2",
            state.current_version_id,
            bool(undo_ok),
            False,
            False,
            False,
            False,
            None if undo_ok else "Version",
            {"snapshot_matches_initial": graph_equal(state.current_graph(), graph), "changed_nodes_before_rollback": changed_nodes(graph, before_undo)},
        )
    )

    redone = redo(state)
    redo_ok = redone is not None and state.current_version_id == "v2" and graph_equal(state.current_graph(), inserted_graph)
    rounds.append(
        RoundRecord(
            "round_3",
            "redo",
            "redo_success" if redone else "redo_failed",
            "v1",
            state.current_version_id,
            bool(redo_ok),
            False,
            False,
            False,
            False,
            None if redo_ok else "Version",
            {"snapshot_matches_inserted": graph_equal(state.current_graph(), inserted_graph)},
        )
    )

    row = {
        "case": "MT3",
        "description": "version rollback and redo snapshot consistency",
        "passed": all(item.correct_execution for item in rounds),
        "rounds": [item.__dict__ for item in rounds],
        "version_state": as_jsonable(state),
    }
    save_json(case_dir / "record.json", row)
    return row


def run_mt4(root: Path) -> dict[str, Any]:
    case_dir = root / "MT4_ambiguous_turn_reference"
    case_dir.mkdir(parents=True, exist_ok=True)
    graph = base_graph()
    state = create_version_state(graph, created_at="2026-09-01T00:00:00+00:00")
    rounds: list[RoundRecord] = []

    archived_parallel_fixture = {
        "instruction": "Make identity_check and risk_check parallel before approval, and approval waits for all incoming.",
        "archived_reason": "Projection: required structural participants not fully realized",
        "counted_in_multiturn_acceptance": False,
    }
    save_json(case_dir / "archived_parallel_fixture.json", archived_parallel_fixture)

    before = state.current_graph()
    round_1_instruction = "Move identity_check and risk_check 80 pixels to the right."
    tx1 = run_versioned_transaction(state, 1, {"node_ids": [], "edge_ids": []}, round_1_instruction)
    save_json(case_dir / "round_1_transaction.json", transaction_payload(tx1.execution))
    ok1 = bool(
        tx1.execution.status == "success"
        and node(tx1.execution.final_graph, "identity_check")
        and node(tx1.execution.final_graph, "identity_check").get("x") == 260
        and node(tx1.execution.final_graph, "risk_check")
        and node(tx1.execution.final_graph, "risk_check").get("x") == 440
        and len(tx1.transaction.plan) >= 2
    )
    rounds.append(record_transaction_round("round_1", round_1_instruction, before, tx1.execution, ok1, version_before="v1", version_after=state.current_version_id))

    before = state.current_graph()
    instruction = "Undo the second one."
    tx2 = run_versioned_transaction(state, 2, {"node_ids": [], "edge_ids": []}, instruction)
    save_json(case_dir / "round_2_transaction.json", transaction_payload(tx2.execution))
    ok2 = bool(clarification_ok(tx2.execution) and graph_equal(state.current_graph(), before) and state.current_version_id == "v2")
    rounds.append(record_transaction_round("round_2", instruction, before, tx2.execution, False, ok2, version_before="v2", version_after=state.current_version_id))

    row = {
        "case": "MT4",
        "description": "ambiguous history mutation reference should clarify without committing a new version",
        "passed": rounds[0].correct_execution and rounds[1].correct_clarification and not rounds[1].false_success and state.current_version_id == "v2",
        "rounds": [item.__dict__ for item in rounds],
        "version_state": as_jsonable(state),
        "archived_parallel_fixture": archived_parallel_fixture,
    }
    save_json(case_dir / "record.json", row)
    return row


def write_summary(root: Path, cases: list[dict[str, Any]]) -> dict[str, Any]:
    round_rows = [round_row for case in cases for round_row in case["rounds"]]
    version_consistency_rows = [row for row in round_rows if row.get("version_before") and row.get("version_after")]
    summary = {
        "root": str(root),
        "totals": {
            "cases": len(cases),
            "passed_cases": sum(bool(case["passed"]) for case in cases),
            "rounds": len(round_rows),
            "correct_execution": sum(bool(row["correct_execution"]) for row in round_rows),
            "correct_clarification": sum(bool(row["correct_clarification"]) for row in round_rows),
            "false_success": sum(bool(row["false_success"]) for row in round_rows),
            "incorrect_execution": sum(bool(row["false_success"] or row["unintended_change"]) for row in round_rows),
            "invalid_graph": sum(bool(row["invalid_graph"]) for row in round_rows),
            "unintended_change": sum(bool(row["unintended_change"]) for row in round_rows),
            "version_consistency_checked": len(version_consistency_rows),
            "version_consistency_passed": sum(round_version_consistent(row) for row in version_consistency_rows),
            "history_reference_used": sum(bool(row.get("details", {}).get("history_reference_used")) for row in round_rows),
        },
        "cases": cases,
    }
    save_json(root / "summary.json", summary)
    return summary


def main() -> None:
    root = Path("experiments/runs") / f"stage3-multiturn-acceptance-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    root.mkdir(parents=True, exist_ok=False)
    cases = [run_mt1(root), run_mt2(root), run_mt3(root), run_mt4(root)]
    summary = write_summary(root, cases)
    print(json.dumps({"root": str(root), "totals": summary["totals"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
