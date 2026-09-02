from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
import time
from typing import Any

from experiments.deterministic_spec_unit_experiment import (
    final_graph_correct,
    grounded_relation_type_correct,
    requirements_semantically_correct,
    spec_units_correct,
    unintended_change,
)
from experiments.final_candidate_spec_unit_regression import REGRESSION_CASES
from experiments.workflow_ir_final_candidate_regression import explicit_obligations, unit_contract_correct
from src.mvp23 import run_transaction
from src.runner import save_json
from src.workflow_ir_core import graph_is_valid


ATTEMPTS = {
    "rename": 10,
    "insert": 10,
    "redirect": 10,
    "delete": 10,
    "move_after": 20,
    "simple_parallel": 20,
    "parallel_retry_exception": 20,
    "mixed_edit": 20,
    "insert_labeled": 20,
    "redirect_labeled": 20,
    "move_after_labeled": 20,
    "failure_success_retry_combo": 20,
}

CASE_BY_NAME = {case.name: case for case, _attempts in REGRESSION_CASES}
ORDERED_CASES = [CASE_BY_NAME[name] for name in ATTEMPTS]
RECORD_REQUIRED_KEYS = {
    "case",
    "status",
    "correct_execution",
    "safe_rejection",
    "incorrect_execution",
    "invalid_graph",
    "unauthorized_mutation",
    "unintended_change",
    "requirements_correct",
    "grounding_correct",
    "requirement_conservation_pass",
    "spec_units_correct",
    "unit_contract_correct",
    "obligation_realization_complete",
    "orphan_obligations",
    "duplicate_realizations",
    "deterministic_facts_count",
    "semantic_slots_count",
    "first_pass_ir_valid",
    "canonicalization_used",
    "projection_used",
    "obligation_coverage",
    "workflow_ir_semantically_correct",
    "property_target_provenance_valid",
    "compiler_correct",
    "final_graph_correct",
    "first_failure_layer",
}
TRANSPORT_ERROR_TOKENS = (
    "incompleteread",
    "timeout",
    "timed out",
    "connection reset",
    "connection aborted",
    "connection refused",
    "getaddrinfo failed",
    "remotedisconnected",
    "urlopen error",
    "temporarily unavailable",
    "bad gateway",
    "service unavailable",
    "gateway timeout",
)


def _trace(result: Any) -> dict[str, Any]:
    specification = result.specification if isinstance(result.specification, dict) else {}
    return specification if isinstance(specification, dict) else {}


def _first_failure_layer(result: Any, row: dict[str, Any]) -> str | None:
    if row["correct_execution"]:
        return None
    trace = _trace(result)
    if trace.get("stop_layer"):
        return trace["stop_layer"]
    if row["incorrect_execution"]:
        return "Evaluator"
    if result.status != "success":
        return {
            "interpretation_error": "Understand",
            "planning_error": "Planner",
            "execution_error": "Executor",
            "unsupported": "Workflow IR",
        }.get(result.status, result.status)
    return "Evaluator"


def _attempt_folder(root: Path, case_name: str, attempt: int) -> Path:
    return root / case_name / f"attempt_{attempt:02d}"


def _load_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def attempt_completed(folder: Path) -> bool:
    record = _load_json(folder / "record.json")
    transaction = _load_json(folder / "transaction_result.json")
    if not isinstance(record, dict) or not isinstance(transaction, dict):
        return False
    if not RECORD_REQUIRED_KEYS <= set(record):
        return False
    if "status" not in transaction or "steps" not in transaction or "specification" not in transaction:
        return False
    return True


def load_completed_row(folder: Path) -> dict[str, Any] | None:
    if not attempt_completed(folder):
        return None
    return _load_json(folder / "record.json")


def is_transport_error(error: BaseException) -> bool:
    message = f"{type(error).__name__}: {error}".lower()
    return any(token in message for token in TRANSPORT_ERROR_TOKENS)


def save_checkpoint(
    root: Path,
    selected: list[Any],
    rows: list[dict[str, Any]],
    transport_stats: dict[str, int],
    current_case: str | None,
    current_attempt: int | None,
    last_error: str | None,
) -> None:
    completed_attempts = len(rows)
    total_attempts = sum(ATTEMPTS[case.name] for case in selected)
    checkpoint = {
        "completed_attempts": completed_attempts,
        "remaining_attempts": total_attempts - completed_attempts,
        "transport_errors": transport_stats["transport_errors"],
        "transport_retries": transport_stats["transport_retries"],
        "transport_recovered": transport_stats["transport_recovered"],
        "current_case": current_case,
        "current_attempt": current_attempt,
        "last_error": last_error,
        "updated_at": datetime.now().isoformat(timespec="seconds"),
    }
    save_json(root / "checkpoint.json", checkpoint)


def run_one(case: Any, folder: Path, transport_retry_count: int = 0) -> dict[str, Any]:
    folder.mkdir(parents=True, exist_ok=True)
    result = run_transaction(case.graph, 1, {"node_ids": [], "edge_ids": []}, case.instruction)
    trace = _trace(result)
    grounded_requirements = trace.get("grounded_requirements", [])
    units = trace.get("spec_units", [])
    unit_contracts = trace.get("unit_contracts", [])
    unit_runs = trace.get("unit_runs", [])
    global_obligations = trace.get("global_obligations", {}) if isinstance(trace.get("global_obligations"), dict) else explicit_obligations([], case.graph)
    invalid_graph = result.final_graph is not None and not graph_is_valid(result.final_graph)
    unauthorized_mutation = bool(result.error and "authorized" in str(result.error).lower())
    unintended = unintended_change(case, result.final_graph)
    target_requirement_satisfied = result.status == "success" and final_graph_correct(case, result.final_graph)
    correct_execution = result.status == "success" and target_requirement_satisfied and not invalid_graph and not unauthorized_mutation and not unintended
    safe_rejection = result.status != "success"
    incorrect_execution = result.status == "success" and not correct_execution
    orphan_obligations = [
        obligation
        for unit_run in unit_runs
        for obligation in unit_run.get("obligation_realization", {}).get("orphan_obligations", [])
    ]
    duplicate_realizations = {
        unit_run.get("unit_id", f"u{index}"): unit_run.get("obligation_realization", {}).get("duplicate_realizations", {})
        for index, unit_run in enumerate(unit_runs, start=1)
        if unit_run.get("obligation_realization", {}).get("duplicate_realizations")
    }
    row = {
        "case": case.name,
        "attempt_folder": str(folder),
        "status": result.status,
        "transport_retry_count": transport_retry_count,
        "correct_execution": correct_execution,
        "safe_rejection": safe_rejection,
        "incorrect_execution": incorrect_execution,
        "invalid_graph": invalid_graph,
        "unauthorized_mutation": unauthorized_mutation,
        "unintended_change": unintended,
        "requirements_correct": requirements_semantically_correct(case, grounded_requirements),
        "grounding_correct": grounded_relation_type_correct(grounded_requirements, case.graph),
        "requirement_conservation_pass": bool(trace.get("requirement_conservation", {}).get("valid")),
        "spec_units_correct": spec_units_correct(case, units),
        "unit_contract_correct": unit_contract_correct(units, unit_contracts, global_obligations) if units or unit_contracts else not global_obligations,
        "obligation_realization_complete": all(
            unit_run.get("obligation_realization", {}).get("obligation_realization_complete", True)
            for unit_run in unit_runs
        ),
        "orphan_obligations": orphan_obligations,
        "duplicate_realizations": duplicate_realizations,
        "deterministic_facts_count": sum(
            unit_run.get("obligation_realization", {}).get("deterministic_facts_count", 0)
            for unit_run in unit_runs
        ),
        "semantic_slots_count": sum(
            unit_run.get("obligation_realization", {}).get("semantic_slots_count", 0)
            for unit_run in unit_runs
        ),
        "first_pass_ir_valid": all(
            unit_run.get("first_pass_ir_valid", True)
            for unit_run in unit_runs
            if unit_run.get("llm_generation_required")
        ),
        "canonicalization_used": any(unit_run.get("canonicalization_used") for unit_run in unit_runs),
        "projection_used": any(unit_run.get("projection_used") for unit_run in unit_runs),
        "obligation_coverage": all(unit_run.get("obligation_coverage", True) for unit_run in unit_runs),
        "workflow_ir_semantically_correct": all(
            unit_run.get("obligation_coverage", True) and unit_run.get("structural_participant_coverage", True)
            for unit_run in unit_runs
        ),
        "property_target_provenance_valid": all(unit_run.get("property_target_provenance_valid", True) for unit_run in unit_runs),
        "compiler_correct": result.status != "planning_error",
        "final_graph_correct": target_requirement_satisfied,
    }
    row["first_failure_layer"] = _first_failure_layer(result, row)

    save_json(folder / "transaction_result.json", {
        "status": result.status,
        "error": result.error,
        "planning_path": result.planning_path,
        "steps": result.steps,
        "final_graph": result.final_graph,
        "target": result.target,
        "graph_context": result.graph_context,
        "specification": result.specification,
        "mutation_authorization": result.mutation_authorization,
    })
    save_json(folder / "record.json", row)
    return row


def summarize(rows: list[dict[str, Any]], transport_stats: dict[str, int]) -> dict[str, Any]:
    totals = {
        "total": len(rows),
        "completed_attempts": len(rows),
        "correct": sum(bool(row["correct_execution"]) for row in rows),
        "safe": sum(bool(row["safe_rejection"]) for row in rows),
        "incorrect": sum(bool(row["incorrect_execution"]) for row in rows),
        "invalid": sum(bool(row["invalid_graph"]) for row in rows),
        "unauthorized": sum(bool(row["unauthorized_mutation"]) for row in rows),
        "unintended": sum(bool(row["unintended_change"]) for row in rows),
        "transport_errors": transport_stats["transport_errors"],
        "transport_retries": transport_stats["transport_retries"],
        "transport_recovered": transport_stats["transport_recovered"],
        "requirements_correct": sum(bool(row["requirements_correct"]) for row in rows),
        "grounding_correct": sum(bool(row["grounding_correct"]) for row in rows),
        "requirement_conservation_pass": sum(bool(row["requirement_conservation_pass"]) for row in rows),
        "spec_units_correct": sum(bool(row["spec_units_correct"]) for row in rows),
        "unit_contract_correct": sum(bool(row["unit_contract_correct"]) for row in rows),
        "obligation_realization_complete": sum(bool(row["obligation_realization_complete"]) for row in rows),
        "orphan_obligations": sum(len(row["orphan_obligations"]) for row in rows),
        "duplicate_realizations": sum(len(row["duplicate_realizations"]) for row in rows),
        "deterministic_facts_count": sum(int(row["deterministic_facts_count"]) for row in rows),
        "semantic_slots_count": sum(int(row["semantic_slots_count"]) for row in rows),
        "first_pass_ir_valid": sum(bool(row["first_pass_ir_valid"]) for row in rows),
        "canonicalization_used": sum(bool(row["canonicalization_used"]) for row in rows),
        "projection_used": sum(bool(row["projection_used"]) for row in rows),
        "obligation_coverage": sum(bool(row["obligation_coverage"]) for row in rows),
        "workflow_ir_semantically_correct": sum(bool(row["workflow_ir_semantically_correct"]) for row in rows),
        "property_target_provenance_valid": sum(bool(row["property_target_provenance_valid"]) for row in rows),
        "compiler_correct": sum(bool(row["compiler_correct"]) for row in rows),
        "final_graph_correct": sum(bool(row["final_graph_correct"]) for row in rows),
    }
    per_case: dict[str, dict[str, Any]] = {}
    first_failure_layers = Counter(row["first_failure_layer"] for row in rows if row["first_failure_layer"])
    case_failures: dict[str, Counter[str]] = defaultdict(Counter)
    for case in ATTEMPTS:
        case_rows = [row for row in rows if row["case"] == case]
        per_case[case] = {
            "total": len(case_rows),
            "correct_execution": sum(bool(row["correct_execution"]) for row in case_rows),
            "safe_rejection": sum(bool(row["safe_rejection"]) for row in case_rows),
            "incorrect_execution": sum(bool(row["incorrect_execution"]) for row in case_rows),
            "invalid_graph": sum(bool(row["invalid_graph"]) for row in case_rows),
            "unauthorized_mutation": sum(bool(row["unauthorized_mutation"]) for row in case_rows),
            "unintended_change": sum(bool(row["unintended_change"]) for row in case_rows),
        }
        for row in case_rows:
            if row["first_failure_layer"]:
                case_failures[case][row["first_failure_layer"]] += 1
    return {
        "totals": totals,
        "per_case": per_case,
        "first_failure_layers": dict(first_failure_layers),
        "case_failure_layers": {case: dict(counter) for case, counter in case_failures.items()},
        "attempts": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", nargs="*", choices=list(ATTEMPTS))
    parser.add_argument("--root")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-transport-retries", type=int, default=3)
    args = parser.parse_args()

    selected = [CASE_BY_NAME[name] for name in args.cases] if args.cases else ORDERED_CASES
    root = Path(args.root) if args.root else Path("experiments/runs") / f"workflow-ir-formal-mainline-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    if args.resume:
        root.mkdir(parents=True, exist_ok=True)
    else:
        root.mkdir(parents=True, exist_ok=False)
    rows: list[dict[str, Any]] = []
    transport_stats = {
        "transport_errors": 0,
        "transport_retries": 0,
        "transport_recovered": 0,
    }

    for case in selected:
        for attempt in range(1, ATTEMPTS[case.name] + 1):
            folder = _attempt_folder(root, case.name, attempt)
            existing = load_completed_row(folder)
            if existing is not None:
                rows.append(existing)
                continue
            if folder.exists():
                save_json(folder / "resume_note.json", {"status": "restarting_incomplete_attempt"})
            retries_used = 0
            while True:
                try:
                    row = run_one(case, folder, transport_retry_count=retries_used)
                    rows.append(row)
                    if retries_used:
                        transport_stats["transport_recovered"] += 1
                    save_checkpoint(root, selected, rows, transport_stats, case.name, attempt, None)
                    break
                except Exception as error:
                    if not is_transport_error(error):
                        raise
                    transport_stats["transport_errors"] += 1
                    if retries_used >= args.max_transport_retries:
                        save_json(
                            folder / "transport_error.json",
                            {
                                "case": case.name,
                                "attempt": attempt,
                                "error_type": type(error).__name__,
                                "error": str(error),
                                "retries_used": retries_used,
                            },
                        )
                        save_checkpoint(root, selected, rows, transport_stats, case.name, attempt, f"{type(error).__name__}: {error}")
                        print(
                            json.dumps(
                                {
                                    "status": "stopped_on_transport_error",
                                    "root": str(root),
                                    "case": case.name,
                                    "attempt": attempt,
                                    "completed_attempts": len(rows),
                                    "remaining_attempts": sum(ATTEMPTS[item.name] for item in selected) - len(rows),
                                    "transport_errors": transport_stats["transport_errors"],
                                    "transport_retries": transport_stats["transport_retries"],
                                    "transport_recovered": transport_stats["transport_recovered"],
                                    "error_type": type(error).__name__,
                                    "error": str(error),
                                },
                                ensure_ascii=False,
                                indent=2,
                            )
                        )
                        print(f"Saved: {root}")
                        return
                    retries_used += 1
                    transport_stats["transport_retries"] += 1
                    save_json(
                        folder / "transport_error.json",
                        {
                            "case": case.name,
                            "attempt": attempt,
                            "error_type": type(error).__name__,
                            "error": str(error),
                            "retries_used": retries_used,
                            "status": "retrying",
                        },
                    )
                    time.sleep(min(5 * retries_used, 15))

    summary = summarize(rows, transport_stats)
    save_json(root / "summary.json", summary)
    save_checkpoint(root, selected, rows, transport_stats, None, None, None)
    print(json.dumps(summary["totals"], ensure_ascii=False, indent=2))
    print(f"Saved: {root}")


if __name__ == "__main__":
    main()
