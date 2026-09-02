from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from experiments.deterministic_spec_unit_experiment import final_graph_correct, unintended_change
from experiments.final_candidate_spec_unit_regression import REGRESSION_CASES
from src.mvp23 import run_transaction
from src.runner import save_json
from src.workflow_ir_core import graph_is_valid


CASE_BY_NAME = {case.name: case for case, _ in REGRESSION_CASES}


def _graph_basic() -> dict[str, Any]:
    return {
        "nodes": [
            {"id": "start", "type": "start", "label": "Start", "x": 0, "y": 0},
            {"id": "review", "type": "task", "label": "Review", "x": 200, "y": 0},
            {"id": "pay", "type": "task", "label": "Pay", "x": 400, "y": 0},
            {"id": "approval", "type": "task", "label": "Approval", "x": 600, "y": 0},
            {"id": "end", "type": "end", "label": "End", "x": 800, "y": 0},
        ],
        "edges": [
            {"id": "e1", "source": "start", "target": "review", "label": ""},
            {"id": "e2", "source": "review", "target": "pay", "label": ""},
            {"id": "e3", "source": "pay", "target": "approval", "label": ""},
            {"id": "e4", "source": "approval", "target": "end", "label": ""},
        ],
    }


def _label(graph: dict[str, Any] | None, node_id: str) -> str | None:
    if not isinstance(graph, dict):
        return None
    node = next((row for row in graph.get("nodes", []) if row.get("id") == node_id), None)
    return node.get("label") if isinstance(node, dict) else None


def _xy(graph: dict[str, Any] | None, node_id: str) -> tuple[Any, Any] | None:
    if not isinstance(graph, dict):
        return None
    node = next((row for row in graph.get("nodes", []) if row.get("id") == node_id), None)
    if not isinstance(node, dict):
        return None
    return node.get("x"), node.get("y")


def _join(graph: dict[str, Any] | None, node_id: str) -> Any:
    if not isinstance(graph, dict):
        return None
    node = next((row for row in graph.get("nodes", []) if row.get("id") == node_id), None)
    return node.get("join") if isinstance(node, dict) else None


def _exact_node_delta_only(graph_before: dict[str, Any], graph_after: dict[str, Any] | None, node_id: str, expected_changes: dict[str, Any]) -> bool:
    if not isinstance(graph_after, dict):
        return False
    before_nodes = {node["id"]: node for node in graph_before["nodes"]}
    after_nodes = {node["id"]: node for node in graph_after["nodes"]}
    if set(before_nodes) != set(after_nodes):
        return False
    if graph_before["edges"] != graph_after["edges"]:
        return False
    for current_id, before in before_nodes.items():
        after = after_nodes[current_id]
        if current_id == node_id:
            for key, value in expected_changes.items():
                if after.get(key) != value:
                    return False
            for key, value in before.items():
                if key not in expected_changes and after.get(key) != value:
                    return False
        elif after != before:
            return False
    return True


@dataclass(frozen=True)
class TargetCase:
    family: str
    name: str
    graph: dict[str, Any]
    instruction: str
    selection: dict[str, Any]
    evaluator: Callable[[Any], bool]
    unintended_evaluator: Callable[[Any], bool]


def _regression_case_case(name: str) -> TargetCase:
    case = CASE_BY_NAME[name]
    return TargetCase(
        family="pollution" if name in {"insert", "mixed_edit", "insert_labeled", "failure_success_retry_combo"} else "protection",
        name=name,
        graph=case.graph,
        instruction=case.instruction,
        selection={"node_ids": [], "edge_ids": []},
        evaluator=lambda result, case=case: result.status == "success" and final_graph_correct(case, result.final_graph),
        unintended_evaluator=lambda result, case=case: unintended_change(case, result.final_graph),
    )


def _custom_cases() -> list[TargetCase]:
    base = _graph_basic()
    return [
        TargetCase(
            family="protection",
            name="coordinate_move",
            graph=base,
            instruction="Move pay 120 pixels to the right",
            selection={"node_ids": [], "edge_ids": []},
            evaluator=lambda result, graph=base: result.status == "success" and _exact_node_delta_only(graph, result.final_graph, "pay", {"x": 520, "y": 0}),
            unintended_evaluator=lambda result, graph=base: result.status == "success" and not _exact_node_delta_only(graph, result.final_graph, "pay", {"x": 520, "y": 0}),
        ),
        TargetCase(
            family="protection",
            name="selected_node_rename",
            graph=base,
            instruction="Rename the selected node to Approved",
            selection={"node_ids": ["review"], "edge_ids": []},
            evaluator=lambda result, graph=base: result.status == "success" and _exact_node_delta_only(graph, result.final_graph, "review", {"label": "Approved"}),
            unintended_evaluator=lambda result, graph=base: result.status == "success" and not _exact_node_delta_only(graph, result.final_graph, "review", {"label": "Approved"}),
        ),
        TargetCase(
            family="protection",
            name="parallel_join",
            graph=CASE_BY_NAME["simple_parallel"].graph,
            instruction=CASE_BY_NAME["simple_parallel"].instruction,
            selection={"node_ids": [], "edge_ids": []},
            evaluator=lambda result, case=CASE_BY_NAME["simple_parallel"]: result.status == "success" and final_graph_correct(case, result.final_graph) and _join(result.final_graph, "approval") == {"mode": "all"},
            unintended_evaluator=lambda result, case=CASE_BY_NAME["simple_parallel"]: unintended_change(case, result.final_graph),
        ),
    ]


def _property_provenance_valid(result: Any) -> bool:
    trace = result.specification if isinstance(result.specification, dict) else {}
    unit_runs = trace.get("unit_runs", []) if isinstance(trace, dict) else []
    return all(unit_run.get("property_target_provenance_valid", True) for unit_run in unit_runs)


def _stop_layer(result: Any) -> str | None:
    specification = result.specification if isinstance(result.specification, dict) else {}
    if isinstance(specification, dict) and specification.get("stop_layer"):
        return specification["stop_layer"]
    if result.status == "success":
        return None
    return {
        "needs_clarification": "Ground",
        "unsupported": "Workflow IR",
        "planning_error": "Planner",
        "execution_error": "Executor",
        "interpretation_error": "Understand",
    }.get(result.status, result.status)


def _run_case(case: TargetCase, attempt: int, root: Path) -> dict[str, Any]:
    result = run_transaction(case.graph, 1, case.selection, case.instruction)
    target_requirement_satisfied = case.evaluator(result)
    invalid_graph = result.final_graph is not None and not graph_is_valid(result.final_graph)
    unauthorized_mutation = bool(result.error and "authorized" in str(result.error).lower())
    unintended = case.unintended_evaluator(result)
    correct_execution = result.status == "success" and target_requirement_satisfied and not invalid_graph and not unauthorized_mutation and not unintended
    row = {
        "family": case.family,
        "case": case.name,
        "attempt": attempt,
        "status": result.status,
        "target_requirement_satisfied": target_requirement_satisfied,
        "unintended_change": unintended,
        "unauthorized_mutation": unauthorized_mutation,
        "incorrect_execution": result.status == "success" and not correct_execution,
        "invalid_graph": invalid_graph,
        "correct_execution": correct_execution,
        "property_target_provenance_valid": _property_provenance_valid(result),
        "first_failure_layer": None if correct_execution else _stop_layer(result),
        "trace": result.specification if isinstance(result.specification, dict) else {},
    }
    folder = root / case.family / case.name / f"attempt_{attempt:02d}"
    folder.mkdir(parents=True, exist_ok=True)
    save_json(folder / "row.json", row)
    save_json(
        folder / "transaction_result.json",
        {
            "status": result.status,
            "error": result.error,
            "steps": result.steps,
            "final_graph": result.final_graph,
            "specification": result.specification,
            "mutation_authorization": result.mutation_authorization,
        },
    )
    return row


def main() -> None:
    root = Path("experiments/runs") / f"stage3-property-ownership-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    root.mkdir(parents=True, exist_ok=False)
    cases = [
        _regression_case_case("insert"),
        _regression_case_case("mixed_edit"),
        _regression_case_case("insert_labeled"),
        _regression_case_case("failure_success_retry_combo"),
        _regression_case_case("rename"),
        _regression_case_case("move_after_labeled"),
        *_custom_cases(),
    ]
    rows = [_run_case(case, attempt, root) for case in cases for attempt in range(1, 6)]
    summary = {
        "root": str(root),
        "totals": {
            "correct_execution": sum(bool(row["correct_execution"]) for row in rows),
            "incorrect_execution": sum(bool(row["incorrect_execution"]) for row in rows),
            "invalid_graph": sum(bool(row["invalid_graph"]) for row in rows),
            "unauthorized_mutation": sum(bool(row["unauthorized_mutation"]) for row in rows),
            "unintended_change": sum(bool(row["unintended_change"]) for row in rows),
            "property_target_provenance_valid": sum(bool(row["property_target_provenance_valid"]) for row in rows),
        },
        "per_case": {},
        "rows": rows,
    }
    for case in cases:
        case_rows = [row for row in rows if row["case"] == case.name]
        summary["per_case"][case.name] = {
            "total": len(case_rows),
            "correct_execution": sum(bool(row["correct_execution"]) for row in case_rows),
            "incorrect_execution": sum(bool(row["incorrect_execution"]) for row in case_rows),
            "invalid_graph": sum(bool(row["invalid_graph"]) for row in case_rows),
            "unauthorized_mutation": sum(bool(row["unauthorized_mutation"]) for row in case_rows),
            "unintended_change": sum(bool(row["unintended_change"]) for row in case_rows),
            "property_target_provenance_valid": sum(bool(row["property_target_provenance_valid"]) for row in case_rows),
        }
    save_json(root / "summary.json", summary)
    print(root)
    print(json.dumps(summary["totals"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
