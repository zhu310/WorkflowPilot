from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from src.mvp23 import run_transaction
from src.runner import save_json


def _graph_scope() -> dict[str, Any]:
    return {
        "nodes": [
            {"id": "draft", "type": "task", "label": "Draft", "x": 40, "y": 40, "width": 120, "height": 60, "stage_id": "stage_intake", "group_ids": ["grp_intake"], "subprocess_id": "sp_intake"},
            {"id": "review_in", "type": "task", "label": "Review", "x": 220, "y": 40, "width": 120, "height": 60, "stage_id": "stage_intake", "group_ids": ["grp_intake"], "subprocess_id": "sp_intake"},
            {"id": "approve", "type": "task", "label": "Approve", "x": 460, "y": 40, "width": 120, "height": 60, "stage_id": "stage_approval", "group_ids": ["grp_approval"], "subprocess_id": "sp_approval"},
            {"id": "pay", "type": "task", "label": "Pay", "x": 640, "y": 40, "width": 120, "height": 60, "stage_id": "stage_approval", "group_ids": ["grp_approval"], "subprocess_id": "sp_approval"},
            {"id": "review_out", "type": "task", "label": "Review", "x": 860, "y": 40, "width": 120, "height": 60, "stage_id": "stage_archive", "group_ids": ["grp_archive"], "subprocess_id": "sp_archive"},
            {"id": "archive", "type": "task", "label": "Archive", "x": 1040, "y": 40, "width": 120, "height": 60, "stage_id": "stage_archive", "group_ids": ["grp_archive"], "subprocess_id": "sp_archive"},
        ],
        "edges": [
            {"id": "e1", "source": "draft", "target": "review_in", "label": ""},
            {"id": "e2", "source": "review_in", "target": "approve", "label": ""},
            {"id": "e3", "source": "approve", "target": "pay", "label": ""},
            {"id": "e4", "source": "pay", "target": "review_out", "label": ""},
            {"id": "e5", "source": "review_out", "target": "archive", "label": ""},
        ],
        "stages": [
            {"id": "stage_intake", "title": "Intake Stage", "node_ids": ["draft", "review_in"]},
            {"id": "stage_approval", "title": "Approval Stage", "node_ids": ["approve", "pay"]},
            {"id": "stage_archive", "title": "Archive Stage", "node_ids": ["review_out", "archive"]},
        ],
        "groups": [
            {"id": "grp_intake", "label": "Intake Group", "node_ids": ["draft", "review_in"]},
            {"id": "grp_approval", "label": "Approval Group", "node_ids": ["approve", "pay"]},
            {"id": "grp_archive", "label": "Archive Group", "node_ids": ["review_out", "archive"]},
        ],
        "subprocesses": [
            {"id": "sp_intake", "name": "Intake Subprocess", "node_ids": ["draft", "review_in"]},
            {"id": "sp_approval", "name": "Approval Subprocess", "node_ids": ["approve", "pay"]},
            {"id": "sp_archive", "name": "Archive Subprocess", "node_ids": ["review_out", "archive"]},
        ],
    }


def _graph_duplicate_stage() -> dict[str, Any]:
    graph = _graph_scope()
    graph["stages"] = [
        {"id": "stage_dup_a", "title": "Review Stage", "node_ids": ["draft", "review_in"]},
        {"id": "stage_dup_b", "title": "Review Stage", "node_ids": ["review_out", "archive"]},
    ]
    for node in graph["nodes"]:
        if node["id"] in {"draft", "review_in"}:
            node["stage_id"] = "stage_dup_a"
        if node["id"] in {"review_out", "archive"}:
            node["stage_id"] = "stage_dup_b"
    return graph


def _resolved_scope_ids(result: Any) -> list[str]:
    specification = result.specification if isinstance(result.specification, dict) else {}
    requirements = specification.get("grounded_requirements", []) if isinstance(specification, dict) else []
    node_ids: set[str] = set()
    for requirement in requirements:
        desired_state = requirement.get("grounded_desired_state", {}) if isinstance(requirement, dict) else {}
        scope_policy = desired_state.get("scope_policy", {}) if isinstance(desired_state, dict) else {}
        for node_id in scope_policy.get("node_ids", []):
            if isinstance(node_id, str):
                node_ids.add(node_id)
    return sorted(node_ids)


def _positions(graph: dict[str, Any]) -> dict[str, tuple[int, int]]:
    return {node["id"]: (node["x"], node["y"]) for node in graph["nodes"]}


def _scope_move_ok(
    original_graph: dict[str, Any],
    final_graph: dict[str, Any] | None,
    scope_ids: set[str],
    dx: int,
) -> bool:
    if not isinstance(final_graph, dict):
        return False
    before = _positions(original_graph)
    after = _positions(final_graph)
    if set(before) != set(after):
        return False
    for node_id, (x, y) in before.items():
        expected = (x + dx, y) if node_id in scope_ids else (x, y)
        if after[node_id] != expected:
            return False
    return original_graph["edges"] == final_graph["edges"]


def _clarification_ok(result: Any) -> bool:
    specification = result.specification if isinstance(result.specification, dict) else {}
    clarification = specification.get("clarification", {}) if isinstance(specification, dict) else {}
    return result.status == "needs_clarification" and bool(clarification.get("question"))


def _stop_layer(result: Any) -> str | None:
    specification = result.specification if isinstance(result.specification, dict) else {}
    if isinstance(specification, dict) and specification.get("stop_layer"):
        return specification["stop_layer"]
    return {
        "needs_clarification": "Ground",
        "unsupported": "Workflow IR",
        "planning_error": "Planner",
        "execution_error": "Executor",
        "interpretation_error": "Understand",
    }.get(result.status)


@dataclass(frozen=True)
class Case:
    name: str
    graph: dict[str, Any]
    instruction: str
    selection: dict[str, Any]
    request_context: dict[str, Any] | None
    expected_scope_ids: list[str]
    evaluator: Callable[[Any], tuple[bool, bool]]


def _run_case(root: Path, case: Case) -> dict[str, Any]:
    result = run_transaction(case.graph, 1, case.selection, case.instruction, request_context=case.request_context)
    correct_execution, correct_clarification = case.evaluator(result)
    resolved_scope_ids = _resolved_scope_ids(result)
    row = {
        "name": case.name,
        "instruction": case.instruction,
        "selection": case.selection,
        "request_context": case.request_context,
        "expected_scope_ids": case.expected_scope_ids,
        "resolved_scope_ids": resolved_scope_ids,
        "scope_correct": resolved_scope_ids == sorted(case.expected_scope_ids),
        "status": result.status,
        "correct_execution": correct_execution,
        "correct_clarification": correct_clarification,
        "unintended_change": result.status == "success" and not correct_execution and result.final_graph != case.graph,
        "unauthorized_mutation": bool(result.error and "authorized" in str(result.error).lower()),
        "false_success": result.status == "success" and result.final_graph == case.graph and not correct_execution,
        "first_failure_layer": None if (correct_execution or correct_clarification) else _stop_layer(result),
        "trace": result.specification if isinstance(result.specification, dict) else {},
    }
    case_dir = root / case.name
    case_dir.mkdir(parents=True, exist_ok=True)
    save_json(case_dir / "result.json", row)
    return row


def main() -> None:
    graph = _graph_scope()
    visible_ids = ["draft", "review_in", "approve"]
    approval_ids = ["approve", "pay"]
    all_ids = [node["id"] for node in graph["nodes"]]

    cases = [
        Case(
            "V1_visible_move",
            graph,
            "Move the currently visible nodes 80 pixels to the right.",
            {"node_ids": [], "edge_ids": []},
            {"visible_area": {"x": 0, "y": 0, "width": 620, "height": 180}},
            visible_ids,
            lambda result, graph=graph: (
                result.status == "success" and _scope_move_ok(graph, result.final_graph, set(visible_ids), 80),
                False,
            ),
        ),
        Case(
            "S1_stage_move",
            graph,
            "Move the Approval Stage nodes 80 pixels to the right.",
            {"node_ids": [], "edge_ids": []},
            None,
            approval_ids,
            lambda result, graph=graph: (
                result.status == "success" and _scope_move_ok(graph, result.final_graph, set(approval_ids), 80),
                False,
            ),
        ),
        Case(
            "G1_global_move",
            graph,
            "Move every node on the whole canvas 80 pixels to the right.",
            {"node_ids": ["draft"], "edge_ids": []},
            None,
            all_ids,
            lambda result, graph=graph: (
                result.status == "success" and _scope_move_ok(graph, result.final_graph, set(all_ids), 80),
                False,
            ),
        ),
        Case(
            "V2_visible_duplicate_labels",
            graph,
            "Move the nodes currently in view 80 pixels to the right.",
            {"node_ids": [], "edge_ids": []},
            {"visible_area": {"left": 0, "top": 0, "right": 360, "bottom": 180}},
            ["draft", "review_in"],
            lambda result, graph=graph: (
                result.status == "success" and _scope_move_ok(graph, result.final_graph, {"draft", "review_in"}, 80),
                False,
            ),
        ),
        Case(
            "S2_group_move",
            graph,
            "Move the Approval Group 80 pixels to the right.",
            {"node_ids": [], "edge_ids": []},
            None,
            approval_ids,
            lambda result, graph=graph: (
                result.status == "success" and _scope_move_ok(graph, result.final_graph, set(approval_ids), 80),
                False,
            ),
        ),
        Case(
            "S3_subprocess_move",
            graph,
            "Move the Intake Subprocess 80 pixels to the right.",
            {"node_ids": [], "edge_ids": []},
            None,
            ["draft", "review_in"],
            lambda result, graph=graph: (
                result.status == "success" and _scope_move_ok(graph, result.final_graph, {"draft", "review_in"}, 80),
                False,
            ),
        ),
        Case(
            "A1_duplicate_stage_title",
            _graph_duplicate_stage(),
            "Move the Review Stage 80 pixels to the right.",
            {"node_ids": [], "edge_ids": []},
            None,
            [],
            lambda result: (False, _clarification_ok(result)),
        ),
        Case(
            "A2_global_over_selection",
            graph,
            "Move the entire workflow 80 pixels to the right.",
            {"node_ids": ["approve"], "edge_ids": []},
            None,
            all_ids,
            lambda result, graph=graph: (
                result.status == "success" and _scope_move_ok(graph, result.final_graph, set(all_ids), 80),
                False,
            ),
        ),
        Case(
            "A3_selected_only",
            graph,
            "Move only the selected nodes 80 pixels to the right.",
            {"node_ids": ["approve", "pay"], "edge_ids": []},
            None,
            approval_ids,
            lambda result, graph=graph: (
                result.status == "success" and _scope_move_ok(graph, result.final_graph, set(approval_ids), 80),
                False,
            ),
        ),
        Case(
            "A4_unclear_here",
            graph,
            "Move everything here 80 pixels to the right.",
            {"node_ids": [], "edge_ids": []},
            None,
            [],
            lambda result: (False, _clarification_ok(result)),
        ),
    ]

    root = Path("experiments/runs") / f"stage3-scope-context-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    root.mkdir(parents=True, exist_ok=False)
    rows = [_run_case(root, case) for case in cases]
    summary = {
        "root": str(root),
        "totals": {
            "scope_correct": sum(bool(row["scope_correct"]) for row in rows),
            "correct_execution": sum(bool(row["correct_execution"]) for row in rows),
            "correct_clarification": sum(bool(row["correct_clarification"]) for row in rows),
            "unintended_change": sum(bool(row["unintended_change"]) for row in rows),
            "unauthorized_mutation": sum(bool(row["unauthorized_mutation"]) for row in rows),
            "false_success": sum(bool(row["false_success"]) for row in rows),
        },
        "rows": rows,
    }
    save_json(root / "summary.json", summary)
    print(root)
    print(json.dumps(summary["totals"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
