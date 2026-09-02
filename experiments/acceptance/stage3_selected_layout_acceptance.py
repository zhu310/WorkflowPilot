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


SELECTED_IDS = ["A", "B", "C"]


@dataclass(frozen=True)
class LayoutCase:
    name: str
    instruction: str
    selection: dict[str, Any]
    evaluator: Callable[[dict[str, Any], Any], tuple[bool, bool, bool, bool, bool, dict[str, Any]]]


def fixture_graph() -> dict[str, Any]:
    return {
        "nodes": [
            {"id": "start", "type": "start", "label": "Start", "x": 0, "y": 0, "width": 120, "height": 60},
            {"id": "A", "type": "task", "label": "Intake", "x": 180, "y": 20, "width": 120, "height": 60},
            {"id": "B", "type": "task", "label": "Review", "x": 370, "y": 110, "width": 120, "height": 60},
            {"id": "C", "type": "task", "label": "Risk", "x": 610, "y": 35, "width": 120, "height": 60},
            {"id": "D", "type": "task", "label": "Neighbor", "x": 800, "y": 80, "width": 120, "height": 60},
            {"id": "E", "type": "end", "label": "End", "x": 980, "y": 80, "width": 120, "height": 60},
        ],
        "edges": [
            {"id": "e1", "source": "start", "target": "A", "label": ""},
            {"id": "e2", "source": "A", "target": "B", "label": ""},
            {"id": "e3", "source": "B", "target": "C", "label": ""},
            {"id": "e4", "source": "C", "target": "D", "label": ""},
            {"id": "e5", "source": "D", "target": "E", "label": ""},
        ],
    }


def nodes_by_id(graph: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    if not isinstance(graph, dict):
        return {}
    return {node["id"]: node for node in graph.get("nodes", []) if isinstance(node, dict) and isinstance(node.get("id"), str)}


def edge_key(edge: dict[str, Any]) -> tuple[str, str, str]:
    return (str(edge.get("source")), str(edge.get("target")), str(edge.get("label", "")))


def edge_set(graph: dict[str, Any] | None) -> set[tuple[str, str, str]]:
    if not isinstance(graph, dict):
        return set()
    return {edge_key(edge) for edge in graph.get("edges", []) if isinstance(edge, dict)}


def changed_node_ids(before: dict[str, Any], after: dict[str, Any] | None) -> list[str]:
    if after is None:
        return []
    before_nodes = nodes_by_id(before)
    after_nodes = nodes_by_id(after)
    return sorted(node_id for node_id in set(before_nodes) | set(after_nodes) if before_nodes.get(node_id) != after_nodes.get(node_id))


def node_property_diffs(before: dict[str, Any], after: dict[str, Any] | None) -> dict[str, dict[str, list[Any]]]:
    if after is None:
        return {}
    before_nodes = nodes_by_id(before)
    after_nodes = nodes_by_id(after)
    diffs: dict[str, dict[str, list[Any]]] = {}
    for node_id in sorted(set(before_nodes) | set(after_nodes)):
        keys = set(before_nodes.get(node_id, {})) | set(after_nodes.get(node_id, {}))
        for key in sorted(keys):
            if before_nodes.get(node_id, {}).get(key) != after_nodes.get(node_id, {}).get(key):
                diffs.setdefault(node_id, {})[key] = [before_nodes.get(node_id, {}).get(key), after_nodes.get(node_id, {}).get(key)]
    return diffs


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
        "success": "Final Validation",
    }.get(result.status)


def graph_unchanged(before: dict[str, Any], after: dict[str, Any] | None) -> bool:
    return after is None or (isinstance(after, dict) and before == after)


def outside_selection_unchanged(before: dict[str, Any], after: dict[str, Any] | None, selected_ids: set[str]) -> bool:
    if not isinstance(after, dict):
        return False
    before_nodes = nodes_by_id(before)
    after_nodes = nodes_by_id(after)
    outside_ids = set(before_nodes) - selected_ids
    if any(before_nodes[node_id] != after_nodes.get(node_id) for node_id in outside_ids):
        return False
    return edge_set(before) == edge_set(after)


def only_selected_position_changed(before: dict[str, Any], after: dict[str, Any] | None, selected_ids: set[str]) -> bool:
    if not isinstance(after, dict):
        return False
    diffs = node_property_diffs(before, after)
    for node_id, changes in diffs.items():
        if node_id not in selected_ids:
            return False
        if set(changes) - {"x", "y"}:
            return False
    return edge_set(before) == edge_set(after)


def selected_move_ok(before: dict[str, Any], after: dict[str, Any] | None, selected_ids: set[str], dx: int, dy: int = 0) -> bool:
    if not isinstance(after, dict):
        return False
    before_nodes = nodes_by_id(before)
    after_nodes = nodes_by_id(after)
    for node_id, before_node in before_nodes.items():
        after_node = after_nodes.get(node_id)
        if not isinstance(after_node, dict):
            return False
        expected_x = before_node["x"] + dx if node_id in selected_ids else before_node["x"]
        expected_y = before_node["y"] + dy if node_id in selected_ids else before_node["y"]
        if after_node.get("x") != expected_x or after_node.get("y") != expected_y:
            return False
    return edge_set(before) == edge_set(after)


def relative_positions_preserved(before: dict[str, Any], after: dict[str, Any] | None, selected_ids: list[str]) -> bool:
    if not isinstance(after, dict):
        return False
    before_nodes = nodes_by_id(before)
    after_nodes = nodes_by_id(after)
    base = selected_ids[0]
    for node_id in selected_ids[1:]:
        before_delta = (before_nodes[node_id]["x"] - before_nodes[base]["x"], before_nodes[node_id]["y"] - before_nodes[base]["y"])
        after_delta = (after_nodes[node_id]["x"] - after_nodes[base]["x"], after_nodes[node_id]["y"] - after_nodes[base]["y"])
        if before_delta != after_delta:
            return False
    return True


def vertically_aligned(after: dict[str, Any] | None, selected_ids: list[str]) -> bool:
    after_nodes = nodes_by_id(after)
    xs = {after_nodes.get(node_id, {}).get("x") for node_id in selected_ids}
    return len(xs) == 1 and None not in xs


def evenly_distributed_horizontally(after: dict[str, Any] | None, selected_ids: list[str]) -> bool:
    if not isinstance(after, dict):
        return False
    after_nodes = nodes_by_id(after)
    xs = sorted(after_nodes.get(node_id, {}).get("x") for node_id in selected_ids)
    if any(not isinstance(x, (int, float)) for x in xs):
        return False
    gaps = [xs[index + 1] - xs[index] for index in range(len(xs) - 1)]
    return len(set(gaps)) == 1 and gaps[0] > 0


def evaluate_move(before: dict[str, Any], result: Any) -> tuple[bool, bool, bool, bool, bool, dict[str, Any]]:
    selected = set(SELECTED_IDS)
    correct = (
        result.status == "success"
        and selected_move_ok(before, result.final_graph, selected, 80)
        and relative_positions_preserved(before, result.final_graph, SELECTED_IDS)
    )
    return correct, False, correct, outside_selection_unchanged(before, result.final_graph, selected), only_selected_position_changed(before, result.final_graph, selected), {}


def evaluate_align(before: dict[str, Any], result: Any) -> tuple[bool, bool, bool, bool, bool, dict[str, Any]]:
    selected = set(SELECTED_IDS)
    correct = (
        result.status == "success"
        and isinstance(result.final_graph, dict)
        and graph_is_valid(result.final_graph)
        and vertically_aligned(result.final_graph, SELECTED_IDS)
        and outside_selection_unchanged(before, result.final_graph, selected)
        and only_selected_position_changed(before, result.final_graph, selected)
    )
    return correct, False, vertically_aligned(result.final_graph, SELECTED_IDS), outside_selection_unchanged(before, result.final_graph, selected), only_selected_position_changed(before, result.final_graph, selected), {}


def evaluate_distribute(before: dict[str, Any], result: Any) -> tuple[bool, bool, bool, bool, bool, dict[str, Any]]:
    selected = set(SELECTED_IDS)
    correct = (
        result.status == "success"
        and isinstance(result.final_graph, dict)
        and graph_is_valid(result.final_graph)
        and evenly_distributed_horizontally(result.final_graph, SELECTED_IDS)
        and outside_selection_unchanged(before, result.final_graph, selected)
        and only_selected_position_changed(before, result.final_graph, selected)
    )
    return correct, False, evenly_distributed_horizontally(result.final_graph, SELECTED_IDS), outside_selection_unchanged(before, result.final_graph, selected), only_selected_position_changed(before, result.final_graph, selected), {}


def evaluate_selected_only(before: dict[str, Any], result: Any) -> tuple[bool, bool, bool, bool, bool, dict[str, Any]]:
    selected = {"A", "B"}
    correct = result.status == "success" and selected_move_ok(before, result.final_graph, selected, 80)
    return correct, False, correct, outside_selection_unchanged(before, result.final_graph, selected), only_selected_position_changed(before, result.final_graph, selected), {}


def evaluate_ambiguous_layout(before: dict[str, Any], result: Any) -> tuple[bool, bool, bool, bool, bool, dict[str, Any]]:
    correct_clarification = clarification_ok(result) and graph_unchanged(before, result.final_graph)
    return False, correct_clarification, False, graph_unchanged(before, result.final_graph), graph_unchanged(before, result.final_graph), {}


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


def run_case(root: Path, case: LayoutCase) -> dict[str, Any]:
    graph = fixture_graph()
    result = run_transaction(graph, 1, case.selection, case.instruction)
    correct_execution, correct_clarification, selected_nodes_correct, outside_unchanged, authorized_changes, details = case.evaluator(graph, result)
    invalid_graph = isinstance(result.final_graph, dict) and not graph_is_valid(result.final_graph)
    false_success = result.status == "success" and graph_unchanged(graph, result.final_graph) and not correct_execution
    unauthorized_change = result.status == "success" and not authorized_changes
    unintended_change = result.status == "success" and not correct_execution and not graph_unchanged(graph, result.final_graph)
    row = {
        "case": case.name,
        "instruction": case.instruction,
        "selection": case.selection,
        "status": result.status,
        "correct_execution": correct_execution,
        "correct_clarification": correct_clarification,
        "false_success": false_success,
        "invalid_graph": invalid_graph,
        "unauthorized_change": unauthorized_change,
        "unintended_change": unintended_change,
        "selected_nodes_correct": selected_nodes_correct,
        "outside_selection_unchanged": outside_unchanged,
        "first_deviation_stage": first_deviation_stage(result, correct_execution, correct_clarification),
        "changed_nodes": changed_node_ids(graph, result.final_graph),
        "node_property_diffs": node_property_diffs(graph, result.final_graph),
        "changed_edges": {
            "added": [] if result.final_graph is None else [list(row) for row in sorted(edge_set(result.final_graph) - edge_set(graph))],
            "removed": [] if result.final_graph is None else [list(row) for row in sorted(edge_set(graph) - edge_set(result.final_graph))],
        },
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
        LayoutCase(
            "L1_selected_multi_node_move",
            "Move the selected nodes 80 pixels to the right.",
            {"node_ids": SELECTED_IDS, "edge_ids": []},
            evaluate_move,
        ),
        LayoutCase(
            "L2_align_selected_vertically",
            "Align the selected nodes vertically.",
            {"node_ids": SELECTED_IDS, "edge_ids": []},
            evaluate_align,
        ),
        LayoutCase(
            "L3_distribute_selected_horizontally",
            "Distribute the selected nodes evenly horizontally.",
            {"node_ids": SELECTED_IDS, "edge_ids": []},
            evaluate_distribute,
        ),
        LayoutCase(
            "L4_selected_only_preserves_neighbor",
            "Move only the selected nodes 80 pixels to the right.",
            {"node_ids": ["A", "B"], "edge_ids": []},
            evaluate_selected_only,
        ),
        LayoutCase(
            "L5_ambiguous_layout_request",
            "Make these nodes look better.",
            {"node_ids": SELECTED_IDS, "edge_ids": []},
            evaluate_ambiguous_layout,
        ),
    ]
    root = Path("experiments/runs") / f"stage3-selected-layout-acceptance-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    root.mkdir(parents=True, exist_ok=False)
    rows = [run_case(root, case) for case in cases]
    summary = {
        "root": str(root),
        "fixture": {
            "selected_node_ids": SELECTED_IDS,
            "node_count": len(fixture_graph()["nodes"]),
            "edge_count": len(fixture_graph()["edges"]),
        },
        "totals": {
            "total": len(rows),
            "correct_execution": sum(bool(row["correct_execution"]) for row in rows),
            "correct_clarification": sum(bool(row["correct_clarification"]) for row in rows),
            "false_success": sum(bool(row["false_success"]) for row in rows),
            "invalid_graph": sum(bool(row["invalid_graph"]) for row in rows),
            "unauthorized_change": sum(bool(row["unauthorized_change"]) for row in rows),
            "unintended_change": sum(bool(row["unintended_change"]) for row in rows),
            "selected_nodes_correct": sum(bool(row["selected_nodes_correct"]) for row in rows),
            "outside_selection_unchanged": sum(bool(row["outside_selection_unchanged"]) for row in rows),
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
