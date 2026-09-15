from __future__ import annotations

import json
import time
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from src.mvp23 import run_transaction
from src.runner import save_json
from src.workflow_ir_core import graph_is_valid, visible_node_ids


OUTPUT_ROOT = Path("experiments/runs/stage3-large-graph-acceptance")
DOC_PATH = Path("docs/stage3_large_graph_acceptance.md")
STAGE_COUNT = 10
NODES_PER_STAGE = 36
DX = 80


@dataclass(frozen=True)
class LargeGraphCase:
    name: str
    request: str
    selection: dict[str, Any]
    request_context: dict[str, Any] | None
    expected_scope_ids: list[str]
    evaluator: Callable[[dict[str, Any], Any], tuple[bool, bool, bool, dict[str, Any]]]


def node_id(stage: int, index: int) -> str:
    if stage == 10 and index == 35:
        return "manual_review"
    return f"s{stage:02d}_n{index:02d}"


def node_label(stage: int, index: int) -> str:
    if stage == 10 and index == 35:
        return "Manual Review"
    if index == 5:
        return "Review"
    if index == 9:
        return "Identity Check"
    if index == 10:
        return "Risk Check"
    if index == 11:
        return "Approval Join"
    if index == 15:
        return "Decision"
    if index == 16:
        return "Fast Path"
    if index == 17:
        return "Exception Review"
    if index == 18:
        return "Audit"
    return f"Stage {stage:02d} Step {index:02d}"


def generate_complex_workflow(stage_count: int = STAGE_COUNT, nodes_per_stage: int = NODES_PER_STAGE) -> dict[str, Any]:
    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    stages: list[dict[str, Any]] = []
    groups: list[dict[str, Any]] = []
    subprocesses: list[dict[str, Any]] = []

    edge_index = 1
    for stage in range(1, stage_count + 1):
        stage_id = f"stage_{stage:02d}"
        group_id = f"group_{stage:02d}"
        subprocess_id = f"subprocess_{stage:02d}"
        stage_node_ids: list[str] = []
        base_x = (stage - 1) * 900
        base_y = 0 if stage % 2 else 260
        for index in range(1, nodes_per_stage + 1):
            nid = node_id(stage, index)
            stage_node_ids.append(nid)
            nodes.append(
                {
                    "id": nid,
                    "type": "human_review" if nid == "manual_review" or index in {5, 17} else "task",
                    "label": node_label(stage, index),
                    "x": base_x + ((index - 1) % 9) * 150,
                    "y": base_y + ((index - 1) // 9) * 95,
                    "width": 120,
                    "height": 58,
                    "stage_id": stage_id,
                    "group_ids": [group_id],
                    "subprocess_id": subprocess_id,
                }
            )
        stages.append({"id": stage_id, "title": f"Stage {stage:02d}", "node_ids": stage_node_ids})
        groups.append({"id": group_id, "label": f"Group {stage:02d}", "node_ids": stage_node_ids})
        subprocesses.append({"id": subprocess_id, "name": f"Subprocess {stage:02d}", "node_ids": stage_node_ids})

        def add_edge(source: str, target: str, label: str = "") -> None:
            nonlocal edge_index
            edges.append({"id": f"e{edge_index:04d}", "source": source, "target": target, "label": label})
            edge_index += 1

        for index in range(1, 8):
            add_edge(node_id(stage, index), node_id(stage, index + 1))
        add_edge(node_id(stage, 8), node_id(stage, 9))
        add_edge(node_id(stage, 8), node_id(stage, 10))
        add_edge(node_id(stage, 9), node_id(stage, 11))
        add_edge(node_id(stage, 10), node_id(stage, 11))
        for index in range(11, 15):
            add_edge(node_id(stage, index), node_id(stage, index + 1))
        add_edge(node_id(stage, 15), node_id(stage, 16), "success")
        add_edge(node_id(stage, 15), node_id(stage, 17), "failure")
        add_edge(node_id(stage, 17), node_id(stage, 15), "retry")
        add_edge(node_id(stage, 10), "manual_review", "exception")
        add_edge(node_id(stage, 16), node_id(stage, 18))
        add_edge(node_id(stage, 17), node_id(stage, 18))
        for index in range(18, nodes_per_stage):
            add_edge(node_id(stage, index), node_id(stage, index + 1))
        if stage < stage_count:
            add_edge(node_id(stage, nodes_per_stage), node_id(stage + 1, 1), "success")

    return {"nodes": nodes, "edges": edges, "stages": stages, "groups": groups, "subprocesses": subprocesses}


def nodes_by_id(graph: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {node["id"]: node for node in graph.get("nodes", []) if isinstance(node, dict) and isinstance(node.get("id"), str)}


def edge_key(edge: dict[str, Any]) -> tuple[str, str, str]:
    return (str(edge.get("source")), str(edge.get("target")), str(edge.get("label", "")))


def edge_set(graph: dict[str, Any]) -> set[tuple[str, str, str]]:
    return {edge_key(edge) for edge in graph.get("edges", []) if isinstance(edge, dict)}


def graph_unchanged(before: dict[str, Any], after: dict[str, Any] | None) -> bool:
    return isinstance(after, dict) and before == after


def changed_node_ids(before: dict[str, Any], after: dict[str, Any] | None) -> set[str]:
    if not isinstance(after, dict):
        return set()
    before_nodes = nodes_by_id(before)
    after_nodes = nodes_by_id(after)
    return {
        nid
        for nid in set(before_nodes) | set(after_nodes)
        if before_nodes.get(nid) != after_nodes.get(nid)
    }


def changed_edges(before: dict[str, Any], after: dict[str, Any] | None) -> dict[str, list[list[str]]]:
    if not isinstance(after, dict):
        return {"added": [], "removed": []}
    added = sorted(edge_set(after) - edge_set(before))
    removed = sorted(edge_set(before) - edge_set(after))
    return {"added": [list(row) for row in added], "removed": [list(row) for row in removed]}


def resolved_scope_ids(result: Any) -> list[str]:
    specification = result.specification if isinstance(result.specification, dict) else {}
    requirements = specification.get("grounded_requirements", []) if isinstance(specification, dict) else []
    ids: set[str] = set()
    for requirement in requirements:
        desired = requirement.get("grounded_desired_state", {}) if isinstance(requirement, dict) else {}
        policy = desired.get("scope_policy", {}) if isinstance(desired, dict) else {}
        ids.update(node_id for node_id in policy.get("node_ids", []) if isinstance(node_id, str))
    return sorted(ids)


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


def relation_label_counts(graph: dict[str, Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for edge in graph.get("edges", []):
        label = str(edge.get("label", ""))
        counts[label] = counts.get(label, 0) + 1
    return counts


def relation_sources(graph: dict[str, Any], label: str) -> set[str]:
    return {edge["source"] for edge in graph.get("edges", []) if isinstance(edge, dict) and edge.get("label", "") == label}


def stage_ids(stage: int) -> list[str]:
    return [node_id(stage, index) for index in range(1, NODES_PER_STAGE + 1)]


def evaluate_stage_rename(before: dict[str, Any], result: Any) -> tuple[bool, bool, bool, dict[str, Any]]:
    after = result.final_graph
    target = "s04_n05"
    if result.status != "success" or not isinstance(after, dict):
        return False, False, False, {}
    before_nodes = nodes_by_id(before)
    after_nodes = nodes_by_id(after)
    expected_changes = {target}
    correct = after_nodes.get(target, {}).get("label") == "Stage 04 Compliance Review"
    outside = changed_node_ids(before, after) <= expected_changes and edge_set(before) == edge_set(after)
    return correct and outside, False, outside, {"changed_node_ids": sorted(changed_node_ids(before, after))}


def evaluate_visible_move(before: dict[str, Any], result: Any) -> tuple[bool, bool, bool, dict[str, Any]]:
    after = result.final_graph
    visible_area = {"x": 1800, "y": -10, "width": 980, "height": 210}
    expected = set(visible_node_ids(before, visible_area))
    if result.status != "success" or not isinstance(after, dict):
        return False, False, False, {"expected_visible_ids": sorted(expected)}
    before_nodes = nodes_by_id(before)
    after_nodes = nodes_by_id(after)
    correct = True
    for nid, node in before_nodes.items():
        expected_x = node["x"] + DX if nid in expected else node["x"]
        if after_nodes.get(nid, {}).get("x") != expected_x or after_nodes.get(nid, {}).get("y") != node["y"]:
            correct = False
            break
    outside = changed_node_ids(before, after) <= expected and edge_set(before) == edge_set(after)
    return correct and outside, False, outside, {"expected_visible_ids": sorted(expected), "changed_node_ids": sorted(changed_node_ids(before, after))}


def evaluate_global_failure_redirect(before: dict[str, Any], result: Any) -> tuple[bool, bool, bool, dict[str, Any]]:
    after = result.final_graph
    if result.status != "success" or not isinstance(after, dict):
        return False, False, False, {}
    before_fail_sources = relation_sources(before, "failure")
    after_fail_edges = {(edge["source"], edge["target"]) for edge in after["edges"] if edge.get("label", "") == "failure"}
    expected_fail_edges = {(source, "manual_review") for source in before_fail_sources}
    non_failure_preserved = {
        key for key in edge_set(before) if key[2] != "failure"
    } == {
        key for key in edge_set(after) if key[2] != "failure"
    }
    nodes_preserved = nodes_by_id(before) == nodes_by_id(after)
    correct = after_fail_edges == expected_fail_edges and len(after_fail_edges) == len(before_fail_sources)
    return correct and non_failure_preserved and nodes_preserved, False, non_failure_preserved and nodes_preserved, {
        "failure_sources": sorted(before_fail_sources),
        "actual_failure_edges": sorted([list(row) for row in after_fail_edges]),
    }


def evaluate_duplicate_ambiguity(before: dict[str, Any], result: Any) -> tuple[bool, bool, bool, dict[str, Any]]:
    correct = result.status == "needs_clarification" and bool(
        isinstance(result.specification, dict)
        and isinstance(result.specification.get("clarification"), dict)
        and result.specification["clarification"].get("question")
    )
    unchanged = result.final_graph is None or graph_unchanged(before, result.final_graph)
    return False, correct and unchanged, unchanged, {}


def evaluate_labeled_relation_edit(before: dict[str, Any], result: Any) -> tuple[bool, bool, bool, dict[str, Any]]:
    after = result.final_graph
    if result.status != "success" or not isinstance(after, dict):
        return False, False, False, {}
    source = "s05_n15"
    expected_added = {(source, "s05_n18", "success")}
    expected_removed = {(source, "s05_n16", "success")}
    added = edge_set(after) - edge_set(before)
    removed = edge_set(before) - edge_set(after)
    label_counts_ok = relation_label_counts(before) == relation_label_counts(after)
    nodes_preserved = nodes_by_id(before) == nodes_by_id(after)
    correct = added == expected_added and removed == expected_removed
    return correct and label_counts_ok and nodes_preserved, False, label_counts_ok and nodes_preserved, {
        "added_edges": sorted([list(row) for row in added]),
        "removed_edges": sorted([list(row) for row in removed]),
        "label_counts_before": relation_label_counts(before),
        "label_counts_after": relation_label_counts(after),
    }


def build_cases(graph: dict[str, Any]) -> list[LargeGraphCase]:
    visible_area = {"x": 1800, "y": -10, "width": 980, "height": 210}
    visible_ids = visible_node_ids(graph, visible_area)
    return [
        LargeGraphCase(
            "case_1_stage_local_rename",
            "Rename Review in Stage 04 to Stage 04 Compliance Review.",
            {"node_ids": [], "edge_ids": []},
            None,
            stage_ids(4),
            evaluate_stage_rename,
        ),
        LargeGraphCase(
            "case_2_visible_area_layout_move",
            "Move the currently visible nodes 80 pixels to the right.",
            {"node_ids": [], "edge_ids": []},
            {"visible_area": visible_area},
            visible_ids,
            evaluate_visible_move,
        ),
        LargeGraphCase(
            "case_3_global_failure_redirect",
            "Across the entire workflow, redirect every failure relation to manual_review.",
            {"node_ids": [node_id(1, 5)], "edge_ids": []},
            None,
            [node["id"] for node in graph["nodes"]],
            evaluate_global_failure_redirect,
        ),
        LargeGraphCase(
            "case_4_duplicate_name_ambiguity",
            "Rename Review to Compliance Review.",
            {"node_ids": [], "edge_ids": []},
            None,
            [],
            evaluate_duplicate_ambiguity,
        ),
        LargeGraphCase(
            "case_5_labeled_relation_edit",
            "Send the success relation from s05_n15 to s05_n18 instead of s05_n16.",
            {"node_ids": [], "edge_ids": []},
            None,
            [],
            evaluate_labeled_relation_edit,
        ),
    ]


def run_case(root: Path, graph: dict[str, Any], case: LargeGraphCase) -> dict[str, Any]:
    case_dir = root / case.name
    case_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    exception_text: str | None = None
    try:
        result = run_transaction(deepcopy(graph), 1, deepcopy(case.selection), case.request, request_context=deepcopy(case.request_context))
    except Exception as error:  # Experiment runner records failures; product code stays untouched.
        result = None
        exception_text = f"{type(error).__name__}: {error}"
    latency = time.perf_counter() - started

    if result is None:
        row = {
            "case": case.name,
            "request": case.request,
            "status": "runner_error",
            "correct_execution": False,
            "correct_clarification": False,
            "scope_correct": False,
            "outside_scope_unchanged": False,
            "invalid_graph": False,
            "unauthorized_mutation": False,
            "unintended_change": False,
            "latency": latency,
            "token_cost": None,
            "first_failure_layer": "Runner/API",
            "error": exception_text,
        }
        save_json(case_dir / "record.json", row)
        return row

    correct_execution, correct_clarification, outside_scope_unchanged, details = case.evaluator(graph, result)
    final_graph = result.final_graph
    outside_scope_unchanged = outside_scope_unchanged or final_graph is None
    invalid_graph = isinstance(final_graph, dict) and not graph_is_valid(final_graph)
    unauthorized_mutation = bool(result.error and "authorized" in str(result.error).lower())
    resolved = resolved_scope_ids(result)
    scope_correct = sorted(case.expected_scope_ids) == resolved if case.expected_scope_ids else correct_clarification or bool(correct_execution)
    unintended_change = result.status == "success" and not outside_scope_unchanged
    row = {
        "case": case.name,
        "node_count": len(graph["nodes"]),
        "edge_count": len(graph["edges"]),
        "stage_count": len(graph.get("stages", [])),
        "request": case.request,
        "selection": case.selection,
        "request_context": case.request_context,
        "expected_scope_ids": sorted(case.expected_scope_ids),
        "resolved_scope_ids": resolved,
        "status": result.status,
        "correct_execution": bool(correct_execution and not invalid_graph and not unauthorized_mutation and not unintended_change),
        "correct_clarification": bool(correct_clarification),
        "scope_correct": bool(scope_correct),
        "outside_scope_unchanged": bool(outside_scope_unchanged),
        "invalid_graph": bool(invalid_graph),
        "unauthorized_mutation": bool(unauthorized_mutation),
        "unintended_change": bool(unintended_change),
        "latency": latency,
        "token_cost": None,
        "first_failure_layer": None if (correct_execution or correct_clarification) and not invalid_graph and not unauthorized_mutation and not unintended_change else first_failure_layer(result) or "Evaluator",
        "changed_nodes": sorted(changed_node_ids(graph, final_graph)),
        "changed_edges": changed_edges(graph, final_graph),
        "details": details,
    }
    save_json(case_dir / "input_graph.json", graph)
    save_json(
        case_dir / "transaction_result.json",
        {
            "status": result.status,
            "error": result.error,
            "planning_path": result.planning_path,
            "steps": result.steps,
            "final_graph": result.final_graph,
            "target": result.target,
            "graph_context": result.graph_context,
            "specification": result.specification,
            "mutation_authorization": result.mutation_authorization,
        },
    )
    save_json(case_dir / "record.json", row)
    return row


def write_report(summary: dict[str, Any]) -> None:
    DOC_PATH.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Stage 3 P0-2 Large Graph Acceptance",
        "",
        f"- Run root: `{summary['root']}`",
        f"- Node count: `{summary['graph']['node_count']}`",
        f"- Edge count: `{summary['graph']['edge_count']}`",
        f"- Stage count: `{summary['graph']['stage_count']}`",
        "",
        "## Totals",
        "",
        "```json",
        json.dumps(summary["totals"], ensure_ascii=False, indent=2),
        "```",
        "",
        "## Cases",
        "",
    ]
    for row in summary["cases"]:
        lines.extend(
            [
                f"### {row['case']}",
                "",
                f"- Status: `{row['status']}`",
                f"- Correct execution: `{row['correct_execution']}`",
                f"- Correct clarification: `{row['correct_clarification']}`",
                f"- Scope correct: `{row['scope_correct']}`",
                f"- Outside scope unchanged: `{row['outside_scope_unchanged']}`",
                f"- First failure layer: `{row['first_failure_layer']}`",
                "",
            ]
        )
    DOC_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    graph = generate_complex_workflow()
    cases = build_cases(graph)
    save_json(OUTPUT_ROOT / "fixture_graph.json", graph)
    rows = [run_case(OUTPUT_ROOT, graph, case) for case in cases]
    totals = {
        "total": len(rows),
        "correct_execution": sum(bool(row["correct_execution"]) for row in rows),
        "correct_clarification": sum(bool(row["correct_clarification"]) for row in rows),
        "scope_correct": sum(bool(row["scope_correct"]) for row in rows),
        "outside_scope_unchanged": sum(bool(row["outside_scope_unchanged"]) for row in rows),
        "invalid_graph": sum(bool(row["invalid_graph"]) for row in rows),
        "unauthorized_mutation": sum(bool(row["unauthorized_mutation"]) for row in rows),
        "unintended_change": sum(bool(row["unintended_change"]) for row in rows),
        "runner_errors": sum(row["status"] == "runner_error" for row in rows),
    }
    summary = {
        "root": str(OUTPUT_ROOT),
        "graph": {
            "node_count": len(graph["nodes"]),
            "edge_count": len(graph["edges"]),
            "stage_count": len(graph["stages"]),
            "group_count": len(graph["groups"]),
            "subprocess_count": len(graph["subprocesses"]),
            "relation_label_counts": relation_label_counts(graph),
        },
        "totals": totals,
        "cases": rows,
    }
    save_json(OUTPUT_ROOT / "summary.json", summary)
    write_report(summary)
    print(json.dumps({"root": str(OUTPUT_ROOT), "doc": str(DOC_PATH), "totals": totals}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
