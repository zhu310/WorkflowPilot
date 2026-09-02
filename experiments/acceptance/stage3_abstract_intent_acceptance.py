from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from src.mvp23 import run_transaction
from src.runner import save_json


def _edge_label(edge: dict[str, Any]) -> str:
    value = edge.get("label", "")
    return value if isinstance(value, str) else ""


def _positions(graph: dict[str, Any]) -> dict[str, tuple[int, int]]:
    return {
        node["id"]: (
            int(node.get("x", 0)) if isinstance(node.get("x"), (int, float)) else 0,
            int(node.get("y", 0)) if isinstance(node.get("y"), (int, float)) else 0,
        )
        for node in graph.get("nodes", [])
        if isinstance(node, dict) and isinstance(node.get("id"), str)
    }


def _node_ids(graph: dict[str, Any]) -> set[str]:
    return {node["id"] for node in graph.get("nodes", []) if isinstance(node, dict) and isinstance(node.get("id"), str)}


def _edges(graph: dict[str, Any], label: str | None = None) -> set[tuple[str, str, str]]:
    pairs = set()
    for edge in graph.get("edges", []):
        if not isinstance(edge, dict):
            continue
        source = edge.get("source")
        target = edge.get("target")
        relation_type = _edge_label(edge)
        if not isinstance(source, str) or not isinstance(target, str):
            continue
        if label is not None and relation_type != label:
            continue
        pairs.add((source, target, relation_type))
    return pairs


def _has_path(graph: dict[str, Any], source: str, target: str, allowed_labels: set[str] | None = None) -> bool:
    frontier = [source]
    seen = {source}
    adjacency: dict[str, list[str]] = {}
    for edge in graph.get("edges", []):
        if not isinstance(edge, dict):
            continue
        relation_type = _edge_label(edge)
        if allowed_labels is not None and relation_type not in allowed_labels:
            continue
        src = edge.get("source")
        dst = edge.get("target")
        if isinstance(src, str) and isinstance(dst, str):
            adjacency.setdefault(src, []).append(dst)
    while frontier:
        current = frontier.pop(0)
        if current == target:
            return True
        for nxt in adjacency.get(current, []):
            if nxt not in seen:
                seen.add(nxt)
                frontier.append(nxt)
    return False


def _graph_diff(before: dict[str, Any], after: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(after, dict):
        return {}
    before_nodes = {node["id"]: node for node in before.get("nodes", []) if isinstance(node, dict) and isinstance(node.get("id"), str)}
    after_nodes = {node["id"]: node for node in after.get("nodes", []) if isinstance(node, dict) and isinstance(node.get("id"), str)}
    changed_nodes = []
    for node_id in sorted(before_nodes.keys() & after_nodes.keys()):
        changes = {}
        for key in ("label", "x", "y", "join"):
            if before_nodes[node_id].get(key) != after_nodes[node_id].get(key):
                changes[key] = {"before": before_nodes[node_id].get(key), "after": after_nodes[node_id].get(key)}
        if changes:
            changed_nodes.append({"node_id": node_id, "changes": changes})
    return {
        "added_nodes": sorted(after_nodes.keys() - before_nodes.keys()),
        "removed_nodes": sorted(before_nodes.keys() - after_nodes.keys()),
        "changed_nodes": changed_nodes,
        "added_edges": sorted(_edges(after) - _edges(before)),
        "removed_edges": sorted(_edges(before) - _edges(after)),
    }


def _stop_layer(result: Any) -> str | None:
    specification = result.specification if isinstance(result.specification, dict) else {}
    if isinstance(specification, dict) and isinstance(specification.get("stop_layer"), str):
        return specification["stop_layer"]
    return {
        "needs_clarification": "Ground",
        "unsupported": "Workflow IR",
        "planning_error": "Planner",
        "execution_error": "Executor",
        "interpretation_error": "Understand",
    }.get(result.status)


def _trace(result: Any) -> dict[str, Any]:
    return result.specification if isinstance(result.specification, dict) else {}


def _clarification_ok(result: Any) -> bool:
    trace = _trace(result)
    question = trace.get("clarification", {}).get("question") if isinstance(trace.get("clarification"), dict) else None
    return result.status == "needs_clarification" and isinstance(question, str) and bool(question.strip())


def _all_labels_preserved(before: dict[str, Any], after: dict[str, Any], node_ids: set[str]) -> bool:
    after_nodes = {node["id"]: node for node in after.get("nodes", []) if isinstance(node, dict) and isinstance(node.get("id"), str)}
    before_nodes = {node["id"]: node for node in before.get("nodes", []) if isinstance(node, dict) and isinstance(node.get("id"), str)}
    return all(before_nodes[node_id].get("label") == after_nodes.get(node_id, {}).get("label") for node_id in node_ids if node_id in before_nodes)


@dataclass(frozen=True)
class Case:
    family: str
    name: str
    graph: dict[str, Any]
    instruction: str
    selection: dict[str, Any]
    request_context: dict[str, Any] | None
    evaluator: Callable[[Any, dict[str, Any]], dict[str, Any]]


def _result_row(case: Case, result: Any) -> dict[str, Any]:
    trace = _trace(result)
    actual = case.evaluator(result, case.graph)
    clarification = trace.get("clarification", {}) if isinstance(trace.get("clarification"), dict) else {}
    row = {
        "family": case.family,
        "name": case.name,
        "instruction": case.instruction,
        "selection": case.selection,
        "request_context": case.request_context,
        "status": result.status,
        "error": result.error,
        "structured_interpretation": trace.get("grounded_requirements", []),
        "grounded_requirements": trace.get("grounded_requirements", []),
        "preserve_constraints": trace.get("global_obligations", {}),
        "deterministic_obligations": [unit.get("obligations", {}) for unit in trace.get("unit_runs", []) if isinstance(unit, dict)],
        "semantic_slots": [unit.get("semantic_slots", {}) for unit in trace.get("unit_runs", []) if isinstance(unit, dict)],
        "actual_outcome": {
            "status": result.status,
            "error": result.error,
            "clarification_message": getattr(result, "clarification_message", None),
            "steps": result.steps,
        },
        "actual_graph_diff": _graph_diff(case.graph, result.final_graph),
        "clarification_reason": clarification.get("reason") or result.error,
        "correct_execution": actual["correct_execution"],
        "correct_clarification": actual["correct_clarification"],
        "unsupported": result.status == "unsupported",
        "false_success": result.status == "success" and not actual["correct_execution"] and result.final_graph == case.graph,
        "incorrect_execution": actual["incorrect_execution"],
        "unintended_change": actual["unintended_change"],
        "first_failure_layer": None if (actual["correct_execution"] or actual["correct_clarification"]) else _stop_layer(result),
        "final_graph": result.final_graph,
        "trace": trace,
    }
    return row


def _success_metrics(
    *,
    correct_execution: bool,
    correct_clarification: bool = False,
    incorrect_execution: bool = False,
    unintended_change: bool = False,
) -> dict[str, Any]:
    return {
        "correct_execution": correct_execution,
        "correct_clarification": correct_clarification,
        "incorrect_execution": incorrect_execution,
        "unintended_change": unintended_change,
    }


def _graph_t1() -> dict[str, Any]:
    return {
        "nodes": [
            {"id": "entry", "type": "task", "label": "Entry", "x": 40, "y": 80},
            {"id": "collect", "type": "task", "label": "Collect", "x": 200, "y": 80},
            {"id": "verify", "type": "task", "label": "Verify", "x": 360, "y": 80},
            {"id": "fix", "type": "task", "label": "Fix", "x": 360, "y": 200},
            {"id": "approve", "type": "task", "label": "Approve", "x": 520, "y": 80},
        ],
        "edges": [
            {"id": "e1", "source": "entry", "target": "collect", "label": ""},
            {"id": "e2", "source": "collect", "target": "verify", "label": ""},
            {"id": "e3", "source": "verify", "target": "fix", "label": "rework"},
            {"id": "e4", "source": "fix", "target": "verify", "label": "retry"},
            {"id": "e5", "source": "verify", "target": "approve", "label": ""},
        ],
        "stages": [{"id": "review_stage", "title": "Review Stage", "node_ids": ["entry", "collect", "verify", "fix", "approve"]}],
    }


def _graph_t2() -> dict[str, Any]:
    return {
        "nodes": [
            {"id": "entry", "type": "task", "label": "Entry", "x": 40, "y": 80},
            {"id": "check_a", "type": "task", "label": "Identity Check", "x": 220, "y": 40},
            {"id": "check_b", "type": "task", "label": "Budget Check", "x": 220, "y": 160},
            {"id": "merge", "type": "task", "label": "Merge Review", "x": 420, "y": 100},
            {"id": "audit", "type": "task", "label": "Audit Check", "x": 620, "y": 100},
            {"id": "approve", "type": "task", "label": "Approve", "x": 820, "y": 100},
        ],
        "edges": [
            {"id": "e1", "source": "entry", "target": "check_a", "label": ""},
            {"id": "e2", "source": "check_a", "target": "merge", "label": ""},
            {"id": "e3", "source": "entry", "target": "check_b", "label": ""},
            {"id": "e4", "source": "check_b", "target": "merge", "label": ""},
            {"id": "e5", "source": "merge", "target": "audit", "label": ""},
            {"id": "e6", "source": "audit", "target": "approve", "label": ""},
        ],
        "stages": [{"id": "approval_stage", "title": "Approval Stage", "node_ids": ["entry", "check_a", "check_b", "merge", "audit", "approve"]}],
    }


def _graph_t3() -> dict[str, Any]:
    return {
        "nodes": [
            {"id": "start", "type": "task", "label": "Start", "x": 40, "y": 100},
            {"id": "auto_check", "type": "task", "label": "Auto Check", "x": 220, "y": 100},
            {"id": "manual_review", "type": "task", "label": "Manual Review", "x": 420, "y": 100},
            {"id": "approve", "type": "task", "label": "Approve", "x": 640, "y": 100},
            {"id": "archive", "type": "task", "label": "Archive", "x": 840, "y": 100},
        ],
        "edges": [
            {"id": "e1", "source": "start", "target": "auto_check", "label": ""},
            {"id": "e2", "source": "auto_check", "target": "manual_review", "label": ""},
            {"id": "e3", "source": "manual_review", "target": "approve", "label": ""},
            {"id": "e4", "source": "approve", "target": "archive", "label": ""},
            {"id": "e5", "source": "auto_check", "target": "manual_review", "label": "uncertain"},
            {"id": "e6", "source": "manual_review", "target": "archive", "label": "failure"},
        ],
    }


def _graph_t4() -> dict[str, Any]:
    return {
        "nodes": [
            {"id": "start", "type": "task", "label": "Start", "x": 60, "y": 120},
            {"id": "ingest", "type": "task", "label": "Ingest", "x": 260, "y": 120},
            {"id": "approve", "type": "task", "label": "Approve", "x": 500, "y": 120},
            {"id": "end", "type": "task", "label": "End", "x": 760, "y": 120},
            {"id": "manual", "type": "task", "label": "Manual Review", "x": 500, "y": 280},
            {"id": "notify", "type": "task", "label": "Notify", "x": 760, "y": 280},
        ],
        "edges": [
            {"id": "e1", "source": "start", "target": "ingest", "label": ""},
            {"id": "e2", "source": "ingest", "target": "approve", "label": ""},
            {"id": "e3", "source": "approve", "target": "end", "label": ""},
            {"id": "e4", "source": "approve", "target": "manual", "label": "exception"},
            {"id": "e5", "source": "manual", "target": "notify", "label": "retry"},
            {"id": "e6", "source": "notify", "target": "end", "label": ""},
        ],
    }


def _graph_t5() -> dict[str, Any]:
    return {
        "nodes": [
            {"id": "draft", "type": "task", "label": "Draft", "x": 40, "y": 100},
            {"id": "review", "type": "task", "label": "Review", "x": 240, "y": 100},
            {"id": "approve", "type": "task", "label": "Approve", "x": 440, "y": 100},
            {"id": "pay", "type": "task", "label": "Pay", "x": 640, "y": 100},
        ],
        "edges": [
            {"id": "e1", "source": "draft", "target": "review", "label": ""},
            {"id": "e2", "source": "review", "target": "approve", "label": ""},
            {"id": "e3", "source": "approve", "target": "pay", "label": ""},
        ],
    }


def _backtracking_score(graph: dict[str, Any]) -> int:
    positions = _positions(graph)
    score = 0
    for edge in graph.get("edges", []):
        if not isinstance(edge, dict):
            continue
        source = edge.get("source")
        target = edge.get("target")
        if source in positions and target in positions and positions[target][0] < positions[source][0]:
            score += 1
    return score


def _t1_eval(result: Any, graph: dict[str, Any]) -> dict[str, Any]:
    if result.status == "needs_clarification":
        return _success_metrics(correct_execution=False, correct_clarification=_clarification_ok(result))
    if result.status != "success" or not isinstance(result.final_graph, dict):
        return _success_metrics(correct_execution=False, incorrect_execution=result.status == "success")
    final_graph = result.final_graph
    required = _node_ids(graph) <= _node_ids(final_graph)
    preserved_business = _edges(graph, "rework") <= _edges(final_graph) and _edges(graph, "retry") <= _edges(final_graph)
    changed = final_graph != graph
    improved = _backtracking_score(final_graph) < _backtracking_score(graph)
    correct = required and preserved_business and changed and improved
    unintended = bool(_node_ids(graph) - _node_ids(final_graph))
    return _success_metrics(correct_execution=correct, incorrect_execution=not correct, unintended_change=unintended)


def _structural_complexity(graph: dict[str, Any]) -> tuple[int, int]:
    return (len(graph.get("edges", [])), len({node.get("id") for node in graph.get("nodes", []) if isinstance(node, dict)}))


def _t2_eval(result: Any, graph: dict[str, Any]) -> dict[str, Any]:
    if result.status == "needs_clarification":
        return _success_metrics(correct_execution=False, correct_clarification=_clarification_ok(result))
    if result.status != "success" or not isinstance(result.final_graph, dict):
        return _success_metrics(correct_execution=False, incorrect_execution=result.status == "success")
    final_graph = result.final_graph
    check_ids = {"check_a", "check_b", "audit"}
    preserved = check_ids <= _node_ids(final_graph)
    labels_preserved = _all_labels_preserved(graph, final_graph, check_ids)
    changed = final_graph != graph
    renamed = any(node.get("id") == "merge" and node.get("label") == "Unified Review" for node in final_graph.get("nodes", []))
    complexity_ok = _structural_complexity(final_graph)[0] <= _structural_complexity(graph)[0]
    correct = preserved and labels_preserved and changed and renamed and complexity_ok
    return _success_metrics(correct_execution=correct, incorrect_execution=not correct, unintended_change=False)


def _t3_eval(result: Any, graph: dict[str, Any]) -> dict[str, Any]:
    if result.status == "needs_clarification":
        return _success_metrics(correct_execution=False, correct_clarification=_clarification_ok(result))
    if result.status != "success" or not isinstance(result.final_graph, dict):
        return _success_metrics(correct_execution=False, incorrect_execution=result.status == "success")
    final_graph = result.final_graph
    normal_flow_skips_manual = _has_path(final_graph, "auto_check", "approve", {""}) and not _has_path(final_graph, "auto_check", "manual_review", {""})
    uncertain_keeps_manual = ("auto_check", "manual_review", "uncertain") in _edges(final_graph)
    failure_keeps_manual = ("manual_review", "archive", "failure") in _edges(final_graph)
    changed = final_graph != graph
    correct = normal_flow_skips_manual and uncertain_keeps_manual and failure_keeps_manual and changed
    unintended = ("auto_check", "manual_review", "uncertain") not in _edges(final_graph) or ("manual_review", "archive", "failure") not in _edges(final_graph)
    return _success_metrics(correct_execution=correct, incorrect_execution=not correct, unintended_change=unintended)


def _t4_eval(result: Any, graph: dict[str, Any]) -> dict[str, Any]:
    if result.status == "needs_clarification":
        return _success_metrics(correct_execution=False, correct_clarification=_clarification_ok(result))
    if result.status != "success" or not isinstance(result.final_graph, dict):
        return _success_metrics(correct_execution=False, incorrect_execution=result.status == "success")
    final_graph = result.final_graph
    topology_preserved = _edges(graph) == _edges(final_graph)
    changed = final_graph != graph
    before = _positions(graph)
    after = _positions(final_graph)
    main_readability_improved = after["ingest"][1] == after["approve"][1] == after["end"][1] and abs(after["manual"][1] - after["approve"][1]) >= abs(before["manual"][1] - before["approve"][1])
    side_preserved = ("approve", "manual", "exception") in _edges(final_graph) and ("manual", "notify", "retry") in _edges(final_graph)
    correct = topology_preserved and changed and main_readability_improved and side_preserved
    return _success_metrics(correct_execution=correct, incorrect_execution=not correct, unintended_change=not topology_preserved)


def _t5_eval(result: Any, _graph: dict[str, Any]) -> dict[str, Any]:
    return _success_metrics(correct_execution=False, correct_clarification=_clarification_ok(result), incorrect_execution=result.status == "success")


def original_cases() -> list[Case]:
    return [
        Case("T1", "reduce_backtracking", _graph_t1(), "Reduce unnecessary backtracking in this stage, but keep all processing steps.", {"node_ids": [], "edge_ids": []}, None, _t1_eval),
        Case("T2", "simplify_without_removing_checks", _graph_t2(), "This stage is too complex. Simplify it, but do not remove any check steps, and rename Merge Review to Unified Review.", {"node_ids": [], "edge_ids": []}, None, _t2_eval),
        Case("T3", "automate_normal_path", _graph_t3(), "Automate the normal path as much as possible. Only send uncertain cases to manual review.", {"node_ids": [], "edge_ids": []}, None, _t3_eval),
        Case("T4", "highlight_main_flow", _graph_t4(), "Make the main flow more prominent, and keep side paths from distracting from the main line.", {"node_ids": [], "edge_ids": []}, None, _t4_eval),
        Case("T5", "maintainability_clarification", _graph_t5(), "Keep all functionality, but make the whole workflow easier to maintain.", {"node_ids": [], "edge_ids": []}, None, _t5_eval),
    ]


def variant_cases() -> list[Case]:
    t1_cn = deepcopy(_graph_t1())
    t1_cn["stages"][0]["title"] = "审核阶段"
    t3_variant = deepcopy(_graph_t3())
    t3_variant["nodes"][1]["label"] = "Rule Engine"
    t3_variant["nodes"][2]["label"] = "Analyst Review"
    t4_cn = deepcopy(_graph_t4())
    return [
        Case("T1_cn", "reduce_backtracking_cn", t1_cn, "减少这个阶段里不必要的来回跳转，但保留所有处理步骤。", {"node_ids": [], "edge_ids": []}, None, _t1_eval),
        Case("T2_variant", "simplify_without_removing_checks_variant", _graph_t2(), "Streamline this approval stage, keep every required check, and call Merge Review Unified Review.", {"node_ids": [], "edge_ids": []}, None, _t2_eval),
        Case("T3_variant", "automate_normal_path_variant", t3_variant, "Let the default path finish automatically, and send only uncertain cases to analyst review.", {"node_ids": [], "edge_ids": []}, None, _t3_eval),
        Case("T4_cn", "highlight_main_flow_cn", t4_cn, "让主流程更突出，旁路不要干扰主线阅读。", {"node_ids": [], "edge_ids": []}, None, _t4_eval),
        Case("T5_cn", "maintainability_clarification_cn", _graph_t5(), "保留所有功能，让整个流程更容易维护。", {"node_ids": [], "edge_ids": []}, None, _t5_eval),
    ]


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for family in sorted({row["family"] for row in rows}):
        family_rows = [row for row in rows if row["family"] == family]
        total = len(family_rows)
        summary[family] = {
            "total": total,
            "correct_execution": sum(row["correct_execution"] for row in family_rows),
            "correct_clarification": sum(row["correct_clarification"] for row in family_rows),
            "unsupported": sum(row["unsupported"] for row in family_rows),
            "false_success": sum(row["false_success"] for row in family_rows),
            "incorrect_execution": sum(row["incorrect_execution"] for row in family_rows),
            "unintended_change": sum(row["unintended_change"] for row in family_rows),
            "first_failure_layers": sorted({row["first_failure_layer"] for row in family_rows if row["first_failure_layer"]}),
        }
    summary["total"] = {
        "runs": len(rows),
        "correct_execution": sum(row["correct_execution"] for row in rows),
        "correct_clarification": sum(row["correct_clarification"] for row in rows),
        "unsupported": sum(row["unsupported"] for row in rows),
        "false_success": sum(row["false_success"] for row in rows),
        "incorrect_execution": sum(row["incorrect_execution"] for row in rows),
        "unintended_change": sum(row["unintended_change"] for row in rows),
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--attempts", type=int, default=5)
    parser.add_argument("--variants", action="store_true")
    args = parser.parse_args()

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    root = Path("experiments/runs") / f"stage3-abstract-intent-{stamp}"
    root.mkdir(parents=True, exist_ok=True)

    cases = original_cases() + (variant_cases() if args.variants else [])
    rows: list[dict[str, Any]] = []
    for case in cases:
        case_dir = root / case.name
        case_dir.mkdir(parents=True, exist_ok=True)
        for attempt in range(1, args.attempts + 1):
            result = run_transaction(deepcopy(case.graph), attempt, deepcopy(case.selection), case.instruction, request_context=deepcopy(case.request_context))
            row = _result_row(case, result)
            row["attempt"] = attempt
            rows.append(row)
            save_json(case_dir / f"attempt_{attempt:02d}.json", row)
    summary = _summarize(rows)
    save_json(root / "summary.json", summary)
    save_json(root / "rows.json", rows)
    print(root)
    print(summary)


if __name__ == "__main__":
    main()
