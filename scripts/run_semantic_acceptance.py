from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.semantic_shadow import graph_diff, run_semantic_shadow

DEFAULT_CASES = ROOT / "evaluation" / "semantic_acceptance_cases.jsonl"
DEFAULT_OUTPUT_DIR = ROOT / "runs" / "semantic_acceptance"


def edge_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return (row.get("source", ""), row.get("target", ""), row.get("label", ""))


def get_path(value: dict[str, Any], path: str) -> Any:
    current: Any = value
    for part in path.split("."):
        if not isinstance(current, dict):
            return None
        current = current.get(part)
    return current


def contains_changes(row: dict[str, Any], changes: dict[str, Any]) -> bool:
    for key, expected in changes.items():
        actual = get_path(row, key)
        if actual != expected:
            return False
    return True


def without_paths(value: dict[str, Any], paths: set[str]) -> dict[str, Any]:
    result = json.loads(json.dumps(value, ensure_ascii=False))

    def prune_empty_containers(current: Any) -> bool:
        if not isinstance(current, dict):
            return False
        empty_children = [key for key, child in current.items() if prune_empty_containers(child)]
        for key in empty_children:
            current.pop(key, None)
        return not current

    for path in paths:
        current: Any = result
        parts = path.split(".")
        for part in parts[:-1]:
            if not isinstance(current, dict):
                current = None
                break
            current = current.get(part)
        if isinstance(current, dict):
            current.pop(parts[-1], None)
    prune_empty_containers(result)
    return result


def strict_preserve_errors(before: dict[str, Any], final_graph: dict[str, Any], expected: dict[str, Any]) -> list[str]:
    before_nodes = {node.get("id"): node for node in before.get("nodes", []) if isinstance(node, dict)}
    final_nodes = {node.get("id"): node for node in final_graph.get("nodes", []) if isinstance(node, dict)}
    before_edges_by_id = {edge.get("id"): edge for edge in before.get("edges", []) if isinstance(edge, dict)}
    final_edges_by_id = {edge.get("id"): edge for edge in final_graph.get("edges", []) if isinstance(edge, dict)}
    before_edge_keys = {edge_key(edge): edge for edge in before.get("edges", []) if isinstance(edge, dict)}
    final_edge_keys = {edge_key(edge): edge for edge in final_graph.get("edges", []) if isinstance(edge, dict)}

    allowed_added_nodes = set(expected.get("required_nodes", []))
    allowed_removed_nodes = set(expected.get("forbidden_nodes", []))
    allowed_added_edges = {edge_key(row) for row in expected.get("required_edges", []) if edge_key(row) not in before_edge_keys}
    allowed_removed_edges = {edge_key(row) for row in expected.get("forbidden_edges", [])}
    allowed_node_properties: dict[str, set[str]] = {}
    allowed_edge_properties: dict[str, set[str]] = {}
    for row in expected.get("required_properties", []):
        changes = row.get("changes", {})
        if not isinstance(changes, dict):
            continue
        if isinstance(row.get("node_id"), str):
            allowed_node_properties.setdefault(row["node_id"], set()).update(changes.keys())
        elif isinstance(row.get("edge_id"), str):
            allowed_edge_properties.setdefault(row["edge_id"], set()).update(changes.keys())

    errors: list[str] = []
    for node_id, before_node in before_nodes.items():
        final_node = final_nodes.get(node_id)
        if final_node is None:
            if node_id not in allowed_removed_nodes:
                errors.append(f"unexpected_node_change: removed {node_id}")
            continue
        allowed_paths = allowed_node_properties.get(node_id, set())
        if without_paths(before_node, allowed_paths) != without_paths(final_node, allowed_paths):
            errors.append(f"unexpected_property_change: node {node_id}")
    for node_id in final_nodes:
        if node_id not in before_nodes and node_id not in allowed_added_nodes:
            errors.append(f"unexpected_node_change: added {node_id}")

    for edge_id, before_edge in before_edges_by_id.items():
        final_edge = final_edges_by_id.get(edge_id)
        before_key = edge_key(before_edge)
        if final_edge is None:
            if before_key not in allowed_removed_edges:
                errors.append(f"unexpected_edge_change: removed {edge_id}")
            continue
        if before_key != edge_key(final_edge):
            continue
        allowed_paths = allowed_edge_properties.get(edge_id, set())
        if without_paths(before_edge, allowed_paths) != without_paths(final_edge, allowed_paths):
            errors.append(f"unexpected_property_change: edge {edge_id}")
    for edge_id, final_edge in final_edges_by_id.items():
        if edge_id not in before_edges_by_id and edge_key(final_edge) not in allowed_added_edges:
            errors.append(f"unexpected_edge_change: added {edge_id}")
    for key in before_edge_keys:
        if key not in final_edge_keys and key not in allowed_removed_edges:
            errors.append(f"unexpected_edge_change: missing relation {key}")
    for key in final_edge_keys:
        if key not in before_edge_keys and key not in allowed_added_edges:
            errors.append(f"unexpected_edge_change: extra relation {key}")
    return errors


def is_legal_new_ref(value: str) -> bool:
    return value.startswith("new_")


def preflight_expected(case_id: str, graph: dict[str, Any], expected: dict[str, Any], step: int | None = None) -> list[dict[str, Any]]:
    nodes = {node.get("id") for node in graph.get("nodes", []) if isinstance(node, dict)}
    edge_ids = {edge.get("id") for edge in graph.get("edges", []) if isinstance(edge, dict)}
    errors: list[dict[str, Any]] = []

    def add(kind: str, fields: Any) -> None:
        row: dict[str, Any] = {"case_id": case_id, "conflict_kind": kind, "fields": fields}
        if step is not None:
            row["step"] = step
        errors.append(row)

    required_nodes = set(expected.get("required_nodes", []))
    forbidden_nodes = set(expected.get("forbidden_nodes", []))
    node_conflicts = sorted(required_nodes & forbidden_nodes)
    if node_conflicts:
        add("required_forbidden_node_conflict", {"node_ids": node_conflicts})

    required_edges = {edge_key(row) for row in expected.get("required_edges", [])}
    forbidden_edges = {edge_key(row) for row in expected.get("forbidden_edges", [])}
    edge_conflicts = sorted(required_edges & forbidden_edges)
    if edge_conflicts:
        add("required_forbidden_edge_conflict", {"edges": edge_conflicts})

    preserved_nodes = set(expected.get("preserved_nodes", []))
    preserved_edges = set(expected.get("preserved_edges", []))
    for row in expected.get("required_properties", []):
        if isinstance(row.get("node_id"), str) and row["node_id"] in preserved_nodes:
            add("preserved_node_property_mutation_conflict", {"node_id": row["node_id"], "required_properties": row.get("changes", {})})
        if isinstance(row.get("edge_id"), str) and row["edge_id"] in preserved_edges:
            add("preserved_edge_property_mutation_conflict", {"edge_id": row["edge_id"], "required_properties": row.get("changes", {})})

    for node_id in required_nodes | forbidden_nodes | preserved_nodes:
        if node_id not in nodes and not is_legal_new_ref(str(node_id)):
            add("unknown_node_reference", {"node_id": node_id})
    for row in expected.get("required_edges", []) + expected.get("forbidden_edges", []):
        for field in ("source", "target"):
            value = str(row.get(field, ""))
            if value not in nodes and not is_legal_new_ref(value):
                add("unknown_edge_endpoint_reference", {"edge": row, "field": field, "node_id": value})
    for edge_id in preserved_edges:
        if edge_id not in edge_ids and not is_legal_new_ref(str(edge_id)):
            add("unknown_edge_reference", {"edge_id": edge_id})
    for row in expected.get("required_properties", []):
        if isinstance(row.get("node_id"), str):
            node_id = row["node_id"]
            if node_id not in nodes and not is_legal_new_ref(node_id):
                add("unknown_property_node_reference", {"node_id": node_id})
        if isinstance(row.get("edge_id"), str):
            edge_id = row["edge_id"]
            if edge_id not in edge_ids and not is_legal_new_ref(edge_id):
                add("unknown_property_edge_reference", {"edge_id": edge_id})
    return errors


def preflight_cases(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    errors: list[dict[str, Any]] = []
    for case in cases:
        case_id = str(case.get("id", ""))
        graph = case.get("initial_graph", {})
        steps = case.get("steps")
        if isinstance(steps, list):
            for index, step in enumerate(steps, start=1):
                expected = step.get("expected", {}) if isinstance(step.get("expected"), dict) else {}
                errors.extend(preflight_expected(case_id, graph, expected, index))
        else:
            expected = case.get("expected", {}) if isinstance(case.get("expected"), dict) else {}
            errors.extend(preflight_expected(case_id, graph, expected))
    return errors


def evaluate_expected(before: dict[str, Any], final_graph: dict[str, Any] | None, expected: dict[str, Any]) -> tuple[bool, str, list[str]]:
    if expected.get("failure_stage"):
        return True, "", []
    if not isinstance(final_graph, dict):
        return False, "final_graph_semantics", ["missing final graph"]

    errors: list[str] = []
    nodes = {node.get("id"): node for node in final_graph.get("nodes", []) if isinstance(node, dict)}
    before_nodes = {node.get("id"): node for node in before.get("nodes", []) if isinstance(node, dict)}
    edges = {edge_key(edge): edge for edge in final_graph.get("edges", []) if isinstance(edge, dict)}
    before_edges_by_id = {edge.get("id"): edge for edge in before.get("edges", []) if isinstance(edge, dict)}
    final_edges_by_id = {edge.get("id"): edge for edge in final_graph.get("edges", []) if isinstance(edge, dict)}

    for node_id in expected.get("required_nodes", []):
        if node_id not in nodes:
            errors.append(f"required node missing: {node_id}")
    for node_id in expected.get("forbidden_nodes", []):
        if node_id in nodes:
            errors.append(f"forbidden node still present: {node_id}")
    for row in expected.get("required_edges", []):
        key = edge_key(row)
        if key not in edges:
            errors.append(f"required edge missing: {key}")
    for row in expected.get("forbidden_edges", []):
        key = edge_key(row)
        if key in edges:
            errors.append(f"forbidden edge still present: {key}")
    for row in expected.get("required_properties", []):
        if isinstance(row.get("node_id"), str):
            target = nodes.get(row["node_id"])
            if not isinstance(target, dict) or not contains_changes(target, row.get("changes", {})):
                errors.append(f"node property mismatch: {row}")
        elif isinstance(row.get("edge_id"), str):
            target = final_edges_by_id.get(row["edge_id"])
            if not isinstance(target, dict) or not contains_changes(target, row.get("changes", {})):
                errors.append(f"edge property mismatch: {row}")
    for node_id in expected.get("preserved_nodes", []):
        if before_nodes.get(node_id) != nodes.get(node_id):
            errors.append(f"preserved node changed: {node_id}")
    for edge_id in expected.get("preserved_edges", []):
        if before_edges_by_id.get(edge_id) != final_edges_by_id.get(edge_id):
            errors.append(f"preserved edge changed: {edge_id}")
    if expected.get("strict_preserve_others") is True:
        errors.extend(strict_preserve_errors(before, final_graph, expected))

    if errors:
        preservation_errors = [error for error in errors if error.startswith("preserved") or error.startswith("unexpected_")]
        return False, "preservation_violation" if preservation_errors else "final_graph_semantics", errors
    return True, "", []


def load_cases(path: Path) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for index, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"case line {index} is not an object")
        cases.append(value)
    return cases


def run_case(case: dict[str, Any]) -> dict[str, Any]:
    graph = case["initial_graph"]
    selection = case.get("selection", {"node_ids": [], "edge_ids": []})
    request_context = case.get("request_context", {})
    steps = case.get("steps")
    if not isinstance(steps, list):
        steps = [{"instruction": case.get("instruction", ""), "expected": case.get("expected", {})}]

    step_results = []
    current_graph = graph
    passed = True
    failure_stage = ""
    failure_reason: Any = ""
    for step_index, step in enumerate(steps, start=1):
        before = current_graph
        instruction = str(step.get("instruction", ""))
        expected = step.get("expected", {}) if isinstance(step.get("expected"), dict) else {}
        shadow = run_semantic_shadow(before, instruction, selection, request_context)
        final_graph = shadow.get("final_shadow_graph") if isinstance(shadow.get("final_shadow_graph"), dict) else None

        expected_stage = expected.get("failure_stage")
        if expected_stage:
            step_passed = shadow.get("failure_stage") == expected_stage
            eval_stage = "" if step_passed else str(shadow.get("failure_stage") or shadow.get("stage"))
            eval_errors = [] if step_passed else [f"expected failure_stage {expected_stage}, got {shadow.get('failure_stage') or shadow.get('stage')}"]
        elif shadow.get("stage") != "success":
            step_passed = False
            eval_stage = str(shadow.get("failure_stage") or shadow.get("stage"))
            eval_errors = [str(shadow.get("failure_reason"))]
        else:
            step_passed, eval_stage, eval_errors = evaluate_expected(before, final_graph, expected)

        if step_passed and isinstance(final_graph, dict):
            current_graph = final_graph
        if not step_passed and passed:
            failure_stage = eval_stage
            failure_reason = eval_errors
        passed = passed and step_passed
        step_results.append(
            {
                "step": step_index,
                "instruction": instruction,
                "passed": step_passed,
                "failure_stage": eval_stage,
                "failure_reason": eval_errors,
                "shadow": shadow,
                "graph_diff": graph_diff(before, final_graph) if isinstance(final_graph, dict) else None,
            }
        )

    return {
        "id": case.get("id", ""),
        "category": case.get("category", ""),
        "passed": passed,
        "failure_stage": failure_stage,
        "failure_reason": failure_reason,
        "steps": step_results,
    }


def write_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    counts = Counter("PASS" if row["passed"] else row["failure_stage"] or "unknown" for row in rows)
    lines = ["# Semantic Acceptance Summary", ""]
    lines.append(f"- total: {len(rows)}")
    lines.append(f"- passed: {sum(1 for row in rows if row['passed'])}")
    lines.append(f"- failed: {sum(1 for row in rows if not row['passed'])}")
    lines.append("")
    lines.append("## Failure Stages")
    for key, count in sorted(counts.items()):
        lines.append(f"- {key}: {count}")
    lines.append("")
    lines.append("## Cases")
    lines.append("| id | category | result | failure_stage |")
    lines.append("| --- | --- | --- | --- |")
    for row in rows:
        result = "PASS" if row["passed"] else "FAIL"
        lines.append(f"| {row['id']} | {row.get('category', '')} | {result} | {row.get('failure_stage', '')} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--case-id", action="append", default=[])
    args = parser.parse_args()

    cases = load_cases(args.cases)
    if args.case_id:
        wanted = set(args.case_id)
        cases = [case for case in cases if case.get("id") in wanted]
    if args.limit:
        cases = cases[: args.limit]

    preflight_errors = preflight_cases(cases)
    if preflight_errors:
        print(json.dumps({"preflight_errors": preflight_errors}, ensure_ascii=False, indent=2))
        return 2

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = args.output_dir / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    rows = [run_case(case) for case in cases]
    (run_dir / "results.json").write_text(json.dumps({"cases": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(run_dir / "summary.md", rows)
    for row in rows:
        print(f"{'PASS' if row['passed'] else 'FAIL'} {row['id']} {row.get('failure_stage', '')}")
    print(f"wrote {run_dir}")
    return 0 if all(row["passed"] for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
