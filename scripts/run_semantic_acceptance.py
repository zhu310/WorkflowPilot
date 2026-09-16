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

    if errors:
        preservation_errors = [error for error in errors if error.startswith("preserved")]
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
