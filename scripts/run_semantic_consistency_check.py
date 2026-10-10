from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT / "runs" / "semantic_consistency"
DEFAULT_ACCEPTANCE_RUN = ROOT / "runs" / "semantic_acceptance" / "20260917_165814" / "results.json"
DEFAULT_FOCUSED_RUN = ROOT / "runs" / "semantic_acceptance" / "20260917_165939" / "results.json"

import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.workflow_ir_runtime import _complete_json


CHECKER_SYSTEM = """Return one JSON object that checks whether a candidate Semantic Spec faithfully and completely expresses the user's current natural-language edit request.

You are not generating Workflow IR, graph patches, planner constraints, or fixes.
You are only comparing:
- raw instruction
- current graph and request context
- candidate Semantic Spec

Return exactly:
{
  "verdict": "consistent" | "inconsistent" | "uncertain",
  "reason": "short explanation"
}

Use these meanings:
- consistent: the candidate Semantic Spec appears to faithfully cover all explicit user requirements without adding important unsupported changes.
- inconsistent: the candidate clearly omits, reverses, weakens, or contradicts an explicit user requirement, or adds an important unsupported change.
- uncertain: the candidate may be right but the instruction/graph/spec comparison is not clear enough to judge confidently.

Judge semantic faithfulness, not implementation details.
Do not require hidden compiler details such as old-edge deletion, fan-in/fan-out, or planner diff.
Do not invent a corrected spec.
Do not output any schema beyond the JSON object above.
"""


def load_results(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get("cases", []) if isinstance(data, dict) else []


def find_case(cases: list[dict[str, Any]], case_id: str) -> dict[str, Any]:
    for row in cases:
        if row.get("id") == case_id:
            return row
    raise KeyError(case_id)


def find_step(case: dict[str, Any], step: int = 1) -> dict[str, Any]:
    for row in case.get("steps", []):
        if row.get("step") == step:
            return row
    raise KeyError(f"{case.get('id')} step {step}")


def sample_from_step(name: str, case_id: str, step: dict[str, Any], category: str, expected_acceptance: str) -> dict[str, Any]:
    shadow = step.get("shadow") if isinstance(step.get("shadow"), dict) else {}
    semantic_spec = shadow.get("semantic_spec") if isinstance(shadow.get("semantic_spec"), dict) else {}
    return {
        "name": name,
        "case_id": case_id,
        "step": step.get("step", 1),
        "category": category,
        "expected_acceptance": expected_acceptance,
        "instruction": step.get("instruction") or shadow.get("instruction") or "",
        "before_graph": shadow.get("before_graph"),
        "selection": shadow.get("selection"),
        "request_context": shadow.get("request_context"),
        "candidate_semantic_spec": semantic_spec.get("normalized") or semantic_spec.get("parsed"),
        "acceptance_passed": step.get("passed"),
        "acceptance_failure_stage": step.get("failure_stage", ""),
    }


def build_samples(full_cases: list[dict[str, Any]], focused_cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []

    stable = find_case(full_cases, "seq_zh_complex_order_four_steps")
    samples.append(
        sample_from_step(
            "stable_failure_complex_step3",
            "seq_zh_complex_order_four_steps",
            find_step(stable, 3),
            "stable_failure",
            "FAIL",
        )
    )

    correct_specs = [
        ("existing_join", "zh_existing_parallel_insert_compliance_after_join", 1),
        ("new_entity_property", "zh_new_human_review_explicit_create", 1),
        ("before_after_correct", "zh_relocate_inventory_before_payment", 1),
        ("rename_property", "property_rename_by_label", 1),
        ("branch_label", "branch_label_edit", 1),
        ("condition_expression", "condition_expression_only", 1),
        ("preserve", "zh_preserve_exception_branch_reorder", 1),
        ("parallel", "linear_parallel_standard", 1),
        ("multi_goal", "zh_multigoal_reorder_and_failure_label", 1),
        ("multi_step_sequence", "seq_en_reorder_then_branch_label", 2),
    ]
    for name, case_id, step_no in correct_specs:
        case = find_case(full_cases, case_id)
        samples.append(sample_from_step(name, case_id, find_step(case, step_no), "correct_pass", "PASS"))

    fluctuation_partial = find_case(full_cases, "seq_zh_parallel_insert_rename")
    samples.append(
        sample_from_step(
            "fluctuation_partial_step2",
            "seq_zh_parallel_insert_rename",
            find_step(fluctuation_partial, 2),
            "fluctuation_partial",
            "FAIL",
        )
    )
    fluctuation_resolved = find_case(focused_cases, "seq_zh_parallel_insert_rename")
    samples.append(
        sample_from_step(
            "fluctuation_resolved_step2",
            "seq_zh_parallel_insert_rename",
            find_step(fluctuation_resolved, 2),
            "fluctuation_resolved",
            "PASS",
        )
    )
    return samples


def run_check(sample: dict[str, Any], repeat_index: int) -> dict[str, Any]:
    payload = {
        "instruction": sample["instruction"],
        "graph": sample["before_graph"],
        "selection": sample["selection"],
        "request_context": sample["request_context"],
        "candidate_semantic_spec": sample["candidate_semantic_spec"],
    }
    checker_run = _complete_json(CHECKER_SYSTEM, payload)
    parsed = checker_run.parsed_json if isinstance(checker_run.parsed_json, dict) else {}
    verdict = parsed.get("verdict") if parsed.get("verdict") in {"consistent", "inconsistent", "uncertain"} else "uncertain"
    reason = parsed.get("reason") if isinstance(parsed.get("reason"), str) else (checker_run.parse_error or "checker did not return valid verdict")
    return {
        "sample": sample["name"],
        "case_id": sample["case_id"],
        "step": sample["step"],
        "category": sample["category"],
        "expected_acceptance": sample["expected_acceptance"],
        "acceptance_passed": sample["acceptance_passed"],
        "acceptance_failure_stage": sample["acceptance_failure_stage"],
        "repeat": repeat_index,
        "verdict": verdict,
        "reason": reason,
        "raw_response": checker_run.raw_response,
        "parse_error": checker_run.parse_error,
    }


def write_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    by_sample: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_sample[row["sample"]].append(row)
    lines = ["# Semantic Consistency Check Summary", ""]
    lines.append(f"- total checks: {len(rows)}")
    counts = Counter(row["verdict"] for row in rows)
    for verdict in ("consistent", "inconsistent", "uncertain"):
        lines.append(f"- {verdict}: {counts.get(verdict, 0)}")
    lines.append("")
    lines.append("| sample | category | case | step | verdicts |")
    lines.append("| --- | --- | --- | --- | --- |")
    for sample, sample_rows in sorted(by_sample.items()):
        verdicts = Counter(row["verdict"] for row in sample_rows)
        verdict_text = ", ".join(f"{key}:{verdicts[key]}" for key in sorted(verdicts))
        first = sample_rows[0]
        lines.append(f"| {sample} | {first['category']} | {first['case_id']} | {first['step']} | {verdict_text} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--acceptance-run", type=Path, default=DEFAULT_ACCEPTANCE_RUN)
    parser.add_argument("--focused-run", type=Path, default=DEFAULT_FOCUSED_RUN)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--repeat", type=int, default=5)
    args = parser.parse_args()

    full_cases = load_results(args.acceptance_run)
    focused_cases = load_results(args.focused_run)
    samples = build_samples(full_cases, focused_cases)
    rows: list[dict[str, Any]] = []
    for sample in samples:
        repeat_count = args.repeat if sample["category"] in {"stable_failure", "fluctuation_partial", "fluctuation_resolved"} else 1
        for index in range(1, repeat_count + 1):
            row = run_check(sample, index)
            rows.append(row)
            print(f"{row['verdict'].upper()} {row['sample']} repeat={index}: {row['reason']}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = args.output_dir / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "results.json").write_text(json.dumps({"checks": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(run_dir / "summary.md", rows)
    print(f"wrote {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
