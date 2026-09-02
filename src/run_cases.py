from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from .evaluator import evaluate
from .runner import run_single_llm, save_json


FIXTURES = Path("fixtures")


def load_case(case_id: str) -> dict[str, Any]:
    path = FIXTURES / f"{case_id}.json"
    if not path.exists():
        raise FileNotFoundError(f"Fixture not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MVP 0 Single-LLM baseline cases.")
    parser.add_argument("--cases", nargs="+", required=True)
    args = parser.parse_args()
    run_dir = Path("runs") / datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=False)
    reports: list[dict[str, Any]] = []

    for case_id in args.cases:
        case = load_case(case_id)
        case_dir = run_dir / case_id
        case_dir.mkdir()
        save_json(case_dir / "input.json", {k: case[k] for k in ("case_id", "instruction", "graph", "context", "editable_scope") if k in case})
        try:
            result = run_single_llm(case["instruction"], case["graph"], case.get("context"), case.get("editable_scope"))
            (case_dir / "raw_response.txt").write_text(result.raw_response, encoding="utf-8")
            if result.parsed_graph is not None:
                save_json(case_dir / "parsed_graph.json", result.parsed_graph)
            evaluation = evaluate(case["graph"], result.parsed_graph, case["acceptance"], result.parse_error)
        except Exception as error:
            evaluation = {"passed": False, "failure_category": "request_error", "checks": {}, "details": [f"{type(error).__name__}: {error}"]}
        report = {
            "case_id": case_id,
            "instruction": case["instruction"],
            "automatic_evaluation": evaluation,
            "human_review_points": case["human_review_points"],
            "human_note": "",
        }
        save_json(case_dir / "report.json", report)
        reports.append(report)
        print(f"{case_id}: {'PASS' if evaluation['passed'] else 'FAIL'} ({evaluation['failure_category'] or 'ok'})")

    save_json(run_dir / "summary.json", {"cases": reports})
    lines = ["# MVP 0 Baseline Summary", ""]
    for report in reports:
        result = report["automatic_evaluation"]
        lines.append(f"- {report['case_id']}: {'PASS' if result['passed'] else 'FAIL'}; {result['failure_category'] or 'ok'}")
        if result["details"]:
            lines.append(f"  - {'; '.join(result['details'])}")
    (run_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved: {run_dir}")


if __name__ == "__main__":
    main()
