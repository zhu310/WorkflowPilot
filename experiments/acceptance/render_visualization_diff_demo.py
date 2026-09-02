from __future__ import annotations

import argparse
import html
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_FORMAL_RUN = Path("experiments/runs/workflow-ir-formal-mainline-20260901-154051")
DEFAULT_ABSTRACT_RUN = Path("experiments/runs/stage3-abstract-intent-20260901-153944")
DEFAULT_OUTPUT_DIR = Path("docs/demo/visualization_diff_demo")


@dataclass(frozen=True)
class DemoCase:
    title: str
    source: Path
    note: str


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def node_id(node: dict[str, Any]) -> str:
    return str(node.get("id", ""))


def edge_key(edge: dict[str, Any]) -> tuple[str, str, str]:
    return (str(edge.get("source", "")), str(edge.get("target", "")), str(edge.get("label", "")))


def node_map(graph: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {node_id(node): node for node in graph.get("nodes", []) if isinstance(node, dict) and node.get("id")}


def edge_map(graph: dict[str, Any]) -> dict[tuple[str, str, str], dict[str, Any]]:
    return {edge_key(edge): edge for edge in graph.get("edges", []) if isinstance(edge, dict)}


def trace_from_artifact(result: dict[str, Any]) -> dict[str, Any]:
    if isinstance(result.get("specification"), dict):
        return result["specification"]
    if isinstance(result.get("trace"), dict):
        return result["trace"]
    return {}


def steps_from_artifact(result: dict[str, Any]) -> list[dict[str, Any]]:
    if isinstance(result.get("steps"), list):
        return result["steps"]
    actual_outcome = result.get("actual_outcome")
    if isinstance(actual_outcome, dict) and isinstance(actual_outcome.get("steps"), list):
        return actual_outcome["steps"]
    return []


def graph_from_transaction(result: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any] | None]:
    trace = trace_from_artifact(result)
    before = trace.get("current_graph") if isinstance(trace.get("current_graph"), dict) else None
    if before is None and isinstance(result.get("graph"), dict):
        before = result["graph"]
    if before is None:
        units = result.get("graph_context", {}).get("units", []) if isinstance(result.get("graph_context"), dict) else []
        before = units[0] if units and isinstance(units[0], dict) else {"nodes": [], "edges": []}
    after = result.get("final_graph") if isinstance(result.get("final_graph"), dict) else None
    return before, after


def diff_graph(before: dict[str, Any], after: dict[str, Any] | None) -> dict[str, Any]:
    after_graph = after or before
    before_nodes = node_map(before)
    after_nodes = node_map(after_graph)
    before_edges = edge_map(before)
    after_edges = edge_map(after_graph)

    added_nodes = sorted(set(after_nodes) - set(before_nodes))
    removed_nodes = sorted(set(before_nodes) - set(after_nodes))
    changed_nodes: list[dict[str, Any]] = []
    for nid in sorted(set(before_nodes) & set(after_nodes)):
        changes = {
            key: {"before": before_nodes[nid].get(key), "after": after_nodes[nid].get(key)}
            for key in sorted(set(before_nodes[nid]) | set(after_nodes[nid]))
            if before_nodes[nid].get(key) != after_nodes[nid].get(key)
        }
        if changes:
            changed_nodes.append({"id": nid, "changes": changes})

    added_edges = sorted(set(after_edges) - set(before_edges))
    removed_edges = sorted(set(before_edges) - set(after_edges))
    preserved_nodes = sorted(set(before_nodes) & set(after_nodes) - {row["id"] for row in changed_nodes})
    preserved_edges = sorted(set(before_edges) & set(after_edges))
    return {
        "added_nodes": added_nodes,
        "removed_nodes": removed_nodes,
        "changed_nodes": changed_nodes,
        "added_edges": [list(item) for item in added_edges],
        "removed_edges": [list(item) for item in removed_edges],
        "preserved_nodes": preserved_nodes,
        "preserved_edges": [list(item) for item in preserved_edges],
    }


def changed_sets(diff: dict[str, Any]) -> tuple[set[str], set[tuple[str, str, str]]]:
    nodes = set(diff["added_nodes"]) | set(diff["removed_nodes"]) | {row["id"] for row in diff["changed_nodes"]}
    edges = {tuple(row) for row in diff["added_edges"]} | {tuple(row) for row in diff["removed_edges"]}
    return nodes, edges


def bounds(graph: dict[str, Any]) -> tuple[float, float, float, float]:
    nodes = graph.get("nodes", [])
    xs = [float(node.get("x", index * 180)) for index, node in enumerate(nodes) if isinstance(node, dict)]
    ys = [float(node.get("y", 0)) for node in nodes if isinstance(node, dict)]
    if not xs or not ys:
        return 0, 0, 600, 240
    return min(xs), min(ys), max(xs), max(ys)


def svg_graph(graph: dict[str, Any], diff: dict[str, Any], title: str) -> str:
    changed_nodes, changed_edges = changed_sets(diff)
    min_x, min_y, max_x, max_y = bounds(graph)
    pad = 70
    width = max(760, int(max_x - min_x + pad * 2 + 180))
    height = max(260, int(max_y - min_y + pad * 2 + 90))

    def px(value: Any, fallback: float) -> float:
        return float(value) - min_x + pad if isinstance(value, (int, float)) else fallback

    def py(value: Any, fallback: float) -> float:
        return float(value) - min_y + pad if isinstance(value, (int, float)) else fallback

    nodes = node_map(graph)
    parts = [
        f'<svg class="graph" viewBox="0 0 {width} {height}" role="img" aria-label="{html.escape(title)}">',
        "<defs><marker id='arrow' markerWidth='10' markerHeight='8' refX='9' refY='4' orient='auto'><path d='M0,0 L10,4 L0,8 z' fill='#39404d'/></marker></defs>",
        f"<text x='18' y='28' class='svg-title'>{html.escape(title)}</text>",
    ]
    for edge in graph.get("edges", []):
        if not isinstance(edge, dict):
            continue
        source = nodes.get(str(edge.get("source")))
        target = nodes.get(str(edge.get("target")))
        if not source or not target:
            continue
        sx = px(source.get("x"), 0) + 72
        sy = py(source.get("y"), 0) + 24
        tx = px(target.get("x"), 0)
        ty = py(target.get("y"), 0) + 24
        key = edge_key(edge)
        klass = "edge changed" if key in changed_edges else "edge"
        label = str(edge.get("label", ""))
        mid_x = (sx + tx) / 2
        mid_y = (sy + ty) / 2 - 8
        parts.append(f"<line class='{klass}' x1='{sx:.1f}' y1='{sy:.1f}' x2='{tx:.1f}' y2='{ty:.1f}' marker-end='url(#arrow)'/>")
        if label:
            parts.append(f"<text class='edge-label' x='{mid_x:.1f}' y='{mid_y:.1f}'>{html.escape(label)}</text>")
    for index, node in enumerate(graph.get("nodes", [])):
        if not isinstance(node, dict):
            continue
        nid = node_id(node)
        x = px(node.get("x"), 60 + index * 120)
        y = py(node.get("y"), 90)
        klass = "node changed" if nid in changed_nodes else "node"
        label = str(node.get("label", node.get("title", nid)))
        ntype = str(node.get("type", ""))
        parts.append(f"<g class='{klass}'>")
        parts.append(f"<rect x='{x:.1f}' y='{y:.1f}' rx='12' ry='12' width='144' height='54'/>")
        parts.append(f"<text class='node-label' x='{x + 72:.1f}' y='{y + 24:.1f}'>{html.escape(label)}</text>")
        parts.append(f"<text class='node-meta' x='{x + 72:.1f}' y='{y + 41:.1f}'>{html.escape(nid)} · {html.escape(ntype)}</text>")
        parts.append("</g>")
    parts.append("</svg>")
    return "\n".join(parts)


def trace_summary(result: dict[str, Any]) -> dict[str, Any]:
    trace = trace_from_artifact(result)
    unit_runs = trace.get("unit_runs", []) if isinstance(trace.get("unit_runs"), list) else []
    grounded = trace.get("grounded_requirements", []) if isinstance(trace.get("grounded_requirements"), list) else []
    return {
        "status": result.get("status"),
        "planning_path": result.get("planning_path"),
        "instruction": trace.get("instruction") or result.get("instruction"),
        "requirements": [row.get("text") for row in grounded if isinstance(row, dict)],
        "unit_count": len(trace.get("spec_units", [])) if isinstance(trace.get("spec_units"), list) else 0,
        "llm_units": sum(1 for row in unit_runs if isinstance(row, dict) and row.get("llm_generation_required")),
        "stop_layer": trace.get("stop_layer"),
        "stop_reason": trace.get("stop_reason"),
        "clarification": trace.get("clarification"),
    }


def json_block(value: Any) -> str:
    return f"<pre>{html.escape(json.dumps(value, ensure_ascii=False, indent=2))}</pre>"


def render_case(case: DemoCase) -> tuple[str, dict[str, Any]]:
    result = load_json(case.source)
    before, after = graph_from_transaction(result)
    diff = diff_graph(before, after)
    summary = trace_summary(result)
    artifact_summary = {
        "title": case.title,
        "source": str(case.source),
        "status": result.get("status"),
        "changed_nodes": len(diff["added_nodes"]) + len(diff["removed_nodes"]) + len(diff["changed_nodes"]),
        "changed_edges": len(diff["added_edges"]) + len(diff["removed_edges"]),
        "preserved_nodes": len(diff["preserved_nodes"]),
        "preserved_edges": len(diff["preserved_edges"]),
        "has_before_graph": bool(before.get("nodes") or before.get("edges")),
        "has_after_graph": after is not None,
        "has_trace": bool(trace_from_artifact(result)),
    }
    after_graph = after or before
    html_parts = [
        "<section class='case-card'>",
        f"<h2>{html.escape(case.title)}</h2>",
        f"<p class='note'>{html.escape(case.note)}</p>",
        f"<p><strong>Instruction:</strong> {html.escape(str(summary.get('instruction') or result.get('error') or ''))}</p>",
        "<div class='graph-pair'>",
        svg_graph(before, diff, "Before Graph"),
        svg_graph(after_graph, diff, "After Graph" if after is not None else "Graph Preserved"),
        "</div>",
        "<div class='columns'>",
        "<div><h3>Changed Nodes / Edges</h3>" + json_block({key: diff[key] for key in ("added_nodes", "removed_nodes", "changed_nodes", "added_edges", "removed_edges")}) + "</div>",
        "<div><h3>Preserved Objects</h3>" + json_block({"nodes": diff["preserved_nodes"], "edges": diff["preserved_edges"]}) + "</div>",
        "</div>",
        "<div class='columns'>",
        "<div><h3>Transaction Steps</h3>" + json_block(steps_from_artifact(result)) + "</div>",
        "<div><h3>Trace Summary</h3>" + json_block(summary) + "</div>",
        "</div>",
        f"<p class='source'>Artifact: {html.escape(str(case.source))}</p>",
        "</section>",
    ]
    return "\n".join(html_parts), artifact_summary


def default_cases(formal_run: Path, abstract_run: Path) -> list[DemoCase]:
    return [
        DemoCase("Single Node Rename", formal_run / "rename/attempt_01/transaction_result.json", "Explicit node property edit."),
        DemoCase("Insert Node With Edges", formal_run / "insert/attempt_01/transaction_result.json", "New node creation plus local topology rewrite."),
        DemoCase("Redirect Edge", formal_run / "redirect/attempt_01/transaction_result.json", "Explicit relation target change."),
        DemoCase("Delete Node", formal_run / "delete/attempt_01/transaction_result.json", "Entity absent target state."),
        DemoCase("Move After", formal_run / "move_after/attempt_01/transaction_result.json", "Topology rewrite with predecessor reconnect."),
        DemoCase("Serial To Parallel", formal_run / "simple_parallel/attempt_01/transaction_result.json", "Parallel structure and join property."),
        DemoCase("Parallel With Retry/Exception", formal_run / "parallel_retry_exception/attempt_01/transaction_result.json", "Main control-flow rewrite while preserving side relations."),
        DemoCase("Mixed Edit", formal_run / "mixed_edit/attempt_01/transaction_result.json", "Combined structural/property/relation target state."),
        DemoCase("Labeled Move After", formal_run / "move_after_labeled/attempt_01/transaction_result.json", "Labeled control-flow provenance."),
        DemoCase("Failure / Success / Retry Combo", formal_run / "failure_success_retry_combo/attempt_01/transaction_result.json", "Separated labeled relation edits."),
        DemoCase("Abstract + Explicit", abstract_run / "simplify_without_removing_checks/attempt_01.json", "Validation requirement plus explicit rename."),
        DemoCase("Abstract Clarification", abstract_run / "maintainability_clarification/attempt_01.json", "No graph mutation; clarification is the correct outcome."),
    ]


def render_html(cases_html: list[str], manifest: dict[str, Any]) -> str:
    css = """
body { margin: 0; font-family: Georgia, 'Times New Roman', serif; color: #20242b; background: #f6f1e8; }
header { padding: 34px 44px; background: linear-gradient(135deg, #263238, #465a64); color: #fff8e6; }
h1 { margin: 0 0 8px; font-size: 34px; }
h2 { margin-top: 0; color: #263238; }
h3 { margin-bottom: 8px; color: #42515a; }
.summary { padding: 18px 44px; background: #fff8e6; border-bottom: 1px solid #decfb8; }
.case-card { margin: 26px auto; max-width: 1180px; padding: 24px; background: #fffdf8; border: 1px solid #dfd1bd; border-radius: 18px; box-shadow: 0 16px 36px rgba(74, 60, 40, 0.12); }
.note, .source { color: #6f6354; }
.graph-pair { display: grid; grid-template-columns: repeat(auto-fit, minmax(420px, 1fr)); gap: 18px; }
.columns { display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 18px; margin-top: 18px; }
pre { overflow: auto; max-height: 360px; padding: 14px; border-radius: 12px; background: #1f272d; color: #f6f1e8; font-size: 12px; line-height: 1.45; }
.graph { width: 100%; min-height: 260px; border-radius: 14px; background: #fbf7ef; border: 1px solid #e4d7c4; }
.svg-title { font: 700 18px Georgia, serif; fill: #263238; }
.edge { stroke: #39404d; stroke-width: 2.2; opacity: .75; }
.edge.changed { stroke: #c45a24; stroke-width: 3.4; opacity: 1; }
.edge-label { font: 12px Georgia, serif; fill: #8a4b1d; paint-order: stroke; stroke: #fbf7ef; stroke-width: 4px; }
.node rect { fill: #f5ead7; stroke: #53616a; stroke-width: 1.5; }
.node.changed rect { fill: #ffd7b8; stroke: #c45a24; stroke-width: 3; }
.node-label { text-anchor: middle; font: 700 13px Georgia, serif; fill: #20242b; }
.node-meta { text-anchor: middle; font: 10px Georgia, serif; fill: #6b7780; }
"""
    return "\n".join(
        [
            "<!doctype html><html><head><meta charset='utf-8'/>",
            "<title>Workflow IR Visualization Diff Demo</title>",
            f"<style>{css}</style></head><body>",
            "<header><h1>Workflow IR Visualization / Diff Demo</h1><p>Generated from frozen transaction artifacts. Product source code is not executed or modified by this report.</p></header>",
            "<div class='summary'><h2>Artifact Coverage</h2>" + json_block(manifest["summary"]) + "</div>",
            *cases_html,
            "</body></html>",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--formal-run", type=Path, default=DEFAULT_FORMAL_RUN)
    parser.add_argument("--abstract-run", type=Path, default=DEFAULT_ABSTRACT_RUN)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    cases_html: list[str] = []
    case_summaries: list[dict[str, Any]] = []
    missing: list[str] = []
    for case in default_cases(args.formal_run, args.abstract_run):
        if not case.source.exists():
            missing.append(str(case.source))
            continue
        rendered, summary = render_case(case)
        cases_html.append(rendered)
        case_summaries.append(summary)

    manifest = {
        "formal_run": str(args.formal_run),
        "abstract_run": str(args.abstract_run),
        "output_dir": str(args.output_dir),
        "summary": {
            "cases_rendered": len(case_summaries),
            "missing_artifacts": missing,
            "cases_with_before_graph": sum(1 for row in case_summaries if row["has_before_graph"]),
            "cases_with_after_graph": sum(1 for row in case_summaries if row["has_after_graph"]),
            "cases_with_trace": sum(1 for row in case_summaries if row["has_trace"]),
        },
        "cases": case_summaries,
    }
    html_path = args.output_dir / "index.html"
    manifest_path = args.output_dir / "manifest.json"
    html_path.write_text(render_html(cases_html, manifest), encoding="utf-8")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"html": str(html_path), "manifest": str(manifest_path), **manifest["summary"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
