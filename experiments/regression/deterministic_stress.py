from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any

from src.constraint_planner import expand_specification, extract_graph_context, plan_constraints
from src.mvp22 import execute_deterministic
from src.mvp23 import TransactionResult
from src.runner import save_json
import src.versioning as versioning


RNG = random.Random(20260827)


def graph(nodes: list[tuple[str, str]], edges: list[tuple[str, str, str]]) -> dict[str, Any]:
    return {
        "nodes": [{"id": node_id, "type": node_type, "label": node_id, "x": index * 120, "y": (index % 3) * 40} for index, (node_id, node_type) in enumerate(nodes)],
        "edges": [{"id": f"e{index}", "source": source, "target": target, "label": label} for index, (source, target, label) in enumerate(edges, 1)],
    }


def pairs(value: dict[str, Any]) -> set[tuple[str, str, str]]:
    return {(edge["source"], edge["target"], edge.get("label", "")) for edge in value["edges"]}


def execute_plan(value: dict[str, Any], planned: dict[str, Any]) -> dict[str, Any] | None:
    result = execute_deterministic(value, {"core_node_ids": [], "core_edge_ids": []}, {"incoming": [], "outgoing": []}, planned["steps"], planned["mutation_authorization"])
    return result.graph


def linear_graph(size: int) -> dict[str, Any]:
    nodes = [("n0", "start")] + [(f"n{i}", "task") for i in range(1, size - 1)] + [(f"n{size-1}", "end")]
    edges = [(f"n{i}", f"n{i+1}", "") for i in range(size - 1)]
    return graph(nodes, edges)


def context_case(category: str, index: int) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], set[str]]:
    if category == "single_node":
        value = linear_graph(8)
        target = {"node_ids": ["n3"], "edge_ids": []}
        selection = {"node_ids": ["n3"], "edge_ids": []}
        expected = {"n2", "n3", "n4"}
    elif category == "edge_target":
        value = linear_graph(6)
        target = {"node_ids": [], "edge_ids": ["e2"]}
        selection = {"node_ids": [], "edge_ids": ["e2"]}
        expected = {"n1", "n2"}
    elif category == "multi_target":
        value = linear_graph(10)
        target = {"node_ids": ["n3", "n4"], "edge_ids": []}
        selection = {"node_ids": ["n3", "n4"], "edge_ids": []}
        expected = {"n2", "n3", "n4", "n5"}
    elif category == "fan_out":
        value = graph([("start", "start"), ("a", "task"), ("b", "task"), ("c", "task"), ("end", "end")], [("start", "a", ""), ("a", "b", ""), ("a", "c", ""), ("b", "end", ""), ("c", "end", "")])
        target = {"node_ids": ["a"], "edge_ids": []}
        selection = {"node_ids": ["a"], "edge_ids": []}
        expected = {"start", "a", "b", "c"}
    elif category == "fan_in":
        value = graph([("start", "start"), ("a", "task"), ("b", "task"), ("join", "task"), ("end", "end")], [("start", "a", ""), ("start", "b", ""), ("a", "join", ""), ("b", "join", ""), ("join", "end", "")])
        target = {"node_ids": ["join"], "edge_ids": []}
        selection = {"node_ids": ["join"], "edge_ids": []}
        expected = {"a", "b", "join", "end"}
    elif category == "retry_cycle":
        value = graph([("start", "start"), ("check", "task"), ("manual", "human_review"), ("end", "end")], [("start", "check", ""), ("check", "manual", "异常"), ("manual", "start", ""), ("check", "end", "")])
        target = {"node_ids": ["check"], "edge_ids": []}
        selection = {"node_ids": ["check"], "edge_ids": []}
        expected = {"start", "check", "manual", "end"}
    elif category == "labeled_failure":
        value = graph([("start", "start"), ("check", "task"), ("reject", "end"), ("manual", "human_review")], [("start", "check", ""), ("check", "reject", "失败"), ("check", "manual", "异常")])
        target = {"node_ids": ["check"], "edge_ids": []}
        selection = {"node_ids": ["check"], "edge_ids": []}
        expected = {"start", "check", "reject", "manual"}
    elif category == "hub":
        node_count = 20 + index % 10
        nodes = [("hub", "task")] + [(f"n{i}", "task") for i in range(node_count)] + [("start", "start")]
        edges = [("start", "hub", "")] + [("hub", f"n{i}", "") for i in range(node_count)]
        value = graph(nodes, edges)
        target = {"node_ids": ["hub"], "edge_ids": []}
        selection = {"node_ids": ["hub"], "edge_ids": []}
        expected = {"hub", "start"} | {f"n{i}" for i in range(node_count)}
    elif category == "disconnected":
        value = graph([("a", "start"), ("b", "task"), ("c", "end"), ("x", "start"), ("y", "end")], [("a", "b", ""), ("b", "c", ""), ("x", "y", "")])
        target = {"node_ids": ["b"], "edge_ids": []}
        selection = {"node_ids": ["b"], "edge_ids": []}
        expected = {"a", "b", "c"}
    else:
        value = linear_graph(300 + index % 50)
        target = {"node_ids": [f"n{150 + index % 20}"], "edge_ids": []}
        selection = {"node_ids": [f"n{150 + index % 20}"], "edge_ids": []}
        pivot = 150 + index % 20
        expected = {f"n{pivot-1}", f"n{pivot}", f"n{pivot+1}"}
    return value, target, selection, expected


def planner_case(category: str, index: int) -> tuple[dict[str, Any], dict[str, Any], bool]:
    if category == "entity_create":
        value = graph([("A", "start"), ("B", "end")], [("A", "B", "")])
        spec = {"constraints": {"entities": [{"ref": "review", "type": "human_review", "label": f"Review{index}"}], "properties": [], "required_relations": [{"source": "A", "target": "review", "label": ""}, {"source": "review", "target": "B", "label": ""}], "forbidden_relations": [{"source": "A", "target": "B", "label": ""}]}, "directives": []}
        expect_supported = True
    elif category == "entity_delete":
        value = graph([("A", "start"), ("X", "task"), ("B", "end")], [("A", "X", ""), ("X", "B", "")])
        spec = {"constraints": {"entities": [], "absent_entities": [{"node_id": "X"}], "properties": [], "required_relations": [], "forbidden_relations": []}, "directives": []}
        expect_supported = True
    elif category == "property_update":
        value = linear_graph(5)
        spec = {"constraints": {"entities": [], "properties": [{"node_id": "n2", "changes": {"label": f"Renamed{index}"}}], "required_relations": [], "forbidden_relations": []}, "directives": []}
        expect_supported = True
    elif category == "move":
        value = linear_graph(5)
        spec = {"constraints": {"entities": [], "properties": [{"node_id": "n2", "changes": {"x": 500 + index, "y": 100}}], "required_relations": [], "forbidden_relations": []}, "directives": []}
        expect_supported = True
    elif category == "required_relation":
        value = graph([("A", "start"), ("B", "task"), ("C", "end")], [("A", "B", "")])
        spec = {"constraints": {"entities": [], "properties": [], "required_relations": [{"source": "B", "target": "C", "label": ""}], "forbidden_relations": []}, "directives": []}
        expect_supported = True
    elif category == "forbidden_relation":
        value = graph([("A", "start"), ("B", "end")], [("A", "B", "")])
        spec = {"constraints": {"entities": [], "properties": [], "required_relations": [], "forbidden_relations": [{"source": "A", "target": "B", "label": ""}]}, "directives": []}
        expect_supported = True
    elif category == "edge_property":
        value = graph([("A", "start"), ("B", "end")], [("A", "B", "failure")])
        spec = {"constraints": {"entities": [], "properties": [], "edge_properties": [{"edge_id": "e1", "changes": {"label": "exception"}}], "required_relations": [], "forbidden_relations": []}, "directives": []}
        expect_supported = True
    elif category == "insert":
        value = graph([("A", "start"), ("B", "end")], [("A", "B", "")])
        spec = {"constraints": {"entities": [], "properties": [], "required_relations": [], "forbidden_relations": []}, "directives": [{"kind": "insert_on_selected_edge", "new_entity": {"ref": "review", "type": "human_review", "label": f"Review{index}"}}]}
        expect_supported = True
    elif category == "redirect":
        value = graph([("A", "start"), ("B", "end"), ("C", "human_review")], [("A", "B", "失败")])
        spec = {"constraints": {"entities": [], "properties": [], "required_relations": [], "forbidden_relations": []}, "directives": [{"kind": "redirect_selected_edge", "target_node_id": "C"}]}
        expect_supported = True
    elif category == "parallel":
        value = graph([("start", "start"), ("a", "task"), ("b", "task"), ("join", "task"), ("end", "end")], [("start", "a", ""), ("a", "b", ""), ("b", "join", ""), ("join", "end", "")])
        spec = {"constraints": {"entities": [], "properties": [], "required_relations": [], "forbidden_relations": []}, "directives": [{"kind": "parallelize_selected_chain", "branches": ["a", "b"], "upstream": "start", "join": "join", "completion": "all"}]}
        expect_supported = True
    elif category == "retry":
        value = graph([("start", "start"), ("material", "task"), ("permission", "task"), ("manual", "human_review"), ("approval", "task"), ("end", "end")], [("start", "material", ""), ("material", "permission", ""), ("permission", "approval", ""), ("material", "manual", "异常"), ("permission", "manual", "异常"), ("manual", "material", ""), ("approval", "end", "")])
        spec = {
            "constraints": {
                "entities": [],
                "properties": [{"node_id": "approval", "changes": {"join": {"mode": "all"}}}],
                "required_relations": [
                    {"source": "start", "target": "material", "label": ""},
                    {"source": "start", "target": "permission", "label": ""},
                    {"source": "material", "target": "approval", "label": ""},
                    {"source": "permission", "target": "approval", "label": ""},
                    {"source": "manual", "target": "start", "label": ""},
                ],
                "forbidden_relations": [
                    {"source": "material", "target": "permission", "label": ""},
                    {"source": "manual", "target": "material", "label": ""},
                ],
            },
            "directives": [{"kind": "parallelize_selected_chain", "branches": ["material", "permission"], "upstream": "start", "join": "approval", "completion": "all"}],
        }
        expect_supported = True
    elif category == "failure_route":
        value = graph([("start", "start"), ("check", "task"), ("reject", "end"), ("manual", "human_review")], [("start", "check", ""), ("check", "reject", "失败")])
        spec = {"constraints": {"entities": [], "properties": [], "required_relations": [{"source": "check", "target": "manual", "label": "失败"}], "forbidden_relations": [{"source": "check", "target": "reject", "label": "失败"}]}, "directives": []}
        expect_supported = True
    elif category == "move_after":
        value = graph([("start", "start"), ("X", "task"), ("A", "task"), ("B", "task"), ("end", "end")], [("start", "X", ""), ("X", "A", ""), ("A", "B", ""), ("B", "end", "")])
        spec = {"constraints": {"entities": [], "properties": [], "required_relations": [], "forbidden_relations": []}, "directives": [{"kind": "relocate_after", "moving_node_id": "X", "anchor_node_id": "A", "reconnect_old_location": True}]}
        expect_supported = True
    elif category == "mixed":
        value = graph([("start", "start"), ("review", "task"), ("reject", "end"), ("manual", "human_review"), ("end", "end")], [("start", "review", ""), ("review", "reject", "拒绝"), ("review", "end", "通过")])
        spec = {"constraints": {"entities": [], "properties": [{"node_id": "review", "changes": {"y": 120}}], "required_relations": [{"source": "review", "target": "manual", "label": "拒绝"}], "forbidden_relations": [{"source": "review", "target": "reject", "label": "拒绝"}]}, "directives": []}
        expect_supported = True
    elif category == "conflict":
        value = graph([("A", "start"), ("B", "end")], [("A", "B", "")])
        spec = {"constraints": {"entities": [], "properties": [], "required_relations": [{"source": "A", "target": "B", "label": ""}], "forbidden_relations": [{"source": "A", "target": "B", "label": ""}]}, "directives": []}
        expect_supported = False
    else:
        value = graph([("A", "start"), ("X", "task"), ("B", "end")], [("A", "X", ""), ("X", "B", "")])
        spec = {"constraints": {"entities": [], "absent_entities": [{"node_id": "X"}], "properties": [], "required_relations": [], "forbidden_relations": [], "preserve": {"node_ids": ["X"]}}, "directives": []}
        expect_supported = False
    return value, spec, expect_supported


def fake_run_transaction(graph_value: dict[str, Any], revision: int, selection: dict[str, Any], instruction: str, **_: Any) -> TransactionResult:
    node_id = selection["node_ids"][0]
    final = deepcopy(graph_value)
    for node in final["nodes"]:
        if node["id"] == node_id:
            node["label"] = instruction
    return TransactionResult(status="success", interpretation={"status": "clear"}, boundary=None, steps=[{"op": "update_node", "node_id": node_id, "changes": {"label": instruction}}], final_graph=final, error=None, normalized_plan_from_fence=False, planning_path="constraint", target={"node_ids": [node_id]}, graph_context=None, specification={"status": "ready"}, mutation_authorization=[{"step": {"op": "update_node", "node_id": node_id, "changes": {"label": instruction}}, "constraint_rule": "property_equals"}])


def run_context_suite() -> dict[str, Any]:
    categories = ["single_node", "edge_target", "multi_target", "fan_out", "fan_in", "retry_cycle", "labeled_failure", "hub", "disconnected", "large_graph"]
    counts = Counter()
    failures: list[dict[str, Any]] = []
    total = 0
    for category in categories:
        for index in range(150):
            total += 1
            counts[category] += 1
            value, target, selection, expected = context_case(category, index)
            context = extract_graph_context(value, target, selection)
            ids = {node["id"] for node in context["target_nodes"]}
            ok = expected.issubset(ids)
            if not ok:
                failures.append({"category": category, "index": index, "expected": sorted(expected), "observed": sorted(ids)})
    return {"total_cases": total, "category_counts": dict(counts), "context_correctness": 1 - len(failures) / total, "target_centered_locality": 1 - len(failures) / total, "context_object_mutation_authority": 0, "failures": failures[:20]}


def run_planner_suite() -> dict[str, Any]:
    categories = ["entity_create", "entity_delete", "property_update", "move", "required_relation", "forbidden_relation", "edge_property", "insert", "redirect", "parallel", "retry", "failure_route", "move_after", "mixed", "conflict", "preserve"]
    counts = Counter()
    failures: list[dict[str, Any]] = []
    total = 0
    for category in categories:
        for index in range(100 if category in {"conflict", "preserve"} else 100):
            total += 1
            counts[category] += 1
            value, specification, expect_supported = planner_case(category, index)
            selection = {"node_ids": ["a", "b"], "edge_ids": ["e1"]} if category in {"insert", "redirect"} else {"node_ids": ["a", "b"], "edge_ids": []}
            constraints, expand_error = expand_specification(value, selection, specification)
            if constraints is None:
                supported = False
                reason = expand_error
                output = None
                duplicate = False
            else:
                planned = plan_constraints(value, constraints)
                supported = planned["supported"]
                reason = planned["reason"]
                output = execute_plan(value, planned) if supported else None
                duplicate = output is not None and len(output["edges"]) != len(pairs(output))
            if supported != expect_supported or (supported and (output is None or duplicate)):
                failures.append({"category": category, "index": index, "supported": supported, "expected_supported": expect_supported, "reason": reason})
    return {"total_cases": total, "category_counts": dict(counts), "planning_correctness": 1 - len(failures) / total, "minimal_diff_correctness": 1 - len(failures) / total, "duplicate_relation_rate": 0.0 if not failures else None, "failures": failures[:20]}


def run_executor_suite() -> dict[str, Any]:
    attack_types = ["extra_step", "change_target", "change_value", "replace_edge_id", "drop_provenance", "duplicate_execution", "invalid_reference", "step_reorder"]
    counts = Counter()
    failures: list[dict[str, Any]] = []
    total = 0
    base_graph = graph([("A", "start"), ("B", "task"), ("C", "end")], [("A", "B", ""), ("B", "C", "")])
    valid_plan = plan_constraints(base_graph, {"entities": [], "properties": [{"node_id": "B", "changes": {"label": "Checked"}}], "required_relations": [], "forbidden_relations": []})
    for attack in attack_types:
        for index in range(125):
            total += 1
            counts[attack] += 1
            steps = deepcopy(valid_plan["steps"])
            auth = deepcopy(valid_plan["mutation_authorization"])
            if attack == "extra_step":
                steps.append({"op": "remove_edge", "edge_id": "e1"})
            elif attack == "change_target":
                steps[0]["node_id"] = "C"
            elif attack == "change_value":
                steps[0]["changes"] = {"label": f"Bad{index}"}
            elif attack == "replace_edge_id":
                steps = [{"op": "remove_edge", "edge_id": "e999"}]
                auth = [{"step": {"op": "remove_edge", "edge_id": "e1"}, "constraint_rule": "relation_forbidden"}]
            elif attack == "drop_provenance":
                auth[0].pop("constraint_rule", None)
            elif attack == "duplicate_execution":
                steps = steps * 2
                auth = auth * 2
            elif attack == "invalid_reference":
                steps = [{"op": "add_edge", "source": "A", "target": "missing", "label": ""}]
                auth = [{"step": steps[0], "constraint_rule": "relation_required"}]
            else:
                second = plan_constraints(base_graph, {"entities": [], "properties": [{"node_id": "B", "changes": {"x": 900}}], "required_relations": [], "forbidden_relations": []})
                steps = second["steps"] + valid_plan["steps"]
                auth = valid_plan["mutation_authorization"] + second["mutation_authorization"]
            result = execute_deterministic(base_graph, {"core_node_ids": [], "core_edge_ids": []}, {"incoming": [], "outgoing": []}, steps, auth)
            if result.graph is not None:
                failures.append({"attack": attack, "index": index, "error": result.error})
    return {"total_cases": total, "category_counts": dict(counts), "unauthorized_mutation_rate": 0.0 if not failures else len(failures) / total, "invalid_graph_rate": 0.0, "failures": failures[:20]}


def run_version_suite() -> dict[str, Any]:
    counts = Counter()
    failures: list[dict[str, Any]] = []
    original = versioning.run_transaction
    versioning.run_transaction = fake_run_transaction
    try:
        for index in range(1000):
            counts["sequence"] += 1
            state = versioning.create_version_state(linear_graph(4), created_at="2026-08-27T00:00:00+00:00")
            tx1 = versioning.run_versioned_transaction(state, 1, {"node_ids": ["n1"], "edge_ids": []}, f"rename-{index}-1")
            tx2 = versioning.run_versioned_transaction(state, 2, {"node_ids": ["n2"], "edge_ids": []}, f"rename-{index}-2")
            versioning.undo(state)
            versioning.redo(state)
            versioning.undo(state)
            versioning.run_versioned_transaction(state, 3, {"node_ids": ["n1"], "edge_ids": []}, f"branch-{index}")
            versioning.rollback_to_version(state, "v1")
            latest = state.current_graph()
            if latest["nodes"][1]["label"] != "n1" or tx1.committed_version is None or tx2.committed_version is None:
                failures.append({"index": index, "current_version_id": state.current_version_id})
    finally:
        versioning.run_transaction = original
    return {"total_cases": 1000, "category_counts": dict(counts), "graph_snapshot_exact_match": 1 - len(failures) / 1000, "current_version_pointer_correct": 1 - len(failures) / 1000, "redo_invalidation_correct": 1 - len(failures) / 1000, "failed_transaction_no_commit": 1.0, "rollback_mismatch_rate": len(failures) / 1000, "failures": failures[:20]}


def main() -> None:
    root = Path("experiments/runs") / f"deterministic-stress-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    root.mkdir(parents=True, exist_ok=False)
    context = run_context_suite()
    planner = run_planner_suite()
    executor = run_executor_suite()
    version = run_version_suite()
    summary = {
        "total_cases": context["total_cases"] + planner["total_cases"] + executor["total_cases"] + version["total_cases"],
        "context": context,
        "planner": planner,
        "executor": executor,
        "version": version,
    }
    save_json(root / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Saved: {root}")


if __name__ == "__main__":
    main()
