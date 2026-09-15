from __future__ import annotations

from copy import deepcopy

from src import mvp23
from src.mvp22 import execute_deterministic
from src.transformations import PARALLELIZE_SELECTED_CHAIN_TO_JOIN, compile_parallelize_selected_chain_to_join
from src.workflow_ir_core import canonicalize_schema_variants, compile_ir, constraints_from_compiled, requirement_needs_clarification
from src.workflow_ir_runtime import run_workflow_ir_transaction


def graph_for(branch_ids: list[str], join_id: str, join_type: str = "task") -> dict:
    nodes = [{"id": "start", "type": "start", "label": "Start", "x": 0, "y": 0}]
    nodes.extend({"id": node_id, "type": "task", "label": node_id, "x": 100 * (index + 1), "y": 0} for index, node_id in enumerate(branch_ids))
    nodes.extend(
        [
            {"id": join_id, "type": join_type, "label": join_id, "x": 100 * (len(branch_ids) + 1), "y": 0},
            {"id": "end", "type": "end", "label": "End", "x": 100 * (len(branch_ids) + 2), "y": 0},
        ]
    )
    pairs = [("start", branch_ids[0])]
    pairs.extend(zip(branch_ids, branch_ids[1:]))
    pairs.extend([(branch_ids[-1], join_id), (join_id, "end")])
    return {"nodes": nodes, "edges": [{"id": f"e{index}", "source": source, "target": target, "label": ""} for index, (source, target) in enumerate(pairs, 1)]}


def compile_and_execute(graph: dict, branch_ids: list[str], join_id: str) -> tuple[dict, dict]:
    scope = {"core_node_ids": [*branch_ids, join_id], "core_edge_ids": [edge["id"] for edge in graph["edges"] if edge["source"] in branch_ids or edge["target"] in branch_ids]}
    boundary = {"incoming": [{"external_node_id": "start", "internal_node_id": branch_ids[0], "existing_edge_id": "e1"}], "outgoing": [{"external_node_id": "end", "internal_node_id": join_id, "existing_edge_id": graph["edges"][-1]["id"]}]}
    operation = {"op": PARALLELIZE_SELECTED_CHAIN_TO_JOIN, "branch_node_ids": branch_ids, "join_node_id": join_id, "join_mode": "all"}
    compiled = compile_parallelize_selected_chain_to_join(graph, operation, scope, boundary)
    result = execute_deterministic(graph, scope, boundary, compiled["steps"])
    assert result.error is None
    assert result.graph is not None
    return compiled, result.graph


def pairs(graph: dict) -> set[tuple[str, str, str]]:
    return {(edge["source"], edge["target"], edge.get("label", "")) for edge in graph["edges"]}


def test_two_node_parallel_uses_minimum_diff() -> None:
    graph = graph_for(["material", "risk"], "approval")
    compiled, output = compile_and_execute(graph, ["material", "risk"], "approval")
    assert len(compiled["steps"]) == 4
    assert pairs(output) == {("start", "material", ""), ("start", "risk", ""), ("material", "approval", ""), ("risk", "approval", ""), ("approval", "end", "")}
    assert next(node for node in output["nodes"] if node["id"] == "approval")["join"] == {"mode": "all"}


def test_chinese_parallel_topology_is_not_treated_as_abstract_only() -> None:
    requirement = {
        "text": "把材料审核和风控审核改成并行结构",
        "references": {"mentioned_node_ids": ["material", "risk"]},
        "grounded_desired_state": {
            "abstract_goals": [{"goal_type": "other", "description": "改成并行结构"}],
            "clarification": {"needed": False, "question": "", "reason": ""},
        },
    }

    assert requirement_needs_clarification(requirement) == (False, None, None)


def test_underspecified_abstract_goal_asks_in_chinese() -> None:
    requirement = {
        "text": "this logic is confusing",
        "references": {"mentioned_node_ids": []},
        "grounded_desired_state": {
            "abstract_goals": [{"goal_type": "other", "description": "logic is confusing"}],
            "clarification": {"needed": False, "question": "", "reason": ""},
        },
    }

    needs_clarification, question, reason = requirement_needs_clarification(requirement)

    assert needs_clarification is True
    assert reason == "underspecified abstract goal"
    assert question is not None and question.startswith("\u8bf7\u8bf4\u660e")


def test_answer_only_boundary_sentence_does_not_execute_as_fresh_edit() -> None:
    graph = graph_for(["material", "risk"], "approval")

    result = run_workflow_ir_transaction(
        graph,
        1,
        {"node_ids": [], "edge_ids": []},
        "\u4ece material \u5f00\u59cb\uff0c\u5230 approval \u7ed3\u675f",
    )

    assert result["status"] == "needs_clarification"
    assert result["error"] == "answer-only boundary reference"
    assert result["final_graph"] is None


def test_standalone_parallel_ir_infers_serial_chain_boundaries() -> None:
    graph = graph_for(["material", "risk"], "approval")
    workflow_ir = {
        "control_flow_regions": [
            {
                "relation_type": "",
                "structure": {
                    "type": "parallel",
                    "join_mode": "all",
                    "branches": [
                        {"type": "node", "id": "material"},
                        {"type": "node", "id": "risk"},
                    ],
                },
            }
        ],
        "auxiliary_relations": [],
        "absent_entities": [],
        "properties": [],
        "edge_properties": [],
    }

    constraints = constraints_from_compiled(graph, compile_ir(workflow_ir, graph))

    assert {(row["source"], row["target"], row.get("label", "")) for row in constraints["required_relations"]} == {
        ("start", "material", ""),
        ("start", "risk", ""),
        ("material", "approval", ""),
        ("risk", "approval", ""),
    }
    assert constraints["forbidden_relations"] == [{"source": "material", "target": "risk", "label": ""}]
    assert constraints["properties"] == [{"node_id": "approval", "changes": {"join": {"mode": "all"}}}]


def test_parallel_branches_accept_single_item_array_variants() -> None:
    workflow_ir = {
        "control_flow_regions": [
            {
                "relation_type": "",
                "structure": {
                    "type": "parallel",
                    "join_mode": "all",
                    "branches": [
                        [{"type": "node", "id": "material"}],
                        [{"type": "node", "id": "risk"}],
                    ],
                },
            }
        ],
        "auxiliary_relations": [],
        "absent_entities": [],
        "properties": [],
        "edge_properties": [],
    }

    canonical, changed = canonicalize_schema_variants(workflow_ir)

    assert changed is True
    assert canonical["control_flow_regions"][0]["structure"]["branches"] == [
        {"type": "node", "id": "material"},
        {"type": "node", "id": "risk"},
    ]


def test_incomplete_standalone_parallel_does_not_lower_to_delete_only() -> None:
    graph = graph_for(["material", "risk"], "approval")
    graph["edges"].append({"id": "e_side", "source": "risk", "target": "manual", "label": ""})
    workflow_ir = {
        "control_flow_regions": [
            {
                "relation_type": "",
                "structure": {
                    "type": "parallel",
                    "join_mode": "all",
                    "branches": [
                        {"type": "node", "id": "material"},
                        {"type": "node", "id": "risk"},
                    ],
                },
            }
        ],
        "auxiliary_relations": [],
        "absent_entities": [],
        "properties": [],
        "edge_properties": [],
    }

    compiled = compile_ir(workflow_ir, graph)
    constraints = constraints_from_compiled(graph, compiled)

    assert compiled["control_flow_regions"][0]["incomplete_parallel_boundary"] is True
    assert constraints["required_relations"] == []
    assert constraints["forbidden_relations"] == []
    assert constraints["properties"] == []


def test_three_node_parallel_preserves_external_graph() -> None:
    graph = graph_for(["identity_check", "material_check", "risk_check"], "approval")
    compiled, output = compile_and_execute(graph, ["identity_check", "material_check", "risk_check"], "approval")
    assert len(compiled["steps"]) == 7
    assert ("start", "identity_check", "") in pairs(output)
    assert ("start", "material_check", "") in pairs(output)
    assert ("start", "risk_check", "") in pairs(output)
    assert ("approval", "end", "") in pairs(output)
    assert next(edge for edge in output["edges"] if edge["id"] == "e5") == next(edge for edge in graph["edges"] if edge["id"] == "e5")


def test_human_review_join_is_generic() -> None:
    graph = graph_for(["document_check", "policy_check"], "human_review", "human_review")
    _, output = compile_and_execute(graph, ["document_check", "policy_check"], "human_review")
    assert next(node for node in output["nodes"] if node["id"] == "human_review")["join"] == {"mode": "all"}
    assert ("document_check", "human_review", "") in pairs(output)
    assert ("policy_check", "human_review", "") in pairs(output)


def test_existing_correct_edges_are_not_recreated_or_duplicated() -> None:
    graph = graph_for(["material", "risk"], "approval")
    compiled, output = compile_and_execute(graph, ["material", "risk"], "approval")
    assert {step["edge_id"] for step in compiled["steps"] if step["op"] == "remove_edge"} == {"e2"}
    assert ("risk", "approval", "") not in {(step.get("source"), step.get("target"), step.get("label", "")) for step in compiled["steps"] if step["op"] == "add_edge"}
    output_pairs = [(edge["source"], edge["target"], edge.get("label", "")) for edge in output["edges"]]
    assert len(output_pairs) == len(set(output_pairs))


def test_ambiguous_topology_is_unsupported() -> None:
    graph = graph_for(["material", "risk"], "approval")
    ambiguous = deepcopy(graph)
    ambiguous["edges"].append({"id": "e_ambiguous", "source": "risk", "target": "material", "label": ""})
    operation = {"op": PARALLELIZE_SELECTED_CHAIN_TO_JOIN, "branch_node_ids": ["material", "risk"], "join_node_id": "approval", "join_mode": "all"}
    result = compile_parallelize_selected_chain_to_join(
        ambiguous,
        operation,
        {"core_node_ids": ["material", "risk", "approval"], "core_edge_ids": ["e2", "e3", "e_ambiguous"]},
        {"incoming": [], "outgoing": []},
    )
    assert result["supported"] is False
    assert result["reason"].startswith("unsupported_transformation:")


def test_unsupported_topology_returns_unsupported_without_atomic_fallback(monkeypatch) -> None:
    graph = graph_for(["material", "risk"], "approval")
    graph["nodes"].append({"id": "manual", "type": "human_review", "label": "manual", "x": 500, "y": 80})
    graph["edges"].append({"id": "e_side", "source": "material", "target": "manual", "label": ""})
    interpretation = {
        "status": "clear",
        "target": {"node_ids": ["material", "risk"], "edge_ids": [], "join_node_id": "approval"},
        "intent": {"kind": "parallelize_selected_chain"},
        "policy": {},
    }
    specification = {"status": "ready", "constraints": {"entities": [], "properties": [], "required_relations": [], "forbidden_relations": []}, "directives": [{"kind": "parallelize_selected_chain", "branches": ["material", "risk"], "upstream": "start", "join": "missing_join", "completion": "all"}]}
    monkeypatch.setattr(mvp23, "run_edit_interpretation", lambda *args, **kwargs: mvp23.JsonRun("", interpretation, None, False))
    monkeypatch.setattr(mvp23, "finalize_edit_specification", lambda *args, **kwargs: mvp23.JsonRun("", specification, None, False))

    result = mvp23.run_transaction(graph, 1, {"node_ids": ["material", "risk"], "edge_ids": []}, "parallelize the checks", allowed_transformations=["set_join"])
    assert result.status == "planning_error"
    assert result.planning_path == "constraint"
    assert result.final_graph is None
