from __future__ import annotations

from src.constraint_planner import expand_specification, extract_graph_context, plan_constraints
from src import mvp23
from src.mvp22 import execute_deterministic


def graph(nodes: list[tuple[str, str]], edges: list[tuple[str, str, str]]) -> dict:
    return {
        "nodes": [{"id": node_id, "type": node_type, "label": node_id, "x": index * 100, "y": 0} for index, (node_id, node_type) in enumerate(nodes)],
        "edges": [{"id": f"e{index}", "source": source, "target": target, "label": label} for index, (source, target, label) in enumerate(edges, 1)],
    }


def pairs(value: dict) -> set[tuple[str, str, str]]:
    return {(edge["source"], edge["target"], edge.get("label", "")) for edge in value["edges"]}


def execute(value: dict, planned: dict) -> dict:
    result = execute_deterministic(value, {"core_node_ids": [], "core_edge_ids": []}, {"incoming": [], "outgoing": []}, planned["steps"], planned["mutation_authorization"])
    assert result.error is None
    assert result.graph is not None
    return result.graph


def test_properties_and_relations_share_one_minimal_diff() -> None:
    value = graph([("start", "start"), ("review", "task"), ("end", "end")], [("start", "review", ""), ("review", "end", "")])
    planned = plan_constraints(value, {"entities": [], "properties": [{"node_id": "review", "changes": {"label": "Reviewed", "x": 250}}], "required_relations": [], "forbidden_relations": []})
    output = execute(value, planned)
    assert next(node for node in output["nodes"] if node["id"] == "review")["label"] == "Reviewed"
    assert next(node for node in output["nodes"] if node["id"] == "review")["x"] == 250
    assert pairs(output) == pairs(value)


def test_insert_and_redirect_use_generic_entity_and_relation_diff() -> None:
    value = graph([("A", "start"), ("B", "task"), ("end", "end")], [("A", "B", ""), ("B", "end", "")])
    spec = {"constraints": {"entities": [], "properties": [], "required_relations": [], "forbidden_relations": []}, "directives": [{"kind": "insert_on_selected_edge", "new_entity": {"ref": "review", "type": "human_review", "label": "Review"}}, {"kind": "redirect_selected_edge", "target_node_id": "end"}]}
    constraints, error = expand_specification(value, {"node_ids": [], "edge_ids": ["e1"]}, spec)
    assert error is None and constraints is not None
    planned = plan_constraints(value, constraints)
    assert planned["supported"]
    output = execute(value, planned)
    assert pairs(output) == {("A", "new_1", ""), ("new_1", "B", ""), ("A", "end", ""), ("B", "end", "")}


def test_parallel_directive_lowers_to_generic_constraints_without_duplicate_edges() -> None:
    value = graph([("start", "start"), ("identity", "task"), ("risk", "task"), ("approval", "human_review"), ("end", "end")], [("start", "identity", ""), ("identity", "risk", ""), ("risk", "approval", ""), ("approval", "end", "")])
    spec = {"constraints": {"entities": [], "properties": [], "required_relations": [], "forbidden_relations": []}, "directives": [{"kind": "parallelize_selected_chain", "branches": ["identity", "risk"], "upstream": "start", "join": "approval", "completion": "all"}]}
    constraints, error = expand_specification(value, {"node_ids": ["identity", "risk"], "edge_ids": []}, spec)
    assert error is None and constraints is not None
    output = execute(value, plan_constraints(value, constraints))
    assert pairs(output) == {("start", "identity", ""), ("start", "risk", ""), ("identity", "approval", ""), ("risk", "approval", ""), ("approval", "end", "")}
    assert next(node for node in output["nodes"] if node["id"] == "approval")["join"] == {"mode": "all"}


def test_move_after_uses_context_facts_and_preserves_external_metadata() -> None:
    value = graph([("start", "start"), ("X", "task"), ("A", "task"), ("B", "task"), ("end", "end")], [("start", "X", ""), ("X", "A", ""), ("A", "B", ""), ("B", "end", "")])
    context = extract_graph_context(value, {"node_ids": ["X", "A"], "moving_node_id": "X", "anchor_node_id": "A"}, {"node_ids": ["X", "A"], "edge_ids": []})
    assert context["facts"]["predecessors"]["X"] == ["start"]
    spec = {"constraints": {"entities": [], "properties": [], "required_relations": [], "forbidden_relations": []}, "directives": [{"kind": "relocate_after", "moving_node_id": "X", "anchor_node_id": "A", "reconnect_old_location": True}]}
    constraints, error = expand_specification(value, {"node_ids": ["X", "A"], "edge_ids": []}, spec)
    assert error is None and constraints is not None
    output = execute(value, plan_constraints(value, constraints))
    assert pairs(output) == {("start", "A", ""), ("A", "X", ""), ("X", "B", ""), ("B", "end", "")}
    assert next(node for node in output["nodes"] if node["id"] == "end") == next(node for node in value["nodes"] if node["id"] == "end")


def test_executor_rejects_incomplete_or_mismatched_authorization() -> None:
    value = graph([("A", "start"), ("B", "end")], [("A", "B", "")])
    result = execute_deterministic(value, {"core_node_ids": [], "core_edge_ids": []}, {"incoming": [], "outgoing": []}, [{"op": "remove_edge", "edge_id": "e1"}], [])
    assert result.graph is None
    assert "authorization" in (result.error or "")


def test_executor_rejects_unauthorized_update_edge() -> None:
    value = graph([("A", "start"), ("B", "end")], [("A", "B", "success")])
    result = execute_deterministic(
        value,
        {"core_node_ids": [], "core_edge_ids": []},
        {"incoming": [], "outgoing": []},
        [{"op": "update_edge", "edge_id": "e1", "changes": {"branch_label": "approved"}}],
        [],
    )
    assert result.graph is None
    assert "authorization" in (result.error or "")


def test_layout_and_edge_properties_lower_to_existing_atomic_operations() -> None:
    value = graph([("A", "start"), ("B", "end")], [("A", "B", "failure")])
    planned = plan_constraints(value, {"entities": [], "properties": [{"node_id": "A", "changes": {"x": 80, "y": 30}}], "edge_properties": [{"edge_id": "e1", "changes": {"label": "exception"}}], "required_relations": [], "forbidden_relations": []})
    assert [step["op"] for step in planned["steps"]] == ["move_node", "update_edge"]
    output = execute(value, planned)
    assert next(node for node in output["nodes"] if node["id"] == "A")["x"] == 80
    assert next(edge for edge in output["edges"] if edge["id"] == "e1")["label"] == "exception"


def test_delete_node_removes_incident_edges_without_reconnect() -> None:
    value = graph([("A", "start"), ("X", "task"), ("B", "end")], [("A", "X", ""), ("X", "B", "")])
    planned = plan_constraints(value, {"entities": [], "absent_entities": [{"node_id": "X"}], "properties": [], "required_relations": [], "forbidden_relations": []})
    assert [step["op"] for step in planned["steps"]] == ["remove_edge", "remove_edge", "remove_node"]
    output = execute(value, planned)
    assert {node["id"] for node in output["nodes"]} == {"A", "B"}
    assert output["edges"] == []


def test_delete_node_with_explicit_reconnect_adds_required_relation() -> None:
    value = graph([("A", "start"), ("X", "task"), ("B", "end")], [("A", "X", ""), ("X", "B", "")])
    planned = plan_constraints(
        value,
        {
            "entities": [],
            "absent_entities": [{"node_id": "X"}],
            "properties": [],
            "required_relations": [{"source": "A", "target": "B", "label": ""}],
            "forbidden_relations": [{"source": "A", "target": "X", "label": ""}, {"source": "X", "target": "B", "label": ""}],
        },
    )
    output = execute(value, planned)
    assert {node["id"] for node in output["nodes"]} == {"A", "B"}
    assert pairs(output) == {("A", "B", "")}


def test_parallel_directive_can_coexist_with_explicit_retry_constraints() -> None:
    value = graph(
        [("start", "start"), ("material", "task"), ("permission", "task"), ("manual", "human_review"), ("approval", "task"), ("end", "end")],
        [
            ("start", "material", ""),
            ("material", "permission", ""),
            ("permission", "approval", ""),
            ("material", "manual", "异常"),
            ("permission", "manual", "异常"),
            ("manual", "material", ""),
            ("approval", "end", ""),
        ],
    )
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
    constraints, error = expand_specification(value, {"node_ids": ["material", "permission"], "edge_ids": []}, spec)
    assert error is None and constraints is not None
    output = execute(value, plan_constraints(value, constraints))
    assert pairs(output) == {
        ("start", "material", ""),
        ("start", "permission", ""),
        ("material", "approval", ""),
        ("permission", "approval", ""),
        ("material", "manual", "异常"),
        ("permission", "manual", "异常"),
        ("manual", "start", ""),
        ("approval", "end", ""),
    }


def test_parallel_directive_requires_explicit_upstream() -> None:
    value = graph([("start", "start"), ("material", "task"), ("permission", "task"), ("approval", "task"), ("end", "end")], [("start", "material", ""), ("material", "permission", ""), ("permission", "approval", ""), ("approval", "end", "")])
    spec = {"constraints": {"entities": [], "properties": [], "required_relations": [], "forbidden_relations": []}, "directives": [{"kind": "parallelize_selected_chain", "branches": ["material", "permission"], "join": "approval", "completion": "all"}]}
    constraints, error = expand_specification(value, {"node_ids": ["material", "permission"], "edge_ids": []}, spec)
    assert constraints is None
    assert error == "parallelize_selected_chain is underspecified"


def test_transaction_uses_specification_planner_not_atomic_steps(monkeypatch) -> None:
    value = graph([("start", "start"), ("review", "task"), ("end", "end")], [("start", "review", ""), ("review", "end", "")])
    interpretation = {"status": "clear", "target": {"node_ids": ["review"], "edge_ids": []}, "intent": {"kind": "rename"}, "policy": {}}
    specification = {"status": "ready", "constraints": {"entities": [], "properties": [{"node_id": "review", "changes": {"label": "Approved"}}], "required_relations": [], "forbidden_relations": []}, "directives": []}
    monkeypatch.setattr(mvp23, "run_edit_interpretation", lambda *args, **kwargs: mvp23.JsonRun("", interpretation, None, False))
    monkeypatch.setattr(mvp23, "finalize_edit_specification", lambda *args, **kwargs: mvp23.JsonRun("", specification, None, False))
    result = mvp23.run_transaction(value, 1, {"node_ids": ["review"], "edge_ids": []}, "rename review")
    assert result.status == "success"
    assert result.planning_path == "constraint"
    assert result.boundary is None
    assert result.mutation_authorization and result.mutation_authorization[0]["constraint_rule"] == "property_equals"
    assert next(node for node in result.final_graph["nodes"] if node["id"] == "review")["label"] == "Approved"


def test_transaction_supports_delete_node_via_absent_entities(monkeypatch) -> None:
    value = graph([("A", "start"), ("X", "task"), ("B", "end")], [("A", "X", ""), ("X", "B", "")])
    interpretation = {"status": "clear", "target": {"node_ids": ["X"], "edge_ids": []}, "intent": {"kind": "delete"}, "policy": {"reconnect": "none"}}
    specification = {
        "status": "ready",
        "constraints": {"entities": [], "absent_entities": [{"node_id": "X"}], "properties": [], "required_relations": [], "forbidden_relations": []},
        "directives": [],
    }
    monkeypatch.setattr(mvp23, "run_edit_interpretation", lambda *args, **kwargs: mvp23.JsonRun("", interpretation, None, False))
    monkeypatch.setattr(mvp23, "finalize_edit_specification", lambda *args, **kwargs: mvp23.JsonRun("", specification, None, False))
    result = mvp23.run_transaction(value, 1, {"node_ids": ["X"], "edge_ids": []}, "delete X and its connections")
    assert result.status == "success"
    assert {node["id"] for node in result.final_graph["nodes"]} == {"A", "B"}
    assert result.final_graph["edges"] == []
