from __future__ import annotations

from types import SimpleNamespace

from src import semantic_shadow
from src.goal_lowering import lower_grounded_goal_to_workflow_ir
from src.semantic_goal import ground_semantic_goal_spec, validate_grounded_goal, validate_semantic_goal_spec


def branch_graph() -> dict:
    return {
        "nodes": [
            {"id": "decision", "type": "decision", "label": "判断", "x": 0, "y": 0},
            {"id": "auto", "type": "task", "label": "自动处理", "x": 180, "y": -80},
            {"id": "manual", "type": "task", "label": "人工处理", "x": 180, "y": 80},
        ],
        "edges": [
            {"id": "e_success", "source": "decision", "target": "auto", "label": "success"},
            {"id": "e_failure", "source": "decision", "target": "manual", "label": "failure"},
        ],
    }


def branch_goal() -> dict:
    return {
        "status": "resolved",
        "semantic_goals": [
            {
                "kind": "branch_property",
                "text": "给失败分支改名为人工审核",
                "object": {"type": "edge_reference", "text": "失败分支"},
                "property": {"name": "label", "value": "人工审核"},
                "explicit_constraints": [],
            }
        ],
        "clarification": {"needed": False, "question": "", "reason": ""},
    }


def branch_graph_cn() -> dict:
    return {
        "nodes": [
            {"id": "decision", "type": "decision", "label": "Decision", "x": 0, "y": 0},
            {"id": "auto", "type": "task", "label": "Auto", "x": 180, "y": -80},
            {"id": "manual", "type": "task", "label": "Manual", "x": 180, "y": 80},
        ],
        "edges": [
            {"id": "e_success", "source": "decision", "target": "auto", "label": "\u6210\u529f\u5206\u652f"},
            {"id": "e_failure", "source": "decision", "target": "manual", "label": "\u5931\u8d25\u5206\u652f"},
        ],
    }


def branch_goal_cn(reference: str = "\u5931\u8d25\u5206\u652f", value: str = "\u4eba\u5de5\u5ba1\u6838") -> dict:
    return {
        "status": "resolved",
        "semantic_goals": [
            {
                "kind": "branch_property",
                "text": f"\u7ed9{reference}\u6539\u540d\u4e3a{value}",
                "object": {"type": "edge_reference", "text": reference},
                "property": {"name": "label", "value": value},
                "explicit_constraints": [],
            }
        ],
        "clarification": {"needed": False, "question": "", "reason": ""},
    }


def edge_by_id(graph: dict, edge_id: str) -> dict:
    return next(edge for edge in graph.get("edges", []) if edge.get("id") == edge_id)


def condition_graph() -> dict:
    return {
        "nodes": [
            {"id": "decision", "type": "decision", "label": "Decision", "x": 0, "y": 0},
            {"id": "auto", "type": "task", "label": "Auto", "x": 180, "y": -80},
            {"id": "manual", "type": "task", "label": "Manual", "x": 180, "y": 80},
        ],
        "edges": [
            {
                "id": "e_success",
                "source": "decision",
                "target": "auto",
                "label": "success",
                "branch_label": "Approved",
                "condition": {"expression": "score >= 60", "language": "cel"},
            },
            {
                "id": "e_failure",
                "source": "decision",
                "target": "manual",
                "label": "failure",
                "branch_label": "Rejected",
                "condition": {"expression": "score < 60", "language": "cel"},
            },
        ],
    }


def condition_goal(edge_text: str = "success branch", value: str = "score >= 90", property_name: str = "condition.expression") -> dict:
    return {
        "status": "resolved",
        "semantic_goals": [
            {
                "id": "g1",
                "goal_category": "object_state_goal",
                "text": f"Set {edge_text} condition to {value}",
                "objects": [{"id": "edge", "type": "edge_reference", "text": edge_text, "role": "object"}],
                "object_desired_states": [{"object": "edge", "state": "property_equals", "property": property_name, "value": value}],
                "relation_intents": [],
                "semantic_constraints": [],
            }
        ],
        "clarification": {"needed": False, "question": "", "reason": ""},
    }


def node_condition_goal() -> dict:
    return {
        "status": "resolved",
        "semantic_goals": [
            {
                "id": "g1",
                "goal_category": "object_state_goal",
                "text": "Set Decision condition to score >= 90",
                "objects": [{"id": "node", "type": "existing_reference", "text": "Decision", "role": "object"}],
                "object_desired_states": [{"object": "node", "state": "property_equals", "property": "condition.expression", "value": "score >= 90"}],
                "relation_intents": [],
                "semantic_constraints": [],
            }
        ],
        "clarification": {"needed": False, "question": "", "reason": ""},
    }


def test_branch_property_lowers_to_edge_properties() -> None:
    grounded = ground_semantic_goal_spec(branch_goal(), branch_graph(), {"node_ids": [], "edge_ids": []})

    lowered = lower_grounded_goal_to_workflow_ir(grounded, branch_graph())

    assert grounded["grounding"]["issues"] == []
    assert grounded["semantic_goals"][0]["bindings"][0]["binding_type"] == "edge"
    assert grounded["semantic_goals"][0]["bindings"][0]["id"] == "e_failure"
    assert lowered["ok"] is True
    assert lowered["semantic_spec"]["workflow_ir"] == {
        "control_flow_regions": [],
        "auxiliary_relations": [],
        "absent_entities": [],
        "properties": [],
        "edge_properties": [{"edge_id": "e_failure", "changes": {"branch_label": "人工审核"}}],
    }


def test_condition_expression_lowers_to_typed_edge_property_with_metadata_merge() -> None:
    graph = condition_graph()
    grounded = ground_semantic_goal_spec(condition_goal(), graph, {"node_ids": [], "edge_ids": []})

    validation = validate_grounded_goal(grounded)
    lowered = lower_grounded_goal_to_workflow_ir(grounded, graph)

    assert grounded["grounding"]["issues"] == []
    assert grounded["semantic_goals"][0]["bindings"][0]["id"] == "e_success"
    assert validation["lowering_ready"] is True
    assert lowered["ok"] is True
    assert lowered["semantic_spec"]["workflow_ir"]["edge_properties"] == [
        {"edge_id": "e_success", "changes": {"condition": {"expression": "score >= 90", "language": "cel"}}}
    ]


def test_condition_property_compatibility_lowers_to_condition_expression() -> None:
    graph = condition_graph()
    grounded = ground_semantic_goal_spec(condition_goal(property_name="condition"), graph, {"node_ids": [], "edge_ids": []})

    lowered = lower_grounded_goal_to_workflow_ir(grounded, graph)

    assert validate_grounded_goal(grounded)["lowering_ready"] is True
    assert lowered["semantic_spec"]["workflow_ir"]["edge_properties"] == [
        {"edge_id": "e_success", "changes": {"condition": {"expression": "score >= 90", "language": "cel"}}}
    ]


def test_failure_condition_update_targets_exact_failure_edge_only() -> None:
    graph = condition_graph()
    grounded = ground_semantic_goal_spec(condition_goal("failure branch", "score < 30"), graph, {"node_ids": [], "edge_ids": []})

    lowered = lower_grounded_goal_to_workflow_ir(grounded, graph)

    assert grounded["grounding"]["issues"] == []
    assert grounded["semantic_goals"][0]["bindings"][0]["id"] == "e_failure"
    assert lowered["semantic_spec"]["workflow_ir"]["edge_properties"] == [
        {"edge_id": "e_failure", "changes": {"condition": {"expression": "score < 30", "language": "cel"}}}
    ]


def test_node_subject_cannot_use_condition_expression() -> None:
    grounded = ground_semantic_goal_spec(node_condition_goal(), condition_graph(), {"node_ids": [], "edge_ids": []})

    validation = validate_grounded_goal(grounded)

    assert grounded["grounding"]["issues"] == []
    assert validation["lowering_ready"] is False
    assert "invalid_property_equals" in {issue["kind"] for issue in validation["issues"]}


def test_condition_expression_shadow_preserves_edge_identity_and_other_fields(monkeypatch) -> None:
    before = condition_graph()
    monkeypatch.setattr(
        semantic_shadow,
        "_complete_json",
        lambda _system, _payload: SimpleNamespace(raw_response="", parsed_json=condition_goal(), parse_error=None, normalized_from_fence=False),
    )

    trace = semantic_shadow.run_semantic_shadow(before, "Only change the success branch condition to score >= 90.")

    assert trace["stage"] == "success"
    success = edge_by_id(trace["final_shadow_graph"], "e_success")
    failure = edge_by_id(trace["final_shadow_graph"], "e_failure")
    assert success["condition"] == {"expression": "score >= 90", "language": "cel"}
    assert success["branch_label"] == "Approved"
    assert success["source"] == "decision"
    assert success["target"] == "auto"
    assert success["label"] == "success"
    assert failure == before["edges"][1]


def test_branch_label_edit_preserves_existing_condition_object(monkeypatch) -> None:
    goal = {
        "status": "resolved",
        "semantic_goals": [
            {
                "id": "g1",
                "goal_category": "object_state_goal",
                "text": "Set success branch label",
                "objects": [{"id": "edge", "type": "edge_reference", "text": "success branch", "role": "object"}],
                "object_desired_states": [{"object": "edge", "state": "property_equals", "property": "branch_label", "value": "Auto OK"}],
                "relation_intents": [],
                "semantic_constraints": [],
            }
        ],
        "clarification": {"needed": False, "question": "", "reason": ""},
    }
    before = condition_graph()
    monkeypatch.setattr(
        semantic_shadow,
        "_complete_json",
        lambda _system, _payload: SimpleNamespace(raw_response="", parsed_json=goal, parse_error=None, normalized_from_fence=False),
    )

    trace = semantic_shadow.run_semantic_shadow(before, "Only change the success branch label.")

    assert trace["stage"] == "success"
    success = edge_by_id(trace["final_shadow_graph"], "e_success")
    assert success["branch_label"] == "Auto OK"
    assert success["condition"] == before["edges"][0]["condition"]
    assert success["label"] == "success"


def test_semantic_shadow_branch_label_only_runs_existing_backend(monkeypatch) -> None:
    monkeypatch.setattr(
        semantic_shadow,
        "_complete_json",
        lambda _system, _payload: SimpleNamespace(raw_response="", parsed_json=branch_goal(), parse_error=None, normalized_from_fence=False),
    )

    trace = semantic_shadow.run_semantic_shadow(branch_graph(), "给失败分支改名为人工审核")

    assert trace["stage"] == "success"
    assert trace["grounded_goal"]["grounding"]["issues"] == []
    assert trace["compiler_input"]["workflow_ir"]["edge_properties"] == [
        {"edge_id": "e_failure", "changes": {"branch_label": "人工审核"}}
    ]
    final_failure = edge_by_id(trace["final_shadow_graph"], "e_failure")
    final_success = edge_by_id(trace["final_shadow_graph"], "e_success")
    assert final_failure["branch_label"] == "人工审核"
    assert final_failure["label"] == "failure"
    assert "branch_label" not in final_success


def test_branch_property_ambiguous_edge_reference_stops_before_lowering(monkeypatch) -> None:
    graph = branch_graph()
    graph["edges"].append({"id": "e_failure_2", "source": "auto", "target": "manual", "label": "failure"})
    monkeypatch.setattr(
        semantic_shadow,
        "_complete_json",
        lambda _system, _payload: SimpleNamespace(raw_response="", parsed_json=branch_goal(), parse_error=None, normalized_from_fence=False),
    )

    trace = semantic_shadow.run_semantic_shadow(graph, "修改失败分支名称")

    assert trace["failure_stage"] == "validation"
    assert trace["validation"]["issues"][0]["kind"] == "ambiguous_edge_reference"
    assert trace["validation"]["issues"][0]["candidates"] == ["e_failure", "e_failure_2"]
    assert trace["lowering_result"] is None


def test_branch_property_contract_rejects_edge_id_and_edge_properties() -> None:
    result = validate_semantic_goal_spec(
        {
            "status": "resolved",
            "semantic_goals": [
                {
                    "kind": "branch_property",
                    "edge_id": "e_failure",
                    "edge_properties": [{"edge_id": "e_failure", "changes": {"branch_label": "x"}}],
                    "object": {"type": "edge_reference", "text": "失败分支"},
                    "property": {"name": "label", "value": "人工审核"},
                }
            ],
        }
    )

    assert result["valid"] is False
    assert "semantic_goal_contract_violation" in {issue["kind"] for issue in result["issues"]}


def test_branch_label_only_preserves_user_edge_phrase_and_runs_existing_backend(monkeypatch) -> None:
    monkeypatch.setattr(
        semantic_shadow,
        "_complete_json",
        lambda _system, _payload: SimpleNamespace(raw_response="", parsed_json=branch_goal_cn(), parse_error=None, normalized_from_fence=False),
    )

    trace = semantic_shadow.run_semantic_shadow(branch_graph_cn(), "\u7ed9\u5931\u8d25\u5206\u652f\u6539\u540d\u4e3a\u4eba\u5de5\u5ba1\u6838")

    goal = trace["semantic_goal"]["normalized"]["semantic_goals"][0]
    assert goal["kind"] == "branch_property"
    assert goal["objects"][0]["reference"] == "\u5931\u8d25\u5206\u652f"
    assert goal["property"] == {"name": "label", "value": "\u4eba\u5de5\u5ba1\u6838"}
    assert trace["stage"] == "success"
    assert trace["grounded_goal"]["semantic_goals"][0]["bindings"][0]["id"] == "e_failure"
    assert trace["compiler_input"]["workflow_ir"]["edge_properties"] == [
        {"edge_id": "e_failure", "changes": {"branch_label": "\u4eba\u5de5\u5ba1\u6838"}}
    ]
    assert edge_by_id(trace["final_shadow_graph"], "e_failure")["branch_label"] == "\u4eba\u5de5\u5ba1\u6838"


def test_branch_this_branch_ambiguous_without_selection_stops_before_lowering(monkeypatch) -> None:
    monkeypatch.setattr(
        semantic_shadow,
        "_complete_json",
        lambda _system, _payload: SimpleNamespace(raw_response="", parsed_json=branch_goal_cn("\u8fd9\u4e2a\u5206\u652f"), parse_error=None, normalized_from_fence=False),
    )

    trace = semantic_shadow.run_semantic_shadow(branch_graph_cn(), "\u7ed9\u8fd9\u4e2a\u5206\u652f\u6539\u540d\u4e3a\u4eba\u5de5\u5ba1\u6838")

    goal = trace["semantic_goal"]["normalized"]["semantic_goals"][0]
    assert goal["objects"][0]["reference"] == "\u8fd9\u4e2a\u5206\u652f"
    assert trace["failure_stage"] == "validation"
    assert trace["stage"] == "goal_ground"
    assert trace["lowering_result"] is None
    assert trace["compiler_input"] is None
    assert trace["validation"]["issues"][0]["kind"] == "ambiguous_edge_reference"
    assert trace["validation"]["issues"][0]["candidates"] == ["e_failure", "e_success"]
    result = semantic_shadow.semantic_trace_to_transaction_result(trace)
    assert result.status == "needs_clarification"


def test_branch_this_branch_uses_unique_selected_edge(monkeypatch) -> None:
    monkeypatch.setattr(
        semantic_shadow,
        "_complete_json",
        lambda _system, _payload: SimpleNamespace(raw_response="", parsed_json=branch_goal_cn("\u8fd9\u4e2a\u5206\u652f"), parse_error=None, normalized_from_fence=False),
    )

    trace = semantic_shadow.run_semantic_shadow(
        branch_graph_cn(),
        "\u7ed9\u8fd9\u4e2a\u5206\u652f\u6539\u540d\u4e3a\u4eba\u5de5\u5ba1\u6838",
        {"node_ids": [], "edge_ids": ["e_failure"]},
    )

    assert trace["stage"] == "success"
    provenance = trace["grounded_goal"]["grounding"]["provenance"]
    assert any(row["reference"] == "\u8fd9\u4e2a\u5206\u652f" and row["resolved_id"] == "e_failure" for row in provenance)
    assert edge_by_id(trace["final_shadow_graph"], "e_failure")["branch_label"] == "\u4eba\u5de5\u5ba1\u6838"


def test_semantic_understanding_cannot_expand_this_branch_to_failure(monkeypatch) -> None:
    expanded = branch_goal_cn("failure")
    monkeypatch.setattr(
        semantic_shadow,
        "_complete_json",
        lambda _system, _payload: SimpleNamespace(raw_response="", parsed_json=expanded, parse_error=None, normalized_from_fence=False),
    )

    trace = semantic_shadow.run_semantic_shadow(branch_graph_cn(), "\u7ed9\u8fd9\u4e2a\u5206\u652f\u6539\u540d\u4e3a\u4eba\u5de5\u5ba1\u6838")

    assert trace["failure_stage"] == "validation"
    assert trace["validation"]["issues"][0]["kind"] == "semantic_goal_contract_violation"
    assert trace["grounded_goal"] is None
