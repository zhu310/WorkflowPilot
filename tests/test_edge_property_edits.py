from __future__ import annotations

from copy import deepcopy

from src import mvp23, workflow_ir_runtime


def graph() -> dict:
    return {
        "nodes": [
            {"id": "start", "type": "start", "label": "Start", "x": 0, "y": 0},
            {"id": "decision", "type": "decision", "label": "Decision", "x": 100, "y": 0},
            {"id": "auto", "type": "task", "label": "Auto", "x": 200, "y": -80},
            {"id": "manual", "type": "task", "label": "Manual", "x": 200, "y": 80},
        ],
        "edges": [
            {"id": "e_start", "source": "start", "target": "decision", "label": ""},
            {
                "id": "e_auto",
                "source": "decision",
                "target": "auto",
                "label": "success",
                "condition": {"expression": "amount <= 1000", "language": "cel"},
                "branch_label": "low_amount",
            },
            {
                "id": "e_manual",
                "source": "decision",
                "target": "manual",
                "label": "failure",
                "condition": {"expression": "amount > 1000", "language": "cel"},
                "branch_label": "high_amount",
            },
        ],
    }


def patch_understand(monkeypatch, requirement: dict) -> None:
    response = {"requirements": [requirement]}
    monkeypatch.setattr(
        workflow_ir_runtime,
        "_complete_json",
        lambda *_args, **_kwargs: workflow_ir_runtime.JsonRun("", response, None, False),
    )


def edges_by_id(value: dict) -> dict[str, dict]:
    return {edge["id"]: edge for edge in value["edges"]}


def test_selected_branch_label_active_edit_preserves_edge_identity_and_condition(monkeypatch) -> None:
    before = graph()
    patch_understand(
        monkeypatch,
        {
            "id": "r1",
            "text": "Rename selected branch label to approved.",
            "references": {},
            "desired_state": {"edge_properties": [{"target": "selected_edge", "changes": {"branch_label": "approved"}}]},
            "clarification": {"needed": False, "question": "", "reason": ""},
        },
    )

    result = mvp23.run_transaction(deepcopy(before), 1, {"node_ids": [], "edge_ids": ["e_auto"]}, "Rename selected branch label to approved.")

    assert result.status == "success"
    after_edges = edges_by_id(result.final_graph)
    assert after_edges["e_auto"]["branch_label"] == "approved"
    assert after_edges["e_auto"]["condition"] == before["edges"][1]["condition"]
    assert after_edges["e_auto"]["source"] == "decision"
    assert after_edges["e_auto"]["target"] == "auto"
    assert after_edges["e_auto"]["label"] == "success"
    assert after_edges["e_manual"] == before["edges"][2]
    assert result.steps == [{"op": "update_edge", "edge_id": "e_auto", "changes": {"branch_label": "approved"}}]
    assert result.mutation_authorization[0]["constraint_rule"] == "edge_property_equals"


def test_selected_condition_expression_replacement_preserves_language_branch_label_and_identity(monkeypatch) -> None:
    before = graph()
    patch_understand(
        monkeypatch,
        {
            "id": "r1",
            "text": "Replace selected branch condition expression with amount <= 5000.",
            "references": {},
            "desired_state": {"edge_properties": [{"target": "selected_edge", "changes": {"condition": {"expression": "amount <= 5000"}}}]},
            "clarification": {"needed": False, "question": "", "reason": ""},
        },
    )

    result = mvp23.run_transaction(deepcopy(before), 1, {"node_ids": [], "edge_ids": ["e_auto"]}, "Replace selected branch condition expression with amount <= 5000.")

    assert result.status == "success"
    after_edges = edges_by_id(result.final_graph)
    assert after_edges["e_auto"]["condition"] == {"expression": "amount <= 5000", "language": "cel"}
    assert after_edges["e_auto"]["branch_label"] == "low_amount"
    assert after_edges["e_auto"]["source"] == "decision"
    assert after_edges["e_auto"]["target"] == "auto"
    assert after_edges["e_auto"]["label"] == "success"
    assert after_edges["e_manual"] == before["edges"][2]


def test_branch_label_does_not_pollute_relation_type_or_edge_label(monkeypatch) -> None:
    before = graph()
    patch_understand(
        monkeypatch,
        {
            "id": "r1",
            "text": "Rename selected branch label to failure.",
            "references": {},
            "desired_state": {"edge_properties": [{"target": "selected_edge", "changes": {"branch_label": "failure"}}]},
            "clarification": {"needed": False, "question": "", "reason": ""},
        },
    )

    result = mvp23.run_transaction(deepcopy(before), 1, {"node_ids": [], "edge_ids": ["e_auto"]}, "Rename selected branch label to failure.")

    edge = edges_by_id(result.final_graph)["e_auto"]
    assert edge["branch_label"] == "failure"
    assert edge["label"] == "success"


def test_ambiguous_branch_property_edit_needs_clarification(monkeypatch) -> None:
    before = graph()
    patch_understand(
        monkeypatch,
        {
            "id": "r1",
            "text": "Rename branch label to approved.",
            "references": {"source_node": "Decision"},
            "desired_state": {"edge_properties": [{"source": "source_node", "changes": {"branch_label": "approved"}}]},
            "clarification": {"needed": False, "question": "", "reason": ""},
        },
    )

    result = mvp23.run_transaction(deepcopy(before), 1, {"node_ids": [], "edge_ids": []}, "Rename branch label to approved.")

    assert result.status == "needs_clarification"
    assert result.final_graph is None
    assert "ambiguous edge reference" in (result.error or "")
