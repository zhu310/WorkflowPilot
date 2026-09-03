from __future__ import annotations

from src.workflow_ir_core import ground_requirement
from src.versioning import EditTransaction, transaction_diff


def graph() -> dict:
    return {
        "nodes": [
            {"id": "start", "type": "start", "label": "Start", "x": 0, "y": 0},
            {"id": "identity_check", "type": "task", "label": "Identity Check", "x": 300, "y": 0},
            {"id": "risk_check", "type": "task", "label": "Risk Check", "x": 420, "y": 0},
            {"id": "end", "type": "end", "label": "End", "x": 600, "y": 0},
        ],
        "edges": [
            {"id": "e1", "source": "start", "target": "identity_check", "label": ""},
            {"id": "e2", "source": "identity_check", "target": "risk_check", "label": ""},
            {"id": "e3", "source": "risk_check", "target": "end", "label": ""},
        ],
    }


def recent_diff() -> dict:
    return {
        "transaction_id": "tx7",
        "from_version_id": "v7",
        "to_version_id": "v8",
        "instruction": "Move identity_check and risk_check to the right.",
        "mutations": [
            {
                "mutation_id": "tx7.m1",
                "kind": "property",
                "target_entity_id": "identity_check",
                "property": "x",
                "before": 180,
                "after": 300,
            },
            {
                "mutation_id": "tx7.m2",
                "kind": "property",
                "target_entity_id": "risk_check",
                "property": "x",
                "before": 300,
                "after": 420,
            },
        ],
    }


def property_rows(grounded: dict) -> list[dict]:
    return grounded["grounded_desired_state"]["properties"]


def test_single_property_restore_from_recent_transaction_diff() -> None:
    grounded = ground_requirement(
        graph(),
        {
            "id": "r1",
            "text": "Undo previous move of identity_check.",
            "references": {"source_node": "identity_check"},
        },
        request_context={"recent_transaction_diff": recent_diff()},
    )

    assert grounded["references"]["history_reference_used"] is True
    assert grounded["references"]["matched_transaction_id"] == "tx7"
    assert grounded["references"]["matched_mutation_id"] == "tx7.m1"
    assert property_rows(grounded) == [
        {
            "node_id": "identity_check",
            "target_entity_ref": "identity_check",
            "source_requirement_id": "r1",
            "derivations": {"x": "historical_mutation_reference"},
            "changes": {"x": 180},
        }
    ]


def test_history_restore_preserves_other_mutation() -> None:
    grounded = ground_requirement(
        graph(),
        {
            "id": "r1",
            "text": "Undo previous move of identity_check and keep risk_check.",
            "references": {"source_node": "identity_check"},
        },
        request_context={"recent_transaction_diff": recent_diff()},
    )

    assert grounded["references"]["matched_mutation_ids"] == ["tx7.m1"]
    assert property_rows(grounded)[0]["node_id"] == "identity_check"
    assert property_rows(grounded)[0]["changes"] == {"x": 180}


def test_history_restore_does_not_fall_back_to_label_change() -> None:
    grounded = ground_requirement(
        graph(),
        {
            "id": "r1",
            "text": "Undo the identity_check move and keep risk_check as it is.",
            "references": {"source_node": "identity_check"},
        },
        request_context={"recent_transaction_diff": recent_diff()},
    )

    assert grounded["references"]["history_reference_used"] is True
    assert property_rows(grounded)[0]["changes"] == {"x": 180}


def test_ordinary_explicit_edit_ignores_recent_transaction_diff() -> None:
    grounded = ground_requirement(
        graph(),
        {
            "id": "r1",
            "text": "Rename identity_check to Approved.",
            "references": {"source_node": "identity_check"},
            "desired_state": {
                "properties": [
                    {"node_id": "identity_check", "changes": {"label": "Approved"}},
                ],
            },
        },
        request_context={"recent_transaction_diff": recent_diff()},
    )

    assert grounded["references"].get("history_reference_used") is None
    assert property_rows(grounded) == [
        {
            "node_id": "identity_check",
            "target_entity_ref": "identity_check",
            "source_requirement_id": "r1",
            "derivations": {"label": "grounded_explicit_label"},
            "changes": {"label": "Approved"},
        }
    ]


def test_no_history_context_requires_clarification() -> None:
    grounded = ground_requirement(
        graph(),
        {
            "id": "r1",
            "text": "Undo previous move of identity_check.",
            "references": {"source_node": "identity_check"},
        },
    )

    clarification = grounded["grounded_desired_state"]["clarification"]
    assert clarification["needed"] is True
    assert clarification["reason"] == "missing recent transaction diff"
    assert property_rows(grounded) == []


def test_ordinal_history_reference_requires_clarification() -> None:
    grounded = ground_requirement(
        graph(),
        {
            "id": "r1",
            "text": "Undo the second one.",
        },
        request_context={"recent_transaction_diff": recent_diff()},
    )

    clarification = grounded["grounded_desired_state"]["clarification"]
    assert clarification["needed"] is True
    assert clarification["reason"] == "missing historical target reference"
    assert property_rows(grounded) == []


def test_transaction_diff_records_property_mutations() -> None:
    before = graph()
    after = graph()
    after["nodes"][1]["x"] = 360
    after["nodes"][2]["label"] = "Risk Review"
    transaction = EditTransaction(
        transaction_id="tx1",
        parent_version_id="v1",
        instruction="move and rename",
        selection={"node_ids": [], "edge_ids": []},
        target=None,
        specification=None,
        plan=[],
        resulting_version_id="v2",
        status="success",
    )

    diff = transaction_diff(transaction, before, after)

    assert diff["transaction_id"] == "tx1"
    assert {
        "mutation_id": "tx1.m1",
        "kind": "property",
        "target_entity_id": "identity_check",
        "property": "x",
        "before": 300,
        "after": 360,
    } in diff["mutations"]
    assert {
        "mutation_id": "tx1.m2",
        "kind": "property",
        "target_entity_id": "risk_check",
        "property": "label",
        "before": "Risk Check",
        "after": "Risk Review",
    } in diff["mutations"]
