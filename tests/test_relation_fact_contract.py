from __future__ import annotations

from src.canonical_goal_state import (
    _validate_goal_state,
    canonical_json,
    normalize_relation_fact,
    relation_fact_endpoint_views,
    semantic_goal_to_canonical_goal_state,
    validate_relation_fact,
)


def relation_fact(members: list[dict], **overrides: object) -> dict:
    value = {
        "relation_id": "r1",
        "kind": "connection",
        "members": members,
        "group_semantics": {},
        "provenance_refs": [],
    }
    value.update(overrides)
    return value


def assert_valid(value: dict) -> dict:
    normalized = normalize_relation_fact(value)
    assert validate_relation_fact(normalized) == []
    return normalized


def test_valid_one_to_one_relation_fact() -> None:
    normalized = assert_valid(relation_fact([{"source_ref": "A", "target_ref": "B"}]))

    assert normalized == {
        "relation_id": "r1",
        "kind": "connection",
        "members": [{"source_ref": "A", "target_ref": "B", "properties": {}, "provenance_refs": []}],
        "group_semantics": {},
        "provenance_refs": [],
    }
    assert relation_fact_endpoint_views(normalized) == {"sources": ["A"], "targets": ["B"], "cardinality": (1, 1)}


def test_valid_one_to_many_relation_fact() -> None:
    normalized = assert_valid(
        relation_fact(
            [
                {"source_ref": "A", "target_ref": "B"},
                {"source_ref": "A", "target_ref": "C"},
            ]
        )
    )

    assert relation_fact_endpoint_views(normalized) == {"sources": ["A"], "targets": ["B", "C"], "cardinality": (1, 2)}


def test_valid_many_to_one_relation_fact() -> None:
    normalized = assert_valid(
        relation_fact(
            [
                {"source_ref": "A", "target_ref": "C"},
                {"source_ref": "B", "target_ref": "C"},
            ],
            kind="precedence",
        )
    )

    assert relation_fact_endpoint_views(normalized) == {"sources": ["A", "B"], "targets": ["C"], "cardinality": (2, 1)}


def test_valid_explicit_mapped_many_to_many_relation_fact() -> None:
    normalized = assert_valid(
        relation_fact(
            [
                {"source_ref": "A", "target_ref": "C"},
                {"source_ref": "B", "target_ref": "D"},
            ]
        )
    )

    assert relation_fact_endpoint_views(normalized) == {"sources": ["A", "B"], "targets": ["C", "D"], "cardinality": (2, 2)}
    assert normalized["members"] == [
        {"source_ref": "A", "target_ref": "C", "properties": {}, "provenance_refs": []},
        {"source_ref": "B", "target_ref": "D", "properties": {}, "provenance_refs": []},
    ]


def test_member_properties_and_provenance_are_preserved() -> None:
    normalized = assert_valid(
        relation_fact(
            [
                {
                    "source_ref": "A",
                    "target_ref": "B",
                    "properties": {"branch_label": "approved", "condition.expression": "score >= 90"},
                    "provenance_refs": ["p_member"],
                }
            ],
            group_semantics={"mode": "choice"},
            provenance_refs=["p_relation"],
        )
    )

    assert normalized["group_semantics"] == {"mode": "choice"}
    assert normalized["provenance_refs"] == ["p_relation"]
    assert normalized["members"][0]["properties"] == {"branch_label": "approved", "condition.expression": "score >= 90"}
    assert normalized["members"][0]["provenance_refs"] == ["p_member"]


def test_canonical_serialization_is_stable_for_relation_facts() -> None:
    left = normalize_relation_fact(
        relation_fact(
            [{"target_ref": "B", "source_ref": "A", "properties": {"branch_label": "ok"}}],
            provenance_refs=["p1"],
        )
    )
    right = {
        "kind": "connection",
        "relation_id": "r1",
        "provenance_refs": ["p1"],
        "group_semantics": {},
        "members": [{"provenance_refs": [], "properties": {"branch_label": "ok"}, "target_ref": "B", "source_ref": "A"}],
    }

    assert canonical_json(left) == canonical_json(right)


def test_relation_fact_rejects_invalid_required_fields() -> None:
    cases = [
        ({}, "missing_relation_id"),
        (relation_fact([{"source_ref": "A", "target_ref": "B"}], relation_id=""), "missing_relation_id"),
        (relation_fact([{"source_ref": "A", "target_ref": "B"}], kind="route"), "invalid_relation_kind"),
        (relation_fact([]), "empty_relation_members"),
        (relation_fact([{"target_ref": "B"}]), "missing_member_source_ref"),
        (relation_fact([{"source_ref": "A"}]), "missing_member_target_ref"),
        (relation_fact([{"source_ref": "A", "target_ref": "B", "properties": []}]), "invalid_member_properties"),
    ]

    for value, expected_kind in cases:
        assert expected_kind in {error["kind"] for error in validate_relation_fact(value)}


def test_sources_targets_and_cardinality_are_derived_not_canonical() -> None:
    raw = relation_fact(
        [{"source_ref": "A", "target_ref": "B"}],
        sources=["wrong"],
        targets=["wrong"],
        cardinality=(9, 9),
    )
    normalized = normalize_relation_fact(raw)

    assert "sources" not in normalized
    assert "targets" not in normalized
    assert "cardinality" not in normalized
    assert relation_fact_endpoint_views(normalized) == {"sources": ["A"], "targets": ["B"], "cardinality": (1, 1)}
    assert {error["kind"] for error in validate_relation_fact(raw)} == {"derived_relation_fact_field"}


def test_existing_canonical_goal_state_without_relation_facts_is_unchanged() -> None:
    goal_state, warnings = semantic_goal_to_canonical_goal_state(
        {
            "status": "resolved",
            "semantic_goals_v2": [
                {
                    "id": "g1",
                    "objects": [{"id": "n1", "type": "existing_reference", "text": "Review", "role": "object"}],
                    "object_desired_states": [{"object": "n1", "state": "exists"}],
                }
            ],
            "clarification": {"needed": False, "question": "", "reason": ""},
        },
        instruction="keep Review",
    )

    assert warnings == []
    assert set(goal_state) == {
        "goal_state_id",
        "revision",
        "references",
        "desired_facts",
        "constraints",
        "open_statements",
    }
    assert "relation_facts" not in goal_state
    assert goal_state["desired_facts"][0]["predicate"] == "entity.exists"


def test_canonical_goal_state_can_carry_valid_relation_facts() -> None:
    goal_state = {
        "goal_state_id": "gs1",
        "revision": "phase1a.shadow.v1",
        "references": [],
        "desired_facts": [],
        "constraints": [],
        "open_statements": [],
        "relation_facts": [
            normalize_relation_fact(relation_fact([{"source_ref": "A", "target_ref": "B"}]))
        ],
    }

    assert _validate_goal_state(goal_state, []) == {"ok": True, "errors": [], "warnings": []}


def test_canonical_goal_state_rejects_stored_relation_fact_endpoint_views() -> None:
    goal_state = {
        "goal_state_id": "gs1",
        "revision": "phase1a.shadow.v1",
        "references": [],
        "desired_facts": [],
        "constraints": [],
        "open_statements": [],
        "relation_facts": [
            relation_fact([{"source_ref": "A", "target_ref": "B"}], sources=["A"])
        ],
    }

    result = _validate_goal_state(goal_state, [])

    assert result["ok"] is False
    assert result["errors"][0]["error"]["kind"] == "derived_relation_fact_field"
