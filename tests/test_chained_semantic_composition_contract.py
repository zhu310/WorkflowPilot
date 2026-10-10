from __future__ import annotations

from src.semantic_goal import validate_semantic_goal_spec


def _spec(objects: list[dict], relations: list[dict]) -> dict:
    return {
        "status": "resolved",
        "semantic_goals": [
            {
                "id": "g1",
                "goal_category": "relation_goal",
                "text": "ordered relation composition",
                "objects": objects,
                "relation_intents": relations,
                "object_desired_states": [],
                "semantic_constraints": [],
            }
        ],
        "clarification": {"needed": False, "question": "", "reason": ""},
    }


def _node(object_id: str, text: str) -> dict:
    return {"id": object_id, "type": "existing_reference", "text": text, "role": "object"}


def _relations(spec: dict) -> list[dict]:
    result = validate_semantic_goal_spec(spec)
    assert result["valid"] is True
    assert result["ready_for_grounding"] is True
    return result["normalized"]["semantic_goals_v2"][0]["relation_intents"]


def test_two_step_chained_ordering_is_expressible_without_schema_change() -> None:
    relations = _relations(
        _spec(
            [_node("stock", "Stock Check"), _node("payment", "Payment"), _node("ship", "Ship")],
            [
                {"source": "stock", "relation": "before", "target": "payment"},
                {"source": "payment", "relation": "before", "target": "ship"},
            ],
        )
    )

    assert relations == [
        {"source": "stock", "relation": "before", "target": "payment"},
        {"source": "payment", "relation": "before", "target": "ship"},
    ]


def test_three_step_chained_ordering_preserves_adjacent_successors() -> None:
    relations = _relations(
        _spec(
            [_node("a", "A"), _node("b", "B"), _node("c", "C"), _node("d", "D")],
            [
                {"source": "a", "relation": "before", "target": "b"},
                {"source": "b", "relation": "before", "target": "c"},
                {"source": "c", "relation": "before", "target": "d"},
            ],
        )
    )

    assert relations == [
        {"source": "a", "relation": "before", "target": "b"},
        {"source": "b", "relation": "before", "target": "c"},
        {"source": "c", "relation": "before", "target": "d"},
    ]


def test_explicit_fan_out_is_not_rewritten_into_chain() -> None:
    relations = _relations(
        _spec(
            [_node("a", "A"), _node("b", "B"), _node("c", "C")],
            [
                {"source": "a", "relation": "before", "target": "b"},
                {"source": "a", "relation": "before", "target": "c"},
            ],
        )
    )

    assert relations == [
        {"source": "a", "relation": "before", "target": "b"},
        {"source": "a", "relation": "before", "target": "c"},
    ]


def test_independent_multi_goal_sources_remain_independent() -> None:
    relations = _relations(
        _spec(
            [_node("a", "A"), _node("b", "B"), _node("c", "C"), _node("d", "D")],
            [
                {"source": "a", "relation": "before", "target": "b"},
                {"source": "c", "relation": "before", "target": "d"},
            ],
        )
    )

    assert relations == [
        {"source": "a", "relation": "before", "target": "b"},
        {"source": "c", "relation": "before", "target": "d"},
    ]


def test_single_relation_ordering_is_unchanged() -> None:
    relations = _relations(
        _spec(
            [_node("stock", "Stock Check"), _node("payment", "Payment")],
            [{"source": "stock", "relation": "before", "target": "payment"}],
        )
    )

    assert relations == [{"source": "stock", "relation": "before", "target": "payment"}]


def test_after_inverse_representation_remains_accepted_for_canonicalizer() -> None:
    relations = _relations(
        _spec(
            [_node("ship", "Ship"), _node("payment", "Payment")],
            [{"source": "ship", "relation": "after", "target": "payment"}],
        )
    )

    assert relations == [{"source": "ship", "relation": "after", "target": "payment"}]
