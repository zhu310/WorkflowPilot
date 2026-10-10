from __future__ import annotations

from copy import deepcopy

from src.canonical_goal_state import relation_fact_endpoint_views, semantic_goal_to_canonical_goal_state


def build_goal(goals: list[dict]) -> dict:
    return {"status": "resolved", "semantic_goals_v2": goals, "clarification": {"needed": False, "question": "", "reason": ""}}


def relation_goal(objects: list[dict], relations: list[dict]) -> dict:
    return build_goal([{"id": "g1", "objects": objects, "relation_intents": relations}])


def node(ref_id: str, text: str) -> dict:
    return {"id": ref_id, "type": "existing_reference", "text": text, "role": "object"}


def edge(ref_id: str, text: str) -> dict:
    return {"id": ref_id, "type": "edge_reference", "text": text, "role": "object"}


def relation_facts(goal_state: dict) -> list[dict]:
    return goal_state.get("relation_facts", [])


def test_before_binary_relation_adapts_to_one_shadow_precedence_relation_fact() -> None:
    goal_state, warnings = semantic_goal_to_canonical_goal_state(
        relation_goal(
            [node("a", "A"), node("b", "B")],
            [{"source": "a", "relation": "before", "target": "b"}],
        ),
        instruction="A before B",
    )

    assert warnings == []
    assert goal_state["desired_facts"][0]["predicate"] == "relation.before"
    assert len(relation_facts(goal_state)) == 1
    shadow = relation_facts(goal_state)[0]
    assert shadow["relation_id"] == "rf:f1"
    assert shadow["kind"] == "precedence"
    assert shadow["members"] == [
        {"source_ref": "r1", "target_ref": "r2", "properties": {}, "provenance_refs": ["p1"]}
    ]
    assert shadow["group_semantics"] == {}
    assert shadow["provenance_refs"] == ["p1"]
    assert relation_fact_endpoint_views(shadow) == {"sources": ["r1"], "targets": ["r2"], "cardinality": (1, 1)}


def test_after_binary_relation_uses_existing_canonical_before_direction() -> None:
    goal_state, _warnings = semantic_goal_to_canonical_goal_state(
        relation_goal(
            [node("a", "A"), node("b", "B")],
            [{"source": "a", "relation": "after", "target": "b"}],
        ),
        instruction="A after B",
    )

    fact = goal_state["desired_facts"][0]
    shadow = goal_state["relation_facts"][0]
    refs = {row["ref_id"]: row for row in goal_state["references"]}

    assert fact["predicate"] == "relation.before"
    assert refs[fact["operands"]["subject"]["ref_id"]]["mention"]["text"] == "B"
    assert refs[fact["operands"]["anchor"]["ref_id"]]["mention"]["text"] == "A"
    assert shadow["members"] == [
        {
            "source_ref": fact["operands"]["subject"]["ref_id"],
            "target_ref": fact["operands"]["anchor"]["ref_id"],
            "properties": {},
            "provenance_refs": ["p1"],
        }
    ]


def test_relation_id_is_runtime_owned_and_deterministic_from_current_fact_identity() -> None:
    source = relation_goal(
        [node("a", "A"), node("b", "B")],
        [{"source": "a", "relation": "before", "target": "b"}],
    )

    left, _ = semantic_goal_to_canonical_goal_state(deepcopy(source), instruction="A before B")
    right, _ = semantic_goal_to_canonical_goal_state(deepcopy(source), instruction="A before B")

    assert left["relation_facts"][0]["relation_id"] == "rf:f1"
    assert left["relation_facts"] == right["relation_facts"]


def test_two_unrelated_before_facts_create_two_shadow_relation_facts_without_grouping() -> None:
    goal_state, _warnings = semantic_goal_to_canonical_goal_state(
        relation_goal(
            [node("a", "A"), node("b", "B"), node("c", "C"), node("d", "D")],
            [
                {"source": "a", "relation": "before", "target": "b"},
                {"source": "c", "relation": "before", "target": "d"},
            ],
        ),
        instruction="A before B and C before D",
    )

    assert len(goal_state["desired_facts"]) == 2
    assert len(goal_state["relation_facts"]) == 2
    assert [len(row["members"]) for row in goal_state["relation_facts"]] == [1, 1]


def test_same_target_before_facts_do_not_auto_group() -> None:
    goal_state, _warnings = semantic_goal_to_canonical_goal_state(
        relation_goal(
            [node("a", "A"), node("b", "B"), node("c", "C")],
            [
                {"source": "a", "relation": "before", "target": "c"},
                {"source": "b", "relation": "before", "target": "c"},
            ],
        ),
        instruction="A and B before C",
    )

    assert len(goal_state["relation_facts"]) == 2
    assert goal_state["relation_facts"][0]["members"] == [
        {"source_ref": "r1", "target_ref": "r3", "properties": {}, "provenance_refs": ["p1"]}
    ]
    assert goal_state["relation_facts"][1]["members"] == [
        {"source_ref": "r2", "target_ref": "r3", "properties": {}, "provenance_refs": ["p1"]}
    ]


def test_same_source_before_facts_do_not_auto_group() -> None:
    goal_state, _warnings = semantic_goal_to_canonical_goal_state(
        relation_goal(
            [node("a", "A"), node("b", "B"), node("c", "C")],
            [
                {"source": "a", "relation": "before", "target": "b"},
                {"source": "a", "relation": "before", "target": "c"},
            ],
        ),
        instruction="A before B and C",
    )

    assert len(goal_state["relation_facts"]) == 2
    assert [row["members"][0]["source_ref"] for row in goal_state["relation_facts"]] == ["r1", "r1"]
    assert [row["members"][0]["target_ref"] for row in goal_state["relation_facts"]] == ["r2", "r3"]


def test_route_to_does_not_migrate_to_shadow_relation_fact() -> None:
    goal_state, _warnings = semantic_goal_to_canonical_goal_state(
        relation_goal(
            [edge("e", "failure branch"), node("n", "Reject")],
            [{"source": "e", "relation": "route_to", "target": "n"}],
        ),
        instruction="send failure branch to Reject",
    )

    assert goal_state["desired_facts"][0]["predicate"] == "relation.route_to"
    assert "relation_facts" not in goal_state


def test_parallel_connected_and_joins_before_do_not_migrate_to_shadow_relation_fact() -> None:
    for relation in ("parallel_with", "connected_to", "joins_before"):
        goal_state, _warnings = semantic_goal_to_canonical_goal_state(
            relation_goal(
                [node("a", "A"), node("b", "B")],
                [{"source": "a", "relation": relation, "target": "b"}],
            ),
            instruction=f"A {relation} B",
        )

        assert goal_state["desired_facts"][0]["predicate"] == f"relation.{relation}"
        assert "relation_facts" not in goal_state


def test_canonical_goal_state_without_relation_facts_remains_valid_for_non_relation_goal() -> None:
    goal_state, warnings = semantic_goal_to_canonical_goal_state(
        build_goal(
            [
                {
                    "id": "g1",
                    "objects": [{"id": "n", "type": "existing_reference", "text": "Review", "role": "object"}],
                    "object_desired_states": [{"object": "n", "state": "exists"}],
                }
            ]
        ),
        instruction="keep Review",
    )

    assert warnings == []
    assert "relation_facts" not in goal_state
    assert goal_state["desired_facts"][0]["predicate"] == "entity.exists"


def test_desired_fact_output_remains_authoritative_and_unchanged_for_before() -> None:
    before_r2_shape, _warnings = semantic_goal_to_canonical_goal_state(
        relation_goal(
            [node("a", "A"), node("b", "B")],
            [{"source": "a", "relation": "before", "target": "b"}],
        ),
        instruction="A before B",
    )

    authoritative = {key: value for key, value in before_r2_shape.items() if key != "relation_facts"}

    assert authoritative == {
        "goal_state_id": "gs1",
        "revision": "phase1a.shadow.v1",
        "references": before_r2_shape["references"],
        "desired_facts": before_r2_shape["desired_facts"],
        "constraints": [],
        "open_statements": [],
    }
    assert before_r2_shape["desired_facts"] == [
        {
            "fact_id": "f1",
            "predicate": "relation.before",
            "predicate_signature_id": "relation.before",
            "operands": {
                "subject": {"operand_type": "reference", "ref_id": "r1"},
                "anchor": {"operand_type": "reference", "ref_id": "r2"},
            },
            "provenance_refs": ["p1"],
            "fact_key": before_r2_shape["desired_facts"][0]["fact_key"],
        }
    ]
