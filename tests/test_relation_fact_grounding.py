from __future__ import annotations

from src.canonical_goal_state import semantic_goal_to_canonical_goal_state
from src.semantic_goal import ground_canonical_relation_facts, ground_semantic_goal_spec, validate_semantic_goal_spec


def graph() -> dict:
    return {
        "nodes": [
            {"id": "a", "label": "A", "type": "task"},
            {"id": "b", "label": "B", "type": "task"},
            {"id": "c", "label": "C", "type": "task"},
            {"id": "d", "label": "D", "type": "task"},
        ],
        "edges": [],
    }


def build_spec(objects: list[dict], relation_facts: list[dict] | None = None, *, states: list[dict] | None = None, relation_intents: list[dict] | None = None) -> dict:
    return {
        "status": "resolved",
        "semantic_goals_v2": [
            {
                "id": "g1",
                "goal_category": "relation_goal",
                "objects": objects,
                "object_desired_states": states or [],
                "relation_intents": relation_intents or [],
                "relation_facts": relation_facts or [],
            }
        ],
        "clarification": {"needed": False, "question": "", "reason": ""},
    }


def node(object_id: str, text: str) -> dict:
    return {"id": object_id, "type": "existing_reference", "text": text, "role": "object"}


def new_node(object_id: str, label: str) -> dict:
    return {"id": object_id, "type": "new_entity", "label": label, "role": "object"}


def grouped(kind: str, members: list[dict], group_semantics: dict | None = None) -> dict:
    return {"kind": kind, "members": members, "group_semantics": group_semantics or {}}


def canonical_and_ground(spec: dict) -> tuple[dict, dict]:
    goal_state, warnings = semantic_goal_to_canonical_goal_state(spec, instruction="deterministic grouped relation")
    assert warnings == []
    grounded = ground_canonical_relation_facts(goal_state, graph(), {"node_ids": [], "edge_ids": []})
    return goal_state, grounded


def test_grouped_one_to_one_survives_canonicalization_and_grounding() -> None:
    spec = build_spec(
        [node("a_obj", "A"), node("b_obj", "B")],
        [grouped("connection", [{"source": "a_obj", "target": "b_obj", "properties": {}}])],
    )

    goal_state, grounded = canonical_and_ground(spec)

    assert len(goal_state["relation_facts"]) == 1
    assert goal_state["relation_facts"][0]["relation_id"] == "rg1"
    assert grounded["relation_facts"][0]["relation_id"] == "rg1"
    assert grounded["relation_facts"][0]["members"][0]["source_binding"]["id"] == "a"
    assert grounded["relation_facts"][0]["members"][0]["target_binding"]["id"] == "b"


def test_grouped_one_to_many_survives_as_one_relation_fact() -> None:
    spec = build_spec(
        [node("a_obj", "A"), node("b_obj", "B"), node("c_obj", "C")],
        [
            grouped(
                "connection",
                [
                    {"source": "a_obj", "target": "b_obj", "properties": {}},
                    {"source": "a_obj", "target": "c_obj", "properties": {}},
                ],
            )
        ],
    )

    goal_state, grounded = canonical_and_ground(spec)

    assert len(goal_state["relation_facts"]) == 1
    assert len(goal_state["relation_facts"][0]["members"]) == 2
    assert [member["source_ref"] for member in goal_state["relation_facts"][0]["members"]] == ["r1", "r1"]
    assert [member["target_binding"]["id"] for member in grounded["relation_facts"][0]["members"]] == ["b", "c"]


def test_grouped_many_to_one_survives_as_one_relation_fact() -> None:
    spec = build_spec(
        [node("a_obj", "A"), node("b_obj", "B"), node("c_obj", "C")],
        [
            grouped(
                "connection",
                [
                    {"source": "a_obj", "target": "c_obj", "properties": {}},
                    {"source": "b_obj", "target": "c_obj", "properties": {}},
                ],
            )
        ],
    )

    goal_state, grounded = canonical_and_ground(spec)

    assert len(goal_state["relation_facts"]) == 1
    assert [member["target_ref"] for member in goal_state["relation_facts"][0]["members"]] == ["r3", "r3"]
    assert [member["source_binding"]["id"] for member in grounded["relation_facts"][0]["members"]] == ["a", "b"]


def test_explicit_mapped_many_to_many_survives_representation_and_grounding() -> None:
    spec = build_spec(
        [node("a_obj", "A"), node("b_obj", "B"), node("c_obj", "C"), node("d_obj", "D")],
        [
            grouped(
                "connection",
                [
                    {"source": "a_obj", "target": "c_obj", "properties": {}},
                    {"source": "b_obj", "target": "d_obj", "properties": {}},
                ],
            )
        ],
    )

    goal_state, grounded = canonical_and_ground(spec)

    assert len(goal_state["relation_facts"]) == 1
    assert [(member["source_ref"], member["target_ref"]) for member in goal_state["relation_facts"][0]["members"]] == [
        ("r1", "r3"),
        ("r2", "r4"),
    ]
    assert [(member["source_binding"]["id"], member["target_binding"]["id"]) for member in grounded["relation_facts"][0]["members"]] == [
        ("a", "c"),
        ("b", "d"),
    ]


def test_grouped_relation_gets_exactly_one_runtime_relation_id() -> None:
    spec = build_spec(
        [node("a_obj", "A"), node("b_obj", "B"), node("c_obj", "C")],
        [
            grouped(
                "precedence",
                [
                    {"source": "a_obj", "target": "c_obj", "properties": {}},
                    {"source": "b_obj", "target": "c_obj", "properties": {}},
                ],
            )
        ],
    )

    goal_state, grounded = canonical_and_ground(spec)

    assert [row["relation_id"] for row in goal_state["relation_facts"]] == ["rg1"]
    assert [row["relation_id"] for row in grounded["relation_facts"]] == ["rg1"]


def test_same_existing_ref_is_reused_across_members() -> None:
    spec = build_spec(
        [node("a_obj", "A"), node("b_obj", "B"), node("c_obj", "C")],
        [
            grouped(
                "connection",
                [
                    {"source": "a_obj", "target": "b_obj", "properties": {}},
                    {"source": "a_obj", "target": "c_obj", "properties": {}},
                ],
            )
        ],
    )

    goal_state, grounded = canonical_and_ground(spec)

    assert [member["source_ref"] for member in goal_state["relation_facts"][0]["members"]] == ["r1", "r1"]
    assert [member["source_binding"]["id"] for member in grounded["relation_facts"][0]["members"]] == ["a", "a"]


def test_same_new_entity_ref_is_reused_across_members() -> None:
    spec = build_spec(
        [node("a_obj", "A"), node("b_obj", "B"), new_node("x_obj", "X"), node("c_obj", "C")],
        [
            grouped(
                "precedence",
                [
                    {"source": "a_obj", "target": "x_obj", "properties": {}},
                    {"source": "b_obj", "target": "x_obj", "properties": {}},
                    {"source": "x_obj", "target": "c_obj", "properties": {}},
                ],
            )
        ],
        states=[{"object": "x_obj", "state": "exists"}],
    )

    goal_state, grounded = canonical_and_ground(spec)

    assert [member["target_ref"] for member in goal_state["relation_facts"][0]["members"][:2]] == ["r3", "r3"]
    assert goal_state["desired_facts"][0]["predicate"] == "entity.exists"
    assert grounded["relation_facts"][0]["members"][0]["target_binding"] == {
        "ref_id": "r3",
        "binding_type": "new_entity",
        "needs_creation": True,
        "label": "X",
    }
    assert grounded["relation_facts"][0]["members"][1]["target_binding"]["ref_id"] == "r3"
    assert grounded["relation_facts"][0]["members"][2]["source_binding"]["ref_id"] == "r3"


def test_member_properties_are_preserved_without_lowering() -> None:
    spec = build_spec(
        [node("a_obj", "A"), node("b_obj", "B"), node("c_obj", "C")],
        [
            grouped(
                "connection",
                [
                    {"source": "a_obj", "target": "b_obj", "properties": {"branch_label": "pass"}},
                    {"source": "a_obj", "target": "c_obj", "properties": {"condition.expression": "score < 60"}},
                ],
            )
        ],
    )

    goal_state, grounded = canonical_and_ground(spec)

    assert [member["properties"] for member in goal_state["relation_facts"][0]["members"]] == [
        {"branch_label": "pass"},
        {"condition.expression": "score < 60"},
    ]
    assert [member["properties"] for member in grounded["relation_facts"][0]["members"]] == [
        {"branch_label": "pass"},
        {"condition.expression": "score < 60"},
    ]


def test_relation_and_member_provenance_are_preserved() -> None:
    spec = build_spec(
        [node("a_obj", "A"), node("b_obj", "B")],
        [grouped("precedence", [{"source": "a_obj", "target": "b_obj", "properties": {}}])],
    )

    goal_state, grounded = canonical_and_ground(spec)

    assert goal_state["relation_facts"][0]["provenance_refs"] == ["p1"]
    assert goal_state["relation_facts"][0]["members"][0]["provenance_refs"] == ["p1"]
    assert grounded["relation_facts"][0]["provenance_refs"] == ["p1"]
    assert grounded["relation_facts"][0]["members"][0]["provenance_refs"] == ["p1"]


def test_group_semantics_shape_is_preserved() -> None:
    spec = build_spec(
        [node("a_obj", "A"), node("b_obj", "B")],
        [grouped("connection", [{"source": "a_obj", "target": "b_obj", "properties": {}}], group_semantics={"join_mode": "all"})],
    )

    goal_state, grounded = canonical_and_ground(spec)

    assert goal_state["relation_facts"][0]["group_semantics"] == {"join_mode": "all"}
    assert grounded["relation_facts"][0]["group_semantics"] == {"join_mode": "all"}


def test_undeclared_grouped_relation_endpoint_is_rejected() -> None:
    spec = build_spec(
        [node("a_obj", "A")],
        [grouped("connection", [{"source": "a_obj", "target": "missing", "properties": {}}])],
    )

    result = validate_semantic_goal_spec(spec, source_instruction="A to missing")

    assert "unknown_grouped_relation_object" in {issue["kind"] for issue in result["issues"]}


def test_missing_endpoint_grounding_reports_missing_node_reference() -> None:
    spec = build_spec(
        [node("a_obj", "A"), node("z_obj", "Z")],
        [grouped("connection", [{"source": "a_obj", "target": "z_obj", "properties": {}}])],
    )

    goal_state, warnings = semantic_goal_to_canonical_goal_state(spec, instruction="A to Z")
    assert warnings == []
    grounded = ground_canonical_relation_facts(goal_state, graph(), {"node_ids": [], "edge_ids": []})

    assert any(issue["kind"] == "missing_node_reference" for issue in grounded["grounding"]["issues"])


def test_unrelated_legacy_binary_facts_never_auto_group() -> None:
    spec = build_spec(
        [node("a_obj", "A"), node("b_obj", "B"), node("c_obj", "C")],
        relation_intents=[
            {"source": "a_obj", "relation": "before", "target": "c_obj"},
            {"source": "b_obj", "relation": "before", "target": "c_obj"},
        ],
    )

    goal_state, warnings = semantic_goal_to_canonical_goal_state(spec, instruction="A and B before C")

    assert warnings == []
    assert len(goal_state["relation_facts"]) == 2
    assert [len(row["members"]) for row in goal_state["relation_facts"]] == [1, 1]


def test_route_to_path_is_unchanged_by_grouped_relation_pipeline() -> None:
    spec = build_spec(
        [
            {"id": "route", "type": "edge_reference", "text": "failure branch", "role": "route"},
            node("target", "C"),
        ],
        relation_intents=[{"source": "route", "relation": "route_to", "target": "target"}],
    )

    goal_state, _warnings = semantic_goal_to_canonical_goal_state(spec, instruction="send failure branch to C")
    grounded = ground_semantic_goal_spec(spec, graph(), {"node_ids": [], "edge_ids": []})

    assert goal_state["desired_facts"][0]["predicate"] == "relation.route_to"
    assert "relation_facts" not in goal_state
    assert grounded["semantic_goals_v2"][0]["relation_intents"] == [{"source": "route", "relation": "route_to", "target": "target"}]
