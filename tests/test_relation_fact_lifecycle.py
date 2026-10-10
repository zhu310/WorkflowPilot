from __future__ import annotations

from copy import deepcopy

from src.phase1b_shadow import (
    empty_goal_state,
    merge_goal_state,
    plan_semantic_delta,
    run_phase1b_shadow,
    semantic_equivalence_key,
    validate_semantic_delta,
)
from src.semantic_goal import ground_canonical_relation_facts


def ref(ref_id: str, text: str) -> dict:
    return {
        "ref_id": ref_id,
        "entity_type": "node",
        "cardinality": "single",
        "mention": {"text": text, "span": {"utterance_id": "u1", "start": 0, "end": len(text)}},
        "provenance_refs": ["p1"],
    }


def exists_fact(ref_id: str) -> dict:
    return {
        "predicate": "entity.exists",
        "predicate_signature_id": "entity.exists",
        "operands": {"subject": {"operand_type": "reference", "ref_id": ref_id}},
        "provenance_refs": ["p1"],
    }


def relation_fact(members: list[tuple[str, str]], *, relation_id: str = "", kind: str = "connection", properties: list[dict] | None = None, provenance_refs: list[str] | None = None) -> dict:
    properties = properties or [{} for _ in members]
    return {
        "relation_id": relation_id,
        "kind": kind,
        "members": [
            {"source_ref": source, "target_ref": target, "properties": props, "provenance_refs": provenance_refs or ["p1"]}
            for (source, target), props in zip(members, properties)
        ],
        "group_semantics": {},
        "provenance_refs": provenance_refs or ["p1"],
    }


def proposal(*, refs: list[dict] | None = None, relation_facts: list[dict] | None = None, desired_facts: list[dict] | None = None) -> dict:
    return {
        "references": refs or [],
        "desired_facts": desired_facts or [],
        "relation_facts": relation_facts or [],
        "constraints": [],
        "open_statements": [],
    }


def turn(relation: str = "CONTINUE") -> dict:
    return {"relation": relation, "task_context_ref": "task1", "evidence_refs": ["p1"], "reason_codes": []}


def apply(goal_state: dict, semantic_proposal: dict, relation: str = "CONTINUE") -> tuple[dict, dict]:
    planned = plan_semantic_delta(goal_state, turn(relation), semantic_proposal)
    assert planned["errors"] == []
    merged = merge_goal_state(goal_state if relation != "NEW_TASK" else empty_goal_state(), planned["semantic_delta"], [{"provenance_id": "p1"}])
    assert merged["ok"] is True
    return planned["semantic_delta"], merged["goal_state"]


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


def test_add_grouped_one_to_many_relation_fact() -> None:
    _, after = apply(
        empty_goal_state(),
        proposal(refs=[ref("r1", "A"), ref("r2", "B"), ref("r3", "C")], relation_facts=[relation_fact([("r1", "r2"), ("r1", "r3")])]),
    )

    assert after["relation_facts"][0]["relation_id"] == "rg1"
    assert len(after["relation_facts"][0]["members"]) == 2


def test_add_grouped_many_to_one_relation_fact() -> None:
    _, after = apply(
        empty_goal_state(),
        proposal(refs=[ref("r1", "A"), ref("r2", "B"), ref("r3", "C")], relation_facts=[relation_fact([("r1", "r3"), ("r2", "r3")])]),
    )

    assert after["relation_facts"][0]["relation_id"] == "rg1"
    assert [member["target_ref"] for member in after["relation_facts"][0]["members"]] == ["r3", "r3"]


def test_continue_unrelated_property_goal_preserves_relation_id() -> None:
    _, before = apply(
        empty_goal_state(),
        proposal(refs=[ref("r1", "A"), ref("r2", "B")], relation_facts=[relation_fact([("r1", "r2")])]),
    )
    _, after = apply(
        before,
        proposal(refs=[ref("r3", "C")], desired_facts=[exists_fact("r3")]),
    )

    assert before["relation_facts"] == after["relation_facts"]
    assert after["relation_facts"][0]["relation_id"] == "rg1"


def test_exact_repeated_grouped_relation_dedupes_without_duplicate() -> None:
    _, before = apply(
        empty_goal_state(),
        proposal(refs=[ref("r1", "A"), ref("r2", "B")], relation_facts=[relation_fact([("r1", "r2")])]),
    )
    delta, after = apply(before, proposal(relation_facts=[relation_fact([("r1", "r2")], provenance_refs=["p2"])]))

    assert delta["operations"] == []
    assert after["relation_facts"] == before["relation_facts"]


def test_correct_target_by_same_sources_preserves_relation_id() -> None:
    _, before = apply(
        empty_goal_state(),
        proposal(refs=[ref("r1", "A"), ref("r2", "B"), ref("r3", "C")], relation_facts=[relation_fact([("r1", "r3"), ("r2", "r3")])]),
    )
    _, after = apply(before, proposal(refs=[ref("r4", "D")], relation_facts=[relation_fact([("r1", "r4"), ("r2", "r4")])]), relation="CORRECT")

    assert after["relation_facts"][0]["relation_id"] == "rg1"
    assert [(member["source_ref"], member["target_ref"]) for member in after["relation_facts"][0]["members"]] == [("r1", "r4"), ("r2", "r4")]


def test_correct_member_set_by_same_target_preserves_relation_id() -> None:
    _, before = apply(
        empty_goal_state(),
        proposal(refs=[ref("r1", "A"), ref("r2", "B"), ref("r3", "C")], relation_facts=[relation_fact([("r1", "r3"), ("r2", "r3")])]),
    )
    _, after = apply(before, proposal(refs=[ref("r4", "D")], relation_facts=[relation_fact([("r1", "r3"), ("r4", "r3")])]), relation="CORRECT")

    assert after["relation_facts"][0]["relation_id"] == "rg1"
    assert [(member["source_ref"], member["target_ref"]) for member in after["relation_facts"][0]["members"]] == [("r1", "r3"), ("r4", "r3")]


def test_ambiguous_relation_fact_correction_does_not_choose_arbitrarily() -> None:
    _, before = apply(
        empty_goal_state(),
        proposal(
            refs=[ref("r1", "A"), ref("r2", "B"), ref("r3", "C"), ref("r4", "D")],
            relation_facts=[relation_fact([("r1", "r3"), ("r2", "r3")]), relation_fact([("r1", "r4"), ("r2", "r4")])],
        ),
    )
    planned = plan_semantic_delta(before, turn("CORRECT"), proposal(refs=[ref("r5", "E")], relation_facts=[relation_fact([("r1", "r5"), ("r2", "r5")])]))

    assert planned["semantic_delta"]["operations"] == []
    assert planned["errors"][0]["message"] == "CORRECT relation_fact target is ambiguous"


def test_retract_exact_existing_relation_fact() -> None:
    _, before = apply(
        empty_goal_state(),
        proposal(refs=[ref("r1", "A"), ref("r2", "B")], relation_facts=[relation_fact([("r1", "r2")])]),
    )
    delta = {"delta_id": "d1", "operations": [{"op": "RETRACT", "element_kind": "relation_fact", "target_id": "rg1", "value": {}, "evidence_refs": []}]}

    assert validate_semantic_delta(delta, before) == []
    merged = merge_goal_state(before, delta, [{"provenance_id": "p1"}])
    assert merged["ok"] is True
    assert merged["goal_state"].get("relation_facts", []) == []


def test_new_task_clears_relation_facts() -> None:
    _, before = apply(
        empty_goal_state(),
        proposal(refs=[ref("r1", "A"), ref("r2", "B")], relation_facts=[relation_fact([("r1", "r2")])]),
    )

    run = run_phase1b_shadow("start over", current_goal_state=before, model_runner=lambda _system, _payload: {"turn_relation": turn("NEW_TASK"), "semantic_proposal": proposal()})

    assert run.ok
    assert "relation_facts" not in run.artifact["after_goal_state"]


def test_multiple_unrelated_relation_facts_preserve_independent_ids_and_reorder() -> None:
    _, before = apply(
        empty_goal_state(),
        proposal(refs=[ref("r1", "A"), ref("r2", "B"), ref("r3", "C")], relation_facts=[relation_fact([("r1", "r2")]), relation_fact([("r2", "r3")])]),
    )
    reordered = deepcopy(before)
    reordered["relation_facts"] = list(reversed(reordered["relation_facts"]))
    _, after = apply(reordered, proposal(refs=[ref("r4", "D")], desired_facts=[exists_fact("r4")]))

    assert {row["relation_id"] for row in after["relation_facts"]} == {"rg1", "rg2"}


def test_semantic_equivalence_ignores_relation_id_and_provenance_but_not_members() -> None:
    base = empty_goal_state()
    base["references"] = [ref("r1", "A"), ref("r2", "B"), ref("r3", "C")]
    left = deepcopy(base)
    right = deepcopy(base)
    changed = deepcopy(base)
    left["relation_facts"] = [relation_fact([("r1", "r2")], relation_id="rg1", provenance_refs=["p1"])]
    right["relation_facts"] = [relation_fact([("r1", "r2")], relation_id="rg99", provenance_refs=["p99"])]
    changed["relation_facts"] = [relation_fact([("r1", "r3")], relation_id="rg1", provenance_refs=["p1"])]

    assert semantic_equivalence_key(left) == semantic_equivalence_key(right)
    assert semantic_equivalence_key(left) != semantic_equivalence_key(changed)


def test_same_new_entity_ref_reused_across_relation_members_after_lifecycle() -> None:
    _, after = apply(
        empty_goal_state(),
        proposal(
            refs=[ref("r1", "A"), ref("r2", "B"), ref("r3", "X"), ref("r4", "C")],
            desired_facts=[exists_fact("r3")],
            relation_facts=[relation_fact([("r1", "r3"), ("r2", "r3"), ("r3", "r4")])],
        ),
    )

    assert after["relation_facts"][0]["members"][0]["target_ref"] == "r3"
    assert after["relation_facts"][0]["members"][1]["target_ref"] == "r3"
    assert after["relation_facts"][0]["members"][2]["source_ref"] == "r3"


def test_legacy_shadow_relation_fact_is_not_managed_grouped_merge_target() -> None:
    before = empty_goal_state()
    before["references"] = [ref("r1", "A"), ref("r2", "B")]
    before["relation_facts"] = [relation_fact([("r1", "r2")], relation_id="rf:f1")]

    delta, after = apply(before, proposal(relation_facts=[relation_fact([("r1", "r2")])]))

    assert delta["operations"][0]["op"] == "ADD"
    assert [row["relation_id"] for row in after["relation_facts"]] == ["rf:f1", "rg1"]


def test_merged_goal_state_relation_facts_ground_after_lifecycle() -> None:
    _, after = apply(
        empty_goal_state(),
        proposal(refs=[ref("r1", "A"), ref("r2", "B"), ref("r3", "C")], relation_facts=[relation_fact([("r1", "r3"), ("r2", "r3")])]),
    )

    grounded = ground_canonical_relation_facts(after, graph(), {"node_ids": [], "edge_ids": []})

    assert grounded["grounding"]["issues"] == []
    assert grounded["relation_facts"][0]["relation_id"] == "rg1"
    assert [(member["source_binding"]["id"], member["target_binding"]["id"]) for member in grounded["relation_facts"][0]["members"]] == [
        ("a", "c"),
        ("b", "c"),
    ]
