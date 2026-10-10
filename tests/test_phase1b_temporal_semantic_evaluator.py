from __future__ import annotations

from copy import deepcopy

from experiments.phase1b_temporal_semantic_evaluator import temporal_semantic_equivalent


def ref(entity_type: str, mention: str | None, cardinality: str = "single") -> dict:
    return {
        "entity_type": entity_type,
        "cardinality": cardinality,
        "mention": None if mention is None else {"text": mention},
    }


def operand(reference: dict) -> dict:
    return {"operand_type": "reference", "reference": deepcopy(reference)}


def state_for_fact(predicate: str, operands: dict, references: list[dict]) -> dict:
    return {
        "references": deepcopy(references),
        "desired_facts": [{"predicate": predicate, "operands": deepcopy(operands)}],
        "constraints": [],
        "open_statements": [],
    }


def property_state(reference: dict, prop: str, value: str) -> dict:
    return state_for_fact(
        "property.equals",
        {
            "subject": operand(reference),
            "property": {"operand_type": "literal", "literal_type": "property_name", "value": prop},
            "value": {"operand_type": "literal", "literal_type": "string", "value": value},
        },
        [reference],
    )


def absent_state(reference: dict) -> dict:
    return state_for_fact("entity.absent", {"subject": operand(reference)}, [reference])


def test_temporal_equivalence_allows_prior_mention_vs_omitted_occurrence_metadata() -> None:
    assert temporal_semantic_equivalent(
        property_state(ref("node", "这个节点"), "display_label", "归档确认"),
        property_state(ref("node", None), "display_label", "归档确认"),
    )


def test_temporal_equivalence_rejects_different_explicit_participants() -> None:
    assert not temporal_semantic_equivalent(absent_state(ref("node", "审核")), absent_state(ref("node", "发布")))


def test_temporal_equivalence_rejects_multi_participant_collapse() -> None:
    edge = ref("edge", None)
    target = ref("node", "人工复核")
    route = state_for_fact(
        "relation.route_to",
        {"subject": operand(edge), "target": operand(target)},
        [edge, target],
    )
    assert not temporal_semantic_equivalent(route, absent_state(edge))


def test_temporal_equivalence_rejects_different_predicates() -> None:
    reference = ref("node", None)
    assert not temporal_semantic_equivalent(absent_state(reference), property_state(reference, "display_label", "归档确认"))


def test_temporal_equivalence_rejects_different_property_or_value() -> None:
    reference = ref("node", None)
    assert not temporal_semantic_equivalent(
        property_state(reference, "display_label", "归档确认"),
        property_state(reference, "branch_label", "归档确认"),
    )
    assert not temporal_semantic_equivalent(
        property_state(reference, "display_label", "归档确认"),
        property_state(reference, "display_label", "retry"),
    )


def test_temporal_equivalence_rejects_different_cardinality() -> None:
    assert not temporal_semantic_equivalent(absent_state(ref("node", None, "single")), absent_state(ref("node", None, "set")))
