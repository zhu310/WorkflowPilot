from __future__ import annotations

from src.canonical_goal_state import (
    CONSTRAINT_SIGNATURES,
    PREDICATE_SIGNATURES,
    build_canonical_shadow_artifact,
    canonical_json,
    model_facing_predicate_signatures,
    predicate_fact_key,
    semantic_goal_to_canonical_goal_state,
)


def build_goal(goals: list[dict]) -> dict:
    return {"status": "resolved", "semantic_goals_v2": goals, "clarification": {"needed": False, "question": "", "reason": ""}}


def walk(value):
    if isinstance(value, dict):
        for key, child in value.items():
            yield key
            yield from walk(child)
    elif isinstance(value, list):
        for child in value:
            yield from walk(child)
    elif isinstance(value, str):
        yield value


def walk_dicts(value):
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from walk_dicts(child)
    elif isinstance(value, list):
        for child in value:
            yield from walk_dicts(child)


def assert_strict_goal_state_schema(goal_state: dict) -> None:
    assert set(goal_state) == {
        "goal_state_id",
        "revision",
        "references",
        "desired_facts",
        "constraints",
        "open_statements",
    }


def test_route_to_adapter_separates_semantic_reference_from_binding_runtime() -> None:
    goal_state, warnings = semantic_goal_to_canonical_goal_state(
        build_goal(
            [
                {
                    "id": "g1",
                    "goal_category": "relation_goal",
                    "objects": [
                        {"id": "route", "type": "unresolved_reference", "role": "source"},
                        {"id": "target", "type": "existing_reference", "text": "End", "role": "target"},
                    ],
                    "relation_intents": [{"source": "route", "relation": "route_to", "target": "target"}],
                }
            ]
        ),
        instruction="send it to End",
    )

    assert warnings == []
    assert_strict_goal_state_schema(goal_state)
    assert "provenance_refs" not in goal_state
    assert "status" not in goal_state
    references = {ref["ref_id"]: ref for ref in goal_state["references"]}
    edge_refs = [ref for ref in references.values() if ref["entity_type"] == "edge"]
    assert len(edge_refs) == 1
    assert edge_refs[0]["mention"] is None
    assert all("semantic_role" not in ref for ref in references.values())
    assert all("binding" not in ref for ref in references.values())
    fact = goal_state["desired_facts"][0]
    assert fact["predicate"] == "relation.route_to"
    assert fact["operands"]["subject"]["ref_id"] == edge_refs[0]["ref_id"]


def test_adapter_has_no_operation_or_runtime_control_leakage() -> None:
    goal_state, _warnings = semantic_goal_to_canonical_goal_state(
        build_goal(
            [
                {
                    "id": "g1",
                    "goal_category": "object_state_goal",
                    "objects": [{"id": "node", "type": "existing_reference", "text": "Review", "role": "object"}],
                    "object_desired_states": [
                        {"object": "node", "state": "property_equals", "property": "display_label", "value": "Manual Review"}
                    ],
                }
            ]
        ),
        instruction="Rename Review to Manual Review",
    )

    forbidden = {"rename", "delete", "reroute", "insert", "pending", "candidate", "confirmation", "slot"}
    leaked = {str(item).lower() for item in walk(goal_state)} & forbidden
    assert leaked == set()
    assert goal_state["desired_facts"][0]["predicate"] == "property.equals"


def test_blank_property_value_is_not_modeled_as_explicit_empty_without_evidence() -> None:
    goal_state, warnings = semantic_goal_to_canonical_goal_state(
        build_goal(
            [
                {
                    "id": "g1",
                    "objects": [{"id": "edge", "type": "edge_reference", "text": "failure", "role": "object"}],
                    "object_desired_states": [
                        {"object": "edge", "state": "property_equals", "property": "branch_label", "value": ""}
                    ],
                }
            ]
        ),
        instruction="update the branch label",
    )

    assert not goal_state["desired_facts"]
    assert goal_state["open_statements"][0]["statement_type"] == "unverified_empty_property_value"
    assert warnings[0]["kind"] == "ambiguous_empty_semantics"
    assert "explicit_empty" not in canonical_json(goal_state)


def test_structured_explicit_empty_source_is_preserved_as_typed_operand() -> None:
    goal_state, warnings = semantic_goal_to_canonical_goal_state(
        build_goal(
            [
                {
                    "id": "g1",
                    "objects": [{"id": "edge", "type": "edge_reference", "text": "failure", "role": "object"}],
                    "object_desired_states": [
                        {
                            "object": "edge",
                            "state": "property_equals",
                            "property": "branch_label",
                            "value": {"operand_type": "explicit_empty"},
                        }
                    ],
                }
            ]
        ),
        instruction="structured source already committed explicit empty",
    )

    assert warnings == []
    fact = goal_state["desired_facts"][0]
    assert fact["predicate"] == "property.equals"
    assert fact["operands"]["value"] == {"operand_type": "explicit_empty"}


def test_shadow_artifact_keeps_evidence_bundle_outside_goal_state() -> None:
    artifact = build_canonical_shadow_artifact(
        build_goal(
            [
                {
                    "id": "g1",
                    "objects": [{"id": "node", "type": "existing_reference", "text": "审核", "role": "object"}],
                    "object_desired_states": [{"object": "node", "state": "exists"}],
                }
            ]
        ),
        instruction="保留审核",
    )

    assert artifact["ok"] is True
    assert_strict_goal_state_schema(artifact["goal_state"])
    assert artifact["evidence_bundle"]
    assert "provenance" not in artifact["goal_state"]
    assert artifact["goal_state"]["references"][0]["provenance_refs"] == ["p1"]
    evidence = artifact["evidence_bundle"][0]
    span = evidence["evidence_span"]
    assert span["text"][span["start"] : span["end"]] == "保留审核"


def test_unstructured_constraint_becomes_open_statement_not_executable_constraint() -> None:
    goal_state, _warnings = semantic_goal_to_canonical_goal_state(
        build_goal(
            [
                {
                    "id": "g1",
                    "semantic_constraints": [
                        {"type": "preserve", "target": "keep the existing flow readable", "text": "keep the existing flow readable"}
                    ],
                }
            ]
        ),
        instruction="Keep the existing flow readable",
    )

    assert goal_state["constraints"] == []
    assert goal_state["open_statements"][0]["statement_type"] == "unstructured_semantic_constraint"


def test_structured_preserve_state_is_constraint_not_desired_fact() -> None:
    goal_state, _warnings = semantic_goal_to_canonical_goal_state(
        build_goal(
            [
                {
                    "id": "g1",
                    "objects": [{"id": "review", "type": "existing_reference", "text": "Review", "role": "object"}],
                    "object_desired_states": [{"object": "review", "state": "preserved"}],
                }
            ]
        ),
        instruction="Preserve Review",
    )

    assert goal_state["desired_facts"] == []
    assert goal_state["constraints"][0]["constraint_predicate"] == "preserve.entity"


def test_deep_schema_invariants_reject_runtime_and_history_fields() -> None:
    goal_state, _warnings = semantic_goal_to_canonical_goal_state(
        build_goal(
            [
                {
                    "id": "g1",
                    "objects": [{"id": "node", "type": "existing_reference", "text": "Review", "role": "object"}],
                    "object_desired_states": [{"object": "node", "state": "exists"}],
                }
            ]
        ),
        instruction="Keep Review",
    )

    assert_strict_goal_state_schema(goal_state)
    forbidden_fields = {
        "provenance",
        "binding",
        "bindings",
        "business_binding_id",
        "candidate",
        "candidates",
        "candidate_id",
        "pending",
        "confirmation",
        "InteractionNeed",
        "unresolved_slot",
        "runtime_slot",
        "operation",
        "planner_step",
        "executor_step",
        "superseded",
        "retracted",
        "cancelled",
        "history",
        "delta_history",
        "semantic_role",
    }
    for item in walk_dicts(goal_state):
        assert not (set(item) & forbidden_fields)


def test_desired_fact_operands_match_predicate_signatures() -> None:
    goal_state, _warnings = semantic_goal_to_canonical_goal_state(
        build_goal(
            [
                {
                    "id": "g1",
                    "objects": [
                        {"id": "route", "type": "edge_reference", "text": "success", "role": "source"},
                        {"id": "target", "type": "existing_reference", "text": "End", "role": "target"},
                    ],
                    "relation_intents": [{"source": "route", "relation": "route_to", "target": "target"}],
                }
            ]
        ),
        instruction="route success to End",
    )

    for fact in goal_state["desired_facts"]:
        signature = PREDICATE_SIGNATURES[fact["predicate"]]
        assert set(fact["operands"]) == set(signature["role_names"])
        assert all(value is not None for value in fact["operands"].values())
        assert predicate_fact_key(fact) == predicate_fact_key(fact)


def test_registry_reflects_current_semantic_and_lowering_capabilities() -> None:
    assert PREDICATE_SIGNATURES["relation.route_to"]["semantic_expressible"] is True
    assert PREDICATE_SIGNATURES["relation.route_to"]["lowering_supported"] is True
    assert PREDICATE_SIGNATURES["relation.connected_to"]["semantic_expressible"] is True
    assert PREDICATE_SIGNATURES["relation.connected_to"]["model_facing_desired_fact"] is False
    assert PREDICATE_SIGNATURES["relation.connected_to"]["lowering_supported"] is False
    assert PREDICATE_SIGNATURES["relation.after"]["model_facing_desired_fact"] is False
    assert PREDICATE_SIGNATURES["property.equals"]["explicit_empty_policy"] == "requires_structured_empty_intent"
    assert CONSTRAINT_SIGNATURES["preserve.join"]["validation_supported"] == "limited_text_hint_in_ordering_lowering"


def test_registry_entries_are_machine_readable_and_complete() -> None:
    for predicate_id, signature in PREDICATE_SIGNATURES.items():
        assert signature["predicate_id"] == predicate_id
        assert signature["role_names"]
        assert set(signature["role_names"]) == set(signature["role_types"])
        assert isinstance(signature["anchor_roles"], list)
        assert isinstance(signature["objective_roles"], list)
        assert signature["symmetry"] in {"none", "unordered"}
        assert "fact_key" in signature["normalization_policy"]
        assert isinstance(signature["semantic_expressible"], bool)
        assert isinstance(signature["model_facing_desired_fact"], bool)
        assert isinstance(signature["lowering_supported"], bool)

    for constraint_id, signature in CONSTRAINT_SIGNATURES.items():
        assert signature["constraint_signature_id"] == constraint_id
        assert signature["role_names"]
        assert set(signature["role_names"]) == set(signature["role_types"])
        assert set(signature["role_names"]) == set(signature["role_cardinality"])
        assert signature["graph_dependency"]
        assert signature["validator_key"]
        assert isinstance(signature["semantic_expressible"], bool)


def test_adapter_output_is_deterministic() -> None:
    source = build_goal(
        [
            {
                "id": "g1",
                "objects": [{"id": "node", "type": "existing_reference", "text": "Review", "role": "object"}],
                "object_desired_states": [{"object": "node", "state": "exists"}],
            }
        ]
    )
    left = build_canonical_shadow_artifact(source, instruction="Keep Review")
    right = build_canonical_shadow_artifact(source, instruction="Keep Review")

    assert left == right


def test_adapter_normalizes_after_relation_to_before() -> None:
    goal_state, _warnings = semantic_goal_to_canonical_goal_state(
        build_goal(
            [
                {
                    "id": "g1",
                    "objects": [
                        {"id": "a", "type": "existing_reference", "text": "A", "role": "source"},
                        {"id": "b", "type": "existing_reference", "text": "B", "role": "target"},
                    ],
                    "relation_intents": [{"source": "a", "relation": "after", "target": "b"}],
                }
            ]
        ),
        instruction="A after B",
    )

    fact = goal_state["desired_facts"][0]
    refs = {row["ref_id"]: row for row in goal_state["references"]}
    assert fact["predicate"] == "relation.before"
    assert refs[fact["operands"]["subject"]["ref_id"]]["mention"]["text"] == "B"
    assert refs[fact["operands"]["anchor"]["ref_id"]]["mention"]["text"] == "A"


def test_connected_to_stays_available_for_semantic_goal_context_adapter() -> None:
    goal_state, _warnings = semantic_goal_to_canonical_goal_state(
        build_goal(
            [
                {
                    "id": "g1",
                    "objects": [
                        {"id": "a", "type": "existing_reference", "text": "A", "role": "source"},
                        {"id": "b", "type": "existing_reference", "text": "B", "role": "target"},
                    ],
                    "relation_intents": [{"source": "a", "relation": "connected_to", "target": "b"}],
                }
            ]
        ),
        instruction="A connected to B",
    )

    assert goal_state["desired_facts"][0]["predicate"] == "relation.connected_to"
    assert "relation.connected_to" not in model_facing_predicate_signatures()
