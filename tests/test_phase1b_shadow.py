from __future__ import annotations

from copy import deepcopy

import pytest

from src import semantic_shadow
from src.canonical_goal_state import CONSTRAINT_SIGNATURES, PREDICATE_SIGNATURES, predicate_fact_key, model_facing_predicate_signatures
from src.phase1b_shadow import (
    PHASE1B_CONTINUITY_SYSTEM,
    PHASE1B_EXTRACTION_SYSTEM,
    PHASE1B_MODEL_GRAMMAR,
    empty_goal_state,
    evaluate_phase1b_invariants,
    merge_goal_state,
    run_phase1b_shadow,
    semantic_equivalence_key,
    validate_semantic_delta,
)
from src.semantic_contract import (
    CANONICAL_DISPLAY_PROPERTY,
    REFERENCE_CARDINALITIES,
    REFERENCE_ENTITY_TYPES,
    SEMANTIC_DELTA_OPS,
    SEMANTIC_ELEMENT_KINDS,
    STAGE_RESULTS,
    STAGE_TYPES,
    TURN_RELATIONS,
    canonical_property_name,
    stage_result_matches,
)


def response(*, relation="CONTINUE", stage=None, acts=None, ops=None) -> dict:
    proposal = {"references": [], "desired_facts": [], "constraints": [], "open_statements": []}
    for op in ops or []:
        if op.get("op") not in {None, "ADD", "CORRECT"}:
            continue
        kind = op.get("element_kind")
        field = {
            "reference": "references",
            "desired_fact": "desired_facts",
            "constraint": "constraints",
            "open_statement": "open_statements",
        }.get(kind)
        if field:
            proposal[field].append(op.get("value", {}))
    return {
        "turn_relation": {
            "relation": relation,
            "task_context_ref": "task1",
            "evidence_refs": ["p1"],
            "reason_codes": ["model_structured"],
        },
        "stage_continuation": stage,
        "semantic_proposal": proposal,
    }


def ref(ref_id: str, entity_type: str, text: str | None, start: int | None = None, end: int | None = None, cardinality: str = "single") -> dict:
    mention = None if text is None else {"text": text, "span": {"utterance_id": "u1", "start": start or 0, "end": end or len(text)}}
    return {"ref_id": ref_id, "entity_type": entity_type, "cardinality": cardinality, "mention": mention, "provenance_refs": ["p1"]}


def reference_operand(ref_id: str) -> dict:
    return {"operand_type": "reference", "ref_id": ref_id}


def literal(value: str) -> dict:
    return {"operand_type": "literal", "literal_type": "string", "value": value}


def property_name(value: str) -> dict:
    return {"operand_type": "literal", "literal_type": "property_name", "value": value}


def fact(fact_id: str, predicate: str, operands: dict) -> dict:
    return {
        "fact_id": fact_id,
        "predicate": predicate,
        "predicate_signature_id": predicate,
        "operands": operands,
        "provenance_refs": ["p1"],
    }


def test_discourse_continuity_none_is_semantic_payload_noop() -> None:
    extraction_payloads = []

    def model(system: str, payload: dict) -> dict:
        if system == PHASE1B_CONTINUITY_SYSTEM:
            return {"discourse_reference_continuity": {"status": "none", "ref_ids": [], "evidence": ""}}
        if system == PHASE1B_EXTRACTION_SYSTEM:
            extraction_payloads.append(deepcopy(payload))
            return response()
        raise AssertionError(system)

    run = run_phase1b_shadow("改到人工复核", model_runner=model)

    assert run.ok
    assert run.artifact["discourse_continuity_integration"] == {"status": "none", "injected": False, "payload_diff": []}
    assert len(extraction_payloads) == 1
    assert "discourse_reference_continuity" not in extraction_payloads[0]


def test_discourse_continuity_injects_only_effective_prior_reference() -> None:
    before = empty_goal_state()
    before["references"] = [ref("r1", "node", "这个节点")]
    extraction_payloads = []

    def model(system: str, payload: dict) -> dict:
        if system == PHASE1B_CONTINUITY_SYSTEM:
            assert set(payload) == {"raw_utterance", "current_goal_state"}
            assert payload["current_goal_state"] == before
            return {"discourse_reference_continuity": {"status": "continued", "ref_ids": ["r1"], "evidence": "它"}}
        if system == PHASE1B_EXTRACTION_SYSTEM:
            extraction_payloads.append(deepcopy(payload))
            return response()
        raise AssertionError(system)

    run = run_phase1b_shadow("改名为归档确认", current_goal_state=before, model_runner=model)

    assert run.ok
    assert run.artifact["discourse_continuity_integration"] == {
        "status": "continued",
        "injected": True,
        "payload_diff": [{"path": "$.discourse_reference_continuity", "kind": "added"}],
    }
    assert extraction_payloads[0]["discourse_reference_continuity"] == {"status": "continued", "ref_ids": ["r1"], "evidence": "它"}


def test_discourse_continuity_new_without_prior_reference_is_noop() -> None:
    extraction_payloads = []

    def model(system: str, payload: dict) -> dict:
        if system == PHASE1B_CONTINUITY_SYSTEM:
            return {"discourse_reference_continuity": {"status": "new", "ref_ids": [], "evidence": "new participant"}}
        if system == PHASE1B_EXTRACTION_SYSTEM:
            extraction_payloads.append(deepcopy(payload))
            return response()
        raise AssertionError(system)

    run = run_phase1b_shadow("新增一个节点", model_runner=model)

    assert run.ok
    assert run.artifact["discourse_continuity_integration"] == {"status": "new", "injected": False, "payload_diff": []}
    assert "discourse_reference_continuity" not in extraction_payloads[0]


def test_semantic_delta_allows_only_add_correct_retract() -> None:
    errors = validate_semantic_delta(
        {
            "delta_id": "d1",
            "operations": [
                {"op": "BIND", "element_kind": "desired_fact", "target_id": None, "value": {}},
                {"op": "NEW_TASK", "element_kind": "desired_fact", "target_id": None, "value": {}},
            ],
        },
        empty_goal_state(),
    )

    assert {error["message"] for error in errors} == {"invalid SemanticDelta op"}


def test_candidate_stage_continuation_does_not_mutate_goal_state() -> None:
    utterance = "第二个"
    stage = {
        "stage_type": "GROUNDING_CANDIDATE",
        "result": "resolved",
        "evidence_refs": ["p1"],
        "consumed_spans": [{"utterance_id": "u1", "start": 0, "end": len(utterance)}],
        "residual_spans": [],
    }

    run = run_phase1b_shadow(utterance, runtime_stage={"stage_type": "GROUNDING_CANDIDATE"}, model_runner=lambda _system, _payload: response(stage=stage))

    assert run.ok
    assert run.artifact["stage_continuation"]["result"] == "resolved"
    assert run.artifact["before_goal_state"] == run.artifact["after_goal_state"]
    assert run.artifact["semantic_delta"]["operations"] == []


def test_confirmation_authorization_does_not_create_desired_fact() -> None:
    utterance = "确认"
    stage = {
        "stage_type": "PREFLIGHT_CONFIRMATION",
        "result": "authorized",
        "evidence_refs": ["p1"],
        "consumed_spans": [{"utterance_id": "u1", "start": 0, "end": len(utterance)}],
        "residual_spans": [],
    }

    run = run_phase1b_shadow(utterance, runtime_stage={"stage_type": "PREFLIGHT_CONFIRMATION"}, model_runner=lambda _system, _payload: response(stage=stage))

    assert run.ok
    assert run.artifact["after_goal_state"]["desired_facts"] == []


def test_candidate_residual_span_keeps_original_utterance_and_ordered_acts() -> None:
    utterance = "第二个，然后把标签改成 retry"
    consumed_end = len("第二个")
    residual_start = len("第二个，然后")
    ops = [
        {"op": "ADD", "element_kind": "reference", "target_id": None, "value": ref("r1", "edge", None)},
        {"op": "ADD", "element_kind": "desired_fact", "target_id": None, "value": fact("f1", "property.equals", {"subject": reference_operand("r1"), "property": property_name("branch_label"), "value": literal("retry")})},
    ]
    stage = {
        "stage_type": "GROUNDING_CANDIDATE",
        "result": "resolved",
        "evidence_refs": ["p1"],
        "consumed_spans": [{"utterance_id": "u1", "start": 0, "end": consumed_end}],
        "residual_spans": [{"utterance_id": "u1", "start": residual_start, "end": len(utterance)}],
    }
    acts = [
        {"turn_act_id": "a1", "sequence_index": 0, "act_kind": "stage_continuation", "input_span_refs": stage["consumed_spans"], "depends_on": []},
        {"turn_act_id": "a2", "sequence_index": 1, "act_kind": "semantic_update", "input_span_refs": stage["residual_spans"], "depends_on": ["a1"]},
    ]

    def model(system, _payload):
        if "runtime stage continuation" in system:
            return {"stage_continuation": stage}
        return response(ops=ops)

    run = run_phase1b_shadow(utterance, runtime_stage={"stage_type": "GROUNDING_CANDIDATE"}, model_runner=model)

    assert run.ok
    residual = run.artifact["stage_continuation"]["residual_spans"][0]
    assert utterance[residual["start"] : residual["end"]] == "把标签改成 retry"
    assert [act["turn_act_id"] for act in run.artifact["turn_acts"]] == ["a1", "a2"]
    assert run.artifact["after_goal_state"]["desired_facts"][0]["operands"]["value"]["value"] == "retry"


@pytest.mark.xfail(reason="Audit-only behavior from reverted Candidate Pending Ownership experiment.")
def test_candidate_residual_semantic_payload_does_not_carry_pending_runtime_stage() -> None:
    utterance = "second, then rename to retry"
    captured_payloads = []
    stage = {
        "stage_type": "GROUNDING_CANDIDATE",
        "result": "resolved",
        "evidence_refs": ["p1"],
        "consumed_spans": [{"utterance_id": "u1", "start": 0, "end": 6}],
        "residual_spans": [{"utterance_id": "u1", "start": 13, "end": len(utterance)}],
    }
    ops = [
        {"op": "ADD", "element_kind": "reference", "target_id": None, "value": ref("r1", "edge", None)},
        {"op": "ADD", "element_kind": "desired_fact", "target_id": None, "value": fact("f1", "property.equals", {"subject": reference_operand("r1"), "property": property_name("display_label"), "value": literal("retry")})},
    ]

    def model(system, payload):
        if "runtime stage continuation" in system:
            return {"stage_continuation": stage}
        if "semantic proposal" in system:
            captured_payloads.append(deepcopy(payload))
            return response(ops=ops)
        return {"discourse_reference_continuity": {"status": "none", "ref_ids": [], "evidence": ""}}

    run = run_phase1b_shadow(utterance, runtime_stage={"stage_type": "GROUNDING_CANDIDATE", "candidates": [{"id": "A"}]}, model_runner=model)

    assert run.ok
    assert captured_payloads[0]["runtime_stage"] == {}
    assert captured_payloads[0]["stage_continuation"]["result"] == "resolved"
    assert captured_payloads[0]["residual_spans"] == stage["residual_spans"]


@pytest.mark.xfail(reason="Audit-only behavior from reverted Candidate Pending Ownership experiment.")
def test_candidate_no_effective_continuation_is_semantic_noop_payload() -> None:
    utterance = "delete publish node"
    captured_payloads = []
    stage = {
        "stage_type": "GROUNDING_CANDIDATE",
        "result": "unresolved",
        "evidence_refs": ["p1"],
        "consumed_spans": [],
        "residual_spans": [{"utterance_id": "u1", "start": 0, "end": len(utterance)}],
    }
    ops = [
        {"op": "ADD", "element_kind": "reference", "target_id": None, "value": ref("r1", "node", "publish node", 7, len(utterance))},
        {"op": "ADD", "element_kind": "desired_fact", "target_id": None, "value": fact("f1", "entity.absent", {"subject": reference_operand("r1")})},
    ]

    def model(system, payload):
        if "runtime stage continuation" in system:
            return {"stage_continuation": stage}
        if "semantic proposal" in system:
            captured_payloads.append(deepcopy(payload))
            return response(relation="NEW_TASK", ops=ops)
        return {"discourse_reference_continuity": {"status": "none", "ref_ids": [], "evidence": ""}}

    run = run_phase1b_shadow(utterance, runtime_stage={"stage_type": "GROUNDING_CANDIDATE", "candidates": [{"id": "A"}]}, model_runner=model)

    assert run.ok
    assert run.artifact["stage_continuation"]["result"] == "unresolved"
    assert run.artifact["stage_runtime_ownership"]["semantic_noop"] is True
    assert captured_payloads[0]["runtime_stage"] == {}
    assert captured_payloads[0]["stage_continuation"] is None
    assert captured_payloads[0]["residual_spans"] == [{"utterance_id": "u1", "start": 0, "end": len(utterance)}]
    assert run.artifact["turn_relation"]["relation"] == "NEW_TASK"


@pytest.mark.xfail(reason="Audit-only behavior from reverted Candidate Pending Ownership experiment.")
def test_candidate_cancel_with_residual_uses_residual_as_normal_semantic_turn() -> None:
    utterance = "cancel that, delete publish node"
    captured_payloads = []
    stage = {
        "stage_type": "GROUNDING_CANDIDATE",
        "result": "cancelled",
        "evidence_refs": ["p1"],
        "consumed_spans": [{"utterance_id": "u1", "start": 0, "end": 11}],
        "residual_spans": [{"utterance_id": "u1", "start": 13, "end": len(utterance)}],
    }
    ops = [
        {"op": "ADD", "element_kind": "reference", "target_id": None, "value": ref("r1", "node", "publish node", 20, len(utterance))},
        {"op": "ADD", "element_kind": "desired_fact", "target_id": None, "value": fact("f1", "entity.absent", {"subject": reference_operand("r1")})},
    ]

    def model(system, payload):
        if "runtime stage continuation" in system:
            return {"stage_continuation": stage}
        if "semantic proposal" in system:
            captured_payloads.append(deepcopy(payload))
            return response(relation="NEW_TASK", ops=ops)
        return {"discourse_reference_continuity": {"status": "none", "ref_ids": [], "evidence": ""}}

    run = run_phase1b_shadow(utterance, runtime_stage={"stage_type": "GROUNDING_CANDIDATE", "candidates": [{"id": "A"}]}, model_runner=model)

    assert run.ok
    assert captured_payloads[0]["runtime_stage"] == {}
    assert captured_payloads[0]["stage_continuation"]["result"] == "cancelled"
    assert captured_payloads[0]["residual_spans"] == stage["residual_spans"]
    assert run.artifact["turn_relation"]["relation"] == "NEW_TASK"


def test_implicit_reference_route_to_uses_null_mention_not_slot() -> None:
    utterance = "改到人工复核"
    ops = [
        {"op": "ADD", "element_kind": "reference", "target_id": None, "value": ref("r1", "edge", None)},
        {"op": "ADD", "element_kind": "reference", "target_id": None, "value": ref("r2", "node", "人工复核", 2, len(utterance))},
        {"op": "ADD", "element_kind": "desired_fact", "target_id": None, "value": fact("f1", "relation.route_to", {"subject": reference_operand("r1"), "target": reference_operand("r2")})},
    ]

    run = run_phase1b_shadow(utterance, model_runner=lambda _system, _payload: response(ops=ops))

    assert run.ok
    edge_ref = next(item for item in run.artifact["after_goal_state"]["references"] if item["entity_type"] == "edge")
    assert edge_ref["mention"] is None
    assert "unresolved_reference" not in str(run.artifact["after_goal_state"])
    assert "subject_edge" not in str(run.artifact["after_goal_state"])


def test_explicit_empty_can_be_extracted_from_structured_raw_turn_evidence() -> None:
    utterance = "把这条边的标签清空"
    ops = [
        {"op": "ADD", "element_kind": "reference", "target_id": None, "value": ref("r1", "edge", "这条边", 1, 4)},
        {"op": "ADD", "element_kind": "desired_fact", "target_id": None, "value": fact("f1", "property.equals", {"subject": reference_operand("r1"), "property": property_name("branch_label"), "value": {"operand_type": "explicit_empty"}})},
    ]

    run = run_phase1b_shadow(utterance, model_runner=lambda _system, _payload: response(ops=ops))

    assert run.ok
    assert run.artifact["after_goal_state"]["desired_facts"][0]["operands"]["value"] == {"operand_type": "explicit_empty"}


def test_no_explicit_empty_without_structured_evidence() -> None:
    utterance = "把这条边的标签改一下"
    ops = [
        {"op": "ADD", "element_kind": "reference", "target_id": None, "value": ref("r1", "edge", "这条边", 1, 4)},
        {"op": "ADD", "element_kind": "desired_fact", "target_id": None, "value": fact("f1", "property.equals", {"subject": reference_operand("r1"), "property": property_name("branch_label"), "value": None})},
    ]

    run = run_phase1b_shadow(utterance, model_runner=lambda _system, _payload: response(ops=ops))

    assert not run.ok
    assert run.error == "semantic_schema_validation_failure"


def test_correction_replaces_old_fact_even_when_fact_key_changes() -> None:
    before = empty_goal_state()
    before["references"] = [ref("r1", "node", "A", 0, 1), ref("r2", "node", "B", 2, 3)]
    before["desired_facts"] = [fact("f1", "relation.before", {"subject": reference_operand("r1"), "anchor": reference_operand("r2")})]
    delta = {
        "delta_id": "d1",
        "operations": [
            {"op": "CORRECT", "element_kind": "desired_fact", "target_id": "f1", "value": fact("f2", "relation.before", {"subject": reference_operand("r2"), "anchor": reference_operand("r1")})}
        ],
    }

    merged = merge_goal_state(before, delta, [{"provenance_id": "p1"}])

    assert merged["ok"]
    assert len(merged["goal_state"]["desired_facts"]) == 1
    operands = merged["goal_state"]["desired_facts"][0]["operands"]
    assert operands["subject"]["ref_id"] == "r2"
    assert operands["anchor"]["ref_id"] == "r1"


def test_delta_planner_corrects_same_fact_key_target() -> None:
    before = empty_goal_state()
    before["references"] = [ref("r1", "edge", None), ref("r2", "node", "Audit", 0, 5)]
    before["desired_facts"] = [fact("f1", "relation.route_to", {"subject": reference_operand("r1"), "target": reference_operand("r2")})]
    ops = [
        {"op": "ADD", "element_kind": "reference", "target_id": None, "value": ref("r3", "node", "End", 0, 3)},
        {"op": "ADD", "element_kind": "desired_fact", "target_id": None, "value": fact("f2", "relation.route_to", {"subject": reference_operand("r1"), "target": reference_operand("r3")})},
    ]

    run = run_phase1b_shadow("not that, route to End", current_goal_state=before, model_runner=lambda _system, _payload: response(relation="CORRECT", ops=ops))

    assert run.ok
    assert run.artifact["semantic_delta"]["operations"][1]["op"] == "CORRECT"
    assert run.artifact["semantic_delta"]["operations"][1]["target_id"] == "f1"
    assert len(run.artifact["after_goal_state"]["desired_facts"]) == 1


def test_delta_planner_corrects_cross_factkey_target_by_shared_participants() -> None:
    before = empty_goal_state()
    before["references"] = [ref("r1", "node", "A", 0, 1), ref("r2", "node", "B", 2, 3)]
    before["desired_facts"] = [fact("f1", "relation.before", {"subject": reference_operand("r1"), "anchor": reference_operand("r2")})]
    ops = [
        {"op": "ADD", "element_kind": "desired_fact", "target_id": None, "value": fact("f2", "relation.before", {"subject": reference_operand("r2"), "anchor": reference_operand("r1")})},
    ]

    run = run_phase1b_shadow("no, B before A", current_goal_state=before, model_runner=lambda _system, _payload: response(relation="CORRECT", ops=ops))

    assert run.ok
    assert run.artifact["semantic_delta"]["operations"][0]["op"] == "CORRECT"
    assert run.artifact["semantic_delta"]["operations"][0]["target_id"] == "f1"
    operands = run.artifact["after_goal_state"]["desired_facts"][0]["operands"]
    assert operands["subject"]["ref_id"] == "r2"
    assert operands["anchor"]["ref_id"] == "r1"


def test_delta_planner_does_not_guess_ambiguous_correction_target() -> None:
    before = empty_goal_state()
    before["references"] = [ref("r1", "node", "A", 0, 1), ref("r2", "node", "B", 2, 3)]
    before["desired_facts"] = [
        fact("f1", "relation.before", {"subject": reference_operand("r1"), "anchor": reference_operand("r2")}),
        fact("f2", "relation.before", {"subject": reference_operand("r1"), "anchor": reference_operand("r2")}),
    ]
    ops = [
        {"op": "ADD", "element_kind": "desired_fact", "target_id": None, "value": fact("f3", "relation.before", {"subject": reference_operand("r1"), "anchor": reference_operand("r2")})},
    ]

    run = run_phase1b_shadow("no, fix the order", current_goal_state=before, model_runner=lambda _system, _payload: response(relation="CORRECT", ops=ops))

    assert not run.ok
    assert run.artifact["after_goal_state"] == before
    assert run.artifact["delta_planner"]["errors"][0]["category"] == "semantic_delta_planning_failure"


def test_new_task_is_turn_relation_not_semantic_delta_op() -> None:
    run = run_phase1b_shadow("另起一个任务", model_runner=lambda _system, _payload: response(relation="NEW_TASK", ops=[]))

    assert run.ok
    assert run.artifact["turn_relation"]["relation"] == "NEW_TASK"
    assert run.artifact["semantic_delta"]["operations"] == []


def test_new_task_applies_proposal_to_fresh_goal_state_without_polluting_old_state() -> None:
    before = empty_goal_state()
    before["references"] = [ref("r1", "node", "Review", 0, 6)]
    before["desired_facts"] = [fact("f1", "entity.exists", {"subject": reference_operand("r1")})]
    before_snapshot = deepcopy(before)
    ops = [
        {"op": "ADD", "element_kind": "reference", "target_id": None, "value": ref("r2", "node", "Archive", 0, 7)},
        {"op": "ADD", "element_kind": "desired_fact", "target_id": None, "value": fact("f2", "entity.exists", {"subject": reference_operand("r2")})},
    ]

    run = run_phase1b_shadow("start another task", current_goal_state=before, model_runner=lambda _system, _payload: response(relation="NEW_TASK", ops=ops))

    assert run.ok
    assert before == before_snapshot
    assert run.artifact["before_goal_state"] == before_snapshot
    assert run.artifact["after_goal_state"]["references"][0]["ref_id"] == "r2"
    assert run.artifact["after_goal_state"]["desired_facts"][0]["predicate"] == "entity.exists"
    assert "Review" not in str(run.artifact["after_goal_state"])
    assert run.artifact["task_lifecycle"]["relation"] == "NEW_TASK"
    assert run.artifact["task_lifecycle"]["preserved_goal_state_key"] == semantic_equivalence_key(before_snapshot)


def test_new_task_applies_open_statement_to_fresh_goal_state() -> None:
    before = empty_goal_state()
    before["references"] = [ref("r1", "node", "Review", 0, 6)]
    ops = [
        {"op": "ADD", "element_kind": "open_statement", "target_id": None, "value": {"statement_id": "s1", "statement_type": "open", "content": {"text": "make it better"}, "provenance_refs": ["p1"]}},
    ]

    run = run_phase1b_shadow("start another vague task", current_goal_state=before, model_runner=lambda _system, _payload: response(relation="NEW_TASK", ops=ops))

    assert run.ok
    assert run.artifact["after_goal_state"]["references"] == []
    assert run.artifact["after_goal_state"]["open_statements"][0]["content"] == {"text": "make it better"}


def test_new_task_applies_constraint_to_fresh_goal_state() -> None:
    before = empty_goal_state()
    before["references"] = [ref("r_old", "node", "Review", 0, 6)]
    ops = [
        {"op": "ADD", "element_kind": "reference", "target_id": None, "value": ref("r1", "node", "Archive", 0, 7)},
        {"op": "ADD", "element_kind": "constraint", "target_id": None, "value": {"constraint_id": "c1", "constraint_predicate": "preserve.entity", "constraint_signature_id": "preserve.entity", "operands": {"target": reference_operand("r1")}, "provenance_refs": ["p1"]}},
    ]

    run = run_phase1b_shadow("start constrained task", current_goal_state=before, model_runner=lambda _system, _payload: response(relation="NEW_TASK", ops=ops))

    assert run.ok
    assert run.artifact["after_goal_state"]["references"][0]["ref_id"] == "r1"
    assert run.artifact["after_goal_state"]["constraints"][0]["constraint_predicate"] == "preserve.entity"
    assert "r_old" not in str(run.artifact["after_goal_state"])


def test_cancel_does_not_create_desired_fact() -> None:
    before = empty_goal_state()
    ops = [
        {"op": "ADD", "element_kind": "reference", "target_id": None, "value": ref("r1", "node", "Archive", 0, 7)},
        {"op": "ADD", "element_kind": "desired_fact", "target_id": None, "value": fact("f1", "entity.exists", {"subject": reference_operand("r1")})},
    ]

    run = run_phase1b_shadow("cancel that", current_goal_state=before, model_runner=lambda _system, _payload: response(relation="CANCEL", ops=ops))

    assert run.ok
    assert run.artifact["after_goal_state"] == before
    assert run.artifact["semantic_delta"]["operations"] == []


def test_selection_and_graph_facts_do_not_create_intent_by_themselves() -> None:
    run = run_phase1b_shadow(
        "",
        selection={"node_ids": ["n1"], "edge_ids": ["e1"]},
        graph_context={"nodes": [{"id": "n1", "label": "Review"}]},
        model_runner=lambda _system, _payload: response(ops=[]),
    )

    assert run.ok
    assert run.artifact["after_goal_state"] == empty_goal_state()


def test_invalid_stage_type_rejects_semantic_clarification_parser() -> None:
    stage = {"stage_type": "SEMANTIC_CLARIFICATION", "result": "resolved", "consumed_spans": [], "residual_spans": []}

    run = run_phase1b_shadow("补充一下", runtime_stage={"stage_type": "SEMANTIC_CLARIFICATION"}, model_runner=lambda _system, _payload: response(stage=stage))

    assert not run.ok
    assert run.artifact["validation"]["errors"][0]["category"] == "stage_continuation_failure"


def test_phase1b_invariants_reject_graph_binding_and_runtime_fields() -> None:
    artifact = {"after_goal_state": empty_goal_state()}
    artifact["after_goal_state"]["references"] = [{"ref_id": "r1", "entity_type": "node", "cardinality": "single", "mention": None, "node_id": "n1"}]
    artifact["semantic_delta"] = {"operations": [{"op": "ADD"}]}

    result = evaluate_phase1b_invariants(artifact)

    assert not result["ok"]
    assert result["errors"][0]["category"] == "invariant_failure"


def test_semantic_equivalence_ignores_transient_ids() -> None:
    left = empty_goal_state()
    left["references"] = [ref("r1", "node", "Review", 0, 6)]
    left["desired_facts"] = [fact("f1", "entity.exists", {"subject": reference_operand("r1")})]
    right = empty_goal_state()
    right["references"] = [ref("x9", "node", "Review", 0, 6)]
    right["desired_facts"] = [fact("z8", "entity.exists", {"subject": reference_operand("x9")})]

    assert semantic_equivalence_key(left) == semantic_equivalence_key(right)


def test_shadow_internal_error_is_fail_open_artifact() -> None:
    before = empty_goal_state()

    run = run_phase1b_shadow("任意输入", current_goal_state=before, model_runner=lambda _system, _payload: (_ for _ in ()).throw(RuntimeError("boom")))

    assert not run.ok
    assert run.error == "shadow_internal_error"
    assert run.artifact["after_goal_state"] == before


def test_production_semantic_shadow_does_not_depend_on_phase1b(monkeypatch) -> None:
    from types import SimpleNamespace

    spec = {
        "status": "resolved",
        "semantic_goals": [
                {
                    "id": "g1",
                    "goal_category": "object_state_goal",
                    "objects": [{"id": "target", "type": "existing_reference", "text": "Review"}],
                "object_desired_states": [{"object": "target", "state": "property_equals", "property": "display_label", "value": "Manual Review"}],
            }
        ],
        "clarification": {"needed": False, "question": "", "reason": ""},
    }
    monkeypatch.setattr(
        semantic_shadow,
        "_complete_json",
        lambda _system, _payload: SimpleNamespace(raw_response="", parsed_json=spec, parse_error=None, normalized_from_fence=False),
    )
    monkeypatch.setenv("SEMANTIC_PHASE1B_SHADOW", "1")
    graph = {
        "nodes": [
            {"id": "start", "type": "start", "label": "Start", "x": 0, "y": 0, "width": 160, "height": 80},
            {"id": "review", "type": "task", "label": "Review", "x": 220, "y": 0, "width": 160, "height": 80},
            {"id": "end", "type": "end", "label": "End", "x": 440, "y": 0, "width": 160, "height": 80},
        ],
        "edges": [
            {"id": "e_start_review", "source": "start", "target": "review", "label": ""},
            {"id": "e_review_end", "source": "review", "target": "end", "label": ""},
        ],
    }

    trace = semantic_shadow.run_semantic_shadow(graph, "Rename Review to Manual Review")

    assert trace["stage"] == "success"
    assert "phase1b_shadow" not in trace


def test_controlled_natural_language_benchmark_uses_model_interface() -> None:
    fixtures = {
        "改到人工复核": response(
            ops=[
                {"op": "ADD", "element_kind": "reference", "target_id": None, "value": ref("r1", "edge", None)},
                {"op": "ADD", "element_kind": "reference", "target_id": None, "value": ref("r2", "node", "人工复核", 2, 6)},
                {"op": "ADD", "element_kind": "desired_fact", "target_id": None, "value": fact("f1", "relation.route_to", {"subject": reference_operand("r1"), "target": reference_operand("r2")})},
            ]
        ),
        "把这条边的标签清空": response(
            ops=[
                {"op": "ADD", "element_kind": "reference", "target_id": None, "value": ref("r1", "edge", "这条边", 1, 4)},
                {"op": "ADD", "element_kind": "desired_fact", "target_id": None, "value": fact("f1", "property.equals", {"subject": reference_operand("r1"), "property": property_name("branch_label"), "value": {"operand_type": "explicit_empty"}})},
            ]
        ),
    }

    def model(_system, payload):
        return fixtures[payload["raw_utterance"]]

    keys = [run_phase1b_shadow(text, model_runner=model).artifact["semantic_equivalence_key"] for text in fixtures]

    assert len(keys) == 2
    assert keys[0]["desired_facts"][0]["predicate"] == "relation.route_to"
    assert keys[1]["desired_facts"][0]["operands"]["value"] == {"operand_type": "explicit_empty"}


def test_repeated_stability_semantic_equivalence() -> None:
    payload = response(
        ops=[
            {"op": "ADD", "element_kind": "reference", "target_id": None, "value": ref("r1", "edge", None)},
            {"op": "ADD", "element_kind": "reference", "target_id": None, "value": ref("r2", "node", "人工复核", 2, 6)},
            {"op": "ADD", "element_kind": "desired_fact", "target_id": None, "value": fact("f1", "relation.route_to", {"subject": reference_operand("r1"), "target": reference_operand("r2")})},
        ]
    )

    runs = [run_phase1b_shadow("改到人工复核", model_runner=lambda _system, _payload: payload) for _ in range(3)]

    assert all(run.ok for run in runs)
    assert len({str(run.artifact["semantic_equivalence_key"]) for run in runs}) == 1


def test_phase1b_model_grammar_covers_predicate_registry() -> None:
    for predicate_id, signature in model_facing_predicate_signatures().items():
        assert predicate_id in PHASE1B_MODEL_GRAMMAR
        for role in signature["role_names"]:
            assert role in PHASE1B_MODEL_GRAMMAR


def test_phase1b_model_grammar_hides_inverse_and_context_predicates() -> None:
    assert "relation.before" in PHASE1B_MODEL_GRAMMAR
    assert "relation.after" not in PHASE1B_MODEL_GRAMMAR
    assert "relation.route_to" in PHASE1B_MODEL_GRAMMAR
    assert "relation.connected_to" not in PHASE1B_MODEL_GRAMMAR


def test_phase1b_model_grammar_covers_reference_enums() -> None:
    for entity_type in REFERENCE_ENTITY_TYPES:
        assert entity_type in PHASE1B_MODEL_GRAMMAR
    for cardinality in REFERENCE_CARDINALITIES:
        assert cardinality in PHASE1B_MODEL_GRAMMAR


def test_phase1b_model_grammar_covers_constraint_registry() -> None:
    for signature_id, signature in CONSTRAINT_SIGNATURES.items():
        assert signature_id in PHASE1B_MODEL_GRAMMAR
        for role in signature["role_names"]:
            assert role in PHASE1B_MODEL_GRAMMAR


def test_phase1b_model_grammar_covers_turn_stage_and_element_enums() -> None:
    for item in SEMANTIC_ELEMENT_KINDS + TURN_RELATIONS + STAGE_TYPES + STAGE_RESULTS:
        assert item in PHASE1B_MODEL_GRAMMAR or item in PHASE1B_EXTRACTION_SYSTEM


def test_phase1b_model_grammar_documents_explicit_empty_position() -> None:
    assert '"operand_type": "explicit_empty"' in PHASE1B_MODEL_GRAMMAR
    assert "property.equals value operand" in PHASE1B_MODEL_GRAMMAR


def test_phase1b_model_output_contract_no_longer_owns_delta_or_target() -> None:
    assert "semantic_delta" not in PHASE1B_EXTRACTION_SYSTEM
    assert "target_id" not in PHASE1B_EXTRACTION_SYSTEM
    assert "ADD|CORRECT|RETRACT" not in PHASE1B_EXTRACTION_SYSTEM
    assert "semantic_delta" not in PHASE1B_MODEL_GRAMMAR
    assert "target_id" not in PHASE1B_MODEL_GRAMMAR


def test_phase1b_benchmark_stage_oracle_accepts_frozen_pending_equivalence() -> None:
    assert stage_result_matches("unresolved", "remain_pending")
    assert stage_result_matches("unresolved", "unresolved")
    assert not stage_result_matches("resolved", "remain_pending")


def test_canonical_property_identity_matches_lowering_display_aliases() -> None:
    assert canonical_property_name("node", "label") == CANONICAL_DISPLAY_PROPERTY
    assert canonical_property_name("node", "name") == CANONICAL_DISPLAY_PROPERTY
    assert canonical_property_name("node", "display_label") == CANONICAL_DISPLAY_PROPERTY
    assert canonical_property_name("edge", "label") == CANONICAL_DISPLAY_PROPERTY
    assert canonical_property_name("edge", "branch_label") == CANONICAL_DISPLAY_PROPERTY
    assert canonical_property_name("edge", "display_label") == CANONICAL_DISPLAY_PROPERTY
    assert canonical_property_name("edge", "condition.expression") == "condition.expression"


def test_property_aliases_normalize_in_goal_state_and_semantic_key() -> None:
    left = empty_goal_state()
    left["references"] = [ref("r1", "edge", None)]
    right = empty_goal_state()
    right["references"] = [ref("r9", "edge", None)]
    left_delta = {
        "delta_id": "d1",
        "operations": [
            {"op": "ADD", "element_kind": "desired_fact", "target_id": None, "value": fact("f1", "property.equals", {"subject": reference_operand("r1"), "property": property_name("branch_label"), "value": literal("retry")})}
        ],
    }
    right_delta = {
        "delta_id": "d1",
        "operations": [
            {"op": "ADD", "element_kind": "desired_fact", "target_id": None, "value": fact("f2", "property.equals", {"subject": reference_operand("r9"), "property": property_name("display_label"), "value": literal("retry")})}
        ],
    }

    merged_left = merge_goal_state(left, left_delta, [{"provenance_id": "p1"}])["goal_state"]
    merged_right = merge_goal_state(right, right_delta, [{"provenance_id": "p1"}])["goal_state"]

    assert merged_left["desired_facts"][0]["operands"]["property"]["value"] == CANONICAL_DISPLAY_PROPERTY
    assert semantic_equivalence_key(merged_left) == semantic_equivalence_key(merged_right)


def test_after_normalizes_to_before_in_goal_state_fact_key_and_semantic_key() -> None:
    before_state = empty_goal_state()
    before_state["references"] = [ref("a", "node", "A"), ref("b", "node", "B")]
    after_fact = fact("f1", "relation.after", {"subject": reference_operand("a"), "anchor": reference_operand("b")})
    before_fact = fact("f2", "relation.before", {"subject": reference_operand("b"), "anchor": reference_operand("a")})
    after_delta = {"delta_id": "d1", "operations": [{"op": "ADD", "element_kind": "desired_fact", "target_id": None, "value": after_fact}]}
    before_delta = {"delta_id": "d1", "operations": [{"op": "ADD", "element_kind": "desired_fact", "target_id": None, "value": before_fact}]}

    merged_after = merge_goal_state(before_state, after_delta, [{"provenance_id": "p1"}])["goal_state"]
    merged_before = merge_goal_state(before_state, before_delta, [{"provenance_id": "p1"}])["goal_state"]

    canonical_fact = merged_after["desired_facts"][0]
    assert canonical_fact["predicate"] == "relation.before"
    assert canonical_fact["operands"] == {"subject": reference_operand("b"), "anchor": reference_operand("a")}
    assert predicate_fact_key(after_fact) == predicate_fact_key(before_fact)
    assert semantic_equivalence_key(merged_after) == semantic_equivalence_key(merged_before)


def test_connected_to_is_not_rewritten_to_route_to() -> None:
    before = empty_goal_state()
    before["references"] = [ref("a", "node", "A"), ref("b", "node", "B")]
    delta = {
        "delta_id": "d1",
        "operations": [
            {
                "op": "ADD",
                "element_kind": "desired_fact",
                "target_id": None,
                "value": fact("f1", "relation.connected_to", {"source": reference_operand("a"), "target": reference_operand("b")}),
            }
        ],
    }

    merged = merge_goal_state(before, delta, [{"provenance_id": "p1"}])

    assert merged["ok"]
    assert merged["goal_state"]["desired_facts"][0]["predicate"] == "relation.connected_to"


def test_open_statement_preserves_explicit_semantic_reference_related_refs() -> None:
    utterance = "把这个节点改一下"
    payload = response(
        ops=[
            {"op": "ADD", "element_kind": "reference", "target_id": None, "value": ref("r_node", "node", "这个节点", 1, 5)},
            {
                "op": "ADD",
                "element_kind": "open_statement",
                "target_id": None,
                "value": {"statement_id": "s1", "statement_type": "open", "content": None, "related_refs": ["r_node"], "provenance_refs": ["p1"]},
            },
        ]
    )

    run = run_phase1b_shadow(
        utterance,
        selection={"node_ids": [], "edge_ids": ["e_decision_report"]},
        model_runner=lambda _system, _payload: payload,
    )

    assert run.ok
    refs = {row["ref_id"]: row for row in run.artifact["after_goal_state"]["references"]}
    assert refs["r_node"]["entity_type"] == "node"
    assert refs["r_node"]["mention"]["text"] == "这个节点"
    assert refs["r_node"]["mention"]["span"] == {"utterance_id": "u1", "start": 1, "end": 5}
    assert run.artifact["after_goal_state"]["open_statements"][0]["related_refs"] == ["r_node"]
    assert "e_decision_report" not in str(run.artifact["after_goal_state"]["open_statements"][0]["related_refs"])


def test_temporal_refinement_prefers_prior_open_statement_node_reference_over_edge_selection() -> None:
    first = response(
        ops=[
            {"op": "ADD", "element_kind": "reference", "target_id": None, "value": ref("r_node", "node", "这个节点", 1, 5)},
            {
                "op": "ADD",
                "element_kind": "open_statement",
                "target_id": None,
                "value": {"statement_id": "s1", "statement_type": "open", "content": None, "related_refs": ["r_node"], "provenance_refs": ["p1"]},
            },
        ]
    )

    def model(_system, payload):
        if payload["raw_utterance"] == "把这个节点改一下":
            return first
        refs = payload["current_goal_state"]["references"]
        assert refs[0]["entity_type"] == "node"
        return response(
            ops=[
                {
                    "op": "ADD",
                    "element_kind": "desired_fact",
                    "target_id": None,
                    "value": fact(
                        "f1",
                        "property.equals",
                        {"subject": reference_operand(refs[0]["ref_id"]), "property": property_name("display_label"), "value": literal("归档确认")},
                    ),
                }
            ]
        )

    first_run = run_phase1b_shadow(
        "把这个节点改一下",
        selection={"node_ids": [], "edge_ids": ["e_decision_report"]},
        model_runner=model,
    )
    second_run = run_phase1b_shadow(
        "改名为归档确认",
        current_goal_state=first_run.artifact["after_goal_state"],
        selection={"node_ids": [], "edge_ids": ["e_decision_report"]},
        model_runner=model,
    )

    assert second_run.ok
    state = second_run.artifact["after_goal_state"]
    refs = {row["ref_id"]: row for row in state["references"]}
    fact_row = state["desired_facts"][0]
    subject_ref = fact_row["operands"]["subject"]["ref_id"]
    assert refs[subject_ref]["entity_type"] == "node"
    assert fact_row["operands"]["property"]["value"] == "display_label"
    assert fact_row["operands"]["value"]["value"] == "归档确认"
    assert state["open_statements"] == []


def test_temporal_refinement_semantic_truth_matches_single_turn_node_rename() -> None:
    first_payload = response(
        ops=[
            {"op": "ADD", "element_kind": "reference", "target_id": None, "value": ref("r_node", "node", "这个节点", 1, 5)},
            {
                "op": "ADD",
                "element_kind": "open_statement",
                "target_id": None,
                "value": {"statement_id": "s1", "statement_type": "open", "content": None, "related_refs": ["r_node"], "provenance_refs": ["p1"]},
            },
        ]
    )
    refine_payload = response(
        ops=[
            {
                "op": "ADD",
                "element_kind": "desired_fact",
                "target_id": None,
                "value": fact(
                    "f1",
                    "property.equals",
                    {"subject": reference_operand("r_node"), "property": property_name("display_label"), "value": literal("归档确认")},
                ),
            }
        ]
    )
    single_payload = response(
        relation="NEW_TASK",
        ops=[
            {"op": "ADD", "element_kind": "reference", "target_id": None, "value": ref("r_node", "node", "这个节点", 1, 5)},
            {
                "op": "ADD",
                "element_kind": "desired_fact",
                "target_id": None,
                "value": fact(
                    "f1",
                    "property.equals",
                    {"subject": reference_operand("r_node"), "property": property_name("display_label"), "value": literal("归档确认")},
                ),
            },
        ],
    )

    def model(_system, payload):
        if payload["raw_utterance"] == "把这个节点改一下":
            return first_payload
        if payload["raw_utterance"] == "改名为归档确认":
            return refine_payload
        return single_payload

    first_run = run_phase1b_shadow("把这个节点改一下", model_runner=model)
    multi = run_phase1b_shadow("改名为归档确认", current_goal_state=first_run.artifact["after_goal_state"], model_runner=model)
    single = run_phase1b_shadow("把这个节点改名为归档确认", model_runner=model)

    assert multi.ok
    assert single.ok
    assert semantic_equivalence_key(multi.artifact["after_goal_state"]) == semantic_equivalence_key(single.artifact["after_goal_state"])


def test_implicit_route_semantic_truth_can_use_ungrounded_edge_reference() -> None:
    payload = response(
        ops=[
            {"op": "ADD", "element_kind": "reference", "target_id": None, "value": ref("r_edge", "edge", None)},
            {"op": "ADD", "element_kind": "reference", "target_id": None, "value": ref("r_human", "node", "人工复核", 2, 6)},
            {
                "op": "ADD",
                "element_kind": "desired_fact",
                "target_id": None,
                "value": fact("f1", "relation.route_to", {"subject": reference_operand("r_edge"), "target": reference_operand("r_human")}),
            },
        ]
    )

    run = run_phase1b_shadow(
        "改到人工复核",
        selection={"node_ids": [], "edge_ids": ["e_decision_report"]},
        model_runner=lambda _system, _payload: payload,
    )

    assert run.ok
    state = run.artifact["after_goal_state"]
    assert state["desired_facts"][0]["predicate"] == "relation.route_to"
    refs = {row["ref_id"]: row for row in state["references"]}
    assert refs["r_edge"]["entity_type"] == "edge"
    assert refs["r_human"]["entity_type"] == "node"


def test_selection_evidence_does_not_override_node_semantic_intent() -> None:
    payload = response(
        ops=[
            {"op": "ADD", "element_kind": "reference", "target_id": None, "value": ref("r_node", "node", "这个节点", 1, 5)},
            {
                "op": "ADD",
                "element_kind": "desired_fact",
                "target_id": None,
                "value": fact(
                    "f1",
                    "property.equals",
                    {"subject": reference_operand("r_node"), "property": property_name("display_label"), "value": literal("归档确认")},
                ),
            },
        ]
    )

    run = run_phase1b_shadow(
        "把这个节点改名为归档确认",
        selection={"node_ids": [], "edge_ids": ["e_decision_report"]},
        model_runner=lambda _system, _payload: payload,
    )

    assert run.ok
    refs = {row["ref_id"]: row for row in run.artifact["after_goal_state"]["references"]}
    subject_ref = run.artifact["after_goal_state"]["desired_facts"][0]["operands"]["subject"]["ref_id"]
    assert refs[subject_ref]["entity_type"] == "node"


def test_open_statement_does_not_create_reference_from_selection_without_explicit_participant() -> None:
    payload = response(
        ops=[
            {
                "op": "ADD",
                "element_kind": "open_statement",
                "target_id": None,
                "value": {"statement_id": "s1", "statement_type": "open", "content": None, "related_refs": ["e_decision_report"], "provenance_refs": ["p1"]},
            }
        ]
    )

    run = run_phase1b_shadow(
        "优化一下这个流程图",
        selection={"node_ids": [], "edge_ids": ["e_decision_report"]},
        model_runner=lambda _system, _payload: payload,
    )

    assert run.ok
    assert run.artifact["after_goal_state"]["references"] == []
    assert run.artifact["after_goal_state"]["open_statements"][0]["related_refs"] == []


def test_unique_predicate_role_type_normalizes_implicit_reference_entity_type() -> None:
    before = empty_goal_state()
    delta = {
        "delta_id": "d1",
        "operations": [
            {"op": "ADD", "element_kind": "reference", "target_id": None, "value": ref("r1", "node", None)},
            {"op": "ADD", "element_kind": "reference", "target_id": None, "value": ref("r2", "node", "End", 0, 3)},
            {"op": "ADD", "element_kind": "desired_fact", "target_id": None, "value": fact("f1", "relation.route_to", {"subject": reference_operand("r1"), "target": reference_operand("r2")})},
        ],
    }

    merged = merge_goal_state(before, delta, [{"provenance_id": "p1"}])

    assert merged["ok"]
    refs = {row["ref_id"]: row for row in merged["goal_state"]["references"]}
    assert refs["r1"]["entity_type"] == "edge"
    assert refs["r2"]["entity_type"] == "node"


def test_unique_predicate_role_type_creates_missing_implicit_reference() -> None:
    before = empty_goal_state()
    delta = {
        "delta_id": "d1",
        "operations": [
            {"op": "ADD", "element_kind": "reference", "target_id": None, "value": ref("r2", "node", "End", 0, 3)},
            {"op": "ADD", "element_kind": "desired_fact", "target_id": None, "value": fact("f1", "relation.route_to", {"subject": reference_operand("r_missing_edge"), "target": reference_operand("r2")})},
        ],
    }

    merged = merge_goal_state(before, delta, [{"provenance_id": "p1"}])

    assert merged["ok"]
    refs = {row["ref_id"]: row for row in merged["goal_state"]["references"]}
    assert refs["r_missing_edge"]["entity_type"] == "edge"
    assert refs["r_missing_edge"]["mention"] is None


def test_open_statement_is_retired_when_later_fact_refines_same_reference() -> None:
    before = empty_goal_state()
    before["references"] = [ref("r1", "edge", None)]
    before["open_statements"] = [{"statement_id": "s1", "statement_type": "modify", "content": None, "provenance_refs": ["p1"]}]
    delta = {
        "delta_id": "d1",
        "operations": [
            {"op": "ADD", "element_kind": "reference", "target_id": None, "value": ref("r2", "node", "End", 8, 11)},
            {"op": "ADD", "element_kind": "desired_fact", "target_id": None, "value": fact("f1", "relation.route_to", {"subject": reference_operand("r1"), "target": reference_operand("r2")})},
        ],
    }

    merged = merge_goal_state(before, delta, [{"provenance_id": "p1"}])

    assert merged["ok"]
    assert merged["goal_state"]["open_statements"] == []


@pytest.mark.xfail(reason="Audit-only behavior from reverted OpenStatement lifecycle experiment.")
def test_open_statement_retirement_handles_multiple_open_statements_individually() -> None:
    before = empty_goal_state()
    before["references"] = [ref("r_edge", "edge", None), ref("r_node", "node", "Review")]
    before["open_statements"] = [
        {"statement_id": "s1", "statement_type": "modify", "content": None, "related_refs": ["r_edge"], "provenance_refs": ["p1"]},
        {"statement_id": "s2", "statement_type": "modify", "content": {"unresolved": "node label"}, "related_refs": ["r_node"], "provenance_refs": ["p1"]},
    ]
    delta = {
        "delta_id": "d1",
        "operations": [
            {"op": "ADD", "element_kind": "reference", "target_id": None, "value": ref("r_target", "node", "End", 0, 3)},
            {"op": "ADD", "element_kind": "desired_fact", "target_id": None, "value": fact("f1", "relation.route_to", {"subject": reference_operand("r_edge"), "target": reference_operand("r_target")})},
        ],
    }

    merged = merge_goal_state(before, delta, [{"provenance_id": "p1"}])

    assert merged["ok"]
    assert [row["statement_id"] for row in merged["goal_state"]["open_statements"]] == ["s2"]


def test_open_statement_retirement_keeps_unrelated_open_statement() -> None:
    before = empty_goal_state()
    before["references"] = [ref("r_edge", "edge", None), ref("r_node", "node", "Review")]
    before["open_statements"] = [
        {"statement_id": "s1", "statement_type": "modify", "content": None, "related_refs": ["r_edge"], "provenance_refs": ["p1"]},
        {"statement_id": "s2", "statement_type": "modify", "content": {"unresolved": "node label"}, "related_refs": ["r_node"], "provenance_refs": ["p1"]},
    ]
    delta = {
        "delta_id": "d1",
        "operations": [
            {"op": "ADD", "element_kind": "reference", "target_id": None, "value": ref("r_new", "node", "Archive", 0, 7)},
            {"op": "ADD", "element_kind": "desired_fact", "target_id": None, "value": fact("f1", "entity.exists", {"subject": reference_operand("r_new")})},
        ],
    }

    merged = merge_goal_state(before, delta, [{"provenance_id": "p1"}])

    assert merged["ok"]
    assert [row["statement_id"] for row in merged["goal_state"]["open_statements"]] == ["s1", "s2"]


def test_open_statement_retirement_handles_correct_replacement() -> None:
    before = empty_goal_state()
    before["references"] = [ref("r_edge", "edge", None), ref("r_old", "node", "Review")]
    before["desired_facts"] = [fact("f_old", "relation.route_to", {"subject": reference_operand("r_edge"), "target": reference_operand("r_old")})]
    before["open_statements"] = [
        {"statement_id": "s1", "statement_type": "open", "content": {"residual": "route target unresolved"}, "related_refs": ["r_edge"], "provenance_refs": ["p1"]}
    ]
    delta = {
        "delta_id": "d1",
        "operations": [
            {"op": "ADD", "element_kind": "reference", "target_id": None, "value": ref("r_new", "node", "End", 0, 3)},
            {"op": "CORRECT", "element_kind": "desired_fact", "target_id": "f_old", "value": fact("f_new", "relation.route_to", {"subject": reference_operand("r_edge"), "target": reference_operand("r_new")})},
        ],
    }

    merged = merge_goal_state(before, delta, [{"provenance_id": "p1"}])

    assert merged["ok"]
    assert merged["goal_state"]["open_statements"] == []
    assert merged["goal_state"]["desired_facts"][0]["operands"]["target"]["ref_id"] == "r_new"


def test_open_statement_retirement_keeps_unscoped_multiple_open_statements() -> None:
    before = empty_goal_state()
    before["references"] = [ref("r_edge", "edge", None)]
    before["open_statements"] = [
        {"statement_id": "s1", "statement_type": "open", "content": None, "provenance_refs": ["p1"]},
        {"statement_id": "s2", "statement_type": "open", "content": None, "provenance_refs": ["p1"]},
    ]
    delta = {
        "delta_id": "d1",
        "operations": [
            {"op": "ADD", "element_kind": "reference", "target_id": None, "value": ref("r_target", "node", "End", 0, 3)},
            {"op": "ADD", "element_kind": "desired_fact", "target_id": None, "value": fact("f1", "relation.route_to", {"subject": reference_operand("r_edge"), "target": reference_operand("r_target")})},
        ],
    }

    merged = merge_goal_state(before, delta, [{"provenance_id": "p1"}])

    assert merged["ok"]
    assert [row["statement_id"] for row in merged["goal_state"]["open_statements"]] == ["s1", "s2"]


def test_open_statement_is_not_retired_without_reference_overlap() -> None:
    before = empty_goal_state()
    before["references"] = [ref("r1", "edge", None)]
    before["open_statements"] = [{"statement_id": "s1", "statement_type": "open", "content": {"unresolved": "target"}, "provenance_refs": ["p1"]}]
    delta = {
        "delta_id": "d1",
        "operations": [
            {"op": "ADD", "element_kind": "reference", "target_id": None, "value": ref("r2", "node", "Review", 0, 6)},
            {"op": "ADD", "element_kind": "desired_fact", "target_id": None, "value": fact("f1", "entity.exists", {"subject": reference_operand("r2")})},
        ],
    }

    merged = merge_goal_state(before, delta, [{"provenance_id": "p1"}])

    assert merged["ok"]
    assert len(merged["goal_state"]["open_statements"]) == 1


def test_contentless_open_statement_is_not_retired_without_scope_evidence() -> None:
    before = empty_goal_state()
    before["open_statements"] = [{"statement_id": "s1", "statement_type": "open", "content": None, "provenance_refs": ["p1"]}]
    delta = {
        "delta_id": "d1",
        "operations": [
            {"op": "ADD", "element_kind": "reference", "target_id": None, "value": ref("r1", "node", "Review", 0, 6)},
            {"op": "ADD", "element_kind": "desired_fact", "target_id": None, "value": fact("f1", "entity.exists", {"subject": reference_operand("r1")})},
        ],
    }

    merged = merge_goal_state(before, delta, [{"provenance_id": "p1"}])

    assert merged["ok"]
    assert len(merged["goal_state"]["open_statements"]) == 1


def test_semantic_equivalence_ignores_open_statement_subtype_and_span_only() -> None:
    left = empty_goal_state()
    left["references"] = [ref("r1", "node", "End", 4, 7)]
    left["open_statements"] = [{"statement_id": "s1", "statement_type": "modify", "content": None, "provenance_refs": ["p1"]}]
    right = empty_goal_state()
    right["references"] = [ref("x1", "node", "End", 0, 3)]
    right["open_statements"] = [{"statement_id": "s9", "statement_type": "vague_modification", "content": None, "provenance_refs": ["p9"]}]

    assert semantic_equivalence_key(left) == semantic_equivalence_key(right)


def test_semantic_equivalence_ignores_edge_deictic_mention() -> None:
    left = empty_goal_state()
    left["references"] = [ref("r1", "edge", None)]
    left["desired_facts"] = [fact("f1", "property.equals", {"subject": reference_operand("r1"), "property": property_name("display_label"), "value": {"operand_type": "explicit_empty"}})]
    right = empty_goal_state()
    right["references"] = [ref("r2", "edge", "this edge", 0, 9)]
    right["desired_facts"] = [fact("f2", "property.equals", {"subject": reference_operand("r2"), "property": property_name("display_label"), "value": {"operand_type": "explicit_empty"}})]

    assert semantic_equivalence_key(left) == semantic_equivalence_key(right)


def test_span_realigns_only_unique_explicit_text() -> None:
    utterance = "route to End"
    ops = [
        {"op": "ADD", "element_kind": "reference", "target_id": None, "value": ref("r1", "node", "End", 99, 102)},
        {"op": "ADD", "element_kind": "desired_fact", "target_id": None, "value": fact("f1", "entity.exists", {"subject": reference_operand("r1")})},
    ]
    acts = [{"turn_act_id": "a1", "sequence_index": 0, "act_kind": "semantic_update", "input_span_refs": [{"utterance_id": "u1", "start": 99, "end": 102, "text": "End"}]}]

    run = run_phase1b_shadow(utterance, model_runner=lambda _system, _payload: response(acts=acts, ops=ops))

    assert run.ok
    mention = run.artifact["after_goal_state"]["references"][0]["mention"]
    assert mention["span"]["start"] == 9


def test_span_does_not_guess_when_text_is_not_unique() -> None:
    utterance = "End then End"
    ops = [
        {"op": "ADD", "element_kind": "reference", "target_id": None, "value": ref("r1", "node", "End", 99, 102)},
        {"op": "ADD", "element_kind": "desired_fact", "target_id": None, "value": fact("f1", "entity.exists", {"subject": reference_operand("r1")})},
    ]

    run = run_phase1b_shadow(utterance, model_runner=lambda _system, _payload: response(ops=ops))

    assert not run.ok
    assert run.error == "semantic_schema_validation_failure"
