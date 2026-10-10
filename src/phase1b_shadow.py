from __future__ import annotations

import json
import os
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Callable

from .canonical_goal_state import (
    CANONICAL_GOAL_STATE_REVISION,
    CONSTRAINT_SIGNATURES,
    PREDICATE_SIGNATURES,
    canonical_json,
    canonicalize_desired_fact,
    normalize_relation_fact,
    predicate_fact_key,
    validate_relation_fact,
)
from .semantic_contract import (
    REFERENCE_CARDINALITIES,
    REFERENCE_ENTITY_TYPES,
    SEMANTIC_DELTA_OPS,
    SEMANTIC_ELEMENT_KINDS,
    STAGE_RESULTS,
    STAGE_TYPES,
    TURN_RELATIONS,
    build_model_facing_semantic_grammar,
    canonical_open_statement_type,
    canonical_property_name,
)
from .relation_fact_runtime import (
    PHASE1B_RELATION_GROUPER_SYSTEM as SHARED_RELATION_GROUPER_SYSTEM,
    attach_managed_relation_grouper,
    build_shadow_relation_group_artifact as shared_build_shadow_relation_group_artifact,
    normalize_relation_group_proposal as shared_normalize_relation_group_proposal,
    relation_grouper_allowed_refs as shared_relation_grouper_allowed_refs,
    relation_grouper_payload as shared_relation_grouper_payload,
    relation_grouper_prior_context as shared_relation_grouper_prior_context,
    relation_groups_to_managed_relation_fact_proposals as shared_relation_groups_to_managed_relation_fact_proposals,
    validate_relation_group_proposal as shared_validate_relation_group_proposal,
)
from .workflow_ir_runtime import _complete_json


PHASE1B_SHADOW_ENV = "SEMANTIC_PHASE1B_SHADOW"
PHASE1B_REVISION = "phase1b.extraction_shadow.v1"

DELTA_OPS = set(SEMANTIC_DELTA_OPS)
ELEMENT_KINDS = set(SEMANTIC_ELEMENT_KINDS) | {"relation_fact"}
PHASE1B_MODEL_GRAMMAR = build_model_facing_semantic_grammar()


PHASE1B_STAGE_SYSTEM = """Return one JSON object for a shadow-only runtime stage continuation decision.

Input contains the raw user utterance, trusted evidence, and the current runtime stage.

Output exactly:
{
  "stage_continuation": {
    "stage_type": "GROUNDING_CANDIDATE|PREFLIGHT_CONFIRMATION",
    "result": "resolved|unresolved|remain_pending|cancelled|authorized|declined",
    "evidence_refs": ["evidence id"],
    "consumed_spans": [{"utterance_id":"u1","start":0,"end":0}],
    "residual_spans": [{"utterance_id":"u1","start":0,"end":0}]
  }
}

Rules:
- This is shadow only. Do not output workflow IR, graph mutations, planner ops, executor ops, pending types, candidate ids, or confirmation ids.
- Candidate answers and confirmation answers are StageContinuation evidence only; they must not create DesiredFacts by themselves.
- Residual semantic content must use spans from the original utterance. Do not rewrite residual text into an authoritative synthetic instruction.
- Return JSON only.
"""


PHASE1B_CONTINUITY_SYSTEM = """Return one JSON object for a shadow-only discourse reference continuity decision.

Input contains current utterance and current canonical GoalState only. It does not contain frontend selection,
node ids, edge ids, graph topology, business entity ids, candidate ids, or grounding facts.

Output exactly:
{
  "discourse_reference_continuity": {
    "status": "continued|replaced|new|none",
    "ref_ids": ["prior semantic ref_id"],
    "evidence": "short text"
  }
}

Responsibilities:
- Decide whether this turn continues to talk about prior semantic participant(s).
- Decide whether this utterance explicitly introduces a new semantic participant.
- Decide whether this utterance explicitly corrects or replaces prior participant(s).
- Return none when no participant can be carried forward.

Do not output predicates, DesiredFacts, constraints, graph ids, node ids, edge ids, target ids,
SemanticDelta, binding decisions, grounding decisions, lowering, planner, or executor information.
Return JSON only.
"""


PHASE1B_EXTRACTION_SYSTEM = """Return one JSON object for a shadow-only semantic proposal.

Input contains the raw user utterance, current canonical GoalState, task context, trusted evidence,
stage continuation result, residual spans, selection facts, and graph facts.

Output exactly:
{
  "turn_relation": {
    "relation": "CONTINUE|CORRECT|CANCEL|NEW_TASK",
    "task_context_ref": "task id or null",
    "evidence_refs": ["evidence id"],
    "reason_codes": ["short_debug_code"]
  },
  "semantic_proposal": {
    "references": [],
    "desired_facts": [],
    "constraints": [],
    "open_statements": []
  }
}

Rules:
- This is shadow only. Do not output workflow IR, graph mutations, steps, graph ids, node ids, edge ids, planner ops, executor ops, pending types, candidate ids, or confirmation ids.
- Do not output delta objects, delta operations, or correction targets. Delta planning is deterministic code ownership.
- NEW_TASK belongs only in turn_relation.relation, never in the semantic proposal.
- Candidate answers and confirmation answers are StageContinuation evidence only; they must not create DesiredFacts by themselves.
- Residual semantic content must use spans from the original utterance. Do not rewrite residual text into an authoritative synthetic instruction.
- Selection and graph facts are trusted evidence, not user intent. Do not create an intent solely because something is selected or present in the graph.
- Explicit empty can be represented only when the user expression itself clearly commits to an empty value. Use {"operand_type":"explicit_empty"} as the value operand.
- For omitted/implicit subjects, create a Reference with mention null and the correct entity type. Do not invent unresolved slots.
- Return JSON only.
""" + PHASE1B_MODEL_GRAMMAR


PHASE1B_RELATION_GROUPER_SYSTEM = SHARED_RELATION_GROUPER_SYSTEM


@dataclass(frozen=True)
class Phase1BRun:
    ok: bool
    artifact: dict[str, Any]
    error: str | None = None


def phase1b_shadow_enabled() -> bool:
    return os.getenv(PHASE1B_SHADOW_ENV, "").strip().lower() in {"1", "true", "yes"}


def empty_goal_state(goal_state_id: str = "gs1") -> dict[str, Any]:
    return {
        "goal_state_id": goal_state_id,
        "revision": CANONICAL_GOAL_STATE_REVISION,
        "references": [],
        "desired_facts": [],
        "constraints": [],
        "open_statements": [],
    }


def run_phase1b_shadow(
    raw_utterance: str,
    *,
    current_goal_state: dict[str, Any] | None = None,
    task_context: dict[str, Any] | None = None,
    trusted_evidence: list[dict[str, Any]] | None = None,
    runtime_stage: dict[str, Any] | None = None,
    graph_context: dict[str, Any] | None = None,
    selection: dict[str, Any] | None = None,
    model_runner: Callable[[str, dict[str, Any]], Any] | None = None,
) -> Phase1BRun:
    before_goal_state = deepcopy(current_goal_state) if isinstance(current_goal_state, dict) else empty_goal_state()
    evidence_bundle = _base_evidence(raw_utterance, trusted_evidence)
    payload = {
        "raw_utterance": raw_utterance,
        "current_goal_state": before_goal_state,
        "task_context": task_context if isinstance(task_context, dict) else {},
        "trusted_evidence": trusted_evidence if isinstance(trusted_evidence, list) else [],
        "runtime_stage": runtime_stage if isinstance(runtime_stage, dict) else {},
        "graph_context": graph_context if isinstance(graph_context, dict) else {},
        "selection": selection if isinstance(selection, dict) else {"node_ids": [], "edge_ids": []},
        "evidence_bundle": evidence_bundle,
    }
    artifact: dict[str, Any] = {
        "enabled": True,
        "revision": PHASE1B_REVISION,
        "raw_turn": {"utterance_id": "u1", "text": raw_utterance},
        "before_goal_state": before_goal_state,
        "evidence_bundle": evidence_bundle,
        "payload_summary": {
            "has_goal_state": bool(before_goal_state),
            "runtime_stage": payload["runtime_stage"].get("stage_type"),
            "selection_counts": {
                "node_ids": len(payload["selection"].get("node_ids", [])) if isinstance(payload["selection"].get("node_ids"), list) else 0,
                "edge_ids": len(payload["selection"].get("edge_ids", [])) if isinstance(payload["selection"].get("edge_ids"), list) else 0,
            },
        },
        "errors": [],
    }
    try:
        runner = model_runner or _default_model_runner
        stage = None
        if payload["runtime_stage"]:
            stage_result = runner(PHASE1B_STAGE_SYSTEM, payload)
            parsed_stage = _parsed_json(stage_result)
            artifact["raw_stage_continuation"] = _raw_model_trace(stage_result, parsed_stage)
            stage = _normalize_stage_continuation(_stage_continuation_value(parsed_stage))
            if stage is not None:
                _realign_spans_in_place(stage, raw_utterance)
        semantic_payload = deepcopy(payload)
        semantic_payload["stage_continuation"] = stage
        semantic_payload["residual_spans"] = _residual_spans_for_semantic(raw_utterance, stage)
        if stage is not None and not semantic_payload["residual_spans"]:
            parsed = {
                "turn_relation": {
                    "relation": "CONTINUE",
                    "task_context_ref": semantic_payload["task_context"].get("task_context_id"),
                    "evidence_refs": stage.get("evidence_refs", []),
                    "reason_codes": ["stage_consumed_turn"],
                },
                "semantic_proposal": {"references": [], "desired_facts": [], "constraints": [], "open_statements": []},
            }
            model_result = parsed
        else:
            continuity_result = runner(
                PHASE1B_CONTINUITY_SYSTEM,
                {
                    "raw_utterance": raw_utterance,
                    "current_goal_state": before_goal_state,
                },
            )
            parsed_continuity = _parsed_json(continuity_result)
            artifact["raw_discourse_reference_continuity"] = _raw_model_trace(continuity_result, parsed_continuity)
            discourse_continuity = normalize_discourse_reference_continuity(_discourse_reference_continuity_value(parsed_continuity))
            artifact["discourse_reference_continuity"] = discourse_continuity
            semantic_payload_for_model, continuity_injected = semantic_payload_with_discourse_continuity(semantic_payload, discourse_continuity)
            artifact["discourse_continuity_integration"] = {
                "status": discourse_continuity.get("status"),
                "injected": continuity_injected,
                "payload_diff": [{"path": "$.discourse_reference_continuity", "kind": "added"}] if continuity_injected else [],
            }
            model_result = runner(PHASE1B_EXTRACTION_SYSTEM, semantic_payload_for_model)
        parsed = _parsed_json(model_result)
        artifact["raw_extraction"] = _raw_model_trace(model_result, parsed)
        if not isinstance(parsed, dict):
            return _failed(artifact, "semantic_extraction_failure", "model did not return a JSON object")
        normalized = normalize_phase1b_extraction(parsed, stage_continuation=stage, residual_spans=semantic_payload["residual_spans"])
        _realign_spans_in_place(normalized, raw_utterance)
        planning_goal_state = _planning_goal_state(before_goal_state, normalized["turn_relation"])
        relation_fact_proposals = _attach_shadow_relation_grouper(artifact, runner, raw_utterance, normalized["semantic_proposal"], planning_goal_state)
        if relation_fact_proposals:
            normalized["semantic_proposal"].setdefault("relation_facts", []).extend(relation_fact_proposals)
        planned = plan_semantic_delta(planning_goal_state, normalized["turn_relation"], normalized["semantic_proposal"])
        artifact["turn_relation"] = normalized["turn_relation"]
        artifact["stage_continuation"] = normalized.get("stage_continuation")
        artifact["turn_acts"] = normalized["turn_acts"]
        artifact["semantic_proposal"] = normalized["semantic_proposal"]
        artifact["semantic_delta"] = planned["semantic_delta"]
        artifact["delta_planner"] = {"ok": not planned["errors"], "errors": planned["errors"]}
        validation = validate_phase1b_extraction(normalized, before_goal_state, raw_utterance)
        validation["errors"].extend(planned["errors"])
        validation["ok"] = not validation["errors"]
        artifact["validation"] = validation
        if not validation["ok"]:
            artifact["after_goal_state"] = before_goal_state
            return Phase1BRun(ok=False, artifact=artifact, error="semantic_schema_validation_failure")
        merge = merge_goal_state(planning_goal_state, planned["semantic_delta"], evidence_bundle)
        artifact["merge"] = {"ok": merge["ok"], "errors": merge["errors"]}
        artifact["after_goal_state"] = merge["goal_state"]
        artifact["task_lifecycle"] = _task_lifecycle_record(before_goal_state, planning_goal_state, normalized["turn_relation"])
        artifact["revision_record"] = build_revision_record(planning_goal_state, merge["goal_state"], planned["semantic_delta"])
        artifact["invariants"] = evaluate_phase1b_invariants(artifact)
        artifact["semantic_equivalence_key"] = semantic_equivalence_key(merge["goal_state"])
        artifact["ok"] = merge["ok"] and artifact["invariants"]["ok"]
        return Phase1BRun(ok=bool(artifact["ok"]), artifact=artifact, error=None if artifact["ok"] else "invariant_failure")
    except Exception as exc:
        artifact["ok"] = False
        artifact["errors"].append({"category": "shadow_internal_error", "message": str(exc)})
        artifact["after_goal_state"] = before_goal_state
        return Phase1BRun(ok=False, artifact=artifact, error="shadow_internal_error")


def _default_model_runner(system_prompt: str, payload: dict[str, Any]) -> Any:
    return _complete_json(system_prompt, payload)


def _planning_goal_state(before_goal_state: dict[str, Any], turn_relation: dict[str, Any]) -> dict[str, Any]:
    if turn_relation.get("relation") == "NEW_TASK":
        return empty_goal_state()
    return before_goal_state


def _task_lifecycle_record(before_goal_state: dict[str, Any], planning_goal_state: dict[str, Any], turn_relation: dict[str, Any]) -> dict[str, Any]:
    relation = turn_relation.get("relation")
    return {
        "relation": relation,
        "preserved_goal_state_key": semantic_equivalence_key(before_goal_state) if relation == "NEW_TASK" else None,
        "planning_goal_state_key": semantic_equivalence_key(planning_goal_state),
    }


def _parsed_json(model_result: Any) -> Any:
    if isinstance(model_result, dict):
        return model_result
    parsed = getattr(model_result, "parsed_json", None)
    if parsed is not None:
        return parsed
    raw = getattr(model_result, "raw_response", "")
    if isinstance(raw, str) and raw.strip():
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return None
    return None


def _raw_model_trace(model_result: Any, parsed: Any) -> dict[str, Any]:
    return {
        "parsed": parsed,
        "parse_error": getattr(model_result, "parse_error", None),
        "normalized_from_fence": getattr(model_result, "normalized_from_fence", False),
    }


def normalize_phase1b_extraction(
    parsed: dict[str, Any],
    *,
    stage_continuation: dict[str, Any] | None = None,
    residual_spans: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    proposal = _normalize_semantic_proposal(parsed.get("semantic_proposal") if "semantic_proposal" in parsed else parsed)
    acts = _build_turn_acts(stage_continuation, residual_spans or [])
    return {
        "turn_relation": _normalize_turn_relation(parsed.get("turn_relation")),
        "stage_continuation": stage_continuation,
        "turn_acts": acts,
        "semantic_proposal": proposal,
        "model_contract_errors": _model_contract_errors(parsed),
    }


def _normalize_turn_relation(value: Any) -> dict[str, Any]:
    row = value if isinstance(value, dict) else {}
    relation = str(row.get("relation") or "NEW_TASK").strip()
    return {
        "relation": relation if relation in TURN_RELATIONS else "NEW_TASK",
        "task_context_ref": row.get("task_context_ref") if row.get("task_context_ref") is None or isinstance(row.get("task_context_ref"), str) else None,
        "evidence_refs": _string_list(row.get("evidence_refs")),
        "reason_codes": _string_list(row.get("reason_codes")),
    }


def _normalize_stage_continuation(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    stage_type = value.get("stage_type")
    if stage_type is None:
        return None
    stage_type = str(stage_type)
    result = value.get("result")
    result = str(result) if result is not None else "unresolved"
    return {
        "stage_type": stage_type,
        "result": result,
        "evidence_refs": _string_list(value.get("evidence_refs")),
        "consumed_spans": _span_list(value.get("consumed_spans")),
        "residual_spans": _span_list(value.get("residual_spans")),
        "stage_result_ref": value.get("stage_result_ref") if isinstance(value.get("stage_result_ref"), str) else None,
    }


def _normalize_turn_acts(value: Any) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    rows = value if isinstance(value, list) else []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            continue
        result.append(
            {
                "turn_act_id": str(row.get("turn_act_id") or f"a{index + 1}"),
                "sequence_index": int(row.get("sequence_index")) if isinstance(row.get("sequence_index"), int) else index,
                "act_kind": str(row.get("act_kind") or "semantic_update"),
                "input_span_refs": _span_list(row.get("input_span_refs")),
                "depends_on": _string_list(row.get("depends_on")),
                "produced_state_ref": row.get("produced_state_ref") if isinstance(row.get("produced_state_ref"), str) else None,
                "requires_resulting_graph_version": bool(row.get("requires_resulting_graph_version")),
            }
        )
    return sorted(result, key=lambda item: item["sequence_index"])


def _normalize_semantic_delta(value: Any) -> dict[str, Any]:
    row = value if isinstance(value, dict) else {}
    operations = row.get("operations") if isinstance(row.get("operations"), list) else []
    normalized_ops = []
    for index, op in enumerate(operations):
        if not isinstance(op, dict):
            continue
        normalized_ops.append(
            {
                "operation_id": str(op.get("operation_id") or f"op{index + 1}"),
                "op": str(op.get("op") or ""),
                "element_kind": str(op.get("element_kind") or ""),
                "target_id": op.get("target_id") if op.get("target_id") is None or isinstance(op.get("target_id"), str) else None,
                "value": deepcopy(op.get("value")) if isinstance(op.get("value"), dict) else {},
                "evidence_refs": _string_list(op.get("evidence_refs")),
            }
        )
    return {"delta_id": str(row.get("delta_id") or "d1"), "operations": normalized_ops}


def _normalize_semantic_proposal(value: Any) -> dict[str, Any]:
    row = value if isinstance(value, dict) else {}
    return {
        "references": [_normalize_reference_element(item, []) for item in row.get("references", []) if isinstance(item, dict)],
        "desired_facts": [_normalize_fact_element(item, []) for item in row.get("desired_facts", []) if isinstance(item, dict)],
        "relation_facts": [_normalize_relation_fact_proposal_element(item) for item in row.get("relation_facts", []) if isinstance(item, dict)],
        "constraints": [_normalize_constraint_element(item, []) for item in row.get("constraints", []) if isinstance(item, dict)],
        "open_statements": [_normalize_open_statement(item, []) for item in row.get("open_statements", []) if isinstance(item, dict)],
    }


def _model_contract_errors(parsed: dict[str, Any]) -> list[dict[str, Any]]:
    forbidden = []
    for row in _walk_dicts(parsed):
        for field in ("semantic_delta", "target_id", "op", "operations"):
            if field in row:
                forbidden.append({"category": "model_contract_failure", "message": "model output includes deterministic-planner field", "field": field})
    return forbidden


def _stage_continuation_value(parsed: Any) -> Any:
    if not isinstance(parsed, dict):
        return None
    if "stage_continuation" in parsed:
        return parsed.get("stage_continuation")
    return parsed


def _discourse_reference_continuity_value(parsed: Any) -> Any:
    if not isinstance(parsed, dict):
        return None
    if "discourse_reference_continuity" in parsed:
        return parsed.get("discourse_reference_continuity")
    return parsed


def normalize_discourse_reference_continuity(value: Any) -> dict[str, Any]:
    row = value if isinstance(value, dict) else {}
    status = str(row.get("status") or "none").strip()
    return {
        "status": status if status in {"continued", "replaced", "new", "none"} else "none",
        "ref_ids": _string_list(row.get("ref_ids")),
        "evidence": row.get("evidence") if isinstance(row.get("evidence"), str) else "",
    }


def discourse_reference_continuity_is_effective(value: dict[str, Any] | None) -> bool:
    if not isinstance(value, dict):
        return False
    if value.get("status") == "none":
        return False
    return bool(value.get("ref_ids"))


def semantic_payload_with_discourse_continuity(payload: dict[str, Any], continuity: dict[str, Any] | None) -> tuple[dict[str, Any], bool]:
    if not discourse_reference_continuity_is_effective(continuity):
        return payload, False
    enriched = deepcopy(payload)
    enriched["discourse_reference_continuity"] = deepcopy(continuity)
    return enriched, True


def _residual_spans_for_semantic(utterance: str, stage: dict[str, Any] | None) -> list[dict[str, Any]]:
    if stage is None:
        return [{"utterance_id": "u1", "start": 0, "end": len(utterance)}] if utterance else []
    residual = _span_list(stage.get("residual_spans"))
    if residual:
        return residual
    consumed = _span_list(stage.get("consumed_spans"))
    if not consumed:
        return [{"utterance_id": "u1", "start": 0, "end": len(utterance)}] if utterance else []
    return []


def _build_turn_acts(stage: dict[str, Any] | None, residual_spans: list[dict[str, Any]]) -> list[dict[str, Any]]:
    acts: list[dict[str, Any]] = []
    if stage is not None:
        acts.append(
            {
                "turn_act_id": "a1",
                "sequence_index": 0,
                "act_kind": "stage_continuation",
                "input_span_refs": _span_list(stage.get("consumed_spans")),
                "depends_on": [],
                "produced_state_ref": stage.get("stage_result_ref"),
                "requires_resulting_graph_version": False,
            }
        )
    if residual_spans:
        acts.append(
            {
                "turn_act_id": f"a{len(acts) + 1}",
                "sequence_index": len(acts),
                "act_kind": "semantic_update",
                "input_span_refs": residual_spans,
                "depends_on": [acts[-1]["turn_act_id"]] if acts else [],
                "produced_state_ref": None,
                "requires_resulting_graph_version": False,
            }
        )
    return acts


def plan_semantic_delta(goal_state: dict[str, Any], turn_relation: dict[str, Any], proposal: dict[str, Any]) -> dict[str, Any]:
    relation = turn_relation.get("relation")
    errors: list[dict[str, Any]] = []
    operations: list[dict[str, Any]] = []
    if relation == "CANCEL":
        return {"semantic_delta": {"delta_id": "d1", "operations": []}, "errors": []}

    grouped_relation_candidate_pairs = _relation_fact_member_pairs(proposal.get("relation_facts", []))
    existing_ref_ids = {
        row.get("ref_id")
        for row in goal_state.get("references", [])
        if isinstance(row, dict) and isinstance(row.get("ref_id"), str)
    }
    for ref_value in proposal.get("references", []):
        ref_id = ref_value.get("ref_id")
        if isinstance(ref_id, str) and ref_id and ref_id in existing_ref_ids:
            continue
        operations.append(_delta_op("ADD", "reference", None, ref_value))
        if isinstance(ref_id, str) and ref_id:
            existing_ref_ids.add(ref_id)

    for fact_value in proposal.get("desired_facts", []):
        if relation == "CORRECT" and grouped_relation_candidate_pairs and _desired_relation_fact_pair(fact_value) in grouped_relation_candidate_pairs:
            continue
        if relation == "CORRECT":
            target_id, target_errors = _resolve_correction_target(goal_state, fact_value)
            errors.extend(target_errors)
            if target_id:
                operations.append(_delta_op("CORRECT", "desired_fact", target_id, fact_value))
            continue
        target_id = _same_fact_key_target(goal_state, fact_value)
        if target_id:
            operations.append(_delta_op("CORRECT", "desired_fact", target_id, fact_value))
        else:
            operations.append(_delta_op("ADD", "desired_fact", None, fact_value))

    for relation_fact_value in proposal.get("relation_facts", []):
        if relation == "CORRECT":
            target_id, target_errors = _resolve_relation_fact_correction_target(goal_state, relation_fact_value)
            errors.extend(target_errors)
            if target_id:
                operations.append(_delta_op("CORRECT", "relation_fact", target_id, relation_fact_value))
            continue
        target_id = _same_relation_fact_key_target(goal_state, relation_fact_value)
        if not target_id:
            operations.append(_delta_op("ADD", "relation_fact", None, relation_fact_value))

    for constraint_value in proposal.get("constraints", []):
        operations.append(_delta_op("ADD", "constraint", None, constraint_value))
    for statement_value in proposal.get("open_statements", []):
        operations.append(_delta_op("ADD", "open_statement", None, statement_value))

    return {"semantic_delta": {"delta_id": "d1", "operations": operations if not errors else []}, "errors": errors}


def _delta_op(op: str, kind: str, target_id: str | None, value: dict[str, Any]) -> dict[str, Any]:
    return {
        "operation_id": "",
        "op": op,
        "element_kind": kind,
        "target_id": target_id,
        "value": deepcopy(value),
        "evidence_refs": _string_list(value.get("provenance_refs")),
    }


def _same_fact_key_target(goal_state: dict[str, Any], fact_value: dict[str, Any]) -> str | None:
    key = predicate_fact_key(fact_value)
    matches = [
        row.get("fact_id")
        for row in goal_state.get("desired_facts", [])
        if isinstance(row, dict) and predicate_fact_key(row) == key and isinstance(row.get("fact_id"), str)
    ]
    return matches[0] if len(matches) == 1 else None


def _relation_fact_member_pairs(relation_facts: Any) -> set[tuple[str, str]]:
    pairs: set[tuple[str, str]] = set()
    for relation_fact in relation_facts if isinstance(relation_facts, list) else []:
        if not isinstance(relation_fact, dict):
            continue
        normalized = normalize_relation_fact(relation_fact)
        for member in normalized["members"]:
            if member["source_ref"] and member["target_ref"]:
                pairs.add((member["source_ref"], member["target_ref"]))
    return pairs


def _desired_relation_fact_pair(fact_value: dict[str, Any]) -> tuple[str, str] | None:
    if not isinstance(fact_value, dict) or not str(fact_value.get("predicate") or "").startswith("relation."):
        return None
    operands = fact_value.get("operands") if isinstance(fact_value.get("operands"), dict) else {}

    def ref_id(role: str) -> str | None:
        operand = operands.get(role)
        if isinstance(operand, dict) and operand.get("operand_type") == "reference" and isinstance(operand.get("ref_id"), str):
            return operand["ref_id"]
        return None

    for source_role, target_role in (("source", "target"), ("subject", "anchor"), ("subject", "target")):
        source = ref_id(source_role)
        target = ref_id(target_role)
        if source and target:
            return (source, target)
    return None


def _resolve_correction_target(goal_state: dict[str, Any], fact_value: dict[str, Any]) -> tuple[str | None, list[dict[str, Any]]]:
    same_key = _same_fact_key_target(goal_state, fact_value)
    if same_key:
        return same_key, []
    candidates = [
        row
        for row in goal_state.get("desired_facts", [])
        if isinstance(row, dict) and _cross_factkey_compatible(row, fact_value)
    ]
    if len(candidates) == 1 and isinstance(candidates[0].get("fact_id"), str):
        return candidates[0]["fact_id"], []
    if not candidates:
        return None, [{"category": "semantic_delta_planning_failure", "message": "CORRECT target could not be resolved"}]
    return None, [{"category": "semantic_delta_planning_failure", "message": "CORRECT target is ambiguous"}]


def _cross_factkey_compatible(before_fact: dict[str, Any], proposal_fact: dict[str, Any]) -> bool:
    if before_fact.get("predicate") != proposal_fact.get("predicate"):
        return False
    before_refs = _reference_operand_ids(before_fact.get("operands"))
    proposal_refs = _reference_operand_ids(proposal_fact.get("operands"))
    return bool(before_refs) and before_refs == proposal_refs


def _managed_relation_facts(goal_state: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        row
        for row in goal_state.get("relation_facts", [])
        if isinstance(row, dict)
        and isinstance(row.get("relation_id"), str)
        and row.get("relation_id")
        and not str(row.get("relation_id")).startswith("rf:")
    ]


def _relation_fact_exact_key(relation_fact: dict[str, Any]) -> str:
    normalized = normalize_relation_fact(relation_fact)
    members = [
        {
            "source_ref": member["source_ref"],
            "target_ref": member["target_ref"],
            "properties": member["properties"],
        }
        for member in normalized["members"]
    ]
    return canonical_json(
        {
            "kind": normalized["kind"],
            "members": sorted(members, key=canonical_json),
            "group_semantics": normalized["group_semantics"],
        }
    )


def _relation_fact_endpoint_sets(relation_fact: dict[str, Any]) -> tuple[frozenset[str], frozenset[str]]:
    normalized = normalize_relation_fact(relation_fact)
    return (
        frozenset(member["source_ref"] for member in normalized["members"] if member["source_ref"]),
        frozenset(member["target_ref"] for member in normalized["members"] if member["target_ref"]),
    )


def _same_relation_fact_key_target(goal_state: dict[str, Any], relation_fact_value: dict[str, Any]) -> str | None:
    key = _relation_fact_exact_key(relation_fact_value)
    matches = [
        row.get("relation_id")
        for row in _managed_relation_facts(goal_state)
        if _relation_fact_exact_key(row) == key
    ]
    return matches[0] if len(matches) == 1 else None


def _resolve_relation_fact_correction_target(goal_state: dict[str, Any], relation_fact_value: dict[str, Any]) -> tuple[str | None, list[dict[str, Any]]]:
    same_key = _same_relation_fact_key_target(goal_state, relation_fact_value)
    if same_key:
        return same_key, []
    proposal = normalize_relation_fact(relation_fact_value)
    proposal_sources, proposal_targets = _relation_fact_endpoint_sets(proposal)
    candidates = []
    for row in _managed_relation_facts(goal_state):
        existing = normalize_relation_fact(row)
        if existing.get("kind") != proposal.get("kind"):
            continue
        existing_sources, existing_targets = _relation_fact_endpoint_sets(existing)
        if proposal_sources and proposal_sources == existing_sources:
            candidates.append(row)
        elif proposal_targets and proposal_targets == existing_targets:
            candidates.append(row)
    unique = {row.get("relation_id"): row for row in candidates if isinstance(row.get("relation_id"), str)}
    if len(unique) == 1:
        return next(iter(unique)), []
    if not unique:
        return None, [{"category": "semantic_delta_planning_failure", "message": "CORRECT relation_fact target could not be resolved"}]
    return None, [{"category": "semantic_delta_planning_failure", "message": "CORRECT relation_fact target is ambiguous"}]


def validate_phase1b_extraction(extraction: dict[str, Any], goal_state: dict[str, Any], utterance: str) -> dict[str, Any]:
    errors: list[dict[str, Any]] = []
    errors.extend(extraction.get("model_contract_errors", []))
    turn_relation = extraction["turn_relation"]
    if turn_relation["relation"] not in TURN_RELATIONS:
        errors.append({"category": "turn_relation_failure", "message": "invalid turn relation"})
    stage = extraction.get("stage_continuation")
    if stage is not None:
        if stage["stage_type"] not in STAGE_TYPES:
            errors.append({"category": "stage_continuation_failure", "message": "semantic clarification stages are not handled here"})
        if stage["result"] not in STAGE_RESULTS:
            errors.append({"category": "stage_continuation_failure", "message": "invalid stage result"})
        for span in stage["consumed_spans"] + stage["residual_spans"]:
            if not _span_valid(span, utterance):
                errors.append({"category": "span_partition_failure", "message": "invalid stage continuation span", "span": span})
    for act in extraction["turn_acts"]:
        for span in act["input_span_refs"]:
            if not _span_valid(span, utterance):
                errors.append({"category": "span_partition_failure", "message": "invalid turn act span", "span": span})
    for span in _spans_in_value(extraction["semantic_proposal"]):
        if not _span_valid(span, utterance):
            errors.append({"category": "span_partition_failure", "message": "invalid semantic proposal span", "span": span})
    errors.extend(validate_semantic_proposal(extraction["semantic_proposal"]))
    errors.extend(_validate_relation_fact_endpoint_refs(extraction["semantic_proposal"], goal_state))
    return {"ok": not errors, "errors": errors}


def validate_semantic_proposal(proposal: dict[str, Any]) -> list[dict[str, Any]]:
    errors: list[dict[str, Any]] = []
    for ref_value in proposal.get("references", []):
        errors.extend(_validate_reference_value(ref_value))
    for fact_value in proposal.get("desired_facts", []):
        errors.extend(_validate_fact_value(fact_value))
    for relation_fact_value in proposal.get("relation_facts", []):
        errors.extend(_validate_relation_fact_value(relation_fact_value, allow_missing_relation_id=True))
    return errors


def _attach_shadow_relation_grouper(
    artifact: dict[str, Any],
    runner: Callable[[str, dict[str, Any]], Any],
    raw_utterance: str,
    semantic_proposal: dict[str, Any],
    planning_goal_state: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    return attach_managed_relation_grouper(
        artifact,
        runner,
        raw_utterance,
        semantic_proposal,
        planning_goal_state=planning_goal_state,
        system_prompt=PHASE1B_RELATION_GROUPER_SYSTEM,
        parsed_json=_parsed_json,
        raw_model_trace=_raw_model_trace,
    )


def relation_grouper_payload(raw_utterance: str, semantic_proposal: dict[str, Any], *, planning_goal_state: dict[str, Any] | None = None) -> dict[str, Any]:
    return shared_relation_grouper_payload(raw_utterance, semantic_proposal, planning_goal_state=planning_goal_state)


def build_shadow_relation_group_artifact(parsed: Any, semantic_proposal: dict[str, Any], *, planning_goal_state: dict[str, Any] | None = None) -> dict[str, Any]:
    return shared_build_shadow_relation_group_artifact(parsed, semantic_proposal, planning_goal_state=planning_goal_state)


def normalize_relation_group_proposal(parsed: Any) -> dict[str, Any]:
    return shared_normalize_relation_group_proposal(parsed)


def validate_relation_group_proposal(proposal: dict[str, Any], semantic_proposal: dict[str, Any], *, planning_goal_state: dict[str, Any] | None = None) -> dict[str, Any]:
    return shared_validate_relation_group_proposal(proposal, semantic_proposal, planning_goal_state=planning_goal_state)


def relation_grouper_allowed_refs(semantic_proposal: dict[str, Any], planning_goal_state: dict[str, Any] | None = None) -> set[str]:
    return shared_relation_grouper_allowed_refs(semantic_proposal, planning_goal_state)


def relation_grouper_prior_context(semantic_proposal: dict[str, Any], planning_goal_state: dict[str, Any] | None = None) -> dict[str, Any]:
    return shared_relation_grouper_prior_context(semantic_proposal, planning_goal_state)


def _active_goal_state_ref_ids(goal_state: dict[str, Any]) -> set[str]:
    return {
        row.get("ref_id")
        for row in goal_state.get("references", [])
        if isinstance(row, dict) and isinstance(row.get("ref_id"), str) and row.get("ref_id")
    }


def _active_managed_relation_ref_ids(goal_state: dict[str, Any]) -> set[str]:
    ref_ids: set[str] = set()
    for relation_fact in _managed_relation_facts(goal_state):
        normalized = normalize_relation_fact(relation_fact)
        for member in normalized["members"]:
            if member["source_ref"]:
                ref_ids.add(member["source_ref"])
            if member["target_ref"]:
                ref_ids.add(member["target_ref"])
    return ref_ids


def _reference_mention_text(reference: dict[str, Any]) -> str | None:
    mention = reference.get("mention")
    if isinstance(mention, dict) and isinstance(mention.get("text"), str):
        return mention["text"]
    return None


def relation_groups_to_managed_relation_fact_proposals(proposal: dict[str, Any], semantic_proposal: dict[str, Any]) -> list[dict[str, Any]]:
    return shared_relation_groups_to_managed_relation_fact_proposals(proposal, semantic_proposal)


def _relation_group_provenance_refs(semantic_proposal: dict[str, Any]) -> list[str]:
    refs: list[str] = []
    for field in ("references", "desired_facts"):
        for row in semantic_proposal.get(field, []):
            if not isinstance(row, dict):
                continue
            for provenance_ref in _string_list(row.get("provenance_refs")):
                if provenance_ref not in refs:
                    refs.append(provenance_ref)
    return refs or ["p1"]


def validate_semantic_delta(delta: dict[str, Any], goal_state: dict[str, Any]) -> list[dict[str, Any]]:
    errors: list[dict[str, Any]] = []
    ids = _element_ids(goal_state)
    for op in delta.get("operations", []):
        op_name = op.get("op")
        element_kind = op.get("element_kind")
        if op_name not in DELTA_OPS:
            errors.append({"category": "semantic_delta_failure", "message": "invalid SemanticDelta op", "op": op_name})
        if element_kind not in ELEMENT_KINDS:
            errors.append({"category": "semantic_delta_failure", "message": "invalid SemanticDelta element kind", "element_kind": element_kind})
        if op_name in {"CORRECT", "RETRACT"} and op.get("target_id") not in ids:
            errors.append({"category": "semantic_delta_failure", "message": "target_id does not identify current semantic truth", "target_id": op.get("target_id")})
        if op_name == "ADD" and op.get("target_id"):
            errors.append({"category": "semantic_delta_failure", "message": "ADD must not target an existing element"})
        if element_kind == "desired_fact" and op_name in {"ADD", "CORRECT"}:
            fact = op.get("value") if isinstance(op.get("value"), dict) else {}
            errors.extend(_validate_fact_value(fact))
        if element_kind == "reference" and op_name in {"ADD", "CORRECT"}:
            errors.extend(_validate_reference_value(op.get("value") if isinstance(op.get("value"), dict) else {}))
        if element_kind == "relation_fact" and op_name in {"ADD", "CORRECT"}:
            errors.extend(_validate_relation_fact_value(op.get("value") if isinstance(op.get("value"), dict) else {}, allow_missing_relation_id=True))
    return errors


def merge_goal_state(goal_state: dict[str, Any], delta: dict[str, Any], evidence_bundle: list[dict[str, Any]]) -> dict[str, Any]:
    next_state = deepcopy(goal_state)
    before_state = deepcopy(goal_state)
    errors = validate_semantic_delta(delta, next_state)
    if errors:
        return {"ok": False, "goal_state": next_state, "errors": errors}
    if not delta.get("operations"):
        return {"ok": True, "goal_state": next_state, "errors": []}
    for op in delta.get("operations", []):
        op_name = op["op"]
        kind = op["element_kind"]
        target_id = op.get("target_id")
        if op_name == "ADD":
            _add_element(next_state, kind, op.get("value", {}), evidence_bundle)
        elif op_name == "CORRECT":
            _remove_element(next_state, kind, target_id)
            value = deepcopy(op.get("value", {}))
            if kind == "relation_fact" and isinstance(target_id, str):
                value["relation_id"] = target_id
            _add_element(next_state, kind, value, evidence_bundle)
        elif op_name == "RETRACT":
            _remove_element(next_state, kind, target_id)
    _canonicalize_reference_entity_types_from_fact_roles(next_state)
    _canonicalize_goal_state(next_state)
    _retire_refined_open_statements(next_state, before_state, delta)
    _renumber_elements(next_state)
    return {"ok": True, "goal_state": next_state, "errors": []}


def build_revision_record(before: dict[str, Any], after: dict[str, Any], delta: dict[str, Any]) -> dict[str, Any]:
    return {
        "revision_id": "rev1",
        "phase": "phase1b_shadow",
        "before_key": semantic_equivalence_key(before),
        "after_key": semantic_equivalence_key(after),
        "delta": deepcopy(delta),
    }


def evaluate_phase1b_invariants(artifact: dict[str, Any]) -> dict[str, Any]:
    errors: list[dict[str, Any]] = []
    goal_state = artifact.get("after_goal_state") if isinstance(artifact.get("after_goal_state"), dict) else {}
    forbidden_fields = {
        "binding",
        "bindings",
        "candidate",
        "candidates",
        "pending",
        "confirmation",
        "semantic_role",
        "node_id",
        "edge_id",
        "operation",
        "history",
        "delta_history",
    }
    for row in _walk_dicts(goal_state):
        overlap = set(row) & forbidden_fields
        if overlap:
            errors.append({"category": "invariant_failure", "message": "forbidden GoalState field", "fields": sorted(overlap)})
    for op in artifact.get("semantic_delta", {}).get("operations", []):
        if op.get("op") not in DELTA_OPS:
            errors.append({"category": "invariant_failure", "message": "forbidden delta op"})
    return {"ok": not errors, "errors": errors}


def semantic_equivalence_key(goal_state: dict[str, Any]) -> dict[str, Any]:
    refs = {
        ref.get("ref_id"): {
            "entity_type": ref.get("entity_type"),
            "cardinality": ref.get("cardinality"),
            "mention": None if ref.get("entity_type") == "edge" else _semantic_mention(ref.get("mention")),
        }
        for ref in goal_state.get("references", [])
        if isinstance(ref, dict)
    }
    facts = []
    for fact in goal_state.get("desired_facts", []):
        if not isinstance(fact, dict):
            continue
        fact = canonicalize_desired_fact(fact)
        semantic_operands = _semantic_operands(fact.get("operands", {}), refs)
        semantic_operands = _canonical_semantic_operands(str(fact.get("predicate") or ""), semantic_operands)
        facts.append({"predicate": fact.get("predicate"), "fact_key": _semantic_fact_key(fact, semantic_operands), "operands": semantic_operands})
    constraints = [
        {"constraint_predicate": row.get("constraint_predicate"), "operands": _semantic_operands(row.get("operands", {}), refs)}
        for row in goal_state.get("constraints", [])
        if isinstance(row, dict)
    ]
    open_statements = [
        {
            "statement_type": canonical_open_statement_type(row.get("statement_type")),
            "content": row.get("content"),
            "related_refs": [_semantic_ref_for_id(ref_id, refs) for ref_id in _related_ref_ids(row)],
        }
        for row in goal_state.get("open_statements", [])
        if isinstance(row, dict)
    ]
    relation_facts = [
        _semantic_relation_fact(row, refs)
        for row in goal_state.get("relation_facts", [])
        if isinstance(row, dict)
        and isinstance(row.get("relation_id"), str)
        and not row.get("relation_id", "").startswith("rf:")
    ]
    result = {
        "references": sorted(refs.values(), key=canonical_json),
        "desired_facts": sorted(facts, key=canonical_json),
        "constraints": sorted(constraints, key=canonical_json),
        "open_statements": sorted(open_statements, key=canonical_json),
    }
    if relation_facts:
        result["relation_facts"] = sorted(relation_facts, key=canonical_json)
    return result


def _semantic_relation_fact(relation_fact: dict[str, Any], refs: dict[str, Any]) -> dict[str, Any]:
    normalized = normalize_relation_fact(relation_fact)
    return {
        "kind": normalized["kind"],
        "members": sorted(
            [
                {
                    "source": _semantic_ref_for_id(member["source_ref"], refs),
                    "target": _semantic_ref_for_id(member["target_ref"], refs),
                    "properties": member["properties"],
                }
                for member in normalized["members"]
            ],
            key=canonical_json,
        ),
        "group_semantics": normalized["group_semantics"],
    }


def _semantic_operands(operands: dict[str, Any], refs: dict[str, Any]) -> dict[str, Any]:
    normalized = {}
    for role, operand in operands.items():
        if isinstance(operand, dict) and operand.get("operand_type") == "reference":
            normalized[role] = {"operand_type": "reference", "reference": refs.get(operand.get("ref_id"), {"missing_ref": operand.get("ref_id")})}
        else:
            normalized[role] = operand
    return normalized


def _semantic_ref_for_id(ref_id: str, refs: dict[str, Any]) -> dict[str, Any]:
    return refs.get(ref_id, {"missing_ref": ref_id})


def _semantic_fact_key(fact: dict[str, Any], semantic_operands: dict[str, Any]) -> str:
    fact = canonicalize_desired_fact(fact)
    predicate = str(fact.get("predicate") or "")
    semantic_operands = _canonical_semantic_operands(predicate, semantic_operands)
    signature = PREDICATE_SIGNATURES.get(predicate)
    if not signature:
        return canonical_json({"predicate": predicate, "operands": semantic_operands})
    roles = [role for role in signature.get("normalization_policy", {}).get("fact_key", []) if role != "predicate"]
    return canonical_json({"predicate": predicate, "anchors": {role: semantic_operands.get(role) for role in roles}})


def _semantic_mention(mention: Any) -> dict[str, Any] | None:
    if not isinstance(mention, dict):
        return None
    text = mention.get("text")
    return {"text": text} if isinstance(text, str) else None


def _canonical_semantic_operands(predicate: str, operands: dict[str, Any]) -> dict[str, Any]:
    normalized = deepcopy(operands)
    if predicate != "property.equals":
        return normalized
    property_operand = normalized.get("property")
    if not isinstance(property_operand, dict) or property_operand.get("operand_type") != "literal":
        return normalized
    subject = normalized.get("subject")
    entity_type = None
    if isinstance(subject, dict):
        reference = subject.get("reference")
        if isinstance(reference, dict):
            entity_type = reference.get("entity_type")
    property_operand["value"] = canonical_property_name(entity_type, property_operand.get("value"))
    return normalized


def _canonicalize_goal_state(goal_state: dict[str, Any]) -> None:
    refs = {row.get("ref_id"): row for row in goal_state.get("references", []) if isinstance(row, dict)}
    facts = goal_state.get("desired_facts", [])
    for index, fact in enumerate(facts):
        if not isinstance(fact, dict):
            continue
        facts[index] = canonicalize_desired_fact(fact)
    for fact in goal_state.get("desired_facts", []):
        if not isinstance(fact, dict) or fact.get("predicate") != "property.equals":
            continue
        operands = fact.get("operands") if isinstance(fact.get("operands"), dict) else {}
        subject = operands.get("subject")
        property_operand = operands.get("property")
        if not isinstance(subject, dict) or not isinstance(property_operand, dict):
            continue
        if property_operand.get("operand_type") != "literal":
            continue
        ref = refs.get(subject.get("ref_id"))
        entity_type = ref.get("entity_type") if isinstance(ref, dict) else None
        property_operand["value"] = canonical_property_name(entity_type, property_operand.get("value"))
    existing_ref_ids = {str(ref_id) for ref_id in refs if isinstance(ref_id, str)}
    for row in goal_state.get("open_statements", []):
        if not isinstance(row, dict):
            continue
        row["related_refs"] = [ref_id for ref_id in _related_ref_ids(row) if ref_id in existing_ref_ids]


def _canonicalize_reference_entity_types_from_fact_roles(goal_state: dict[str, Any]) -> None:
    refs = {row.get("ref_id"): row for row in goal_state.get("references", []) if isinstance(row, dict)}
    for fact in goal_state.get("desired_facts", []):
        if not isinstance(fact, dict):
            continue
        signature = PREDICATE_SIGNATURES.get(str(fact.get("predicate") or ""))
        if not signature:
            continue
        operands = fact.get("operands") if isinstance(fact.get("operands"), dict) else {}
        role_types = signature.get("role_types", {})
        for role, allowed_types in role_types.items():
            if not isinstance(allowed_types, list) or len(allowed_types) != 1:
                continue
            operand = operands.get(role)
            if not isinstance(operand, dict) or operand.get("operand_type") != "reference":
                continue
            ref = refs.get(operand.get("ref_id"))
            if ref is None and isinstance(operand.get("ref_id"), str):
                ref = {
                    "ref_id": operand["ref_id"],
                    "entity_type": allowed_types[0],
                    "cardinality": "single",
                    "mention": None,
                    "provenance_refs": list(fact.get("provenance_refs", [])) if isinstance(fact.get("provenance_refs"), list) else [],
                }
                goal_state.setdefault("references", []).append(ref)
                refs[operand["ref_id"]] = ref
            if not isinstance(ref, dict):
                continue
            if ref.get("mention") is None or ref.get("entity_type") == "unknown":
                ref["entity_type"] = allowed_types[0]


def _retire_refined_open_statements(next_state: dict[str, Any], before_state: dict[str, Any], delta: dict[str, Any]) -> None:
    before_open_ids = {
        row.get("statement_id")
        for row in before_state.get("open_statements", [])
        if isinstance(row, dict) and isinstance(row.get("statement_id"), str)
    }
    if len(before_open_ids) != 1:
        return
    before_ref_ids = {
        row.get("ref_id")
        for row in before_state.get("references", [])
        if isinstance(row, dict) and isinstance(row.get("ref_id"), str)
    }
    if not before_ref_ids:
        return
    refined_ref_ids: set[str] = set()
    for op in delta.get("operations", []):
        if op.get("op") not in {"ADD", "CORRECT"} or op.get("element_kind") != "desired_fact":
            continue
        value = op.get("value") if isinstance(op.get("value"), dict) else {}
        refined_ref_ids.update(_reference_operand_ids(value.get("operands") if isinstance(value.get("operands"), dict) else {}))
    if not (before_ref_ids & refined_ref_ids):
        return
    next_state["open_statements"] = [
        row
        for row in next_state.get("open_statements", [])
        if not isinstance(row, dict) or row.get("statement_id") not in before_open_ids
    ]


def _reference_operand_ids(value: Any) -> set[str]:
    ids: set[str] = set()
    if isinstance(value, dict):
        if value.get("operand_type") == "reference" and isinstance(value.get("ref_id"), str):
            ids.add(value["ref_id"])
        for child in value.values():
            ids.update(_reference_operand_ids(child))
    elif isinstance(value, list):
        for child in value:
            ids.update(_reference_operand_ids(child))
    return ids


def _related_ref_ids(value: dict[str, Any]) -> set[str]:
    return {
        str(item)
        for item in value.get("related_refs", [])
        if isinstance(item, str) and item
    } if isinstance(value.get("related_refs"), list) else set()


def _spans_in_value(value: Any):
    if isinstance(value, dict):
        if {"utterance_id", "start", "end"} <= set(value):
            yield value
        for child in value.values():
            yield from _spans_in_value(child)
    elif isinstance(value, list):
        for child in value:
            yield from _spans_in_value(child)


def _add_element(goal_state: dict[str, Any], kind: str, value: dict[str, Any], evidence_bundle: list[dict[str, Any]]) -> None:
    if kind == "reference":
        goal_state.setdefault("references", []).append(_normalize_reference_element(value, evidence_bundle))
    elif kind == "desired_fact":
        goal_state.setdefault("desired_facts", []).append(_normalize_fact_element(value, evidence_bundle))
    elif kind == "constraint":
        goal_state.setdefault("constraints", []).append(_normalize_constraint_element(value, evidence_bundle))
    elif kind == "open_statement":
        goal_state.setdefault("open_statements", []).append(_normalize_open_statement(value, evidence_bundle))
    elif kind == "relation_fact":
        goal_state.setdefault("relation_facts", []).append(_normalize_relation_fact_element(value, evidence_bundle))


def _remove_element(goal_state: dict[str, Any], kind: str, element_id: str | None) -> None:
    field, id_key = _field_for_kind(kind)
    goal_state[field] = [row for row in goal_state.get(field, []) if not isinstance(row, dict) or row.get(id_key) != element_id]


def _renumber_elements(goal_state: dict[str, Any]) -> None:
    for index, row in enumerate(goal_state.get("references", []), start=1):
        if isinstance(row, dict) and not row.get("ref_id"):
            row["ref_id"] = f"r{index}"
    for index, row in enumerate(goal_state.get("desired_facts", []), start=1):
        if isinstance(row, dict) and not row.get("fact_id"):
            row["fact_id"] = f"f{index}"
        if isinstance(row, dict):
            row["fact_key"] = predicate_fact_key(row)
    for index, row in enumerate(goal_state.get("constraints", []), start=1):
        if isinstance(row, dict) and not row.get("constraint_id"):
            row["constraint_id"] = f"c{index}"
    for index, row in enumerate(goal_state.get("open_statements", []), start=1):
        if isinstance(row, dict) and not row.get("statement_id"):
            row["statement_id"] = f"s{index}"
    used_relation_ids = {
        row.get("relation_id")
        for row in goal_state.get("relation_facts", [])
        if isinstance(row, dict) and isinstance(row.get("relation_id"), str) and row.get("relation_id")
    }
    next_relation_index = 1
    for row in goal_state.get("relation_facts", []):
        if not isinstance(row, dict) or row.get("relation_id"):
            continue
        while f"rg{next_relation_index}" in used_relation_ids:
            next_relation_index += 1
        relation_id = f"rg{next_relation_index}"
        row["relation_id"] = relation_id
        used_relation_ids.add(relation_id)


def _field_for_kind(kind: str) -> tuple[str, str]:
    return {
        "reference": ("references", "ref_id"),
        "desired_fact": ("desired_facts", "fact_id"),
        "constraint": ("constraints", "constraint_id"),
        "open_statement": ("open_statements", "statement_id"),
        "relation_fact": ("relation_facts", "relation_id"),
    }[kind]


def _normalize_reference_element(value: dict[str, Any], evidence_bundle: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "ref_id": str(value.get("ref_id") or ""),
        "entity_type": str(value.get("entity_type") or "unknown"),
        "cardinality": str(value.get("cardinality") or "single"),
        "mention": deepcopy(value.get("mention")) if isinstance(value.get("mention"), dict) else None,
        "provenance_refs": _provenance_refs(value, evidence_bundle),
    }


def _normalize_fact_element(value: dict[str, Any], evidence_bundle: list[dict[str, Any]]) -> dict[str, Any]:
    predicate = str(value.get("predicate") or "")
    return canonicalize_desired_fact({
        "fact_id": str(value.get("fact_id") or ""),
        "predicate": predicate,
        "predicate_signature_id": str(value.get("predicate_signature_id") or predicate),
        "operands": deepcopy(value.get("operands")) if isinstance(value.get("operands"), dict) else {},
        "provenance_refs": _provenance_refs(value, evidence_bundle),
    })


def _normalize_constraint_element(value: dict[str, Any], evidence_bundle: list[dict[str, Any]]) -> dict[str, Any]:
    predicate = str(value.get("constraint_predicate") or "")
    return {
        "constraint_id": str(value.get("constraint_id") or ""),
        "constraint_predicate": predicate,
        "constraint_signature_id": str(value.get("constraint_signature_id") or predicate),
        "operands": deepcopy(value.get("operands")) if isinstance(value.get("operands"), dict) else {},
        "provenance_refs": _provenance_refs(value, evidence_bundle),
    }


def _normalize_open_statement(value: dict[str, Any], evidence_bundle: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "statement_id": str(value.get("statement_id") or ""),
        "statement_type": canonical_open_statement_type(value.get("statement_type")),
        "content": deepcopy(value.get("content")),
        "related_refs": _string_list(value.get("related_refs")),
        "provenance_refs": _provenance_refs(value, evidence_bundle),
    }


def _normalize_relation_fact_element(value: dict[str, Any], evidence_bundle: list[dict[str, Any]]) -> dict[str, Any]:
    normalized = normalize_relation_fact(value)
    provenance_refs = _provenance_refs(normalized, evidence_bundle)
    normalized["provenance_refs"] = provenance_refs
    for member in normalized["members"]:
        if not member.get("provenance_refs"):
            member["provenance_refs"] = list(provenance_refs)
    return normalized


def _normalize_relation_fact_proposal_element(value: dict[str, Any]) -> dict[str, Any]:
    normalized = _normalize_relation_fact_element(value, [])
    normalized["relation_id"] = ""
    return normalized


def _provenance_refs(value: dict[str, Any], evidence_bundle: list[dict[str, Any]]) -> list[str]:
    refs = _string_list(value.get("provenance_refs"))
    if refs:
        return refs
    return [evidence_bundle[0]["provenance_id"]] if evidence_bundle else []


def _validate_fact_value(fact: dict[str, Any]) -> list[dict[str, Any]]:
    errors = []
    predicate = fact.get("predicate")
    signature = PREDICATE_SIGNATURES.get(str(predicate or ""))
    if not signature:
        return [{"category": "semantic_schema_validation_failure", "message": "unknown predicate", "predicate": predicate}]
    operands = fact.get("operands") if isinstance(fact.get("operands"), dict) else {}
    if set(operands) != set(signature["role_names"]):
        errors.append({"category": "semantic_schema_validation_failure", "message": "predicate operands do not match signature", "predicate": predicate})
    for role, operand in operands.items():
        if operand is None:
            errors.append({"category": "semantic_schema_validation_failure", "message": "operand must be typed, not null", "role": role})
        elif isinstance(operand, dict) and operand.get("operand_type") == "reference" and not isinstance(operand.get("ref_id"), str):
            errors.append({"category": "semantic_schema_validation_failure", "message": "reference operand requires ref_id", "role": role})
        elif isinstance(operand, dict) and operand.get("operand_type") in {"literal", "explicit_empty"}:
            continue
        elif not isinstance(operand, dict):
            errors.append({"category": "semantic_schema_validation_failure", "message": "operand must be a typed object", "role": role})
    return errors


def _validate_relation_fact_value(relation_fact: dict[str, Any], *, allow_missing_relation_id: bool) -> list[dict[str, Any]]:
    value = normalize_relation_fact(relation_fact)
    if allow_missing_relation_id and not value.get("relation_id"):
        value["relation_id"] = "__proposal_relation__"
    return [
        {
            "category": "semantic_schema_validation_failure",
            "message": "invalid relation_fact",
            "error": error,
        }
        for error in validate_relation_fact(value)
    ]


def _validate_relation_fact_endpoint_refs(proposal: dict[str, Any], goal_state: dict[str, Any]) -> list[dict[str, Any]]:
    known_refs = {
        row.get("ref_id")
        for row in goal_state.get("references", [])
        if isinstance(row, dict) and isinstance(row.get("ref_id"), str) and row.get("ref_id")
    }
    known_refs.update(
        row.get("ref_id")
        for row in proposal.get("references", [])
        if isinstance(row, dict) and isinstance(row.get("ref_id"), str) and row.get("ref_id")
    )
    errors: list[dict[str, Any]] = []
    for relation_index, relation_fact in enumerate(proposal.get("relation_facts", [])):
        normalized = normalize_relation_fact(relation_fact)
        for member_index, member in enumerate(normalized["members"]):
            for field in ("source_ref", "target_ref"):
                ref_id = member.get(field)
                if ref_id and ref_id not in known_refs:
                    errors.append(
                        {
                            "category": "semantic_schema_validation_failure",
                            "message": "relation_fact member references unknown ref",
                            "relation_index": relation_index,
                            "member_index": member_index,
                            "field": field,
                            "ref_id": ref_id,
                        }
                    )
    return errors


def _validate_reference_value(ref: dict[str, Any]) -> list[dict[str, Any]]:
    errors = []
    if ref.get("entity_type") not in REFERENCE_ENTITY_TYPES:
        errors.append({"category": "semantic_schema_validation_failure", "message": "invalid reference entity_type"})
    if ref.get("cardinality") not in REFERENCE_CARDINALITIES:
        errors.append({"category": "semantic_schema_validation_failure", "message": "invalid reference cardinality"})
    for forbidden in ("semantic_role", "binding", "candidate_id", "node_id", "edge_id"):
        if forbidden in ref:
            errors.append({"category": "semantic_schema_validation_failure", "message": "reference leaks runtime or graph binding", "field": forbidden})
    return errors


def _element_ids(goal_state: dict[str, Any]) -> set[str]:
    ids: set[str] = set()
    for field, key in (("references", "ref_id"), ("desired_facts", "fact_id"), ("constraints", "constraint_id"), ("open_statements", "statement_id"), ("relation_facts", "relation_id")):
        for row in goal_state.get(field, []):
            if isinstance(row, dict) and isinstance(row.get(key), str):
                ids.add(row[key])
    return ids


def _base_evidence(raw_utterance: str, trusted_evidence: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    bundle = [
        {
            "provenance_id": "p1",
            "source_type": "user_utterance",
            "source_ref": "u1",
            "evidence_span": {"utterance_id": "u1", "start": 0, "end": len(raw_utterance), "text": raw_utterance} if raw_utterance else None,
            "graph_version": None,
        }
    ]
    for index, row in enumerate(trusted_evidence or [], start=2):
        if not isinstance(row, dict):
            continue
        item = deepcopy(row)
        item.setdefault("provenance_id", f"p{index}")
        bundle.append(item)
    return bundle


def _span_list(value: Any) -> list[dict[str, Any]]:
    rows = value if isinstance(value, list) else []
    result = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        if isinstance(row.get("start"), int) and isinstance(row.get("end"), int):
            span = {"utterance_id": str(row.get("utterance_id") or "u1"), "start": row["start"], "end": row["end"]}
            if isinstance(row.get("text"), str):
                span["text"] = row["text"]
            result.append(span)
    return result


def _span_valid(span: dict[str, Any], utterance: str) -> bool:
    return span.get("utterance_id") == "u1" and isinstance(span.get("start"), int) and isinstance(span.get("end"), int) and 0 <= span["start"] <= span["end"] <= len(utterance)


def _realign_spans_in_place(value: Any, utterance: str) -> None:
    if isinstance(value, dict):
        mention = value.get("mention")
        if isinstance(mention, dict) and isinstance(mention.get("text"), str) and isinstance(mention.get("span"), dict):
            _realign_span_to_text(mention["span"], mention["text"], utterance)
        if isinstance(value.get("text"), str) and isinstance(value.get("start"), int) and isinstance(value.get("end"), int):
            _realign_span_to_text(value, value["text"], utterance)
        for child in value.values():
            _realign_spans_in_place(child, utterance)
    elif isinstance(value, list):
        for child in value:
            _realign_spans_in_place(child, utterance)


def _realign_span_to_text(span: dict[str, Any], text: str, utterance: str) -> None:
    if _span_valid(span, utterance):
        return
    if not text:
        return
    starts: list[int] = []
    offset = 0
    while True:
        index = utterance.find(text, offset)
        if index < 0:
            break
        starts.append(index)
        offset = index + 1
    if len(starts) != 1:
        return
    span["utterance_id"] = "u1"
    span["start"] = starts[0]
    span["end"] = starts[0] + len(text)


def _string_list(value: Any) -> list[str]:
    return [str(item) for item in value] if isinstance(value, list) else []


def _walk_dicts(value: Any):
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from _walk_dicts(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_dicts(child)


def _failed(artifact: dict[str, Any], category: str, message: str) -> Phase1BRun:
    artifact["ok"] = False
    artifact["errors"].append({"category": category, "message": message})
    artifact.setdefault("after_goal_state", artifact.get("before_goal_state"))
    return Phase1BRun(ok=False, artifact=artifact, error=category)
