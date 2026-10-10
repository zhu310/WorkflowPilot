from __future__ import annotations

import json
import os
from copy import deepcopy
from typing import Any

from .canonical_goal_state import build_canonical_shadow_artifact
from .constraint_planner import plan_constraints
from .goal_lowering import lower_grounded_goal_to_workflow_ir, partial_execution_contract_for_grounded_goal
from .mvp23 import TransactionResult
from .mvp22 import execute_deterministic
from .phase1b_shadow import empty_goal_state, merge_goal_state, plan_semantic_delta
from .relation_fact_runtime import attach_managed_relation_grouper
from .semantic_goal import (
    ground_canonical_relation_facts,
    ground_semantic_goal_spec,
    normalize_semantic_goal_spec,
    validate_grounded_goal,
    validate_semantic_goal_spec,
)
from .workflow_ir_core import (
    compile_grounded_semantic_spec,
    graph_is_valid,
    ground_understand_semantic_spec,
    normalize_request_context,
    normalize_understand_semantic_spec,
)
from .workflow_ir_runtime import SEMANTIC_GOAL_UNDERSTANDING_SYSTEM, _complete_json


LEGACY_WORKFLOW_IR_COMPAT_ENV = "SEMANTIC_LEGACY_WORKFLOW_IR_COMPAT"
CANONICAL_SHADOW_ENV = "SEMANTIC_CANONICAL_SHADOW"


def _legacy_workflow_ir_compat_enabled() -> bool:
    return os.getenv(LEGACY_WORKFLOW_IR_COMPAT_ENV, "").strip().lower() in {"1", "true", "yes"}


def _canonical_shadow_enabled() -> bool:
    return os.getenv(CANONICAL_SHADOW_ENV, "").strip().lower() in {"1", "true", "yes"}


def _semantic_goal_contract_violation(field: str) -> dict[str, Any]:
    return {
        "kind": "semantic_goal_contract_violation",
        "message": f"Semantic Goal output must not include {field}; Workflow IR output is only allowed in explicit legacy compatibility mode.",
        "severity": "error",
        "location": field,
    }


def _attach_canonical_shadow_trace(trace: dict[str, Any], normalized_goal: dict[str, Any], instruction: str) -> None:
    if not _canonical_shadow_enabled():
        return
    try:
        trace["canonical_shadow"] = build_canonical_shadow_artifact(normalized_goal, instruction=instruction)
    except Exception as exc:
        trace["canonical_shadow"] = {
            "enabled": True,
            "ok": False,
            "failure_stage": "canonical_shadow",
            "error": str(exc),
        }


def _canonical_artifact_for_semantic_goal(normalized_goal: dict[str, Any], instruction: str) -> dict[str, Any]:
    try:
        return build_canonical_shadow_artifact(normalized_goal, instruction=instruction)
    except Exception as exc:  # Managed RelationFact support must be fail-open in semantic_shadow.
        return {
            "enabled": True,
            "ok": False,
            "failure_stage": "canonical_goal_state",
            "error": str(exc),
        }


def _semantic_proposal_from_canonical_goal_state(goal_state: dict[str, Any]) -> dict[str, Any]:
    return {
        "references": deepcopy(goal_state.get("references", [])) if isinstance(goal_state.get("references"), list) else [],
        "desired_facts": deepcopy(goal_state.get("desired_facts", [])) if isinstance(goal_state.get("desired_facts"), list) else [],
        "constraints": deepcopy(goal_state.get("constraints", [])) if isinstance(goal_state.get("constraints"), list) else [],
        "open_statements": deepcopy(goal_state.get("open_statements", [])) if isinstance(goal_state.get("open_statements"), list) else [],
        "relation_facts": [],
    }


def _object_phrase(item: dict[str, Any]) -> str:
    if item.get("type") == "new_entity":
        return str(item.get("label") or "").strip()
    return str(item.get("reference") or item.get("label") or "").strip()


def _object_entity_type(item: dict[str, Any]) -> str:
    item_type = str(item.get("type") or "").strip()
    if item_type == "edge_reference":
        return "edge"
    if item_type in {"existing_reference", "new_entity"}:
        return "node"
    return "unknown"


def _reference_text(ref: dict[str, Any]) -> str:
    mention = ref.get("mention") if isinstance(ref.get("mention"), dict) else {}
    return str(mention.get("text") or "").strip()


def _semantic_ref_to_v2_object_ids(normalized_goal: dict[str, Any], goal_state: dict[str, Any]) -> dict[str, str]:
    object_keys: dict[tuple[str, str], list[str]] = {}
    for goal in normalized_goal.get("semantic_goals_v2", []) if isinstance(normalized_goal.get("semantic_goals_v2"), list) else []:
        if not isinstance(goal, dict):
            continue
        for item in goal.get("objects", []) if isinstance(goal.get("objects"), list) else []:
            if not isinstance(item, dict) or not isinstance(item.get("id"), str) or not item.get("id"):
                continue
            key = (_object_entity_type(item), _object_phrase(item))
            if not key[1]:
                continue
            object_keys.setdefault(key, []).append(item["id"])

    mapping: dict[str, str] = {}
    for ref in goal_state.get("references", []) if isinstance(goal_state.get("references"), list) else []:
        if not isinstance(ref, dict) or not isinstance(ref.get("ref_id"), str):
            continue
        key = (str(ref.get("entity_type") or "unknown"), _reference_text(ref))
        object_ids = object_keys.get(key, [])
        if len(object_ids) == 1:
            mapping[ref["ref_id"]] = object_ids[0]
    return mapping


def _attach_v2_object_ids_to_grounded_relation_facts(relation_facts: list[dict[str, Any]], ref_to_object_id: dict[str, str]) -> None:
    for relation_fact in relation_facts:
        if not isinstance(relation_fact, dict):
            continue
        for member in relation_fact.get("members", []) if isinstance(relation_fact.get("members"), list) else []:
            if not isinstance(member, dict):
                continue
            for field in ("source_binding", "target_binding"):
                binding = member.get(field)
                if not isinstance(binding, dict):
                    continue
                ref_id = binding.get("ref_id")
                if binding.get("needs_creation") is True and isinstance(ref_id, str) and ref_id in ref_to_object_id:
                    binding["v2_object_id"] = ref_to_object_id[ref_id]


def _endpoint_key(binding: Any) -> tuple[str, str] | None:
    if not isinstance(binding, dict):
        return None
    if binding.get("binding_type") == "node" and isinstance(binding.get("id"), str) and binding["id"]:
        return ("node", binding["id"])
    if binding.get("needs_creation") is True:
        v2_object_id = binding.get("v2_object_id")
        if isinstance(v2_object_id, str) and v2_object_id.strip():
            return ("new", f"v2:{v2_object_id.strip()}")
        ref_id = binding.get("ref_id")
        if isinstance(ref_id, str) and ref_id.strip():
            return ("new", f"ref:{ref_id.strip()}")
    return None


def _covered_relation_fact_pairs(relation_facts: list[dict[str, Any]]) -> set[tuple[tuple[str, str], tuple[str, str]]]:
    covered: set[tuple[tuple[str, str], tuple[str, str]]] = set()
    for relation_fact in relation_facts:
        if not isinstance(relation_fact, dict):
            continue
        relation_id = relation_fact.get("relation_id")
        if not (isinstance(relation_id, str) and relation_id and not relation_id.startswith("rf:")):
            continue
        if relation_fact.get("kind") != "precedence":
            continue
        members = relation_fact.get("members") if isinstance(relation_fact.get("members"), list) else []
        if len(members) < 2:
            continue
        for member in members:
            if not isinstance(member, dict):
                continue
            source_key = _endpoint_key(member.get("source_binding"))
            target_key = _endpoint_key(member.get("target_binding"))
            if source_key is not None and target_key is not None:
                covered.add((source_key, target_key))
    return covered


def _suppress_covered_legacy_relation_goals(grounded_goal: dict[str, Any], relation_facts: list[dict[str, Any]]) -> dict[str, Any]:
    covered_pairs = _covered_relation_fact_pairs(relation_facts)
    if not covered_pairs:
        return grounded_goal
    next_goal = deepcopy(grounded_goal)
    kept: list[dict[str, Any]] = []
    suppressed: list[dict[str, Any]] = []
    for goal in next_goal.get("semantic_goals", []) if isinstance(next_goal.get("semantic_goals"), list) else []:
        if not isinstance(goal, dict):
            kept.append(goal)
            continue
        bindings = {
            binding.get("role"): binding
            for binding in goal.get("bindings", [])
            if isinstance(binding, dict) and isinstance(binding.get("role"), str)
        }
        pair = (_endpoint_key(bindings.get("subject")), _endpoint_key(bindings.get("anchor")))
        if goal.get("kind") == "ordering" and goal.get("relation") == "before" and pair[0] is not None and pair[1] is not None and pair in covered_pairs:
            suppressed.append({"source_subgoal_id": goal.get("source_subgoal_id"), "endpoint_pair": pair})
            continue
        kept.append(goal)
    next_goal["semantic_goals"] = kept
    next_goal["suppressed_legacy_relation_goals"] = suppressed
    return next_goal


def _attach_managed_relation_facts_to_grounded_goal(
    trace: dict[str, Any],
    normalized_goal: dict[str, Any],
    grounded_goal: dict[str, Any],
    before: dict[str, Any],
    normalized_selection: dict[str, Any],
    normalized_context: dict[str, Any],
    instruction: str,
) -> dict[str, Any]:
    canonical_artifact = _canonical_artifact_for_semantic_goal(normalized_goal, instruction)
    trace["managed_relation_runtime"] = {
        "enabled": True,
        "semantic_execution_goalstate_persistence_gap": True,
        "reason": "semantic_shadow has no persisted production GoalState; this entrypoint uses single-turn R3B lifecycle only.",
    }
    if not canonical_artifact.get("ok"):
        trace["managed_relation_runtime"]["ok"] = False
        trace["managed_relation_runtime"]["failure_stage"] = "canonical_goal_state"
        trace["managed_relation_runtime"]["error"] = canonical_artifact.get("error")
        return grounded_goal

    before_goal_state = empty_goal_state()
    semantic_proposal = _semantic_proposal_from_canonical_goal_state(canonical_artifact.get("goal_state", {}))
    relation_fact_proposals = attach_managed_relation_grouper(
        trace,
        _complete_json,
        instruction,
        semantic_proposal,
        planning_goal_state=before_goal_state,
    )
    trace["managed_relation_runtime"]["relation_fact_proposal_count"] = len(relation_fact_proposals)
    if not relation_fact_proposals:
        trace["managed_relation_runtime"]["ok"] = True
        trace["managed_relation_runtime"]["relation_facts_attached"] = False
        return grounded_goal

    semantic_proposal["relation_facts"] = relation_fact_proposals
    planned = plan_semantic_delta(before_goal_state, {"relation": "CONTINUE"}, semantic_proposal)
    trace["managed_relation_runtime"]["semantic_delta"] = planned.get("semantic_delta")
    if planned.get("errors"):
        trace["managed_relation_runtime"]["ok"] = False
        trace["managed_relation_runtime"]["failure_stage"] = "semantic_delta"
        trace["managed_relation_runtime"]["errors"] = planned.get("errors")
        return grounded_goal

    merged = merge_goal_state(before_goal_state, planned.get("semantic_delta", {}), canonical_artifact.get("evidence_bundle", []))
    trace["managed_relation_runtime"]["merge"] = {"ok": merged.get("ok"), "errors": merged.get("errors", [])}
    if merged.get("ok") is not True:
        trace["managed_relation_runtime"]["ok"] = False
        trace["managed_relation_runtime"]["failure_stage"] = "goal_state_merge"
        return grounded_goal

    goal_state = merged.get("goal_state") if isinstance(merged.get("goal_state"), dict) else {}
    grounded_relations = ground_canonical_relation_facts(goal_state, before, normalized_selection, normalized_context)
    ref_to_object_id = _semantic_ref_to_v2_object_ids(normalized_goal, goal_state)
    relation_facts = grounded_relations.get("relation_facts", []) if isinstance(grounded_relations.get("relation_facts"), list) else []
    _attach_v2_object_ids_to_grounded_relation_facts(relation_facts, ref_to_object_id)
    trace["managed_relation_runtime"]["after_goal_state"] = goal_state
    trace["managed_relation_runtime"]["grounded_relation_facts"] = relation_facts
    trace["managed_relation_runtime"]["ref_to_v2_object_id"] = ref_to_object_id

    grounding = grounded_relations.get("grounding") if isinstance(grounded_relations.get("grounding"), dict) else {}
    relation_issues = grounding.get("issues") if isinstance(grounding.get("issues"), list) else []
    next_goal = _suppress_covered_legacy_relation_goals(grounded_goal, relation_facts)
    if relation_issues:
        next_grounding = deepcopy(next_goal.get("grounding")) if isinstance(next_goal.get("grounding"), dict) else {"issues": []}
        next_grounding.setdefault("issues", [])
        if isinstance(next_grounding["issues"], list):
            next_grounding["issues"].extend(deepcopy(relation_issues))
        next_goal["grounding"] = next_grounding
    next_goal["relation_facts"] = relation_facts
    trace["managed_relation_runtime"]["ok"] = True
    trace["managed_relation_runtime"]["relation_facts_attached"] = True
    trace["managed_relation_runtime"]["suppressed_legacy_relation_goals"] = next_goal.get("suppressed_legacy_relation_goals", [])
    return next_goal


def graph_diff(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    before_nodes = {node["id"]: node for node in before.get("nodes", []) if isinstance(node, dict) and isinstance(node.get("id"), str)}
    after_nodes = {node["id"]: node for node in after.get("nodes", []) if isinstance(node, dict) and isinstance(node.get("id"), str)}
    before_edges = {edge["id"]: edge for edge in before.get("edges", []) if isinstance(edge, dict) and isinstance(edge.get("id"), str)}
    after_edges = {edge["id"]: edge for edge in after.get("edges", []) if isinstance(edge, dict) and isinstance(edge.get("id"), str)}
    changed_nodes = []
    changed_edges = []
    for node_id in sorted(set(before_nodes) & set(after_nodes)):
        changes = {
            key: {"before": before_nodes[node_id].get(key), "after": after_nodes[node_id].get(key)}
            for key in sorted(set(before_nodes[node_id]) | set(after_nodes[node_id]))
            if before_nodes[node_id].get(key) != after_nodes[node_id].get(key)
        }
        if changes:
            changed_nodes.append({"id": node_id, "changes": changes})
    for edge_id in sorted(set(before_edges) & set(after_edges)):
        changes = {
            key: {"before": before_edges[edge_id].get(key), "after": after_edges[edge_id].get(key)}
            for key in sorted(set(before_edges[edge_id]) | set(after_edges[edge_id]))
            if before_edges[edge_id].get(key) != after_edges[edge_id].get(key)
        }
        if changes:
            changed_edges.append({"id": edge_id, "changes": changes})
    return {
        "added_nodes": [after_nodes[node_id] for node_id in sorted(set(after_nodes) - set(before_nodes))],
        "removed_nodes": [before_nodes[node_id] for node_id in sorted(set(before_nodes) - set(after_nodes))],
        "changed_nodes": changed_nodes,
        "added_edges": [after_edges[edge_id] for edge_id in sorted(set(after_edges) - set(before_edges))],
        "removed_edges": [before_edges[edge_id] for edge_id in sorted(set(before_edges) - set(after_edges))],
        "changed_edges": changed_edges,
    }


def _graph_summary(graph: dict[str, Any]) -> dict[str, Any]:
    return {
        "node_count": len(graph.get("nodes", [])) if isinstance(graph.get("nodes"), list) else 0,
        "edge_count": len(graph.get("edges", [])) if isinstance(graph.get("edges"), list) else 0,
        "node_ids": [node.get("id") for node in graph.get("nodes", []) if isinstance(node, dict)],
        "edge_ids": [edge.get("id") for edge in graph.get("edges", []) if isinstance(edge, dict)],
    }


def _semantic_refinement_evidence_text(instruction: str, request_context: dict[str, Any]) -> str:
    phrases = [instruction]
    refinement = request_context.get("semantic_refinement") if isinstance(request_context.get("semantic_refinement"), dict) else {}
    partial = refinement.get("partial_semantic_state") if isinstance(refinement.get("partial_semantic_state"), dict) else {}
    semantic_goal = partial.get("semantic_goal") if isinstance(partial.get("semantic_goal"), dict) else {}
    goals = semantic_goal.get("semantic_goals_v2") if isinstance(semantic_goal.get("semantic_goals_v2"), list) else []
    for goal in goals:
        if not isinstance(goal, dict):
            continue
        for item in goal.get("objects", []) if isinstance(goal.get("objects"), list) else []:
            if not isinstance(item, dict):
                continue
            phrase = item.get("label") if item.get("type") == "new_entity" else item.get("reference")
            if isinstance(phrase, str) and phrase.strip():
                phrases.append(phrase)
    return " ".join(phrases)


def run_semantic_shadow(
    graph: dict[str, Any],
    instruction: str,
    selection: dict[str, Any] | None = None,
    request_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    before = deepcopy(graph)
    normalized_selection = selection if isinstance(selection, dict) else {"node_ids": [], "edge_ids": []}
    normalized_context = normalize_request_context(normalized_selection, request_context)
    trace: dict[str, Any] = {
        "enabled": True,
        "stage": "start",
        "instruction": instruction,
        "selection": normalized_selection,
        "request_context": normalized_context,
        "before_graph": before,
        "graph_context": _graph_summary(before),
        "semantic_goal": None,
        "grounded_goal": None,
        "goal_validation": None,
        "lowering_result": None,
        "partial_execution": None,
        "semantic_spec": None,
        "grounded_spec": None,
        "validation": None,
        "compiler_input": None,
        "compiled": None,
        "constraints": None,
        "planner": None,
        "executor": None,
        "final_shadow_graph": None,
        "graph_diff": None,
    }
    if not graph_is_valid(before):
        trace.update({"stage": "input", "failure_stage": "input", "failure_reason": "invalid graph"})
        return trace

    payload = {
        "instruction": instruction,
        "graph": before,
        "selection": normalized_selection,
        "request_context": normalized_context,
        "conversation_history": normalized_context.get("conversation_history", []),
    }
    trace["stage"] = "understand"
    semantic_goal_override = normalized_context.get("semantic_goal_override") if isinstance(normalized_context.get("semantic_goal_override"), dict) else None
    if isinstance(semantic_goal_override, dict):
        understand_run = type(
            "SemanticGoalOverride",
            (),
            {
                "raw_response": "",
                "parsed_json": deepcopy(semantic_goal_override),
                "parse_error": None,
                "normalized_from_fence": False,
            },
        )()
    else:
        understand_run = _complete_json(SEMANTIC_GOAL_UNDERSTANDING_SYSTEM, payload)
    trace["semantic_goal"] = {
        "raw_response": understand_run.raw_response,
        "parsed": understand_run.parsed_json,
        "parse_error": understand_run.parse_error,
        "normalized_from_fence": understand_run.normalized_from_fence,
    }
    if understand_run.parsed_json is None:
        trace.update({"failure_stage": "understand", "failure_reason": understand_run.parse_error or "semantic goal parse failed"})
        return trace

    if isinstance(understand_run.parsed_json.get("workflow_ir"), dict):
        if not _legacy_workflow_ir_compat_enabled():
            issue = _semantic_goal_contract_violation("workflow_ir")
            trace["validation"] = {"valid": False, "deterministic_ready": False, "issues": [issue]}
            trace.update({"failure_stage": "validation", "failure_reason": [issue]})
            return trace
        semantic_spec_input = understand_run.parsed_json
        trace["semantic_spec"] = {
            "raw_response": understand_run.raw_response,
            "parsed": understand_run.parsed_json,
            "parse_error": understand_run.parse_error,
            "normalized_from_fence": understand_run.normalized_from_fence,
        }
    else:
        normalized_goal = normalize_semantic_goal_spec(understand_run.parsed_json)
        evidence_text = _semantic_refinement_evidence_text(instruction, normalized_context)
        goal_validation = validate_semantic_goal_spec(understand_run.parsed_json, source_instruction=evidence_text)
        trace["semantic_goal"]["normalized"] = normalized_goal
        _attach_canonical_shadow_trace(trace, normalized_goal, instruction)
        trace["goal_validation"] = goal_validation
        if not goal_validation.get("ready_for_grounding"):
            trace["partial_execution"] = partial_execution_contract_for_grounded_goal(normalized_goal)
            trace["validation"] = {"valid": goal_validation.get("valid", False), "deterministic_ready": False, "issues": goal_validation.get("issues", [])}
            trace.update({"failure_stage": "validation", "failure_reason": goal_validation.get("issues", [])})
            return trace

        trace["stage"] = "goal_ground"
        grounded_goal = ground_semantic_goal_spec(normalized_goal, before, normalized_selection, normalized_context)
        grounded_goal = _attach_managed_relation_facts_to_grounded_goal(
            trace,
            normalized_goal,
            grounded_goal,
            before,
            normalized_selection,
            normalized_context,
            instruction,
        )
        trace["grounded_goal"] = grounded_goal
        grounded_goal_validation = validate_grounded_goal(grounded_goal)
        trace["goal_validation"] = grounded_goal_validation
        if not grounded_goal_validation.get("lowering_ready"):
            trace["partial_execution"] = partial_execution_contract_for_grounded_goal(grounded_goal)
            trace["validation"] = {"valid": grounded_goal_validation.get("valid", False), "deterministic_ready": False, "issues": grounded_goal_validation.get("issues", [])}
            trace.update({"failure_stage": "validation", "failure_reason": grounded_goal_validation.get("issues", [])})
            return trace

        trace["stage"] = "lowering"
        if normalized_context.get("allow_route_same_target_confirmation") is True:
            grounded_goal["_allow_route_same_target_confirmation"] = True
        lowering = lower_grounded_goal_to_workflow_ir(grounded_goal, before)
        trace["lowering_result"] = lowering
        trace["partial_execution"] = lowering.get("partial_execution")
        if lowering.get("ok") is not True:
            issues = lowering.get("issues", [])
            trace["validation"] = {"valid": False, "deterministic_ready": False, "issues": issues}
            trace.update({"failure_stage": "validation", "failure_reason": issues})
            return trace
        semantic_spec_input = lowering.get("semantic_spec") if isinstance(lowering.get("semantic_spec"), dict) else {}
        trace["semantic_spec"] = {
            "raw_response": understand_run.raw_response,
            "parsed": semantic_spec_input,
            "parse_error": None,
            "normalized_from_fence": understand_run.normalized_from_fence,
        }

    normalized_spec = normalize_understand_semantic_spec(semantic_spec_input)
    trace["semantic_spec"]["normalized"] = normalized_spec

    trace["stage"] = "ground"
    grounded = ground_understand_semantic_spec(normalized_spec, before, normalized_selection)
    trace["grounded_spec"] = grounded

    trace["stage"] = "validation"
    shadow = compile_grounded_semantic_spec(grounded, before)
    trace["validation"] = shadow.get("validation")
    trace["compiler_input"] = {"workflow_ir": grounded.get("workflow_ir")}
    if shadow.get("stage") == "validation":
        trace.update({"failure_stage": "validation", "failure_reason": shadow.get("issues", [])})
        return trace

    trace["stage"] = "compile"
    trace["compiled"] = shadow.get("compiled")
    trace["constraints"] = shadow.get("constraints")
    if shadow.get("stage") == "compile":
        trace.update({"failure_stage": "compile", "failure_reason": shadow.get("issues", [])})
        return trace

    trace["stage"] = "planner"
    constraints = shadow.get("constraints")
    planner = plan_constraints(before, constraints if isinstance(constraints, dict) else {})
    trace["planner"] = planner
    if not planner.get("supported"):
        trace.update({"failure_stage": "planner", "failure_reason": planner.get("reason")})
        return trace

    trace["stage"] = "executor"
    execution = execute_deterministic(
        before,
        {"core_node_ids": [], "core_edge_ids": []},
        {"incoming": [], "outgoing": []},
        planner.get("steps", []),
        planner.get("mutation_authorization"),
    )
    trace["executor"] = {
        "steps": planner.get("steps", []),
        "mutation_authorization": planner.get("mutation_authorization", []),
        "error": execution.error,
        "applied_steps": execution.applied_steps,
    }
    if execution.graph is None or execution.error:
        trace.update({"failure_stage": "executor", "failure_reason": execution.error})
        return trace
    trace["final_shadow_graph"] = execution.graph
    trace["graph_diff"] = graph_diff(before, execution.graph)
    trace["stage"] = "success"
    return trace


def _validation_clarification_message(trace: dict[str, Any]) -> str | None:
    grounded = trace.get("grounded_spec")
    if isinstance(grounded, dict):
        clarification = grounded.get("clarification")
        if isinstance(clarification, dict):
            question = clarification.get("question")
            if isinstance(question, str) and question.strip():
                return question
    semantic_goal = trace.get("semantic_goal")
    normalized = semantic_goal.get("normalized") if isinstance(semantic_goal, dict) else None
    if isinstance(normalized, dict):
        clarification = normalized.get("clarification")
        if isinstance(clarification, dict):
            question = clarification.get("question")
            if isinstance(question, str) and question.strip():
                return question
    issues = trace.get("validation", {}).get("issues") if isinstance(trace.get("validation"), dict) else trace.get("failure_reason")
    if isinstance(issues, list):
        for issue in issues:
            if isinstance(issue, dict) and issue.get("message"):
                return str(issue["message"])
    return None


def _partial_execution_message(trace: dict[str, Any]) -> str | None:
    partial = trace.get("partial_execution")
    if not isinstance(partial, dict):
        return None
    subgoals = partial.get("subgoals")
    if not isinstance(subgoals, list):
        return None
    executed = [row for row in subgoals if isinstance(row, dict) and row.get("status") == "executed"]
    incomplete = [row for row in subgoals if isinstance(row, dict) and row.get("status") != "executed"]
    if not incomplete:
        return None
    done = "; ".join(str(row.get("description", row.get("id", ""))) for row in executed) or "none"
    not_done = "; ".join(
        f"{row.get('description', row.get('id', ''))}: {row.get('status')} ({row.get('reason', '')})"
        for row in incomplete
    )
    return f"Executed subgoals: {done}. Not completed: {not_done}."


def _diff_has_changes(diff: dict[str, Any] | None) -> bool:
    if not isinstance(diff, dict):
        return False
    return any(
        bool(diff.get(key))
        for key in ("added_nodes", "removed_nodes", "changed_nodes", "added_edges", "removed_edges", "changed_edges")
    )


def semantic_trace_to_transaction_result(trace: dict[str, Any]) -> TransactionResult:
    normalized_spec = None
    semantic_spec = trace.get("semantic_spec")
    if isinstance(semantic_spec, dict):
        normalized_spec = semantic_spec.get("normalized")
    trace_specification = {
        "pipeline": "semantic",
        "trace": trace,
        "semantic_goal": trace.get("semantic_goal"),
        "grounded_goal": trace.get("grounded_goal"),
        "goal_validation": trace.get("goal_validation"),
        "lowering_result": trace.get("lowering_result"),
        "partial_execution": trace.get("partial_execution"),
        "semantic_spec": normalized_spec,
        "grounded_spec": trace.get("grounded_spec"),
        "validation": trace.get("validation"),
        "compiled": trace.get("compiled"),
        "constraints": trace.get("constraints"),
        "planner": trace.get("planner"),
    }
    if trace.get("stage") == "success":
        executor = trace.get("executor") if isinstance(trace.get("executor"), dict) else {}
        partial_execution = trace.get("partial_execution") if isinstance(trace.get("partial_execution"), dict) else {}
        applied_steps = executor.get("applied_steps") if isinstance(executor.get("applied_steps"), list) else []
        if not _diff_has_changes(trace.get("graph_diff")) and not applied_steps:
            trace_specification["partial_execution"] = {
                **partial_execution,
                "final_status": "unsupported",
            }
            return TransactionResult(
                status="unsupported",
                interpretation=normalized_spec if isinstance(normalized_spec, dict) else None,
                boundary=None,
                steps=[],
                final_graph=None,
                error="no graph change was produced by the executable plan",
                normalized_plan_from_fence=False,
                planning_path="semantic",
                target={"selection": trace.get("selection", {})},
                graph_context=trace.get("graph_context"),
                specification=trace_specification,
                mutation_authorization=None,
                clarification_message=None,
            )
        result_status = "partial_success" if partial_execution.get("final_status") == "partial_success" else "success"
        return TransactionResult(
            status=result_status,
            interpretation=normalized_spec if isinstance(normalized_spec, dict) else None,
            boundary=None,
            steps=list(executor.get("steps", [])),
            final_graph=trace.get("final_shadow_graph") if isinstance(trace.get("final_shadow_graph"), dict) else None,
            error=None,
            normalized_plan_from_fence=False,
            planning_path="semantic",
            target={"selection": trace.get("selection", {})},
            graph_context=trace.get("graph_context"),
            specification=trace_specification,
            mutation_authorization=executor.get("mutation_authorization") if isinstance(executor.get("mutation_authorization"), list) else None,
            clarification_message=_partial_execution_message(trace) if result_status == "partial_success" else None,
        )

    failure_stage = str(trace.get("failure_stage") or trace.get("stage") or "semantic")
    failure_reason = trace.get("failure_reason")
    error = failure_reason if isinstance(failure_reason, str) else json.dumps(failure_reason, ensure_ascii=False)
    status = {
        "understand": "interpretation_error",
        "ground": "interpretation_error",
        "validation": "unsupported",
        "compile": "unsupported",
        "planner": "planning_error",
        "executor": "execution_error",
        "input": "execution_error",
    }.get(failure_stage, "unsupported")

    validation = trace.get("validation")
    issues = validation.get("issues", []) if isinstance(validation, dict) else []
    if failure_stage == "validation" and isinstance(issues, list):
        issue_kinds = {issue.get("kind") for issue in issues if isinstance(issue, dict)}
        if issue_kinds & {
            "clarification_needed",
            "partial_spec",
            "unresolved_semantic_goals",
            "ambiguous_node_reference",
            "missing_node_reference",
            "ambiguous_edge_reference",
            "missing_edge_reference",
            "stale_node_reference",
            "stale_edge_reference",
            "route_same_source_target_confirmation",
        }:
            status = "needs_clarification"
    partial_execution = trace.get("partial_execution") if isinstance(trace.get("partial_execution"), dict) else {}
    clarification_issue = bool(
        issue_kinds
        & {
            "clarification_needed",
            "partial_spec",
            "unresolved_semantic_goals",
            "ambiguous_node_reference",
            "missing_node_reference",
            "ambiguous_edge_reference",
            "missing_edge_reference",
            "stale_node_reference",
            "stale_edge_reference",
            "route_same_source_target_confirmation",
        }
    )
    if partial_execution.get("final_status") == "needs_clarification":
        status = "needs_clarification"
    elif partial_execution.get("final_status") == "unsupported" and not clarification_issue:
        status = partial_execution["final_status"]

    return TransactionResult(
        status=status,
        interpretation=normalized_spec if isinstance(normalized_spec, dict) else None,
        boundary=None,
        steps=[],
        final_graph=None,
        error=error,
        normalized_plan_from_fence=False,
        planning_path="semantic",
        target={"selection": trace.get("selection", {})},
        graph_context=trace.get("graph_context"),
        specification=trace_specification,
        mutation_authorization=None,
        clarification_message=_validation_clarification_message(trace) if status == "needs_clarification" else None,
    )


def run_semantic_transaction(
    graph: dict[str, Any],
    instruction: str,
    selection: dict[str, Any] | None = None,
    request_context: dict[str, Any] | None = None,
) -> TransactionResult:
    trace = run_semantic_shadow(graph, instruction, selection, request_context)
    result = semantic_trace_to_transaction_result(trace)
    if result.status in {"success", "partial_success"} and not graph_is_valid(result.final_graph):
        return TransactionResult(
            status="execution_error",
            interpretation=result.interpretation,
            boundary=None,
            steps=result.steps,
            final_graph=None,
            error="commit rejected: final graph failed structural validation",
            normalized_plan_from_fence=False,
            planning_path="semantic",
            target=result.target,
            graph_context=result.graph_context,
            specification=result.specification,
            mutation_authorization=result.mutation_authorization,
        )
    return result
