from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from .config import get_settings
from .constraint_planner import plan_constraints
from .llm import DeepSeekAdapter
from .mvp22 import execute_deterministic, parse_json_with_fence
from .workflow_ir_core import (
    build_unit_contract,
    canonicalize_schema_variants,
    compile_ir,
    constraints_from_compiled,
    explicit_obligations,
    form_spec_units,
    generation_contract,
    graph_context_for_unit,
    graph_is_valid,
    graph_satisfies_obligations,
    ground_requirement,
    is_schema_structural_error,
    llm_requirements,
    merge_constraints,
    merge_skeleton_and_generated,
    normalize_request_context,
    normalize_ir,
    obligation_coverage,
    obligation_key,
    obligation_realization,
    project_generated_ir,
    required_obligations,
    requirement_conservation,
    requirement_needs_clarification,
    requirement_local_facts,
    selection_conflict,
    skeleton_from_obligations,
    structural_participant_coverage,
    validate_compiled_property_provenance,
    validate_ir,
    with_entity_specs,
)


UNDERSTAND_SYSTEM = """Extract user requirements and only explicit graph references from the request.
Return exactly:
{
  "requirements": [
    {
      "id": "r1",
      "text": "...",
      "references": {
        "source_node": "name or id explicitly stated by user",
        "target_node": "name or id explicitly stated by user",
        "relation_type": "explicit route label/type from user text",
        "selected_edge": {"source":"name or id","target":"name or id","relation_type":"explicit label"}
      },
        "desired_state": {
        "abstract_goals": [
          {"goal_type":"reduce_backtracking"|"simplify"|"increase_automation"|"highlight_main_flow"|"other","description":"requested high-level outcome"}
        ],
        "properties": [
          {"target":"source_node"|"target_node"|"selected_node"|"scope_nodes","node_id":"exact existing id/name if explicit","changes":{"label":"...","join":{"mode":"all"}}}
        ],
        "edge_properties": [
          {"target":"selected_edge","edge_id":"exact existing edge id if explicit","source":"source_node","target_node":"target_node","relation_type":"explicit route label/type when needed","changes":{"branch_label":"...","condition":{"expression":"explicit expression from user","language":"optional explicit language"}}}
        ],
        "position_offsets": [
          {"target":"source_node"|"target_node"|"selected_node"|"scope_nodes","node_id":"exact existing id/name if explicit","dx":0,"dy":0}
        ],
        "explicit_relations": [
          {"source":"source_node"|"target_node"|"selected_node","source_node":"exact existing id/name if explicit","target":"target_node"|"selected_node","target_node":"exact existing id/name if explicit","relation_type":"explicit route label/type from user text or \"\" when explicitly unlabeled"}
        ],
        "absent_entities": [{"node_id":"exact existing id/name explicitly requested for deletion"}],
        "selected_only": true
      },
      "scope": {
        "kind": "selected_only"|"visible_area"|"stage"|"group"|"subprocess"|"global",
        "name": "exact stage/group/subprocess name from user text when applicable"
      },
      "clarification": {"needed": false, "question": "", "reason": ""}
    }
  ]
}
Rules:
- Preserve all requested edit semantics; do not omit, weaken, or add requirements.
- Requirements may be coarse or split; they do not need perfect Specify granularity.
- Keep references only when explicit in the request wording.
- Distinguish the user's requested future goal from current graph facts. Do not rewrite a requested improvement into an already-existing current relation or property unless the user explicitly asked for that exact fact to be preserved.
- Use desired_state whenever the user explicitly specifies a target property value, coordinate move offset, explicit relation destination, deletion, or selected-only scope.
- Use desired_state.edge_properties for explicit existing-edge property edits only. branch_label is distinct from relation_type / edge label. Do not change relation_type, label, source, target, condition, or other metadata for a branch_label request.
- For condition.expression edits, only copy an explicit expression supplied by the user. Do not generate, optimize, infer, or rewrite business condition logic.
- Use desired_state.abstract_goals when the user asks for a high-level outcome that is not itself a concrete node/edge/property edit. Abstract goals describe requested final-state properties for validation; they are not edit actions.
- Include only properties explicitly requested by the user. Do not invent join/wait properties for rename, move, redirect, or delete requests.
- If the request says "selected node", "selected step", or equivalent, use `source_node: "$selected_node"` and/or desired_state targets with `target:"selected_node"`.
- If the request applies the same move/property change to the current visible nodes, a named stage/group/subprocess, or the whole canvas, set `scope` and use `target:"scope_nodes"` for that repeated change.
- Scope membership is determined by the program from request_context and graph metadata. Do not guess visible members or stage/group members in free text.
- Use `scope.kind="global"` when the instruction clearly says the whole canvas / entire workflow / all nodes / 整张画布 / 整体流程.
- Use `scope.kind="selected_only"` only when the instruction explicitly restricts the edit to the selected objects.
- For coordinate moves such as "move right by 120", express the offset in desired_state.position_offsets instead of describing an operation.
- For topology intents such as after, before, parallel, sequence, or branch structure, keep the semantics in requirement text plus grounded node references. Do not restate normal control-flow structure as desired_state.explicit_relations.
- Use desired_state.explicit_relations only when the user explicitly changes a concrete relation destination or existence, such as redirect/retry/exception/failure/success routing.
- If the request is a high-level improvement goal and the exact final structure is not deterministic from graph facts alone, keep the requested goal in the requirement text and put only the high-level validation goal in desired_state.abstract_goals.
- If a high-level improvement request does not provide enough concrete criteria to choose one reasonable rewrite over other materially different rewrites, set clarification.needed=true instead of inventing one.
- If the request already states a concrete improvement criterion together with preserve constraints, such as reducing backtracking while keeping steps, simplifying without removing checks, making the default path more automatic while preserving exception handling, or making the main flow clearer while keeping side paths, keep it as a clear requirement instead of asking for implementation details.
- When the graph already contains one obvious unlabeled/default flow and separately labeled side routes, treat those graph facts as sufficient grounding context for abstract improvement requests about the main/default flow unless the user asks for a different business scope.
- relation_type may only name a real graph edge label/type from the graph or the empty string "" for unlabeled edges.
- Never write natural-language position or waiting phrases into relation_type, such as after, before, wait for, or wait for all incoming.
- If the request is ambiguous, uses local words like "here/these" without enough scope context, or conflicts with an explicit selected-only scope, set clarification.needed=true and provide exactly one short clarification question.
- Do not output steps, operations, directives, units, boundaries, target scopes, or a final graph.
- Return JSON only.
"""


IR_SYSTEM = """Return one JSON object describing only the target workflow structure for the current unit.

Return exactly:
{
  "control_flow_regions": [
    {
      "relation_type": "exact edge label string, or "" for unlabeled control flow",
      "structure": {
        "type": "node" | "sequence" | "parallel" | "choice",
        "id": "existing node id",
        "ref": "new node ref",
        "node_type": "type for a new node",
        "label": "label for a new node",
        "items": [],
        "branches": [],
        "join_mode": "all" | "any"
      }
    }
  ],
  "auxiliary_relations": [
    {
      "source": "existing node id",
      "relation_type": "exact edge label string, or "" for unlabeled explicit relation",
      "targets": ["existing node id"]
    }
  ],
  "absent_entities": [
    {
      "node_id": "existing node id"
    }
  ],
  "properties": [
    {
      "node_id": "existing node id",
      "changes": {"join": {"mode": "all"}} or {"label": "New label"}
    }
  ],
  "edge_properties": [
    {
      "edge_id": "existing edge id",
      "changes": {"branch_label": "New branch label"} or {"condition": {"expression": "explicit expression", "language": "optional explicit language"}}
    }
  ]
}

Rules:
- Describe the final workflow structure, not edit steps.
- The payload includes a unit_contract. You may only write to the allowed target-state channels in that contract.
- Use node/sequence/parallel/choice only as structural IR primitives.
- For existing nodes, use {"type":"node","id":"..."}.
- For new nodes, use {"type":"node","ref":"new ref","node_type":"...","label":"..."}.
- Use sequence to express ordered flow, parallel to express concurrent branches, and choice for branch alternatives.
- Do not output add/remove edge instructions, exact outgoing target groups for the whole graph, required/forbidden relations, path constraints, or directives.
- control_flow_regions are for structural control flow owned by this unit, and relation_type may be any real graph label including "", success, failure, retry, or exception.
- auxiliary_relations represent explicit relation targets owned by this unit, for any real relation_type including "".
- If the user does not explicitly change a relation group, omit it and let the program inherit it from the current graph.
- For unlabeled relations, relation_type must be "" only when the grounded relation is truly unlabeled.
- If a downstream node must wait for all incoming work, use canonical property {"join":{"mode":"all"}} or parallel.join_mode="all".
- If a node must be renamed, use properties with {"label":"..."}.
- If an already authorized existing edge branch_label or condition.expression must be changed, use edge_properties. Do not use relation_type / label for branch_label, and do not invent condition expressions.
- For layout/readability improvements, use properties such as {"x": ...} and {"y": ...} on allowed existing nodes instead of rewriting business topology when topology does not need to change.
- If auxiliary_relations are allowed for the current unit, you may use them to rewrite the final targets of an owned relation group when that is the clearest way to realize the requested goal.
- If an existing node must be deleted, include it in absent_entities.
- The original instruction is only semantic context. You may describe only the current unit.
- Return JSON only.
"""


SCHEMA_RETRY_SYSTEM = """Previous IR is structurally invalid. Regenerate the complete IR without changing the requested semantics.

Return exactly one JSON object using the same Workflow IR schema as before.
Keep the same unit contract boundaries.
Fix only the structural/schema problem reported by the validator.
Return JSON only.
"""


@dataclass
class JsonRun:
    raw_response: str
    parsed_json: dict[str, Any] | None
    parse_error: str | None
    normalized_from_fence: bool


def _complete_json(system_prompt: str, payload: dict[str, Any]) -> JsonRun:
    raw_response = DeepSeekAdapter(get_settings()).complete(system_prompt, json.dumps(payload, ensure_ascii=False, indent=2))
    normalized = parse_json_with_fence(raw_response)
    return JsonRun(
        raw_response=raw_response,
        parsed_json=normalized.payload,
        parse_error=normalized.parse_error,
        normalized_from_fence=(normalized.normalized_text is not None and normalized.normalized_text != raw_response),
    )


def _safe_normalize_ir(value: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    try:
        return normalize_ir(value)
    except (KeyError, TypeError, AttributeError):
        return None


def _abstract_goal_violations(
    original_graph: dict[str, Any],
    final_graph: dict[str, Any],
    grounded_requirements: list[dict[str, Any]],
) -> list[str]:
    violations: list[str] = []

    def node_ids(graph_data: dict[str, Any]) -> set[str]:
        return {
            node["id"]
            for node in graph_data.get("nodes", [])
            if isinstance(node, dict) and isinstance(node.get("id"), str)
        }

    def edge_set(graph_data: dict[str, Any]) -> set[tuple[str, str, str]]:
        return {
            (edge.get("source"), edge.get("target"), edge.get("label", ""))
            for edge in graph_data.get("edges", [])
            if isinstance(edge, dict) and isinstance(edge.get("source"), str) and isinstance(edge.get("target"), str)
        }

    def positions(graph_data: dict[str, Any]) -> dict[str, tuple[float, float]]:
        return {
            node["id"]: (
                float(node.get("x", 0)) if isinstance(node.get("x"), (int, float)) else 0.0,
                float(node.get("y", 0)) if isinstance(node.get("y"), (int, float)) else 0.0,
            )
            for node in graph_data.get("nodes", [])
            if isinstance(node, dict) and isinstance(node.get("id"), str)
        }

    def backtracking_score(graph_data: dict[str, Any]) -> int:
        pos = positions(graph_data)
        return sum(
            1
            for edge in graph_data.get("edges", [])
            if isinstance(edge, dict)
            and edge.get("source") in pos
            and edge.get("target") in pos
            and pos[edge["target"]][0] < pos[edge["source"]][0]
        )

    def check_nodes_preserved(scope_nodes: set[str]) -> bool:
        final_nodes = node_ids(final_graph)
        original_check_nodes = {
            node["id"]
            for node in original_graph.get("nodes", [])
            if isinstance(node, dict)
            and isinstance(node.get("id"), str)
            and node["id"] in scope_nodes
            and "check" in str(node.get("label", node["id"])).lower()
        }
        return original_check_nodes <= final_nodes

    for requirement in grounded_requirements:
        desired_state = requirement.get("grounded_desired_state", {}) if isinstance(requirement.get("grounded_desired_state"), dict) else {}
        abstract_goals = desired_state.get("abstract_goals", []) if isinstance(desired_state.get("abstract_goals"), list) else []
        scope_policy = desired_state.get("scope_policy", {}) if isinstance(desired_state.get("scope_policy"), dict) else {}
        scope_nodes = {node_id for node_id in scope_policy.get("node_ids", []) if isinstance(node_id, str)}
        for row in abstract_goals:
            if not isinstance(row, dict):
                continue
            goal_type = row.get("goal_type")
            if goal_type == "reduce_backtracking" and backtracking_score(final_graph) >= backtracking_score(original_graph):
                violations.append("abstract goal reduce_backtracking was not satisfied")
            elif goal_type == "simplify":
                if not scope_nodes <= node_ids(final_graph):
                    violations.append("abstract goal simplify removed scoped nodes")
                elif not check_nodes_preserved(scope_nodes):
                    violations.append("abstract goal simplify removed required check nodes")
                elif len(edge_set(final_graph)) > len(edge_set(original_graph)):
                    violations.append("abstract goal simplify increased edge complexity")
            elif goal_type == "highlight_main_flow":
                if edge_set(final_graph) != edge_set(original_graph):
                    violations.append("abstract goal highlight_main_flow changed topology")
                elif positions(final_graph) == positions(original_graph):
                    violations.append("abstract goal highlight_main_flow did not change layout")
            elif goal_type in {"increase_automation", "other", None, ""}:
                violations.append(f"abstract goal {goal_type or 'unknown'} lacks deterministic satisfaction rule")
    return violations


def _unsupported_validation_goals(per_unit_obligations: list[dict[str, Any]]) -> list[dict[str, str]]:
    supported_goal_types = {"reduce_backtracking", "simplify", "highlight_main_flow"}
    unsupported: list[dict[str, str]] = []
    for obligations in per_unit_obligations:
        for row in obligations.get("abstract_goals", []):
            if not isinstance(row, dict):
                continue
            goal_type = str(row.get("goal_type") or "unknown")
            if goal_type not in supported_goal_types:
                unsupported.append(
                    {
                        "requirement_id": str(row.get("requirement_id") or ""),
                        "goal_type": goal_type,
                        "reason": "no deterministic validation standard",
                    }
                )
    return unsupported


def _orchestration_trace(
    graph: dict[str, Any],
    instruction: str,
    selection: dict[str, Any],
    request_context: dict[str, Any] | None,
    grounded_requirements: list[dict[str, Any]],
    units: list[dict[str, Any]],
    conservation: dict[str, Any],
    unit_contracts: list[dict[str, Any]],
    global_obligations: dict[str, Any],
    unit_runs: list[dict[str, Any]],
    merged_constraints: dict[str, Any] | None,
    planner: dict[str, Any] | None,
    stop_layer: str | None,
    stop_reason: str | None,
) -> dict[str, Any]:
    return {
        "instruction": instruction,
        "selection": selection,
        "request_context": request_context,
        "grounded_requirements": grounded_requirements,
        "grounded_references": [requirement.get("references", {}) for requirement in grounded_requirements],
        "spec_units": units,
        "requirement_conservation": conservation,
        "unit_contracts": unit_contracts,
        "global_obligations": global_obligations,
        "unit_runs": unit_runs,
        "merged_constraints": merged_constraints,
        "planner": planner,
        "stop_layer": stop_layer,
        "stop_reason": stop_reason,
        "current_graph": graph,
    }


def _clarification_response(
    graph: dict[str, Any],
    selection: dict[str, Any],
    request_context: dict[str, Any] | None,
    instruction: str,
    grounded_requirements: list[dict[str, Any]],
    units: list[dict[str, Any]],
    conservation: dict[str, Any],
    unit_contracts: list[dict[str, Any]],
    global_obligations: dict[str, Any],
    unit_runs: list[dict[str, Any]],
    stop_layer: str,
    reason: str,
    question: str,
    normalized_from_fence: bool,
) -> dict[str, Any]:
    return {
        "status": "needs_clarification",
        "error": reason,
        "clarification_message": question,
        "steps": [],
        "final_graph": None,
        "target": {"selection": selection},
        "graph_context": {"units": [graph_context_for_unit(graph, unit) for unit in units]},
        "specification": {
            **_orchestration_trace(
                graph,
                instruction,
                selection,
                request_context,
                grounded_requirements,
                units,
                conservation,
                unit_contracts,
                global_obligations,
                unit_runs,
                None,
                None,
                stop_layer,
                reason,
            ),
            "clarification": {"needed": True, "question": question, "reason": reason},
        },
        "mutation_authorization": None,
        "normalized_from_fence": normalized_from_fence,
    }


def run_workflow_ir_transaction(
    graph: dict[str, Any],
    revision: int,
    selection: dict[str, Any],
    instruction: str,
    short_context: list[dict[str, str]] | None = None,
    resolved_constraints: list[dict[str, Any]] | None = None,
    allowed_transformations: list[str] | None = None,
    request_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    del revision, short_context, resolved_constraints, allowed_transformations
    normalized_request_context = normalize_request_context(selection, request_context)
    understand_run = _complete_json(
        UNDERSTAND_SYSTEM,
        {"instruction": instruction, "graph": graph, "selection": selection, "request_context": normalized_request_context},
    )
    if understand_run.parsed_json is None:
        return {
            "status": "interpretation_error",
            "error": understand_run.parse_error,
            "steps": [],
            "final_graph": None,
            "target": {"selection": selection},
            "graph_context": None,
            "specification": {"stop_layer": "Understand", "raw_response": understand_run.raw_response},
            "mutation_authorization": None,
            "normalized_from_fence": understand_run.normalized_from_fence,
        }

    raw_requirements = (
        understand_run.parsed_json.get("requirements", [])
        if isinstance(understand_run.parsed_json, dict) and isinstance(understand_run.parsed_json.get("requirements"), list)
        else []
    )
    grounded_requirements = [
        ground_requirement(graph, requirement, selection, normalized_request_context)
        for requirement in raw_requirements
        if isinstance(requirement, dict) and requirement.get("id") and requirement.get("text")
    ]
    if instruction.strip() and not grounded_requirements:
        return {
            "status": "unsupported",
            "error": "no grounded requirements extracted from instruction",
            "steps": [],
            "final_graph": None,
            "target": {"selection": selection},
            "graph_context": {"units": []},
            "specification": {"stop_layer": "Understand", "raw_response": understand_run.raw_response},
            "mutation_authorization": None,
            "normalized_from_fence": understand_run.normalized_from_fence,
        }
    grounded_references = [requirement["references"] for requirement in grounded_requirements]
    local_graph_facts = [requirement_local_facts(graph, requirement) for requirement in grounded_requirements]
    units = form_spec_units(grounded_requirements, grounded_references, local_graph_facts) if grounded_requirements else []
    conservation = requirement_conservation(grounded_requirements, units)
    unit_contracts = [build_unit_contract(graph, unit) for unit in units]
    global_obligations = explicit_obligations(grounded_requirements, graph)
    unit_runs: list[dict[str, Any]] = []
    merged_constraints: dict[str, Any] | None = None
    planner: dict[str, Any] | None = None
    stop_layer: str | None = None
    stop_reason: str | None = None
    normalized_from_fence = understand_run.normalized_from_fence

    for requirement in grounded_requirements:
        needs_clarification, question, reason = requirement_needs_clarification(requirement)
        if needs_clarification:
            return _clarification_response(
                graph,
                selection,
                normalized_request_context,
                instruction,
                grounded_requirements,
                units,
                conservation,
                unit_contracts,
                global_obligations,
                unit_runs,
                "Ground",
                reason or "clarification required",
                question or "Please clarify the unresolved edit target.",
                normalized_from_fence,
            )

    has_selection_conflict, conflict_question = selection_conflict(grounded_requirements, selection)
    if has_selection_conflict:
        return _clarification_response(
            graph,
            selection,
            normalized_request_context,
            instruction,
            grounded_requirements,
            units,
            conservation,
            unit_contracts,
            global_obligations,
            unit_runs,
            "Ground",
            "selection scope conflict",
            conflict_question or "Please clarify the selected-only scope.",
            normalized_from_fence,
        )

    if not conservation["valid"]:
        stop_layer = "Requirement Ownership"
        stop_reason = json.dumps(
            {"missing_owners": conservation["missing_owners"], "duplicate_owners": conservation["duplicate_owners"]},
            ensure_ascii=False,
            sort_keys=True,
        )

    per_unit_obligations: list[dict[str, Any]] = []
    if stop_layer is None:
        for unit, unit_contract in zip(units, unit_contracts, strict=True):
            obligations = with_entity_specs(required_obligations(unit, graph, unit_contract, global_obligations), unit, graph)
            per_unit_obligations.append(obligations)

        unsupported_validation_goals = _unsupported_validation_goals(per_unit_obligations)
        if unsupported_validation_goals:
            return _clarification_response(
                graph,
                selection,
                normalized_request_context,
                instruction,
                grounded_requirements,
                units,
                conservation,
                unit_contracts,
                global_obligations,
                unit_runs,
                "Obligation Realization",
                json.dumps({"unsupported_validation_goals": unsupported_validation_goals}, ensure_ascii=False, sort_keys=True),
                "What concrete success criterion should I use to validate this high-level goal?",
                normalized_from_fence,
            )

        has_abstract_goal = any(obligations.get("abstract_goals") for obligations in per_unit_obligations)
        has_non_abstract_target = any(
            obligations.get("properties")
            or obligations.get("edge_properties")
            or obligations.get("auxiliary")
            or obligations.get("absent_entities")
            or obligations.get("entity_specs")
            or obligations.get("structural_participants")
            for obligations in per_unit_obligations
        )
        if has_abstract_goal and not has_non_abstract_target:
            return _clarification_response(
                graph,
                selection,
                normalized_request_context,
                instruction,
                grounded_requirements,
                units,
                conservation,
                unit_contracts,
                global_obligations,
                unit_runs,
                "Obligation Realization",
                "abstract goal has no concrete target change",
                "What concrete edit should I apply before validating this high-level goal?",
                normalized_from_fence,
            )

        owned_keys: set[str] = set()
        duplicate_owner_keys: dict[str, int] = defaultdict(int)
        all_keys: list[str] = []
        for obligations in per_unit_obligations:
            unit_keys = [
                *(obligation_key("property", row) for row in obligations.get("properties", [])),
                *(obligation_key("edge_property", row) for row in obligations.get("edge_properties", [])),
                *(obligation_key("explicit_relation", row) for row in obligations.get("auxiliary", [])),
                *(obligation_key("absent_entity", row) for row in obligations.get("absent_entities", [])),
                *(obligation_key("entity_new", row) for row in obligations.get("entity_specs", [])),
                *(obligation_key("structural_participants", row) for row in obligations.get("structural_participants", [])),
                *(obligation_key("abstract_goal", row) for row in obligations.get("abstract_goals", [])),
            ]
            all_keys.extend(unit_keys)
        for key in all_keys:
            owned_keys.add(key)
            duplicate_owner_keys[key] += 1
        global_expected_keys = {
            *(obligation_key("property", row) for row in global_obligations.get("properties", [])),
            *(obligation_key("edge_property", row) for row in global_obligations.get("edge_properties", [])),
            *(obligation_key("explicit_relation", row) for row in global_obligations.get("auxiliary", [])),
            *(obligation_key("absent_entity", row) for row in global_obligations.get("absent_entities", [])),
        }
        for obligations in per_unit_obligations:
            global_expected_keys.update(obligation_key("entity_new", row) for row in obligations.get("entity_specs", []))
            global_expected_keys.update(
                obligation_key("structural_participants", row) for row in obligations.get("structural_participants", [])
            )
            global_expected_keys.update(obligation_key("abstract_goal", row) for row in obligations.get("abstract_goals", []))
        orphan_obligations = sorted(global_expected_keys - owned_keys)
        if global_obligations.get("entity_create") and not any(obligations.get("entity_specs") for obligations in per_unit_obligations):
            orphan_obligations = sorted(set(orphan_obligations + ["entity_new:{}"]))
        duplicate_owners = {key: count for key, count in duplicate_owner_keys.items() if count > 1}
        if orphan_obligations or duplicate_owners:
            stop_layer = "Requirement Ownership"
            stop_reason = json.dumps(
                {"orphan_obligations": orphan_obligations, "duplicate_owners": duplicate_owners},
                ensure_ascii=False,
                sort_keys=True,
            )

    unit_constraints: list[dict[str, Any]] = []
    any_realized_content = False
    if stop_layer is None:
        for index, (unit, full_unit_contract, obligations) in enumerate(
            zip(units, unit_contracts, per_unit_obligations, strict=True),
            start=1,
        ):
            skeleton, structural_slots, abstract_slots, deterministic_entities = skeleton_from_obligations(unit, graph, obligations)
            realization = obligation_realization(obligations, skeleton, structural_slots, abstract_slots, unit.get("unit_id", f"u{index}"))
            generation_view = generation_contract(full_unit_contract, structural_slots, [], deterministic_entities)
            generation_requirements, generation_references = llm_requirements(unit, structural_slots, [])
            llm_generation_required = bool(structural_slots)

            unit_record: dict[str, Any] = {
                "unit_id": unit.get("unit_id", f"u{index}"),
                "requirements": unit.get("requirements", []),
                "references": unit.get("references", []),
                "graph_context": graph_context_for_unit(graph, unit),
                "unit_contract": full_unit_contract,
                "obligations": obligations,
                "target_skeleton": skeleton,
                "semantic_slots": {"structural_regions": structural_slots},
                "abstract_goal_validation": abstract_slots,
                "obligation_realization": realization,
                "llm_generation_required": llm_generation_required,
            }
            unit_runs.append(unit_record)
            any_realized_content = any_realized_content or bool(
                obligations.get("properties")
                or obligations.get("edge_properties")
                or obligations.get("auxiliary")
                or obligations.get("absent_entities")
                or obligations.get("entity_specs")
                or structural_slots
            )

            if not realization["obligation_realization_complete"]:
                stop_layer = "Obligation Realization"
                stop_reason = json.dumps(
                    {
                        "orphan_obligations": realization["orphan_obligations"],
                        "duplicate_realizations": realization["duplicate_realizations"],
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
                break

            projected_ir = {"control_flow_regions": [], "auxiliary_relations": [], "absent_entities": [], "properties": [], "edge_properties": []}
            raw_ir = None
            parse_error = None
            schema_error = None
            canonicalization_used = False
            projection_used = False
            schema_retry_triggered = False
            schema_retry_recovered = False
            first_pass_ir_valid = True

            if llm_generation_required:
                payload = {
                    "instruction": instruction,
                    "requirements": generation_requirements,
                    "references": generation_references,
                    "graph_context": unit_record["graph_context"],
                    "unit_contract": generation_view,
                    "target_skeleton": skeleton,
                    "semantic_slots": {"structural_regions": structural_slots},
                    "preserve_constraints": {"outside_scope_preserved": True},
                }
                first_run = _complete_json(IR_SYSTEM, payload)
                raw_ir = first_run.raw_response
                parse_error = first_run.parse_error
                normalized_from_fence = normalized_from_fence or first_run.normalized_from_fence
                canonical_ir, canonicalized = canonicalize_schema_variants(first_run.parsed_json)
                canonicalization_used = canonicalized
                projected_candidate = project_generated_ir(canonical_ir if isinstance(canonical_ir, dict) else None, generation_view)
                normalized_candidate = _safe_normalize_ir(canonical_ir if isinstance(canonical_ir, dict) else None)
                projection_used = isinstance(normalized_candidate, dict) and projected_candidate != normalized_candidate
                first_pass_ir_valid, schema_error = validate_ir(projected_candidate, graph, generation_view) if parse_error is None else (False, parse_error)
                projected_ir = projected_candidate

                if not first_pass_ir_valid and is_schema_structural_error(schema_error):
                    schema_retry_triggered = True
                    retry_payload = {
                        **payload,
                        "validator_error": schema_error,
                        "previous_ir": canonical_ir if isinstance(canonical_ir, dict) else raw_ir,
                    }
                    retry_run = _complete_json(SCHEMA_RETRY_SYSTEM, retry_payload)
                    normalized_from_fence = normalized_from_fence or retry_run.normalized_from_fence
                    retry_ir, retry_canonicalized = canonicalize_schema_variants(retry_run.parsed_json)
                    canonicalization_used = canonicalization_used or retry_canonicalized
                    retry_projected = project_generated_ir(retry_ir if isinstance(retry_ir, dict) else None, generation_view)
                    normalized_retry = _safe_normalize_ir(retry_ir if isinstance(retry_ir, dict) else None)
                    projection_used = projection_used or (isinstance(normalized_retry, dict) and retry_projected != normalized_retry)
                    retry_valid, retry_error = validate_ir(retry_projected, graph, generation_view) if retry_run.parse_error is None else (False, retry_run.parse_error)
                    raw_ir = retry_run.raw_response
                    parse_error = retry_run.parse_error
                    if retry_valid:
                        schema_retry_recovered = True
                        projected_ir = retry_projected
                        schema_error = None
                    else:
                        projected_ir = {"control_flow_regions": [], "auxiliary_relations": [], "absent_entities": [], "properties": [], "edge_properties": []}
                        schema_error = retry_error

            participant_coverage = structural_participant_coverage(projected_ir, structural_slots)
            final_ir = merge_skeleton_and_generated(skeleton, projected_ir)
            obligation_passed, obligation_errors = obligation_coverage(final_ir, obligations)

            unit_record.update(
                {
                    "generation_view": generation_view,
                    "generation_requirements": generation_requirements,
                    "generation_references": generation_references,
                    "raw_ir": raw_ir,
                    "parse_error": parse_error,
                    "first_pass_ir_valid": first_pass_ir_valid,
                    "schema_retry_triggered": schema_retry_triggered,
                    "schema_retry_recovered": schema_retry_recovered,
                    "canonicalization_used": canonicalization_used,
                    "projection_used": projection_used,
                    "projected_ir": projected_ir,
                    "structural_participant_coverage": participant_coverage,
                    "final_ir": final_ir,
                    "obligation_coverage": obligation_passed,
                    "obligation_errors": obligation_errors,
                }
            )

            if llm_generation_required and schema_error is not None:
                stop_layer = "Schema"
                stop_reason = schema_error
                break
            if not participant_coverage:
                stop_layer = "Projection"
                stop_reason = "required structural participants not fully realized"
                break
            if not obligation_passed:
                stop_layer = "Obligation Coverage"
                stop_reason = "; ".join(obligation_errors)
                break

            compiled = compile_ir(final_ir, graph)
            property_provenance_valid, property_provenance_errors = validate_compiled_property_provenance(
                graph,
                final_ir,
                compiled,
                obligations,
            )
            constraints = constraints_from_compiled(graph, compiled)
            unit_record["compiled"] = compiled
            unit_record["constraints"] = constraints
            unit_record["property_target_provenance_valid"] = property_provenance_valid
            unit_record["property_target_provenance_errors"] = property_provenance_errors
            if not property_provenance_valid:
                stop_layer = "Target Skeleton"
                stop_reason = "; ".join(property_provenance_errors)
                break
            unit_constraints.append(constraints)

    if stop_layer is None and grounded_requirements and not any_realized_content:
        stop_layer = "Obligation Realization"
        stop_reason = "grounded requirements did not realize any deterministic facts or semantic slots"

    if stop_layer is None:
        merged_constraints, merge_error = merge_constraints(graph, unit_constraints)
        if merged_constraints is None:
            stop_layer = "Target Skeleton"
            stop_reason = merge_error
    if stop_layer is None:
        planner = plan_constraints(graph, merged_constraints)
        if not planner.get("supported"):
            stop_layer = "Planner"
            stop_reason = planner.get("reason")

    if stop_layer is not None:
        return {
            "status": "unsupported",
            "error": stop_reason,
            "steps": [],
            "final_graph": None,
            "target": {"selection": selection},
            "graph_context": {"units": [graph_context_for_unit(graph, unit) for unit in units]},
            "specification": _orchestration_trace(
                graph,
                instruction,
                selection,
                normalized_request_context,
                grounded_requirements,
                units,
                conservation,
                unit_contracts,
                global_obligations,
                unit_runs,
                merged_constraints,
                planner,
                stop_layer,
                stop_reason,
            ),
            "mutation_authorization": None,
            "normalized_from_fence": normalized_from_fence,
        }

    execution = execute_deterministic(
        graph,
        {"core_node_ids": [], "core_edge_ids": []},
        {"incoming": [], "outgoing": []},
        planner["steps"],
        planner.get("mutation_authorization"),
    )
    if execution.graph is None or not graph_is_valid(execution.graph):
        error = execution.error or "final graph failed structural validation"
        return {
            "status": "execution_error",
            "error": error,
            "steps": planner["steps"],
            "final_graph": None,
            "target": {"selection": selection},
            "graph_context": {"units": [graph_context_for_unit(graph, unit) for unit in units]},
            "specification": _orchestration_trace(
                graph,
                instruction,
                selection,
                normalized_request_context,
                grounded_requirements,
                units,
                conservation,
                unit_contracts,
                global_obligations,
                unit_runs,
                merged_constraints,
                planner,
                "Executor",
                error,
            ),
            "mutation_authorization": planner.get("mutation_authorization"),
            "normalized_from_fence": normalized_from_fence,
        }

    no_graph_change = execution.graph == graph
    no_steps = not planner["steps"]
    obligations_already_satisfied = graph_satisfies_obligations(graph, global_obligations)
    has_abstract_goals = any(
        isinstance(unit_run.get("obligations"), dict) and unit_run["obligations"].get("abstract_goals")
        for unit_run in unit_runs
        if isinstance(unit_run, dict)
    )
    realized_target_exists = bool(
        merged_constraints
        and (
            merged_constraints.get("entities")
            or merged_constraints.get("absent_entities")
            or merged_constraints.get("properties")
            or merged_constraints.get("required_relations")
            or merged_constraints.get("forbidden_relations")
        )
    )
    if no_graph_change and no_steps and has_abstract_goals:
        return {
            "status": "unsupported",
            "error": "abstract change intent did not produce a validated target change",
            "steps": [],
            "final_graph": None,
            "target": {"selection": selection},
            "graph_context": {"units": [graph_context_for_unit(graph, unit) for unit in units]},
            "specification": _orchestration_trace(
                graph,
                instruction,
                selection,
                normalized_request_context,
                grounded_requirements,
                units,
                conservation,
                unit_contracts,
                global_obligations,
                unit_runs,
                merged_constraints,
                planner,
                "Final Validation",
                "abstract success invariant blocked unchanged graph",
            ),
            "mutation_authorization": planner.get("mutation_authorization"),
            "normalized_from_fence": normalized_from_fence,
        }
    abstract_violations = _abstract_goal_violations(graph, execution.graph, grounded_requirements) if has_abstract_goals else []
    if abstract_violations:
        return {
            "status": "unsupported",
            "error": "; ".join(abstract_violations),
            "steps": [],
            "final_graph": None,
            "target": {"selection": selection},
            "graph_context": {"units": [graph_context_for_unit(graph, unit) for unit in units]},
            "specification": _orchestration_trace(
                graph,
                instruction,
                selection,
                normalized_request_context,
                grounded_requirements,
                units,
                conservation,
                unit_contracts,
                global_obligations,
                unit_runs,
                merged_constraints,
                planner,
                "Final Validation",
                "; ".join(abstract_violations),
            ),
            "mutation_authorization": planner.get("mutation_authorization"),
            "normalized_from_fence": normalized_from_fence,
        }
    if no_graph_change and no_steps and not (obligations_already_satisfied or realized_target_exists):
        return {
            "status": "unsupported",
            "error": "grounded requirements did not produce a realized change or already-satisfied target",
            "steps": [],
            "final_graph": None,
            "target": {"selection": selection},
            "graph_context": {"units": [graph_context_for_unit(graph, unit) for unit in units]},
            "specification": _orchestration_trace(
                graph,
                instruction,
                selection,
                normalized_request_context,
                grounded_requirements,
                units,
                conservation,
                unit_contracts,
                global_obligations,
                unit_runs,
                merged_constraints,
                planner,
                "Final Validation",
                "false success invariant blocked unchanged graph",
            ),
            "mutation_authorization": planner.get("mutation_authorization"),
            "normalized_from_fence": normalized_from_fence,
        }

    return {
        "status": "success",
        "error": None,
        "steps": planner["steps"],
        "final_graph": execution.graph,
        "target": {"selection": selection},
        "graph_context": {"units": [graph_context_for_unit(graph, unit) for unit in units]},
        "specification": _orchestration_trace(
            graph,
            instruction,
            selection,
            normalized_request_context,
            grounded_requirements,
            units,
            conservation,
            unit_contracts,
            global_obligations,
            unit_runs,
            merged_constraints,
            planner,
            None,
            None,
        ),
        "mutation_authorization": planner.get("mutation_authorization"),
        "normalized_from_fence": normalized_from_fence,
    }
