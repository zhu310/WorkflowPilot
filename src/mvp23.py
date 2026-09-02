from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from .config import get_settings
from .constraint_planner import expand_specification, extract_graph_context, plan_constraints
from .llm import DeepSeekAdapter
from .mvp21 import compute_boundary
from .mvp22 import ExecutorResult, execute_deterministic, parse_json_with_fence
from .transformations import PARALLELIZE_SELECTED_CHAIN_TO_JOIN, compile_parallelize_selected_chain_to_join, has_unambiguous_serial_chain
from .workflow_ir_runtime import run_workflow_ir_transaction


REQUEST_INTERPRETATION_SYSTEM = """You interpret workflow edit requests for a visual process canvas.
Do not edit the graph. Return exactly one JSON object:
{
  "status": "clear" | "needs_clarification",
  "ambiguity_type": "",
  "question": "",
  "constraint_key": "",
  "candidate_constraints": [
    {
      "key": "",
      "value": "",
      "label": ""
    }
  ],
  "resolved_scope": {
    "core_node_ids": [],
    "core_edge_ids": []
  },
  "scope_expansion": {
    "added_node_ids": [],
    "added_edge_ids": []
  },
  "allowed_transformations": [],
  "notes": ""
}

Rules:
- selection is a front-end fact. Never rewrite it.
- resolved_constraints are already-confirmed user facts for this same request. Treat them as binding constraints, not soft conversation hints.
- Do not ask again about an ambiguity already resolved by resolved_constraints unless the current graph makes that constraint invalid.
- Nodes and edges are first-class objects.
- resolved_scope identifies the local process region to be reworked, not every future add/remove operation.
- Prioritize scope and instruction-binding ambiguities over optional style or implementation preferences.
- If the instruction only concerns the selected objects, return clear.
- If the instruction explicitly names an object outside selection and the operation is clear, return clear and include it in resolved_scope.
- If the instruction says only the selected objects should be edited, but also asks to edit a concrete object outside selection, return needs_clarification.
- For edge-only edits, include the selected edge in resolved_scope.core_edge_ids and include any current endpoint nodes that must participate in the reconnection inside resolved_scope.core_node_ids.
- If a named existing node must receive a semantic change such as join.mode, include that node in resolved_scope.core_node_ids.
- If multiple scope interpretations would produce materially different graph edits and no resolved constraint already chooses one, return needs_clarification.
- When returning needs_clarification, ask exactly one short question and provide explicit candidate_constraints using stable key/value pairs.
- Do not ask vague follow-up questions about implementation style when the user already gave a concrete structural edit.
- Use exact graph ids.
"""


CLARIFICATION_RESOLUTION_SYSTEM = """You resolve a user's clarification answer for the current pending workflow edit request.
Do not edit the graph. Return exactly one JSON object:
{
  "status": "resolved" | "needs_clarification",
  "resolved_constraint": {
    "key": "",
    "value": "",
    "source": "user_clarification"
  },
  "question": "",
  "notes": ""
}

Rules:
- Choose exactly one candidate constraint only if the user's answer clearly selects it.
- If the user's answer is still ambiguous, return needs_clarification and ask one short follow-up question.
- resolved_constraint must preserve the candidate key and value exactly.
"""


STRUCTURED_PLAN_SYSTEM = """You create a structured workflow edit plan.
Do not edit the graph. Return exactly JSON:
{
  "steps": [
    {
      "op": "add_node" | "update_node" | "move_node" | "add_edge" | "remove_edge" | "set_join" | "update_edge",
      "instruction": "",
      "node_id": "",
      "edge_id": "",
      "source": "",
      "target": "",
      "label": "",
      "changes": {},
      "x": 0,
      "y": 0,
      "join": {"mode": "all"},
      "node": {}
    }
  ],
  "notes": ""
}

Rules:
- The steps are both planning output and executable edit script.
- Respect resolved_constraints as binding facts.
- Use boundary to understand how the local scope connects to the external graph.
- Include every required operation and no no-op steps.
- Use exact existing ids when removing or updating existing objects.
- Do not wrap the JSON in markdown.
"""


SEMANTIC_DECISION_SYSTEM = """You decide whether a workflow edit should use one known semantic transformation.
Do not edit the graph. Return exactly one JSON object:
{
  "decision": "semantic" | "atomic",
  "semantic_operation": {
    "op": "parallelize_selected_chain_to_join",
    "branch_node_ids": [],
    "join_node_id": "",
    "join_mode": "all"
  },
  "notes": ""
}

Rules:
- Choose semantic only when the user explicitly asks to convert the current selected serial checks into parallel branches and to wait at one existing downstream join node until all branches complete.
- For semantic, output only parallelize_selected_chain_to_join with exact existing ids. Do not output upstream nodes, edge lists, or atomic steps.
- Choose atomic for every other request, including node edits, edge edits, layout edits, clarification-driven edits, failure routing, or an unclear topology request.
- The compiler will validate the graph topology. Never guess a semantic operation.
- Do not wrap JSON in markdown.
"""


EDIT_INTERPRETATION_SYSTEM = """You interpret a workflow edit request. Do not edit the graph.
Return exactly one JSON object:
{
  "status": "clear" | "needs_clarification",
  "question": "",
  "missing_user_decision": "",
  "constraint_key": "",
  "candidate_constraints": [{"key": "", "value": "", "label": ""}],
  "target": {"node_ids": [], "edge_ids": [], "moving_node_id": "", "anchor_node_id": "", "join_node_id": ""},
  "intent": {"kind": ""},
  "policy": {},
  "notes": ""
}

Rules:
- Target contains only user-facing objects and anchors the request is about. Never include predecessors, successors, upstream/downstream nodes, or edge ids merely because execution may later need them.
- Selection is a front-end fact. Do not rewrite it.
- Use exact existing ids for existing target objects. A selected edge can be included in target.edge_ids.
- Record only business choices explicitly supplied by the user in policy.
- Treat delete/remove of an existing selected or explicitly named node as clear when the user already chose the deletion policy, for example "delete it and its connections" or "delete it and keep the flow continuous".
- Graph facts such as current edge endpoints, predecessor, successor, position, and edge metadata are read by the program later. They are never a reason to ask clarification.
- If the user says any exception/failure from named branches should go to one existing handler node, and the graph already has one branch-specific exception/failure edge per named branch, that is clear. Do not ask whether to keep both branch edges or merge them.
- When an instruction refers to an existing node by its visible label, role, or unambiguous type (for example "manual handling" / a human-review node), resolve that node from the supplied graph and put its exact id in Target. Do not ask for its type, label, position, or id.
- Redirecting a selected edge to one unambiguously identifiable existing node is clear. Moving selected nodes by a stated offset is also clear; current coordinates are graph facts.
- If the user asks to change a selected edge or branch to an existing destination, interpret that as redirecting the selected edge unless the instruction explicitly says to move a node.
- Clarification is allowed only when at least two materially different business goals remain compatible with the instruction, selection, and graph facts, and the program cannot choose between them from those facts alone.
- A merely imaginable alternative is not enough for clarification.
- If the instruction explicitly says to edit only selected objects while also explicitly requests a semantic/property/layout change to an existing object outside selection, this is a real business conflict. Return needs_clarification with candidate_constraints for allowing or excluding that named object; do not silently include it in Target.
- If status is needs_clarification, missing_user_decision must name the unresolved business choice, and that choice must not be derivable from graph facts alone.
- Ask one short clarification only if a real business/design choice has multiple materially different outcomes.
- resolved_constraints are binding answers to prior clarification for this request.
- If resolved_constraints contains {"key": "allow_unselected", "value": "false"}, the originally requested unselected-object edit is explicitly excluded. Return clear with only the permitted selected target; do not ask the same conflict again.
- If resolved_constraints contains {"key": "allow_unselected", "value": "true"}, the named unselected object is permitted and should be included in Target.
- Do not output executable operations or atomic steps.
"""


EDIT_SPECIFICATION_SYSTEM = """You create a declarative workflow graph edit specification, not executable steps.
Return exactly one JSON object:
{
  "status": "ready" | "needs_clarification" | "unsupported",
  "question": "",
  "missing_user_decision": "",
  "constraints": {
    "entities": [{"ref": "", "type": "", "label": "", "x": 0, "y": 0}],
    "absent_entities": [{"node_id": ""}],
    "properties": [{"node_id": "", "changes": {}}],
    "edge_properties": [{"edge_id": "", "changes": {}}],
    "required_relations": [{"source": "", "target": "", "label": ""}],
    "forbidden_relations": [{"source": "", "target": "", "label": ""}],
    "preserve": {"outside_specification": true}
  },
  "directives": [
    {"kind": "insert_on_selected_edge", "new_entity": {"ref": "", "type": "", "label": ""}},
    {"kind": "redirect_selected_edge", "target_node_id": ""},
    {"kind": "parallelize_selected_chain", "branches": [], "upstream": "", "join": "", "completion": "all"},
    {"kind": "relocate_after", "moving_node_id": "", "anchor_node_id": "", "reconnect_old_location": true}
  ],
  "notes": ""
}

Rules:
- State only the desired entities, property values, required relations, forbidden relations, and preserve constraints. Never output add_edge, remove_edge, update_node, move_node, set_join, edge ids, or an executable script.
- Delete an existing node by listing it in `absent_entities`. If the user wants continuity, also state the required replacement relation explicitly. If the user wants only deletion, do not invent a reconnection.
- Direct constraints are preferred. Directives are permitted only for the listed topology intentions; the program expands them deterministically into the same generic constraints.
- Use target and read-only graph context. Do not invent graph facts that context already supplies.
- If Target or graph_context identifies an existing destination node, it is an existing node by definition. A selected-edge redirect to that id is ready and must not ask whether to create a node or request its id/type/coordinates.
- A new entity uses a temporary ref, never a final id.
- `entities` is only for a node the user explicitly asks to create. Never represent an existing graph node as a new entity: rename, type, join, and layout changes to an existing id must be written in `properties`.
- For every Target existing node that a stated offset moves, emit one `properties` item with that same node_id and its final x/y. Do not create a replacement node.
- For a selected-edge redirect to an existing target, emit the redirect_selected_edge directive; do not create either endpoint.
- For a serial-to-parallel conversion, emit exactly one parallelize_selected_chain directive and make it self-contained: `branches`, `upstream`, `join`, and `completion`.
- `branches` must be an array of exact existing branch node ids as plain strings; `upstream` and `join` must each be one exact existing node id string; `completion` states the join semantics such as `all`.
- The parallel directive is the authoritative statement of the normal parallel topology. Do not omit `upstream` even if graph context makes it obvious.
- If the request also changes retry, exception, failure, or other surrounding routing, use direct required_relations, forbidden_relations, edge labels, and join properties only for those external business relations outside the directive-owned normal parallel topology.
- If the instruction explicitly mentions an external exception/failure/retry relation, write that relation explicitly in required_relations or forbidden_relations even when the current graph already has it. Do not rely on preserve-only behavior for named external routing.
- Example: if parallel branches keep `material -> manual` and `permission -> manual` as exception routes, and retry changes from `manual -> material` to `manual -> start`, then required_relations must include both exception edges and the new retry edge, and forbidden_relations must include the old retry edge.
- Raw specifications must be internally consistent. Never require and forbid the same relation in one specification.
- When a parallel shorthand is present, never mark any branch-to-join edge as forbidden. The obsolete serial edges are only the current branch-to-branch chain edges, plus any old retry edge the instruction explicitly reroutes elsewhere.
- When a parallel shorthand is present, it owns the normal fan-out, normal fan-in, serial-chain removal, and completion join semantics for that local topology. Do not also restate those owned construction relations or the join.mode property in direct constraints.
- With a parallel shorthand, use direct required_relations and forbidden_relations only for business relations outside that owned local topology, such as exception/failure/retry or other labeled edges the user explicitly mentions.
- For coordinate edits, use the current target node coordinates from context and state final x/y values.
- If status is needs_clarification, missing_user_decision must name the unresolved business choice. If a real business choice remains unresolved, return needs_clarification. If the desired result cannot be expressed safely, return unsupported. Do not fall back to atomic steps.
"""


@dataclass
class JsonRun:
    raw_response: str
    parsed_json: dict[str, Any] | None
    parse_error: str | None
    normalized_from_fence: bool


@dataclass
class TransactionResult:
    status: str
    interpretation: dict[str, Any] | None
    boundary: dict[str, Any] | None
    steps: list[dict[str, Any]]
    final_graph: dict[str, Any] | None
    error: str | None
    normalized_plan_from_fence: bool
    planning_path: str = "atomic"
    semantic_operation: dict[str, Any] | None = None
    compiler_reason: str | None = None
    target: dict[str, Any] | None = None
    graph_context: dict[str, Any] | None = None
    specification: dict[str, Any] | None = None
    mutation_authorization: list[dict[str, Any]] | None = None


def _normalize_text(value: str) -> str:
    return re.sub(r"\s+", "", value).lower()


def _resolve_candidate_constraint_locally(
    candidate_constraints: list[dict[str, Any]],
    answer: str,
) -> dict[str, Any] | None:
    normalized = _normalize_text(answer)
    if not normalized:
        return None
    for row in candidate_constraints:
        key = row.get("key")
        value = row.get("value")
        label = row.get("label", "")
        if not isinstance(key, str) or not isinstance(value, str):
            continue
        tokens = {_normalize_text(value), _normalize_text(label)}
        if any(token and token in normalized for token in tokens):
            return {"key": key, "value": value, "source": "user_clarification"}
    if len(candidate_constraints) == 2 and {row.get("value") for row in candidate_constraints} == {"true", "false"}:
        truthy = ("允许", "可以", "是", "要", "true", "yes")
        falsy = ("不要", "不允许", "不可以", "否", "false", "no", "仅", "只")
        if any(token in normalized for token in truthy) and not any(token in normalized for token in falsy):
            return {"key": candidate_constraints[0]["key"], "value": "true", "source": "user_clarification"}
        if any(token in normalized for token in falsy):
            return {"key": candidate_constraints[0]["key"], "value": "false", "source": "user_clarification"}
    return None


def _complete_json(system_prompt: str, payload: dict[str, Any], allow_fence: bool = False) -> JsonRun:
    adapter = DeepSeekAdapter(get_settings())
    raw_response = adapter.complete(system_prompt, json.dumps(payload, ensure_ascii=False, indent=2))
    if allow_fence:
        normalized = parse_json_with_fence(raw_response)
        return JsonRun(
            raw_response=raw_response,
            parsed_json=normalized.payload,
            parse_error=normalized.parse_error,
            normalized_from_fence=(normalized.normalized_text is not None and normalized.normalized_text != raw_response),
        )
    try:
        parsed = json.loads(raw_response)
        if not isinstance(parsed, dict):
            raise TypeError("top-level JSON is not an object")
        return JsonRun(raw_response=raw_response, parsed_json=parsed, parse_error=None, normalized_from_fence=False)
    except (json.JSONDecodeError, TypeError) as error:
        return JsonRun(raw_response=raw_response, parsed_json=None, parse_error=f"{type(error).__name__}: {error}", normalized_from_fence=False)


def run_request_interpretation(
    graph: dict[str, Any],
    revision: int,
    selection: dict[str, Any],
    instruction: str,
    short_context: list[dict[str, str]] | None = None,
    resolved_constraints: list[dict[str, Any]] | None = None,
    clarification_turns: list[dict[str, Any]] | None = None,
) -> JsonRun:
    payload: dict[str, Any] = {
        "graph": graph,
        "revision": revision,
        "selection": selection,
        "instruction": instruction,
        "resolved_constraints": resolved_constraints or [],
    }
    if short_context:
        payload["short_context"] = short_context
    if clarification_turns:
        payload["clarification_turns"] = clarification_turns
    return _complete_json(REQUEST_INTERPRETATION_SYSTEM, payload)


def run_edit_interpretation(
    graph: dict[str, Any],
    revision: int,
    selection: dict[str, Any],
    instruction: str,
    short_context: list[dict[str, str]] | None = None,
    resolved_constraints: list[dict[str, Any]] | None = None,
    clarification_turns: list[dict[str, Any]] | None = None,
) -> JsonRun:
    payload: dict[str, Any] = {
        "graph": graph,
        "revision": revision,
        "selection": selection,
        "instruction": instruction,
        "resolved_constraints": resolved_constraints or [],
    }
    if short_context:
        payload["short_context"] = short_context
    if clarification_turns:
        payload["clarification_turns"] = clarification_turns
    result = _complete_json(EDIT_INTERPRETATION_SYSTEM, payload, allow_fence=True)
    parsed = result.parsed_json
    if parsed and parsed.get("intent", {}).get("kind") in {"redirect_edge", "change_edge_target"}:
        target = parsed.setdefault("target", {})
        edge_ids = target.get("edge_ids") or selection.get("edge_ids", [])
        candidates = [node for node in graph.get("nodes", []) if node.get("type") == "human_review"]
        if len(edge_ids) == 1 and len(candidates) == 1 and not target.get("join_node_id"):
            target["edge_ids"] = edge_ids
            target["join_node_id"] = candidates[0]["id"]
            if parsed.get("status") == "needs_clarification":
                parsed["status"] = "clear"
                parsed["question"] = ""
                parsed["notes"] = "resolved the unique existing human_review target from graph facts"
    if parsed and any(item.get("key") == "allow_unselected" and item.get("value") == "false" for item in (resolved_constraints or [])):
        target = parsed.setdefault("target", {})
        target["node_ids"] = list(selection.get("node_ids", []))
        target["edge_ids"] = list(selection.get("edge_ids", []))
        if target.get("moving_node_id") not in target["node_ids"]:
            target["moving_node_id"] = ""
        if parsed.get("status") == "needs_clarification":
            parsed["status"] = "clear"
            parsed["question"] = ""
            parsed["notes"] = "applied resolved constraint excluding unselected mutations"
    return result


def finalize_edit_specification(
    instruction: str,
    target: dict[str, Any],
    intent: dict[str, Any],
    policy: dict[str, Any],
    graph_context: dict[str, Any],
    resolved_constraints: list[dict[str, Any]] | None = None,
) -> JsonRun:
    result = _complete_json(
        EDIT_SPECIFICATION_SYSTEM,
        {
            "instruction": instruction,
            "target": target,
            "intent": intent,
            "policy": policy,
            "graph_context": graph_context,
            "resolved_constraints": resolved_constraints or [],
        },
        allow_fence=True,
    )
    if (
        result.parsed_json
        and result.parsed_json.get("status") == "needs_clarification"
        and intent.get("kind") in {"redirect_edge", "change_edge_target"}
        and isinstance(target.get("join_node_id"), str)
        and target.get("join_node_id")
    ):
        result.parsed_json = {
            "status": "ready",
            "constraints": {"entities": [], "absent_entities": [], "properties": [], "edge_properties": [], "required_relations": [], "forbidden_relations": [], "preserve": {"outside_specification": True}},
            "directives": [{"kind": "redirect_selected_edge", "target_node_id": target["join_node_id"]}],
            "notes": "normalized selected-edge redirect to an existing graph target",
        }
    return result


def _normalize_scope(
    graph: dict[str, Any],
    selection: dict[str, Any],
    interpretation: dict[str, Any],
) -> dict[str, Any]:
    resolved_scope = interpretation.get("resolved_scope", {})
    core_node_ids = set(resolved_scope.get("core_node_ids", []))
    core_edge_ids = set(resolved_scope.get("core_edge_ids", []))

    # Keep explicit selected edges inside the core scope. Their endpoints are
    # part of the local region even when the user only selected an edge.
    for edge_id in selection.get("edge_ids", []):
        core_edge_ids.add(edge_id)
    edge_map = {edge.get("id"): edge for edge in graph.get("edges", [])}
    for edge_id in list(core_edge_ids):
        edge = edge_map.get(edge_id)
        if edge is None:
            continue
        if edge.get("source"):
            core_node_ids.add(edge["source"])
        if edge.get("target"):
            core_node_ids.add(edge["target"])

    return {
        "core_node_ids": list(core_node_ids),
        "core_edge_ids": list(core_edge_ids),
    }


def resolve_clarification_answer(
    question: str,
    candidate_constraints: list[dict[str, Any]],
    answer: str,
    graph: dict[str, Any],
    selection: dict[str, Any],
    instruction: str,
) -> JsonRun:
    local = _resolve_candidate_constraint_locally(candidate_constraints, answer)
    if local is not None:
        return JsonRun(
            raw_response=json.dumps({"status": "resolved", "resolved_constraint": local, "question": "", "notes": "resolved locally from user clarification"}, ensure_ascii=False),
            parsed_json={"status": "resolved", "resolved_constraint": local, "question": "", "notes": "resolved locally from user clarification"},
            parse_error=None,
            normalized_from_fence=False,
        )
    return _complete_json(
        CLARIFICATION_RESOLUTION_SYSTEM,
        {
            "question": question,
            "candidate_constraints": candidate_constraints,
            "answer": answer,
            "graph": graph,
            "selection": selection,
            "instruction": instruction,
        },
    )


def run_structured_plan(
    graph: dict[str, Any],
    selection: dict[str, Any],
    instruction: str,
    resolved_scope: dict[str, Any],
    boundary: dict[str, Any],
    resolved_constraints: list[dict[str, Any]],
    allowed_transformations: list[str],
    short_context: list[dict[str, str]] | None = None,
) -> JsonRun:
    payload: dict[str, Any] = {
        "graph": graph,
        "selection": selection,
        "instruction": instruction,
        "resolved_scope": resolved_scope,
        "boundary": boundary,
        "resolved_constraints": resolved_constraints,
        "allowed_transformations": allowed_transformations,
    }
    if short_context:
        payload["short_context"] = short_context
    return _complete_json(STRUCTURED_PLAN_SYSTEM, payload, allow_fence=True)


def run_planning_decision(
    graph: dict[str, Any],
    selection: dict[str, Any],
    instruction: str,
    resolved_scope: dict[str, Any],
    boundary: dict[str, Any],
    resolved_constraints: list[dict[str, Any]],
    short_context: list[dict[str, str]] | None = None,
) -> JsonRun:
    payload: dict[str, Any] = {
        "graph": graph,
        "selection": selection,
        "instruction": instruction,
        "resolved_scope": resolved_scope,
        "boundary": boundary,
        "resolved_constraints": resolved_constraints,
        "known_semantic_transformations": [PARALLELIZE_SELECTED_CHAIN_TO_JOIN],
    }
    if short_context:
        payload["short_context"] = short_context
    return _complete_json(SEMANTIC_DECISION_SYSTEM, payload, allow_fence=True)


def _should_consider_semantic_decision(
    graph: dict[str, Any],
    selection: dict[str, Any],
    allowed_transformations: list[str],
) -> bool:
    # Keep all non-topology requests on the established Atomic path without another LLM call.
    selected_nodes = selection.get("node_ids", [])
    return "set_join" in allowed_transformations and isinstance(selected_nodes, list) and has_unambiguous_serial_chain(graph, selected_nodes)


def create_pending_request(
    request_id: str,
    graph: dict[str, Any],
    revision: int,
    selection: dict[str, Any],
    instruction: str,
    short_context: list[dict[str, str]] | None,
    interpretation: dict[str, Any],
) -> dict[str, Any]:
    return {
        "request_id": request_id,
        "base_revision": revision,
        "graph_snapshot": graph,
        "selection": selection,
        "original_instruction": instruction,
        "short_context": short_context or [],
        "clarification_turns": [
            {
                "ambiguity_type": interpretation.get("ambiguity_type", ""),
                "question": interpretation.get("question", ""),
                "constraint_key": interpretation.get("constraint_key", ""),
                "candidate_constraints": interpretation.get("candidate_constraints", []),
            }
        ],
        "resolved_constraints": [],
        "status": "waiting_clarification",
    }


def resume_pending_request(
    pending_request: dict[str, Any],
    clarification_answer: str,
    current_graph: dict[str, Any],
    current_revision: int,
) -> tuple[dict[str, Any], JsonRun, JsonRun]:
    clarification_turns = list(pending_request.get("clarification_turns", []))
    current_turn = dict(clarification_turns[-1]) if clarification_turns else {"question": "", "candidate_constraints": []}
    current_turn["answer"] = clarification_answer
    clarification_turns[-1] = current_turn

    resolution = resolve_clarification_answer(
        current_turn.get("question", ""),
        current_turn.get("candidate_constraints", []),
        clarification_answer,
        current_graph,
        pending_request["selection"],
        pending_request["original_instruction"],
    )
    resolved_constraints = list(pending_request.get("resolved_constraints", []))
    if resolution.parsed_json and resolution.parsed_json.get("status") == "resolved":
        resolved_constraint = resolution.parsed_json.get("resolved_constraint")
        if isinstance(resolved_constraint, dict):
            resolved_constraints = [constraint for constraint in resolved_constraints if constraint.get("key") != resolved_constraint.get("key")]
            resolved_constraints.append(resolved_constraint)

    interpretation = run_edit_interpretation(
        current_graph,
        current_revision,
        pending_request["selection"],
        pending_request["original_instruction"],
        pending_request.get("short_context", []),
        resolved_constraints,
        clarification_turns,
    )
    updated_pending = {
        **pending_request,
        "base_revision": current_revision,
        "graph_snapshot": current_graph,
        "clarification_turns": clarification_turns,
        "resolved_constraints": resolved_constraints,
        "status": "resolved" if interpretation.parsed_json and interpretation.parsed_json.get("status") == "clear" else "waiting_clarification",
    }
    return updated_pending, resolution, interpretation


def run_transaction(
    graph: dict[str, Any],
    revision: int,
    selection: dict[str, Any],
    instruction: str,
    short_context: list[dict[str, str]] | None = None,
    resolved_constraints: list[dict[str, Any]] | None = None,
    allowed_transformations: list[str] | None = None,
    request_context: dict[str, Any] | None = None,
) -> TransactionResult:
    if run_edit_interpretation is not _DEFAULT_RUN_EDIT_INTERPRETATION or finalize_edit_specification is not _DEFAULT_FINALIZE_EDIT_SPECIFICATION:
        return _run_transaction_legacy(
            graph,
            revision,
            selection,
            instruction,
            short_context=short_context,
            resolved_constraints=resolved_constraints,
            allowed_transformations=allowed_transformations,
            request_context=request_context,
        )

    result = run_workflow_ir_transaction(
        graph,
        revision,
        selection,
        instruction,
        short_context=short_context,
        resolved_constraints=resolved_constraints,
        allowed_transformations=allowed_transformations,
        request_context=request_context,
    )
    return TransactionResult(
        status=result["status"],
        interpretation=result.get("specification"),
        boundary=None,
        steps=result.get("steps", []),
        final_graph=result.get("final_graph"),
        error=result.get("error"),
        normalized_plan_from_fence=bool(result.get("normalized_from_fence")),
        planning_path="workflow_ir",
        target=result.get("target"),
        graph_context=result.get("graph_context"),
        specification=result.get("specification"),
        mutation_authorization=result.get("mutation_authorization"),
    )


def _run_transaction_legacy(
    graph: dict[str, Any],
    revision: int,
    selection: dict[str, Any],
    instruction: str,
    short_context: list[dict[str, str]] | None = None,
    resolved_constraints: list[dict[str, Any]] | None = None,
    allowed_transformations: list[str] | None = None,
    request_context: dict[str, Any] | None = None,
) -> TransactionResult:
    del request_context
    # Legacy constraint path retained only for test monkeypatch compatibility.
    interpreted = run_edit_interpretation(graph, revision, selection, instruction, short_context, resolved_constraints or [])
    if interpreted.parsed_json is None:
        return TransactionResult("interpretation_error", None, None, [], None, interpreted.parse_error, interpreted.normalized_from_fence, "constraint")
    interpretation = interpreted.parsed_json
    if interpretation.get("status") != "clear":
        return TransactionResult("needs_clarification", interpretation, None, [], None, None, interpreted.normalized_from_fence, "constraint")
    target = interpretation.get("target")
    intent = interpretation.get("intent", {})
    policy = interpretation.get("policy", {})
    if not isinstance(target, dict) or not isinstance(intent, dict) or not isinstance(policy, dict):
        return TransactionResult("interpretation_error", interpretation, None, [], None, "interpretation target, intent, or policy is invalid", interpreted.normalized_from_fence, "constraint")

    graph_context = extract_graph_context(graph, target, selection)
    specification_run = finalize_edit_specification(instruction, target, intent, policy, graph_context, resolved_constraints or [])
    if specification_run.parsed_json is None:
        return TransactionResult("specification_error", interpretation, None, [], None, specification_run.parse_error, specification_run.normalized_from_fence, "constraint", target=target, graph_context=graph_context)
    specification = specification_run.parsed_json
    if specification.get("status") == "needs_clarification":
        clarification = {**interpretation, "status": "needs_clarification", "question": specification.get("question", "")}
        return TransactionResult("needs_clarification", clarification, None, [], None, None, specification_run.normalized_from_fence, "constraint", target=target, graph_context=graph_context, specification=specification)
    if specification.get("status") != "ready":
        return TransactionResult("unsupported", interpretation, None, [], None, specification.get("notes") or "unsupported edit specification", specification_run.normalized_from_fence, "constraint", target=target, graph_context=graph_context, specification=specification)

    constraints, expansion_error = expand_specification(graph, selection, specification)
    if constraints is None:
        return TransactionResult("unsupported", interpretation, None, [], None, expansion_error, specification_run.normalized_from_fence, "constraint", target=target, graph_context=graph_context, specification=specification)
    if any(item.get("key") == "allow_unselected" and item.get("value") == "false" for item in (resolved_constraints or [])):
        # This is an explicit user preserve constraint, not a general target
        # whitelist: topology directives may still derive adjacent edge edits.
        permitted_properties = set(selection.get("node_ids", []))
        constraints["properties"] = [item for item in constraints["properties"] if item.get("node_id") in permitted_properties]
    planned = plan_constraints(graph, constraints)
    if not planned["supported"]:
        return TransactionResult("planning_error", interpretation, None, [], None, planned["reason"], specification_run.normalized_from_fence, "constraint", target=target, graph_context=graph_context, specification=specification)

    steps = planned["steps"]
    # Scope and boundary are legacy Executor parameters.  The new safety
    # contract is the complete planner authorization supplied below.
    executor_result: ExecutorResult = execute_deterministic(graph, {"core_node_ids": [], "core_edge_ids": []}, {"incoming": [], "outgoing": []}, steps, planned["mutation_authorization"])
    if executor_result.graph is None:
        return TransactionResult("execution_error", interpretation, None, steps, None, executor_result.error, specification_run.normalized_from_fence, "constraint", target=target, graph_context=graph_context, specification=specification, mutation_authorization=planned["mutation_authorization"])
    return TransactionResult("success", interpretation, None, steps, executor_result.graph, None, specification_run.normalized_from_fence, "constraint", target=target, graph_context=graph_context, specification=specification, mutation_authorization=planned["mutation_authorization"])


_DEFAULT_RUN_EDIT_INTERPRETATION = run_edit_interpretation
_DEFAULT_FINALIZE_EDIT_SPECIFICATION = finalize_edit_specification
