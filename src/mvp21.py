from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from .config import get_settings
from .llm import DeepSeekAdapter
from .runner import parse_model_graph


REQUEST_INTERPRETATION_SYSTEM = """You interpret natural-language workflow edit requests for a visual flow canvas.
Do not edit the graph. Return exactly one JSON object with this shape:
{
  "status": "clear" | "needs_clarification",
  "ambiguity_type": "",
  "question": "",
  "candidate_interpretations": [],
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

Definitions:
- selection is a front-end fact. Never rewrite or pretend the user selected more than they did.
- resolved_scope is the local process region to be reworked in this turn. Nodes and edges are first-class and equal.
- resolved_scope is not a complete whitelist of every future change. Planning will decide concrete adds/removes later.

Interpretation rules:
- If the instruction only concerns the selected objects, return clear.
- If the instruction explicitly names a concrete object outside selection and the operation is clear, return clear and include that object in resolved_scope.
- If the instruction uses ambiguous references such as overall, here, these, that one, or similar wording and multiple reasonable scopes exist, return needs_clarification.
- If the instruction and selection create a real interpretation conflict that cannot both be satisfied without user choice, return needs_clarification.
- If a reference from short_context has exactly one reasonable candidate, resolve it directly.
- If a reference from short_context has multiple reasonable candidates, return needs_clarification.
- When status is needs_clarification, ask exactly one short clarification question that would unblock the edit.
- Use exact graph ids from the input.
- Keep notes brief.
"""


STRUCTURED_PLAN_SYSTEM = """You create a structured multi-step workflow edit plan.
Do not edit the graph. Return exactly JSON:
{
  "steps": [
    {
      "op": "add_node" | "update_node" | "move_node" | "add_edge" | "remove_edge" | "set_join",
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
- Nodes and edges are first-class planning objects.
- resolved_scope defines the core local region being reworked.
- boundary defines the current interfaces between the core scope and the external graph.
- You may create or replace boundary-crossing edges when required by the user request.
- External boundary endpoint nodes themselves must remain unchanged in id, label, type, x, y unless the instruction explicitly targets them.
- Include every required operation and no no-op steps.
- Prefer remove_edge + add_edge for branch retargeting.
- Use exact existing ids when removing or updating existing objects.
"""


EXECUTE_STEP_SYSTEM = """You execute exactly one workflow graph edit step for a visual process canvas.
Return exactly one JSON object and no markdown.
The JSON must have this shape: {"graph": {"nodes": [...], "edges": [...]}}.
Return the complete modified graph, not a patch.

Execution rules:
- Apply only current_step. Do not perform future steps early.
- Nodes and edges are first-class objects.
- Preserve all existing ids and fields unless current_step requires a change.
- resolved_scope.core_node_ids and resolved_scope.core_edge_ids describe the local region being reworked.
- Boundary endpoints are external graph connectors. They may participate in necessary cross-boundary edge changes, but external boundary nodes must keep id, label, type, x, and y unchanged unless explicitly targeted by the instruction.
- Do not rewrite unrelated nodes, edges, labels, positions, or topology outside the local rework.
- Every edge source and target must refer to a node in the returned graph.
- Available node types are start, end, task, decision, human_review.
- A node may optionally include join: {"mode": "all"}.
"""


@dataclass
class JsonRun:
    raw_response: str
    parsed_json: dict[str, Any] | None
    parse_error: str | None


@dataclass
class GraphRun:
    raw_response: str
    parsed_graph: dict[str, Any] | None
    parse_error: str | None


def _complete_json(system_prompt: str, payload: dict[str, Any]) -> JsonRun:
    adapter = DeepSeekAdapter(get_settings())
    raw_response = adapter.complete(system_prompt, json.dumps(payload, ensure_ascii=False, indent=2))
    try:
        parsed = json.loads(raw_response)
        if not isinstance(parsed, dict):
            raise TypeError("top-level JSON is not an object")
        return JsonRun(raw_response=raw_response, parsed_json=parsed, parse_error=None)
    except (json.JSONDecodeError, TypeError) as error:
        return JsonRun(raw_response=raw_response, parsed_json=None, parse_error=f"{type(error).__name__}: {error}")


def run_request_interpretation(
    graph: dict[str, Any],
    revision: int,
    selection: dict[str, Any],
    instruction: str,
    short_context: list[dict[str, str]] | None = None,
    clarification_history: list[dict[str, str]] | None = None,
) -> JsonRun:
    payload: dict[str, Any] = {
        "graph": graph,
        "revision": revision,
        "selection": selection,
        "instruction": instruction,
    }
    if short_context:
        payload["short_context"] = short_context
    if clarification_history:
        payload["clarification_history"] = clarification_history
    return _complete_json(REQUEST_INTERPRETATION_SYSTEM, payload)


def compute_boundary(graph: dict[str, Any], resolved_scope: dict[str, Any]) -> dict[str, Any]:
    core_node_ids = set(resolved_scope.get("core_node_ids", []))
    incoming: list[dict[str, Any]] = []
    outgoing: list[dict[str, Any]] = []
    for edge in graph.get("edges", []):
        source = edge.get("source")
        target = edge.get("target")
        if source not in core_node_ids and target in core_node_ids:
            incoming.append(
                {
                    "external_node_id": source,
                    "internal_node_id": target,
                    "existing_edge_id": edge.get("id"),
                    "direction": "outside_to_inside",
                }
            )
        elif source in core_node_ids and target not in core_node_ids:
            outgoing.append(
                {
                    "external_node_id": target,
                    "internal_node_id": source,
                    "existing_edge_id": edge.get("id"),
                    "direction": "inside_to_outside",
                }
            )
    return {"incoming": incoming, "outgoing": outgoing}


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
                "candidate_interpretations": interpretation.get("candidate_interpretations", []),
            }
        ],
        "status": "waiting_clarification",
    }


def resume_pending_request(
    pending_request: dict[str, Any],
    clarification_answer: str,
    current_graph: dict[str, Any],
    current_revision: int,
) -> tuple[dict[str, Any], JsonRun]:
    clarification_turns = list(pending_request.get("clarification_turns", []))
    if clarification_turns:
        clarification_turns[-1] = {**clarification_turns[-1], "answer": clarification_answer}
    else:
        clarification_turns.append({"question": "", "answer": clarification_answer})

    interpretation = run_request_interpretation(
        current_graph,
        current_revision,
        pending_request["selection"],
        pending_request["original_instruction"],
        pending_request.get("short_context", []),
        clarification_turns,
    )
    updated_pending = {
        **pending_request,
        "base_revision": current_revision,
        "graph_snapshot": current_graph,
        "clarification_turns": clarification_turns,
        "status": "resolved" if interpretation.parsed_json and interpretation.parsed_json.get("status") == "clear" else "waiting_clarification",
    }
    return updated_pending, interpretation


def run_structured_plan(
    graph: dict[str, Any],
    selection: dict[str, Any],
    instruction: str,
    resolved_scope: dict[str, Any],
    boundary: dict[str, Any],
    allowed_transformations: list[str],
    short_context: list[dict[str, str]] | None = None,
    clarification_history: list[dict[str, str]] | None = None,
) -> JsonRun:
    payload: dict[str, Any] = {
        "graph": graph,
        "selection": selection,
        "instruction": instruction,
        "resolved_scope": resolved_scope,
        "boundary": boundary,
        "allowed_transformations": allowed_transformations,
    }
    if short_context:
        payload["short_context"] = short_context
    if clarification_history:
        payload["clarification_history"] = clarification_history
    return _complete_json(STRUCTURED_PLAN_SYSTEM, payload)


def run_execute_step(
    graph: dict[str, Any],
    selection: dict[str, Any],
    instruction: str,
    resolved_scope: dict[str, Any],
    boundary: dict[str, Any],
    current_step: dict[str, Any],
) -> GraphRun:
    adapter = DeepSeekAdapter(get_settings())
    payload = {
        "graph": graph,
        "selection": selection,
        "instruction": instruction,
        "resolved_scope": resolved_scope,
        "boundary": boundary,
        "current_step": current_step,
    }
    raw_response = adapter.complete(EXECUTE_STEP_SYSTEM, json.dumps(payload, ensure_ascii=False, indent=2))
    parsed_graph, parse_error = parse_model_graph(raw_response)
    return GraphRun(raw_response=raw_response, parsed_graph=parsed_graph, parse_error=parse_error)
