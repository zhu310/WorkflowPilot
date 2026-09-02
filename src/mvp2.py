from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from .config import get_settings
from .llm import DeepSeekAdapter
from .runner import parse_model_graph


INTERPRET_SCOPE_SYSTEM = """You interpret workflow edit scope. Do not edit the graph.
Return exactly one JSON object with this shape:
{
  "status": "clear" | "ambiguous",
  "question": "",
  "focus_scope": {
    "anchor_node_ids": [],
    "boundary_node_ids": [],
    "anchor_edge_ids": []
  },
  "allowed_transformations": [],
  "notes": ""
}

Rules:
- Selection is a strong context anchor, not a hard permission boundary.
- If the instruction only concerns the selected objects, return clear.
- If the instruction explicitly names a concrete object outside selection and its operation, return clear and include that object in focus_scope.anchor_node_ids.
- If the instruction uses ambiguous referents such as overall, here, these, that one, or similar wording and multiple reasonable scopes exist, return ambiguous.
- If the selection and instruction create a real interpretation conflict that cannot be satisfied under one coherent scope, return ambiguous.
- When status is ambiguous, do not suggest edits. Ask one short clarification question in question.
- Use exact graph ids from the input. Keep notes brief.
"""


FREE_PLAN_SYSTEM = """You are an offline single-model planner for workflow graph editing.
Do not edit the graph. Return exactly JSON:
{
  "steps": [
    {
      "instruction": "..."
    }
  ]
}

Rules:
- Produce ordered, independently executable steps.
- Each step instruction must mention exact graph ids.
- Use explicit English operation phrases such as "add edge start -> risk", "remove edge e2", "set join approval all".
- Include every required topology change and no no-op steps.
- Respect the supplied focus_scope and allowed_transformations.
"""


STRUCTURED_PLAN_SYSTEM = """You are an offline single-model planner for workflow graph editing.
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
  ]
}

Rules:
- Only include fields needed by that step. Leave unrelated fields absent.
- Produce ordered, independently executable steps.
- Include every required structural change and no no-op steps.
- Respect the supplied focus_scope and allowed_transformations.
- Use exact graph ids.
- "set_join" uses node_id and join.
- "remove_edge" should prefer edge_id for an existing edge.
"""


COMPLETENESS_CHECK_SYSTEM = """You check planning completeness for workflow graph editing.
Do not edit the graph. Return exactly JSON:
{
  "status": "complete" | "incomplete",
  "missing_operations": [],
  "question": "",
  "notes": "",
  "revised_steps": []
}

Rules:
- Compare the draft steps against the original instruction, graph, focus_scope, allowed_transformations, and explicit checklist.
- If any required operation is missing, mark incomplete and return a complete revised_steps list.
- If the draft is already complete, mark complete and still return the full revised_steps list.
- Keep question empty unless the instruction is actually ambiguous.
- Use exact graph ids and the same structured step schema as the draft.
"""


EXECUTE_STEP_SYSTEM = """You execute exactly one workflow graph edit step.
Return exactly one JSON object and no markdown.
The JSON must have this shape: {"graph": {"nodes": [...], "edges": [...]}}.
Return the complete modified graph, not a patch.

Execution rules:
- Apply only the supplied current_step. Do not perform future steps early.
- Preserve existing ids and fields unless the current_step requires a change.
- Keep objects outside the described focus region unchanged.
- focus_scope.anchor_node_ids are the primary editable region.
- focus_scope.boundary_node_ids may keep the same node fields, but their incident edges may be modified only when the current_step requires a connection between a boundary node and an anchor node.
- New nodes, if any, must be conceptually inside the focus region.
- allowed_transformations tells you which kinds of edits are permitted for this run.
- Every edge source and target must refer to a node in the returned graph.
- Available node types for this experiment are start, end, task, decision, human_review.
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


def run_scope_interpretation(instruction: str, graph: dict[str, Any], selection: dict[str, Any]) -> JsonRun:
    return _complete_json(
        INTERPRET_SCOPE_SYSTEM,
        {"instruction": instruction, "graph": graph, "selection": selection},
    )


def run_free_plan(
    instruction: str,
    graph: dict[str, Any],
    focus_scope: dict[str, Any],
    allowed_transformations: list[str],
) -> JsonRun:
    return _complete_json(
        FREE_PLAN_SYSTEM,
        {
            "instruction": instruction,
            "graph": graph,
            "focus_scope": focus_scope,
            "allowed_transformations": allowed_transformations,
        },
    )


def run_structured_plan(
    instruction: str,
    graph: dict[str, Any],
    focus_scope: dict[str, Any],
    allowed_transformations: list[str],
) -> JsonRun:
    return _complete_json(
        STRUCTURED_PLAN_SYSTEM,
        {
            "instruction": instruction,
            "graph": graph,
            "focus_scope": focus_scope,
            "allowed_transformations": allowed_transformations,
        },
    )


def run_completeness_check(
    instruction: str,
    graph: dict[str, Any],
    focus_scope: dict[str, Any],
    allowed_transformations: list[str],
    draft_steps: list[dict[str, Any]],
    checklist: list[str],
) -> JsonRun:
    return _complete_json(
        COMPLETENESS_CHECK_SYSTEM,
        {
            "instruction": instruction,
            "graph": graph,
            "focus_scope": focus_scope,
            "allowed_transformations": allowed_transformations,
            "draft_steps": draft_steps,
            "checklist": checklist,
        },
    )


def run_execute_step(
    instruction: str,
    current_graph: dict[str, Any],
    focus_scope: dict[str, Any],
    allowed_transformations: list[str],
    current_step: dict[str, Any],
) -> GraphRun:
    adapter = DeepSeekAdapter(get_settings())
    payload = {
        "instruction": instruction,
        "current_graph": current_graph,
        "focus_scope": focus_scope,
        "allowed_transformations": allowed_transformations,
        "current_step": current_step,
    }
    raw_response = adapter.complete(EXECUTE_STEP_SYSTEM, json.dumps(payload, ensure_ascii=False, indent=2))
    parsed_graph, parse_error = parse_model_graph(raw_response)
    return GraphRun(raw_response=raw_response, parsed_graph=parsed_graph, parse_error=parse_error)
