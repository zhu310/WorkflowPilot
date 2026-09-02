from __future__ import annotations

import json
from typing import Any


SYSTEM_PROMPT = """You edit workflow graphs. Return exactly one JSON object and no markdown.
The JSON must have this shape: {\"graph\": {\"nodes\": [...], \"edges\": [...]}}.
Return the complete modified graph, not a patch. Preserve existing ids and fields unless the instruction requires a change. Do not change unrelated nodes or edges. Every edge source and target must refer to a node in the returned graph. Available node types for this experiment are start, end, task, decision, human_review.
When editable_scope is supplied, only the listed existing node_ids and edge_ids may be modified. Every existing node and edge outside that scope must be returned unchanged in every field, including id, type, label, x, y, source, and target. In scoped edits, do not add or delete nodes or edges.
A node may optionally include join: {\"mode\": \"all\"}. This means the node may proceed only after all incoming edges complete. Without join.mode=all, multiple incoming edges do not imply AND-join semantics."""


def build_user_prompt(
    instruction: str,
    graph: dict[str, Any],
    context: dict[str, Any] | None,
    editable_scope: dict[str, Any] | None = None,
) -> str:
    payload: dict[str, Any] = {"instruction": instruction, "graph": graph}
    if context:
        payload["context"] = context
    if editable_scope is not None:
        payload["editable_scope"] = editable_scope
    return json.dumps(payload, ensure_ascii=False, indent=2)
