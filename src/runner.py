from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import get_settings
from .llm import DeepSeekAdapter
from .prompt import SYSTEM_PROMPT, build_user_prompt


@dataclass
class ModelRun:
    raw_response: str
    parsed_graph: dict[str, Any] | None
    parse_error: str | None


def parse_model_graph(raw_response: str) -> tuple[dict[str, Any] | None, str | None]:
    """Parse only; deliberately do not strip fences or repair malformed output."""
    try:
        payload = json.loads(raw_response)
        graph = payload["graph"]
        if not isinstance(graph, dict):
            raise TypeError("graph is not an object")
        return graph, None
    except (json.JSONDecodeError, KeyError, TypeError) as error:
        return None, f"{type(error).__name__}: {error}"


def run_single_llm(
    instruction: str,
    graph: dict[str, Any],
    context: dict[str, Any] | None,
    editable_scope: dict[str, Any] | None = None,
) -> ModelRun:
    adapter = DeepSeekAdapter(get_settings())
    raw_response = adapter.complete(SYSTEM_PROMPT, build_user_prompt(instruction, graph, context, editable_scope))
    parsed_graph, parse_error = parse_model_graph(raw_response)
    return ModelRun(raw_response=raw_response, parsed_graph=parsed_graph, parse_error=parse_error)


def save_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
