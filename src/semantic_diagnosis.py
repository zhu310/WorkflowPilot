from __future__ import annotations

import json
import re
from typing import Any, Protocol

from .config import get_settings
from .llm import DeepSeekAdapter


class CompletionAdapter(Protocol):
    def complete(self, system_prompt: str, user_prompt: str) -> str:
        ...


DIAGNOSIS_SYSTEM = """You are the semantic diagnosis stage of a workflow graph editor.

Return exactly one JSON object:
{
  "status": "needs_confirmation" | "needs_clarification",
  "analysis": "short user-facing diagnosis in Chinese",
  "options": [
    {
      "id": "option_1",
      "title": "short Chinese title",
      "description": "user-facing option description in Chinese, no internal ids",
      "tradeoff": "main benefit/risk in Chinese",
      "draft_instruction": "hidden executable edit goal, using exact node ids from graph_summary"
    }
  ],
  "recommended_option_id": "option_1",
  "question": "one Chinese confirmation or clarification question"
}

Responsibilities:
- Diagnose abstract, advisory, fact-checking, and ambiguous workflow questions only. Never claim the graph was modified.
- Answer the user's current question first. If the user is only asking, explaining, or discussing, do not push a graph edit.
- If the user only asks for current graph facts, answer the facts and set options to an empty list. Leave question empty unless the user asks what to do next.
- If the user only asks whether a logic problem exists, answer yes/no with grounded reasons first. Do not propose edit options unless the user also asks for a fix or recommendation.
- Do not output edit options merely because you found a problem. Options are allowed only when the user asks how to fix, asks for recommendations, asks for alternatives, or expresses an edit intent.
- If options is empty, question must be empty or a concrete clarifying question about the user's scope or business semantics. Never ask the user to choose a方案/option when no options were returned.
- Ground the answer in graph_summary: mention the relevant visible node labels, edge labels, paths, or local range.
- Reduce ambiguity only when it changes the possible graph mutation. Do not ask clarification for details that can be read from graph_summary.
- Recommend proportionally. If one direction is clearly best, recommend that one direction. Offer multiple options only when there are real tradeoffs.
- Use semantic option titles when options are needed, such as "人工处理后再继续"; do not rely on "方案1/方案2/方案3" as the meaning.
- If the user appears to want an edit but the request is still abstract, include a natural, user-facing edit sentence they can send next.
- Do not use a rigid response template. The fields should be concise and shaped by the current question.
- Return needs_clarification when no safe local recommendation or answer can be given without a real business choice.
- The graph_summary is the latest source of truth. Conversation history is only reference context.
- If conversation history describes an older graph but graph_summary already changed, diagnose the latest graph_summary.
- Resolve words such as "here", "this part", "the previous change", "刚才那个", "前面的", "就这么改", and "第一个"
  only when recent conversation history contains one unique, clear referent and graph_summary still supports it.
- If there are multiple candidate referents, multiple earlier option sets, or old unresolved topics, do not guess. Ask the user to restate the desired target in natural language.
- When focus_context is present and the user says "here", "this part", "this change", or similar,
  diagnose the focus_context area first. Do not drift to unrelated graph regions.
- focus_graph_summary is the resolved current subgraph for that focus area. Use it before the full graph
  when explaining "here" or "this change".
- If focus_context says the last edit succeeded, first verify that exact edited area in graph_summary.
  If it is now structurally consistent, say it is already correct instead of recommending the same change again.
- If the user asks about a current workflow fact, answer the fact from graph_summary. Do not create options and do not steer into editing.
- If the user gives a fully explicit edit instruction, this stage should not run the edit. Return status "needs_clarification", set options to [], set recommended_option_id to "", leave question empty, and say in analysis that the request is already clear enough for the formal Workflow IR edit path.
- Treat direct commands such as changing named nodes to parallel, serial, renamed, inserted, deleted, or reconnected relationships as fully explicit edit instructions when the affected visible nodes and target relationship are both present.
- If the user requests an unsafe or semantically conflicting topology change, explain the dependency conflict and ask for the business meaning needed to proceed.

Boundaries:
- User-facing fields must mention visible labels, not internal ids.
- Do not output English user-facing errors or backend terms in analysis/title/description/tradeoff/question.
- draft_instruction is hidden and used only after user confirmation.
- draft_instruction must be a concrete final-state edit goal, not analysis and not a script.
- draft_instruction must use exact existing node ids from graph_summary.
- Keep draft_instruction local: mention only affected nodes and required/forbidden relations.
- If an option is a recommendation, its description should make the intended user-visible graph change clear enough that the user can restate it.
- Do not require the user to copy internal ids. Internal ids are allowed only in draft_instruction.
- Do not list unrelated preserve-edges; the downstream executor preserves outside scope.
- Do not invent new node ids unless the option explicitly creates a new node.
- If the current graph already satisfies the user's stated target, say so in analysis and offer only remaining improvement options if any.
- For a clear parallel recommendation in an advisory answer, the hidden draft must include fan-out, fan-in, serial-edge removal, and AND-join. This does not apply to direct explicit edit commands, which should stay out of diagnosis.

Good hidden draft examples:
- "把 graph_summary 中已有的 node_a、node_b、node_c 改成串行主流程：node_a -> node_b -> node_c。删除 node_a -> node_c 的直接连线；如果 node_a -> node_b 只有旧的侧边，则目标里只保留 node_a -> node_b 的无标签主流程边。"
- "把 graph_summary 中已有的 node_entry、node_branch_1、node_branch_2、node_join 改成并行汇聚结构：node_entry 同时连接 node_branch_1 和 node_branch_2；node_branch_1 和 node_branch_2 都连接到 node_join；删除 node_branch_1 -> node_branch_2 的串行连线；将 node_join 标记为 AND-join。"
"""


def extract_json(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        match = re.search(r"```(?:json)?\s*(.*?)```", stripped, re.DOTALL | re.IGNORECASE)
        if match:
            stripped = match.group(1).strip()
    try:
        value = json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise
        value = json.loads(stripped[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError("LLM response must be a JSON object")
    return value


def graph_summary(graph: dict[str, Any]) -> dict[str, Any]:
    return {
        "nodes": [
            {
                "id": node.get("id"),
                "label": node.get("label"),
                "type": node.get("type"),
                "join": node.get("join"),
            }
            for node in graph.get("nodes", [])
            if isinstance(node, dict)
        ],
        "edges": [
            {
                "id": edge.get("id"),
                "source": edge.get("source"),
                "target": edge.get("target"),
                "label": edge.get("label", ""),
            }
            for edge in graph.get("edges", [])
            if isinstance(edge, dict)
        ],
    }


def focus_graph_summary(graph: dict[str, Any], focus_context: dict[str, Any]) -> dict[str, Any]:
    changed_area = focus_context.get("changed_area") if isinstance(focus_context.get("changed_area"), dict) else {}
    focus_ids = {
        str(node_id)
        for node_id in changed_area.get("node_ids", [])
        if isinstance(node_id, str)
    }
    edges = [
        edge
        for edge in graph.get("edges", [])
        if isinstance(edge, dict) and (edge.get("source") in focus_ids or edge.get("target") in focus_ids)
    ]
    for edge in edges:
        if isinstance(edge.get("source"), str):
            focus_ids.add(str(edge["source"]))
        if isinstance(edge.get("target"), str):
            focus_ids.add(str(edge["target"]))
    return {
        "nodes": [
            {
                "id": node.get("id"),
                "label": node.get("label"),
                "type": node.get("type"),
                "join": node.get("join"),
            }
            for node in graph.get("nodes", [])
            if isinstance(node, dict) and node.get("id") in focus_ids
        ],
        "edges": [
            {
                "id": edge.get("id"),
                "source": edge.get("source"),
                "target": edge.get("target"),
                "label": edge.get("label", ""),
            }
            for edge in edges
        ],
    }


def _clean_option(row: Any, index: int) -> dict[str, str] | None:
    if not isinstance(row, dict):
        return None
    title = str(row.get("title", "")).strip()
    description = str(row.get("description", "")).strip()
    draft = str(row.get("draft_instruction", "")).strip()
    if not title or not description:
        return None
    return {
        "id": str(row.get("id", f"option_{index}")).strip() or f"option_{index}",
        "title": title,
        "description": description,
        "tradeoff": str(row.get("tradeoff", "")).strip(),
        "draft_instruction": draft,
    }


def looks_like_target_state_instruction(draft: str, graph: dict[str, Any]) -> bool:
    if not draft:
        return False
    node_ids = {
        str(node.get("id"))
        for node in graph.get("nodes", [])
        if isinstance(node, dict) and isinstance(node.get("id"), str)
    }
    mentioned = {node_id for node_id in node_ids if node_id in draft}
    target_tokens = (
        "->",
        "\u6539\u6210",
        "\u6539\u4e3a",
        "\u76ee\u6807\u7ed3\u6784",
        "\u4e32\u884c",
        "\u5e76\u884c",
        "\u5220\u9664",
        "\u7981\u6b62",
        "\u4e0d\u8981\u4fdd\u7559",
        "\u4e0d\u4fdd\u7559",
        "AND-join",
    )
    return len(mentioned) >= 2 and any(token in draft for token in target_tokens)


def _normalize_diagnosis(raw: dict[str, Any], graph: dict[str, Any]) -> dict[str, Any]:
    status = raw.get("status") if raw.get("status") in {"needs_confirmation", "needs_clarification"} else "needs_clarification"
    options = [
        option
        for index, row in enumerate(raw.get("options", []), 1)
        if (option := _clean_option(row, index)) is not None
    ]
    recommended = str(raw.get("recommended_option_id", "")).strip()
    if recommended and recommended not in {row["id"] for row in options}:
        recommended = options[0]["id"] if options else ""
    if status == "needs_confirmation" and not options:
        status = "needs_clarification"
    recommended_instruction = next((row["draft_instruction"] for row in options if row["id"] == recommended), "") if recommended else ""
    if not looks_like_target_state_instruction(recommended_instruction, graph):
        recommended_instruction = ""
    question = str(raw.get("question", "")).strip()
    if not question and status == "needs_confirmation":
        question = "\u4f60\u5e0c\u671b\u6211\u6309\u54ea\u4e2a\u65b9\u6848\u7ee7\u7eed\uff1f"
    return {
        "status": status,
        "diagnosis": {
            "analysis": str(raw.get("analysis", "")).strip() or "\u6211\u9700\u8981\u66f4\u591a\u4e1a\u52a1\u4fe1\u606f\u624d\u80fd\u5224\u65ad\u3002",
            "options": options if status == "needs_confirmation" else [],
            "recommended_option_id": recommended if status == "needs_confirmation" else "",
            "recommended_instruction": recommended_instruction,
            "question": question,
            "requires_confirmation": status == "needs_confirmation",
        },
    }


def diagnose_workflow(
    graph: dict[str, Any],
    message: str,
    conversation_history: list[dict[str, str]] | None = None,
    focus_context: dict[str, Any] | None = None,
    adapter: CompletionAdapter | None = None,
) -> dict[str, Any]:
    client = adapter or DeepSeekAdapter(get_settings())
    payload = {
        "user_message": message,
        "graph_summary": graph_summary(graph),
        "conversation_history": conversation_history or [],
        "focus_context": focus_context or {},
        "focus_graph_summary": focus_graph_summary(graph, focus_context or {}),
    }
    raw = client.complete(DIAGNOSIS_SYSTEM, json.dumps(payload, ensure_ascii=False, indent=2))
    return _normalize_diagnosis(extract_json(raw), graph)
