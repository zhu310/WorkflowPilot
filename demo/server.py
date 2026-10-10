from __future__ import annotations

import json
import os
import re
import sys
import time
from copy import deepcopy
from dataclasses import asdict
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
DIST = ROOT / "demo" / "frontend" / "dist"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.stage3_large_graph_acceptance import build_cases, generate_complex_workflow, graph_is_valid  # noqa: E402
from src.semantic_diagnosis import diagnose_workflow  # noqa: E402
from src.product_feedback import build_user_feedback, derive_interaction_need, extract_user_candidates  # noqa: E402
from src.semantic_shadow import run_semantic_shadow  # noqa: E402
from src.versioning import (  # noqa: E402
    VersionState,
    as_jsonable,
    create_version_state,
    redo,
    run_versioned_semantic_transaction,
    run_versioned_transaction,
    transaction_diff,
    undo,
)

SESSIONS: dict[str, VersionState] = {}


SELECTION_WORDS = {
    "选第一个": 1,
    "选 1": 1,
    "选择第一个": 1,
    "用第一个": 1,
    "first": 1,
    "选第二个": 2,
    "选 2": 2,
    "选择第二个": 2,
    "用第二个": 2,
    "second": 2,
    "选第三个": 3,
    "选 3": 3,
    "选择第三个": 3,
    "用第三个": 3,
    "third": 3,
}

CONFIRMATION_YES = {"确认", "是的", "继续", "仍然修改", "确认修改", "yes", "y", "confirm", "continue"}
CONFIRMATION_NO = {"取消", "不要", "否", "不用", "no", "n", "cancel"}


def _llm_error_message(error: Exception) -> str:
    detail = str(error)
    if "Insufficient Balance" in detail or "402" in detail:
        return "真实 LLM 调用失败：模型服务余额不足，请充值或更换可用 API Key 后重试。"
    if "DEEPSEEK_API_KEY" in detail:
        return "真实 LLM 调用失败：缺少 DEEPSEEK_API_KEY，请检查 .env 配置。"
    if "timed out" in detail.lower() or "timeout" in detail.lower():
        return "真实 LLM 调用失败：模型服务响应超时，请稍后重试。"
    return "真实 LLM 调用失败：请检查 API Key、模型服务余额或网络连接后重试。"


def diagnose_graph(payload: dict[str, Any]) -> dict[str, Any]:
    graph = payload.get("graph")
    if not isinstance(graph, dict) or not graph_is_valid(graph):
        session_id, state = _get_session(payload)
        graph = state.current_graph() if state is not None else None
    if not isinstance(graph, dict) or not graph_is_valid(graph):
        return {"status": "error", "error": "invalid graph for diagnosis"}

    history = payload.get("conversation_history") if isinstance(payload.get("conversation_history"), list) else []
    focus_context = payload.get("focus_context") if isinstance(payload.get("focus_context"), dict) else {}
    try:
        return _sanitize_diagnosis_response(diagnose_workflow(graph, str(payload.get("message", "")), history, focus_context))
    except Exception as error:
        message = _llm_error_message(error)
        return {
            "status": "llm_error",
            "error": message,
            "diagnosis": {
                "analysis": message,
                "options": [],
                "recommended_option_id": "",
                "recommended_instruction": "",
                "question": "配置恢复后，可以继续让我检查流程逻辑并给出推荐方案。",
                "requires_confirmation": False,
            },
        }


def _sanitize_diagnosis_response(result: dict[str, Any]) -> dict[str, Any]:
    diagnosis = result.get("diagnosis")
    if not isinstance(diagnosis, dict):
        return result
    options = diagnosis.get("options")
    if isinstance(options, list):
        option_instructions = {
            str(option.get("id", "")): option.get("draft_instruction", "")
            for option in options
            if isinstance(option, dict) and option.get("draft_instruction")
        }
        if option_instructions:
            diagnosis["option_instructions"] = option_instructions
        diagnosis["options"] = [
            {key: value for key, value in option.items() if key != "draft_instruction"}
            if isinstance(option, dict)
            else option
            for option in options
        ]
    return result


def _json_response(handler: SimpleHTTPRequestHandler, status: int, payload: dict[str, Any]) -> None:
    body = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.send_header("Access-Control-Allow-Origin", "*")
    handler.send_header("Access-Control-Allow-Headers", "content-type")
    handler.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
    handler.end_headers()
    handler.wfile.write(body)


def _read_json(handler: SimpleHTTPRequestHandler) -> dict[str, Any]:
    length = int(handler.headers.get("Content-Length", "0"))
    if length <= 0:
        return {}
    payload = json.loads(handler.rfile.read(length).decode("utf-8"))
    return payload if isinstance(payload, dict) else {}


def _nodes_by_id(graph: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        node["id"]: node
        for node in graph.get("nodes", [])
        if isinstance(node, dict) and isinstance(node.get("id"), str)
    }


def _edges_by_id(graph: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        edge["id"]: edge
        for edge in graph.get("edges", [])
        if isinstance(edge, dict) and isinstance(edge.get("id"), str)
    }


def graph_diff(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    before_nodes = _nodes_by_id(before)
    after_nodes = _nodes_by_id(after)
    before_edges = _edges_by_id(before)
    after_edges = _edges_by_id(after)
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


def _session_payload(session_id: str, state: VersionState) -> dict[str, Any]:
    return {
        "session_id": session_id,
        "current_version_id": state.current_version_id,
        "graph": state.current_graph(),
        "state": as_jsonable(state),
    }


def _get_session(payload: dict[str, Any]) -> tuple[str | None, VersionState | None]:
    session_id = payload.get("session_id")
    if not isinstance(session_id, str) or session_id not in SESSIONS:
        return None, None
    return session_id, SESSIONS[session_id]


def _flatten_pending_candidates(pending: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not isinstance(pending, dict):
        return []
    rows: list[dict[str, Any]] = []
    for group in pending.get("candidates", []):
        if not isinstance(group, dict):
            continue
        for index, candidate in enumerate(group.get("candidates", []) if isinstance(group.get("candidates"), list) else [], start=1):
            if not isinstance(candidate, dict) or not isinstance(candidate.get("id"), str):
                continue
            rows.append(
                {
                    **candidate,
                    "index": len(rows) + 1,
                    "group_index": index,
                    "location": group.get("location", ""),
                    "reference": group.get("reference", ""),
                    "binding_type": group.get("type", candidate.get("type", "")),
                }
            )
    return rows


def _candidate_selected_ids(selection: dict[str, Any], candidate_type: str) -> list[str]:
    key = "edge_ids" if candidate_type == "edge" else "node_ids"
    values = selection.get(key) if isinstance(selection, dict) else []
    return [item for item in values if isinstance(item, str)] if isinstance(values, list) else []


def _candidate_type(rows: list[dict[str, Any]]) -> str:
    for row in rows:
        binding_type = str(row.get("binding_type") or row.get("type") or "")
        if binding_type in {"edge", "node"}:
            return binding_type
    return "object"


def _candidate_by_current_selection(selection: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    candidate_type = _candidate_type(rows)
    if candidate_type not in {"edge", "node"}:
        return None
    candidate_ids = {str(row.get("id")) for row in rows if isinstance(row.get("id"), str)}
    selected = [item for item in _candidate_selected_ids(selection, candidate_type) if item in candidate_ids]
    selected = sorted(set(selected))
    if len(selected) != 1:
        return None
    return next((row for row in rows if row.get("id") == selected[0]), None)


CHINESE_ORDINAL_DIGITS = {
    "一": 1,
    "二": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
    "十": 10,
}


def _parse_chinese_ordinal(value: str) -> int | None:
    text = value.strip()
    match = re.search(r"第\s*(\d{1,3})\s*(?:个|条|项|位|)", text)
    if match:
        return int(match.group(1))
    match = re.search(r"第\s*([一二三四五六七八九十]{1,3})\s*(?:个|条|项|位|)", text)
    if not match:
        return None
    raw = match.group(1)
    if raw == "十":
        return 10
    if raw.startswith("十"):
        return 10 + CHINESE_ORDINAL_DIGITS.get(raw[1:], 0)
    if "十" in raw:
        left, _, right = raw.partition("十")
        return CHINESE_ORDINAL_DIGITS.get(left, 0) * 10 + CHINESE_ORDINAL_DIGITS.get(right, 0)
    return CHINESE_ORDINAL_DIGITS.get(raw)


def _compact_candidate_text(value: Any) -> str:
    return re.sub(r"[^0-9a-zA-Z\u4e00-\u9fff]+", "", str(value or "").lower())


def _candidate_by_constrained_text(answer: str, rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    ordinal = _parse_chinese_ordinal(answer)
    if ordinal is not None:
        return rows[ordinal - 1] if 0 < ordinal <= len(rows) else {"invalid": True}
    compact_answer = _compact_candidate_text(answer)
    if not compact_answer:
        return None
    matches = []
    for row in rows:
        fields = [
            row.get("id"),
            row.get("label"),
            row.get("reference"),
            row.get("reason"),
        ]
        compact_fields = [_compact_candidate_text(field) for field in fields if field]
        if any(compact_answer == field or compact_answer in field or field in compact_answer for field in compact_fields if field):
            matches.append(row)
    unique_ids = {row.get("id") for row in matches}
    return matches[0] if len(unique_ids) == 1 else None


def _resolve_pending_candidate(
    answer: str,
    selection: dict[str, Any],
    pending: dict[str, Any] | None,
) -> dict[str, Any] | None:
    rows = _flatten_pending_candidates(pending)
    if not rows:
        return None
    selected = _candidate_by_current_selection(selection, rows)
    if selected is not None:
        return selected
    parsed = _parse_candidate_selection(answer, pending)
    if parsed is not None:
        return parsed
    return _candidate_by_constrained_text(answer, rows)


def _parse_candidate_selection(answer: str, pending: dict[str, Any] | None) -> dict[str, Any] | None:
    normalized = answer.strip().lower().replace("　", " ")
    rows = _flatten_pending_candidates(pending)
    if not normalized or not rows:
        return None
    selected_index = None
    compact = normalized.replace(" ", "")
    for phrase, index in SELECTION_WORDS.items():
        if phrase.replace(" ", "").lower() in compact:
            selected_index = index
            break
    if selected_index is not None:
        return rows[selected_index - 1] if 0 < selected_index <= len(rows) else {"invalid": True}
    for row in rows:
        candidate_id = str(row.get("id", "")).lower()
        label = str(row.get("label", "")).lower()
        if candidate_id and candidate_id in normalized:
            return row
        if label and label in normalized:
            return row
    return None


def _candidate_pending_response(state: VersionState, session_id: str, pending: dict[str, Any]) -> dict[str, Any]:
    interaction = pending.get("interaction") if isinstance(pending.get("interaction"), dict) else {}
    candidates = pending.get("candidates") if isinstance(pending.get("candidates"), list) else []
    prompt = str(interaction.get("prompt") or "请从候选中选择一个对象。")
    lines = [prompt]
    for group in candidates:
        if not isinstance(group, dict):
            continue
        rows = group.get("candidates") if isinstance(group.get("candidates"), list) else []
        for index, candidate in enumerate(rows[:5], start=1):
            if isinstance(candidate, dict):
                lines.append(f"{index}. {candidate.get('label')}")
    message = "\n".join(line for line in lines if line).strip()
    feedback = {
        "status": "needs_clarification",
        "message": message,
        "executed": [],
        "incomplete": [],
        "interaction": interaction,
        "clarification": {"question": prompt, "candidates": candidates},
    }
    return {
        "status": "needs_clarification",
        "error": None,
        "clarification_message": prompt,
        "message": message,
        "user_feedback": feedback,
        "pending_id": pending.get("pending_id"),
        "pipeline": os.getenv("AI_EDIT_PIPELINE", "legacy").strip().lower() or "legacy",
        "session_id": session_id,
        "current_version_id": state.current_version_id,
        "graph": state.current_graph(),
        "state": as_jsonable(state),
    }


def _looks_like_candidate_selection(answer: str) -> bool:
    compact = answer.strip().lower().replace(" ", "").replace("　", "")
    if not compact:
        return False
    if any(phrase.replace(" ", "").lower() in compact for phrase in SELECTION_WORDS):
        return True
    return compact.startswith(("选择", "选", "用")) and any(ch.isdigit() for ch in compact)


def _candidate_exists(graph: dict[str, Any], row: dict[str, Any]) -> bool:
    candidate_id = row.get("id")
    if row.get("binding_type") == "edge" or row.get("type") == "edge":
        return isinstance(candidate_id, str) and candidate_id in _edges_by_id(graph)
    return isinstance(candidate_id, str) and candidate_id in _nodes_by_id(graph)


def _compact_text(value: str) -> str:
    return value.replace(" ", "").strip().lower()


def _looks_like_confirmation_yes(value: str) -> bool:
    compact = _compact_text(value)
    return compact in {_compact_text(item) for item in CONFIRMATION_YES}


def _looks_like_confirmation_no(value: str) -> bool:
    compact = _compact_text(value)
    return compact in {_compact_text(item) for item in CONFIRMATION_NO}


def _context_with_candidate_binding(request_context: dict[str, Any], row: dict[str, Any]) -> dict[str, Any]:
    updated = dict(request_context)
    candidate_id = str(row.get("id", ""))
    reference = str(row.get("reference", ""))
    location = str(row.get("location", ""))
    binding_type = "edge" if row.get("binding_type") == "edge" or row.get("type") == "edge" else "node"
    bindings = {
        "by_location": {location: candidate_id} if location else {},
        "by_reference": {reference: candidate_id} if reference else {},
        "edges": {reference: candidate_id} if binding_type == "edge" and reference else {},
        "nodes": {reference: candidate_id} if binding_type == "node" and reference else {},
    }
    updated["candidate_bindings"] = bindings
    updated["clarification_resolution"] = {
        "source": "pending_candidate",
        "candidate_id": candidate_id,
        "binding_type": binding_type,
        "reference": reference,
        "location": location,
    }
    return updated


def _pending_interaction_type(pending: dict[str, Any] | None) -> str:
    return str(pending.get("type") or "") if isinstance(pending, dict) else ""


def _pending_is_stale(state: VersionState) -> bool:
    pending = state.pending_clarification
    if not isinstance(pending, dict):
        return False
    origin = pending.get("originating_version_id")
    return isinstance(origin, str) and bool(origin) and origin != state.current_version_id


def _single_selected_edge(selection: dict[str, Any], graph: dict[str, Any]) -> str | None:
    raw_edge_ids = selection.get("edge_ids") if isinstance(selection, dict) else []
    if not isinstance(raw_edge_ids, list):
        return None
    edges = _edges_by_id(graph)
    edge_ids = [edge_id for edge_id in raw_edge_ids if isinstance(edge_id, str) and edge_id in edges]
    return edge_ids[0] if len(edge_ids) == 1 else None


def _context_with_pending_resume(request_context: dict[str, Any], pending: dict[str, Any], resume: dict[str, Any]) -> dict[str, Any]:
    updated = dict(request_context)
    pending_context = pending.get("request_context")
    if isinstance(pending_context, dict):
        updated = {**pending_context, **updated}
    updated["pending_interaction"] = pending.get("interaction")
    updated["pending_resume"] = resume
    partial_semantic_state = pending.get("partial_semantic_state")
    if isinstance(partial_semantic_state, dict):
        updated["semantic_refinement"] = {
            "mode": "refine_partial_goal",
            "partial_semantic_state": partial_semantic_state,
            "originating_instruction": pending.get("original_instruction", ""),
            "originating_selection": pending.get("selection", {"node_ids": [], "edge_ids": []}),
        }
    return updated


def _context_with_pending_candidate_binding(
    request_context: dict[str, Any],
    pending: dict[str, Any],
    row: dict[str, Any],
) -> dict[str, Any]:
    pending_context = pending.get("request_context")
    base = {**pending_context, **request_context} if isinstance(pending_context, dict) else dict(request_context)
    updated = _context_with_candidate_binding(base, row)
    updated["pending_interaction"] = pending.get("interaction")
    updated["pending_resume"] = {
        "type": "choose_candidate",
        "candidate_id": row.get("id"),
        "binding_type": row.get("binding_type") or row.get("type"),
        "location": row.get("location", ""),
        "reference": row.get("reference", ""),
    }
    semantic_goal = pending.get("semantic_goal")
    if isinstance(semantic_goal, dict):
        parsed = semantic_goal.get("parsed")
        if isinstance(parsed, dict):
            updated["semantic_goal_override"] = parsed
    return updated


def _resume_missing_slot_interaction(
    pending: dict[str, Any] | None,
    instruction: str,
    selection: dict[str, Any],
    request_context: dict[str, Any],
    graph: dict[str, Any],
) -> tuple[str, dict[str, Any], dict[str, Any], bool, str]:
    if _pending_interaction_type(pending) != "provide_missing_slot":
        return instruction, selection, request_context, False, ""
    interaction = pending.get("interaction") if isinstance(pending, dict) else None
    if not isinstance(interaction, dict):
        return instruction, selection, request_context, False, ""
    slot = str(interaction.get("slot") or "")
    if slot == "subject_edge":
        edge_id = _single_selected_edge(selection, graph)
        if edge_id is None:
            prompt = str(interaction.get("prompt") or "Please select one connection, then try again.")
            return instruction, selection, request_context, False, prompt
        known_target = str(interaction.get("known_target") or "").strip()
        if not known_target:
            prompt = str(interaction.get("prompt") or "Please describe the target node.")
            return instruction, selection, request_context, False, prompt
        updated_selection = {"node_ids": [], "edge_ids": [edge_id]}
        updated_context = _context_with_pending_resume(
            request_context,
            pending,
            {"type": "provide_missing_slot", "slot": slot, "edge_id": edge_id, "known_target": known_target},
        )
        return f"Route selected edge to {known_target}", updated_selection, updated_context, True, ""
    if slot == "route_target":
        pending_selection = pending.get("selection")
        updated_selection = pending_selection if isinstance(pending_selection, dict) else selection
        updated_context = _context_with_pending_resume(
            request_context,
            pending,
            {"type": "provide_missing_slot", "slot": slot, "target_reply": instruction},
        )
        return f"Route selected edge to {instruction}", updated_selection, updated_context, True, ""
    if slot == "transformation":
        pending_selection = pending.get("selection")
        updated_selection = pending_selection if isinstance(pending_selection, dict) else selection
        updated_context = _context_with_pending_resume(
            request_context,
            pending,
            {"type": "provide_missing_slot", "slot": slot, "reply": instruction},
        )
        return instruction, updated_selection, updated_context, True, ""
    return instruction, selection, request_context, False, str(interaction.get("prompt") or "")


def _route_confirmation_issue(specification: dict[str, Any] | None) -> dict[str, Any] | None:
    spec = specification if isinstance(specification, dict) else {}
    validation = spec.get("validation") if isinstance(spec.get("validation"), dict) else {}
    issues = validation.get("issues") if isinstance(validation.get("issues"), list) else []
    return next(
        (issue for issue in issues if isinstance(issue, dict) and issue.get("kind") == "route_same_source_target_confirmation"),
        None,
    )


def _store_pending_interaction(
    state: VersionState,
    instruction: str,
    execution: Any,
    graph: dict[str, Any],
    interaction_need: dict[str, Any],
    selection: dict[str, Any] | None = None,
    request_context: dict[str, Any] | None = None,
) -> str | None:
    interaction = interaction_need.get("interaction") if isinstance(interaction_need, dict) else {}
    if not isinstance(interaction, dict):
        state.pending_clarification = None
        return None
    interaction_type = str(interaction.get("type") or "")
    if interaction_type not in {"provide_missing_slot", "choose_candidate", "confirm_change"}:
        state.pending_clarification = None
        return None
    pending_id = uuid4().hex
    pending: dict[str, Any] = {
        "pending_id": pending_id,
        "type": interaction_type,
        "originating_version_id": state.current_version_id,
        "original_instruction": instruction,
        "selection": selection if isinstance(selection, dict) else {"node_ids": [], "edge_ids": []},
        "request_context": request_context if isinstance(request_context, dict) else {},
        "interaction": interaction,
        "created_at": time.time(),
    }
    if interaction_type == "confirm_change":
        pending["confirmation_kind"] = "route_same_source_target"
        pending["issue"] = _route_confirmation_issue(execution.specification)
    elif interaction_type == "choose_candidate":
        pending["resume_stage"] = "grounding"
        pending["semantic_goal"] = (execution.specification or {}).get("semantic_goal") if isinstance(execution.specification, dict) else None
        pending["grounding_state"] = (execution.specification or {}).get("grounded_goal") if isinstance(execution.specification, dict) else None
        pending["candidates"] = interaction_need.get("candidates") or extract_user_candidates(execution.specification, graph)
    elif interaction_type == "provide_missing_slot":
        pending["issue"] = {"slot": interaction.get("slot"), "prompt": interaction.get("prompt")}
        specification = execution.specification if isinstance(execution.specification, dict) else {}
        semantic_goal = specification.get("semantic_goal") if isinstance(specification.get("semantic_goal"), dict) else {}
        normalized = semantic_goal.get("normalized") if isinstance(semantic_goal.get("normalized"), dict) else None
        if isinstance(normalized, dict):
            pending["partial_semantic_state"] = {
                "semantic_goal": deepcopy(normalized),
                "interaction": deepcopy(interaction),
            }
    state.pending_clarification = pending
    return pending_id


def _store_pending_clarification(
    state: VersionState,
    instruction: str,
    execution: Any,
    graph: dict[str, Any],
    selection: dict[str, Any] | None = None,
    request_context: dict[str, Any] | None = None,
) -> str | None:
    interaction_need = derive_interaction_need(
        execution.status,
        execution.specification if isinstance(execution.specification, dict) else None,
        graph,
        getattr(execution, "clarification_message", None),
        getattr(execution, "error", None),
    )
    return _store_pending_interaction(state, instruction, execution, graph, interaction_need, selection, request_context)


def _clarification_response(state: VersionState, session_id: str, message: str, status: str = "needs_clarification") -> dict[str, Any]:
    feedback = build_user_feedback(status, clarification_message=message, graph=state.current_graph())
    return {
        "status": status,
        "error": None,
        "clarification_message": message,
        "pipeline": os.getenv("AI_EDIT_PIPELINE", "legacy").strip().lower() or "legacy",
        "session_id": session_id,
        "current_version_id": state.current_version_id,
        "graph": state.current_graph(),
        "state": as_jsonable(state),
        "user_feedback": feedback,
    }


def create_session(payload: dict[str, Any]) -> dict[str, Any]:
    graph = payload.get("graph")
    if not isinstance(graph, dict) or not graph_is_valid(graph):
        return {"status": "error", "error": "invalid initial graph"}
    session_id = uuid4().hex
    SESSIONS[session_id] = create_version_state(graph)
    return {"status": "success", **_session_payload(session_id, SESSIONS[session_id])}


def run_edit(payload: dict[str, Any]) -> dict[str, Any]:
    session_id, state = _get_session(payload)
    if state is None or session_id is None:
        return {"status": "error", "error": "missing or unknown session_id"}
    instruction = str(payload.get("instruction", ""))
    selection = payload.get("selection") if isinstance(payload.get("selection"), dict) else {"node_ids": [], "edge_ids": []}
    request_context = payload.get("request_context") if isinstance(payload.get("request_context"), dict) else {}
    before = state.current_graph()
    if _pending_is_stale(state):
        state.pending_clarification = None
        return _clarification_response(state, session_id, "流程已经发生变化，请重新选择或重新描述。")
    pending_type = _pending_interaction_type(state.pending_clarification)
    if isinstance(state.pending_clarification, dict) and pending_type in {"confirmation", "confirm_change"}:
        if _looks_like_confirmation_yes(instruction):
            pending_selection = state.pending_clarification.get("selection")
            if isinstance(pending_selection, dict):
                selection = pending_selection
            pending_context = state.pending_clarification.get("request_context")
            if isinstance(pending_context, dict):
                request_context = {**pending_context, **request_context}
            request_context = dict(request_context)
            request_context["allow_route_same_target_confirmation"] = True
            instruction = str(state.pending_clarification.get("original_instruction", instruction))
        elif _looks_like_confirmation_no(instruction):
            state.pending_clarification = None
            return _clarification_response(state, session_id, "已取消上一轮待确认的修改。", status="cancelled")
    pending_selection = _parse_candidate_selection(instruction, state.pending_clarification)
    if pending_type == "choose_candidate":
        pending_selection = _resolve_pending_candidate(instruction, selection, state.pending_clarification)
        if pending_selection is None:
            return _candidate_pending_response(state, session_id, state.pending_clarification)
        if pending_selection.get("invalid"):
            return _candidate_pending_response(state, session_id, state.pending_clarification)
        if not _candidate_exists(before, pending_selection):
            state.pending_clarification = None
            return _clarification_response(state, session_id, "上一次候选对象已经不存在，请重新描述要修改的对象。")
        request_context = _context_with_pending_candidate_binding(request_context, state.pending_clarification, pending_selection)
        pending_instruction = state.pending_clarification.get("original_instruction")
        if isinstance(pending_instruction, str) and pending_instruction.strip():
            instruction = pending_instruction
        pending_selection_snapshot = state.pending_clarification.get("selection")
        if isinstance(pending_selection_snapshot, dict):
            selection = pending_selection_snapshot
    elif pending_selection is not None:
        if pending_selection.get("invalid"):
            return _clarification_response(state, session_id, "没有找到对应编号的候选，请重新选择。")
        if not _candidate_exists(before, pending_selection):
            state.pending_clarification = None
            return _clarification_response(state, session_id, "上一次候选对象已经不存在，请重新描述要修改的对象。")
        request_context = _context_with_candidate_binding(request_context, pending_selection)
        instruction = str(state.pending_clarification.get("original_instruction", instruction))
    elif state.pending_clarification is None and _looks_like_candidate_selection(instruction):
        return _clarification_response(state, session_id, "当前没有待确认的候选，请先描述你想修改哪个对象。")
    elif pending_type == "provide_missing_slot":
        instruction, selection, request_context, resumed, resume_message = _resume_missing_slot_interaction(
            state.pending_clarification,
            instruction,
            selection,
            request_context,
            before,
        )
        if not resumed:
            return _clarification_response(state, session_id, resume_message or "请补充需要修改的对象或目标。")
    pipeline = os.getenv("AI_EDIT_PIPELINE", "legacy").strip().lower() or "legacy"
    if pipeline not in {"legacy", "semantic"}:
        pipeline = "legacy"
    try:
        runner = run_versioned_semantic_transaction if pipeline == "semantic" else run_versioned_transaction
        result = runner(
            state,
            revision=len(state.version_order),
            selection=selection,
            instruction=instruction,
            request_context=request_context,
        )
    except Exception as error:
        message = _llm_error_message(error)
        return {
            "status": "llm_error",
            "error": message,
            "pipeline": pipeline,
            "session_id": session_id,
            "current_version_id": state.current_version_id,
            "graph": before,
            "state": as_jsonable(state),
        }
    execution = result.execution
    after = result.committed_version.graph_snapshot if result.committed_version else before
    response_diff = graph_diff(before, after)
    feedback_specification = deepcopy(execution.specification) if isinstance(execution.specification, dict) else execution.specification
    if isinstance(feedback_specification, dict) and any(
        bool(response_diff.get(key))
        for key in ("added_nodes", "removed_nodes", "changed_nodes", "added_edges", "removed_edges", "changed_edges")
    ):
        trace = feedback_specification.get("trace")
        if not isinstance(trace, dict):
            trace = {}
            feedback_specification["trace"] = trace
        trace["graph_diff"] = response_diff
    interaction_need = derive_interaction_need(execution.status, feedback_specification, after, execution.clarification_message, execution.error)
    interaction = interaction_need.get("interaction") if isinstance(interaction_need.get("interaction"), dict) else {}
    interaction_type = str(interaction.get("type") or "") if isinstance(interaction, dict) else ""
    public_status = execution.status
    if interaction_type in {"provide_missing_slot", "choose_candidate", "confirm_change"}:
        public_status = "needs_clarification"
        result.transaction.status = public_status
        if state.transactions and state.transactions[-1].transaction_id == result.transaction.transaction_id:
            state.transactions[-1].status = public_status
    feedback = build_user_feedback(public_status, feedback_specification, after, execution.clarification_message, execution.error)
    pending_id = None
    if interaction_type in {"provide_missing_slot", "choose_candidate", "confirm_change"}:
        pending_id = _store_pending_interaction(state, instruction, execution, after, interaction_need, selection, request_context)
    elif execution.status in {"success", "partial_success"}:
        state.pending_clarification = None
    public_question = ""
    if isinstance(feedback.get("clarification"), dict):
        public_question = str(feedback["clarification"].get("question") or "")
    public_error = public_question or str(feedback.get("message") or "")
    response = {
        "status": public_status,
        "error": public_error if execution.error else None,
        "clarification_message": public_question if execution.clarification_message else None,
        "message": feedback.get("message"),
        "user_feedback": feedback,
        "pending_id": pending_id,
        "pipeline": pipeline,
        "session_id": session_id,
        "current_version_id": state.current_version_id,
        "transaction": asdict(result.transaction),
        "committed_version": asdict(result.committed_version) if result.committed_version else None,
        "steps": execution.steps,
        "final_graph": execution.final_graph,
        "graph": state.current_graph(),
        "target": execution.target,
        "graph_context": execution.graph_context,
        "specification": execution.specification,
        "mutation_authorization": execution.mutation_authorization,
        "diff": response_diff,
        "transaction_diff": transaction_diff(result.transaction, before, after) if result.committed_version else None,
        "state": as_jsonable(state),
    }
    if os.getenv("SEMANTIC_SHADOW") == "1":
        try:
            response["semantic_shadow"] = run_semantic_shadow(before, instruction, selection, request_context)
        except Exception as error:
            response["semantic_shadow"] = {
                "enabled": True,
                "stage": "shadow_error",
                "failure_stage": "shadow_error",
                "failure_reason": str(error),
                "instruction": instruction,
                "selection": selection,
                "request_context": request_context,
                "before_graph": before,
            }
    return response


def run_undo(payload: dict[str, Any]) -> dict[str, Any]:
    session_id, state = _get_session(payload)
    if state is None or session_id is None:
        return {"status": "error", "error": "missing or unknown session_id"}
    version = undo(state)
    return {"status": "success" if version else "noop", **_session_payload(session_id, state)}


def run_redo(payload: dict[str, Any]) -> dict[str, Any]:
    session_id, state = _get_session(payload)
    if state is None or session_id is None:
        return {"status": "error", "error": "missing or unknown session_id"}
    version = redo(state)
    return {"status": "success" if version else "noop", **_session_payload(session_id, state)}


def run_large_graph_test(payload: dict[str, Any]) -> dict[str, Any]:
    graph = generate_complex_workflow()
    session_id = uuid4().hex
    state = create_version_state(graph)
    SESSIONS[session_id] = state
    cases = build_cases(graph)
    limit = int(payload.get("limit", len(cases)))
    rows: list[dict[str, Any]] = []
    for case in cases[:limit]:
        before = state.current_graph()
        started = time.perf_counter()
        try:
            result = run_versioned_transaction(
                state,
                revision=len(state.version_order),
                selection=case.selection,
                instruction=case.request,
                request_context=case.request_context,
            )
            latency = time.perf_counter() - started
            final_graph = result.execution.final_graph
            rows.append(
                {
                    "case": case.name,
                    "request": case.request,
                    "status": result.execution.status,
                    "error": result.execution.error,
                    "latency": latency,
                    "node_count": len(before.get("nodes", [])),
                    "edge_count": len(before.get("edges", [])),
                    "valid_final_graph": graph_is_valid(final_graph) if isinstance(final_graph, dict) else False,
                    "committed_version_id": result.committed_version.version_id if result.committed_version else None,
                    "step_count": len(result.execution.steps),
                    "diff": graph_diff(before, result.committed_version.graph_snapshot) if result.committed_version else None,
                }
            )
        except Exception as error:
            rows.append(
                {
                    "case": case.name,
                    "request": case.request,
                    "status": "llm_error",
                    "error": f"{type(error).__name__}: {error}",
                    "node_count": len(before.get("nodes", [])),
                    "edge_count": len(before.get("edges", [])),
                }
            )
            break
    return {
        "status": "success",
        "session_id": session_id,
        "current_version_id": state.current_version_id,
        "graph": state.current_graph(),
        "large_graph": {
            "node_count": len(state.current_graph().get("nodes", [])),
            "edge_count": len(state.current_graph().get("edges", [])),
            "cases": rows,
        },
        "state": as_jsonable(state),
    }


class DemoHandler(SimpleHTTPRequestHandler):
    def do_OPTIONS(self) -> None:
        _json_response(self, 204, {})

    def do_GET(self) -> None:
        if urlparse(self.path).path == "/api/health":
            _json_response(self, 200, {"status": "ok", "mode": "real_llm"})
            return
        super().do_GET()

    def do_POST(self) -> None:
        routes = {
            "/api/session/new": create_session,
            "/api/diagnose": diagnose_graph,
            "/api/ai-edit": run_edit,
            "/api/undo": run_undo,
            "/api/redo": run_redo,
            "/api/large-graph-test": run_large_graph_test,
        }
        route = routes.get(urlparse(self.path).path)
        if route is None:
            _json_response(self, 404, {"status": "error", "error": "not found"})
            return
        try:
            _json_response(self, 200, route(_read_json(self)))
        except Exception as error:  # pragma: no cover - HTTP server guard
            _json_response(self, 500, {"status": "error", "error": f"{type(error).__name__}: {error}"})

    def translate_path(self, path: str) -> str:
        relative = urlparse(path).path.lstrip("/")
        if relative == "" or relative.startswith("assets/"):
            return str(DIST / (relative or "index.html"))
        target = DIST / relative
        return str(target if target.exists() else DIST / "index.html")


def main() -> None:
    if not DIST.exists():
        raise SystemExit("demo/frontend/dist does not exist. Run `npm install` and `npm run build` in demo/frontend first.")
    port = 8080
    server = ThreadingHTTPServer(("127.0.0.1", port), DemoHandler)
    print(f"AI-Flow-Canvas real LLM demo is running at http://127.0.0.1:{port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
