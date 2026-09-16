from __future__ import annotations

import json
import os
import sys
import time
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
from src.semantic_shadow import run_semantic_shadow  # noqa: E402
from src.versioning import (  # noqa: E402
    VersionState,
    as_jsonable,
    create_version_state,
    redo,
    run_versioned_transaction,
    transaction_diff,
    undo,
)

SESSIONS: dict[str, VersionState] = {}


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
    try:
        result = run_versioned_transaction(
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
            "session_id": session_id,
            "current_version_id": state.current_version_id,
            "graph": before,
            "state": as_jsonable(state),
        }
    execution = result.execution
    after = result.committed_version.graph_snapshot if result.committed_version else before
    response = {
        "status": execution.status,
        "error": execution.error,
        "clarification_message": execution.clarification_message,
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
        "diff": graph_diff(before, after),
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
