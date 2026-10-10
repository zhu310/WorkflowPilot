from __future__ import annotations

import os
from collections import Counter
from copy import deepcopy
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import pytest

from semantic_goal_contract_helpers import semantic_goal_contract_violations
from src import semantic_shadow
from src.config import get_settings
from src.product_feedback import build_user_feedback


@dataclass(frozen=True)
class AcceptanceCase:
    case_id: str
    instruction: str
    graph: dict[str, Any]
    mock_goal: dict[str, Any]
    selection: dict[str, Any] | None = None
    request_context: dict[str, Any] | None = None


def _obj(object_id: str, object_type: str, text: str | None = None, label: str | None = None) -> dict[str, Any]:
    row = {"id": object_id, "type": object_type}
    if text is not None:
        row["text"] = text
    if label is not None:
        row["label"] = label
    return row


def _spec(goals: list[dict[str, Any]] | None = None, clarification: str | None = None) -> dict[str, Any]:
    return {
        "status": "clarification" if clarification else "resolved",
        "semantic_goals": goals or [],
        "clarification": {
            "needed": clarification is not None,
            "question": clarification or "",
            "reason": clarification or "",
        },
    }


def _state(text: str, objects: list[dict[str, Any]], states: list[dict[str, Any]], constraints: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    return {
        "id": "g1",
        "goal_category": "object_state_goal",
        "text": text,
        "objects": objects,
        "object_desired_states": states,
        "semantic_constraints": constraints or [],
    }


def _relation(text: str, objects: list[dict[str, Any]], relations: list[dict[str, Any]], constraints: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    return {
        "id": "g1",
        "goal_category": "relation_goal",
        "text": text,
        "objects": objects,
        "relation_intents": relations,
        "semantic_constraints": constraints or [],
    }


def _composite(
    text: str,
    objects: list[dict[str, Any]],
    states: list[dict[str, Any]] | None = None,
    relations: list[dict[str, Any]] | None = None,
    constraints: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "id": "g1",
        "goal_category": "composite_goal",
        "text": text,
        "objects": objects,
        "object_desired_states": states or [],
        "relation_intents": relations or [],
        "semantic_constraints": constraints or [],
    }


def _base_graph() -> dict[str, Any]:
    return {
        "nodes": [
            {"id": "start", "type": "start", "label": "开始", "x": 0, "y": 0, "width": 120, "height": 72},
            {"id": "approval", "type": "task", "label": "审批节点", "x": 170, "y": 0, "width": 120, "height": 72},
            {"id": "risk_control", "type": "task", "label": "风控", "x": 340, "y": 0, "width": 120, "height": 72},
            {"id": "risk_check", "type": "task", "label": "风险检查", "x": 510, "y": 0, "width": 120, "height": 72},
            {"id": "audit", "type": "task", "label": "审计", "x": 680, "y": 0, "width": 120, "height": 72},
            {"id": "inventory", "type": "task", "label": "库存检查", "x": 850, "y": 0, "width": 120, "height": 72},
            {"id": "shipping", "type": "task", "label": "发货", "x": 1020, "y": 0, "width": 120, "height": 72},
            {"id": "decision", "type": "decision", "label": "结果判断", "x": 1190, "y": 0, "width": 120, "height": 72},
            {"id": "success", "type": "task", "label": "成功流程", "x": 1360, "y": -100, "width": 120, "height": 72},
            {"id": "manual", "type": "task", "label": "人工处理", "x": 1360, "y": 70, "width": 120, "height": 72},
            {"id": "exception", "type": "task", "label": "异常处理", "x": 1360, "y": 240, "width": 120, "height": 72},
            {"id": "end", "type": "end", "label": "结束", "x": 1530, "y": 0, "width": 120, "height": 72},
        ],
        "edges": [
            {"id": "e_start_approval", "source": "start", "target": "approval", "label": ""},
            {"id": "e_approval_risk", "source": "approval", "target": "risk_control", "label": ""},
            {"id": "e_risk_riskcheck", "source": "risk_control", "target": "risk_check", "label": ""},
            {"id": "e_riskcheck_audit", "source": "risk_check", "target": "audit", "label": ""},
            {"id": "e_audit_inventory", "source": "audit", "target": "inventory", "label": ""},
            {"id": "e_inventory_shipping", "source": "inventory", "target": "shipping", "label": ""},
            {"id": "e_shipping_decision", "source": "shipping", "target": "decision", "label": ""},
            {"id": "e_success", "source": "decision", "target": "success", "label": "success", "branch_label": "成功"},
            {"id": "e_failure", "source": "decision", "target": "manual", "label": "failure", "branch_label": "失败"},
            {"id": "e_exception", "source": "decision", "target": "exception", "label": "exception", "branch_label": "异常"},
            {"id": "e_success_end", "source": "success", "target": "end", "label": ""},
            {"id": "e_manual_end", "source": "manual", "target": "end", "label": ""},
            {"id": "e_exception_end", "source": "exception", "target": "end", "label": ""},
        ],
    }


def _duplicate_graph() -> dict[str, Any]:
    graph = _base_graph()
    graph["nodes"].extend(
        [
            {"id": "dup_a", "type": "task", "label": "重复审批", "x": 170, "y": 220, "width": 120, "height": 72},
            {"id": "dup_b", "type": "task", "label": "重复审批", "x": 340, "y": 220, "width": 120, "height": 72},
        ]
    )
    return graph


def _multi_failure_graph() -> dict[str, Any]:
    graph = _base_graph()
    graph["nodes"].append({"id": "manual_2", "type": "task", "label": "人工处理二", "x": 1360, "y": 400, "width": 120, "height": 72})
    graph["edges"].append({"id": "e_failure_2", "source": "approval", "target": "manual_2", "label": "failure", "branch_label": "失败"})
    return graph


def _no_exception_edge_graph() -> dict[str, Any]:
    graph = _base_graph()
    graph["edges"] = [edge for edge in graph["edges"] if edge.get("id") != "e_exception"]
    return graph


def build_acceptance_cases() -> list[AcceptanceCase]:
    cases: list[AcceptanceCase] = []
    add = cases.append
    add(AcceptanceCase("T1", "增加一个人工审核节点", _base_graph(), _spec([_state("人工审核存在", [_obj("n", "new_entity", label="人工审核")], [{"object": "n", "state": "exists"}])])))
    add(AcceptanceCase("T2", "增加一个通知步骤", _base_graph(), _spec([_state("通知存在", [_obj("n", "new_entity", label="通知")], [{"object": "n", "state": "exists"}])])))
    add(AcceptanceCase("T3", "增加一个异常处理节点", _base_graph(), _spec([_state("异常处理存在", [_obj("n", "new_entity", label="异常处理")], [{"object": "n", "state": "exists"}])])))
    add(AcceptanceCase("T4", "删除库存检查", _base_graph(), _spec([_state("库存检查 absent", [_obj("n", "existing_reference", text="库存检查")], [{"object": "n", "state": "absent"}])])))
    add(AcceptanceCase("T5", "去掉重复审批", _duplicate_graph(), _spec([_state("重复审批 absent", [_obj("n", "existing_reference", text="重复审批")], [{"object": "n", "state": "absent"}])])))
    add(AcceptanceCase("T6", "把审批节点名称改成人工审核", _base_graph(), _spec([_state("审批节点改名", [_obj("n", "existing_reference", text="审批节点")], [{"object": "n", "state": "property_equals", "property": "display_label", "value": "人工审核"}])])))
    add(AcceptanceCase("T7", "把失败分支名称改成人工审核", _base_graph(), _spec([_state("失败分支改名", [_obj("e", "edge_reference", text="失败分支")], [{"object": "e", "state": "property_equals", "property": "display_label", "value": "人工审核"}])])))
    add(AcceptanceCase("T8", "把成功分支名称改成自动通过", _base_graph(), _spec([_state("成功分支改名", [_obj("e", "edge_reference", text="成功分支")], [{"object": "e", "state": "property_equals", "property": "display_label", "value": "自动通过"}])])))
    add(AcceptanceCase("T9", "把审核移动到风控后面", _base_graph(), _spec([_relation("审核 after 风控", [_obj("a", "existing_reference", text="审核"), _obj("r", "existing_reference", text="风控")], [{"source": "a", "relation": "after", "target": "r"}])])))
    add(AcceptanceCase("T10", "让风险检查和审计并行执行", _base_graph(), _spec([_relation("风险检查 parallel 审计", [_obj("a", "existing_reference", text="风险检查"), _obj("b", "existing_reference", text="审计")], [{"source": "a", "relation": "parallel_with", "target": "b"}])])))
    add(AcceptanceCase("T11", "失败流程转向人工处理", _base_graph(), _spec([_relation("失败 route", [_obj("e", "edge_reference", text="失败流程"), _obj("m", "existing_reference", text="人工处理")], [{"source": "e", "relation": "route_to", "target": "m"}])])))
    add(AcceptanceCase("T12", "异常流程接到人工处理", _base_graph(), _spec([_relation("异常 route", [_obj("e", "edge_reference", text="异常流程"), _obj("m", "existing_reference", text="人工处理")], [{"source": "e", "relation": "route_to", "target": "m"}])])))
    add(AcceptanceCase("T13", "增加人工审核节点，并让失败分支走人工审核", _base_graph(), _spec([_composite("新增并路由", [_obj("e", "edge_reference", text="失败分支"), _obj("n", "new_entity", label="人工审核")], [{"object": "n", "state": "exists"}], [{"source": "e", "relation": "route_to", "target": "n"}])])))
    add(AcceptanceCase("T14", "把审核放到风控后面，同时失败分支改成人工处理", _base_graph(), _spec([_composite("移动并改分支", [_obj("a", "existing_reference", text="审核"), _obj("r", "existing_reference", text="风控"), _obj("e", "edge_reference", text="失败分支")], [{"object": "e", "state": "property_equals", "property": "display_label", "value": "人工处理"}], [{"source": "a", "relation": "after", "target": "r"}])])))
    add(AcceptanceCase("T15", "增加通知节点，失败和异常都走通知", _base_graph(), _spec([_composite("通知多路由", [_obj("f", "edge_reference", text="失败"), _obj("x", "edge_reference", text="异常"), _obj("n", "new_entity", label="通知")], [{"object": "n", "state": "exists"}], [{"source": "f", "relation": "route_to", "target": "n"}, {"source": "x", "relation": "route_to", "target": "n"}])])))
    add(AcceptanceCase("T16", "增加通知节点，但不要影响成功流程", _base_graph(), _spec([_composite("通知 preserve success", [_obj("n", "new_entity", label="通知")], [{"object": "n", "state": "exists"}], [], [{"type": "preserve", "target": "成功流程", "text": "不要影响成功流程"}])])))
    add(AcceptanceCase("T17", "发货前增加出库确认，但保留异常处理", _base_graph(), _spec([_composite("出库确认 before 发货", [_obj("n", "new_entity", label="出库确认"), _obj("s", "existing_reference", text="发货")], [{"object": "n", "state": "exists"}], [{"source": "n", "relation": "before", "target": "s"}], [{"type": "preserve", "target": "异常处理", "text": "保留异常处理"}])])))
    add(AcceptanceCase("T18", "不改变已有审批节点，只把失败分支改成人工审核", _base_graph(), _spec([_composite("preserve approval rename failure", [_obj("a", "existing_reference", text="审批节点"), _obj("e", "edge_reference", text="失败分支")], [{"object": "a", "state": "preserved"}, {"object": "e", "state": "property_equals", "property": "display_label", "value": "人工审核"}])])))
    add(AcceptanceCase("T19", "把这个节点移动到右边", _base_graph(), _spec(clarification="请说明这个节点要放到哪个节点右边。")))
    add(AcceptanceCase("T20", "把这个节点移动到右边", _base_graph(), _spec(clarification="请说明选中的节点要放到哪个节点右边。"), {"node_ids": ["approval"], "edge_ids": []}))
    add(AcceptanceCase("T21", "给这个分支改名为人工处理", _base_graph(), _spec([_state("这个分支改名", [_obj("e", "edge_reference", text="这个分支")], [{"object": "e", "state": "property_equals", "property": "display_label", "value": "人工处理"}])])))
    add(AcceptanceCase("T22", "给这个分支改名为人工处理", _base_graph(), _spec([_state("这个分支改名", [_obj("e", "edge_reference", text="这个分支")], [{"object": "e", "state": "property_equals", "property": "display_label", "value": "人工处理"}])]), {"node_ids": [], "edge_ids": ["e_failure"]}))
    recent_graph = _base_graph()
    recent_graph["nodes"].append({"id": "manual_recent", "type": "task", "label": "人工审核", "x": 100, "y": 300, "width": 120, "height": 72})
    recent_context = {"recent_transaction_diff": {"transaction_id": "tx-prev", "created_nodes": [{"id": "manual_recent", "label": "人工审核", "type": "node"}], "changed_nodes": [], "affected_nodes": ["manual_recent"], "affected_edges": []}}
    add(AcceptanceCase("T23", "第一轮增加人工审核节点；第二轮把这个节点移动到风控后面", recent_graph, _spec([_relation("这个节点 after 风控", [_obj("n", "existing_reference", text="这个节点"), _obj("r", "existing_reference", text="风控")], [{"source": "n", "relation": "after", "target": "r"}])]), request_context=recent_context))
    multi_context = {"recent_transaction_diff": {"transaction_id": "tx-prev", "created_nodes": [], "changed_nodes": [{"id": "approval", "label": "审批节点", "type": "node"}, {"id": "risk_control", "label": "风控", "type": "node"}], "affected_nodes": ["approval", "risk_control"], "affected_edges": []}}
    add(AcceptanceCase("T24", "recent transaction 多 affected nodes；第二轮把这个节点移动到风控后面", _base_graph(), _spec([_relation("这个节点 after 风控", [_obj("n", "existing_reference", text="这个节点"), _obj("r", "existing_reference", text="风控")], [{"source": "n", "relation": "after", "target": "r"}])]), request_context=multi_context))
    add(AcceptanceCase("T25", "无 selection / 无 recent，把这个节点移动到风控后面", _base_graph(), _spec([_relation("这个节点 after 风控", [_obj("n", "existing_reference", text="这个节点"), _obj("r", "existing_reference", text="风控")], [{"source": "n", "relation": "after", "target": "r"}])])))
    add(AcceptanceCase("T26", "失败情况转人工审核", _base_graph(), _spec([_relation("失败情况 route", [_obj("e", "edge_reference", text="失败情况"), _obj("m", "new_entity", label="人工审核")], [{"source": "e", "relation": "route_to", "target": "m"}], [])])))
    add(AcceptanceCase("T27", "异常流程接到人工处理", _base_graph(), _spec([_relation("异常流程 route", [_obj("e", "edge_reference", text="异常流程"), _obj("m", "existing_reference", text="人工处理")], [{"source": "e", "relation": "route_to", "target": "m"}])])))
    add(AcceptanceCase("T28", "多个 failure edge，失败流程转人工审核", _multi_failure_graph(), _spec([_relation("失败流程 route", [_obj("e", "edge_reference", text="失败流程"), _obj("m", "new_entity", label="人工审核")], [{"source": "e", "relation": "route_to", "target": "m"}])])))
    add(AcceptanceCase("T29", "这个分支转人工处理", _base_graph(), _spec([_relation("这个分支 route", [_obj("e", "edge_reference", text="这个分支"), _obj("m", "existing_reference", text="人工处理")], [{"source": "e", "relation": "route_to", "target": "m"}])]), {"node_ids": [], "edge_ids": ["e_exception"]}))
    add(AcceptanceCase("T30", "无异常 edge，异常流程接到人工处理", _no_exception_edge_graph(), _spec([_relation("异常流程 route", [_obj("e", "edge_reference", text="异常流程"), _obj("m", "existing_reference", text="人工处理")], [{"source": "e", "relation": "route_to", "target": "m"}])])))
    add(AcceptanceCase("T31", "优化一下这个流程", _base_graph(), _spec(clarification="请说明你希望优化的具体目标。")))
    add(AcceptanceCase("T32", "让流程更合理，但不要删节点", _base_graph(), _spec(clarification="请先选择一个具体优化方向。")))
    add(AcceptanceCase("T33", "给我几个优化方案", _base_graph(), _spec(clarification="这是建议请求，请先通过诊断链路确认方案。")))
    assert len(cases) == 33
    return cases


def _classify(trace: dict[str, Any], feedback: dict[str, Any], contract_violations: list[dict[str, str]]) -> str:
    if contract_violations:
        return "C"
    validation = trace.get("validation") if isinstance(trace.get("validation"), dict) else {}
    issues = validation.get("issues") if isinstance(validation.get("issues"), list) else trace.get("failure_reason")
    issue_kinds = {
        issue.get("kind")
        for issue in issues
        if isinstance(issue, dict)
    } if isinstance(issues, list) else set()
    if "semantic_goal_contract_violation" in issue_kinds:
        return "C"
    if not feedback.get("message"):
        return "G"
    result = semantic_shadow.semantic_trace_to_transaction_result(trace)
    if result.status in {"success", "partial_success"}:
        return "A"
    if result.status == "needs_clarification":
        return "B"
    if result.status == "unsupported" and result.error == "no graph change was produced by the executable plan":
        return "E"
    if trace.get("failure_stage") in {"understand"}:
        return "C"
    if trace.get("failure_stage") == "validation":
        return "E"
    if trace.get("failure_stage") in {"compile", "planner", "executor", "input"}:
        return "F"
    return "H"


def _run_case(case: AcceptanceCase, mode: str, monkeypatch: pytest.MonkeyPatch | None = None) -> dict[str, Any]:
    if mode == "MOCK_SEMANTIC_GOAL":
        assert monkeypatch is not None
        monkeypatch.setattr(
            semantic_shadow,
            "_complete_json",
            lambda _system, _payload, parsed=deepcopy(case.mock_goal): SimpleNamespace(
                raw_response="",
                parsed_json=parsed,
                parse_error=None,
                normalized_from_fence=False,
            ),
        )
    trace = semantic_shadow.run_semantic_shadow(
        deepcopy(case.graph),
        case.instruction,
        deepcopy(case.selection or {"node_ids": [], "edge_ids": []}),
        deepcopy(case.request_context or {}),
    )
    result = semantic_shadow.semantic_trace_to_transaction_result(trace)
    parsed = trace.get("semantic_goal", {}).get("parsed") if isinstance(trace.get("semantic_goal"), dict) else None
    contract_violations = semantic_goal_contract_violations(parsed) if isinstance(parsed, dict) else []
    feedback = build_user_feedback(result.status, result.specification, trace.get("final_shadow_graph") or case.graph, result.clarification_message, result.error)
    classification = _classify(trace, feedback, contract_violations)
    return {
        "case": case.case_id,
        "instruction": case.instruction,
        "mode": mode,
        "semantic_goal": parsed,
        "validation": trace.get("goal_validation") or trace.get("validation"),
        "forbidden_check": contract_violations,
        "grounding": (trace.get("grounded_goal") or {}).get("grounding") if isinstance(trace.get("grounded_goal"), dict) else None,
        "lowering": trace.get("lowering_result"),
        "partial_execution": (result.specification or {}).get("partial_execution") if isinstance(result.specification, dict) else None,
        "product_feedback": feedback,
        "final_status": result.status,
        "classification": classification,
        "reason": result.error or result.clarification_message or "",
    }


def _summary(rows: list[dict[str, Any]]) -> str:
    counts = Counter(row["classification"] for row in rows)
    lines = [f"counts={dict(counts)}"]
    for row in rows:
        feedback = row.get("product_feedback", {}).get("message", "").splitlines()[0] if isinstance(row.get("product_feedback"), dict) else ""
        lines.append(
            f"{row['case']}|{row['mode']}|{row['final_status']}|{row['classification']}|"
            f"violations={row['forbidden_check']}|feedback={feedback}|reason={row['reason']}"
        )
    return "\n".join(lines)


def test_mock_semantic_goal_33_case_full_pipeline(monkeypatch: pytest.MonkeyPatch) -> None:
    rows = [_run_case(case, "MOCK_SEMANTIC_GOAL", monkeypatch) for case in build_acceptance_cases()]
    bad = [row for row in rows if row["classification"] in {"C", "D", "F", "G", "H"}]
    assert not bad, _summary(rows)
    assert any(row["final_status"] == "partial_success" for row in rows), _summary(rows)
    assert any(row["final_status"] == "needs_clarification" for row in rows), _summary(rows)


@pytest.mark.skipif(os.getenv("RUN_REAL_LLM_TESTS") != "1", reason="REAL_LLM_FULL_PIPELINE_NOT_RUN: set RUN_REAL_LLM_TESTS=1")
def test_real_llm_33_case_full_pipeline() -> None:
    try:
        get_settings()
    except RuntimeError as error:
        pytest.skip(f"REAL_LLM_FULL_PIPELINE_NOT_RUN: {error}")
    rows = [_run_case(case, "REAL_LLM_FULL_PIPELINE") for case in build_acceptance_cases()]
    bad = [row for row in rows if row["classification"] in {"C", "D", "F", "G", "H"}]
    assert not bad, _summary(rows)
