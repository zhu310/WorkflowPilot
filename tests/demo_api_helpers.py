from __future__ import annotations

import re
from types import SimpleNamespace

from demo import server
from src import semantic_shadow


INTERNAL_TOKENS = (
    "property_equals",
    "route_to",
    "right_of",
    "grounding",
    "executable binding",
    "semantic relation could not be lowered",
    "supported semantic relation",
    "supported semantic state",
    "relation intent was not lowered",
    "relation intent",
    "object references must preserve phrases from the user instruction",
    "semantic understanding must not pre-resolve graph objects",
    "semantic goal contract",
    "semantic goal",
    "lowering",
    "not lowered",
    "must preserve phrases",
    "pre-resolve graph objects",
    "validator",
    "deterministic",
    "multiple graph edges match reference",
    "no graph edge matches reference",
    "no graph node matches reference",
    "ambiguous_edge_reference",
    "missing_edge_reference",
    "ambiguous_node_reference",
    "missing_node_reference",
    "workflow_ir",
    "inserted",
    "connection:",
    "branch:",
    "between",
    "traceback",
    "exception",
)
INTERNAL_PATTERNS = (
    re.compile(r"\bo\d+\b", re.IGNORECASE),
    re.compile(r"\bg\d+\b", re.IGNORECASE),
    re.compile(r"\be_[A-Za-z0-9_]+\b"),
    re.compile(r"\{[^{}]*['\"][A-Za-z_]+['\"]\s*:", re.IGNORECASE),
)


def mock_understanding(monkeypatch, spec: dict) -> None:
    monkeypatch.setattr(
        semantic_shadow,
        "_complete_json",
        lambda _system, _payload: SimpleNamespace(
            raw_response="",
            parsed_json=spec,
            parse_error=None,
            normalized_from_fence=False,
        ),
    )


def create_session(graph: dict) -> str:
    result = server.create_session({"graph": graph})
    assert result["status"] == "success"
    return result["session_id"]


def assert_readable_response(response: dict) -> None:
    message = str(response.get("message") or "")
    feedback_message = str(response.get("user_feedback", {}).get("message") or "")
    assert message
    assert feedback_message
    assert message == feedback_message
    assert_no_internal_visible_text(message)
    assert_no_internal_visible_text(feedback_message)
    feedback = response.get("user_feedback", {})
    if isinstance(feedback, dict):
        for row in feedback.get("executed", []) if isinstance(feedback.get("executed"), list) else []:
            if isinstance(row, dict):
                assert_no_internal_visible_text(str(row.get("summary") or ""))
        for row in feedback.get("incomplete", []) if isinstance(feedback.get("incomplete"), list) else []:
            if isinstance(row, dict):
                assert_no_internal_visible_text(str(row.get("summary") or ""))
                assert_no_internal_visible_text(str(row.get("reason") or ""))
        clarification = feedback.get("clarification")
        if isinstance(clarification, dict):
            assert_no_internal_visible_text(str(clarification.get("question") or ""))
            for group in clarification.get("candidates", []) if isinstance(clarification.get("candidates"), list) else []:
                if not isinstance(group, dict):
                    continue
                assert_no_internal_visible_text(str(group.get("reason") or ""))
                for candidate in group.get("candidates", []) if isinstance(group.get("candidates"), list) else []:
                    if isinstance(candidate, dict):
                        assert_no_internal_visible_text(str(candidate.get("label") or ""))
                        assert_no_internal_visible_text(str(candidate.get("reason") or ""))


def assert_no_internal_visible_text(text: str) -> None:
    lowered = text.lower()
    for token in INTERNAL_TOKENS:
        assert token not in lowered
    for pattern in INTERNAL_PATTERNS:
        assert pattern.search(text) is None


def demo_graph() -> dict:
    return {
        "nodes": [
            {"id": "n_start", "type": "start", "label": "用户需求", "x": 110, "y": 260, "width": 96, "height": 96},
            {"id": "n_submit", "type": "task", "label": "解析任务", "x": 280, "y": 260, "width": 140, "height": 72},
            {"id": "n_approval_decision", "type": "decision", "label": "结果判断", "x": 680, "y": 260, "width": 104, "height": 104},
            {"id": "n_report", "type": "llm_reasoning", "label": "生成报告", "x": 890, "y": 220, "width": 140, "height": 72},
            {"id": "n_publish", "type": "output", "label": "发布结果", "x": 1090, "y": 220, "width": 140, "height": 72},
            {"id": "n_human", "type": "human_review", "label": "人工复核", "x": 890, "y": 420, "width": 140, "height": 72},
            {"id": "n_end", "type": "end", "label": "结束", "x": 1280, "y": 260, "width": 96, "height": 96},
        ],
        "edges": [
            {"id": "e_start_submit", "source": "n_start", "target": "n_submit", "label": ""},
            {"id": "e_submit_decision", "source": "n_submit", "target": "n_approval_decision", "label": ""},
            {"id": "e_decision_report", "source": "n_approval_decision", "target": "n_report", "label": "success"},
            {"id": "e_decision_human", "source": "n_approval_decision", "target": "n_human", "label": "failure"},
            {"id": "e_report_publish", "source": "n_report", "target": "n_publish", "label": ""},
            {"id": "e_report_human", "source": "n_report", "target": "n_human", "label": "failure"},
            {"id": "e_human_publish", "source": "n_human", "target": "n_publish", "label": ""},
            {"id": "e_publish_end", "source": "n_publish", "target": "n_end", "label": ""},
        ],
    }
