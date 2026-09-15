from __future__ import annotations

import json

from demo import server
from src.semantic_diagnosis import diagnose_workflow, focus_graph_summary, graph_summary


class FakeAdapter:
    def __init__(self, payload: dict) -> None:
        self.payload = payload
        self.calls: list[tuple[str, str]] = []

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        self.calls.append((system_prompt, user_prompt))
        return json.dumps(self.payload, ensure_ascii=False)


def graph() -> dict:
    return {
        "nodes": [
            {"id": "n_report", "type": "llm_reasoning", "label": "生成报告", "x": 0, "y": 0},
            {"id": "n_human", "type": "human_review", "label": "人工复核", "x": 100, "y": 0},
            {"id": "n_publish", "type": "output", "label": "发布结果", "x": 200, "y": 0},
        ],
        "edges": [
            {"id": "e1", "source": "n_report", "target": "n_publish", "label": ""},
            {"id": "e2", "source": "n_report", "target": "n_human", "label": "failure"},
            {"id": "e3", "source": "n_human", "target": "n_publish", "label": ""},
        ],
    }


def full_graph() -> dict:
    return {
        "nodes": [
            {"id": "n_approval_decision", "type": "decision", "label": "结果判断", "x": 0, "y": 0},
            {"id": "n_report", "type": "llm_reasoning", "label": "生成报告", "x": 100, "y": 0},
            {"id": "n_human", "type": "human_review", "label": "人工复核", "x": 100, "y": 100},
            {"id": "n_publish", "type": "output", "label": "发布结果", "x": 200, "y": 0},
        ],
        "edges": [
            {"id": "e_decision_report", "source": "n_approval_decision", "target": "n_report", "label": "success"},
            {"id": "e_decision_human", "source": "n_approval_decision", "target": "n_human", "label": "failure"},
            {"id": "e_report_publish", "source": "n_report", "target": "n_publish", "label": ""},
            {"id": "e_report_human", "source": "n_report", "target": "n_human", "label": "failure"},
            {"id": "e_human_publish", "source": "n_human", "target": "n_publish", "label": ""},
        ],
    }


def test_semantic_diagnosis_normalizes_llm_options_without_executing() -> None:
    adapter = FakeAdapter(
        {
            "status": "needs_confirmation",
            "analysis": "当前流程可能存在发布绕过人工复核的问题。",
            "options": [
                {
                    "id": "option_1",
                    "title": "严格复核",
                    "description": "生成报告后先人工复核，再发布结果。",
                    "tradeoff": "合规性强，自动化程度低",
                    "draft_instruction": "把 n_report、n_human、n_publish 改成 n_report -> n_human -> n_publish。",
                }
            ],
            "recommended_option_id": "option_1",
            "question": "是否按严格复核方案修改？",
        }
    )

    result = diagnose_workflow(graph(), "这里是不是逻辑有问题？", adapter=adapter)

    assert result["status"] == "needs_confirmation"
    diagnosis = result["diagnosis"]
    assert diagnosis["requires_confirmation"] is True
    assert diagnosis["recommended_instruction"].startswith("把 n_report")
    assert diagnosis["options"][0]["title"] == "严格复核"
    assert "graph_summary" in adapter.calls[0][1]


def test_demo_diagnose_endpoint_delegates_to_semantic_diagnosis(monkeypatch) -> None:
    def fake_diagnose(graph_data, message, history, focus_context):
        assert graph_data["nodes"][0]["id"] == "n_report"
        assert message == "哪种改法更合理？"
        assert history == []
        assert focus_context == {"changed_area": {"node_ids": ["n_report"]}}
        return {
            "status": "needs_confirmation",
            "diagnosis": {
                "analysis": "建议先确认发布是否必须人工复核。",
                "options": [
                    {
                        "id": "option_1",
                        "title": "严格复核",
                        "description": "生成报告后先人工复核，再发布。",
                        "tradeoff": "更安全",
                        "draft_instruction": "把 n_report、n_human、n_publish 改成 n_report -> n_human -> n_publish。",
                    }
                ],
                "recommended_option_id": "",
                "recommended_instruction": "",
                "question": "是否需要所有发布都经过人工复核？",
                "requires_confirmation": True,
            },
        }

    monkeypatch.setattr(server, "diagnose_workflow", fake_diagnose)

    result = server.diagnose_graph(
        {
            "graph": graph(),
            "message": "哪种改法更合理？",
            "focus_context": {"changed_area": {"node_ids": ["n_report"]}},
        }
    )

    assert result["status"] == "needs_confirmation"
    assert result["diagnosis"]["question"] == "是否需要所有发布都经过人工复核？"
    assert "draft_instruction" not in result["diagnosis"]["options"][0]
    assert result["diagnosis"]["option_instructions"] == {
        "option_1": "把 n_report、n_human、n_publish 改成 n_report -> n_human -> n_publish。"
    }


def test_diagnosis_graph_summary_includes_join_state() -> None:
    summary = graph_summary(
        {
            "nodes": [
                {
                    "id": "n_approval_decision",
                    "type": "decision",
                    "label": "结果判断",
                    "join": {"mode": "all"},
                }
            ],
            "edges": [],
        }
    )

    assert summary["nodes"][0]["join"] == {"mode": "all"}


def test_focus_graph_summary_resolves_changed_area() -> None:
    graph_data = {
        "nodes": [
            {"id": "n_submit", "label": "解析任务"},
            {"id": "n_retrieval", "label": "知识检索"},
            {"id": "n_tool", "label": "工具调用"},
            {"id": "n_approval_decision", "label": "结果判断", "join": {"mode": "all"}},
        ],
        "edges": [
            {"id": "e1", "source": "n_submit", "target": "n_retrieval", "label": ""},
            {"id": "e2", "source": "n_submit", "target": "n_tool", "label": ""},
            {"id": "e3", "source": "n_retrieval", "target": "n_approval_decision", "label": ""},
            {"id": "e4", "source": "n_tool", "target": "n_approval_decision", "label": ""},
        ],
    }

    summary = focus_graph_summary(graph_data, {"changed_area": {"node_ids": ["n_approval_decision"]}})

    assert {edge["id"] for edge in summary["edges"]} == {"e3", "e4"}
    assert any(node["join"] == {"mode": "all"} for node in summary["nodes"])


def test_demo_server_llm_errors_are_user_facing_chinese() -> None:
    message = server._llm_error_message(RuntimeError('DeepSeek HTTP 402: {"message":"Insufficient Balance"}'))

    assert "余额不足" in message
    assert "Insufficient Balance" not in message
