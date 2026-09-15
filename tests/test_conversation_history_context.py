from __future__ import annotations

from src import workflow_ir_runtime


def graph() -> dict:
    return {
        "nodes": [
            {"id": "a", "type": "task", "label": "A", "x": 0, "y": 0},
            {"id": "b", "type": "task", "label": "B", "x": 160, "y": 0},
        ],
        "edges": [{"id": "e_ab", "source": "a", "target": "b", "label": ""}],
    }


def test_conversation_history_reaches_understand_only(monkeypatch) -> None:
    captured_payloads: list[dict] = []

    def fake_complete(_system_prompt: str, payload: dict) -> workflow_ir_runtime.JsonRun:
        captured_payloads.append(payload)
        return workflow_ir_runtime.JsonRun("", {"requirements": []}, None, False)

    monkeypatch.setattr(workflow_ir_runtime, "_complete_json", fake_complete)
    history = [
        {"role": "user", "content": "这里应该怎么改？"},
        {"role": "assistant", "content": "建议让 A 和 B 并行后汇聚。"},
    ]

    workflow_ir_runtime.run_workflow_ir_transaction(
        graph(),
        revision=1,
        selection={"node_ids": [], "edge_ids": []},
        instruction="就按刚才建议的改",
        request_context={
            "conversation_history": history,
            "visible_area": {"left": 0, "top": 0, "right": 400, "bottom": 300},
            "graph_context": {"ignored": True},
        },
    )

    assert captured_payloads
    understand_payload = captured_payloads[0]
    assert understand_payload["conversation_history"] == history
    assert "conversation_history" not in understand_payload["request_context"]
    assert "graph_context" not in understand_payload["request_context"]
