from __future__ import annotations

from src import mvp23


def test_resolve_clarification_answer_locally_for_binary_choice() -> None:
    result = mvp23.resolve_clarification_answer(
        "允许修改未选中的付款节点吗？",
        [
            {"key": "allow_unselected", "value": "true", "label": "允许一起修改"},
            {"key": "allow_unselected", "value": "false", "label": "只改选中对象"},
        ],
        "不要修改付款，只改选中的预算节点。",
        {"nodes": [], "edges": []},
        {"node_ids": ["budget"], "edge_ids": []},
        "只修改我选中的节点，同时把付款节点往右移动 120。",
    )
    assert result.parsed_json is not None
    assert result.parsed_json["status"] == "resolved"
    assert result.parsed_json["resolved_constraint"] == {
        "key": "allow_unselected",
        "value": "false",
        "source": "user_clarification",
    }


def test_resume_pending_request_persists_resolved_constraint(monkeypatch) -> None:
    interpretation = {
        "status": "needs_clarification",
        "question": "允许修改未选中的付款节点吗？",
        "constraint_key": "allow_unselected",
        "candidate_constraints": [
            {"key": "allow_unselected", "value": "true", "label": "允许一起修改"},
            {"key": "allow_unselected", "value": "false", "label": "只改选中对象"},
        ],
        "target": {"node_ids": ["budget"], "edge_ids": []},
        "intent": {"kind": "move"},
        "policy": {},
    }
    clear = {
        "status": "clear",
        "question": "",
        "target": {"node_ids": ["budget"], "edge_ids": []},
        "intent": {"kind": "move"},
        "policy": {},
    }
    monkeypatch.setattr(mvp23, "run_edit_interpretation", lambda *args, **kwargs: mvp23.JsonRun("", clear, None, False))
    pending = mvp23.create_pending_request(
        "req1",
        {"nodes": [], "edges": []},
        1,
        {"node_ids": ["budget"], "edge_ids": []},
        "只修改我选中的节点，同时把付款节点往右移动 120。",
        None,
        interpretation,
    )
    updated, resolution, followup = mvp23.resume_pending_request(pending, "不要修改付款，只改选中的预算节点。", {"nodes": [], "edges": []}, 2)
    assert resolution.parsed_json is not None
    assert resolution.parsed_json["status"] == "resolved"
    assert updated["resolved_constraints"] == [{"key": "allow_unselected", "value": "false", "source": "user_clarification"}]
    assert followup.parsed_json is not None
    assert followup.parsed_json["status"] == "clear"
