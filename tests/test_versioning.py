from __future__ import annotations

from src import mvp23
from src.versioning import as_jsonable, create_version_state, redo, rollback_to_version, run_versioned_transaction, undo


def graph() -> dict:
    return {
        "nodes": [
            {"id": "start", "type": "start", "label": "Start", "x": 0, "y": 0},
            {"id": "review", "type": "task", "label": "Review", "x": 100, "y": 0},
            {"id": "end", "type": "end", "label": "End", "x": 200, "y": 0},
        ],
        "edges": [
            {"id": "e1", "source": "start", "target": "review", "label": ""},
            {"id": "e2", "source": "review", "target": "end", "label": ""},
        ],
    }


def test_version_commit_undo_redo_and_rollback(monkeypatch) -> None:
    state = create_version_state(graph(), created_at="2026-08-27T00:00:00")
    interpretation = {"status": "clear", "target": {"node_ids": ["review"], "edge_ids": []}, "intent": {"kind": "rename"}, "policy": {}}
    specification = {"status": "ready", "constraints": {"entities": [], "properties": [{"node_id": "review", "changes": {"label": "Approved"}}], "required_relations": [], "forbidden_relations": []}, "directives": []}
    monkeypatch.setattr(mvp23, "run_edit_interpretation", lambda *args, **kwargs: mvp23.JsonRun("", interpretation, None, False))
    monkeypatch.setattr(mvp23, "finalize_edit_specification", lambda *args, **kwargs: mvp23.JsonRun("", specification, None, False))

    committed = run_versioned_transaction(state, 1, {"node_ids": ["review"], "edge_ids": []}, "rename review")
    assert committed.execution.status == "success"
    assert committed.committed_version is not None
    assert state.current_version_id == "v2"
    assert state.current_graph()["nodes"][1]["label"] == "Approved"

    undone = undo(state)
    assert undone is not None
    assert state.current_version_id == "v1"
    assert state.current_graph()["nodes"][1]["label"] == "Review"

    redone = redo(state)
    assert redone is not None
    assert state.current_version_id == "v2"
    assert state.current_graph()["nodes"][1]["label"] == "Approved"

    rolled = rollback_to_version(state, "v1")
    assert rolled is not None
    assert state.current_version_id == "v1"
    assert state.current_graph()["nodes"][1]["label"] == "Review"


def test_new_edit_after_undo_invalidates_linear_redo(monkeypatch) -> None:
    state = create_version_state(graph(), created_at="2026-08-27T00:00:00")
    interpretation = {"status": "clear", "target": {"node_ids": ["review"], "edge_ids": []}, "intent": {"kind": "rename"}, "policy": {}}
    first_spec = {"status": "ready", "constraints": {"entities": [], "properties": [{"node_id": "review", "changes": {"label": "Approved"}}], "required_relations": [], "forbidden_relations": []}, "directives": []}
    second_spec = {"status": "ready", "constraints": {"entities": [], "properties": [{"node_id": "review", "changes": {"label": "Checked"}}], "required_relations": [], "forbidden_relations": []}, "directives": []}
    monkeypatch.setattr(mvp23, "run_edit_interpretation", lambda *args, **kwargs: mvp23.JsonRun("", interpretation, None, False))
    monkeypatch.setattr(mvp23, "finalize_edit_specification", lambda *args, **kwargs: mvp23.JsonRun("", first_spec, None, False))
    run_versioned_transaction(state, 1, {"node_ids": ["review"], "edge_ids": []}, "rename review")
    undo(state)
    monkeypatch.setattr(mvp23, "finalize_edit_specification", lambda *args, **kwargs: mvp23.JsonRun("", second_spec, None, False))
    run_versioned_transaction(state, 2, {"node_ids": ["review"], "edge_ids": []}, "rename review again")

    assert state.current_version_id == "v2"
    assert redo(state) is None
    payload = as_jsonable(state)
    assert payload["version_order"] == ["v1", "v2"]
    assert payload["transactions"][-1]["resulting_version_id"] == "v2"
