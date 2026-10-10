from __future__ import annotations

from types import SimpleNamespace

from demo import server
from src import mvp23, semantic_shadow
from src.versioning import create_version_state, redo, run_versioned_semantic_transaction, undo


def graph() -> dict:
    return {
        "nodes": [
            {"id": "start", "type": "start", "label": "Start", "x": 0, "y": 0},
            {"id": "review", "type": "task", "label": "Review", "x": 100, "y": 0},
            {"id": "end", "type": "end", "label": "End", "x": 200, "y": 0},
        ],
        "edges": [
            {"id": "e_start_review", "source": "start", "target": "review", "label": ""},
            {"id": "e_review_end", "source": "review", "target": "end", "label": ""},
        ],
    }


def fake_json(value: dict) -> SimpleNamespace:
    return SimpleNamespace(raw_response="", parsed_json=value, parse_error=None, normalized_from_fence=False)


def patch_semantic_specs(monkeypatch, specs: list[dict]) -> None:
    monkeypatch.setenv("SEMANTIC_LEGACY_WORKFLOW_IR_COMPAT", "1")
    remaining = list(specs)

    def complete_json(_system: str, _payload: dict) -> SimpleNamespace:
        assert remaining
        return fake_json(remaining.pop(0))

    monkeypatch.setattr(semantic_shadow, "_complete_json", complete_json)


def rename_spec(label: str) -> dict:
    return {
        "status": "resolved",
        "workflow_ir": {"properties": [{"node_id": "review", "changes": {"label": label}}]},
    }


def partial_spec() -> dict:
    return {
        "status": "partial",
        "workflow_ir": {},
        "semantic_goals": [{"goal_type": "other", "description": "need more detail"}],
        "clarification": {"needed": True, "question": "Which node should I edit?", "reason": "ambiguous"},
    }


def invalid_spec() -> dict:
    return {
        "status": "resolved",
        "workflow_ir": {
            "control_flow_regions": [
                {"relation_type": "", "structure": {"type": "sequence", "items": [{"type": "node", "id": "missing"}]}}
            ]
        },
    }


def test_semantic_versioned_transaction_commits_and_supports_undo_redo(monkeypatch) -> None:
    state = create_version_state(graph(), created_at="2026-09-22T00:00:00")
    patch_semantic_specs(monkeypatch, [rename_spec("Checked"), rename_spec("Approved")])

    first = run_versioned_semantic_transaction(state, 1, {"node_ids": [], "edge_ids": []}, "rename review")
    assert first.execution.status == "success"
    assert first.committed_version is not None
    assert state.current_version_id == "v2"
    assert state.current_graph()["nodes"][1]["label"] == "Checked"

    second = run_versioned_semantic_transaction(state, 2, {"node_ids": [], "edge_ids": []}, "rename again")
    assert second.execution.status == "success"
    assert second.committed_version is not None
    assert state.current_version_id == "v3"
    assert state.current_graph()["nodes"][1]["label"] == "Approved"

    assert undo(state) is not None
    assert state.current_version_id == "v2"
    assert state.current_graph()["nodes"][1]["label"] == "Checked"
    assert redo(state) is not None
    assert state.current_version_id == "v3"
    assert state.current_graph()["nodes"][1]["label"] == "Approved"


def test_semantic_partial_and_validation_failure_do_not_commit(monkeypatch) -> None:
    state = create_version_state(graph(), created_at="2026-09-22T00:00:00")
    patch_semantic_specs(monkeypatch, [partial_spec(), invalid_spec()])

    partial = run_versioned_semantic_transaction(state, 1, {"node_ids": [], "edge_ids": []}, "unclear")
    assert partial.execution.status == "needs_clarification"
    assert partial.committed_version is None
    assert state.current_version_id == "v1"
    assert state.current_graph()["nodes"][1]["label"] == "Review"

    invalid = run_versioned_semantic_transaction(state, 1, {"node_ids": [], "edge_ids": []}, "invalid")
    assert invalid.execution.status != "success"
    assert invalid.committed_version is None
    assert state.current_version_id == "v1"
    assert state.current_graph()["nodes"][1]["label"] == "Review"


def test_demo_ai_edit_semantic_pipeline_commits_without_legacy_fallback(monkeypatch) -> None:
    server.SESSIONS.clear()
    monkeypatch.setenv("AI_EDIT_PIPELINE", "semantic")
    patch_semantic_specs(monkeypatch, [rename_spec("Semantic Review")])
    created = server.create_session({"graph": graph()})

    result = server.run_edit(
        {
            "session_id": created["session_id"],
            "instruction": "rename review",
            "selection": {"node_ids": [], "edge_ids": []},
            "request_context": {},
        }
    )

    assert result["pipeline"] == "semantic"
    assert result["status"] == "success"
    assert result["current_version_id"] == "v2"
    assert result["committed_version"] is not None
    assert result["graph"]["nodes"][1]["label"] == "Semantic Review"


def test_demo_ai_edit_semantic_failure_does_not_fallback_to_legacy(monkeypatch) -> None:
    server.SESSIONS.clear()
    monkeypatch.setenv("AI_EDIT_PIPELINE", "semantic")
    patch_semantic_specs(monkeypatch, [partial_spec()])
    monkeypatch.setattr(
        mvp23,
        "run_edit_interpretation",
        lambda *args, **kwargs: mvp23.JsonRun(
            "",
            {"status": "clear", "target": {"node_ids": ["review"], "edge_ids": []}, "intent": {"kind": "rename"}, "policy": {}},
            None,
            False,
        ),
    )
    monkeypatch.setattr(
        mvp23,
        "finalize_edit_specification",
        lambda *args, **kwargs: mvp23.JsonRun(
            "",
            {"status": "ready", "constraints": {"entities": [], "properties": [{"node_id": "review", "changes": {"label": "Legacy"}}], "required_relations": [], "forbidden_relations": []}, "directives": []},
            None,
            False,
        ),
    )
    created = server.create_session({"graph": graph()})

    result = server.run_edit({"session_id": created["session_id"], "instruction": "unclear", "selection": {"node_ids": [], "edge_ids": []}})

    assert result["pipeline"] == "semantic"
    assert result["status"] == "needs_clarification"
    assert result["committed_version"] is None
    assert result["current_version_id"] == "v1"
    assert result["graph"]["nodes"][1]["label"] == "Review"


def test_demo_ai_edit_legacy_pipeline_still_available(monkeypatch) -> None:
    server.SESSIONS.clear()
    monkeypatch.setenv("AI_EDIT_PIPELINE", "legacy")
    monkeypatch.setattr(
        mvp23,
        "run_edit_interpretation",
        lambda *args, **kwargs: mvp23.JsonRun(
            "",
            {"status": "clear", "target": {"node_ids": ["review"], "edge_ids": []}, "intent": {"kind": "rename"}, "policy": {}},
            None,
            False,
        ),
    )
    monkeypatch.setattr(
        mvp23,
        "finalize_edit_specification",
        lambda *args, **kwargs: mvp23.JsonRun(
            "",
            {"status": "ready", "constraints": {"entities": [], "properties": [{"node_id": "review", "changes": {"label": "Legacy"}}], "required_relations": [], "forbidden_relations": []}, "directives": []},
            None,
            False,
        ),
    )
    created = server.create_session({"graph": graph()})

    result = server.run_edit({"session_id": created["session_id"], "instruction": "rename", "selection": {"node_ids": ["review"], "edge_ids": []}})

    assert result["pipeline"] == "legacy"
    assert result["status"] == "success"
    assert result["current_version_id"] == "v2"
    assert result["graph"]["nodes"][1]["label"] == "Legacy"
