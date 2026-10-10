from types import SimpleNamespace

from src import semantic_shadow
from src.versioning import create_version_state, recent_successful_transaction_diff, run_versioned_semantic_transaction


def mock_understanding_sequence(monkeypatch, specs: list[dict]) -> None:
    responses = list(specs)

    def fake_complete(_system, _payload):
        spec = responses.pop(0) if responses else specs[-1]
        return SimpleNamespace(raw_response="", parsed_json=spec, parse_error=None, normalized_from_fence=False)

    monkeypatch.setattr(semantic_shadow, "_complete_json", fake_complete)


def base_graph() -> dict:
    return {
        "nodes": [
            {"id": "start", "type": "start", "label": "Start", "x": 0, "y": 0, "width": 120, "height": 72},
            {"id": "risk", "type": "task", "label": "Risk", "x": 240, "y": 0, "width": 120, "height": 72},
            {"id": "end", "type": "end", "label": "End", "x": 480, "y": 0, "width": 120, "height": 72},
        ],
        "edges": [
            {"id": "e_start_risk", "source": "start", "target": "risk", "label": ""},
            {"id": "e_risk_end", "source": "risk", "target": "end", "label": ""},
        ],
    }


def exists_plus_unsupported(label: str = "Notice") -> dict:
    return {
        "status": "resolved",
        "semantic_goals": [
            {
                "id": "g1",
                "goal_category": "composite_goal",
                "text": f"Add {label} and connect it to End",
                "objects": [
                    {"id": "notice", "type": "new_entity", "label": label},
                    {"id": "end_obj", "type": "existing_reference", "text": "End"},
                ],
                "object_desired_states": [{"object": "notice", "state": "exists"}],
                "relation_intents": [{"source": "notice", "relation": "connected_to", "target": "end_obj"}],
            }
        ],
        "clarification": {"needed": False, "question": "", "reason": ""},
    }


def move_recent_goal() -> dict:
    return {
        "status": "resolved",
        "semantic_goals": [
            {
                "id": "g1",
                "goal_category": "relation_goal",
                "text": "Move that node to the right of Risk",
                "objects": [
                    {"id": "moving", "type": "existing_reference", "text": "that node"},
                    {"id": "anchor", "type": "existing_reference", "text": "Risk"},
                ],
                "relation_intents": [{"source": "moving", "relation": "right_of", "target": "anchor"}],
            }
        ],
        "clarification": {"needed": False, "question": "", "reason": ""},
    }


def test_partial_success_recent_transaction_uses_only_real_created_node(monkeypatch) -> None:
    state = create_version_state(base_graph())
    mock_understanding_sequence(monkeypatch, [exists_plus_unsupported("Notice"), move_recent_goal()])

    first = run_versioned_semantic_transaction(
        state,
        revision=1,
        selection={"node_ids": [], "edge_ids": []},
        instruction="Add Notice and connect it to End",
    )
    assert first.execution.status == "partial_success"
    diff = recent_successful_transaction_diff(state)
    assert diff is not None
    assert [node["label"] for node in diff["created_nodes"]] == ["Notice"]
    assert all("connected_to" not in str(value) for value in diff.values())

    second = run_versioned_semantic_transaction(
        state,
        revision=2,
        selection={"node_ids": [], "edge_ids": []},
        instruction="Move that node to the right of Risk",
    )
    assert second.execution.status == "success"
    moved = next(node for node in second.execution.final_graph["nodes"] if node.get("label") == "Notice")
    risk = next(node for node in second.execution.final_graph["nodes"] if node.get("id") == "risk")
    assert moved["x"] > risk["x"]


def test_partial_success_without_graph_diff_does_not_create_recent_candidate(monkeypatch) -> None:
    state = create_version_state(base_graph())
    unsupported_only = {
        "status": "resolved",
        "semantic_goals": [
            {
                "id": "g1",
                "goal_category": "relation_goal",
                "text": "Connect that node to End",
                "objects": [{"id": "end_obj", "type": "existing_reference", "text": "End"}],
                "relation_intents": [{"source": "missing", "relation": "connected_to", "target": "end_obj"}],
            }
        ],
        "clarification": {"needed": False, "question": "", "reason": ""},
    }
    mock_understanding_sequence(monkeypatch, [unsupported_only, move_recent_goal()])

    first = run_versioned_semantic_transaction(
        state,
        revision=1,
        selection={"node_ids": [], "edge_ids": []},
        instruction="Connect that node to End",
    )
    assert first.execution.status in {"unsupported", "needs_clarification"}
    assert recent_successful_transaction_diff(state) is None

    second = run_versioned_semantic_transaction(
        state,
        revision=1,
        selection={"node_ids": [], "edge_ids": []},
        instruction="Move that node to the right of Risk",
    )
    assert second.execution.status == "needs_clarification"


def test_recent_transaction_multiple_changed_nodes_stays_ambiguous(monkeypatch) -> None:
    state = create_version_state(base_graph())
    multi_change = {
        "status": "resolved",
        "semantic_goals": [
            {
                "id": "g1",
                "goal_category": "composite_goal",
                "text": "Rename multiple nodes",
                "objects": [
                    {"id": "start_obj", "type": "existing_reference", "text": "Start"},
                    {"id": "risk_obj", "type": "existing_reference", "text": "Risk"},
                ],
                "object_desired_states": [
                    {"object": "start_obj", "state": "property_equals", "property": "display_label", "value": "Begin"},
                    {"object": "risk_obj", "state": "property_equals", "property": "display_label", "value": "Risk Control"},
                ],
            }
        ],
        "clarification": {"needed": False, "question": "", "reason": ""},
    }
    mock_understanding_sequence(monkeypatch, [multi_change, move_recent_goal()])

    first = run_versioned_semantic_transaction(
        state,
        revision=1,
        selection={"node_ids": [], "edge_ids": []},
        instruction="Rename Start to Begin and Risk to Risk Control",
    )
    assert first.execution.status == "success"
    diff = recent_successful_transaction_diff(state)
    assert diff is not None
    assert set(diff["affected_nodes"]) >= {"start", "risk"}

    second = run_versioned_semantic_transaction(
        state,
        revision=2,
        selection={"node_ids": [], "edge_ids": []},
        instruction="Move that node to the right of Risk",
    )
    assert second.execution.status == "needs_clarification"
    assert "ambiguous_node_reference" in (second.execution.error or "")
