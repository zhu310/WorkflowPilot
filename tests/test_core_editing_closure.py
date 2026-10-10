from __future__ import annotations

from types import SimpleNamespace

from src import semantic_shadow


def mock_understanding(monkeypatch, spec: dict) -> None:
    monkeypatch.setattr(
        semantic_shadow,
        "_complete_json",
        lambda _system, _payload: SimpleNamespace(raw_response="", parsed_json=spec, parse_error=None, normalized_from_fence=False),
    )


def base_graph() -> dict:
    return {
        "nodes": [
            {"id": "start", "type": "start", "label": "Start", "x": 0, "y": 0, "width": 160, "height": 80},
            {"id": "review", "type": "task", "label": "Review", "x": 220, "y": 0, "width": 160, "height": 80},
            {"id": "risk", "type": "task", "label": "Risk", "x": 440, "y": 0, "width": 160, "height": 80},
            {"id": "end", "type": "end", "label": "End", "x": 660, "y": 0, "width": 160, "height": 80},
        ],
        "edges": [
            {"id": "e_start_review", "source": "start", "target": "review", "label": ""},
            {"id": "e_review_risk", "source": "review", "target": "risk", "label": ""},
            {"id": "e_risk_end", "source": "risk", "target": "end", "label": ""},
        ],
    }


def resolved_goal(goal: dict) -> dict:
    return {"status": "resolved", "semantic_goals": [goal], "clarification": {"needed": False, "question": "", "reason": ""}}


def test_exists_state_creates_node_through_existing_backend(monkeypatch) -> None:
    mock_understanding(
        monkeypatch,
        resolved_goal(
            {
                "id": "g1",
                "goal_category": "object_state_goal",
                "text": "Add Manual Review.",
                "objects": [{"id": "manual", "type": "new_entity", "label": "Manual Review"}],
                "object_desired_states": [{"object": "manual", "state": "exists"}],
            }
        ),
    )

    trace = semantic_shadow.run_semantic_shadow({"nodes": [{"id": "start", "type": "start", "label": "Start"}], "edges": []}, "Add Manual Review.")

    assert trace["stage"] == "success"
    labels = {node.get("label") for node in trace["final_shadow_graph"]["nodes"]}
    assert "Manual Review" in labels
    assert trace["compiler_input"]["workflow_ir"]["control_flow_regions"]


def test_node_property_equals_renames_node_label(monkeypatch) -> None:
    mock_understanding(
        monkeypatch,
        resolved_goal(
            {
                "id": "g1",
                "goal_category": "object_state_goal",
                "text": "Rename Review to Manual Review.",
                "objects": [{"id": "target", "type": "existing_reference", "text": "Review"}],
                "object_desired_states": [
                    {"object": "target", "state": "property_equals", "property": "display_label", "value": "Manual Review"}
                ],
            }
        ),
    )

    trace = semantic_shadow.run_semantic_shadow(base_graph(), "Rename Review to Manual Review.")

    assert trace["stage"] == "success"
    review = next(node for node in trace["final_shadow_graph"]["nodes"] if node["id"] == "review")
    assert review["label"] == "Manual Review"
    assert trace["compiler_input"]["workflow_ir"]["properties"] == [{"node_id": "review", "changes": {"label": "Manual Review"}}]


def test_absent_state_deletes_node_without_reconnect(monkeypatch) -> None:
    mock_understanding(
        monkeypatch,
        resolved_goal(
            {
                "id": "g1",
                "goal_category": "object_state_goal",
                "text": "Delete Review.",
                "objects": [{"id": "target", "type": "existing_reference", "text": "Review"}],
                "object_desired_states": [{"object": "target", "state": "absent"}],
            }
        ),
    )

    trace = semantic_shadow.run_semantic_shadow(base_graph(), "Delete Review.")

    assert trace["stage"] == "success"
    assert "review" not in {node["id"] for node in trace["final_shadow_graph"]["nodes"]}
    assert all(edge.get("source") != "review" and edge.get("target") != "review" for edge in trace["final_shadow_graph"]["edges"])
    assert trace["compiler_input"]["workflow_ir"]["absent_entities"] == [{"node_id": "review"}]


def test_route_to_new_target_creates_target_then_reroutes(monkeypatch) -> None:
    graph = {
        "nodes": [
            {"id": "decision", "type": "task", "label": "Decision"},
            {"id": "reject", "type": "task", "label": "Reject"},
        ],
        "edges": [{"id": "e_fail", "source": "decision", "target": "reject", "label": "failure", "branch_label": "Failure"}],
    }
    mock_understanding(
        monkeypatch,
        resolved_goal(
            {
                "id": "g1",
                "goal_category": "composite_goal",
                "text": "Add Manual Review and route Failure to Manual Review.",
                "objects": [
                    {"id": "route", "type": "edge_reference", "text": "Failure"},
                    {"id": "manual", "type": "new_entity", "label": "Manual Review"},
                ],
                "object_desired_states": [{"object": "manual", "state": "exists"}],
                "relation_intents": [{"source": "route", "relation": "route_to", "target": "manual"}],
            }
        ),
    )

    trace = semantic_shadow.run_semantic_shadow(graph, "Add Manual Review and route Failure to Manual Review.")

    assert trace["stage"] == "success"
    manual = next(node for node in trace["final_shadow_graph"]["nodes"] if node.get("label") == "Manual Review")
    assert any(edge.get("source") == "decision" and edge.get("target") == manual["id"] and edge.get("label") == "failure" for edge in trace["final_shadow_graph"]["edges"])
    assert trace["partial_execution"]["final_status"] == "success"


def test_parallel_with_lowers_only_simple_direct_serial_chain(monkeypatch) -> None:
    mock_understanding(
        monkeypatch,
        resolved_goal(
            {
                "id": "g1",
                "goal_category": "relation_goal",
                "text": "Make Review and Risk parallel.",
                "objects": [
                    {"id": "review_obj", "type": "existing_reference", "text": "Review"},
                    {"id": "risk_obj", "type": "existing_reference", "text": "Risk"},
                ],
                "relation_intents": [{"source": "review_obj", "relation": "parallel_with", "target": "risk_obj"}],
            }
        ),
    )

    trace = semantic_shadow.run_semantic_shadow(base_graph(), "Make Review and Risk parallel.")

    assert trace["stage"] == "success"
    region = trace["compiler_input"]["workflow_ir"]["control_flow_regions"][0]
    assert region["structure"]["type"] == "parallel"
    assert any(edge.get("source") == "start" and edge.get("target") == "risk" for edge in trace["final_shadow_graph"]["edges"])
    assert any(edge.get("source") == "review" and edge.get("target") == "end" for edge in trace["final_shadow_graph"]["edges"])


def test_conversation_history_is_last_resort_for_deictic_node_reference(monkeypatch) -> None:
    mock_understanding(
        monkeypatch,
        resolved_goal(
            {
                "id": "g1",
                "goal_category": "relation_goal",
                "text": "Move that node to the right of Risk.",
                "objects": [
                    {"id": "moving", "type": "existing_reference", "text": "that node"},
                    {"id": "anchor", "type": "existing_reference", "text": "Risk"},
                ],
                "relation_intents": [{"source": "moving", "relation": "right_of", "target": "anchor"}],
            }
        ),
    )

    graph = {
        "nodes": [
            {"id": "risk", "type": "task", "label": "Risk", "x": 440, "y": 0, "width": 160, "height": 80},
            {"id": "manual", "type": "task", "label": "Manual Review", "x": 0, "y": 200, "width": 160, "height": 80},
        ],
        "edges": [],
    }
    trace = semantic_shadow.run_semantic_shadow(
        graph,
        "Move that node to the right of Risk.",
        request_context={"conversation_history": [{"role": "assistant", "content": "Created Manual Review."}]},
    )

    assert trace["stage"] == "success"
    provenance = trace["grounded_goal"]["grounding"]["provenance"]
    assert any(row.get("resolved_id") == "manual" and row.get("resolution_type") == "conversation_history" for row in provenance)
    assert trace["compiler_input"]["workflow_ir"]["properties"][0]["node_id"] == "manual"


def test_composite_goal_reports_partial_success_when_supported_subgoal_executes(monkeypatch) -> None:
    mock_understanding(
        monkeypatch,
        resolved_goal(
            {
                "id": "g1",
                "goal_category": "composite_goal",
                "text": "Add Manual Review and connect it to End.",
                "objects": [
                    {"id": "manual", "type": "new_entity", "label": "Manual Review"},
                    {"id": "end_obj", "type": "existing_reference", "text": "End"},
                ],
                "object_desired_states": [{"object": "manual", "state": "exists"}],
                "relation_intents": [{"source": "manual", "relation": "connected_to", "target": "end_obj"}],
            }
        ),
    )

    trace = semantic_shadow.run_semantic_shadow(base_graph(), "Add Manual Review and connect it to End.")
    result = semantic_shadow.semantic_trace_to_transaction_result(trace)

    assert trace["stage"] == "success"
    assert result.status == "partial_success"
    assert trace["partial_execution"]["executed_count"] == 1
    incomplete = [row for row in trace["partial_execution"]["subgoals"] if row["status"] != "executed"]
    assert incomplete[0]["relation"] == "connected_to"
