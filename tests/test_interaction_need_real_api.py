from __future__ import annotations

from types import SimpleNamespace

from demo import server
from demo_api_helpers import assert_no_internal_visible_text
from src import semantic_shadow


def graph() -> dict:
    return {
        "nodes": [
            {"id": "decision", "type": "decision", "label": "Decision"},
            {"id": "report", "type": "task", "label": "Report"},
            {"id": "human", "type": "task", "label": "Human Review"},
            {"id": "end", "type": "end", "label": "End"},
        ],
        "edges": [
            {"id": "e_success", "source": "decision", "target": "report", "label": "success", "branch_label": "success"},
            {"id": "e_failure", "source": "decision", "target": "human", "label": "failure", "branch_label": "failure"},
            {"id": "e_report_end", "source": "report", "target": "end", "label": ""},
        ],
    }


def candidate_graph() -> dict:
    return {
        "nodes": [
            {"id": "approve", "type": "decision", "label": "Approve"},
            {"id": "review", "type": "task", "label": "Review"},
            {"id": "archive", "type": "task", "label": "Archive"},
            {"id": "finish", "type": "end", "label": "Finish"},
        ],
        "edges": [
            {"id": "e_yes", "source": "approve", "target": "review", "label": "yes", "branch_label": "yes"},
            {"id": "e_timeout", "source": "approve", "target": "archive", "label": "timeout", "branch_label": "timeout"},
            {"id": "e_review_finish", "source": "review", "target": "finish", "label": ""},
            {"id": "e_archive_finish", "source": "archive", "target": "finish", "label": ""},
        ],
    }


def mock_understanding(monkeypatch, spec: dict) -> None:
    monkeypatch.setattr(
        semantic_shadow,
        "_complete_json",
        lambda _system, _payload: SimpleNamespace(raw_response="", parsed_json=spec, parse_error=None, normalized_from_fence=False),
    )


def route_spec(objects: list[dict], relations: list[dict]) -> dict:
    return {
        "status": "resolved",
        "semantic_goals": [
            {
                "id": "g1",
                "goal_category": "relation_goal",
                "objects": objects,
                "relation_intents": relations,
            }
        ],
        "clarification": {"needed": False, "question": "", "reason": ""},
    }


def partial_route_spec(objects: list[dict], relations: list[dict], question: str = "Please clarify.") -> dict:
    return {
        "status": "clarification",
        "semantic_goals": [
            {
                "id": "g1",
                "goal_category": "relation_goal",
                "objects": objects,
                "relation_intents": relations,
            }
        ],
        "clarification": {"needed": True, "question": question, "reason": "missing semantic role"},
    }


def generic_transformation_spec(object_type: str, reference: str, question: str = "How should this object change?") -> dict:
    return {
        "status": "clarification",
        "semantic_goals": [
            {
                "id": "g1",
                "goal_category": "",
                "objects": [
                    {"id": "o1", "type": object_type, "reference": reference, "role": "object"},
                ],
                "unresolved_dimensions": [
                    {"dimension": "transformation", "object": "o1"},
                ],
            }
        ],
        "clarification": {"needed": True, "question": question, "reason": "missing transformation"},
    }


def edge_label_spec(reference: str, value: str) -> dict:
    return {
        "status": "resolved",
        "semantic_goals": [
            {
                "id": "g1",
                "goal_category": "object_state_goal",
                "objects": [{"id": "o1", "type": "edge_reference", "reference": reference, "role": "object"}],
                "object_desired_states": [
                    {"object": "o1", "state": "property_equals", "property": "display_label", "value": value}
                ],
            }
        ],
        "clarification": {"needed": False, "question": "", "reason": ""},
    }


def node_rename_spec(reference: str, value: str) -> dict:
    return {
        "status": "resolved",
        "semantic_goals": [
            {
                "id": "g1",
                "goal_category": "object_state_goal",
                "objects": [{"id": "o1", "type": "existing_reference", "reference": reference, "role": "object"}],
                "object_desired_states": [
                    {"object": "o1", "state": "property_equals", "property": "display_label", "value": value}
                ],
            }
        ],
        "clarification": {"needed": False, "question": "", "reason": ""},
    }


def run_api(monkeypatch, spec: dict, selection: dict, instruction: str = "edit") -> dict:
    server.SESSIONS.clear()
    monkeypatch.setenv("AI_EDIT_PIPELINE", "semantic")
    mock_understanding(monkeypatch, spec)
    session_id = server.create_session({"graph": graph()})["session_id"]
    return server.run_edit(
        {
            "session_id": session_id,
            "instruction": instruction,
            "selection": selection,
            "request_context": {},
        }
    )


def assert_conversational(response: dict) -> None:
    message = response["user_feedback"]["message"]
    assert_no_internal_visible_text(message)
    assert "未完成" not in message
    assert "不能安全执行" not in message
    assert "候选 id" not in message


def test_real_api_missing_subject_interaction_preserves_known_target(monkeypatch) -> None:
    response = run_api(
        monkeypatch,
        route_spec(
            [{"id": "target", "type": "existing_reference", "reference": "Human Review"}],
            [{"source": "route", "relation": "route_to", "target": "target"}],
        ),
        {"node_ids": [], "edge_ids": []},
        "route to Human Review",
    )

    interaction = response["user_feedback"]["interaction"]
    assert interaction["type"] == "provide_missing_slot"
    assert interaction["slot"] == "subject_edge"
    assert interaction["known_target"] == "Human Review"
    assert "Human Review" in response["message"]
    assert_conversational(response)


def test_real_api_missing_target_interaction_preserves_selected_edge(monkeypatch) -> None:
    response = run_api(
        monkeypatch,
        route_spec(
            [{"id": "route", "type": "edge_reference", "reference": "this edge"}],
            [{"source": "route", "relation": "route_to", "target": "target"}],
        ),
        {"node_ids": [], "edge_ids": ["e_success"]},
        "change this edge",
    )

    interaction = response["user_feedback"]["interaction"]
    assert interaction["type"] == "provide_missing_slot"
    assert interaction["slot"] == "route_target"
    assert "Decision -> Report" in interaction["known_subject"]
    assert_conversational(response)


def test_real_api_ambiguous_interaction_has_real_candidates_only(monkeypatch) -> None:
    response = run_api(
        monkeypatch,
        route_spec(
            [
                {"id": "route", "type": "edge_reference", "reference": "this connection"},
                {"id": "target", "type": "existing_reference", "reference": "End"},
            ],
            [{"source": "route", "relation": "route_to", "target": "target"}],
        ),
        {"node_ids": [], "edge_ids": ["e_success", "e_report_end"]},
        "route this connection to End",
    )

    interaction = response["user_feedback"]["interaction"]
    assert response["status"] == "needs_clarification"
    assert interaction["type"] == "choose_candidate"
    assert len(interaction["candidates"]) == 2
    assert "e_success" not in response["message"]
    assert_conversational(response)


def test_real_api_confirmation_interaction_has_no_generic_failure_copy(monkeypatch) -> None:
    response = run_api(
        monkeypatch,
        route_spec(
            [
                {"id": "route", "type": "edge_reference", "reference": "this edge"},
                {"id": "target", "type": "existing_reference", "reference": "Human Review"},
            ],
            [{"source": "route", "relation": "route_to", "target": "target"}],
        ),
        {"node_ids": [], "edge_ids": ["e_success"]},
        "route this edge to Human Review",
    )

    interaction = response["user_feedback"]["interaction"]
    assert response["status"] == "needs_clarification"
    assert interaction["type"] == "confirm_change"
    assert "Decision" in response["message"]
    assert "Human Review" in response["message"]
    assert_conversational(response)


def test_missing_subject_pending_resumes_from_current_edge_selection(monkeypatch) -> None:
    server.SESSIONS.clear()
    monkeypatch.setenv("AI_EDIT_PIPELINE", "semantic")

    missing_subject = route_spec(
        [{"id": "target", "type": "existing_reference", "reference": "Human Review"}],
        [{"source": "route", "relation": "route_to", "target": "target"}],
    )
    resumed_route = route_spec(
        [
            {"id": "route", "type": "edge_reference", "reference": "selected edge"},
            {"id": "target", "type": "existing_reference", "reference": "Human Review"},
        ],
        [{"source": "route", "relation": "route_to", "target": "target"}],
    )

    def complete(_system, payload):
        spec = resumed_route if payload["instruction"] == "Route selected edge to Human Review" else missing_subject
        return SimpleNamespace(raw_response="", parsed_json=spec, parse_error=None, normalized_from_fence=False)

    monkeypatch.setattr(semantic_shadow, "_complete_json", complete)
    session_id = server.create_session({"graph": graph()})["session_id"]

    first = server.run_edit(
        {
            "session_id": session_id,
            "instruction": "route to Human Review",
            "selection": {"node_ids": [], "edge_ids": []},
            "request_context": {},
        }
    )
    assert first["status"] == "needs_clarification"
    pending = first["state"]["pending_clarification"]
    assert pending["type"] == "provide_missing_slot"
    assert pending["originating_version_id"] == first["current_version_id"]

    second = server.run_edit(
        {
            "session_id": session_id,
            "instruction": "this one",
            "selection": {"node_ids": [], "edge_ids": ["e_report_end"]},
            "request_context": {},
        }
    )
    assert second["status"] == "success"
    assert second["transaction"]["selection"]["edge_ids"] == ["e_report_end"]
    assert second["state"]["pending_clarification"] is None
    changed = {edge["id"]: edge for edge in second["graph"]["edges"]}
    assert changed["e_report_end"]["target"] == "human"


def test_missing_target_pending_resumes_from_target_reply(monkeypatch) -> None:
    server.SESSIONS.clear()
    monkeypatch.setenv("AI_EDIT_PIPELINE", "semantic")

    missing_target = route_spec(
        [{"id": "route", "type": "edge_reference", "reference": "selected edge"}],
        [{"source": "route", "relation": "route_to", "target": "target"}],
    )
    resumed_route = route_spec(
        [
            {"id": "route", "type": "edge_reference", "reference": "selected edge"},
            {"id": "target", "type": "existing_reference", "reference": "End"},
        ],
        [{"source": "route", "relation": "route_to", "target": "target"}],
    )

    def complete(_system, payload):
        spec = resumed_route if payload["instruction"] == "Route selected edge to End" else missing_target
        return SimpleNamespace(raw_response="", parsed_json=spec, parse_error=None, normalized_from_fence=False)

    monkeypatch.setattr(semantic_shadow, "_complete_json", complete)
    session_id = server.create_session({"graph": graph()})["session_id"]

    first = server.run_edit(
        {
            "session_id": session_id,
            "instruction": "change this connection",
            "selection": {"node_ids": [], "edge_ids": ["e_success"]},
            "request_context": {},
        }
    )
    assert first["status"] == "needs_clarification"
    assert first["state"]["pending_clarification"]["type"] == "provide_missing_slot"

    second = server.run_edit(
        {
            "session_id": session_id,
            "instruction": "End",
            "selection": {"node_ids": [], "edge_ids": []},
            "request_context": {},
        }
    )
    assert second["status"] == "success"
    assert second["transaction"]["selection"]["edge_ids"] == ["e_success"]
    changed = {edge["id"]: edge for edge in second["graph"]["edges"]}
    assert changed["e_success"]["target"] == "end"


def test_pending_invalidates_on_version_mismatch(monkeypatch) -> None:
    server.SESSIONS.clear()
    monkeypatch.setenv("AI_EDIT_PIPELINE", "semantic")
    mock_understanding(
        monkeypatch,
        route_spec(
            [{"id": "target", "type": "existing_reference", "reference": "Human Review"}],
            [{"source": "route", "relation": "route_to", "target": "target"}],
        ),
    )
    session_id = server.create_session({"graph": graph()})["session_id"]
    first = server.run_edit(
        {
            "session_id": session_id,
            "instruction": "route to Human Review",
            "selection": {"node_ids": [], "edge_ids": []},
            "request_context": {},
        }
    )
    assert first["state"]["pending_clarification"]["originating_version_id"] == "v1"
    server.SESSIONS[session_id].pending_clarification["originating_version_id"] = "stale"

    second = server.run_edit(
        {
            "session_id": session_id,
            "instruction": "this one",
            "selection": {"node_ids": [], "edge_ids": ["e_success"]},
            "request_context": {},
        }
    )
    assert second["status"] == "needs_clarification"
    assert second["state"]["pending_clarification"] is None


def test_runtime_pending_does_not_depend_on_user_feedback(monkeypatch) -> None:
    server.SESSIONS.clear()
    monkeypatch.setenv("AI_EDIT_PIPELINE", "semantic")
    mock_understanding(
        monkeypatch,
        route_spec(
            [{"id": "target", "type": "existing_reference", "reference": "Human Review"}],
            [{"source": "route", "relation": "route_to", "target": "target"}],
        ),
    )
    monkeypatch.setattr(
        server,
        "build_user_feedback",
        lambda status, specification=None, graph=None, clarification_message=None, error=None: {
            "status": status,
            "message": "patched feedback",
            "clarification": {},
            "interaction": {},
        },
    )
    session_id = server.create_session({"graph": graph()})["session_id"]

    response = server.run_edit(
        {
            "session_id": session_id,
            "instruction": "route to Human Review",
            "selection": {"node_ids": [], "edge_ids": []},
            "request_context": {},
        }
    )
    assert response["status"] == "needs_clarification"
    assert response["state"]["pending_clarification"]["type"] == "provide_missing_slot"


def test_partial_semantic_goal_missing_subject_resumes_to_confirmation(monkeypatch) -> None:
    server.SESSIONS.clear()
    monkeypatch.setenv("AI_EDIT_PIPELINE", "semantic")

    turn1_partial = partial_route_spec(
        [
            {"id": "source", "type": "unresolved_reference", "role": "source"},
            {"id": "target", "type": "existing_reference", "reference": "Human Review", "role": "target"},
        ],
        [{"source": "source", "relation": "route_to", "target": "target"}],
        question="Which route should go to Human Review?",
    )
    resumed_route = route_spec(
        [
            {"id": "route", "type": "edge_reference", "reference": "selected edge"},
            {"id": "target", "type": "existing_reference", "reference": "Human Review"},
        ],
        [{"source": "route", "relation": "route_to", "target": "target"}],
    )

    def complete(_system, payload):
        spec = resumed_route if payload["instruction"] == "Route selected edge to Human Review" else turn1_partial
        return SimpleNamespace(raw_response="", parsed_json=spec, parse_error=None, normalized_from_fence=False)

    monkeypatch.setattr(semantic_shadow, "_complete_json", complete)
    session_id = server.create_session({"graph": graph()})["session_id"]

    first = server.run_edit(
        {
            "session_id": session_id,
            "instruction": "route to Human Review",
            "selection": {"node_ids": [], "edge_ids": []},
            "request_context": {},
        }
    )
    interaction = first["user_feedback"]["interaction"]
    assert first["status"] == "needs_clarification"
    assert first["specification"]["goal_validation"]["ready_for_grounding"] is False
    assert first["specification"]["semantic_goal"]["normalized"]["semantic_goals_v2"][0]["objects"][0]["type"] == "unresolved_reference"
    assert interaction["type"] == "provide_missing_slot"
    assert interaction["slot"] == "subject_edge"
    assert interaction["known_target"] == "Human Review"
    assert first["state"]["pending_clarification"]["interaction"]["slot"] == "subject_edge"

    second = server.run_edit(
        {
            "session_id": session_id,
            "instruction": "this one",
            "selection": {"node_ids": [], "edge_ids": ["e_success"]},
            "request_context": {},
        }
    )
    assert second["status"] == "needs_clarification"
    assert second["user_feedback"]["interaction"]["type"] == "confirm_change"
    assert second["transaction"]["selection"]["edge_ids"] == ["e_success"]


def test_partial_semantic_goal_missing_target_resumes_from_reply(monkeypatch) -> None:
    server.SESSIONS.clear()
    monkeypatch.setenv("AI_EDIT_PIPELINE", "semantic")

    turn1_partial = partial_route_spec(
        [
            {"id": "source", "type": "edge_reference", "reference": "this edge", "role": "source"},
            {"id": "target", "type": "unresolved_reference", "role": "target"},
        ],
        [{"source": "source", "relation": "route_to", "target": "target"}],
        question="Where should this edge go?",
    )
    resumed_route = route_spec(
        [
            {"id": "route", "type": "edge_reference", "reference": "selected edge"},
            {"id": "target", "type": "existing_reference", "reference": "End"},
        ],
        [{"source": "route", "relation": "route_to", "target": "target"}],
    )

    def complete(_system, payload):
        spec = resumed_route if payload["instruction"] == "Route selected edge to End" else turn1_partial
        return SimpleNamespace(raw_response="", parsed_json=spec, parse_error=None, normalized_from_fence=False)

    monkeypatch.setattr(semantic_shadow, "_complete_json", complete)
    session_id = server.create_session({"graph": graph()})["session_id"]

    first = server.run_edit(
        {
            "session_id": session_id,
            "instruction": "change this route",
            "selection": {"node_ids": [], "edge_ids": ["e_success"]},
            "request_context": {},
        }
    )
    interaction = first["user_feedback"]["interaction"]
    assert first["status"] == "needs_clarification"
    assert interaction["type"] == "provide_missing_slot"
    assert interaction["slot"] == "route_target"
    assert "Decision -> Report" in interaction["known_subject"]

    second = server.run_edit(
        {
            "session_id": session_id,
            "instruction": "End",
            "selection": {"node_ids": [], "edge_ids": []},
            "request_context": {},
        }
    )
    assert second["status"] == "success"
    assert second["transaction"]["selection"]["edge_ids"] == ["e_success"]
    changed = {edge["id"]: edge for edge in second["graph"]["edges"]}
    assert changed["e_success"]["target"] == "end"


def test_generic_edge_edit_pending_refines_to_route_without_text_concatenation(monkeypatch) -> None:
    server.SESSIONS.clear()
    monkeypatch.setenv("AI_EDIT_PIPELINE", "semantic")
    turn1_partial = generic_transformation_spec("edge_reference", "this edge", "How should this edge change?")
    turn2_route = route_spec(
        [
            {"id": "route", "type": "edge_reference", "reference": "this edge"},
            {"id": "target", "type": "existing_reference", "reference": "End"},
        ],
        [{"source": "route", "relation": "route_to", "target": "target"}],
    )
    seen_payloads = []

    def complete(_system, payload):
        seen_payloads.append(payload)
        if payload["instruction"] == "send it to End":
            return SimpleNamespace(raw_response="", parsed_json=turn2_route, parse_error=None, normalized_from_fence=False)
        return SimpleNamespace(raw_response="", parsed_json=turn1_partial, parse_error=None, normalized_from_fence=False)

    monkeypatch.setattr(semantic_shadow, "_complete_json", complete)
    session_id = server.create_session({"graph": graph()})["session_id"]

    first = server.run_edit(
        {
            "session_id": session_id,
            "instruction": "change this edge",
            "selection": {"node_ids": [], "edge_ids": ["e_success"]},
            "request_context": {},
        }
    )
    assert first["status"] == "needs_clarification"
    interaction = first["user_feedback"]["interaction"]
    assert interaction["type"] == "provide_missing_slot"
    assert interaction["slot"] == "transformation"
    pending = first["state"]["pending_clarification"]
    assert pending["partial_semantic_state"]["semantic_goal"]["semantic_goals_v2"][0]["goal_category"] == ""

    second = server.run_edit(
        {
            "session_id": session_id,
            "instruction": "send it to End",
            "selection": {"node_ids": [], "edge_ids": []},
            "request_context": {},
        }
    )

    assert second["status"] == "success"
    assert second["transaction"]["selection"]["edge_ids"] == ["e_success"]
    assert seen_payloads[-1]["instruction"] == "send it to End"
    assert seen_payloads[-1]["request_context"]["semantic_refinement"]["mode"] == "refine_partial_goal"
    changed = {edge["id"]: edge for edge in second["graph"]["edges"]}
    assert changed["e_success"]["target"] == "end"


def test_generic_edge_edit_pending_refines_to_label_edit(monkeypatch) -> None:
    server.SESSIONS.clear()
    monkeypatch.setenv("AI_EDIT_PIPELINE", "semantic")
    turn1_partial = generic_transformation_spec("edge_reference", "this edge")
    turn2_label = edge_label_spec("this edge", "retry")

    def complete(_system, payload):
        spec = turn2_label if payload["instruction"] == "set the label to retry" else turn1_partial
        return SimpleNamespace(raw_response="", parsed_json=spec, parse_error=None, normalized_from_fence=False)

    monkeypatch.setattr(semantic_shadow, "_complete_json", complete)
    session_id = server.create_session({"graph": graph()})["session_id"]

    first = server.run_edit(
        {
            "session_id": session_id,
            "instruction": "change this edge",
            "selection": {"node_ids": [], "edge_ids": ["e_success"]},
            "request_context": {},
        }
    )
    assert first["status"] == "needs_clarification"

    second = server.run_edit(
        {
            "session_id": session_id,
            "instruction": "set the label to retry",
            "selection": {"node_ids": [], "edge_ids": []},
            "request_context": {},
        }
    )

    assert second["status"] == "success"
    changed = {edge["id"]: edge for edge in second["graph"]["edges"]}
    assert changed["e_success"]["branch_label"] == "retry"


def test_generic_node_edit_pending_refines_to_rename(monkeypatch) -> None:
    server.SESSIONS.clear()
    monkeypatch.setenv("AI_EDIT_PIPELINE", "semantic")
    turn1_partial = generic_transformation_spec("existing_reference", "this node")
    turn2_rename = node_rename_spec("this node", "Archive Check")

    def complete(_system, payload):
        spec = turn2_rename if payload["instruction"] == "rename it to Archive Check" else turn1_partial
        return SimpleNamespace(raw_response="", parsed_json=spec, parse_error=None, normalized_from_fence=False)

    monkeypatch.setattr(semantic_shadow, "_complete_json", complete)
    session_id = server.create_session({"graph": graph()})["session_id"]

    first = server.run_edit(
        {
            "session_id": session_id,
            "instruction": "change this node",
            "selection": {"node_ids": ["report"], "edge_ids": []},
            "request_context": {},
        }
    )
    assert first["status"] == "needs_clarification"
    assert first["user_feedback"]["interaction"]["slot"] == "transformation"

    second = server.run_edit(
        {
            "session_id": session_id,
            "instruction": "rename it to Archive Check",
            "selection": {"node_ids": [], "edge_ids": []},
            "request_context": {},
        }
    )

    assert second["status"] == "success"
    changed = {node["id"]: node for node in second["graph"]["nodes"]}
    assert changed["report"]["label"] == "Archive Check"


def test_choose_candidate_pending_resumes_grounding_with_chinese_ordinal(monkeypatch) -> None:
    server.SESSIONS.clear()
    monkeypatch.setenv("AI_EDIT_PIPELINE", "semantic")
    spec = route_spec(
        [
            {"id": "route", "type": "edge_reference", "reference": "this edge"},
            {"id": "target", "type": "existing_reference", "reference": "Human Review"},
        ],
        [{"source": "route", "relation": "route_to", "target": "target"}],
    )
    calls = []

    def complete(_system, payload):
        calls.append(payload)
        return SimpleNamespace(raw_response="", parsed_json=spec, parse_error=None, normalized_from_fence=False)

    monkeypatch.setattr(semantic_shadow, "_complete_json", complete)
    session_id = server.create_session({"graph": graph()})["session_id"]

    first = server.run_edit(
        {
            "session_id": session_id,
            "instruction": "route this edge to Human Review",
            "selection": {"node_ids": [], "edge_ids": ["e_success", "e_report_end"]},
            "request_context": {},
        }
    )
    assert first["status"] == "needs_clarification"
    pending = first["state"]["pending_clarification"]
    assert pending["type"] == "choose_candidate"
    assert pending["resume_stage"] == "grounding"
    assert [row["id"] for row in pending["candidates"][0]["candidates"]] == ["e_success", "e_report_end"]

    second = server.run_edit(
        {
            "session_id": session_id,
            "instruction": "\u7b2c\u4e8c\u4e2a",
            "selection": {"node_ids": [], "edge_ids": []},
            "request_context": {},
        }
    )

    assert second["status"] == "success"
    assert len(calls) == 1
    assert second["transaction"]["selection"]["edge_ids"] == ["e_success", "e_report_end"]
    assert second["state"]["pending_clarification"] is None
    changed = {edge["id"]: edge for edge in second["graph"]["edges"]}
    assert changed["e_report_end"]["target"] == "human"
    assert changed["e_success"]["target"] == "report"


def test_choose_candidate_pending_resolves_from_current_selection(monkeypatch) -> None:
    server.SESSIONS.clear()
    monkeypatch.setenv("AI_EDIT_PIPELINE", "semantic")
    spec = route_spec(
        [
            {"id": "route", "type": "edge_reference", "reference": "this edge"},
            {"id": "target", "type": "existing_reference", "reference": "End"},
        ],
        [{"source": "route", "relation": "route_to", "target": "target"}],
    )
    calls = []

    def complete(_system, payload):
        calls.append(payload)
        return SimpleNamespace(raw_response="", parsed_json=spec, parse_error=None, normalized_from_fence=False)

    monkeypatch.setattr(semantic_shadow, "_complete_json", complete)
    session_id = server.create_session({"graph": graph()})["session_id"]
    first = server.run_edit(
        {
            "session_id": session_id,
            "instruction": "route this edge to End",
            "selection": {"node_ids": [], "edge_ids": ["e_success", "e_report_end"]},
            "request_context": {},
        }
    )
    assert first["status"] == "needs_clarification"

    second = server.run_edit(
        {
            "session_id": session_id,
            "instruction": "this one",
            "selection": {"node_ids": [], "edge_ids": ["e_success"]},
            "request_context": {},
        }
    )

    assert second["status"] == "success"
    assert len(calls) == 1
    changed = {edge["id"]: edge for edge in second["graph"]["edges"]}
    assert changed["e_success"]["target"] == "end"


def test_choose_candidate_unresolved_reply_keeps_pending(monkeypatch) -> None:
    server.SESSIONS.clear()
    monkeypatch.setenv("AI_EDIT_PIPELINE", "semantic")
    spec = route_spec(
        [
            {"id": "route", "type": "edge_reference", "reference": "this edge"},
            {"id": "target", "type": "existing_reference", "reference": "End"},
        ],
        [{"source": "route", "relation": "route_to", "target": "target"}],
    )
    calls = []

    def complete(_system, payload):
        calls.append(payload)
        return SimpleNamespace(raw_response="", parsed_json=spec, parse_error=None, normalized_from_fence=False)

    monkeypatch.setattr(semantic_shadow, "_complete_json", complete)
    session_id = server.create_session({"graph": graph()})["session_id"]
    first = server.run_edit(
        {
            "session_id": session_id,
            "instruction": "route this edge to End",
            "selection": {"node_ids": [], "edge_ids": ["e_success", "e_report_end"]},
            "request_context": {},
        }
    )
    pending_id = first["pending_id"]

    second = server.run_edit(
        {
            "session_id": session_id,
            "instruction": "not sure",
            "selection": {"node_ids": [], "edge_ids": []},
            "request_context": {},
        }
    )

    assert second["status"] == "needs_clarification"
    assert second["pending_id"] == pending_id
    assert second["state"]["pending_clarification"]["type"] == "choose_candidate"
    assert len(second["user_feedback"]["clarification"]["candidates"][0]["candidates"]) == 2
    assert len(calls) == 1


def test_non_demo_candidate_pending_resolves_by_candidate_descriptor(monkeypatch) -> None:
    server.SESSIONS.clear()
    monkeypatch.setenv("AI_EDIT_PIPELINE", "semantic")
    spec = route_spec(
        [
            {"id": "route", "type": "edge_reference", "reference": "this edge"},
            {"id": "target", "type": "existing_reference", "reference": "Finish"},
        ],
        [{"source": "route", "relation": "route_to", "target": "target"}],
    )
    calls = []

    def complete(_system, payload):
        calls.append(payload)
        return SimpleNamespace(raw_response="", parsed_json=spec, parse_error=None, normalized_from_fence=False)

    monkeypatch.setattr(semantic_shadow, "_complete_json", complete)
    session_id = server.create_session({"graph": candidate_graph()})["session_id"]
    first = server.run_edit(
        {
            "session_id": session_id,
            "instruction": "route this edge to Finish",
            "selection": {"node_ids": [], "edge_ids": ["e_yes", "e_timeout"]},
            "request_context": {},
        }
    )
    assert first["status"] == "needs_clarification"

    second = server.run_edit(
        {
            "session_id": session_id,
            "instruction": "timeout",
            "selection": {"node_ids": [], "edge_ids": []},
            "request_context": {},
        }
    )

    assert second["status"] == "success"
    assert len(calls) == 1
    changed = {edge["id"]: edge for edge in second["graph"]["edges"]}
    assert changed["e_timeout"]["target"] == "finish"
    assert changed["e_yes"]["target"] == "review"
