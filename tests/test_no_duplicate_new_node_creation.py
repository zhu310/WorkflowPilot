from types import SimpleNamespace

from src import semantic_shadow


def graph() -> dict:
    return {
        "nodes": [
            {"id": "start", "type": "start", "label": "Start", "x": 0, "y": 0},
            {"id": "end", "type": "end", "label": "End", "x": 300, "y": 0},
        ],
        "edges": [{"id": "e_start_end", "source": "start", "target": "end", "label": ""}],
    }


def add_exception_before_end_spec() -> dict:
    return {
        "status": "resolved",
        "semantic_goals": [
            {
                "id": "g1",
                "goal_category": "composite_goal",
                "text": "Add exception handling before End",
                "objects": [
                    {"id": "created", "type": "new_entity", "label": "Exception Handling Node"},
                    {"id": "end", "type": "existing_reference", "text": "End"},
                ],
                "object_desired_states": [{"object": "created", "state": "exists"}],
                "relation_intents": [{"source": "created", "relation": "before", "target": "end"}],
            }
        ],
        "clarification": {"needed": False, "question": "", "reason": ""},
    }


def test_same_new_entity_in_state_and_relation_is_created_once(monkeypatch) -> None:
    monkeypatch.setattr(
        semantic_shadow,
        "_complete_json",
        lambda _system, _payload: SimpleNamespace(
            raw_response="",
            parsed_json=add_exception_before_end_spec(),
            parse_error=None,
            normalized_from_fence=False,
        ),
    )

    trace = semantic_shadow.run_semantic_shadow(graph(), "Add Exception Handling Node and put Exception Handling before End")

    assert trace["stage"] == "success"
    created = [
        node
        for node in trace["final_shadow_graph"]["nodes"]
        if str(node.get("label", "")).lower().startswith("exception handling")
    ]
    assert len(created) == 1
