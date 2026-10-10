from __future__ import annotations

from types import SimpleNamespace

from src import semantic_shadow
from src.goal_lowering import DEFAULT_LAYOUT_SPACING, lower_grounded_goal_to_workflow_ir
from src.semantic_goal import ground_semantic_goal_spec


def graph() -> dict:
    return {
        "nodes": [
            {"id": "parse", "type": "task", "label": "Parse Task", "x": 100, "y": 80, "width": 160, "height": 80},
            {"id": "knowledge", "type": "task", "label": "Knowledge Retrieval", "x": 20, "y": 20, "width": 160, "height": 80},
        ],
        "edges": [{"id": "e_parse_knowledge", "source": "parse", "target": "knowledge", "label": ""}],
    }


def test_right_of_lowers_to_workflow_ir_properties_without_topology() -> None:
    goal = {
        "status": "resolved",
        "semantic_goals": [
            {
                "kind": "layout",
                "text": "Put Knowledge Retrieval to the right of Parse Task.",
                "objects": [
                    {"reference": "Knowledge Retrieval", "role": "target"},
                    {"reference": "Parse Task", "role": "anchor"},
                ],
                "relation": "right_of",
                "explicit_constraints": ["preserve_topology"],
            }
        ],
    }
    grounded = ground_semantic_goal_spec(goal, graph(), {"node_ids": [], "edge_ids": []})

    lowered = lower_grounded_goal_to_workflow_ir(grounded, graph())

    assert lowered["ok"] is True
    semantic_spec = lowered["semantic_spec"]
    assert semantic_spec["status"] == "resolved"
    assert semantic_spec["semantic_goals"] == []
    assert semantic_spec["workflow_ir"]["control_flow_regions"] == []
    assert semantic_spec["workflow_ir"]["auxiliary_relations"] == []
    assert semantic_spec["workflow_ir"]["edge_properties"] == []
    assert semantic_spec["workflow_ir"]["absent_entities"] == []
    assert semantic_spec["workflow_ir"]["properties"] == [
        {
            "node_id": "knowledge",
            "changes": {"x": 100 + 160 + DEFAULT_LAYOUT_SPACING, "y": 80.0},
        }
    ]


def test_semantic_shadow_layout_goal_runs_existing_backend(monkeypatch) -> None:
    goal = {
        "status": "resolved",
        "semantic_goals": [
            {
                "kind": "layout",
                "text": "Put Knowledge Retrieval to the right of Parse Task.",
                "objects": [
                    {"reference": "Knowledge Retrieval", "role": "target"},
                    {"reference": "Parse Task", "role": "anchor"},
                ],
                "relation": "right_of",
                "explicit_constraints": ["preserve_topology"],
            }
        ],
    }

    monkeypatch.setattr(
        semantic_shadow,
        "_complete_json",
        lambda _system, _payload: SimpleNamespace(raw_response="", parsed_json=goal, parse_error=None, normalized_from_fence=False),
    )

    trace = semantic_shadow.run_semantic_shadow(graph(), "Put Knowledge Retrieval to the right of Parse Task.")

    assert trace["stage"] == "success"
    assert trace["compiler_input"]["workflow_ir"]["properties"] == [
        {"node_id": "knowledge", "changes": {"x": 340.0, "y": 80.0}}
    ]
    assert trace["final_shadow_graph"]["nodes"][1]["x"] == 340.0
    assert trace["final_shadow_graph"]["nodes"][1]["y"] == 80.0
    assert trace["graph_diff"]["changed_nodes"][0]["id"] == "knowledge"


def test_semantic_shadow_rejects_workflow_ir_without_legacy_compat(monkeypatch) -> None:
    monkeypatch.delenv("SEMANTIC_LEGACY_WORKFLOW_IR_COMPAT", raising=False)
    legacy_spec = {
        "status": "resolved",
        "workflow_ir": {"properties": [{"node_id": "knowledge", "changes": {"x": 999}}]},
    }
    monkeypatch.setattr(
        semantic_shadow,
        "_complete_json",
        lambda _system, _payload: SimpleNamespace(raw_response="", parsed_json=legacy_spec, parse_error=None, normalized_from_fence=False),
    )

    trace = semantic_shadow.run_semantic_shadow(graph(), "Move Knowledge Retrieval.")

    assert trace["failure_stage"] == "validation"
    assert trace["validation"]["issues"][0]["kind"] == "semantic_goal_contract_violation"
    assert trace["final_shadow_graph"] is None
