from __future__ import annotations

from types import SimpleNamespace

from src import semantic_shadow
from src.constraint_planner import plan_constraints
from src.goal_lowering import lower_grounded_goal_to_workflow_ir
from src.semantic_goal import ground_semantic_goal_spec, validate_grounded_goal, validate_semantic_goal_spec
from src.workflow_ir_core import compile_ir, constraints_from_compiled, normalize_ir


def shipping_graph() -> dict:
    return {
        "nodes": [
            {"id": "order", "type": "task", "label": "鎺ュ崟", "x": 0, "y": 0},
            {"id": "shipping", "type": "task", "label": "鍙戣揣", "x": 220, "y": 0},
            {"id": "done", "type": "task", "label": "瀹屾垚", "x": 440, "y": 0},
        ],
        "edges": [
            {"id": "e_order_shipping", "source": "order", "target": "shipping", "label": ""},
            {"id": "e_shipping_done", "source": "shipping", "target": "done", "label": ""},
        ],
    }


def ordering_goal() -> dict:
    return {
        "status": "resolved",
        "semantic_goals": [
            {
                "kind": "ordering",
                "text": "insert confirmation before shipping",
                "subject": {"type": "new_entity", "label": "鍑哄簱纭"},
                "anchor": {"type": "existing_reference", "text": "鍙戣揣"},
                "relation": "before",
                "explicit_constraints": [],
            }
        ],
        "clarification": {"needed": False, "question": "", "reason": ""},
    }


def existing_join_graph() -> dict:
    return {
        "nodes": [
            {"id": "audit", "type": "task", "label": "瀹℃牳", "x": 0, "y": 0},
            {"id": "risk", "type": "task", "label": "椋庢帶", "x": 0, "y": 120},
            {"id": "merge", "type": "task", "label": "鍚堝苟", "x": 220, "y": 60, "join": {"mode": "all"}},
            {"id": "done", "type": "end", "label": "瀹屾垚", "x": 440, "y": 60},
        ],
        "edges": [
            {"id": "e_audit_merge", "source": "audit", "target": "merge", "label": ""},
            {"id": "e_risk_merge", "source": "risk", "target": "merge", "label": ""},
            {"id": "e_merge_done", "source": "merge", "target": "done", "label": ""},
        ],
    }


def existing_join_goal() -> dict:
    return {
        "status": "resolved",
        "semantic_goals": [
            {
                "kind": "ordering",
                "text": "鍦ㄥ悎骞跺悗澧炲姞閫氱煡鑺傜偣",
                "subject": {"type": "new_entity", "label": "閫氱煡鑺傜偣"},
                "anchor": {"type": "existing_reference", "text": "鍚堝苟"},
                "relation": "after",
                "explicit_constraints": ["preserve_existing_join"],
            }
        ],
        "clarification": {"needed": False, "question": "", "reason": ""},
    }


def existing_join_grouped_insert_goal() -> dict:
    return {
        "status": "resolved",
        "semantic_goals": [
            {
                "id": "g1",
                "goal_category": "composite_goal",
                "text": "after audit and risk jointly complete, add compliance, then continue to done",
                "objects": [
                    {"id": "audit_obj", "type": "existing_reference", "text": "瀹℃牳", "role": "source"},
                    {"id": "risk_obj", "type": "existing_reference", "text": "椋庢帶", "role": "source"},
                    {"id": "confirm_obj", "type": "new_entity", "label": "鍚堣纭", "role": "target"},
                    {"id": "done_obj", "type": "existing_reference", "text": "瀹屾垚", "role": "target"},
                ],
                "object_desired_states": [{"object": "confirm_obj", "state": "exists"}],
                "relation_intents": [
                    {"source": "audit_obj", "relation": "before", "target": "confirm_obj"},
                    {"source": "risk_obj", "relation": "before", "target": "confirm_obj"},
                    {"source": "confirm_obj", "relation": "before", "target": "done_obj"},
                ],
                "semantic_constraints": [],
            }
        ],
        "clarification": {"needed": False, "question": "", "reason": ""},
    }


def edge_set(graph: dict) -> set[tuple[str, str, str]]:
    return {
        (edge["source"], edge["target"], edge.get("label", ""))
        for edge in graph.get("edges", [])
        if isinstance(edge, dict)
    }


def chain_graph() -> dict:
    return {
        "nodes": [
            {"id": "start", "type": "start", "label": "Start", "x": 0, "y": 0},
            {"id": "a", "type": "task", "label": "A", "x": 100, "y": 0},
            {"id": "b", "type": "task", "label": "B", "x": 200, "y": 0},
            {"id": "c", "type": "task", "label": "C", "x": 300, "y": 0},
            {"id": "d", "type": "task", "label": "D", "x": 400, "y": 0},
            {"id": "reject", "type": "task", "label": "Reject", "x": 300, "y": 120},
        ],
        "edges": [
            {"id": "e_start_b", "source": "start", "target": "b", "label": ""},
            {"id": "e_b_a", "source": "b", "target": "a", "label": ""},
            {"id": "e_a_c", "source": "a", "target": "c", "label": "success", "branch_label": "OK"},
            {"id": "e_a_reject", "source": "a", "target": "reject", "label": "failure", "branch_label": "No stock"},
        ],
    }


def insert_graph() -> dict:
    return {
        "nodes": [
            {"id": "x", "type": "task", "label": "Intake", "x": 0, "y": 0},
            {"id": "a", "type": "task", "label": "Validate", "x": 100, "y": 0},
            {"id": "b", "type": "task", "label": "Enrich", "x": 200, "y": 0},
            {"id": "c", "type": "task", "label": "Review", "x": 300, "y": 0},
        ],
        "edges": [
            {"id": "e_x_a", "source": "x", "target": "a", "label": ""},
            {"id": "e_a_b", "source": "a", "target": "b", "label": ""},
            {"id": "e_b_c", "source": "b", "target": "c", "label": ""},
        ],
    }


def insert_human_review_goal() -> dict:
    return {
        "status": "resolved",
        "semantic_goals": [
            {
                "id": "g1",
                "goal_category": "composite_goal",
                "text": "Insert a Human Review step between Validate and Enrich.",
                "objects": [
                    {"id": "validate", "type": "existing_reference", "text": "Validate", "role": "object"},
                    {"id": "human", "type": "new_entity", "label": "Human Review", "role": "object"},
                    {"id": "enrich", "type": "existing_reference", "text": "Enrich", "role": "object"},
                ],
                "object_desired_states": [{"object": "human", "state": "exists"}],
                "relation_intents": [
                    {"source": "validate", "relation": "before", "target": "human"},
                    {"source": "human", "relation": "before", "target": "enrich"},
                ],
                "semantic_constraints": [],
            }
        ],
        "clarification": {"needed": False, "question": "", "reason": ""},
    }


def parallel_graph(join_target: str = "c") -> dict:
    edges = [
        {"id": "e_x_a", "source": "x", "target": "a", "label": ""},
        {"id": "e_a_b", "source": "a", "target": "b", "label": ""},
        {"id": "e_b_d", "source": "b", "target": "d", "label": ""},
    ]
    if join_target == "c":
        edges[-1] = {"id": "e_b_c", "source": "b", "target": "c", "label": ""}
    return {
        "nodes": [
            {"id": "x", "type": "task", "label": "Start", "x": 0, "y": 0},
            {"id": "a", "type": "task", "label": "A", "x": 100, "y": 0},
            {"id": "b", "type": "task", "label": "B", "x": 200, "y": 0},
            {"id": "c", "type": "task", "label": "C", "x": 300, "y": 0},
            {"id": "d", "type": "task", "label": "D", "x": 400, "y": 0},
            {"id": "reject", "type": "task", "label": "Reject", "x": 200, "y": 120},
        ],
        "edges": [
            *edges,
            {"id": "e_a_reject", "source": "a", "target": "reject", "label": "failure", "branch_label": "No"},
        ],
    }


def parallel_continuation_goal(
    *,
    use_joins_before: bool = False,
    include_parallel: bool = True,
    include_a_continuation: bool = True,
    target_b: str = "c",
    include_b_continuation: bool = True,
    include_property: bool = False,
) -> dict:
    relation = "joins_before" if use_joins_before else "before"
    objects = [
        {"id": "a", "type": "existing_reference", "text": "A", "role": "object"},
        {"id": "b", "type": "existing_reference", "text": "B", "role": "object"},
        {"id": "c", "type": "existing_reference", "text": "C", "role": "object"},
        {"id": "d", "type": "existing_reference", "text": "D", "role": "object"},
    ]
    relation_intents = []
    if include_parallel:
        relation_intents.append({"source": "a", "relation": "parallel_with", "target": "b"})
    if include_a_continuation:
        relation_intents.append({"source": "a", "relation": relation, "target": "c"})
    if include_b_continuation:
        relation_intents.append({"source": "b", "relation": relation, "target": target_b})
    states = []
    if include_property:
        states.append({"object": "a", "state": "property_equals", "property": "display_label", "value": "A+"})
    return {
        "status": "resolved",
        "semantic_goals": [
            {
                "id": "g1",
                "goal_category": "composite_goal",
                "text": "parallel continuation",
                "objects": objects,
                "object_desired_states": states,
                "relation_intents": relation_intents,
                "semantic_constraints": [],
            }
        ],
        "clarification": {"needed": False, "question": "", "reason": ""},
    }


def unresolved_anchor_ordering_goal() -> dict:
    return {
        "status": "clarification",
        "semantic_goals": [
            {
                "id": "g1",
                "goal_category": "relation_goal",
                "objects": [
                    {"id": "validate", "type": "existing_reference", "text": "Validate", "role": "object"},
                    {"id": "missing", "type": "unresolved_reference", "role": "anchor"},
                ],
                "relation_intents": [{"source": "validate", "relation": "before", "target": "missing"}],
                "unresolved_dimensions": [],
            }
        ],
        "clarification": {"needed": True, "question": "Which target?", "reason": "missing target"},
    }


def v2_ordering_goal(relations: list[tuple[str, str]]) -> dict:
    labels = {"a": "A", "b": "B", "c": "C", "d": "D"}
    object_ids = sorted({node_id for relation in relations for node_id in relation})
    return {
        "status": "resolved",
        "semantic_goals": [
            {
                "id": "g1",
                "goal_category": "relation_goal",
                "text": "ordering relations",
                "objects": [
                    {"id": node_id, "type": "existing_reference", "text": labels[node_id], "role": "object"}
                    for node_id in object_ids
                ],
                "relation_intents": [
                    {"source": source, "relation": "before", "target": target}
                    for source, target in relations
                ],
                "object_desired_states": [],
                "semantic_constraints": [],
            }
        ],
        "clarification": {"needed": False, "question": "", "reason": ""},
    }


def business_chain_graph() -> dict:
    return {
        "nodes": [
            {"id": "n_start", "type": "start", "label": "\u7528\u6237\u9700\u6c42"},
            {"id": "n_submit", "type": "task", "label": "\u89e3\u6790\u4efb\u52a1"},
            {"id": "n_retrieval", "type": "task", "label": "\u77e5\u8bc6\u68c0\u7d22"},
            {"id": "n_tool", "type": "task", "label": "\u5de5\u5177\u8c03\u7528"},
            {"id": "n_decision", "type": "decision", "label": "\u7ed3\u679c\u5224\u65ad"},
            {"id": "n_report", "type": "task", "label": "\u751f\u6210\u62a5\u544a"},
            {"id": "n_publish", "type": "task", "label": "\u53d1\u5e03\u7ed3\u679c"},
            {"id": "n_human", "type": "task", "label": "\u4eba\u5de5\u590d\u6838"},
            {"id": "n_end", "type": "end", "label": "\u7ed3\u675f"},
        ],
        "edges": [
            {"id": "e_start_submit", "source": "n_start", "target": "n_submit", "label": ""},
            {"id": "e_submit_retrieval", "source": "n_submit", "target": "n_retrieval", "label": ""},
            {"id": "e_retrieval_tool", "source": "n_retrieval", "target": "n_tool", "label": ""},
            {"id": "e_tool_decision", "source": "n_tool", "target": "n_decision", "label": ""},
            {"id": "e_decision_report", "source": "n_decision", "target": "n_report", "label": "success", "branch_label": "\u6210\u529f"},
            {"id": "e_decision_human", "source": "n_decision", "target": "n_human", "label": "failure", "branch_label": "\u5931\u8d25"},
            {"id": "e_report_publish", "source": "n_report", "target": "n_publish", "label": ""},
            {"id": "e_report_human", "source": "n_report", "target": "n_human", "label": "failure", "branch_label": "\u5931\u8d25"},
            {"id": "e_human_publish", "source": "n_human", "target": "n_publish", "label": ""},
            {"id": "e_publish_end", "source": "n_publish", "target": "n_end", "label": ""},
        ],
    }


def business_reorder_goal(*, complete: bool) -> dict:
    relation_intents = [{"source": "tool", "relation": "before", "target": "retrieval"}]
    objects = [
        {"id": "tool", "type": "existing_reference", "text": "\u5de5\u5177\u8c03\u7528", "role": "object"},
        {"id": "retrieval", "type": "existing_reference", "text": "\u77e5\u8bc6\u68c0\u7d22", "role": "object"},
    ]
    if complete:
        objects.extend(
            [
                {"id": "submit", "type": "existing_reference", "text": "\u89e3\u6790\u4efb\u52a1", "role": "object"},
                {"id": "decision", "type": "existing_reference", "text": "\u7ed3\u679c\u5224\u65ad", "role": "object"},
            ]
        )
        relation_intents = [
            {"source": "tool", "relation": "before", "target": "retrieval"},
            {"source": "submit", "relation": "before", "target": "tool"},
            {"source": "retrieval", "relation": "before", "target": "decision"},
        ]
    return {
        "status": "resolved",
        "semantic_goals": [
            {
                "id": "g1",
                "goal_category": "relation_goal",
                "text": "reorder local chain",
                "objects": objects,
                "relation_intents": relation_intents,
                "object_desired_states": [],
                "semantic_constraints": [{"type": "preserve", "target": "\u5176\u4ed6\u8282\u70b9\u548c\u8fde\u7ebf"}],
            }
        ],
        "clarification": {"needed": False, "question": "", "reason": ""},
    }


def run_business_reorder(monkeypatch, *, complete: bool, relation_groups: list[dict]) -> dict:
    goal = business_reorder_goal(complete=complete)
    instruction = (
        "\u628a\u201c\u5de5\u5177\u8c03\u7528\u201d\u8c03\u6574\u5230\u201c\u77e5\u8bc6\u68c0\u7d22\u201d\u4e4b\u524d\uff0c"
        "\u4fdd\u6301\u5b83\u4eec\u4ecd\u7136\u4f4d\u4e8e\u201c\u89e3\u6790\u4efb\u52a1\u201d\u548c\u201c\u7ed3\u679c\u5224\u65ad\u201d\u4e4b\u95f4\u3002"
    )

    def complete_json(system: str, _payload: dict) -> SimpleNamespace:
        if "relation_groups" in system:
            return SimpleNamespace(raw_response="", parsed_json={"relation_groups": relation_groups}, parse_error=None, normalized_from_fence=False)
        return SimpleNamespace(raw_response="", parsed_json=goal, parse_error=None, normalized_from_fence=False)

    monkeypatch.setattr(semantic_shadow, "_complete_json", complete_json)
    return semantic_shadow.run_semantic_shadow(business_chain_graph(), instruction)


def assert_business_reorder_graph(trace: dict) -> None:
    assert trace["stage"] == "success"
    final_edges = edge_set(trace["final_shadow_graph"])
    assert ("n_submit", "n_tool", "") in final_edges
    assert ("n_tool", "n_retrieval", "") in final_edges
    assert ("n_retrieval", "n_decision", "") in final_edges
    assert ("n_submit", "n_retrieval", "") not in final_edges
    assert ("n_retrieval", "n_tool", "") not in final_edges
    assert ("n_tool", "n_decision", "") not in final_edges
    assert ("n_decision", "n_report", "success") in final_edges
    assert ("n_decision", "n_human", "failure") in final_edges
    assert ("n_report", "n_publish", "") in final_edges
    assert len(trace["final_shadow_graph"]["nodes"]) == 9
    assert len(trace["final_shadow_graph"]["edges"]) == 10


def lowered_sequence_regions(relations: list[tuple[str, str]]) -> list[list[str]]:
    grounded = ground_semantic_goal_spec(v2_ordering_goal(relations), chain_graph(), {"node_ids": [], "edge_ids": []})
    lowered = lower_grounded_goal_to_workflow_ir(grounded, chain_graph())
    assert lowered["ok"] is True
    regions = lowered["semantic_spec"]["workflow_ir"]["control_flow_regions"]
    return [
        [item["id"] for item in region["structure"]["items"]]
        for region in regions
        if region["structure"].get("type") == "sequence"
    ]


def test_contiguous_ordering_fragments_lower_to_one_sequence() -> None:
    assert lowered_sequence_regions([("a", "b"), ("b", "c")]) == [["a", "b", "c"]]


def test_three_contiguous_ordering_fragments_lower_to_one_sequence() -> None:
    assert lowered_sequence_regions([("a", "b"), ("b", "c"), ("c", "d")]) == [["a", "b", "c", "d"]]


def test_contiguous_ordering_fragment_input_order_does_not_change_sequence() -> None:
    assert lowered_sequence_regions([("b", "c"), ("a", "b")]) == [["a", "b", "c"]]


def test_fan_out_ordering_fragments_are_not_merged() -> None:
    assert lowered_sequence_regions([("a", "b"), ("a", "c")]) == [["a", "b"], ["a", "c"]]


def test_fan_in_ordering_fragments_are_not_merged() -> None:
    assert lowered_sequence_regions([("a", "c"), ("b", "c")]) == [["a", "c"], ["b", "c"]]


def test_independent_ordering_fragments_remain_independent() -> None:
    assert lowered_sequence_regions([("a", "b"), ("c", "d")]) == [["a", "b"], ["c", "d"]]


def test_cyclic_ordering_fragments_are_not_merged() -> None:
    assert lowered_sequence_regions([("a", "b"), ("b", "a")]) == [["a", "b"], ["b", "a"]]


def test_single_ordering_relation_lowering_is_unchanged() -> None:
    assert lowered_sequence_regions([("a", "b")]) == [["a", "b"]]


def test_joins_before_normalizes_to_executable_before_ordering() -> None:
    grounded = ground_semantic_goal_spec(
        parallel_continuation_goal(use_joins_before=True),
        parallel_graph(),
        {"node_ids": [], "edge_ids": []},
    )

    relations = [
        (goal["kind"], goal["relation"], [(item["role"], item.get("reference")) for item in goal["objects"]])
        for goal in grounded["semantic_goals"]
    ]

    assert ("parallel", "parallel_with", [("left", "A"), ("right", "B")]) in relations
    assert ("ordering", "before", [("subject", "A"), ("anchor", "C")]) in relations
    assert ("ordering", "before", [("subject", "B"), ("anchor", "C")]) in relations
    assert all(goal["relation"] != "joins_before" for goal in grounded["semantic_goals"])


def test_parallel_with_common_before_continuation_lowers_to_nested_sequence() -> None:
    grounded = ground_semantic_goal_spec(parallel_continuation_goal(), parallel_graph(), {"node_ids": [], "edge_ids": []})

    lowered = lower_grounded_goal_to_workflow_ir(grounded, parallel_graph())

    assert lowered["ok"] is True
    regions = lowered["semantic_spec"]["workflow_ir"]["control_flow_regions"]
    assert regions == [
        {
            "relation_type": "",
            "structure": {
                "type": "sequence",
                "items": [
                    {
                        "type": "parallel",
                        "join_mode": "all",
                        "branches": [{"type": "node", "id": "a"}, {"type": "node", "id": "b"}],
                    },
                    {"type": "node", "id": "c"},
                ],
            },
        }
    ]


def test_parallel_explicit_join_target_does_not_use_old_graph_successor() -> None:
    graph = parallel_graph(join_target="d")
    grounded = ground_semantic_goal_spec(parallel_continuation_goal(), graph, {"node_ids": [], "edge_ids": []})

    lowered = lower_grounded_goal_to_workflow_ir(grounded, graph)

    assert lowered["ok"] is True
    items = lowered["semantic_spec"]["workflow_ir"]["control_flow_regions"][0]["structure"]["items"]
    assert items[0]["type"] == "parallel"
    assert items[1] == {"type": "node", "id": "c"}


def test_parallel_only_lowering_keeps_simple_parallel_region() -> None:
    grounded = ground_semantic_goal_spec(
        parallel_continuation_goal(include_a_continuation=False, include_b_continuation=False),
        parallel_graph(),
        {"node_ids": [], "edge_ids": []},
    )

    lowered = lower_grounded_goal_to_workflow_ir(grounded, parallel_graph())

    assert lowered["ok"] is True
    assert lowered["semantic_spec"]["workflow_ir"]["control_flow_regions"] == [
        {
            "relation_type": "",
            "structure": {
                "type": "parallel",
                "join_mode": "all",
                "branches": [{"type": "node", "id": "a"}, {"type": "node", "id": "b"}],
            },
        }
    ]


def test_common_before_without_parallel_fact_does_not_create_parallel() -> None:
    grounded = ground_semantic_goal_spec(
        parallel_continuation_goal(include_parallel=False),
        parallel_graph(),
        {"node_ids": [], "edge_ids": []},
    )

    lowered = lower_grounded_goal_to_workflow_ir(grounded, parallel_graph())

    assert lowered["ok"] is True
    structures = [region["structure"] for region in lowered["semantic_spec"]["workflow_ir"]["control_flow_regions"]]
    assert all(structure["type"] == "sequence" for structure in structures)
    assert all("parallel" not in {item.get("type") for item in structure["items"]} for structure in structures)


def test_conflicting_parallel_continuation_targets_do_not_guess_join() -> None:
    grounded = ground_semantic_goal_spec(
        parallel_continuation_goal(target_b="d"),
        parallel_graph(),
        {"node_ids": [], "edge_ids": []},
    )

    lowered = lower_grounded_goal_to_workflow_ir(grounded, parallel_graph())

    assert lowered["ok"] is True
    regions = lowered["semantic_spec"]["workflow_ir"]["control_flow_regions"]
    assert len(regions) == 3
    assert regions[0]["structure"]["type"] == "parallel"
    assert [region["structure"]["type"] for region in regions[1:]] == ["sequence", "sequence"]


def test_incomplete_parallel_continuation_does_not_fill_missing_branch() -> None:
    grounded = ground_semantic_goal_spec(
        parallel_continuation_goal(include_b_continuation=False),
        parallel_graph(),
        {"node_ids": [], "edge_ids": []},
    )

    lowered = lower_grounded_goal_to_workflow_ir(grounded, parallel_graph())

    assert lowered["ok"] is True
    regions = lowered["semantic_spec"]["workflow_ir"]["control_flow_regions"]
    assert len(regions) == 2
    assert regions[0]["structure"]["type"] == "parallel"
    assert regions[1]["structure"] == {"type": "sequence", "items": [{"type": "node", "id": "a"}, {"type": "node", "id": "c"}]}


def test_multigoal_property_and_parallel_common_continuation_lower_together() -> None:
    grounded = ground_semantic_goal_spec(
        parallel_continuation_goal(include_property=True),
        parallel_graph(),
        {"node_ids": [], "edge_ids": []},
    )

    lowered = lower_grounded_goal_to_workflow_ir(grounded, parallel_graph())

    assert lowered["ok"] is True
    workflow_ir = lowered["semantic_spec"]["workflow_ir"]
    assert workflow_ir["properties"] == [{"node_id": "a", "changes": {"label": "A+"}}]
    assert workflow_ir["control_flow_regions"][0]["structure"]["items"][0]["type"] == "parallel"
    assert workflow_ir["control_flow_regions"][0]["structure"]["items"][1] == {"type": "node", "id": "c"}


def test_parallel_common_continuation_shadow_preserves_branch_metadata(monkeypatch) -> None:
    monkeypatch.setattr(
        semantic_shadow,
        "_complete_json",
        lambda _system, _payload: SimpleNamespace(
            raw_response="",
            parsed_json=parallel_continuation_goal(),
            parse_error=None,
            normalized_from_fence=False,
        ),
    )

    trace = semantic_shadow.run_semantic_shadow(parallel_graph(), "Run A and B in parallel, then both continue to C.")

    assert trace["stage"] == "success"
    final_edges = edge_set(trace["final_shadow_graph"])
    assert ("x", "a", "") in final_edges
    assert ("x", "b", "") in final_edges
    assert ("a", "c", "") in final_edges
    assert ("b", "c", "") in final_edges
    assert ("a", "b", "") not in final_edges
    failure_edge = next(edge for edge in trace["final_shadow_graph"]["edges"] if edge.get("id") == "e_a_reject")
    assert failure_edge["source"] == "a"
    assert failure_edge["target"] == "reject"
    assert failure_edge["label"] == "failure"
    assert failure_edge["branch_label"] == "No"


def test_parallel_common_continuation_does_not_commit_incomplete_boundary(monkeypatch) -> None:
    monkeypatch.setattr(
        semantic_shadow,
        "_complete_json",
        lambda _system, _payload: SimpleNamespace(
            raw_response="",
            parsed_json=parallel_continuation_goal(),
            parse_error=None,
            normalized_from_fence=False,
        ),
    )

    graph = parallel_graph()
    graph["nodes"].append({"id": "y", "type": "task", "label": "Other Start"})
    graph["edges"].append({"id": "e_y_a", "source": "y", "target": "a", "label": ""})

    result = semantic_shadow.run_semantic_transaction(graph, "Run A and B in parallel, then both continue to C.")

    assert result.status in {"needs_clarification", "unsupported"}
    assert result.final_graph is None
    issues = result.specification["trace"]["failure_reason"]
    assert issues[0]["kind"] == "ambiguous_structural_boundary"
    assert "boundary" in issues[0]["message"]


def test_parallel_without_explicit_continuation_infers_serial_chain_boundary(monkeypatch) -> None:
    goal = parallel_continuation_goal(include_a_continuation=False, include_b_continuation=False)
    goal["semantic_goals"][0]["objects"] = goal["semantic_goals"][0]["objects"][:2]
    monkeypatch.setattr(
        semantic_shadow,
        "_complete_json",
        lambda _system, _payload: SimpleNamespace(
            raw_response="",
            parsed_json=goal,
            parse_error=None,
            normalized_from_fence=False,
        ),
    )

    trace = semantic_shadow.run_semantic_shadow(parallel_graph(), "Run A and B in parallel.")

    assert trace["stage"] == "success"
    final_edges = edge_set(trace["final_shadow_graph"])
    assert ("x", "a", "") in final_edges
    assert ("x", "b", "") in final_edges
    assert ("a", "c", "") in final_edges
    assert ("b", "c", "") in final_edges
    assert ("a", "b", "") not in final_edges


def test_parallel_inference_stops_for_labeled_internal_business_edge(monkeypatch) -> None:
    goal = parallel_continuation_goal(include_a_continuation=False, include_b_continuation=False)
    goal["semantic_goals"][0]["objects"] = goal["semantic_goals"][0]["objects"][:2]
    monkeypatch.setattr(
        semantic_shadow,
        "_complete_json",
        lambda _system, _payload: SimpleNamespace(
            raw_response="",
            parsed_json=goal,
            parse_error=None,
            normalized_from_fence=False,
        ),
    )
    graph = parallel_graph()
    for edge in graph["edges"]:
        if edge["id"] == "e_a_b":
            edge["label"] = "failure"
            edge["branch_label"] = "No"

    result = semantic_shadow.run_semantic_transaction(graph, "Run A and B in parallel.")

    assert result.status in {"needs_clarification", "unsupported"}
    assert result.final_graph is None


def test_existing_before_new_entity_is_valid_ordering_participant() -> None:
    grounded = ground_semantic_goal_spec(insert_human_review_goal(), insert_graph(), {"node_ids": [], "edge_ids": []})

    validation = validate_semantic_goal_spec(insert_human_review_goal(), "Insert a Human Review step between Validate and Enrich.")
    # Semantic validation remains ready; grounded validation is the contract repaired here.
    assert validation["ready_for_grounding"] is True
    grounded_validation = validate_grounded_goal(grounded)

    assert grounded["grounding"]["issues"] == []
    assert "missing_ordering_anchor" not in {issue["kind"] for issue in grounded_validation["issues"]}
    assert grounded_validation["lowering_ready"] is True


def test_existing_new_existing_ordering_lowers_to_one_sequence_and_one_new_node() -> None:
    grounded = ground_semantic_goal_spec(insert_human_review_goal(), insert_graph(), {"node_ids": [], "edge_ids": []})

    lowered = lower_grounded_goal_to_workflow_ir(grounded, insert_graph())

    assert lowered["ok"] is True
    workflow_ir = lowered["semantic_spec"]["workflow_ir"]
    items = workflow_ir["control_flow_regions"][0]["structure"]["items"]
    assert items[0] == {"type": "node", "id": "a"}
    assert items[2] == {"type": "node", "id": "b"}
    assert items[1] == {"type": "node", "ref": "new_human"}
    entity = workflow_ir["entities"][0]
    assert entity["ref"] == "new_human"
    assert entity["node_type"] == "task"
    assert entity["label"] == "Human Review"
    assert isinstance(entity["x"], (int, float))
    assert isinstance(entity["y"], (int, float))
    assert len([item for item in items if item.get("ref") == "new_human"]) == 1


def test_insert_human_review_shadow_creates_one_node_and_rewires(monkeypatch) -> None:
    monkeypatch.setattr(
        semantic_shadow,
        "_complete_json",
        lambda _system, _payload: SimpleNamespace(
            raw_response="",
            parsed_json=insert_human_review_goal(),
            parse_error=None,
            normalized_from_fence=False,
        ),
    )

    trace = semantic_shadow.run_semantic_shadow(insert_graph(), "Insert a Human Review step between Validate and Enrich.")

    assert trace["stage"] == "success"
    new_nodes = [node for node in trace["final_shadow_graph"]["nodes"] if node.get("label") == "Human Review"]
    assert len(new_nodes) == 1
    new_id = new_nodes[0]["id"]
    final_edges = edge_set(trace["final_shadow_graph"])
    assert ("a", new_id, "") in final_edges
    assert (new_id, "b", "") in final_edges
    assert ("a", "b", "") not in final_edges
    assert ("x", "a", "") in final_edges
    assert ("b", "c", "") in final_edges


def shared_new_graph() -> dict:
    return {
        "nodes": [
            {"id": "A", "type": "task", "label": "A", "x": 0, "y": 0},
            {"id": "B", "type": "task", "label": "B", "x": 220, "y": 0},
            {"id": "C", "type": "task", "label": "C", "x": 440, "y": 0},
            {"id": "D", "type": "task", "label": "D", "x": 660, "y": 0},
        ],
        "edges": [],
    }


def scoped_join_graph() -> dict:
    return {
        "nodes": [
            {"id": "A", "type": "task", "label": "A", "x": 0, "y": 0},
            {"id": "B", "type": "task", "label": "B", "x": 0, "y": 120},
            {"id": "J", "type": "task", "label": "J", "x": 220, "y": 60},
            {"id": "C", "type": "task", "label": "C", "x": 440, "y": 60},
        ],
        "edges": [
            {"id": "e_a_j", "source": "A", "target": "J", "label": ""},
            {"id": "e_b_j", "source": "B", "target": "J", "label": ""},
            {"id": "e_j_c", "source": "J", "target": "C", "label": ""},
        ],
    }


def incompatible_scope_graph() -> dict:
    return {
        "nodes": [
            {"id": "A", "type": "task", "label": "A"},
            {"id": "B", "type": "task", "label": "B"},
            {"id": "JA", "type": "task", "label": "JA"},
            {"id": "JB", "type": "task", "label": "JB"},
        ],
        "edges": [
            {"id": "e_a_ja", "source": "A", "target": "JA", "label": ""},
            {"id": "e_b_jb", "source": "B", "target": "JB", "label": ""},
        ],
    }


def _existing_binding(role: str, node_id: str) -> dict:
    return {"role": role, "binding_type": "node", "id": node_id}


def _new_binding(role: str, object_id: str, label: str = "X") -> dict:
    return {
        "role": role,
        "type": "new_entity",
        "label": label,
        "v2_object_id": object_id,
        "needs_creation": True,
    }


def _ordering_goal(subject: dict, anchor: dict) -> dict:
    return {"kind": "ordering", "relation": "before", "bindings": [subject, anchor]}


def _grounded_orderings(goals: list[dict]) -> dict:
    return {
        "status": "resolved",
        "semantic_goals": goals,
        "semantic_goals_v2": [],
        "grounding": {"issues": []},
        "clarification": {"needed": False, "question": "", "reason": ""},
    }


def _lower_shared_new(goals: list[dict]) -> dict:
    lowered = lower_grounded_goal_to_workflow_ir(_grounded_orderings(goals), shared_new_graph())
    assert lowered["ok"] is True
    return lowered["semantic_spec"]["workflow_ir"]


def _grounded_relation_fact_goal(relation_facts: list[dict], semantic_goals: list[dict] | None = None) -> dict:
    return {
        "status": "resolved",
        "semantic_goals": semantic_goals or [],
        "relation_facts": relation_facts,
        "semantic_goals_v2": [],
        "grounding": {"issues": []},
        "clarification": {"needed": False, "question": "", "reason": ""},
    }


def _grounded_precedence(relation_id: str, pairs: list[tuple[dict, dict]]) -> dict:
    return {
        "relation_id": relation_id,
        "kind": "precedence",
        "members": [
            {"source_binding": source_binding, "target_binding": target_binding, "properties": {}}
            for source_binding, target_binding in pairs
        ],
    }


def _rf_node(node_id: str) -> dict:
    return {"binding_type": "node", "id": node_id}


def _rf_new(ref_id: str, label: str = "X") -> dict:
    return {"ref_id": ref_id, "binding_type": "new_entity", "needs_creation": True, "label": label}


def _constraints_for(ir: dict, graph: dict) -> dict:
    return constraints_from_compiled(graph, compile_ir(ir, graph))


def _compiled_relations(ir: dict) -> set[tuple[str, str, str]]:
    compiled = compile_ir(ir, shared_new_graph())
    constraints = constraints_from_compiled(shared_new_graph(), compiled)
    return {(row["source"], row["target"], row.get("label", "")) for row in constraints["required_relations"]}


def test_goal_lowering_declares_simple_new_entity_once_for_existing_new_existing_ordering() -> None:
    ir = _lower_shared_new(
        [
            _ordering_goal(_existing_binding("subject", "A"), _new_binding("anchor", "x")),
            _ordering_goal(_new_binding("subject", "x"), _existing_binding("anchor", "B")),
        ]
    )

    entity = ir["entities"][0]
    assert {key: entity[key] for key in ("ref", "node_type", "label")} == {"ref": "new_x", "node_type": "task", "label": "X"}
    assert isinstance(entity["x"], (int, float))
    assert isinstance(entity["y"], (int, float))
    assert ir["control_flow_regions"][0]["structure"]["items"] == [
        {"type": "node", "id": "A"},
        {"type": "node", "ref": "new_x"},
        {"type": "node", "id": "B"},
    ]
    assert _compiled_relations(ir) == {("A", "new_x", ""), ("new_x", "B", "")}


def test_goal_lowering_reuses_new_entity_for_shared_predecessor_ordering() -> None:
    ir = _lower_shared_new(
        [
            _ordering_goal(_new_binding("subject", "x"), _existing_binding("anchor", "A")),
            _ordering_goal(_new_binding("subject", "x"), _existing_binding("anchor", "B")),
        ]
    )

    assert [row["ref"] for row in ir["entities"]] == ["new_x"]
    assert _compiled_relations(ir) == {("new_x", "A", ""), ("new_x", "B", "")}


def test_goal_lowering_reuses_new_entity_for_shared_successor_ordering() -> None:
    ir = _lower_shared_new(
        [
            _ordering_goal(_existing_binding("subject", "A"), _new_binding("anchor", "x")),
            _ordering_goal(_existing_binding("subject", "B"), _new_binding("anchor", "x")),
        ]
    )

    assert [row["ref"] for row in ir["entities"]] == ["new_x"]
    assert _compiled_relations(ir) == {("A", "new_x", ""), ("B", "new_x", "")}


def test_goal_lowering_preserves_general_shared_new_node_relations_without_parallel_inference() -> None:
    ir = _lower_shared_new(
        [
            _ordering_goal(_existing_binding("subject", "A"), _new_binding("anchor", "x")),
            _ordering_goal(_existing_binding("subject", "B"), _new_binding("anchor", "x")),
            _ordering_goal(_new_binding("subject", "x"), _existing_binding("anchor", "C")),
        ]
    )

    assert [row["ref"] for row in ir["entities"]] == ["new_x"]
    assert _compiled_relations(ir) == {("A", "new_x", ""), ("B", "new_x", ""), ("new_x", "C", "")}
    assert all(region["structure"].get("type") == "sequence" for region in ir["control_flow_regions"])


def test_grouped_relation_scope_preserves_existing_join_boundary_for_shared_new_target() -> None:
    graph = scoped_join_graph()
    grounded = _grounded_relation_fact_goal(
        [
            _grounded_precedence(
                "rg1",
                [
                    (_rf_node("A"), _rf_new("r_x", "X")),
                    (_rf_node("B"), _rf_new("r_x", "X")),
                ],
            )
        ],
        [_ordering_goal(_new_binding("subject", "r_x", "X"), _existing_binding("anchor", "C"))],
    )

    lowered = lower_grounded_goal_to_workflow_ir(grounded, graph)
    assert lowered["ok"] is True
    ir = lowered["semantic_spec"]["workflow_ir"]
    assert [row["ref"] for row in ir["entities"]] == ["new_r_x"]
    scoped_regions = [row for row in ir["control_flow_regions"] if row.get("lowering_scope_id") == "relation_fact:rg1"]
    assert len(scoped_regions) == 2

    constraints = _constraints_for(ir, graph)
    required = {(row["source"], row["target"], row.get("label", "")) for row in constraints["required_relations"]}
    forbidden = {(row["source"], row["target"], row.get("label", "")) for row in constraints["forbidden_relations"]}
    assert ("A", "J", "") in required
    assert ("B", "J", "") in required
    assert ("J", "new_r_x", "") in required
    assert ("new_r_x", "C", "") in required
    assert ("A", "new_r_x", "") not in required
    assert ("B", "new_r_x", "") not in required
    assert ("J", "C", "") in forbidden


def test_all_scoped_grouped_relation_owns_replaced_continuation_edge() -> None:
    graph = scoped_join_graph()
    lowered = lower_grounded_goal_to_workflow_ir(
        _grounded_relation_fact_goal(
            [
                _grounded_precedence(
                    "rg1",
                    [
                        (_rf_node("A"), _rf_new("r_x", "X")),
                        (_rf_node("B"), _rf_new("r_x", "X")),
                        (_rf_new("r_x", "X"), _rf_node("C")),
                    ],
                )
            ]
        ),
        graph,
    )
    assert lowered["ok"] is True
    ir = lowered["semantic_spec"]["workflow_ir"]
    assert len([row for row in ir["control_flow_regions"] if row.get("lowering_scope_id") == "relation_fact:rg1"]) == 3

    compiled = compile_ir(ir, graph)
    scoped = compiled["control_flow_regions"][0]
    assert ("J", "C", "") in {tuple(row) for row in scoped["owned_current_edges"]}
    constraints = constraints_from_compiled(graph, compiled)
    required = {(row["source"], row["target"], row.get("label", "")): row for row in constraints["required_relations"]}
    forbidden = {(row["source"], row["target"], row.get("label", "")) for row in constraints["forbidden_relations"]}
    assert {("A", "J", ""), ("B", "J", ""), ("J", "new_r_x", ""), ("new_r_x", "C", "")} <= set(required)
    assert required[("J", "new_r_x", "")].get("id") == "e_j_c"
    assert ("J", "C", "") in forbidden
    assert ("A", "new_r_x", "") not in set(required)
    assert ("B", "new_r_x", "") not in set(required)


def test_all_scoped_and_mixed_scoped_continuation_have_equivalent_topology_constraints() -> None:
    graph = scoped_join_graph()
    mixed_ir = lower_grounded_goal_to_workflow_ir(
        _grounded_relation_fact_goal(
            [
                _grounded_precedence(
                    "rg1",
                    [
                        (_rf_node("A"), _rf_new("r_x", "X")),
                        (_rf_node("B"), _rf_new("r_x", "X")),
                    ],
                )
            ],
            [_ordering_goal(_new_binding("subject", "r_x", "X"), _existing_binding("anchor", "C"))],
        ),
        graph,
    )["semantic_spec"]["workflow_ir"]
    all_scoped_ir = lower_grounded_goal_to_workflow_ir(
        _grounded_relation_fact_goal(
            [
                _grounded_precedence(
                    "rg1",
                    [
                        (_rf_node("A"), _rf_new("r_x", "X")),
                        (_rf_node("B"), _rf_new("r_x", "X")),
                        (_rf_new("r_x", "X"), _rf_node("C")),
                    ],
                )
            ]
        ),
        graph,
    )["semantic_spec"]["workflow_ir"]

    mixed_constraints = constraints_from_compiled(graph, compile_ir(mixed_ir, graph))
    all_scoped_constraints = constraints_from_compiled(graph, compile_ir(all_scoped_ir, graph))

    def topology(constraints: dict) -> tuple[set[tuple[str, str, str]], set[tuple[str, str, str]]]:
        required = {(row["source"], row["target"], row.get("label", "")) for row in constraints["required_relations"]}
        forbidden = {(row["source"], row["target"], row.get("label", "")) for row in constraints["forbidden_relations"]}
        return required, forbidden

    assert topology(all_scoped_constraints) == topology(mixed_constraints)


def test_scoped_replacement_ownership_is_not_fan_in_specific() -> None:
    graph = {
        "nodes": [
            {"id": "A", "type": "task", "label": "A"},
            {"id": "C", "type": "task", "label": "C"},
        ],
        "edges": [{"id": "e_a_c", "source": "A", "target": "C", "label": ""}],
    }
    lowered = lower_grounded_goal_to_workflow_ir(
        _grounded_relation_fact_goal(
            [_grounded_precedence("rg1", [(_rf_node("A"), _rf_new("r_x", "X")), (_rf_new("r_x", "X"), _rf_node("C"))])]
        ),
        graph,
    )

    constraints = constraints_from_compiled(graph, compile_ir(lowered["semantic_spec"]["workflow_ir"], graph))
    required = {(row["source"], row["target"], row.get("label", "")): row for row in constraints["required_relations"]}
    forbidden = {(row["source"], row["target"], row.get("label", "")) for row in constraints["forbidden_relations"]}
    assert {("A", "new_r_x", ""), ("new_r_x", "C", "")} <= set(required)
    assert required[("A", "new_r_x", "")].get("id") == "e_a_c"
    assert ("A", "C", "") in forbidden


def test_scoped_replacement_ownership_preserves_unrelated_outgoing_edge() -> None:
    graph = scoped_join_graph()
    graph["nodes"].append({"id": "D", "type": "task", "label": "D", "x": 440, "y": 120})
    graph["edges"].append({"id": "e_j_d", "source": "J", "target": "D", "label": ""})
    lowered = lower_grounded_goal_to_workflow_ir(
        _grounded_relation_fact_goal(
            [
                _grounded_precedence(
                    "rg1",
                    [
                        (_rf_node("A"), _rf_new("r_x", "X")),
                        (_rf_node("B"), _rf_new("r_x", "X")),
                        (_rf_new("r_x", "X"), _rf_node("C")),
                    ],
                )
            ]
        ),
        graph,
    )

    constraints = constraints_from_compiled(graph, compile_ir(lowered["semantic_spec"]["workflow_ir"], graph))
    forbidden = {(row["source"], row["target"], row.get("label", "")) for row in constraints["forbidden_relations"]}
    assert ("J", "C", "") in forbidden
    assert ("J", "D", "") not in forbidden


def test_scoped_relation_does_not_rewire_unrelated_incoming_edge_to_region_entry() -> None:
    graph = {
        "nodes": [
            {"id": "report", "type": "task", "label": "Report"},
            {"id": "human", "type": "task", "label": "Human Review"},
            {"id": "publish", "type": "task", "label": "Publish"},
        ],
        "edges": [
            {"id": "e_report_publish", "source": "report", "target": "publish", "label": ""},
            {"id": "e_report_human_failure", "source": "report", "target": "human", "label": "failure", "branch_label": "failure"},
            {"id": "e_human_publish", "source": "human", "target": "publish", "label": ""},
        ],
    }
    lowered = lower_grounded_goal_to_workflow_ir(
        _grounded_relation_fact_goal(
            [
                _grounded_precedence(
                    "rg1",
                    [
                        (_rf_node("human"), _rf_new("r_confirm", "Confirm")),
                        (_rf_new("r_confirm", "Confirm"), _rf_node("publish")),
                    ],
                )
            ]
        ),
        graph,
    )

    constraints = constraints_from_compiled(graph, compile_ir(lowered["semantic_spec"]["workflow_ir"], graph))
    required = {(row["source"], row["target"], row.get("label", "")) for row in constraints["required_relations"]}
    forbidden = {(row["source"], row["target"], row.get("label", "")) for row in constraints["forbidden_relations"]}
    assert ("human", "new_r_confirm", "") in required
    assert ("new_r_confirm", "publish", "") in required
    assert ("report", "human", "") not in required
    assert ("report", "human", "failure") not in forbidden
    assert ("report", "publish", "") not in forbidden
    assert ("human", "publish", "") in forbidden


def test_grounded_preserve_edge_enters_executable_constraints() -> None:
    graph = {
        "nodes": [
            {"id": "report", "type": "task", "label": "Report"},
            {"id": "human", "type": "task", "label": "Human Review"},
            {"id": "publish", "type": "task", "label": "Publish"},
        ],
        "edges": [
            {"id": "e_report_publish", "source": "report", "target": "publish", "label": ""},
            {"id": "e_report_human_failure", "source": "report", "target": "human", "label": "failure"},
            {"id": "e_human_publish", "source": "human", "target": "publish", "label": ""},
        ],
    }
    semantic_goal = {
        "status": "resolved",
        "semantic_goals": [
            {
                "id": "g1",
                "goal_category": "constraint_goal",
                "objects": [],
                "object_desired_states": [],
                "relation_intents": [],
                "semantic_constraints": [
                    {"type": "preserve", "target": "Report Publish", "text": "preserve Report Publish"}
                ],
            }
        ],
        "clarification": {"needed": False, "question": "", "reason": ""},
    }
    grounded = ground_semantic_goal_spec(semantic_goal, graph)
    assert grounded["grounding"]["issues"] == []
    assert grounded["preserve_constraints"]["edge_ids"] == ["e_report_publish"]

    lowered = lower_grounded_goal_to_workflow_ir(
        {
            **_grounded_relation_fact_goal(
                [
                    _grounded_precedence(
                        "rg1",
                        [
                            (_rf_node("human"), _rf_new("r_confirm", "Confirm")),
                            (_rf_new("r_confirm", "Confirm"), _rf_node("publish")),
                        ],
                    )
                ]
            ),
            "preserve_constraints": grounded["preserve_constraints"],
        },
        graph,
    )
    constraints = constraints_from_compiled(graph, compile_ir(lowered["semantic_spec"]["workflow_ir"], graph))
    assert constraints["preserve"]["edge_ids"] == ["e_report_publish"]


def test_abstract_preserve_protects_edges_outside_compiled_scope() -> None:
    graph = {
        "nodes": [
            {"id": "report", "type": "task", "label": "Report"},
            {"id": "human", "type": "task", "label": "Human Review"},
            {"id": "publish", "type": "task", "label": "Publish"},
        ],
        "edges": [
            {"id": "e_report_publish", "source": "report", "target": "publish", "label": ""},
            {"id": "e_report_human_failure", "source": "report", "target": "human", "label": "failure"},
            {"id": "e_human_publish", "source": "human", "target": "publish", "label": ""},
        ],
    }
    lowered = lower_grounded_goal_to_workflow_ir(
        {
            **_grounded_relation_fact_goal(
                [
                    _grounded_precedence(
                        "rg1",
                        [
                            (_rf_node("human"), _rf_new("r_confirm", "Confirm")),
                            (_rf_new("r_confirm", "Confirm"), _rf_node("publish")),
                        ],
                    )
                ]
            ),
            "preserve_constraints": {
                "node_ids": [],
                "edge_ids": [],
                "abstract": [{"target": "the rest of the workflow", "text": "keep the rest unchanged"}],
            },
        },
        graph,
    )
    constraints = constraints_from_compiled(graph, compile_ir(lowered["semantic_spec"]["workflow_ir"], graph))
    assert "e_report_publish" in constraints["preserve"]["edge_ids"]
    assert "e_report_human_failure" in constraints["preserve"]["edge_ids"]
    assert "e_human_publish" not in constraints["preserve"]["edge_ids"]
    planner = plan_constraints(graph, constraints)
    assert planner["supported"] is True
    removed = {step["edge_id"] for step in planner["steps"] if step["op"] == "remove_edge"}
    updated = {step["edge_id"] for step in planner["steps"] if step["op"] == "update_edge"}
    assert removed == set()
    assert updated == {"e_human_publish"}


def test_abstract_preserve_blocks_unowned_edge_removal() -> None:
    graph = {
        "nodes": [
            {"id": "report", "type": "task", "label": "Report"},
            {"id": "publish", "type": "task", "label": "Publish"},
        ],
        "edges": [{"id": "e_report_publish", "source": "report", "target": "publish", "label": ""}],
    }
    constraints = {
        "entities": [],
        "absent_entities": [],
        "properties": [],
        "edge_properties": [],
        "required_relations": [],
        "forbidden_relations": [{"source": "report", "target": "publish", "label": ""}],
        "preserve": {"node_ids": [], "edge_ids": ["e_report_publish"], "abstract": [{"target": "rest"}]},
    }
    planner = plan_constraints(graph, constraints)
    assert planner["supported"] is False
    assert planner["reason"] == "preserve constraint blocks edge removal"


def test_ambiguous_preserve_reference_blocks_grounding() -> None:
    graph = {
        "nodes": [
            {"id": "a", "type": "task", "label": "A"},
            {"id": "b", "type": "task", "label": "B"},
            {"id": "c", "type": "task", "label": "C"},
        ],
        "edges": [
            {"id": "e_ab", "source": "a", "target": "b", "label": "same"},
            {"id": "e_ac", "source": "a", "target": "c", "label": "same"},
        ],
    }
    semantic_goal = {
        "status": "resolved",
        "semantic_goals": [
            {
                "id": "g1",
                "goal_category": "constraint_goal",
                "semantic_constraints": [{"type": "preserve", "target": "same", "text": "preserve same"}],
            }
        ],
        "clarification": {"needed": False, "question": "", "reason": ""},
    }

    grounded = ground_semantic_goal_spec(semantic_goal, graph)
    assert any(issue["kind"] == "ambiguous_preserve_reference" for issue in grounded["grounding"]["issues"])
    assert validate_grounded_goal(grounded)["lowering_ready"] is False


def test_scoped_replacement_does_not_forbid_edge_that_remains_target_edge() -> None:
    graph = {
        "nodes": [
            {"id": "A", "type": "task", "label": "A"},
            {"id": "C", "type": "task", "label": "C"},
        ],
        "edges": [{"id": "e_a_c", "source": "A", "target": "C", "label": ""}],
    }
    ir = {
        "entities": [{"ref": "new_x", "node_type": "task", "label": "X"}],
        "control_flow_regions": [
            {
                "relation_type": "",
                "lowering_scope_id": "relation_fact:rg1",
                "structure": {"type": "sequence", "items": [{"type": "node", "id": "A"}, {"type": "node", "id": "C"}]},
            },
            {
                "relation_type": "",
                "lowering_scope_id": "relation_fact:rg1",
                "structure": {"type": "sequence", "items": [{"type": "node", "id": "A"}, {"type": "node", "ref": "new_x"}, {"type": "node", "id": "C"}]},
            },
        ],
        "auxiliary_relations": [],
        "absent_entities": [],
        "properties": [],
        "edge_properties": [],
    }

    constraints = constraints_from_compiled(graph, compile_ir(ir, graph))
    forbidden = {(row["source"], row["target"], row.get("label", "")) for row in constraints["forbidden_relations"]}
    assert ("A", "C", "") not in forbidden


def test_managed_relation_fact_marks_suppressed_legacy_subgoals_executed() -> None:
    graph = {
        "nodes": [
            {"id": "A", "type": "task", "label": "A"},
            {"id": "C", "type": "task", "label": "C"},
        ],
        "edges": [{"id": "e_a_c", "source": "A", "target": "C", "label": ""}],
    }
    grounded = _grounded_relation_fact_goal(
        [_grounded_precedence("rg1", [(_rf_node("A"), _rf_new("r_x", "X")), (_rf_new("r_x", "X"), _rf_node("C"))])],
    )
    grounded["semantic_goals_v2"] = [
        {
            "id": "g1",
            "objects": [
                {"id": "a", "type": "existing_reference", "reference": "A"},
                {"id": "x", "type": "new_entity", "label": "X"},
                {"id": "c", "type": "existing_reference", "reference": "C"},
            ],
            "object_desired_states": [{"object": "x", "state": "exists"}],
            "relation_intents": [
                {"source": "a", "relation": "before", "target": "x"},
                {"source": "x", "relation": "before", "target": "c"},
            ],
        }
    ]
    grounded["suppressed_legacy_relation_goals"] = [
        {"source_subgoal_id": "g1:relation:0"},
        {"source_subgoal_id": "g1:relation:1"},
    ]

    lowered = lower_grounded_goal_to_workflow_ir(grounded, graph)
    by_id = {row["id"]: row for row in lowered["partial_execution"]["subgoals"]}
    assert by_id["g1:relation:0"]["status"] == "executed"
    assert by_id["g1:relation:1"]["status"] == "executed"


def test_independent_binary_same_target_does_not_get_grouped_scope() -> None:
    graph = scoped_join_graph()
    ir = lower_grounded_goal_to_workflow_ir(
        _grounded_orderings(
            [
                _ordering_goal(_existing_binding("subject", "A"), _new_binding("anchor", "x")),
                _ordering_goal(_existing_binding("subject", "B"), _new_binding("anchor", "x")),
                _ordering_goal(_new_binding("subject", "x"), _existing_binding("anchor", "C")),
            ]
        ),
        graph,
    )["semantic_spec"]["workflow_ir"]

    assert all("lowering_scope_id" not in row for row in ir["control_flow_regions"])
    constraints = _constraints_for(ir, graph)
    required = {(row["source"], row["target"], row.get("label", "")) for row in constraints["required_relations"]}
    assert ("A", "new_x", "") in required
    assert ("B", "new_x", "") in required


def test_grouped_relation_scope_safe_stops_without_compatible_boundary() -> None:
    graph = incompatible_scope_graph()
    lowered = lower_grounded_goal_to_workflow_ir(
        _grounded_relation_fact_goal(
            [
                _grounded_precedence(
                    "rg1",
                    [
                        (_rf_node("A"), _rf_new("r_x", "X")),
                        (_rf_node("B"), _rf_new("r_x", "X")),
                    ],
                )
            ]
        ),
        graph,
    )
    assert lowered["ok"] is True

    compiled = compile_ir(lowered["semantic_spec"]["workflow_ir"], graph)
    scoped = [row for row in compiled["control_flow_regions"] if row.get("lowering_scope_id") == "relation_fact:rg1"]
    assert scoped and scoped[0]["compile_issues"][0]["kind"] == "unsupported_grouped_scope_boundary"
    constraints = constraints_from_compiled(graph, compiled)
    required = {(row["source"], row["target"], row.get("label", "")) for row in constraints["required_relations"]}
    assert ("A", "new_r_x", "") not in required
    assert ("B", "new_r_x", "") not in required


def test_lowering_scope_id_is_compile_metadata_not_constraint_output() -> None:
    ir = {
        "entities": [{"ref": "new_x", "node_type": "task", "label": "X"}],
        "control_flow_regions": [
            {
                "relation_type": "",
                "lowering_scope_id": "relation_fact:rg1",
                "structure": {"type": "sequence", "items": [{"type": "node", "id": "A"}, {"type": "node", "ref": "new_x"}]},
            },
            {
                "relation_type": "",
                "lowering_scope_id": "relation_fact:rg1",
                "structure": {"type": "sequence", "items": [{"type": "node", "id": "B"}, {"type": "node", "ref": "new_x"}]},
            },
        ],
        "auxiliary_relations": [],
        "absent_entities": [],
        "properties": [],
        "edge_properties": [],
    }

    normalized = normalize_ir(ir)
    assert normalized["control_flow_regions"][0]["lowering_scope_id"] == "relation_fact:rg1"
    compiled = compile_ir(normalized, scoped_join_graph())
    assert compiled["control_flow_regions"][0]["lowering_scope_id"] == "relation_fact:rg1"
    constraints = constraints_from_compiled(scoped_join_graph(), compiled)
    assert "lowering_scope_id" not in constraints


def test_goal_lowering_reuses_new_entity_for_more_than_two_refs() -> None:
    ir = _lower_shared_new(
        [
            _ordering_goal(_existing_binding("subject", "A"), _new_binding("anchor", "x")),
            _ordering_goal(_existing_binding("subject", "B"), _new_binding("anchor", "x")),
            _ordering_goal(_existing_binding("subject", "D"), _new_binding("anchor", "x")),
            _ordering_goal(_new_binding("subject", "x"), _existing_binding("anchor", "C")),
        ]
    )

    assert [row["ref"] for row in ir["entities"]] == ["new_x"]
    assert _compiled_relations(ir) == {
        ("A", "new_x", ""),
        ("B", "new_x", ""),
        ("D", "new_x", ""),
        ("new_x", "C", ""),
    }


def test_goal_lowering_keeps_independent_new_entities_separate() -> None:
    ir = _lower_shared_new(
        [
            _ordering_goal(_existing_binding("subject", "A"), _new_binding("anchor", "x", "X")),
            _ordering_goal(_existing_binding("subject", "B"), _new_binding("anchor", "y", "Y")),
        ]
    )

    assert {row["ref"] for row in ir["entities"]} == {"new_x", "new_y"}
    assert _compiled_relations(ir) == {("A", "new_x", ""), ("B", "new_y", "")}


def test_goal_lowering_does_not_merge_same_label_different_semantic_identity() -> None:
    ir = _lower_shared_new(
        [
            _ordering_goal(_existing_binding("subject", "A"), _new_binding("anchor", "x1", "Review")),
            _ordering_goal(_existing_binding("subject", "B"), _new_binding("anchor", "x2", "Review")),
        ]
    )

    assert sorted((row["ref"], row["label"]) for row in ir["entities"]) == [
        ("new_x1", "Review"),
        ("new_x2", "Review"),
    ]
    assert _compiled_relations(ir) == {("A", "new_x1", ""), ("B", "new_x2", "")}


def test_unresolved_ordering_anchor_still_safe_stops() -> None:
    grounded = {
        "status": "resolved",
        "semantic_goals": [
            {
                "kind": "ordering",
                "bindings": [
                    {"role": "subject", "binding_type": "node", "id": "a"},
                    {"role": "anchor", "type": "unresolved_reference"},
                ],
                "relation": "before",
            }
        ],
        "grounding": {"issues": []},
        "clarification": {"needed": False, "question": "", "reason": ""},
    }

    grounded_validation = validate_grounded_goal(grounded)

    kinds = {issue["kind"] for issue in grounded_validation["issues"]}
    assert "missing_ordering_anchor" in kinds
    assert grounded_validation["lowering_ready"] is False


def test_contiguous_ordering_shadow_preserves_failure_branch_metadata(monkeypatch) -> None:
    monkeypatch.setattr(
        semantic_shadow,
        "_complete_json",
        lambda _system, _payload: SimpleNamespace(
            raw_response="",
            parsed_json=v2_ordering_goal([("a", "b"), ("b", "c")]),
            parse_error=None,
            normalized_from_fence=False,
        ),
    )

    trace = semantic_shadow.run_semantic_shadow(chain_graph(), "Put A before B, then continue to C.")

    assert trace["stage"] == "success"
    items = trace["compiler_input"]["workflow_ir"]["control_flow_regions"][0]["structure"]["items"]
    assert [item["id"] for item in items] == ["a", "b", "c"]
    final_edges = edge_set(trace["final_shadow_graph"])
    assert ("start", "a", "") in final_edges
    assert ("a", "b", "") in final_edges
    assert ("b", "c", "") in final_edges
    assert ("b", "a", "") not in final_edges
    assert ("c", "a", "") not in final_edges
    failure_edge = next(edge for edge in trace["final_shadow_graph"]["edges"] if edge.get("id") == "e_a_reject")
    assert failure_edge["source"] == "a"
    assert failure_edge["target"] == "reject"
    assert failure_edge["label"] == "failure"
    assert failure_edge["branch_label"] == "No stock"


def test_contiguous_ordering_does_not_commit_incomplete_boundary(monkeypatch) -> None:
    monkeypatch.setattr(
        semantic_shadow,
        "_complete_json",
        lambda _system, _payload: SimpleNamespace(
            raw_response="",
            parsed_json=v2_ordering_goal([("a", "b"), ("b", "c")]),
            parse_error=None,
            normalized_from_fence=False,
        ),
    )

    graph = chain_graph()
    graph["nodes"].append({"id": "other", "type": "task", "label": "Other"})
    graph["edges"].append({"id": "e_other_b", "source": "other", "target": "b", "label": ""})
    result = semantic_shadow.run_semantic_transaction(graph, "Put A before B, then continue to C.")

    assert result.status == "unsupported"
    assert result.final_graph is None
    issues = result.specification["trace"]["failure_reason"]
    assert issues[0]["kind"] == "ambiguous_structural_boundary"
    assert "incoming boundary" in issues[0]["message"]


def test_business_short_reorder_swaps_unique_local_chain(monkeypatch) -> None:
    trace = run_business_reorder(monkeypatch, complete=False, relation_groups=[])

    assert trace["managed_relation_runtime"]["relation_facts_attached"] is False
    assert_business_reorder_graph(trace)


def test_business_complete_reorder_without_group_matches_short_path(monkeypatch) -> None:
    trace = run_business_reorder(monkeypatch, complete=True, relation_groups=[])

    assert trace["managed_relation_runtime"]["relation_facts_attached"] is False
    items = trace["compiler_input"]["workflow_ir"]["control_flow_regions"][0]["structure"]["items"]
    assert [item["id"] for item in items] == ["n_submit", "n_tool", "n_retrieval", "n_decision"]
    assert_business_reorder_graph(trace)


def test_business_complete_reorder_with_group_matches_ungrouped_path(monkeypatch) -> None:
    trace = run_business_reorder(
        monkeypatch,
        complete=True,
        relation_groups=[
            {
                "kind": "precedence",
                "members": [
                    {"source": "r1", "target": "r2"},
                    {"source": "r3", "target": "r1"},
                    {"source": "r2", "target": "r4"},
                ],
            }
        ],
    )

    assert trace["managed_relation_runtime"]["relation_facts_attached"] is True
    regions = trace["compiler_input"]["workflow_ir"]["control_flow_regions"]
    assert len(regions) == 1
    assert regions[0]["lowering_scope_id"].startswith("relation_fact:")
    items = regions[0]["structure"]["items"]
    assert [item["id"] for item in items] == ["n_submit", "n_tool", "n_retrieval", "n_decision"]
    assert_business_reorder_graph(trace)


def test_structural_boundary_stop_is_not_empty_clarification(monkeypatch) -> None:
    monkeypatch.setattr(
        semantic_shadow,
        "_complete_json",
        lambda _system, _payload: SimpleNamespace(
            raw_response="",
            parsed_json=v2_ordering_goal([("a", "b"), ("b", "c")]),
            parse_error=None,
            normalized_from_fence=False,
        ),
    )
    graph = chain_graph()
    graph["nodes"].append({"id": "other", "type": "task", "label": "Other"})
    graph["edges"].append({"id": "e_other_b", "source": "other", "target": "b", "label": ""})

    result = semantic_shadow.run_semantic_transaction(graph, "Put A before B, then continue to C.")

    assert result.status == "unsupported"
    assert result.final_graph is None
    assert result.specification["trace"]["failure_reason"][0]["kind"] == "ambiguous_structural_boundary"


def submit_graph(ambiguous: bool = True) -> dict:
    nodes = [
        {"id": "start", "type": "start", "label": "Start", "x": 0, "y": 0},
        {"id": "submit_order", "type": "task", "label": "\u63d0\u4ea4\u8ba2\u5355", "x": 220, "y": 0},
        {"id": "done", "type": "end", "label": "Done", "x": 440, "y": 0},
    ]
    edges = [
        {"id": "e_start_submit", "source": "start", "target": "submit_order", "label": ""},
        {"id": "e_submit_done", "source": "submit_order", "target": "done", "label": ""},
    ]
    if ambiguous:
        nodes.append({"id": "submit_approval", "type": "task", "label": "\u63d0\u4ea4\u5ba1\u6279", "x": 220, "y": 120})
    return {"nodes": nodes, "edges": edges}


def submit_ordering_goal(anchor_text: str = "\u63d0\u4ea4") -> dict:
    return {
        "status": "resolved",
        "semantic_goals": [
            {
                "kind": "ordering",
                "text": "\u5728\u63d0\u4ea4\u524d\u589e\u52a0\u5ba1\u6838",
                "subject": {"type": "new_entity", "label": "\u5ba1\u6838"},
                "anchor": {"type": "existing_reference", "text": anchor_text},
                "relation": "before",
                "explicit_constraints": [],
            }
        ],
        "clarification": {"needed": False, "question": "", "reason": ""},
    }


def test_ordering_before_new_entity_lowers_to_workflow_ir_sequence() -> None:
    grounded = ground_semantic_goal_spec(ordering_goal(), shipping_graph(), {"node_ids": [], "edge_ids": []})

    lowered = lower_grounded_goal_to_workflow_ir(grounded, shipping_graph())

    assert lowered["ok"] is True
    workflow_ir = lowered["semantic_spec"]["workflow_ir"]
    assert workflow_ir["properties"] == []
    assert workflow_ir["auxiliary_relations"] == []
    assert workflow_ir["edge_properties"] == []
    assert workflow_ir["absent_entities"] == []
    entity = workflow_ir["entities"][0]
    assert isinstance(entity["x"], (int, float))
    assert isinstance(entity["y"], (int, float))
    created_ref = entity["ref"]
    entity.pop("x")
    entity.pop("y")
    assert workflow_ir["control_flow_regions"] == [
        {
            "relation_type": "",
            "structure": {
                "type": "sequence",
                "items": [
                    {"type": "node", "ref": created_ref},
                    {"type": "node", "id": "shipping"},
                ],
            },
        }
    ]


def test_semantic_shadow_insert_shipping_confirm_runs_existing_backend(monkeypatch) -> None:
    monkeypatch.setattr(
        semantic_shadow,
        "_complete_json",
        lambda _system, _payload: SimpleNamespace(raw_response="", parsed_json=ordering_goal(), parse_error=None, normalized_from_fence=False),
    )

    spec = ordering_goal()
    goal = spec["semantic_goals"][0]
    source = f"{goal['subject']['label']} {goal['anchor']['text']}"
    trace = semantic_shadow.run_semantic_shadow(shipping_graph(), source)

    assert trace["stage"] == "success"
    workflow_ir = trace["compiler_input"]["workflow_ir"]
    entity = workflow_ir["entities"][0]
    assert isinstance(entity["x"], (int, float))
    assert isinstance(entity["y"], (int, float))
    created_ref = entity["ref"]
    entity.pop("x")
    entity.pop("y")
    assert workflow_ir["control_flow_regions"][0]["structure"]["items"] == [
        {"type": "node", "ref": created_ref},
        {"type": "node", "id": "shipping"},
    ]
    new_node = next(node for node in trace["final_shadow_graph"]["nodes"] if node.get("label") == "鍑哄簱纭")
    final_edges = edge_set(trace["final_shadow_graph"])
    assert ("order", new_node["id"], "") in final_edges
    assert (new_node["id"], "shipping", "") in final_edges
    assert ("order", "shipping", "") not in final_edges
    assert ("shipping", "done", "") in final_edges


def test_ordering_existing_reference_ambiguity_returns_candidates() -> None:
    graph = shipping_graph()
    graph["nodes"].append({"id": "shipping_backup", "type": "task", "label": "鍙戣揣", "x": 220, "y": 100})

    grounded = ground_semantic_goal_spec(ordering_goal(), graph, {"node_ids": [], "edge_ids": []})

    issues = grounded["grounding"]["issues"]
    assert issues[0]["kind"] == "ambiguous_node_reference"
    assert issues[0]["candidates"] == ["shipping", "shipping_backup"]


def test_ordering_new_entity_does_not_enter_node_matching_even_when_label_exists() -> None:
    graph = shipping_graph()
    graph["nodes"].append({"id": "existing_confirm", "type": "task", "label": "鍑哄簱纭", "x": 100, "y": 100})

    grounded = ground_semantic_goal_spec(ordering_goal(), graph, {"node_ids": [], "edge_ids": []})

    assert grounded["grounding"]["issues"] == []
    subject = next(binding for binding in grounded["semantic_goals"][0]["bindings"] if binding["role"] == "subject")
    assert subject == {"type": "new_entity", "label": "鍑哄簱纭", "role": "subject", "needs_creation": True}
    assert all(binding.get("id") != "existing_confirm" for binding in grounded["semantic_goals"][0]["bindings"])


def test_existing_join_insert_after_merge_lowers_with_preserved_join_context() -> None:
    grounded = ground_semantic_goal_spec(existing_join_goal(), existing_join_graph(), {"node_ids": [], "edge_ids": []})

    lowered = lower_grounded_goal_to_workflow_ir(grounded, existing_join_graph())

    assert lowered["ok"] is True
    workflow_ir = lowered["semantic_spec"]["workflow_ir"]
    assert workflow_ir["properties"] == []
    assert workflow_ir["auxiliary_relations"] == []
    assert workflow_ir["edge_properties"] == []
    assert workflow_ir["absent_entities"] == []
    entity = workflow_ir["entities"][0]
    assert isinstance(entity["x"], (int, float))
    assert isinstance(entity["y"], (int, float))
    created_ref = entity["ref"]
    entity.pop("x")
    entity.pop("y")
    assert workflow_ir["control_flow_regions"] == [
        {
            "relation_type": "",
            "structure": {
                "type": "sequence",
                "items": [
                    {
                        "type": "parallel",
                        "join_mode": "all",
                        "branches": [{"type": "node", "id": "audit"}, {"type": "node", "id": "risk"}],
                    },
                    {"type": "node", "id": "merge"},
                    {"type": "node", "ref": created_ref},
                ],
            },
        }
    ]


def test_semantic_shadow_existing_join_insert_after_merge_runs_existing_backend(monkeypatch) -> None:
    monkeypatch.setattr(
        semantic_shadow,
        "_complete_json",
        lambda _system, _payload: SimpleNamespace(raw_response="", parsed_json=existing_join_goal(), parse_error=None, normalized_from_fence=False),
    )

    spec = existing_join_goal()
    goal = spec["semantic_goals"][0]
    source = f"{goal['anchor']['text']} {goal['subject']['label']}"
    trace = semantic_shadow.run_semantic_shadow(existing_join_graph(), source)

    assert trace["stage"] == "success"
    grounding = trace["grounded_goal"]["grounding"]
    assert grounding["issues"] == []
    merge_provenance = next(row for row in grounding["provenance"] if row["resolved_id"] == "merge")
    assert merge_provenance["graph_facts"]["is_join_anchor"] is True
    assert trace["semantic_goal"]["normalized"]["semantic_goals"][0]["explicit_constraints"] == ["preserve_existing_join"]
    workflow_ir = trace["compiler_input"]["workflow_ir"]
    assert workflow_ir["control_flow_regions"][0]["structure"]["items"][0] == {
        "type": "parallel",
        "join_mode": "all",
        "branches": [{"type": "node", "id": "audit"}, {"type": "node", "id": "risk"}],
    }

    new_node = next(node for node in trace["final_shadow_graph"]["nodes"] if node.get("label") == "閫氱煡鑺傜偣")
    final_edges = edge_set(trace["final_shadow_graph"])
    assert ("audit", "merge", "") in final_edges
    assert ("risk", "merge", "") in final_edges
    assert ("merge", new_node["id"], "") in final_edges
    assert (new_node["id"], "done", "") in final_edges
    assert ("merge", "done", "") not in final_edges
    merge_node = next(node for node in trace["final_shadow_graph"]["nodes"] if node["id"] == "merge")
    assert merge_node["join"] == {"mode": "all"}


def test_semantic_shadow_attaches_managed_relation_group_before_lowering(monkeypatch) -> None:
    spec = existing_join_grouped_insert_goal()
    goal = spec["semantic_goals"][0]
    source = " ".join(
        [
            goal["objects"][0]["text"],
            goal["objects"][1]["text"],
            goal["objects"][2]["label"],
            goal["objects"][3]["text"],
        ]
    )

    def complete(system: str, _payload: dict) -> SimpleNamespace:
        if "Relation Grouper" in system:
            return SimpleNamespace(
                raw_response="",
                parsed_json={
                    "relation_groups": [
                        {
                            "kind": "precedence",
                            "members": [
                                {"source": "r1", "target": "r3"},
                                {"source": "r2", "target": "r3"},
                            ],
                        }
                    ]
                },
                parse_error=None,
                normalized_from_fence=False,
            )
        return SimpleNamespace(raw_response="", parsed_json=spec, parse_error=None, normalized_from_fence=False)

    monkeypatch.setattr(semantic_shadow, "_complete_json", complete)

    trace = semantic_shadow.run_semantic_shadow(existing_join_graph(), source)

    assert trace["stage"] == "success"
    runtime = trace["managed_relation_runtime"]
    assert runtime["relation_facts_attached"] is True
    assert runtime["after_goal_state"]["relation_facts"][0]["relation_id"] == "rg1"
    assert len(runtime["suppressed_legacy_relation_goals"]) == 2

    workflow_ir = trace["compiler_input"]["workflow_ir"]
    scoped_regions = [row for row in workflow_ir["control_flow_regions"] if row.get("lowering_scope_id") == "relation_fact:rg1"]
    assert len(scoped_regions) == 2
    assert workflow_ir["entities"][0]["ref"] == "new_confirm_obj"

    new_node = next(node for node in trace["final_shadow_graph"]["nodes"] if node.get("label") == goal["objects"][2]["label"])
    final_edges = edge_set(trace["final_shadow_graph"])
    assert ("audit", "merge", "") in final_edges
    assert ("risk", "merge", "") in final_edges
    assert ("merge", new_node["id"], "") in final_edges
    assert (new_node["id"], "done", "") in final_edges
    assert ("audit", new_node["id"], "") not in final_edges
    assert ("risk", new_node["id"], "") not in final_edges
    assert ("merge", "done", "") not in final_edges


def test_semantic_shadow_without_relation_group_keeps_legacy_ordering_path(monkeypatch) -> None:
    spec = existing_join_grouped_insert_goal()
    goal = spec["semantic_goals"][0]
    source = " ".join(
        [
            goal["objects"][0]["text"],
            goal["objects"][1]["text"],
            goal["objects"][2]["label"],
            goal["objects"][3]["text"],
        ]
    )

    def complete(system: str, _payload: dict) -> SimpleNamespace:
        if "Relation Grouper" in system:
            return SimpleNamespace(raw_response="", parsed_json={"relation_groups": []}, parse_error=None, normalized_from_fence=False)
        return SimpleNamespace(raw_response="", parsed_json=spec, parse_error=None, normalized_from_fence=False)

    monkeypatch.setattr(semantic_shadow, "_complete_json", complete)

    trace = semantic_shadow.run_semantic_shadow(existing_join_graph(), source)

    assert trace["managed_relation_runtime"]["relation_facts_attached"] is False
    workflow_ir = trace["compiler_input"]["workflow_ir"]
    assert all("lowering_scope_id" not in row for row in workflow_ir["control_flow_regions"])


def report_human_publish_graph() -> dict:
    return {
        "nodes": [
            {"id": "report", "type": "task", "label": "Report"},
            {"id": "human", "type": "task", "label": "Human Review"},
            {"id": "publish", "type": "task", "label": "Publish"},
        ],
        "edges": [
            {"id": "e_report_publish", "source": "report", "target": "publish", "label": ""},
            {"id": "e_report_human_failure", "source": "report", "target": "human", "label": "failure", "branch_label": "failure"},
            {"id": "e_human_publish", "source": "human", "target": "publish", "label": ""},
        ],
    }


def insert_confirm_preserve_paths_goal() -> dict:
    return {
        "status": "resolved",
        "semantic_goals_v2": [
            {
                "id": "g1",
                "goal_category": "composite_goal",
                "text": "Insert Confirm after Human Review before Publish and preserve existing paths.",
                "objects": [
                    {"id": "human", "type": "existing_reference", "reference": "Human Review"},
                    {"id": "confirm", "type": "new_entity", "label": "Confirm"},
                    {"id": "publish", "type": "existing_reference", "reference": "Publish"},
                    {"id": "report_publish", "type": "edge_reference", "reference": "Report Publish"},
                    {"id": "report_human_failure", "type": "edge_reference", "reference": "failure Report Human Review"},
                ],
                "object_desired_states": [
                    {"object": "confirm", "state": "exists"},
                    {"object": "report_publish", "state": "preserved"},
                    {"object": "report_human_failure", "state": "preserved"},
                ],
                "relation_intents": [
                    {"source": "human", "relation": "before", "target": "confirm"},
                    {"source": "confirm", "relation": "before", "target": "publish"},
                ],
                "semantic_constraints": [
                    {"type": "preserve", "target": "the rest of the workflow", "text": "preserve all unrelated paths"}
                ],
            }
        ],
        "clarification": {"needed": False, "question": "", "reason": ""},
    }


def _run_insert_confirm(monkeypatch, relation_groups: list[dict]) -> dict:
    spec = insert_confirm_preserve_paths_goal()

    def complete(system: str, _payload: dict) -> SimpleNamespace:
        if "Relation Grouper" in system:
            return SimpleNamespace(raw_response="", parsed_json={"relation_groups": relation_groups}, parse_error=None, normalized_from_fence=False)
        return SimpleNamespace(raw_response="", parsed_json=spec, parse_error=None, normalized_from_fence=False)

    monkeypatch.setattr(semantic_shadow, "_complete_json", complete)
    return semantic_shadow.run_semantic_shadow(
        report_human_publish_graph(),
        "Insert Confirm after Human Review before Publish; preserve Report Publish and failure Report Human Review.",
    )


def _assert_insert_confirm_final_graph(trace: dict) -> None:
    assert trace["stage"] == "success"
    final_edges = edge_set(trace["final_shadow_graph"])
    confirm = next(node for node in trace["final_shadow_graph"]["nodes"] if node.get("label") == "Confirm")
    assert ("report", "publish", "") in final_edges
    assert ("report", "human", "failure") in final_edges
    assert ("human", confirm["id"], "") in final_edges
    assert (confirm["id"], "publish", "") in final_edges
    assert ("human", "publish", "") not in final_edges
    assert ("report", "human", "") not in final_edges
    failure_edge = next(edge for edge in trace["final_shadow_graph"]["edges"] if edge.get("id") == "e_report_human_failure")
    assert failure_edge["branch_label"] == "failure"


def test_semantic_shadow_insert_confirm_preserve_paths_with_relation_group(monkeypatch) -> None:
    trace = _run_insert_confirm(
        monkeypatch,
        [
            {
                "kind": "precedence",
                "members": [
                    {"source": "r1", "target": "r2"},
                    {"source": "r2", "target": "r3"},
                ],
            }
        ],
    )

    assert trace["managed_relation_runtime"]["relation_facts_attached"] is True
    assert trace["grounded_spec"]["workflow_ir"].get("preserve", {}).get("abstract")
    assert "e_report_publish" in trace["constraints"]["preserve"]["edge_ids"]
    assert "e_report_human_failure" in trace["constraints"]["preserve"]["edge_ids"]
    _assert_insert_confirm_final_graph(trace)


def test_semantic_shadow_insert_confirm_preserve_paths_without_relation_group(monkeypatch) -> None:
    trace = _run_insert_confirm(monkeypatch, [])

    assert trace["managed_relation_runtime"]["relation_facts_attached"] is False
    assert all("lowering_scope_id" not in row for row in trace["compiler_input"]["workflow_ir"]["control_flow_regions"])
    assert trace["grounded_spec"]["workflow_ir"].get("preserve", {}).get("abstract")
    assert "e_report_publish" in trace["constraints"]["preserve"]["edge_ids"]
    assert "e_report_human_failure" in trace["constraints"]["preserve"]["edge_ids"]
    _assert_insert_confirm_final_graph(trace)
    preserved = {
        row["id"]: row
        for row in trace["partial_execution"]["subgoals"]
        if row.get("type") == "object_desired_state" and row.get("state") == "preserved"
    }
    assert preserved["g1:state:1"]["status"] == "executed"
    assert preserved["g1:state:2"]["status"] == "executed"


def test_existing_join_merge_reference_ambiguity_needs_clarification(monkeypatch) -> None:
    graph = existing_join_graph()
    graph["nodes"].append({"id": "merge_backup", "type": "task", "label": "鍚堝苟", "x": 220, "y": 180})
    monkeypatch.setattr(
        semantic_shadow,
        "_complete_json",
        lambda _system, _payload: SimpleNamespace(raw_response="", parsed_json=existing_join_goal(), parse_error=None, normalized_from_fence=False),
    )

    spec = existing_join_goal()
    goal = spec["semantic_goals"][0]
    source = f"{goal['anchor']['text']} {goal['subject']['label']}"
    trace = semantic_shadow.run_semantic_shadow(graph, source)

    assert trace["failure_stage"] == "validation"
    assert trace["validation"]["issues"][0]["kind"] == "ambiguous_node_reference"
    assert trace["validation"]["issues"][0]["candidates"] == ["merge", "merge_backup"]


def test_insert_before_submit_ambiguous_preserves_user_referent_and_stops_before_lowering(monkeypatch) -> None:
    monkeypatch.setattr(
        semantic_shadow,
        "_complete_json",
        lambda _system, _payload: SimpleNamespace(raw_response="", parsed_json=submit_ordering_goal(), parse_error=None, normalized_from_fence=False),
    )

    trace = semantic_shadow.run_semantic_shadow(submit_graph(ambiguous=True), "\u5728\u63d0\u4ea4\u524d\u589e\u52a0\u5ba1\u6838")

    goal = trace["semantic_goal"]["normalized"]["semantic_goals"][0]
    anchor = next(item for item in goal["objects"] if item["role"] == "anchor")
    assert anchor["reference"] == "\u63d0\u4ea4"
    assert trace["failure_stage"] == "validation"
    assert trace["stage"] == "goal_ground"
    assert trace["lowering_result"] is None
    assert trace["compiler_input"] is None
    assert trace["planner"] is None
    assert trace["executor"] is None
    assert trace["validation"]["issues"][0]["kind"] == "ambiguous_node_reference"
    assert trace["validation"]["issues"][0]["candidates"] == ["submit_approval", "submit_order"]
    result = semantic_shadow.semantic_trace_to_transaction_result(trace)
    assert result.status == "needs_clarification"


def test_insert_before_submit_unique_short_referent_enters_ordering_lowering(monkeypatch) -> None:
    monkeypatch.setattr(
        semantic_shadow,
        "_complete_json",
        lambda _system, _payload: SimpleNamespace(raw_response="", parsed_json=submit_ordering_goal(), parse_error=None, normalized_from_fence=False),
    )

    trace = semantic_shadow.run_semantic_shadow(submit_graph(ambiguous=False), "\u5728\u63d0\u4ea4\u524d\u589e\u52a0\u5ba1\u6838")

    assert trace["stage"] == "success"
    provenance = trace["grounded_goal"]["grounding"]["provenance"]
    assert any(row["reference"] == "\u63d0\u4ea4" and row["resolved_id"] == "submit_order" for row in provenance)
    workflow_ir = trace["compiler_input"]["workflow_ir"]
    items = workflow_ir["control_flow_regions"][0]["structure"]["items"]
    entity = workflow_ir["entities"][0]
    assert isinstance(entity["x"], (int, float))
    assert isinstance(entity["y"], (int, float))
    created_ref = entity["ref"]
    assert items == [
        {"type": "node", "ref": created_ref},
        {"type": "node", "id": "submit_order"},
    ]


def test_semantic_understanding_cannot_expand_submit_to_specific_graph_label(monkeypatch) -> None:
    expanded = submit_ordering_goal("Submit Order")
    monkeypatch.setattr(
        semantic_shadow,
        "_complete_json",
        lambda _system, _payload: SimpleNamespace(raw_response="", parsed_json=expanded, parse_error=None, normalized_from_fence=False),
    )

    trace = semantic_shadow.run_semantic_shadow(submit_graph(ambiguous=True), "\u5728\u63d0\u4ea4\u524d\u589e\u52a0\u5ba1\u6838")

    assert trace["failure_stage"] == "validation"
    assert trace["validation"]["issues"][0]["kind"] == "semantic_goal_contract_violation"
    assert trace["grounded_goal"] is None
    validation = validate_semantic_goal_spec(expanded, source_instruction="\u5728\u63d0\u4ea4\u524d\u589e\u52a0\u5ba1\u6838")
    assert validation["ready_for_grounding"] is False
