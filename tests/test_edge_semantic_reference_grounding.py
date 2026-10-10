from __future__ import annotations

from types import SimpleNamespace

from src import semantic_shadow
from src.semantic_goal import ground_semantic_goal_spec


def graph_with_edges(edges: list[dict]) -> dict:
    return {
        "nodes": [
            {"id": "decision", "type": "decision", "label": "\u5224\u65ad"},
            {"id": "success", "type": "task", "label": "\u6210\u529f\u5904\u7406"},
            {"id": "manual", "type": "task", "label": "\u4eba\u5de5\u5904\u7406"},
            {"id": "payment", "type": "task", "label": "\u652f\u4ed8"},
            {"id": "shipping", "type": "task", "label": "\u53d1\u8d27"},
        ],
        "edges": edges,
    }


def edge_goal(reference: str) -> dict:
    return {
        "status": "resolved",
        "semantic_goals": [
            {
                "goal_category": "relation_goal",
                "text": "Route referenced edge to manual handling.",
                "objects": [
                    {"id": "route", "type": "edge_reference", "text": reference, "role": "route"},
                    {"id": "target", "type": "existing_reference", "text": "\u4eba\u5de5\u5904\u7406", "role": "target"},
                ],
                "relation_intents": [{"source": "route", "relation": "route_to", "target": "target"}],
            }
        ],
        "clarification": {"needed": False, "question": "", "reason": ""},
    }


def test_edge_branch_label_reference_binds() -> None:
    graph = graph_with_edges([
        {"id": "edge_failure", "source": "decision", "target": "manual", "branch_label": "\u5931\u8d25"},
    ])

    grounded = ground_semantic_goal_spec(edge_goal("\u5931\u8d25\u60c5\u51b5"), graph, {"edge_ids": []})

    assert grounded["grounding"]["issues"] == []
    binding = grounded["semantic_goals"][0]["bindings"][0]
    assert binding["binding_type"] == "edge"
    assert binding["id"] == "edge_failure"


def test_edge_condition_expression_reference_binds() -> None:
    graph = graph_with_edges([
        {"id": "edge_exception", "source": "decision", "target": "manual", "condition": {"expression": "status == exception"}},
    ])

    grounded = ground_semantic_goal_spec(edge_goal("\u5f02\u5e38\u6d41\u7a0b"), graph, {"edge_ids": []})

    assert grounded["grounding"]["issues"] == []
    binding = grounded["semantic_goals"][0]["bindings"][0]
    assert binding["binding_type"] == "edge"
    assert binding["id"] == "edge_exception"


def test_edge_reference_multiple_candidates_stops_for_clarification() -> None:
    graph = graph_with_edges([
        {"id": "edge_branch_failure", "source": "decision", "target": "manual", "branch_label": "\u5931\u8d25"},
        {"id": "edge_condition_failure", "source": "payment", "target": "manual", "condition": {"expression": "status failed"}},
    ])

    grounded = ground_semantic_goal_spec(edge_goal("\u5931\u8d25\u60c5\u51b5"), graph, {"edge_ids": []})

    issues = grounded["grounding"]["issues"]
    assert issues[0]["kind"] == "ambiguous_edge_reference"
    assert issues[0]["candidates"] == ["edge_branch_failure", "edge_condition_failure"]


def test_edge_selection_priority_skips_semantic_candidates() -> None:
    graph = graph_with_edges([
        {"id": "edge_selected", "source": "decision", "target": "manual", "label": "success"},
        {"id": "edge_failure", "source": "payment", "target": "manual", "branch_label": "\u5931\u8d25"},
    ])

    grounded = ground_semantic_goal_spec(
        edge_goal("\u8fd9\u4e2a\u5206\u652f"),
        graph,
        {"selected_edge_ids": ["edge_selected"]},
    )

    assert grounded["grounding"]["issues"] == []
    binding = grounded["semantic_goals"][0]["bindings"][0]
    assert binding["id"] == "edge_selected"
    provenance = grounded["grounding"]["provenance"][0]
    assert provenance["resolution_type"] == "selection"


def test_edge_reference_falls_back_to_unique_selected_edge_ids_when_text_does_not_match() -> None:
    graph = graph_with_edges([
        {"id": "edge_selected", "source": "decision", "target": "success", "label": "success"},
        {"id": "edge_failure", "source": "decision", "target": "manual", "label": "failure"},
    ])

    grounded = ground_semantic_goal_spec(
        edge_goal("\u8fd9\u6761\u8fde\u7ebf"),
        graph,
        {"node_ids": [], "edge_ids": ["edge_selected"]},
    )

    assert grounded["grounding"]["issues"] == []
    binding = grounded["semantic_goals"][0]["bindings"][0]
    assert binding["binding_type"] == "edge"
    assert binding["id"] == "edge_selected"


def test_edge_reference_multiple_selected_edges_stops_for_clarification() -> None:
    graph = graph_with_edges([
        {"id": "edge_success", "source": "decision", "target": "success", "label": "success"},
        {"id": "edge_failure", "source": "decision", "target": "manual", "label": "failure"},
    ])

    grounded = ground_semantic_goal_spec(
        edge_goal("\u8fd9\u6761\u8fde\u7ebf"),
        graph,
        {"node_ids": [], "edge_ids": ["edge_success", "edge_failure"]},
    )

    issues = grounded["grounding"]["issues"]
    assert issues[0]["kind"] == "ambiguous_edge_reference"
    assert issues[0]["candidates"] == ["edge_failure", "edge_success"]


def test_edge_reference_low_confidence_missing() -> None:
    graph = graph_with_edges([
        {"id": "edge_success", "source": "decision", "target": "success", "label": "success"},
        {"id": "edge_payment", "source": "payment", "target": "shipping", "label": "payment"},
    ])

    grounded = ground_semantic_goal_spec(edge_goal("\u5f02\u5e38\u6d41\u7a0b"), graph, {"edge_ids": []})

    issues = grounded["grounding"]["issues"]
    assert issues[0]["kind"] == "missing_edge_reference"


def review_publish_graph() -> dict:
    return {
        "nodes": [
            {"id": "n_decision", "type": "decision", "label": "结果判断"},
            {"id": "n_report", "type": "task", "label": "生成报告"},
            {"id": "n_human", "type": "task", "label": "人工复核"},
            {"id": "n_publish", "type": "task", "label": "发布结果"},
        ],
        "edges": [
            {"id": "e_decision_human", "source": "n_decision", "target": "n_human", "label": "failure", "branch_label": "failure"},
            {"id": "e_report_human", "source": "n_report", "target": "n_human", "label": "failure", "branch_label": "failure"},
            {"id": "e_report_publish", "source": "n_report", "target": "n_publish", "label": ""},
            {"id": "e_decision_report", "source": "n_decision", "target": "n_report", "label": "success", "branch_label": "success"},
            {"id": "e_human_publish", "source": "n_human", "target": "n_publish", "label": ""},
        ],
    }


def preserve_edge_goal(reference: str) -> dict:
    return {
        "status": "resolved",
        "semantic_goals": [
            {
                "id": "g1",
                "goal_category": "state_goal",
                "text": reference,
                "objects": [{"id": "edge", "type": "edge_reference", "text": reference, "role": "object"}],
                "object_desired_states": [{"object": "edge", "state": "preserved"}],
                "relation_intents": [],
            }
        ],
        "clarification": {"needed": False, "question": "", "reason": ""},
    }


def bound_edge_id(reference: str, graph: dict | None = None) -> str:
    grounded = ground_semantic_goal_spec(preserve_edge_goal(reference), graph or review_publish_graph(), {"node_ids": [], "edge_ids": []})
    assert grounded["grounding"]["issues"] == []
    return grounded["semantic_goals"][0]["bindings"][0]["id"]


def test_real_failure_branch_reference_uses_explicit_source_node() -> None:
    assert bound_edge_id("结果判断的失败分支") == "e_decision_human"
    assert bound_edge_id("生成报告的失败分支") == "e_report_human"


def test_direct_path_reference_prefers_explicit_source_and_target_over_path_context_label() -> None:
    assert bound_edge_id("生成报告成功后直接进入发布结果的路径") == "e_report_publish"


def test_same_label_same_target_explicit_source_disambiguates_edge() -> None:
    graph = review_publish_graph()
    assert bound_edge_id("生成报告 failure branch", graph) == "e_report_human"


def test_same_source_different_target_explicit_target_disambiguates_edge() -> None:
    graph = review_publish_graph()
    assert bound_edge_id("结果判断到人工复核的失败分支", graph) == "e_decision_human"


def test_endpoint_evidence_overrides_conflicting_label_on_other_edge() -> None:
    graph = review_publish_graph()
    assert bound_edge_id("生成报告 success 发布结果", graph) == "e_report_publish"


def test_relation_only_failure_branch_remains_ambiguous() -> None:
    grounded = ground_semantic_goal_spec(preserve_edge_goal("failure branch"), review_publish_graph(), {"node_ids": [], "edge_ids": []})
    issues = grounded["grounding"]["issues"]
    assert issues[0]["kind"] == "ambiguous_edge_reference"
    assert issues[0]["candidates"] == ["e_decision_human", "e_report_human"]


def test_same_endpoint_different_labels_are_not_endpoint_merged() -> None:
    graph = {
        "nodes": [
            {"id": "a", "type": "task", "label": "A"},
            {"id": "b", "type": "task", "label": "B"},
        ],
        "edges": [
            {"id": "e_ab_success", "source": "a", "target": "b", "label": "success", "branch_label": "success"},
            {"id": "e_ab_failure", "source": "a", "target": "b", "label": "failure", "branch_label": "failure"},
        ],
    }
    grounded = ground_semantic_goal_spec(preserve_edge_goal("A to B"), graph, {"node_ids": [], "edge_ids": []})
    issues = grounded["grounding"]["issues"]
    assert issues[0]["kind"] == "ambiguous_edge_reference"
    assert issues[0]["candidates"] == ["e_ab_failure", "e_ab_success"]


def full_insert_review_confirm_goal() -> dict:
    return {
        "status": "resolved",
        "semantic_goals": [
            {
                "id": "g1",
                "goal_category": "composite_goal",
                "text": REAL_REVIEW_CONFIRM_INSTRUCTION,
                "objects": [
                    {"id": "human", "type": "existing_reference", "text": "人工复核", "role": "object"},
                    {"id": "confirm", "type": "new_entity", "label": "“复核确认”", "role": "object"},
                    {"id": "publish", "type": "existing_reference", "text": "发布结果", "role": "object"},
                    {"id": "decision_failure", "type": "edge_reference", "text": "“结果判断”的失败分支", "role": "object"},
                    {"id": "report_failure", "type": "edge_reference", "text": "“生成报告”的失败分支", "role": "object"},
                    {"id": "report_publish", "type": "edge_reference", "text": "“生成报告”成功后直接进入“发布结果”的路径", "role": "object"},
                ],
                "object_desired_states": [
                    {"object": "confirm", "state": "exists"},
                    {"object": "decision_failure", "state": "preserved"},
                    {"object": "report_failure", "state": "preserved"},
                    {"object": "report_publish", "state": "preserved"},
                ],
                "relation_intents": [
                    {"source": "human", "relation": "before", "target": "confirm"},
                    {"source": "confirm", "relation": "before", "target": "publish"},
                ],
                "semantic_constraints": [
                    {"type": "preserve", "target": "其他业务边", "text": "保留原有的失败分支连接，也不要改变生成报告成功后直接进入发布结果的路径"}
                ],
            }
        ],
        "clarification": {"needed": False, "question": "", "reason": ""},
    }


REAL_REVIEW_CONFIRM_INSTRUCTION = (
    "在“结果判断”的失败分支和“生成报告”的失败分支汇合到“人工复核”之后，"
    "增加一个“复核确认”节点，再进入“发布结果”。"
    "保留原有的失败分支连接，也不要改变“生成报告”成功后直接进入“发布结果”的路径。"
)


def edge_set(graph: dict) -> set[tuple[str, str, str]]:
    return {
        (edge["source"], edge["target"], edge.get("label", ""))
        for edge in graph.get("edges", [])
        if isinstance(edge, dict)
    }


def run_review_confirm_transaction(monkeypatch, relation_groups: list[dict]) -> dict:
    spec = full_insert_review_confirm_goal()

    def complete(system: str, _payload: dict) -> SimpleNamespace:
        if "Relation Grouper" in system:
            return SimpleNamespace(raw_response="", parsed_json={"relation_groups": relation_groups}, parse_error=None, normalized_from_fence=False)
        return SimpleNamespace(raw_response="", parsed_json=spec, parse_error=None, normalized_from_fence=False)

    monkeypatch.setattr(semantic_shadow, "_complete_json", complete)
    return semantic_shadow.run_semantic_shadow(review_publish_graph(), REAL_REVIEW_CONFIRM_INSTRUCTION)


def assert_review_confirm_graph(trace: dict) -> None:
    assert trace["stage"] == "success"
    bindings = {
        binding["v2_object_id"]: binding["id"]
        for goal in trace["grounded_goal"]["semantic_goals"]
        for binding in goal.get("bindings", [])
        if isinstance(binding, dict) and binding.get("binding_type") == "edge"
    }
    assert bindings["decision_failure"] == "e_decision_human"
    assert bindings["report_failure"] == "e_report_human"
    assert bindings["report_publish"] == "e_report_publish"
    preserve_edges = set(trace["constraints"]["preserve"]["edge_ids"])
    assert {"e_decision_human", "e_report_human", "e_report_publish"} <= preserve_edges
    final_edges = edge_set(trace["final_shadow_graph"])
    confirm = next(node for node in trace["final_shadow_graph"]["nodes"] if node.get("label") == "复核确认")
    assert ("n_decision", "n_human", "failure") in final_edges
    assert ("n_report", "n_human", "failure") in final_edges
    assert ("n_report", "n_publish", "") in final_edges
    assert ("n_human", confirm["id"], "") in final_edges
    assert (confirm["id"], "n_publish", "") in final_edges
    assert ("n_human", "n_publish", "") not in final_edges
    assert ("n_report", "n_human", "") not in final_edges


def test_real_review_confirm_transaction_with_group(monkeypatch) -> None:
    trace = run_review_confirm_transaction(
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
    assert_review_confirm_graph(trace)


def test_real_review_confirm_transaction_without_group(monkeypatch) -> None:
    trace = run_review_confirm_transaction(monkeypatch, [])
    assert trace["managed_relation_runtime"]["relation_facts_attached"] is False
    assert_review_confirm_graph(trace)


def review_confirm_v3_graph() -> dict:
    graph = review_publish_graph()
    graph["nodes"].append({"id": "n_confirm", "type": "task", "label": "复核结果确认"})
    graph["nodes"][-1]["label"] = "复核结果确认"
    graph["edges"] = [
        edge
        for edge in graph["edges"]
        if edge.get("id") != "e_human_publish"
    ]
    graph["edges"].extend(
        [
            {"id": "e_human_confirm", "source": "n_human", "target": "n_confirm", "label": ""},
            {"id": "e_confirm_publish", "source": "n_confirm", "target": "n_publish", "label": ""},
        ]
    )
    return graph


DELETE_CONFIRM_RESTORE_DIRECT_INSTRUCTION = (
    "删除“复核结果确认”节点，让“人工复核”重新直接连接“发布结果”。"
    "保留“结果判断”和“生成报告”进入“人工复核”的两条失败分支，"
    "也不要改变“生成报告”直接进入“发布结果”的路径。"
)


def delete_confirm_restore_direct_goal() -> dict:
    return {
        "status": "resolved",
        "semantic_goals": [
            {
                "id": "g1",
                "goal_category": "composite_goal",
                "text": DELETE_CONFIRM_RESTORE_DIRECT_INSTRUCTION,
                "objects": [
                    {"id": "confirm", "type": "existing_reference", "text": "复核结果确认", "role": "object"},
                    {"id": "human", "type": "existing_reference", "text": "人工复核", "role": "source"},
                    {"id": "publish", "type": "existing_reference", "text": "发布结果", "role": "target"},
                    {"id": "decision_failure", "type": "edge_reference", "text": "“结果判断”进入“人工复核”的失败分支", "role": "object"},
                    {"id": "report_failure", "type": "edge_reference", "text": "“生成报告”进入“人工复核”的失败分支", "role": "object"},
                    {"id": "report_publish", "type": "edge_reference", "text": "“生成报告”直接进入“发布结果”的路径", "role": "object"},
                ],
                "object_desired_states": [
                    {"object": "confirm", "state": "absent"},
                    {"object": "decision_failure", "state": "preserved"},
                    {"object": "report_failure", "state": "preserved"},
                    {"object": "report_publish", "state": "preserved"},
                ],
                "relation_intents": [
                    {"source": "human", "relation": "route_to", "target": "publish"},
                ],
                "semantic_constraints": [
                    {"type": "preserve", "target": "其他业务连接", "text": "保留两条失败分支和生成报告直接发布路径"}
                ],
            }
        ],
        "clarification": {"needed": False, "question": "", "reason": ""},
    }


DELETE_CONFIRM_RESTORE_DIRECT_INSTRUCTION_CLEAN = (
    "删除“复核结果确认”节点，让“人工复核”重新直接连接“发布结果”。"
    "保留“结果判断”进入“人工复核”的失败分支，"
    "保留“生成报告”进入“人工复核”的失败分支，"
    "也不要改变“生成报告”直接进入“发布结果”的路径。"
)


def delete_confirm_restore_direct_goal_clean() -> dict:
    return {
        "status": "resolved",
        "semantic_goals": [
            {
                "id": "g1",
                "goal_category": "composite_goal",
                "text": DELETE_CONFIRM_RESTORE_DIRECT_INSTRUCTION_CLEAN,
                "objects": [
                    {"id": "confirm", "type": "existing_reference", "text": "复核结果确认", "role": "object"},
                    {"id": "human", "type": "existing_reference", "text": "人工复核", "role": "source"},
                    {"id": "publish", "type": "existing_reference", "text": "发布结果", "role": "target"},
                    {"id": "decision_failure", "type": "edge_reference", "text": "“结果判断”进入“人工复核”的失败分支", "role": "object"},
                    {"id": "report_failure", "type": "edge_reference", "text": "“生成报告”进入“人工复核”的失败分支", "role": "object"},
                    {"id": "report_publish", "type": "edge_reference", "text": "“生成报告”直接进入“发布结果”的路径", "role": "object"},
                ],
                "object_desired_states": [
                    {"object": "confirm", "state": "absent"},
                    {"object": "decision_failure", "state": "preserved"},
                    {"object": "report_failure", "state": "preserved"},
                    {"object": "report_publish", "state": "preserved"},
                ],
                "relation_intents": [
                    {"source": "human", "relation": "route_to", "target": "publish"},
                ],
                "semantic_constraints": [
                    {"type": "preserve", "target": "其他业务连接", "text": "保留两条失败分支和生成报告直接发布路径"}
                ],
            }
        ],
        "clarification": {"needed": False, "question": "", "reason": ""},
    }


def delete_confirm_restore_direct_goal_split() -> dict:
    return {
        "status": "resolved",
        "semantic_goals": [
            {
                "id": "g1",
                "goal_category": "object_state_goal",
                "text": "删除“复核结果确认”节点。",
                "objects": [
                    {"id": "confirm", "type": "existing_reference", "text": "复核结果确认", "role": "object"},
                ],
                "object_desired_states": [
                    {"object": "confirm", "state": "absent"},
                ],
            },
            {
                "id": "g2",
                "goal_category": "relation_goal",
                "text": "让“人工复核”重新直接连接“发布结果”。",
                "objects": [
                    {"id": "human", "type": "existing_reference", "text": "人工复核", "role": "source"},
                    {"id": "publish", "type": "existing_reference", "text": "发布结果", "role": "target"},
                ],
                "relation_intents": [
                    {"source": "human", "relation": "route_to", "target": "publish"},
                ],
            },
            {
                "id": "g3",
                "goal_category": "constraint_goal",
                "text": "保留原有失败分支和生成报告直接发布路径。",
                "objects": [
                    {"id": "decision_failure", "type": "edge_reference", "text": "“结果判断”进入“人工复核”的失败分支", "role": "object"},
                    {"id": "report_failure", "type": "edge_reference", "text": "“生成报告”进入“人工复核”的失败分支", "role": "object"},
                    {"id": "report_publish", "type": "edge_reference", "text": "“生成报告”直接进入“发布结果”的路径", "role": "object"},
                ],
                "object_desired_states": [
                    {"object": "decision_failure", "state": "preserved"},
                    {"object": "report_failure", "state": "preserved"},
                    {"object": "report_publish", "state": "preserved"},
                ],
            },
        ],
        "clarification": {"needed": False, "question": "", "reason": ""},
    }


def test_delete_node_and_restore_direct_connection_does_not_ground_source_as_edge(monkeypatch) -> None:
    spec = delete_confirm_restore_direct_goal_clean()

    def complete(system: str, _payload: dict) -> SimpleNamespace:
        if "Relation Grouper" in system:
            return SimpleNamespace(raw_response="", parsed_json={"relation_groups": []}, parse_error=None, normalized_from_fence=False)
        return SimpleNamespace(raw_response="", parsed_json=spec, parse_error=None, normalized_from_fence=False)

    monkeypatch.setattr(semantic_shadow, "_complete_json", complete)
    trace = semantic_shadow.run_semantic_shadow(review_confirm_v3_graph(), DELETE_CONFIRM_RESTORE_DIRECT_INSTRUCTION_CLEAN)

    assert trace["stage"] == "success"
    assert trace["grounded_goal"]["grounding"]["issues"] == []
    connection_goal = next(goal for goal in trace["grounded_goal"]["semantic_goals"] if goal.get("kind") == "connection")
    bindings = {binding["role"]: binding for binding in connection_goal["bindings"]}
    assert bindings["source"]["binding_type"] == "node"
    assert bindings["source"]["id"] == "n_human"
    assert bindings["target"]["binding_type"] == "node"
    assert bindings["target"]["id"] == "n_publish"
    final = trace["final_shadow_graph"]
    assert len(final["nodes"]) == 4
    assert len(final["edges"]) == 5
    assert "n_confirm" not in {node["id"] for node in final["nodes"]}
    final_edges = edge_set(final)
    assert ("n_decision", "n_human", "failure") in final_edges
    assert ("n_decision", "n_report", "success") in final_edges
    assert ("n_report", "n_human", "failure") in final_edges
    assert ("n_report", "n_publish", "") in final_edges
    assert ("n_human", "n_publish", "") in final_edges
    assert ("n_human", "n_confirm", "") not in final_edges
    assert ("n_confirm", "n_publish", "") not in final_edges


def test_split_goals_delete_node_and_restore_direct_connection(monkeypatch) -> None:
    spec = delete_confirm_restore_direct_goal_split()

    def complete(system: str, _payload: dict) -> SimpleNamespace:
        if "Relation Grouper" in system:
            return SimpleNamespace(raw_response="", parsed_json={"relation_groups": []}, parse_error=None, normalized_from_fence=False)
        return SimpleNamespace(raw_response="", parsed_json=spec, parse_error=None, normalized_from_fence=False)

    monkeypatch.setattr(semantic_shadow, "_complete_json", complete)
    trace = semantic_shadow.run_semantic_shadow(review_confirm_v3_graph(), DELETE_CONFIRM_RESTORE_DIRECT_INSTRUCTION_CLEAN)

    assert trace["stage"] == "success"
    connection_goal = next(goal for goal in trace["grounded_goal"]["semantic_goals"] if goal.get("kind") == "connection")
    assert connection_goal["relation"] == "connected_to"
    final_edges = edge_set(trace["final_shadow_graph"])
    assert "n_confirm" not in {node["id"] for node in trace["final_shadow_graph"]["nodes"]}
    assert ("n_human", "n_publish", "") in final_edges
    assert ("n_report", "n_publish", "") in final_edges
    assert ("n_decision", "n_human", "failure") in final_edges
    assert ("n_report", "n_human", "failure") in final_edges


def test_delete_unrelated_node_does_not_convert_failure_route(monkeypatch) -> None:
    instruction = "删除“复核结果确认”节点，把“结果判断”的失败分支改到“发布结果”。"
    spec = {
        "status": "resolved",
        "semantic_goals": [
            {
                "id": "g1",
                "goal_category": "object_state_goal",
                "text": "删除“复核结果确认”节点。",
                "objects": [{"id": "confirm", "type": "existing_reference", "text": "复核结果确认", "role": "object"}],
                "object_desired_states": [{"object": "confirm", "state": "absent"}],
            },
            {
                "id": "g2",
                "goal_category": "relation_goal",
                "text": "把“结果判断”的失败分支改到“发布结果”。",
                "objects": [
                    {"id": "route", "type": "edge_reference", "text": "“结果判断”的失败分支", "role": "source"},
                    {"id": "publish", "type": "existing_reference", "text": "发布结果", "role": "target"},
                ],
                "relation_intents": [{"source": "route", "relation": "route_to", "target": "publish"}],
            },
        ],
        "clarification": {"needed": False, "question": "", "reason": ""},
    }

    def complete(system: str, _payload: dict) -> SimpleNamespace:
        if "Relation Grouper" in system:
            return SimpleNamespace(raw_response="", parsed_json={"relation_groups": []}, parse_error=None, normalized_from_fence=False)
        return SimpleNamespace(raw_response="", parsed_json=spec, parse_error=None, normalized_from_fence=False)

    monkeypatch.setattr(semantic_shadow, "_complete_json", complete)
    trace = semantic_shadow.run_semantic_shadow(review_confirm_v3_graph(), instruction)

    route_goal = next(goal for goal in trace["grounded_goal"]["semantic_goals"] if goal.get("source_subgoal_id") == "g2:relation:0")
    assert route_goal["kind"] == "route"
    assert next(binding for binding in route_goal["bindings"] if binding["role"] == "route")["id"] == "e_decision_human"


def test_explicit_node_to_node_connected_to_adds_direct_edge(monkeypatch) -> None:
    instruction = "让“人工复核”直接连接“发布结果”。"
    spec = {
        "status": "resolved",
        "semantic_goals": [
            {
                "id": "g1",
                "goal_category": "relation_goal",
                "text": instruction,
                "objects": [
                    {"id": "human", "type": "existing_reference", "text": "人工复核", "role": "source"},
                    {"id": "publish", "type": "existing_reference", "text": "发布结果", "role": "target"},
                ],
                "relation_intents": [
                    {"source": "human", "relation": "connected_to", "target": "publish"},
                ],
            }
        ],
        "clarification": {"needed": False, "question": "", "reason": ""},
    }

    def complete(system: str, _payload: dict) -> SimpleNamespace:
        if "Relation Grouper" in system:
            return SimpleNamespace(raw_response="", parsed_json={"relation_groups": []}, parse_error=None, normalized_from_fence=False)
        return SimpleNamespace(raw_response="", parsed_json=spec, parse_error=None, normalized_from_fence=False)

    graph = review_publish_graph()
    graph["edges"] = [edge for edge in graph["edges"] if not (edge["source"] == "n_human" and edge["target"] == "n_publish")]
    monkeypatch.setattr(semantic_shadow, "_complete_json", complete)

    trace = semantic_shadow.run_semantic_shadow(graph, instruction)

    assert trace["stage"] == "success"
    assert ("n_human", "n_publish", "") in edge_set(trace["final_shadow_graph"])
