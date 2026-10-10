from __future__ import annotations

from demo.server import graph_diff
from demo_api_helpers import assert_no_internal_visible_text
from src.product_feedback import build_user_feedback


def graph() -> dict:
    return {
        "nodes": [
            {"id": "a", "type": "task", "label": "\u751f\u6210\u62a5\u544a"},
            {"id": "b", "type": "task", "label": "\u53d1\u5e03\u7ed3\u679c"},
            {"id": "c", "type": "end", "label": "\u7ed3\u675f"},
        ],
        "edges": [{"id": "e_a_b", "source": "a", "target": "b", "label": ""}],
    }


def structure_graph() -> dict:
    return {
        "nodes": [
            {"id": "parse", "type": "task", "label": "\u89e3\u6790\u4efb\u52a1"},
            {"id": "retrieval", "type": "task", "label": "\u77e5\u8bc6\u68c0\u7d22"},
            {"id": "tool", "type": "task", "label": "\u5de5\u5177\u8c03\u7528"},
            {"id": "judge", "type": "task", "label": "\u7ed3\u679c\u5224\u65ad"},
        ],
        "edges": [
            {"id": "e_parse_retrieval", "source": "parse", "target": "retrieval", "label": ""},
            {"id": "e_retrieval_tool", "source": "retrieval", "target": "tool", "label": ""},
            {"id": "e_tool_judge", "source": "tool", "target": "judge", "label": ""},
        ],
    }


def assert_zh_visible(text: str) -> None:
    for token in ("Inserted", "connection:", "branch:", "between", "multiple graph edges match reference"):
        assert token not in text
    assert_no_internal_visible_text(text)


def test_insert_feedback_uses_chinese_template() -> None:
    before = graph()
    after = {
        "nodes": [*before["nodes"], {"id": "x", "type": "task", "label": "\u68c0\u67e5"}],
        "edges": [
            {"id": "e_a_b", "source": "a", "target": "x", "label": ""},
            {"id": "e_x_b", "source": "x", "target": "b", "label": ""},
        ],
    }
    feedback = build_user_feedback("success", {"trace": {"graph_diff": graph_diff(before, after)}, "partial_execution": {"subgoals": [{"status": "executed"}]}}, after)

    assert "\u5df2\u5728" in feedback["message"]
    assert "\u4e4b\u95f4\u65b0\u589e" in feedback["message"]
    assert_zh_visible(feedback["message"])


def test_insert_feedback_strips_outer_reference_quotes_from_new_node_label() -> None:
    before = graph()
    after = {
        "nodes": [*before["nodes"], {"id": "x", "type": "task", "label": "\u201c\u590d\u6838\u786e\u8ba4\u201d"}],
        "edges": [
            {"id": "e_a_b", "source": "a", "target": "x", "label": ""},
            {"id": "e_x_b", "source": "x", "target": "b", "label": ""},
        ],
    }
    feedback = build_user_feedback(
        "success",
        {"trace": {"graph_diff": graph_diff(before, after)}, "partial_execution": {"subgoals": [{"status": "executed"}]}},
        after,
    )

    assert "\u65b0\u589e\u201c\u590d\u6838\u786e\u8ba4\u201d" in feedback["message"]
    assert "\u201c\u201c\u590d\u6838\u786e\u8ba4\u201d\u201d" not in feedback["message"]
    assert_zh_visible(feedback["message"])


def test_parallel_structure_success_feedback_reports_parallel_goal() -> None:
    before = structure_graph()
    after = {
        "nodes": [
            before["nodes"][0],
            before["nodes"][1],
            before["nodes"][2],
            {**before["nodes"][3], "join": {"mode": "all"}},
        ],
        "edges": [
            {"id": "e_parse_retrieval", "source": "parse", "target": "retrieval", "label": ""},
            {"id": "e_tool_judge", "source": "tool", "target": "judge", "label": ""},
            {"id": "e_parse_tool", "source": "parse", "target": "tool", "label": ""},
            {"id": "e_retrieval_judge", "source": "retrieval", "target": "judge", "label": ""},
        ],
    }
    spec = {
        "grounded_goal": {
            "semantic_goals": [
                {
                    "kind": "parallel",
                    "relation": "parallel_with",
                    "source_subgoal_id": "g1:relation:0",
                    "bindings": [
                        {"role": "left", "binding_type": "node", "id": "retrieval"},
                        {"role": "right", "binding_type": "node", "id": "tool"},
                    ],
                }
            ]
        },
        "trace": {"graph_diff": graph_diff(before, after)},
        "partial_execution": {
            "subgoals": [{"id": "g1:relation:0", "type": "relation_intent", "status": "executed"}],
            "final_status": "success",
        },
    }

    feedback = build_user_feedback("success", spec, after)

    assert "\u5e76\u884c\u7ed3\u6784" in feedback["message"]
    assert "\u77e5\u8bc6\u68c0\u7d22" in feedback["message"]
    assert "\u5de5\u5177\u8c03\u7528" in feedback["message"]
    assert "\u7ed3\u679c\u5224\u65ad" in feedback["message"]
    assert_zh_visible(feedback["message"])


def test_reorder_structure_success_feedback_reports_ordering_goal() -> None:
    before = structure_graph()
    after = {
        "nodes": before["nodes"],
        "edges": [
            {"id": "e_parse_tool", "source": "parse", "target": "tool", "label": ""},
            {"id": "e_tool_retrieval", "source": "tool", "target": "retrieval", "label": ""},
            {"id": "e_retrieval_judge", "source": "retrieval", "target": "judge", "label": ""},
        ],
    }
    spec = {
        "grounded_goal": {
            "semantic_goals": [
                {
                    "kind": "ordering",
                    "relation": "before",
                    "source_subgoal_id": "g1:relation:0",
                    "bindings": [
                        {"role": "subject", "binding_type": "node", "id": "tool"},
                        {"role": "anchor", "binding_type": "node", "id": "retrieval"},
                    ],
                }
            ],
            "semantic_goals_v2": [
                {"semantic_constraints": [{"type": "preserve", "target": "\u5176\u4ed6\u6d41\u7a0b"}]}
            ],
        },
        "trace": {"graph_diff": graph_diff(before, after)},
        "partial_execution": {
            "subgoals": [{"id": "g1:relation:0", "type": "relation_intent", "status": "executed"}],
            "final_status": "success",
        },
    }

    feedback = build_user_feedback("success", spec, after)

    assert "\u5df2\u5c06\u201c\u5de5\u5177\u8c03\u7528\u201d\u8c03\u6574\u5230\u201c\u77e5\u8bc6\u68c0\u7d22\u201d\u4e4b\u524d" in feedback["message"]
    assert "\u5176\u4ed6\u672a\u6d89\u53ca\u7684\u6d41\u7a0b\u8fde\u63a5\u4fdd\u6301\u4e0d\u53d8" in feedback["message"]
    assert_zh_visible(feedback["message"])


def test_grouped_reorder_feedback_prefers_relation_changed_by_transaction() -> None:
    before = structure_graph()
    after = {
        "nodes": before["nodes"],
        "edges": [
            {"id": "e_parse_tool", "source": "parse", "target": "tool", "label": ""},
            {"id": "e_tool_retrieval", "source": "tool", "target": "retrieval", "label": ""},
            {"id": "e_retrieval_judge", "source": "retrieval", "target": "judge", "label": ""},
        ],
    }
    spec = {
        "grounded_goal": {
            "semantic_goals": [],
            "suppressed_legacy_relation_goals": [
                {"source_subgoal_id": "g1:relation:0"},
                {"source_subgoal_id": "g1:relation:1"},
                {"source_subgoal_id": "g1:relation:2"},
            ],
            "relation_facts": [
                {
                    "relation_id": "rg1",
                    "kind": "precedence",
                    "members": [
                        {
                            "source_binding": {"binding_type": "node", "id": "parse"},
                            "target_binding": {"binding_type": "node", "id": "tool"},
                        },
                        {
                            "source_binding": {"binding_type": "node", "id": "tool"},
                            "target_binding": {"binding_type": "node", "id": "retrieval"},
                        },
                        {
                            "source_binding": {"binding_type": "node", "id": "retrieval"},
                            "target_binding": {"binding_type": "node", "id": "judge"},
                        },
                    ],
                }
            ],
            "semantic_goals_v2": [
                {"semantic_constraints": [{"type": "preserve", "target": "\u5176\u4ed6\u6d41\u7a0b"}]}
            ],
        },
        "trace": {"before_graph": before, "graph_diff": graph_diff(before, after)},
        "partial_execution": {
            "subgoals": [
                {"id": "g1:relation:0", "type": "relation_intent", "status": "executed"},
                {"id": "g1:relation:1", "type": "relation_intent", "status": "executed"},
                {"id": "g1:relation:2", "type": "relation_intent", "status": "executed"},
            ],
            "final_status": "success",
        },
    }

    feedback = build_user_feedback("success", spec, after)

    assert "\u5df2\u5c06\u201c\u5de5\u5177\u8c03\u7528\u201d\u8c03\u6574\u5230\u201c\u77e5\u8bc6\u68c0\u7d22\u201d\u4e4b\u524d" in feedback["message"]
    assert "\u5df2\u5c06\u201c\u89e3\u6790\u4efb\u52a1\u201d\u8c03\u6574\u5230\u201c\u5de5\u5177\u8c03\u7528\u201d\u4e4b\u524d" not in feedback["message"]
    assert_zh_visible(feedback["message"])


def test_blocked_structure_feedback_does_not_report_executed_relation_goal() -> None:
    before = structure_graph()
    spec = {
        "grounded_goal": {
            "semantic_goals": [
                {
                    "kind": "ordering",
                    "relation": "before",
                    "source_subgoal_id": "g1:relation:0",
                    "bindings": [
                        {"role": "subject", "binding_type": "node", "id": "tool"},
                        {"role": "anchor", "binding_type": "node", "id": "retrieval"},
                    ],
                }
            ]
        },
        "validation": {"issues": [{"kind": "ambiguous_structural_boundary"}]},
        "partial_execution": {
            "subgoals": [{"id": "g1:relation:0", "type": "relation_intent", "status": "unsupported"}],
            "final_status": "unsupported",
        },
    }

    feedback = build_user_feedback("needs_clarification", spec, before)

    assert "\u8c03\u6574\u5230" not in feedback["message"]
    assert "\u5df2\u5b8c\u6210" not in feedback["message"]
    assert_zh_visible(feedback["message"])


def test_success_feedback_summarizes_only_verified_preserved_edges() -> None:
    before = graph()
    after = {
        "nodes": [*before["nodes"], {"id": "x", "type": "task", "label": "\u68c0\u67e5"}],
        "edges": [*before["edges"], {"id": "e_x_b", "source": "x", "target": "b", "label": ""}],
    }
    spec = {
        "constraints": {"preserve": {"edge_ids": ["e_a_b"]}},
        "grounded_goal": {
            "semantic_goals": [
                {
                    "kind": "object_state",
                    "object_state": {"state": "preserved"},
                    "bindings": [{"role": "object", "binding_type": "edge", "id": "e_a_b"}],
                }
            ],
            "semantic_goals_v2": [
                {
                    "semantic_constraints": [
                        {"type": "preserve", "target": "\u751f\u6210\u62a5\u544a\u5230\u53d1\u5e03\u7ed3\u679c"}
                    ]
                }
            ],
        },
        "trace": {"graph_diff": graph_diff(before, after)},
        "partial_execution": {"subgoals": [{"status": "executed"}]},
    }
    feedback = build_user_feedback("success", spec, after)

    assert "\u5df2\u4fdd\u7559" in feedback["message"]
    assert "\u751f\u6210\u62a5\u544a" in feedback["message"]
    assert "\u53d1\u5e03\u7ed3\u679c" in feedback["message"]
    assert_zh_visible(feedback["message"])


def test_success_feedback_does_not_enumerate_background_preserve_edges() -> None:
    before = graph()
    after = {
        "nodes": [*before["nodes"], {"id": "x", "type": "task", "label": "\u68c0\u67e5"}],
        "edges": [*before["edges"], {"id": "e_x_b", "source": "x", "target": "b", "label": ""}],
    }
    spec = {
        "constraints": {"preserve": {"edge_ids": ["e_a_b", "e_x_b"], "outside_specification": True}},
        "trace": {"graph_diff": graph_diff(before, after)},
        "partial_execution": {"subgoals": [{"status": "executed"}]},
    }
    feedback = build_user_feedback("success", spec, after)

    assert "\u5df2\u4fdd\u7559" not in feedback["message"]
    assert "\u751f\u6210\u62a5\u544a\u5230\u53d1\u5e03\u7ed3\u679c" not in feedback["message"]
    assert_zh_visible(feedback["message"])


def test_success_feedback_does_not_claim_changed_preserve_was_kept() -> None:
    before = graph()
    after = {"nodes": before["nodes"], "edges": [{"id": "e_a_b", "source": "a", "target": "c", "label": ""}]}
    spec = {
        "constraints": {"preserve": {"edge_ids": ["e_a_b"]}},
        "grounded_goal": {
            "semantic_goals": [
                {
                    "kind": "object_state",
                    "object_state": {"state": "preserved"},
                    "bindings": [{"role": "object", "binding_type": "edge", "id": "e_a_b"}],
                }
            ],
        },
        "trace": {"graph_diff": graph_diff(before, after)},
        "partial_execution": {"subgoals": [{"status": "executed"}]},
    }
    feedback = build_user_feedback("success", spec, after)

    assert "\u5df2\u4fdd\u7559" not in feedback["message"]
    assert_zh_visible(feedback["message"])


def test_route_feedback_uses_chinese_connection_template() -> None:
    before = graph()
    after = {"nodes": before["nodes"], "edges": [{"id": "e_a_b", "source": "a", "target": "c", "label": ""}]}
    feedback = build_user_feedback("success", {"trace": {"graph_diff": graph_diff(before, after)}, "partial_execution": {"subgoals": [{"status": "executed"}]}}, after)

    assert "\u8fde\u7ebf" in feedback["message"]
    assert_zh_visible(feedback["message"])


def test_ambiguous_candidates_use_chinese_edge_words() -> None:
    spec = {"validation": {"issues": [{"kind": "ambiguous_edge_reference", "reference": "\u8fd9\u4e2a\u5206\u652f", "candidates": ["e_a_b"], "message": "multiple graph edges match reference"}]}}
    ambiguous_graph = graph()
    ambiguous_graph["edges"] = [
        *ambiguous_graph["edges"],
        {"id": "e_a_c", "source": "a", "target": "c", "label": "backup", "branch_label": "\u5907\u9009"},
    ]
    spec["validation"]["issues"][0]["candidates"] = ["e_a_b", "e_a_c"]
    feedback = build_user_feedback("needs_clarification", spec, ambiguous_graph)

    assert "\u8fde\u7ebf" in feedback["message"]
    assert "\u4f60\u60f3\u4fee\u6539\u54ea\u4e00\u4e2a" in feedback["message"]
    assert "\u672a\u5b8c\u6210" not in feedback["message"]
    assert_zh_visible(feedback["message"])


def test_same_target_confirmation_uses_chinese_visible_template() -> None:
    spec = {
        "trace": {
            "grounded_goal": {
                "grounding": {
                    "issues": [
                        {
                            "kind": "route_same_source_target_confirmation",
                            "source": "a",
                            "target": "b",
                            "relation_type": "success",
                            "conflicting_branch_label": "failure",
                        }
                    ]
                }
            }
        }
    }
    feedback = build_user_feedback("needs_clarification", spec, graph())

    assert "\u662f\u5426" in feedback["message"]
    assert "\u672a\u5b8c\u6210" not in feedback["message"]
    assert_zh_visible(feedback["message"])


def test_missing_edge_without_candidates_does_not_render_empty_candidate_section() -> None:
    spec = {
        "trace": {
            "grounded_goal": {
                "grounding": {
                    "issues": [
                        {
                            "kind": "missing_edge_reference",
                            "reference": "\u8fd9\u6761\u8fde\u7ebf",
                            "candidates": [],
                            "reason": "no graph edge matches reference",
                        }
                    ]
                }
            }
        }
    }
    feedback = build_user_feedback("needs_clarification", spec, graph())

    assert "\u5019\u9009" not in feedback["message"]
    assert feedback["clarification"]["candidates"] == []
    assert "\u5206\u652f" in feedback["clarification"]["question"]
