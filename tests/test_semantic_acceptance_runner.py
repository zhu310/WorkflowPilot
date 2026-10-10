from __future__ import annotations

from scripts.run_semantic_acceptance import evaluate_expected, preflight_cases


def graph() -> dict:
    return {
        "nodes": [
            {"id": "x", "type": "task", "label": "X"},
            {"id": "a", "type": "task", "label": "A", "owner": "ops"},
            {"id": "b", "type": "task", "label": "B"},
            {"id": "c", "type": "task", "label": "C"},
            {"id": "review", "type": "task", "label": "Review"},
        ],
        "edges": [
            {"id": "e_x_a", "source": "x", "target": "a", "label": ""},
            {"id": "e_a_b", "source": "a", "target": "b", "label": ""},
            {"id": "e_b_c", "source": "b", "target": "c", "label": ""},
            {"id": "e_a_review", "source": "a", "target": "review", "label": "success", "branch_label": "ok", "condition": {"expression": "score >= 80"}},
            {"id": "e_b_review", "source": "b", "target": "review", "label": "failure", "branch_label": "fail"},
        ],
    }


def test_strict_preserve_allows_declared_property_change() -> None:
    before = graph()
    final = graph()
    final["nodes"][1]["label"] = "A+"

    passed, stage, errors = evaluate_expected(
        before,
        final,
        {"required_properties": [{"node_id": "a", "changes": {"label": "A+"}}], "strict_preserve_others": True},
    )

    assert passed is True
    assert stage == ""
    assert errors == []


def test_strict_preserve_allows_declared_nested_leaf_on_new_parent() -> None:
    before = graph()
    final = graph()
    final["nodes"][3]["join"] = {"mode": "all"}

    passed, stage, errors = evaluate_expected(
        before,
        final,
        {"required_properties": [{"node_id": "c", "changes": {"join.mode": "all"}}], "strict_preserve_others": True},
    )

    assert passed is True
    assert stage == ""
    assert errors == []


def test_strict_preserve_rejects_undeclared_nested_leaf_on_new_parent() -> None:
    before = graph()
    final = graph()
    final["nodes"][3]["join"] = {"mode": "all", "strategy": "barrier"}

    passed, stage, errors = evaluate_expected(
        before,
        final,
        {"required_properties": [{"node_id": "c", "changes": {"join.mode": "all"}}], "strict_preserve_others": True},
    )

    assert passed is False
    assert stage == "preservation_violation"
    assert "unexpected_property_change: node c" in errors


def test_strict_preserve_allows_declared_nested_leaf_update() -> None:
    before = graph()
    before["nodes"][3]["join"] = {"mode": "any"}
    final = graph()
    final["nodes"][3]["join"] = {"mode": "all"}

    passed, stage, errors = evaluate_expected(
        before,
        final,
        {"required_properties": [{"node_id": "c", "changes": {"join.mode": "all"}}], "strict_preserve_others": True},
    )

    assert passed is True
    assert stage == ""
    assert errors == []


def test_strict_preserve_rejects_extra_node_label_change() -> None:
    before = graph()
    final = graph()
    final["nodes"][1]["label"] = "A+"
    final["nodes"][2]["label"] = "B+"

    passed, stage, errors = evaluate_expected(
        before,
        final,
        {"required_properties": [{"node_id": "a", "changes": {"label": "A+"}}], "strict_preserve_others": True},
    )

    assert passed is False
    assert stage == "preservation_violation"
    assert "unexpected_property_change: node b" in errors


def test_strict_preserve_rejects_unrelated_edge_deletion() -> None:
    before = graph()
    final = graph()
    final["edges"] = [edge for edge in final["edges"] if edge["id"] != "e_b_c"]

    passed, stage, errors = evaluate_expected(before, final, {"strict_preserve_others": True})

    assert passed is False
    assert stage == "preservation_violation"
    assert "unexpected_edge_change: removed e_b_c" in errors


def test_strict_preserve_rejects_extra_edge_property_change() -> None:
    before = graph()
    final = graph()
    final["edges"][3]["condition"] = {"expression": "score >= 90"}
    final["edges"][3]["branch_label"] = "approved"

    passed, stage, errors = evaluate_expected(
        before,
        final,
        {
            "required_properties": [{"edge_id": "e_a_review", "changes": {"condition.expression": "score >= 90"}}],
            "strict_preserve_others": True,
        },
    )

    assert passed is False
    assert stage == "preservation_violation"
    assert "unexpected_property_change: edge e_a_review" in errors


def test_strict_preserve_allows_declared_reorder_edges() -> None:
    before = graph()
    final = graph()
    final["edges"] = [
        {"id": "new_x_b", "source": "x", "target": "b", "label": ""},
        {"id": "new_b_a", "source": "b", "target": "a", "label": ""},
        {"id": "new_a_c", "source": "a", "target": "c", "label": ""},
        final["edges"][3],
        final["edges"][4],
    ]

    passed, stage, errors = evaluate_expected(
        before,
        final,
        {
            "required_edges": [
                {"source": "x", "target": "b", "label": ""},
                {"source": "b", "target": "a", "label": ""},
                {"source": "a", "target": "c", "label": ""},
            ],
            "forbidden_edges": [
                {"source": "x", "target": "a", "label": ""},
                {"source": "a", "target": "b", "label": ""},
                {"source": "b", "target": "c", "label": ""},
            ],
            "strict_preserve_others": True,
        },
    )

    assert passed is True
    assert stage == ""
    assert errors == []


def test_strict_preserve_treats_same_id_different_relation_as_topology_replacement() -> None:
    before = graph()
    final = graph()
    final["edges"][1] = {"id": "e_a_b", "source": "a", "target": "c", "label": ""}

    passed, stage, errors = evaluate_expected(
        before,
        final,
        {
            "required_edges": [{"source": "a", "target": "c", "label": ""}],
            "forbidden_edges": [{"source": "a", "target": "b", "label": ""}],
            "strict_preserve_others": True,
        },
    )

    assert passed is True
    assert stage == ""
    assert errors == []


def test_strict_preserve_rejects_metadata_relation_deletion() -> None:
    before = graph()
    final = graph()
    final["edges"] = [edge for edge in final["edges"] if edge["id"] != "e_b_review"]

    passed, stage, errors = evaluate_expected(before, final, {"strict_preserve_others": True})

    assert passed is False
    assert stage == "preservation_violation"
    assert "unexpected_edge_change: removed e_b_review" in errors


def case_with_expected(expected: dict, *, case_id: str = "case") -> dict:
    return {"id": case_id, "initial_graph": graph(), "expected": expected}


def test_preflight_rejects_required_and_forbidden_same_edge() -> None:
    errors = preflight_cases(
        [
            case_with_expected(
                {
                    "required_edges": [{"source": "x", "target": "a", "label": ""}],
                    "forbidden_edges": [{"source": "x", "target": "a", "label": ""}],
                }
            )
        ]
    )

    assert errors[0]["conflict_kind"] == "required_forbidden_edge_conflict"
    assert errors[0]["case_id"] == "case"


def test_preflight_rejects_preserved_edge_with_property_mutation() -> None:
    errors = preflight_cases(
        [
            case_with_expected(
                {
                    "preserved_edges": ["e_a_review"],
                    "required_properties": [{"edge_id": "e_a_review", "changes": {"branch_label": "approved"}}],
                }
            )
        ]
    )

    assert errors[0]["conflict_kind"] == "preserved_edge_property_mutation_conflict"
    assert errors[0]["fields"]["edge_id"] == "e_a_review"


def test_preflight_rejects_preserved_node_with_property_mutation() -> None:
    errors = preflight_cases(
        [
            case_with_expected(
                {
                    "preserved_nodes": ["a"],
                    "required_properties": [{"node_id": "a", "changes": {"label": "A+"}}],
                }
            )
        ]
    )

    assert errors[0]["conflict_kind"] == "preserved_node_property_mutation_conflict"
    assert errors[0]["fields"]["node_id"] == "a"


def test_preflight_accepts_legal_strict_preserve_case() -> None:
    errors = preflight_cases(
        [
            case_with_expected(
                {
                    "required_properties": [{"node_id": "a", "changes": {"label": "A+"}}],
                    "strict_preserve_others": True,
                }
            )
        ]
    )

    assert errors == []


def test_preflight_accepts_legal_topology_replacement() -> None:
    errors = preflight_cases(
        [
            case_with_expected(
                {
                    "required_edges": [{"source": "x", "target": "b", "label": ""}],
                    "forbidden_edges": [{"source": "x", "target": "a", "label": ""}],
                }
            )
        ]
    )

    assert errors == []


def test_preflight_reports_sequential_step_for_conflict() -> None:
    errors = preflight_cases(
        [
            {
                "id": "seq",
                "initial_graph": graph(),
                "steps": [
                    {"instruction": "ok", "expected": {"required_edges": [{"source": "x", "target": "a", "label": ""}]}},
                    {
                        "instruction": "bad",
                        "expected": {
                            "required_edges": [{"source": "a", "target": "b", "label": ""}],
                            "forbidden_edges": [{"source": "a", "target": "b", "label": ""}],
                        },
                    },
                ],
            }
        ]
    )

    assert errors[0]["case_id"] == "seq"
    assert errors[0]["step"] == 2
    assert errors[0]["conflict_kind"] == "required_forbidden_edge_conflict"
