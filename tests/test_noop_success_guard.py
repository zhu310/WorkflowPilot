from src.semantic_shadow import semantic_trace_to_transaction_result


def test_success_trace_without_steps_or_diff_is_not_reported_as_success() -> None:
    trace = {
        "stage": "success",
        "selection": {"node_ids": [], "edge_ids": []},
        "graph_context": {"node_count": 1, "edge_count": 0},
        "semantic_spec": {"normalized": {"status": "resolved", "workflow_ir": {}}},
        "partial_execution": {
            "subgoals": [{"id": "g1", "description": "property_equals no-op", "status": "executed"}],
            "executed_count": 1,
            "incomplete_count": 0,
            "final_status": "success",
        },
        "executor": {"steps": [], "applied_steps": [], "mutation_authorization": []},
        "final_shadow_graph": {"nodes": [{"id": "a", "label": "A"}], "edges": []},
        "graph_diff": {
            "added_nodes": [],
            "removed_nodes": [],
            "changed_nodes": [],
            "added_edges": [],
            "removed_edges": [],
            "changed_edges": [],
        },
    }

    result = semantic_trace_to_transaction_result(trace)

    assert result.status == "unsupported"
    assert result.final_graph is None
    assert result.steps == []
    assert result.specification["partial_execution"]["final_status"] == "unsupported"
