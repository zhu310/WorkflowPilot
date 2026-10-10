from __future__ import annotations

from demo import server
from demo_api_helpers import INTERNAL_TOKENS, create_session, demo_graph
from src.mvp23 import TransactionResult
from src.versioning import EditTransaction, GraphVersion, VersionedTransactionResult


def _assert_no_internal_tokens(value: str) -> None:
    lowered = value.lower()
    for token in INTERNAL_TOKENS:
        assert token not in lowered


def test_demo_api_message_uses_response_diff_when_spec_trace_diff_is_missing(monkeypatch) -> None:
    server.SESSIONS.clear()
    monkeypatch.setenv("AI_EDIT_PIPELINE", "semantic")
    session_id = create_session(demo_graph())

    def fake_run_versioned_semantic_transaction(state, revision, selection, instruction, request_context=None):
        before = state.current_graph()
        after = {
            **before,
            "nodes": [
                {**node, "label": "生成正式报告"} if node.get("id") == "n_report" else node
                for node in before["nodes"]
            ],
        }
        execution = TransactionResult(
            status="success",
            interpretation=None,
            boundary=None,
            steps=[{"kind": "property", "node_id": "n_report"}],
            final_graph=after,
            error=None,
            normalized_plan_from_fence=False,
            planning_path="semantic",
            target={"selection": selection},
            graph_context=None,
            specification={
                "partial_execution": {
                    "subgoals": [
                        {"id": "g1:state:0", "description": "o1 property_equals", "status": "executed"}
                    ],
                    "executed_count": 1,
                    "incomplete_count": 0,
                    "final_status": "success",
                },
                "trace": {},
            },
            mutation_authorization=None,
        )
        committed = GraphVersion(
            version_id="v2",
            parent_version_id=state.current_version_id,
            graph_snapshot=after,
            created_at="2026-09-28T00:00:00+00:00",
        )
        state.versions[committed.version_id] = committed
        state.version_order.append(committed.version_id)
        state.current_version_id = committed.version_id
        transaction = EditTransaction(
            transaction_id="tx1",
            parent_version_id=committed.parent_version_id or "v1",
            instruction=instruction,
            selection=selection,
            target=execution.target,
            specification=execution.specification,
            plan=execution.steps,
            resulting_version_id=committed.version_id,
            status=execution.status,
            error=None,
        )
        state.transactions.append(transaction)
        return VersionedTransactionResult(transaction=transaction, execution=execution, committed_version=committed)

    monkeypatch.setattr(server, "run_versioned_semantic_transaction", fake_run_versioned_semantic_transaction)

    response = server.run_edit(
        {
            "session_id": session_id,
            "instruction": "把生成报告节点名称改成生成正式报告",
            "selection": {"node_ids": [], "edge_ids": []},
        }
    )

    assert response["status"] == "success"
    assert response["diff"]["changed_nodes"] == [
        {"id": "n_report", "changes": {"label": {"before": "生成报告", "after": "生成正式报告"}}}
    ]
    assert response["message"] == response["user_feedback"]["message"]
    _assert_no_internal_tokens(response["message"])
    _assert_no_internal_tokens(response["user_feedback"]["message"])
    for row in response["user_feedback"]["executed"]:
        _assert_no_internal_tokens(row["summary"])
    assert "生成报告" in response["message"]
    assert "生成正式报告" in response["message"]
    assert response["user_feedback"]["executed"][0]["summary"] != "o1 property_equals"
