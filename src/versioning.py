from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any

from .mvp23 import TransactionResult, run_transaction


def _snapshot(graph: dict[str, Any]) -> dict[str, Any]:
    return deepcopy(graph)


def _graph_is_valid(graph: dict[str, Any] | None) -> bool:
    if not isinstance(graph, dict):
        return False
    nodes = graph.get("nodes")
    edges = graph.get("edges")
    if not isinstance(nodes, list) or not isinstance(edges, list):
        return False
    node_ids = [node.get("id") for node in nodes if isinstance(node, dict)]
    edge_ids = [edge.get("id") for edge in edges if isinstance(edge, dict)]
    if len(node_ids) != len(nodes) or len(edge_ids) != len(edges):
        return False
    if len(node_ids) != len(set(node_ids)) or len(edge_ids) != len(set(edge_ids)) or None in set(node_ids) or None in set(edge_ids):
        return False
    node_id_set = set(node_ids)
    return all(edge.get("source") in node_id_set and edge.get("target") in node_id_set for edge in edges if isinstance(edge, dict))


@dataclass
class GraphVersion:
    version_id: str
    parent_version_id: str | None
    graph_snapshot: dict[str, Any]
    created_at: str


@dataclass
class EditTransaction:
    transaction_id: str
    parent_version_id: str
    instruction: str
    selection: dict[str, Any]
    target: dict[str, Any] | None
    specification: dict[str, Any] | None
    plan: list[dict[str, Any]]
    resulting_version_id: str | None
    status: str
    error: str | None = None


@dataclass
class VersionState:
    versions: dict[str, GraphVersion]
    version_order: list[str]
    current_version_id: str
    transactions: list[EditTransaction] = field(default_factory=list)

    def current_graph(self) -> dict[str, Any]:
        return _snapshot(self.versions[self.current_version_id].graph_snapshot)


@dataclass
class VersionedTransactionResult:
    transaction: EditTransaction
    execution: TransactionResult
    committed_version: GraphVersion | None


def create_version_state(graph: dict[str, Any], created_at: str | None = None) -> VersionState:
    timestamp = created_at or datetime.now(UTC).isoformat(timespec="seconds")
    initial = GraphVersion(version_id="v1", parent_version_id=None, graph_snapshot=_snapshot(graph), created_at=timestamp)
    return VersionState(versions={"v1": initial}, version_order=["v1"], current_version_id="v1")


def _next_version_id(state: VersionState) -> str:
    return f"v{len(state.version_order) + 1}"


def _next_transaction_id(state: VersionState) -> str:
    return f"tx{len(state.transactions) + 1}"


def _nodes_by_id(graph: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        node["id"]: node
        for node in graph.get("nodes", [])
        if isinstance(node, dict) and isinstance(node.get("id"), str)
    }


def _edge_key(edge: dict[str, Any]) -> tuple[str, str, str]:
    return (str(edge.get("source")), str(edge.get("target")), str(edge.get("label", "")))


def _edges_by_key(graph: dict[str, Any]) -> dict[tuple[str, str, str], dict[str, Any]]:
    return {
        _edge_key(edge): edge
        for edge in graph.get("edges", [])
        if isinstance(edge, dict) and isinstance(edge.get("source"), str) and isinstance(edge.get("target"), str)
    }


def transaction_diff(
    transaction: EditTransaction,
    before_graph: dict[str, Any],
    after_graph: dict[str, Any],
) -> dict[str, Any]:
    before_nodes = _nodes_by_id(before_graph)
    after_nodes = _nodes_by_id(after_graph)
    mutations: list[dict[str, Any]] = []
    index = 1
    for node_id in sorted(set(before_nodes) & set(after_nodes)):
        before = before_nodes[node_id]
        after = after_nodes[node_id]
        for property_name in sorted(set(before) | set(after)):
            if property_name == "id":
                continue
            if before.get(property_name) == after.get(property_name):
                continue
            mutations.append(
                {
                    "mutation_id": f"{transaction.transaction_id}.m{index}",
                    "kind": "property",
                    "target_entity_id": node_id,
                    "property": property_name,
                    "before": deepcopy(before.get(property_name)),
                    "after": deepcopy(after.get(property_name)),
                }
            )
            index += 1
    before_edges = _edges_by_key(before_graph)
    after_edges = _edges_by_key(after_graph)
    for source, target, label in sorted(set(before_edges) - set(after_edges)):
        mutations.append(
            {
                "mutation_id": f"{transaction.transaction_id}.m{index}",
                "kind": "relation_absent",
                "source": source,
                "target": target,
                "relation_type": label,
            }
        )
        index += 1
    for source, target, label in sorted(set(after_edges) - set(before_edges)):
        mutations.append(
            {
                "mutation_id": f"{transaction.transaction_id}.m{index}",
                "kind": "relation_present",
                "source": source,
                "target": target,
                "relation_type": label,
            }
        )
        index += 1
    return {
        "transaction_id": transaction.transaction_id,
        "from_version_id": transaction.parent_version_id,
        "to_version_id": transaction.resulting_version_id,
        "instruction": transaction.instruction,
        "mutations": mutations,
    }


def recent_successful_transaction_diff(state: VersionState) -> dict[str, Any] | None:
    for transaction in reversed(state.transactions):
        if transaction.status != "success" or transaction.resulting_version_id != state.current_version_id:
            continue
        parent = state.versions.get(transaction.parent_version_id)
        current = state.versions.get(transaction.resulting_version_id)
        if parent is None or current is None:
            return None
        return transaction_diff(transaction, parent.graph_snapshot, current.graph_snapshot)
    return None


def run_versioned_transaction(
    state: VersionState,
    revision: int,
    selection: dict[str, Any],
    instruction: str,
    short_context: list[dict[str, str]] | None = None,
    resolved_constraints: list[dict[str, Any]] | None = None,
    allowed_transformations: list[str] | None = None,
    request_context: dict[str, Any] | None = None,
) -> VersionedTransactionResult:
    parent_version_id = state.current_version_id
    graph = state.current_graph()
    transaction_request_context = deepcopy(request_context) if isinstance(request_context, dict) else {}
    recent_diff = recent_successful_transaction_diff(state)
    if recent_diff is not None and "recent_transaction_diff" not in transaction_request_context:
        transaction_request_context["recent_transaction_diff"] = recent_diff
    execution = run_transaction(
        graph,
        revision,
        selection,
        instruction,
        short_context=short_context,
        resolved_constraints=resolved_constraints,
        allowed_transformations=allowed_transformations,
        request_context=transaction_request_context,
    )
    committed_version: GraphVersion | None = None
    resulting_version_id: str | None = None
    if execution.status == "success" and _graph_is_valid(execution.final_graph):
        parent_index = state.version_order.index(parent_version_id)
        stale_future = state.version_order[parent_index + 1 :]
        for version_id in stale_future:
            state.versions.pop(version_id, None)
        if stale_future:
            state.version_order = state.version_order[: parent_index + 1]
        version_id = _next_version_id(state)
        committed_version = GraphVersion(
            version_id=version_id,
            parent_version_id=parent_version_id,
            graph_snapshot=_snapshot(execution.final_graph),
            created_at=datetime.now(UTC).isoformat(timespec="seconds"),
        )
        state.versions[version_id] = committed_version
        state.version_order.append(version_id)
        state.current_version_id = version_id
        resulting_version_id = version_id
    elif execution.status == "success":
        execution = TransactionResult(
            status="execution_error",
            interpretation=execution.interpretation,
            boundary=execution.boundary,
            steps=execution.steps,
            final_graph=None,
            error="commit rejected: final graph failed structural validation",
            normalized_plan_from_fence=execution.normalized_plan_from_fence,
            planning_path=execution.planning_path,
            semantic_operation=execution.semantic_operation,
            compiler_reason=execution.compiler_reason,
            target=execution.target,
            graph_context=execution.graph_context,
            specification=execution.specification,
            mutation_authorization=execution.mutation_authorization,
        )
    transaction = EditTransaction(
        transaction_id=_next_transaction_id(state),
        parent_version_id=parent_version_id,
        instruction=instruction,
        selection=deepcopy(selection),
        target=deepcopy(execution.target),
        specification=deepcopy(execution.specification),
        plan=deepcopy(execution.steps),
        resulting_version_id=resulting_version_id,
        status=execution.status,
        error=execution.error,
    )
    state.transactions.append(transaction)
    return VersionedTransactionResult(transaction=transaction, execution=execution, committed_version=committed_version)


def undo(state: VersionState) -> GraphVersion | None:
    current = state.versions[state.current_version_id]
    if current.parent_version_id is None:
        return None
    state.current_version_id = current.parent_version_id
    return state.versions[state.current_version_id]


def redo(state: VersionState) -> GraphVersion | None:
    index = state.version_order.index(state.current_version_id)
    if index + 1 >= len(state.version_order):
        return None
    state.current_version_id = state.version_order[index + 1]
    return state.versions[state.current_version_id]


def rollback_to_version(state: VersionState, version_id: str) -> GraphVersion | None:
    if version_id not in state.versions:
        return None
    state.current_version_id = version_id
    return state.versions[version_id]


def as_jsonable(state: VersionState) -> dict[str, Any]:
    return {
        "versions": {version_id: asdict(version) for version_id, version in state.versions.items()},
        "version_order": list(state.version_order),
        "current_version_id": state.current_version_id,
        "transactions": [asdict(transaction) for transaction in state.transactions],
    }
