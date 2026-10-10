from pathlib import Path


FRONTEND_MAIN = Path("demo/frontend/src/main.ts")


def source() -> str:
    return FRONTEND_MAIN.read_text(encoding="utf-8")


def block(start: str, end: str) -> str:
    text = source()
    return text[text.index(start) : text.index(end)]


def test_same_source_target_edges_are_aggregated_in_display_graph_only() -> None:
    text = source()
    workflow_block = block("function workflowToLogicFlow", "function readLogicFlowSelection")

    assert "groupEdgesByEndpoints(graph.edges)" in workflow_block
    assert "__afc_display_edge__" in text
    assert "displayEdge: edges.length > 1" in workflow_block
    assert "underlying_edge_ids" in workflow_block
    assert "combinedEdgeLabel(edges)" in workflow_block
    assert "edges: displayEdges" in workflow_block


def test_business_edges_are_not_reconstructed_from_display_edges() -> None:
    text = source()
    sync_block = block("function mergeDisplayNodePositionsIntoWorkflow", "function syncDisplayGraphAfterLayout")

    assert "function logicFlowToWorkflow" not in text
    assert "edges: workflowGraph.edges.map((edge) => ({ ...edge }))" in sync_block
    assert "workflowGraph = logicFlowToWorkflow()" not in text


def test_display_selection_maps_to_underlying_business_edge_ids() -> None:
    text = source()

    assert "displayEdgeLookup.get(id)?.underlying_edge_ids ?? [id]" in text
    assert "toggleEdgeIds(frontendSelectionState.edge_ids, displayEdgeIdsForEvent(id))" in text
    assert "const isSelected = underlying.some((edgeId) => selectedEdgeIds.has(edgeId))" in text
    assert "selection: selectionBeforeSend" in text


def test_aggregated_label_uses_public_branch_descriptors_without_internal_ids() -> None:
    descriptor_block = block("function userVisibleEdgeLabel", "function combinedEdgeLabel")

    assert "edge.branch_label" in descriptor_block
    assert "edge.label" in descriptor_block
    assert "condition[key]" in descriptor_block
    assert "edge.id" not in descriptor_block
    assert "route_to" not in descriptor_block
    assert "workflow_ir" not in descriptor_block
    assert "return ''" in descriptor_block


def test_display_projection_has_runtime_coverage_invariant() -> None:
    text = source()
    workflow_block = block("function workflowToLogicFlow", "function readLogicFlowSelection")

    assert "function assertDisplayProjectionCoverage" in text
    assert "assertDisplayProjectionCoverage(graph.edges, displayEdges)" in workflow_block
    assert "AI_DISPLAY_PROJECTION_MISSING_BUSINESS_EDGE" in text
    assert "AI_DISPLAY_PROJECTION_DUPLICATE_UNDERLYING_EDGE" in text


def test_selection_summary_groups_selected_underlying_edges_by_endpoint() -> None:
    text = source()

    assert "function summarizeSelectedEdges" in text
    assert "groupEdgesByEndpoints(workflowGraph.edges.filter((edge) => selected.has(edge.id)))" in text
    assert "summarizeSelectedEdges(selection.edge_ids)" in text
