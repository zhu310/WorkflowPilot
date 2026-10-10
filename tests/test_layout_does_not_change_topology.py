from __future__ import annotations

from pathlib import Path


FRONTEND_MAIN = Path("demo/frontend/src/main.ts")


def source() -> str:
    return FRONTEND_MAIN.read_text(encoding="utf-8")


def function_block(text: str, name: str, next_name: str) -> str:
    return text[text.index(f"function {name}") : text.index(f"function {next_name}")]


def test_workflow_to_logicflow_preserves_edge_source_and_target() -> None:
    text = source()
    block = function_block(text, "workflowToLogicFlow", "readLogicFlowSelection")

    assert "sourceNodeId: source" in block
    assert "targetNodeId: target" in block
    assert "type: 'polyline'" in block
    assert "groupEdgesByEndpoints(graph.edges)" in block
    assert "underlying_edge_ids" in block


def test_logicflow_display_graph_is_not_a_business_reconstruction_api() -> None:
    text = source()

    assert "function logicFlowToWorkflow" not in text
    assert "workflowGraph = logicFlowToWorkflow()" not in text
    assert "source: edge.sourceNodeId" not in text
    assert "target: edge.targetNodeId" not in text
    assert "label: edge.properties?.label" not in text


def test_official_layout_sync_only_updates_display_graph_from_logicflow() -> None:
    text = source()
    merge_block = function_block(text, "mergeDisplayNodePositionsIntoWorkflow", "syncDisplayGraphAfterLayout")
    sync_block = function_block(text, "syncDisplayGraphAfterLayout", "completeOfficialLayout")

    assert "workflowGraph = logicFlowToWorkflow()" not in sync_block
    assert "x: displayNode.x" in merge_block
    assert "y: displayNode.y" in merge_block
    assert "edges: workflowGraph.edges.map((edge) => ({ ...edge }))" in merge_block
    assert "lf.render(" not in sync_block
    assert "workflowToLogicFlow(" not in sync_block
    assert "applySelectionHighlight()" in sync_block
    assert "fitReadableView()" in sync_block
    fit_block = text[text.index("function fitReadableView") : text.index("const rawRenderRawData")]
    assert "computeDisplayGeometryBBox()" in fit_block
    assert "lf.zoom(nextZoom" in fit_block
    assert "lf.focusOn({ x: bbox.centerX, y: bbox.centerY })" in fit_block
    for block in (merge_block, sync_block):
        assert "addEdge" not in block
        assert "deleteEdge" not in block
        assert "removeEdge" not in block


def test_layout_function_does_not_create_planner_or_backend_mutations() -> None:
    text = source()
    block = function_block(text, "runOfficialLayout", "render")

    forbidden = [
        "fetch('/api/ai-edit'",
        "fetch(\"/api/ai-edit\"",
        "addEdge",
        "deleteEdge",
        "removeEdge",
        "sourceNodeId =",
        "targetNodeId =",
    ]
    for token in forbidden:
        assert token not in block
