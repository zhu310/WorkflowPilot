from __future__ import annotations

from pathlib import Path


FRONTEND_MAIN = Path("demo/frontend/src/main.ts")


def source() -> str:
    return FRONTEND_MAIN.read_text(encoding="utf-8")


def test_app_layer_no_longer_writes_final_edge_points() -> None:
    text = source()
    block = text[text.index("function workflowToLogicFlow") : text.index("function readLogicFlowSelection")]

    assert "pointsList" not in block
    assert "startPoint" not in block
    assert "endPoint" not in block
    assert "text: label" in block
    assert "sourceNodeId: source" in block
    assert "targetNodeId: target" in block
    assert "underlying_edge_ids" in block


def test_obsolete_app_layer_edge_policy_is_removed() -> None:
    text = source()

    assert "function edgeVisualPolicy" not in text
    assert "function routedPoints" not in text
    assert "function externalHorizontalRoutes" not in text
    assert "function externalVerticalRoutes" not in text
    assert "function validatedRoute" not in text
    assert "function pathCrossesIntermediateNode" not in text


def test_native_elk_sections_are_the_edge_routing_owner() -> None:
    text = source()
    layout_block = text[text.index("function runOfficialLayout") : text.index("function render(graph")]

    assert "readGeometryEngine()" in layout_block
    assert "runNativeElkLayout" in layout_block
    assert "runLegacyOfficialLayout" in layout_block
    assert "extensions.dagre?.layout" not in layout_block
    assert "extensions.elkLayout?.layout" not in layout_block


def test_layout_sync_does_not_rerender_over_official_edge_route() -> None:
    text = source()
    sync_block = text[text.index("function syncDisplayGraphAfterLayout") : text.index("function completeOfficialLayout")]

    assert "mergeDisplayNodePositionsIntoWorkflow()" in sync_block
    assert "validateGeometryContract()" in sync_block
    assert "lf.render(" not in sync_block
    assert "workflowToLogicFlow(" not in sync_block
    assert "applySelectionHighlight()" in sync_block


def test_failed_visual_separation_reroute_is_removed_from_production() -> None:
    text = source()

    assert "function applyVisualEdgeSeparationToLogicFlow" not in text
    assert "function visualSeparationOffsets" not in text
    assert "function routeNeedsVisualSeparation" not in text
    assert "function separateDisplayRoute" not in text
    assert "function routeMidpoint" not in text


def test_stable_port_contract_replaces_post_layout_reroute() -> None:
    text = source()
    workflow_block = text[text.index("function workflowToLogicFlow") : text.index("function readLogicFlowSelection")]
    port_block = text[text.index("function assignStablePorts") : text.index("function validateGeometryContract")]

    assert "isDefaultAnchor: false" in text
    assert "const anchorOffsetsByNode = assignStablePorts(graph.nodes, displayEdges)" in workflow_block
    assert "sourceAnchorId" in workflow_block
    assert "targetAnchorId" in workflow_block
    assert "anchorsOffset: anchorOffsetsByNode.get(node.id) ?? []" in workflow_block
    assert "sourceAnchorId = id" in port_block
    assert "targetAnchorId = id" in port_block
    assert "sourcePort: edge.sourceAnchorId" in text
    assert "targetPort: edge.targetAnchorId" in text
    assert "oppositeNodeSortValue(side, a.opposite)" in port_block
    assert "displayEdgeSortKey(a.edge).localeCompare(displayEdgeSortKey(b.edge))" in port_block


def test_overlap_detector_is_validator_only() -> None:
    text = source()
    validator_block = text[text.index("function validateGeometryContract") : text.index("function applySelectionHighlight")]

    assert "function independentRouteOverlap" in text
    assert "overlapLength" in text
    assert "overlapRatio" in text
    assert "AI_GEOMETRY_INDEPENDENT_EDGE_OVERLAP" in validator_block
    assert "renderRawData" not in validator_block
    assert "pointsList:" not in validator_block


def test_stable_port_algorithm_is_not_demo_specific() -> None:
    text = source()
    port_block = text[text.index("function anchorId") : text.index("function validateGeometryContract")]

    forbidden = [
        "e_report_publish",
        "e_report_human",
        "n_report",
        "n_publish",
        "鐢熸垚鎶ュ憡",
        "鍙戝竷缁撴灉",
        "浜哄伐澶嶆牳",
        "success",
        "failure",
    ]
    for token in forbidden:
        assert token not in port_block
