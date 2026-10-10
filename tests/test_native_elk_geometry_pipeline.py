from __future__ import annotations

import json
from pathlib import Path


FRONTEND_MAIN = Path("demo/frontend/src/main.ts")
PACKAGE_JSON = Path("demo/frontend/package.json")


def source() -> str:
    return FRONTEND_MAIN.read_text(encoding="utf-8")


def block(text: str, start: str, end: str) -> str:
    return text[text.index(start) : text.index(end)]


def test_native_elk_is_a_direct_frontend_dependency() -> None:
    dependencies = json.loads(PACKAGE_JSON.read_text(encoding="utf-8"))["dependencies"]

    assert dependencies["elkjs"].startswith("^0.11")


def test_native_elk_graph_uses_display_projection_and_stable_ports() -> None:
    text = source()
    elk_block = block(text, "function buildNativeElkGraph", "function pointKey")

    assert "graphData.nodes.map" in elk_block
    assert "graphData.edges.map" in elk_block
    assert "nativeElkPortFromAnchor" in elk_block
    assert "'elk.portConstraints': 'FIXED_SIDE'" in elk_block
    assert "'elk.port.side'" in text
    assert "'elk.port.index'" not in text
    assert "sourcePort: edge.sourceAnchorId" in elk_block
    assert "targetPort: edge.targetAnchorId" in elk_block
    assert "sources: [edge.sourceAnchorId ?? edge.sourceNodeId]" in elk_block
    assert "targets: [edge.targetAnchorId ?? edge.targetNodeId]" in elk_block
    assert "workflowGraph" not in elk_block


def test_native_elk_options_prioritize_orthogonal_correctness() -> None:
    text = source()
    options_block = block(text, "function nativeElkOptions", "function buildNativeElkGraph")

    assert "'elk.algorithm': 'layered'" in options_block
    assert "'elk.direction': 'RIGHT'" in options_block
    assert "'elk.edgeRouting': 'ORTHOGONAL'" in options_block
    assert "'elk.separateConnectedComponents': 'true'" in options_block
    assert "'elk.spacing.edgeNode'" in options_block
    assert "'elk.spacing.edgeEdge'" in options_block
    assert "'elk.layered.crossingMinimization.strategy'" in options_block


def test_elk_sections_are_mapped_to_logicflow_points_without_process_edges() -> None:
    text = source()
    mapping_block = block(text, "function applyNativeElkLayoutToGraphData", "function validateGeometryContract")
    native_runner_block = block(text, "async function runNativeElkLayout", "function runOfficialLayout")

    assert "flattenElkSections(edge.id, elkEdge?.sections)" in mapping_block
    assert "startPoint: pointsList[0]" in mapping_block
    assert "endPoint: pointsList[pointsList.length - 1]" in mapping_block
    assert "pointsList" in mapping_block
    assert "const renderRawData = (lf as any).renderRawData?.bind(lf)" in native_runner_block
    assert "renderRawData?.(nextGraphData)" in native_runner_block
    assert "rawRenderRawData?.(nextGraphData)" not in native_runner_block
    assert "extensions.dagre" not in native_runner_block
    assert "extensions.elkLayout" not in native_runner_block


def test_multi_section_edges_are_not_silently_truncated() -> None:
    text = source()
    flatten_block = block(text, "function flattenElkSections", "function anchorsOffsetFromElkPorts")

    assert "sections.length === 1" in flatten_block
    assert "section.outgoingSections" in flatten_block
    assert "ordered.forEach" in flatten_block
    assert "AI_NATIVE_ELK_MULTI_SECTION_EDGE_FLATTENED" in flatten_block
    assert "sections[0]" in flatten_block


def test_native_path_and_legacy_fallback_are_mutually_exclusive() -> None:
    text = source()
    run_block = block(text, "function runOfficialLayout", "function render(graph")
    legacy_block = block(text, "function runLegacyOfficialLayout", "async function runNativeElkLayout")

    assert "if (engine === 'native_elk')" in run_block
    assert "return;" in run_block
    assert "runLegacyOfficialLayout(reason, runId)" in run_block
    assert "extensions.dagre?.layout" in legacy_block
    assert "extensions.elkLayout?.layout" in legacy_block
    assert "extensions.dagre?.layout" not in run_block
    assert "extensions.elkLayout?.layout" not in run_block


def test_display_aggregation_stays_before_native_elk() -> None:
    text = source()
    workflow_block = block(text, "function workflowToLogicFlow", "function readLogicFlowSelection")
    render_block = block(text, "function render(graph", "function pretty")

    assert "groupEdgesByEndpoints(graph.edges)" in workflow_block
    assert "edges: displayEdges" in workflow_block
    assert "const graphData = workflowToLogicFlow(workflowGraph)" in render_block
    assert "runOfficialLayout(reason, graphData)" in render_block
