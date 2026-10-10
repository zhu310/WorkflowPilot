from __future__ import annotations

from pathlib import Path


FRONTEND_MAIN = Path("demo/frontend/src/main.ts")


def source() -> str:
    return FRONTEND_MAIN.read_text(encoding="utf-8")


def block(text: str, start: str, end: str) -> str:
    return text[text.index(start) : text.index(end)]


def test_server_edit_results_are_rendered_through_layout_path() -> None:
    text = source()
    server_block = text[text.index("function applyServerState") :]
    render_block = block(text, "function render", "function pretty")

    assert "if (result.graph) render(result.graph)" in server_block
    assert "else if (result.final_graph) render(result.final_graph)" in server_block
    assert "runOfficialLayout(reason, graphData)" in render_block


def test_manual_layout_button_uses_same_official_layout_path() -> None:
    text = source()

    assert "id=\"fitView\"" in text
    assert "整理布局" in text
    assert "runOfficialLayout('manual')" in text


def test_layout_preserves_selection_debug_and_request_selection_source() -> None:
    text = source()

    assert "const selectionBeforeSend = getSelection()" in text
    assert "writeSelectionDebug(selectionBeforeSend, selectionBeforeSend)" in text
    assert "selection: selectionBeforeSend" in text
    assert "applySelectionHighlight()" in text


def test_layout_sync_occurs_after_native_elk_sections_are_rendered() -> None:
    text = source()
    hook_block = block(text, "const rawRenderRawData", "function runOfficialLayout")
    native_block = block(text, "async function runNativeElkLayout", "function runOfficialLayout")
    layout_block = block(text, "function runOfficialLayout", "function render(graph")

    assert "void runNativeElkLayout(reason, runId, displayGraphData)" in layout_block
    assert "runLegacyOfficialLayout(reason, runId)" in layout_block
    assert "extensions.dagre?.layout" not in layout_block
    assert "extensions.elkLayout?.layout" not in layout_block
    assert "completeOfficialLayout(pending.id, pending.reason)" in hook_block
    assert "const renderRawData = (lf as any).renderRawData?.bind(lf)" in native_block
    assert "renderRawData?.(nextGraphData)" in native_block
    assert "rawRenderRawData?.(nextGraphData)" not in native_block
    assert "window.requestAnimationFrame(() => syncDisplayGraphAfterLayout(reason, runId))" in text
    assert "window.setTimeout(() => syncDisplayGraphAfterLayout(reason), 80)" not in layout_block
    assert "window.setTimeout(() => syncDisplayGraphAfterLayout(reason), 280)" not in layout_block


def test_layout_fit_view_has_readability_floor_without_changing_layout_data() -> None:
    text = source()
    fit_block = block(text, "function fitReadableView", "const rawRenderRawData")

    assert "computeDisplayGeometryBBox()" in fit_block
    assert "const fitZoom = Math.min(availableWidth / bbox.width, availableHeight / bbox.height)" in fit_block
    assert "const nextZoom = clampZoom(fitZoom)" in fit_block
    assert "lf.zoom(nextZoom, [canvasWidth / 2, canvasHeight / 2])" in fit_block
    assert "lf.focusOn({ x: bbox.centerX, y: bbox.centerY })" in fit_block
    assert "fitReadableView()" in block(text, "function syncDisplayGraphAfterLayout", "function completeOfficialLayout")


def test_display_geometry_bbox_includes_nodes_edges_and_padding() -> None:
    text = source()
    bbox_block = block(text, "function computeDisplayGeometryBBox", "function clampZoom")

    assert "canonicalShapeGeometry(node)" in bbox_block
    assert "routePointsFromDisplayEdge(edge).forEach" in bbox_block
    assert "DISPLAY_BBOX_LABEL_PADDING" in bbox_block


def test_native_elk_spacing_uses_compact_layer_distance() -> None:
    text = source()
    options_block = block(text, "function nativeElkOptions", "function buildNativeElkGraph")

    assert "'elk.layered.spacing.nodeNodeBetweenLayers': '120'" in options_block
    assert "'elk.layered.spacing.nodeNodeBetweenLayers': '170'" not in options_block
