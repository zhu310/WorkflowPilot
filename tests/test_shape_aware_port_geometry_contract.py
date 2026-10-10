from __future__ import annotations

from pathlib import Path
from math import isclose, sqrt


FRONTEND_MAIN = Path("demo/frontend/src/main.ts")


def source() -> str:
    return FRONTEND_MAIN.read_text(encoding="utf-8")


def block(text: str, start: str, end: str) -> str:
    return text[text.index(start) : text.index(end)]


def test_canonical_shape_geometry_drives_logicflow_and_elk_dimensions() -> None:
    text = source()
    geometry_block = block(text, "function canonicalShapeGeometry", "function sideForRelativeGeometry")
    workflow_block = block(text, "function workflowToLogicFlow", "function readLogicFlowSelection")
    elk_block = block(text, "function buildNativeElkGraph", "function pointKey")

    assert "workflowType === 'decision'" in geometry_block
    assert "workflowType === 'start' || workflowType === 'end'" in geometry_block
    assert "return { kind, width, height, rx: width / 2, ry: height / 2, radius: 0 }" in geometry_block
    assert "const geometry = canonicalShapeGeometry(node)" in workflow_block
    assert "rx: geometry.rx" in workflow_block
    assert "ry: geometry.ry" in workflow_block
    assert "r: geometry.rx" in workflow_block
    assert "width: geometry.width" in workflow_block
    assert "height: geometry.height" in workflow_block
    assert "const geometry = canonicalShapeGeometry(node)" in elk_block
    assert "const { width, height } = geometry" in elk_block


def test_shape_aware_boundary_formulas_cover_rect_diamond_and_ellipse() -> None:
    text = source()
    boundary_block = block(text, "function distributedPortCoordinate", "function assignStablePorts")

    assert "kind === 'rect'" in boundary_block
    assert "kind === 'diamond'" in boundary_block
    assert "1 - Math.abs(y) / ry" in boundary_block
    assert "1 - Math.abs(x) / rx" in boundary_block
    assert "Math.sqrt(Math.max(0, 1 - (y * y) / (ry * ry)))" in boundary_block
    assert "Math.sqrt(Math.max(0, 1 - (x * x) / (rx * rx)))" in boundary_block
    assert "shapeAwareBoundaryPoint(side, index, sorted.length, geometry)" in text


def test_elk_fixed_side_ports_keep_side_semantics_without_fixed_coordinates() -> None:
    text = source()
    port_block = block(text, "function nativeElkPortFromAnchor", "function nativeElkOptions")

    assert "centerX = geometry.width / 2 + anchor.x" in port_block
    assert "centerY = geometry.height / 2 + anchor.y" in port_block
    assert "'elk.port.side': elkSide(side)" in port_block
    assert "'elk.portConstraints': 'FIXED_SIDE'" in text
    assert "'elk.portConstraints': 'FIXED_POS'" not in text


def test_endpoint_projection_uses_final_elk_segment_direction_and_shape_boundary() -> None:
    text = source()
    projection_block = block(text, "function shapeBoundaryHorizontalX", "function clampRouteEndpointsToCanonicalAnchors")
    clamp_block = block(text, "function clampRouteEndpointsToCanonicalAnchors", "function applyNativeElkLayoutToGraphData")

    assert "function projectEndpointToShapeBoundary" in projection_block
    assert "shapeBoundaryHorizontalX(geometry, localY)" in projection_block
    assert "shapeBoundaryVerticalY(geometry, localX)" in projection_block
    assert "projectEndpointToShapeBoundary(laidOutNodesById.get(edge.sourceNodeId), pointsList[0], pointsList[1])" in clamp_block
    assert "pointsList[pointsList.length - 1]" in clamp_block
    assert "pointsList[pointsList.length - 2]" in clamp_block


def test_shape_geometry_is_not_demo_specific() -> None:
    text = source()
    geometry_block = block(text, "function canonicalShapeKind", "function assignStablePorts")

    forbidden = [
        "n_approval_decision",
        "e_decision_report",
        "e_decision_human",
        "结果判断",
        "生成报告",
        "人工复核",
    ]
    for token in forbidden:
        assert token not in geometry_block


def distributed(side: str, index: int, count: int, width: float, height: float) -> float:
    span = height if side in {"left", "right"} else width
    usable = max(0.0, span - 20.0)
    return 0.0 if count <= 1 else -usable / 2 + (usable * index) / (count - 1)


def boundary_point(kind: str, side: str, index: int, count: int, width: float, height: float) -> tuple[float, float]:
    rx = width / 2
    ry = height / 2
    stable = distributed(side, index, count, width, height)
    if kind == "rect":
        if side == "left":
            return -width / 2, stable
        if side == "right":
            return width / 2, stable
        if side == "top":
            return stable, -height / 2
        return stable, height / 2
    if kind == "diamond":
        if side in {"left", "right"}:
            y = max(-ry, min(ry, stable))
            x = rx * max(0.0, 1 - abs(y) / ry)
            return (x if side == "right" else -x), y
        x = max(-rx, min(rx, stable))
        y = ry * max(0.0, 1 - abs(x) / rx)
        return x, y if side == "bottom" else -y
    if side in {"left", "right"}:
        y = max(-ry, min(ry, stable))
        x = rx * sqrt(max(0.0, 1 - (y * y) / (ry * ry)))
        return (x if side == "right" else -x), y
    x = max(-rx, min(rx, stable))
    y = ry * sqrt(max(0.0, 1 - (x * x) / (rx * rx)))
    return x, y if side == "bottom" else -y


def boundary_error(kind: str, point: tuple[float, float], width: float, height: float) -> float:
    x, y = point
    rx = width / 2
    ry = height / 2
    if kind == "rect":
        return min(abs(abs(x) - rx), abs(abs(y) - ry))
    if kind == "diamond":
        return abs(abs(x) / rx + abs(y) / ry - 1)
    return abs((x * x) / (rx * rx) + (y * y) / (ry * ry) - 1)


def test_synthetic_fan_in_out_ports_land_on_shape_boundaries() -> None:
    cases = [
        ("rect", 140.0, 72.0, ["left", "right", "top", "bottom"], [3]),
        ("diamond", 104.0, 104.0, ["left", "right", "top", "bottom"], [2, 3]),
        ("ellipse", 96.0, 96.0, ["left", "right", "top", "bottom"], [3]),
        ("ellipse", 120.0, 80.0, ["left", "right", "top", "bottom"], [3]),
    ]
    max_error = 0.0
    for kind, width, height, sides, counts in cases:
        for side in sides:
            for count in counts:
                for index in range(count):
                    point = boundary_point(kind, side, index, count, width, height)
                    error = boundary_error(kind, point, width, height)
                    max_error = max(max_error, error)
                    assert isclose(error, 0.0, abs_tol=1e-9)
    assert max_error == 0.0
