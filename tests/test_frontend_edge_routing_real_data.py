from __future__ import annotations

from pathlib import Path


FRONTEND_MAIN = Path("demo/frontend/src/main.ts")


def source() -> str:
    return FRONTEND_MAIN.read_text(encoding="utf-8")


def test_frontend_does_not_define_a_custom_edge_router() -> None:
    text = source()

    forbidden = [
        "function routedPoints",
        "function edgePoints",
        "function anchorPoint",
        "function segmentIntersectsRect",
        "function externalHorizontalRoutes",
        "function externalVerticalRoutes",
    ]
    for token in forbidden:
        assert token not in text


def test_logicflow_edges_are_plain_polyline_inputs_before_official_layout() -> None:
    text = source()
    block = text[text.index("function workflowToLogicFlow") : text.index("function readLogicFlowSelection")]

    assert "type: 'polyline'" in block
    assert "sourceNodeId: source" in block
    assert "targetNodeId: target" in block
    assert "sourceAnchorId" in block
    assert "targetAnchorId" in block
    assert "properties: {" in block
    assert "label" in block
    assert "underlying_edge_ids" in block
    assert "pointsList" not in block


def test_official_render_raw_data_is_the_layout_completion_boundary() -> None:
    text = source()
    hook_block = text[text.index("const rawRenderRawData") : text.index("function runOfficialLayout")]

    assert "rawRenderRawData(graphData)" in hook_block
    assert "completeOfficialLayout(pending.id, pending.reason)" in hook_block


def test_labels_are_not_positioned_by_app_layer_offsets() -> None:
    text = source()
    block = text[text.index("function workflowToLogicFlow") : text.index("function readLogicFlowSelection")]

    assert "edgeText(" not in text
    assert "labelOffset" not in block
    assert "text: label" in block
