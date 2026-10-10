from pathlib import Path


FRONTEND_MAIN = Path("demo/frontend/src/main.ts")


def source() -> str:
    return FRONTEND_MAIN.read_text(encoding="utf-8")


def workflow_block() -> str:
    text = source()
    return text[text.index("function workflowToLogicFlow") : text.index("function readLogicFlowSelection")]


def test_same_endpoint_label_collision_is_resolved_by_display_aggregation() -> None:
    block = workflow_block()

    assert "text: label" in block
    assert "combinedEdgeLabel(edges)" in block
    assert "underlying_edges" in block
    assert "pointsList" not in block


def test_app_layer_does_not_reintroduce_path_or_label_offsets() -> None:
    text = source()
    block = workflow_block()

    forbidden = ["pathOffset", "labelOffset", "edgeText(", "tangentShift", "labelInsideAnyNode"]
    for token in forbidden:
        assert token not in text
    assert "type: 'polyline'" in block


def test_display_aggregation_is_generic_not_branch_name_specific() -> None:
    block = workflow_block()

    assert "success" not in block
    assert "failure" not in block
    assert "exception" not in block
    assert "groupEdgesByEndpoints" in block


def test_unlabeled_edges_do_not_receive_synthetic_branch_text() -> None:
    text = source()
    label_block = text[text.index("function userVisibleEdgeLabel") : text.index("function combinedEdgeLabel")]
    combined_block = text[text.index("function combinedEdgeLabel") : text.index("function displayEdgeId(")]

    assert "return ''" in label_block
    assert "branch ${" not in label_block
    assert "'branch'" not in label_block
    assert ".filter(Boolean)" in combined_block
    assert "new Set(labels)" in combined_block
