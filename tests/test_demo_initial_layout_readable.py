from __future__ import annotations

import re
from pathlib import Path


FRONTEND_MAIN = Path("demo/frontend/src/main.ts")


def source() -> str:
    return FRONTEND_MAIN.read_text(encoding="utf-8")


def node_position(node_id: str) -> tuple[int, int]:
    pattern = re.compile(
        r"\{\s*id:\s*'" + re.escape(node_id) + r"'.*?x:\s*(?P<x>\d+),\s*y:\s*(?P<y>\d+)",
        re.DOTALL,
    )
    match = pattern.search(source())
    assert match, f"missing node {node_id}"
    return int(match.group("x")), int(match.group("y"))


def test_demo_reset_layout_gives_report_human_publish_end_enough_space() -> None:
    report = node_position("n_report")
    human = node_position("n_human")
    publish = node_position("n_publish")
    end = node_position("n_end")

    assert abs(report[1] - human[1]) >= 340
    assert abs(publish[0] - report[0]) >= 240
    assert abs(end[0] - publish[0]) >= 240
    assert abs(end[1] - publish[1]) >= 120


def test_demo_reset_layout_keeps_branch_targets_visually_separated() -> None:
    decision = node_position("n_approval_decision")
    report = node_position("n_report")
    human = node_position("n_human")

    assert abs(report[1] - human[1]) >= 340
    assert abs(report[0] - decision[0]) >= 220
    assert abs(human[0] - decision[0]) >= 220
