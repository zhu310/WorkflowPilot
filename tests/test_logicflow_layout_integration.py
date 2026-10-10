from __future__ import annotations

import json
from pathlib import Path


FRONTEND_MAIN = Path("demo/frontend/src/main.ts")
PACKAGE_JSON = Path("demo/frontend/package.json")


def source() -> str:
    return FRONTEND_MAIN.read_text(encoding="utf-8")


def package_json() -> dict:
    return json.loads(PACKAGE_JSON.read_text(encoding="utf-8"))


def test_logicflow_layout_dependency_is_declared() -> None:
    dependencies = package_json()["dependencies"]

    assert "@logicflow/layout" in dependencies
    assert "elkjs" in dependencies


def test_official_layout_plugins_are_registered_with_logicflow() -> None:
    text = source()

    assert "import { Dagre, ElkLayout } from '@logicflow/layout';" in text
    assert "plugins: [Dagre, ElkLayout]" in text
    assert "pluginsOptions" in text
    assert "elkLayout" in text
    assert "rankdir: 'LR'" in text


def test_layout_runtime_prefers_native_elk_sections_and_keeps_legacy_fallback() -> None:
    text = source()

    assert "function runOfficialLayout" in text
    assert "const DEFAULT_GEOMETRY_ENGINE: GeometryEngine = 'native_elk'" in text
    assert "function runNativeElkLayout" in text
    assert "nativeElk.layout" in text
    assert "applyNativeElkLayoutToGraphData" in text
    assert "runLegacyOfficialLayout" in text
    assert "extensions.elkLayout?.layout" in text
    assert "extensions.dagre?.layout" in text
    assert "AI_NATIVE_ELK_LAYOUT_FAILED" in text
    assert "AI_FLOW_LAYOUT_UNAVAILABLE" in text
    assert "AI_FLOW_LAYOUT_FAILED" in text


def test_layout_runs_on_initial_reset_manual_and_server_state_paths() -> None:
    text = source()

    assert "render(initialGraph, 'initial')" in text
    assert "render(initialGraph, 'reset')" in text
    assert "runOfficialLayout('manual')" in text
    assert "function applyServerState" in text
    assert "if (result.graph) render(result.graph)" in text
    assert "else if (result.final_graph) render(result.final_graph)" in text
