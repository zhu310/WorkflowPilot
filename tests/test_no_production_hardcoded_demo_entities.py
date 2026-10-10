from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

FORBIDDEN_SNIPPETS = (
    "结果判断",
    "人工复核",
    "生成报告",
    "发布结果",
    "异常处理节点",
    "e_decision_report",
    "e_decision_human",
    "e_report_publish",
    "e_publish_end",
    "e_report_human",
    "n_report",
    "n_publish",
    "n_human",
    "n_end",
    "这个分支转人工复核",
    "这个分支转结束",
    "在结束前增加异常处理节点",
    "把生成报告节点名称改成生成正式报告",
    "优化一下这个流程",
)

SCAN_ROOTS = (ROOT / "src", ROOT / "demo")
EXCLUDED_PARTS = {
    "dist",
    "node_modules",
    "__pycache__",
}
EXCLUDED_SUFFIXES = {
    ".md",
    ".map",
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".ico",
}


def production_files():
    for scan_root in SCAN_ROOTS:
        for path in scan_root.rglob("*"):
            if not path.is_file():
                continue
            relative_parts = set(path.relative_to(ROOT).parts)
            if relative_parts & EXCLUDED_PARTS:
                continue
            if path.suffix.lower() in EXCLUDED_SUFFIXES:
                continue
            yield path


def strip_allowed_demo_initial_graph(path: Path, text: str) -> str:
    if path.as_posix().endswith("demo/frontend/src/main.ts"):
        start = text.find("const initialGraph: WorkflowGraph =")
        end = text.find("const nodeTypeMap", start)
        if start != -1 and end != -1:
            return text[:start] + text[end:]
    return text


def test_no_demo_specific_entities_are_hardcoded_in_production_logic() -> None:
    violations: list[str] = []
    for path in production_files():
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        text = strip_allowed_demo_initial_graph(path, text)
        for snippet in FORBIDDEN_SNIPPETS:
            if snippet in text:
                violations.append(f"{path.relative_to(ROOT)} contains {snippet!r}")

    assert violations == []
