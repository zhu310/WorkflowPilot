from __future__ import annotations

from copy import deepcopy
from typing import Any


CLARIFICATION_KINDS = {
    "ambiguous_node_reference",
    "missing_node_reference",
    "ambiguous_edge_reference",
    "missing_edge_reference",
    "stale_node_reference",
    "stale_edge_reference",
}
INTERNAL_TOKENS = (
    "o1",
    "o2",
    "property_equals",
    "route_to",
    "right_of",
    "grounding",
    "executable binding",
    "semantic relation could not be lowered",
    "supported semantic relation",
    "supported semantic state",
    "relation intent was not lowered",
    "relation intent",
    "not lowered",
    "lowering",
    "semantic goal",
    "semantic understanding",
    "semantic goal contract",
    "object references must preserve phrases from the user instruction",
    "semantic understanding must not pre-resolve graph objects",
    "must preserve phrases",
    "pre-resolve graph objects",
    "validator",
    "deterministic",
    "multiple graph edges match reference",
    "no graph edge matches reference",
    "no graph node matches reference",
    "ambiguous_edge_reference",
    "missing_edge_reference",
    "ambiguous_node_reference",
    "missing_node_reference",
    "validation",
    "workflow_ir",
    "inserted",
    "connection:",
    "branch:",
    "between",
)
TECHNICAL_REASON_MARKERS = (
    "_",
    ":",
    "{",
    "}",
    "[",
    "]",
    "traceback",
    "exception",
    "failed",
    "error",
    "could not",
    "did not",
    "must preserve",
    "not produce",
    "not found",
    "matches reference",
)

ZH_BRANCH = "\u5206\u652f"
ZH_CONNECTION = "\u8fde\u7ebf"
ZH_INSERT_BETWEEN = "\u5df2\u5728\u201c{source}\u201d\u548c\u201c{target}\u201d\u4e4b\u95f4\u65b0\u589e\u201c{label}\u201d"
ZH_REROUTE_CONNECTION = "\u5df2\u5c06\u201c{source} -> {old_target}\u201d\u7684\u8fde\u7ebf\u6539\u63a5\u5230\u201c{new_target}\u201d"
ZH_REROUTE_BRANCH = "\u5df2\u5c06\u201c{label}\u201d\u5206\u652f\u6539\u63a5\u5230\u201c{new_target}\u201d"
ZH_AMBIGUOUS_EDGE_REASON = "\u6ca1\u80fd\u786e\u5b9a\u4f60\u8bf4\u7684\u662f\u54ea\u4e00\u6761\u5206\u652f\uff0c\u8bf7\u4ece\u5019\u9009\u4e2d\u9009\u62e9\u3002"
ZH_MISSING_EDGE_REASON = "\u5f53\u524d\u65e0\u6cd5\u786e\u5b9a\u4f60\u8981\u4fee\u6539\u54ea\u6761\u5206\u652f\u3002\u8bf7\u5148\u5728\u753b\u5e03\u4e2d\u9009\u62e9\u4e00\u6761\u5206\u652f\uff0c\u6216\u63cf\u8ff0\u5b83\u7684\u8d77\u70b9\u548c\u7ec8\u70b9\u3002"
ZH_AMBIGUOUS_NODE_REASON = "\u6ca1\u80fd\u552f\u4e00\u786e\u5b9a\u4f60\u8bf4\u7684\u662f\u54ea\u4e2a\u8282\u70b9\uff0c\u8bf7\u4ece\u5019\u9009\u4e2d\u9009\u62e9\u3002"
ZH_MISSING_NODE_REASON = "\u5f53\u524d\u56fe\u4e2d\u627e\u4e0d\u5230\u4f60\u63cf\u8ff0\u7684\u8282\u70b9\u3002\u8bf7\u6362\u4e00\u4e2a\u66f4\u660e\u786e\u7684\u540d\u79f0\u6216\u5148\u5728\u753b\u5e03\u4e2d\u9009\u62e9\u5b83\u3002"
ZH_UNIQUE_BINDING_REASON = "\u8fd8\u4e0d\u80fd\u552f\u4e00\u786e\u5b9a\u8981\u4fee\u6539\u7684\u5bf9\u8c61\uff0c\u8bf7\u4ece\u5019\u9009\u4e2d\u9009\u62e9\u3002"
ZH_NEEDS_CONFIRMATION_REASON = "\u5f53\u524d\u76ee\u6807\u8fd8\u9700\u8981\u8fdb\u4e00\u6b65\u786e\u8ba4\u3002"
ZH_UNSUPPORTED_REASON = "\u5f53\u524d\u8bf7\u6c42\u8fd8\u4e0d\u80fd\u5b89\u5168\u8f6c\u6362\u4e3a\u53ef\u6267\u884c\u7684\u6d41\u7a0b\u4fee\u6539\u3002"
ZH_EXECUTION_FAILED_REASON = "\u6267\u884c\u4fee\u6539\u65f6\u9047\u5230\u95ee\u9898\uff0c\u6d41\u7a0b\u56fe\u6ca1\u6709\u88ab\u66f4\u65b0\u3002\u8bf7\u6362\u4e00\u79cd\u66f4\u660e\u786e\u7684\u8bf4\u6cd5\u91cd\u8bd5\u3002"
ZH_UNKNOWN_REASON = "\u8fd9\u4e2a\u76ee\u6807\u6682\u65f6\u65e0\u6cd5\u5b89\u5168\u5b8c\u6210\uff0c\u9700\u8981\u66f4\u660e\u786e\u7684\u4fee\u6539\u5bf9\u8c61\u6216\u76ee\u6807\u3002"
ZH_UNKNOWN_CLARIFICATION = "\u76ee\u524d\u8fd8\u65e0\u6cd5\u786e\u5b9a\u4f60\u5e0c\u671b\u4fee\u6539\u7684\u5bf9\u8c61\uff0c\u8bf7\u6362\u4e00\u4e2a\u66f4\u660e\u786e\u7684\u63cf\u8ff0\uff0c\u6216\u5148\u5728\u753b\u5e03\u4e2d\u9009\u62e9\u76ee\u6807\u3002"
ZH_UNKNOWN_UNSUPPORTED = "\u5f53\u524d\u8fd9\u4e2a\u4fee\u6539\u6682\u65f6\u65e0\u6cd5\u5b89\u5168\u6267\u884c\uff0c\u9700\u8981\u8fdb\u4e00\u6b65\u786e\u8ba4\u6216\u6362\u4e00\u79cd\u66f4\u660e\u786e\u7684\u4fee\u6539\u65b9\u5f0f\u3002"
ZH_PARTIAL_INCOMPLETE = "\u8fd8\u6709\u4e00\u4e2a\u4fee\u6539\u76ee\u6807\u672a\u5b8c\u6210"
OUTER_QUOTE_PAIRS = {
    '"': '"',
    "'": "'",
    "\u201c": "\u201d",
    "\u2018": "\u2019",
    "\u300c": "\u300d",
    "\u300e": "\u300f",
}


def _strip_outer_reference_quotes(value: Any) -> str:
    text = str(value or "").strip()
    if len(text) >= 2:
        closing = OUTER_QUOTE_PAIRS.get(text[0])
        if closing and text[-1] == closing:
            inner = text[1:-1].strip()
            if inner:
                return inner
    return text


def _nodes_by_id(graph: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    graph = graph if isinstance(graph, dict) else {}
    return {
        node["id"]: node
        for node in graph.get("nodes", [])
        if isinstance(node, dict) and isinstance(node.get("id"), str)
    }


def _edges_by_id(graph: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    graph = graph if isinstance(graph, dict) else {}
    return {
        edge["id"]: edge
        for edge in graph.get("edges", [])
        if isinstance(edge, dict) and isinstance(edge.get("id"), str)
    }


def _edge_display(edge: dict[str, Any], nodes: dict[str, dict[str, Any]]) -> str:
    source_id = edge.get("source")
    target_id = edge.get("target")
    source = _strip_outer_reference_quotes(nodes.get(source_id, {}).get("label")) if isinstance(source_id, str) else ""
    target = _strip_outer_reference_quotes(nodes.get(target_id, {}).get("label")) if isinstance(target_id, str) else ""
    label = _strip_outer_reference_quotes(edge.get("branch_label") or edge.get("label") or edge.get("condition"))
    route = f"{source} -> {target}" if source or target else ""
    if label:
        return f"{label} {ZH_BRANCH}\uff1a{route}"
    return f"{ZH_CONNECTION}\uff1a{route}" if route else ZH_CONNECTION


def _node_label(graph: dict[str, Any] | None, node_id: Any) -> str:
    if not isinstance(node_id, str):
        return ""
    node = _nodes_by_id(graph).get(node_id)
    return _strip_outer_reference_quotes(node.get("label")) if node else ""


def _edge_label(edge: dict[str, Any]) -> str:
    return _strip_outer_reference_quotes(edge.get("branch_label") or edge.get("label")) or ZH_CONNECTION


def _edge_readable_name(edge: dict[str, Any], graph: dict[str, Any] | None) -> str:
    label = _strip_outer_reference_quotes(edge.get("branch_label") or edge.get("label"))
    if label:
        return f"{label} {ZH_BRANCH}"
    nodes = _nodes_by_id(graph)
    source_id = edge.get("source")
    target_id = edge.get("target")
    source = _strip_outer_reference_quotes(nodes.get(source_id, {}).get("label")) if isinstance(source_id, str) else ""
    target = _strip_outer_reference_quotes(nodes.get(target_id, {}).get("label")) if isinstance(target_id, str) else ""
    if source or target:
        return f"{ZH_CONNECTION}\uff1a{source} -> {target}"
    return "流程连接"


def _edge_route_label(graph: dict[str, Any] | None, source_id: Any, target_id: Any) -> str:
    source = _node_label(graph, source_id)
    target = _node_label(graph, target_id)
    if source and target:
        return f"{source} -> {target}"
    return source or target or ZH_CONNECTION


def _looks_internal_or_technical(text: str) -> bool:
    lowered = text.lower()
    if any(token in lowered for token in INTERNAL_TOKENS):
        return True
    if any(marker in lowered for marker in TECHNICAL_REASON_MARKERS):
        return True
    return False


def _readable_issue_reason(issue: dict[str, Any]) -> str:
    kind = str(issue.get("kind") or "")
    if kind == "ambiguous_edge_reference":
        return ZH_AMBIGUOUS_EDGE_REASON
    if kind == "missing_edge_reference":
        return ZH_MISSING_EDGE_REASON
    if kind == "ambiguous_node_reference":
        return ZH_AMBIGUOUS_NODE_REASON
    if kind == "missing_node_reference":
        return ZH_MISSING_NODE_REASON
    if kind in {"stale_edge_reference", "stale_node_reference"}:
        return "\u4f60\u9009\u62e9\u6216\u5f15\u7528\u7684\u5bf9\u8c61\u5df2\u4e0d\u5728\u5f53\u524d\u7248\u672c\u4e2d\uff0c\u8bf7\u91cd\u65b0\u9009\u62e9\u6216\u63cf\u8ff0\u3002"
    if kind in {"clarification_needed", "partial_spec", "unresolved_semantic_goals"}:
        return ZH_NEEDS_CONFIRMATION_REASON
    if kind == "ambiguous_structural_boundary":
        return ZH_UNKNOWN_UNSUPPORTED
    if kind:
        return _readable_reason(issue.get("reason") or issue.get("message") or kind)
    return _readable_reason(issue.get("reason") or issue.get("message") or "")


def _readable_issue_question(issue: dict[str, Any], graph: dict[str, Any] | None) -> str:
    kind = str(issue.get("kind") or "")
    reference = str(issue.get("reference") or "").strip()
    if kind == "ambiguous_edge_reference":
        if len(_candidate_rows(issue, graph)) >= 2:
            return "\u627e\u5230\u4e86\u591a\u6761\u53ef\u80fd\u7684\u5206\u652f\uff0c\u8bf7\u4ece\u5019\u9009\u4e2d\u9009\u62e9\u4e00\u6761\u3002"
        return "\u8fd8\u4e0d\u80fd\u552f\u4e00\u786e\u5b9a\u8981\u4fee\u6539\u54ea\u6761\u5206\u652f\u3002\u8bf7\u5148\u5728\u753b\u5e03\u4e2d\u9009\u62e9\u4e00\u6761\u5206\u652f\uff0c\u6216\u63cf\u8ff0\u5b83\u7684\u8d77\u70b9\u548c\u7ec8\u70b9\u3002"
    if kind == "missing_edge_reference":
        if reference:
            return f"\u5f53\u524d\u65e0\u6cd5\u786e\u5b9a\u201c{reference}\u201d\u5bf9\u5e94\u54ea\u6761\u5206\u652f\u3002\u8bf7\u5148\u5728\u753b\u5e03\u4e2d\u9009\u62e9\u4e00\u6761\u5206\u652f\uff0c\u6216\u63cf\u8ff0\u5b83\u7684\u8d77\u70b9\u548c\u7ec8\u70b9\u3002"
        return ZH_MISSING_EDGE_REASON
    if kind == "ambiguous_node_reference":
        if len(_candidate_rows(issue, graph)) >= 2:
            return "\u627e\u5230\u4e86\u591a\u4e2a\u53ef\u80fd\u7684\u8282\u70b9\uff0c\u8bf7\u4ece\u5019\u9009\u4e2d\u9009\u62e9\u4e00\u4e2a\u3002"
        return "\u8fd8\u4e0d\u80fd\u552f\u4e00\u786e\u5b9a\u8981\u4fee\u6539\u54ea\u4e2a\u8282\u70b9\u3002\u8bf7\u6362\u4e00\u4e2a\u66f4\u660e\u786e\u7684\u540d\u79f0\uff0c\u6216\u5148\u5728\u753b\u5e03\u4e2d\u9009\u62e9\u5b83\u3002"
    if kind == "missing_node_reference":
        if reference:
            return f"\u5f53\u524d\u56fe\u4e2d\u6ca1\u6709\u627e\u5230\u201c{reference}\u201d\u5bf9\u5e94\u7684\u8282\u70b9\u3002\u8bf7\u6362\u4e00\u4e2a\u66f4\u660e\u786e\u7684\u540d\u79f0\uff0c\u6216\u5148\u5728\u753b\u5e03\u4e2d\u9009\u62e9\u5b83\u3002"
        return ZH_MISSING_NODE_REASON
    if kind == "route_same_source_target_confirmation":
        return _same_target_confirmation_question(issue, graph)
    if kind == "layout_collision":
        return _layout_collision_question(issue, graph)
    return _readable_issue_reason(issue)


def _insert_split_summaries(diff: dict[str, Any], graph: dict[str, Any] | None) -> tuple[list[str], set[str], set[str], set[str]]:
    added_nodes = [row for row in diff.get("added_nodes", []) if isinstance(row, dict) and isinstance(row.get("id"), str)]
    added_edges = [row for row in diff.get("added_edges", []) if isinstance(row, dict)]
    changed_edges = [row for row in diff.get("changed_edges", []) if isinstance(row, dict)]
    summaries: list[str] = []
    grouped_node_ids: set[str] = set()
    grouped_added_edge_ids: set[str] = set()
    grouped_changed_edge_ids: set[str] = set()
    added_by_source: dict[Any, list[dict[str, Any]]] = {}
    for edge in added_edges:
        added_by_source.setdefault(edge.get("source"), []).append(edge)

    for node in added_nodes:
        node_id = node.get("id")
        outgoing = added_by_source.get(node_id, [])
        if not isinstance(node_id, str) or len(outgoing) != 1:
            continue
        continuation = outgoing[0]
        old_target = continuation.get("target")
        if not isinstance(old_target, str):
            continue
        matching_changed = []
        for row in changed_edges:
            changes = row.get("changes") if isinstance(row.get("changes"), dict) else {}
            target_change = changes.get("target")
            if (
                isinstance(target_change, dict)
                and target_change.get("before") == old_target
                and target_change.get("after") == node_id
            ):
                matching_changed.append(row)
        if len(matching_changed) != 1:
            continue
        changed = matching_changed[0]
        changed_edge = _edges_by_id(graph).get(changed.get("id"), {})
        source_id = changed_edge.get("source")
        node_label = _public_label(node.get("label"), "\u65b0\u8282\u70b9")
        summaries.append(
            ZH_INSERT_BETWEEN.format(
                source=_node_label(graph, source_id),
                target=_node_label(graph, old_target),
                label=node_label,
            )
        )
        grouped_node_ids.add(node_id)
        if isinstance(continuation.get("id"), str):
            grouped_added_edge_ids.add(continuation["id"])
        if isinstance(changed.get("id"), str):
            grouped_changed_edge_ids.add(changed["id"])
    return summaries, grouped_node_ids, grouped_added_edge_ids, grouped_changed_edge_ids


def _readable_reason(reason: Any) -> str:
    text = str(reason or "")
    lowered = text.lower()
    if not text:
        return ""
    if "no graph edge matches reference" in lowered:
        return ZH_MISSING_EDGE_REASON
    if "no graph node matches reference" in lowered:
        return ZH_MISSING_NODE_REASON
    if "multiple graph edges match reference" in lowered:
        return ZH_AMBIGUOUS_EDGE_REASON
    if "multiple graph nodes match reference" in lowered:
        return ZH_AMBIGUOUS_NODE_REASON
    if "grounding did not produce a unique executable binding" in lowered:
        return ZH_UNIQUE_BINDING_REASON
    if "supported semantic relation could not be lowered" in lowered or "semantic relation could not be lowered" in lowered:
        return ZH_NEEDS_CONFIRMATION_REASON
    if "grounding did not produce a unique executable binding" in lowered:
        return "没能唯一确定你指的是哪一个对象，请从候选中选择。"
    if "unsupported" in lowered or "not supported" in lowered:
        return "当前能力还不能安全执行这个目标。"
    if "execution" in lowered and ("failed" in lowered or "error" in lowered):
        return ZH_EXECUTION_FAILED_REASON
    if _looks_internal_or_technical(text):
        return ZH_UNKNOWN_REASON
    if any(token in lowered for token in INTERNAL_TOKENS):
        return "当前目标还需要进一步确认。"
    return text


def _readable_question(question: Any, *, fallback_for_unsafe_lowering: bool = False) -> str:
    text = str(question or "")
    if not text:
        return ""
    lowered = text.lower()
    if "semantic relation could not be lowered" in lowered and "supported semantic relation could not be lowered" not in lowered:
        return "\u5f53\u524d\u76ee\u6807\u8fd8\u4e0d\u80fd\u5b89\u5168\u6267\u884c\uff0c\u8bf7\u8865\u5145\u66f4\u660e\u786e\u7684\u4fee\u6539\u65b9\u5f0f\u3002"
    readable = _readable_reason(text)
    if readable != text:
        return readable
    if fallback_for_unsafe_lowering and _looks_internal_or_technical(text):
        return ZH_UNKNOWN_REASON
    if fallback_for_unsafe_lowering and any(token in lowered for token in INTERNAL_TOKENS):
        return "\u5f53\u524d\u76ee\u6807\u8fd8\u9700\u8981\u8fdb\u4e00\u6b65\u786e\u8ba4\u3002"
    return text


def _readable_issue_reason(issue: dict[str, Any]) -> str:
    kind = str(issue.get("kind") or "")
    if kind == "ambiguous_edge_reference":
        return ZH_AMBIGUOUS_EDGE_REASON
    if kind == "missing_edge_reference":
        return ZH_MISSING_EDGE_REASON
    if kind == "ambiguous_node_reference":
        return ZH_AMBIGUOUS_NODE_REASON
    if kind == "missing_node_reference":
        return ZH_MISSING_NODE_REASON
    if kind in {"stale_edge_reference", "stale_node_reference"}:
        return "\u4f60\u9009\u62e9\u6216\u5f15\u7528\u7684\u5bf9\u8c61\u5df2\u4e0d\u5728\u5f53\u524d\u7248\u672c\u4e2d\uff0c\u8bf7\u91cd\u65b0\u9009\u62e9\u6216\u63cf\u8ff0\u3002"
    if kind in {"clarification_needed", "partial_spec", "unresolved_semantic_goals"}:
        return ZH_NEEDS_CONFIRMATION_REASON
    if kind == "ambiguous_structural_boundary":
        return ZH_UNKNOWN_UNSUPPORTED
    if kind in {"semantic_goal_contract_violation", "invalid_semantic_goal", "invalid_existing_reference", "invalid_edge_reference", "invalid_new_entity"}:
        return ZH_UNKNOWN_UNSUPPORTED
    if "lowering" in kind or "unsupported" in kind:
        return ZH_UNKNOWN_UNSUPPORTED
    if "execution" in kind:
        return ZH_EXECUTION_FAILED_REASON
    return _readable_reason(issue.get("reason") or issue.get("message") or kind)


def _readable_reason(reason: Any) -> str:
    text = str(reason or "")
    lowered = text.lower()
    if not text:
        return ""
    if "no graph edge matches reference" in lowered:
        return ZH_MISSING_EDGE_REASON
    if "no graph node matches reference" in lowered:
        return ZH_MISSING_NODE_REASON
    if "multiple graph edges match reference" in lowered:
        return ZH_AMBIGUOUS_EDGE_REASON
    if "multiple graph nodes match reference" in lowered:
        return ZH_AMBIGUOUS_NODE_REASON
    if "grounding did not produce a unique executable binding" in lowered:
        return ZH_UNIQUE_BINDING_REASON
    if "supported semantic relation could not be lowered" in lowered or "semantic relation could not be lowered" in lowered:
        return ZH_UNKNOWN_UNSUPPORTED
    if "supported semantic state could not be lowered" in lowered or "relation intent was not lowered" in lowered:
        return ZH_UNKNOWN_UNSUPPORTED
    if "object references must preserve phrases from the user instruction" in lowered:
        return ZH_UNKNOWN_UNSUPPORTED
    if "semantic understanding must not pre-resolve graph objects" in lowered:
        return ZH_UNKNOWN_UNSUPPORTED
    if "unsupported" in lowered or "not supported" in lowered:
        return ZH_UNKNOWN_UNSUPPORTED
    if "execution" in lowered and ("failed" in lowered or "error" in lowered):
        return ZH_EXECUTION_FAILED_REASON
    return ZH_UNKNOWN_REASON


def _readable_question(question: Any, *, fallback_for_unsafe_lowering: bool = False) -> str:
    text = str(question or "")
    if not text:
        return ""
    lowered = text.lower()
    if "semantic relation could not be lowered" in lowered and "supported semantic relation could not be lowered" not in lowered:
        return ZH_UNKNOWN_UNSUPPORTED
    readable = _readable_reason(text)
    if readable and readable != ZH_UNKNOWN_REASON:
        return readable
    if fallback_for_unsafe_lowering and _looks_internal_or_technical(text):
        return ZH_UNKNOWN_REASON
    return text


def _readable_description(row: dict[str, Any]) -> str:
    description = str(row.get("description") or row.get("id") or "")
    lowered = description.lower()
    if _looks_internal_or_technical(description):
        if str(row.get("status") or "") == "executed":
            return "\u5df2\u5b8c\u6210\u4e00\u4e2a\u4fee\u6539\u76ee\u6807"
        return "\u8fd8\u6709\u4e00\u4e2a\u76ee\u6807\u9700\u8981\u786e\u8ba4"
    if any(token in lowered for token in INTERNAL_TOKENS):
        return "还有一个目标需要确认"
    return description or "未完成的目标"


def _same_target_confirmation_question(issue: dict[str, Any], graph: dict[str, Any] | None) -> str:
    source = _node_label(graph, issue.get("source"))
    target = _node_label(graph, issue.get("target"))
    current = str(issue.get("relation_type") or "当前")
    existing = str(issue.get("conflicting_branch_label") or issue.get("conflicting_relation_type") or "另一条")
    if source and target:
        return f"“{source}”节点已经有 {existing} 分支指向“{target}”。是否仍要把 {current} 分支也指向“{target}”？"
    return "该目标已经被同一来源的其他分支指向。是否仍要继续这次分支改向？"


def _layout_collision_question(issue: dict[str, Any], graph: dict[str, Any] | None) -> str:
    target = _node_label(graph, issue.get("target"))
    conflict = _node_label(graph, issue.get("conflicting_node"))
    if target and conflict:
        return f"移动“{target}”后会与“{conflict}”重叠，请指定更明确的目标位置或参考节点。"
    return "移动后会与已有节点重叠，请指定更明确的目标位置或参考节点。"



def _public_label(value: Any, fallback: str) -> str:
    text = _strip_outer_reference_quotes(value)
    lowered = text.lower()
    if not text or any(token in lowered for token in INTERNAL_TOKENS):
        return fallback
    return text


def _public_node_label(graph: dict[str, Any] | None, node_id: Any, fallback: str = '\u76ee\u6807\u8282\u70b9') -> str:
    return _public_label(_node_label(graph, node_id), fallback)


def _edge_pairs(graph: dict[str, Any] | None) -> set[tuple[str, str]]:
    return {
        (edge["source"], edge["target"])
        for edge in (graph.get("edges", []) if isinstance(graph, dict) else [])
        if isinstance(edge, dict) and isinstance(edge.get("source"), str) and isinstance(edge.get("target"), str)
    }


def _has_path(graph: dict[str, Any] | None, source: str, target: str) -> bool:
    edges = _edge_pairs(graph)
    adjacency: dict[str, set[str]] = {}
    for left, right in edges:
        adjacency.setdefault(left, set()).add(right)
    seen: set[str] = set()
    stack = [source]
    while stack:
        node_id = stack.pop()
        if node_id == target:
            return True
        if node_id in seen:
            continue
        seen.add(node_id)
        stack.extend(sorted(adjacency.get(node_id, set()) - seen))
    return False


def _binding_node_id(goal: dict[str, Any], role: str) -> str | None:
    for binding in goal.get("bindings", []) if isinstance(goal.get("bindings"), list) else []:
        if (
            isinstance(binding, dict)
            and binding.get("role") == role
            and binding.get("binding_type") == "node"
            and isinstance(binding.get("id"), str)
        ):
            return binding["id"]
    return None


def _executed_subgoal_ids(partial: dict[str, Any]) -> set[str]:
    return {
        str(row.get("id"))
        for row in partial.get("subgoals", [])
        if isinstance(row, dict) and row.get("status") == "executed" and row.get("id")
    }


def _suppressed_relation_subgoal_ids(grounded: dict[str, Any]) -> set[str]:
    return {
        str(row.get("source_subgoal_id"))
        for row in grounded.get("suppressed_legacy_relation_goals", [])
        if isinstance(row, dict) and row.get("source_subgoal_id")
    }


def _common_successor(graph: dict[str, Any] | None, left: str, right: str) -> str | None:
    outgoing: dict[str, set[str]] = {}
    for source, target in _edge_pairs(graph):
        outgoing.setdefault(source, set()).add(target)
    common = sorted(outgoing.get(left, set()) & outgoing.get(right, set()))
    return common[0] if len(common) == 1 else None


def _structural_relation_summaries(
    specification: dict[str, Any] | None,
    graph: dict[str, Any] | None,
    diff: dict[str, Any],
) -> list[str]:
    if not (diff.get("added_edges") or diff.get("removed_edges") or diff.get("changed_edges")):
        return []
    partial = _partial_execution(specification)
    executed_ids = _executed_subgoal_ids(partial)
    if not executed_ids:
        return []
    grounded = _grounded_goal(specification)
    before_graph = _before_graph(specification)
    suppressed_ids = _suppressed_relation_subgoal_ids(grounded)
    summaries: list[str] = []
    seen: set[tuple[str, str, str]] = set()
    for goal in grounded.get("semantic_goals", []) if isinstance(grounded.get("semantic_goals"), list) else []:
        if not isinstance(goal, dict):
            continue
        source_subgoal_id = str(goal.get("source_subgoal_id") or "")
        if not source_subgoal_id or source_subgoal_id not in executed_ids or source_subgoal_id in suppressed_ids:
            continue
        kind = str(goal.get("kind") or "")
        relation = str(goal.get("relation") or "")
        if kind == "parallel" and relation == "parallel_with":
            left_id = _binding_node_id(goal, "left")
            right_id = _binding_node_id(goal, "right")
            if not (left_id and right_id):
                continue
            if (left_id, right_id) in _edge_pairs(graph) or (right_id, left_id) in _edge_pairs(graph):
                continue
            key = ("parallel", left_id, right_id)
            if key in seen:
                continue
            seen.add(key)
            left = _public_node_label(graph, left_id)
            right = _public_node_label(graph, right_id)
            join_id = _common_successor(graph, left_id, right_id)
            if join_id:
                join = _public_node_label(graph, join_id)
                summaries.append(f"\u5df2\u5c06\u201c{left}\u201d\u548c\u201c{right}\u201d\u8c03\u6574\u4e3a\u5e76\u884c\u7ed3\u6784\uff0c\u5e76\u5728\u201c{join}\u201d\u5904\u6c47\u5408")
            else:
                summaries.append(f"\u5df2\u5c06\u201c{left}\u201d\u548c\u201c{right}\u201d\u8c03\u6574\u4e3a\u5e76\u884c\u7ed3\u6784")
        elif kind == "ordering" and relation in {"before", "after"}:
            subject_id = _binding_node_id(goal, "subject")
            anchor_id = _binding_node_id(goal, "anchor")
            if not (subject_id and anchor_id):
                continue
            before_id, after_id = (subject_id, anchor_id) if relation == "before" else (anchor_id, subject_id)
            if not _has_path(graph, before_id, after_id) or _has_path(before_graph, before_id, after_id):
                continue
            key = ("ordering", before_id, after_id)
            if key in seen:
                continue
            seen.add(key)
            before_label = _public_node_label(graph, before_id)
            after_label = _public_node_label(graph, after_id)
            summaries.append(f"\u5df2\u5c06\u201c{before_label}\u201d\u8c03\u6574\u5230\u201c{after_label}\u201d\u4e4b\u524d")
        elif kind == "connection" and relation in {"connected_to", ""}:
            source_id = _binding_node_id(goal, "source")
            target_id = _binding_node_id(goal, "target")
            if not (source_id and target_id and (source_id, target_id) in _edge_pairs(graph)):
                continue
            key = ("connection", source_id, target_id)
            if key in seen:
                continue
            seen.add(key)
            source = _public_node_label(graph, source_id)
            target = _public_node_label(graph, target_id)
            summaries.append(f"\u5df2\u5efa\u7acb\u201c{source}\u201d\u5230\u201c{target}\u201d\u7684\u76f4\u63a5\u8fde\u63a5")
    if not summaries:
        for relation_fact in grounded.get("relation_facts", []) if isinstance(grounded.get("relation_facts"), list) else []:
            if not isinstance(relation_fact, dict) or relation_fact.get("kind") != "precedence":
                continue
            members = relation_fact.get("members") if isinstance(relation_fact.get("members"), list) else []
            if len(members) < 2:
                continue
            for member in members:
                if not isinstance(member, dict):
                    continue
                source_binding = member.get("source_binding") if isinstance(member.get("source_binding"), dict) else {}
                target_binding = member.get("target_binding") if isinstance(member.get("target_binding"), dict) else {}
                source_id = source_binding.get("id")
                target_id = target_binding.get("id")
                if not (
                    isinstance(source_id, str)
                    and isinstance(target_id, str)
                    and _has_path(graph, source_id, target_id)
                    and not _has_path(before_graph, source_id, target_id)
                ):
                    continue
                before_label = _public_node_label(graph, source_id)
                after_label = _public_node_label(graph, target_id)
                summaries.append(f"\u5df2\u5c06\u201c{before_label}\u201d\u8c03\u6574\u5230\u201c{after_label}\u201d\u4e4b\u524d")
                break
            if summaries:
                break
    return summaries


def _diff_summaries(diff: dict[str, Any], graph: dict[str, Any] | None) -> list[str]:
    summaries: list[str] = []
    grouped_summaries, grouped_node_ids, _grouped_added_edge_ids, grouped_changed_edge_ids = _insert_split_summaries(diff, graph)
    summaries.extend(grouped_summaries)
    for row in diff.get('added_nodes', []) if isinstance(diff.get('added_nodes'), list) else []:
        if isinstance(row, dict):
            if row.get('id') in grouped_node_ids:
                continue
            label = _public_label(row.get('label') or row.get('id'), '\u65b0\u8282\u70b9')
            summaries.append(f'\u65b0\u589e\u201c{label}\u201d\u8282\u70b9')
    for row in diff.get('removed_nodes', []) if isinstance(diff.get('removed_nodes'), list) else []:
        if isinstance(row, dict):
            label = _public_label(row.get('label') or row.get('id'), '\u8282\u70b9')
            summaries.append(f'\u5220\u9664\u201c{label}\u201d\u8282\u70b9')
    for row in diff.get('changed_nodes', []) if isinstance(diff.get('changed_nodes'), list) else []:
        if not isinstance(row, dict):
            continue
        changes = row.get('changes') if isinstance(row.get('changes'), dict) else {}
        if 'label' in changes and isinstance(changes['label'], dict):
            after = _public_label(changes['label'].get('after'), '\u65b0\u540d\u79f0')
            summaries.append(f'\u5df2\u5c06\u8282\u70b9\u540d\u79f0\u6539\u4e3a\u201c{after}\u201d')
        elif {'x', 'y'} & set(changes):
            label = _public_node_label(graph, row.get('id'), '\u8282\u70b9')
            summaries.append(f'\u5df2\u79fb\u52a8\u201c{label}\u201d\u8282\u70b9')
        else:
            label = _public_node_label(graph, row.get('id'), '\u8282\u70b9')
            summaries.append(f'\u5df2\u66f4\u65b0\u201c{label}\u201d\u8282\u70b9')
    for row in diff.get('changed_edges', []) if isinstance(diff.get('changed_edges'), list) else []:
        if not isinstance(row, dict):
            continue
        if row.get('id') in grouped_changed_edge_ids:
            continue
        changes = row.get('changes') if isinstance(row.get('changes'), dict) else {}
        edge = _edges_by_id(graph).get(row.get('id'), {})
        label = _edge_readable_name(edge, graph)
        if 'target' in changes and isinstance(changes['target'], dict):
            new_target = _public_node_label(graph, changes['target'].get('after'))
            raw_label = str(edge.get('branch_label') or edge.get('label') or '').strip() if isinstance(edge, dict) else ''
            if raw_label and not _looks_internal_or_technical(raw_label):
                summaries.append(ZH_REROUTE_BRANCH.format(label=raw_label, new_target=new_target))
            else:
                source = _public_node_label(graph, edge.get('source') if isinstance(edge, dict) else None, '\u8d77\u70b9')
                old_target = _public_node_label(graph, changes['target'].get('before'), '\u539f\u76ee\u6807\u8282\u70b9')
                summaries.append(ZH_REROUTE_CONNECTION.format(source=source, old_target=old_target, new_target=new_target))
        elif 'branch_label' in changes and isinstance(changes['branch_label'], dict):
            after = _public_label(changes['branch_label'].get('after'), '\u65b0\u5206\u652f\u540d\u79f0')
            summaries.append(f'\u5df2\u5c06\u5206\u652f\u540d\u79f0\u6539\u4e3a\u201c{after}\u201d')
        else:
            safe_label = _public_label(label or row.get('id'), '\u8fd9\u6761\u5206\u652f')
            summaries.append(f'\u5df2\u66f4\u65b0\u201c{safe_label}\u201d\u5206\u652f')
    return summaries


def _readable_diff_summaries(diff: dict[str, Any], graph: dict[str, Any] | None) -> list[str]:
    summaries = [
        summary if not any(token in summary.lower() for token in INTERNAL_TOKENS) else '\u5df2\u5b8c\u6210\u4e00\u4e2a\u4fee\u6539\u76ee\u6807'
        for summary in _diff_summaries(diff, graph)
    ]
    index = 0
    for row in diff.get('added_nodes', []) if isinstance(diff.get('added_nodes'), list) else []:
        if isinstance(row, dict):
            index += 1
    for row in diff.get('removed_nodes', []) if isinstance(diff.get('removed_nodes'), list) else []:
        if isinstance(row, dict):
            index += 1
    for row in diff.get('changed_nodes', []) if isinstance(diff.get('changed_nodes'), list) else []:
        if not isinstance(row, dict):
            continue
        changes = row.get('changes') if isinstance(row.get('changes'), dict) else {}
        if 'label' in changes and isinstance(changes['label'], dict) and index < len(summaries):
            before = _public_label(changes['label'].get('before') or _node_label(graph, row.get('id')), '\u539f\u8282\u70b9')
            after = _public_label(changes['label'].get('after'), '\u65b0\u540d\u79f0')
            summaries[index] = f'\u5df2\u5c06\u201c{before}\u201d\u8282\u70b9\u540d\u79f0\u6539\u4e3a\u201c{after}\u201d'
        index += 1
    return summaries

def _candidate_rows(issue: dict[str, Any], graph: dict[str, Any] | None) -> list[dict[str, Any]]:
    nodes = _nodes_by_id(graph)
    edges = _edges_by_id(graph)
    kind = str(issue.get("kind", ""))
    object_type = "edge" if "edge" in kind else "node"
    rows = []
    for candidate_id in issue.get("candidates", []) if isinstance(issue.get("candidates"), list) else []:
        if not isinstance(candidate_id, str):
            continue
        if object_type == "edge":
            edge = edges.get(candidate_id, {})
            if not edge:
                continue
            rows.append(
                {
                    "id": candidate_id,
                    "type": "edge",
                    "label": _edge_display(edge, nodes),
                    "reason": _readable_issue_reason(issue),
                }
            )
        else:
            node = nodes.get(candidate_id, {})
            if not node:
                continue
            rows.append(
                {
                    "id": candidate_id,
                    "type": "node",
                    "label": str(node.get("label") or "\u8282\u70b9"),
                    "reason": _readable_issue_reason(issue),
                }
            )
    if object_type == "edge":
        def edge_rank(row: dict[str, Any]) -> tuple[int, int, str]:
            edge = edges.get(row.get("id"), {})
            source_node = nodes.get(edge.get("source"), {})
            has_label = bool(edge.get("branch_label") or edge.get("label"))
            from_decision = source_node.get("type") == "decision"
            return (0 if has_label else 1, 0 if from_decision else 1, str(row.get("label") or ""))

        rows.sort(key=edge_rank)
    return rows


def _grounding_issues(specification: dict[str, Any] | None) -> list[dict[str, Any]]:
    spec = specification if isinstance(specification, dict) else {}
    trace = spec.get("trace") if isinstance(spec.get("trace"), dict) else {}
    grounded = trace.get("grounded_goal") if isinstance(trace.get("grounded_goal"), dict) else spec.get("grounded_goal")
    grounding = grounded.get("grounding") if isinstance(grounded, dict) and isinstance(grounded.get("grounding"), dict) else {}
    issues = grounding.get("issues")
    if isinstance(issues, list) and issues:
        return [issue for issue in issues if isinstance(issue, dict)]
    validation = spec.get("validation") if isinstance(spec.get("validation"), dict) else trace.get("validation")
    fallback = validation.get("issues") if isinstance(validation, dict) else []
    return [issue for issue in fallback if isinstance(issue, dict)] if isinstance(fallback, list) else []


def _partial_execution(specification: dict[str, Any] | None) -> dict[str, Any]:
    spec = specification if isinstance(specification, dict) else {}
    partial = spec.get("partial_execution")
    return partial if isinstance(partial, dict) else {}


def _trace_graph_diff(specification: dict[str, Any] | None) -> dict[str, Any]:
    spec = specification if isinstance(specification, dict) else {}
    trace = spec.get("trace") if isinstance(spec.get("trace"), dict) else {}
    diff = trace.get("graph_diff")
    return diff if isinstance(diff, dict) else {}


def _before_graph(specification: dict[str, Any] | None) -> dict[str, Any] | None:
    spec = specification if isinstance(specification, dict) else {}
    trace = spec.get("trace") if isinstance(spec.get("trace"), dict) else {}
    graph = trace.get("before_graph")
    return graph if isinstance(graph, dict) else None


def _constraints(specification: dict[str, Any] | None) -> dict[str, Any]:
    spec = specification if isinstance(specification, dict) else {}
    trace = spec.get("trace") if isinstance(spec.get("trace"), dict) else {}
    for value in (spec.get("constraints"), trace.get("constraints")):
        if isinstance(value, dict):
            return value
    return {}


def _grounded_goal(specification: dict[str, Any] | None) -> dict[str, Any]:
    spec = specification if isinstance(specification, dict) else {}
    trace = spec.get("trace") if isinstance(spec.get("trace"), dict) else {}
    for value in (spec.get("grounded_goal"), trace.get("grounded_goal")):
        if isinstance(value, dict):
            return value
    return {}


def _join_readable_items(items: list[str]) -> str:
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]}\u548c{items[1]}"
    return "\u3001".join(items[:-1]) + f"\u548c{items[-1]}"


def _display_branch_label(value: Any) -> str:
    label = _strip_outer_reference_quotes(value)
    lowered = label.lower()
    if lowered == "failure":
        return "\u5931\u8d25"
    if lowered == "success":
        return "\u6210\u529f"
    return label


def _preserved_edge_display(edge: dict[str, Any], nodes: dict[str, dict[str, Any]]) -> str:
    source = _strip_outer_reference_quotes(nodes.get(edge.get("source"), {}).get("label"))
    target = _strip_outer_reference_quotes(nodes.get(edge.get("target"), {}).get("label"))
    label = _display_branch_label(edge.get("branch_label") or edge.get("label"))
    if source and target and label:
        return f"{source}\u5230{target}\u7684{label}\u5206\u652f"
    if source and target:
        return f"{source}\u5230{target}\u7684\u8fde\u63a5"
    return ZH_CONNECTION


def _explicit_preserved_edge_ids(specification: dict[str, Any] | None) -> list[str]:
    grounded = _grounded_goal(specification)
    edge_ids: list[str] = []
    for goal in grounded.get("semantic_goals", []) if isinstance(grounded.get("semantic_goals"), list) else []:
        if not isinstance(goal, dict) or goal.get("kind") != "object_state":
            continue
        state = goal.get("object_state") if isinstance(goal.get("object_state"), dict) else {}
        if state.get("state") != "preserved":
            continue
        for binding in goal.get("bindings", []) if isinstance(goal.get("bindings"), list) else []:
            if (
                isinstance(binding, dict)
                and binding.get("role") == "object"
                and binding.get("binding_type") == "edge"
                and isinstance(binding.get("id"), str)
            ):
                edge_ids.append(binding["id"])
    result: list[str] = []
    for edge_id in edge_ids:
        if edge_id not in result:
            result.append(edge_id)
    return result


def _has_user_abstract_preserve(specification: dict[str, Any] | None) -> bool:
    for goal in _semantic_goals_v2(specification):
        constraints = goal.get("semantic_constraints") if isinstance(goal.get("semantic_constraints"), list) else []
        if any(isinstance(row, dict) and row.get("type") == "preserve" for row in constraints):
            return True
    return False


def _preserved_edge_summaries(
    specification: dict[str, Any] | None,
    graph: dict[str, Any] | None,
    diff: dict[str, Any],
) -> list[str]:
    edge_ids = _explicit_preserved_edge_ids(specification)
    removed = {
        row.get("id")
        for row in diff.get("removed_edges", [])
        if isinstance(row, dict) and isinstance(row.get("id"), str)
    }
    changed = {
        row.get("id")
        for row in diff.get("changed_edges", [])
        if isinstance(row, dict) and isinstance(row.get("id"), str)
    }
    edges = _edges_by_id(graph)
    nodes = _nodes_by_id(graph)
    explicit_displays: list[str] = []
    for edge_id in edge_ids:
        if not isinstance(edge_id, str) or edge_id in removed or edge_id in changed:
            continue
        edge = edges.get(edge_id)
        if not isinstance(edge, dict):
            continue
        display = _preserved_edge_display(edge, nodes)
        if display:
            explicit_displays.append(display)
    summaries: list[str] = []
    if explicit_displays:
        summaries.append(f"\u5df2\u4fdd\u7559{_join_readable_items(explicit_displays)}")
    if _has_user_abstract_preserve(specification):
        summaries.append("\u5176\u4ed6\u672a\u6d89\u53ca\u7684\u6d41\u7a0b\u8fde\u63a5\u4fdd\u6301\u4e0d\u53d8")
    return summaries


def _diff_has_changes(diff: dict[str, Any]) -> bool:
    return any(
        bool(diff.get(key))
        for key in ("added_nodes", "removed_nodes", "changed_nodes", "added_edges", "removed_edges", "changed_edges")
    )


def extract_user_candidates(specification: dict[str, Any] | None, graph: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    candidates = []
    for issue in _grounding_issues(specification):
        kind = str(issue.get("kind", ""))
        if kind not in CLARIFICATION_KINDS:
            continue
        rows = _candidate_rows(issue, graph)
        if not rows:
            continue
        if kind.startswith("ambiguous") and len(rows) < 2:
            continue
        candidates.append(
            {
                "location": issue.get("location", ""),
                "reference": issue.get("reference", ""),
                "type": "edge" if "edge" in kind else "node",
                "reason": _readable_issue_reason(issue),
                "candidates": rows,
            }
        )
    return candidates


def _readable_incomplete_status(status: Any) -> str:
    value = str(status or "")
    if value in {"missing", "ambiguous"}:
        return ZH_NEEDS_CONFIRMATION_REASON
    if value == "skipped_due_to_dependency":
        return "\u9700\u8981\u5148\u5b8c\u6210\u524d\u7f6e\u4fee\u6539\u76ee\u6807"
    return value or ZH_PARTIAL_INCOMPLETE


def _node_display(graph: dict[str, Any] | None, node_id: Any) -> str:
    label = _node_label(graph, node_id)
    return _public_label(label, "\u76ee\u6807\u8282\u70b9")


def _edge_by_id(graph: dict[str, Any] | None, edge_id: Any) -> dict[str, Any] | None:
    if not isinstance(edge_id, str):
        return None
    edge = _edges_by_id(graph).get(edge_id)
    return edge if isinstance(edge, dict) else None


def _edge_display_by_id(graph: dict[str, Any] | None, edge_id: Any) -> str:
    edge = _edge_by_id(graph, edge_id)
    if not edge:
        return ZH_CONNECTION
    return _edge_display(edge, _nodes_by_id(graph))


def _semantic_goals_v2(specification: dict[str, Any] | None) -> list[dict[str, Any]]:
    spec = specification if isinstance(specification, dict) else {}
    candidates: list[Any] = []
    trace = spec.get("trace") if isinstance(spec.get("trace"), dict) else {}
    for container in (
        spec.get("grounded_goal"),
        trace.get("grounded_goal"),
        spec.get("semantic_goal", {}).get("normalized") if isinstance(spec.get("semantic_goal"), dict) else None,
        trace.get("semantic_goal", {}).get("normalized") if isinstance(trace.get("semantic_goal"), dict) else None,
    ):
        if isinstance(container, dict):
            candidates.append(container.get("semantic_goals_v2"))
    for value in candidates:
        if isinstance(value, list):
            return [row for row in value if isinstance(row, dict)]
    return []


def _grounded_provenance(specification: dict[str, Any] | None) -> list[dict[str, Any]]:
    spec = specification if isinstance(specification, dict) else {}
    trace = spec.get("trace") if isinstance(spec.get("trace"), dict) else {}
    grounded = trace.get("grounded_goal") if isinstance(trace.get("grounded_goal"), dict) else spec.get("grounded_goal")
    grounding = grounded.get("grounding") if isinstance(grounded, dict) and isinstance(grounded.get("grounding"), dict) else {}
    provenance = grounding.get("provenance")
    return [row for row in provenance if isinstance(row, dict)] if isinstance(provenance, list) else []


def _selected_edge_ids(specification: dict[str, Any] | None) -> list[str]:
    spec = specification if isinstance(specification, dict) else {}
    trace = spec.get("trace") if isinstance(spec.get("trace"), dict) else {}
    selection = trace.get("selection") if isinstance(trace.get("selection"), dict) else {}
    edge_ids = selection.get("edge_ids")
    return [edge_id for edge_id in edge_ids if isinstance(edge_id, str)] if isinstance(edge_ids, list) else []


def _resolved_object_ids(specification: dict[str, Any] | None) -> set[str]:
    resolved: set[str] = set()
    for row in _grounded_provenance(specification):
        resolved_id = row.get("resolved_id")
        reference = row.get("reference")
        if isinstance(resolved_id, str):
            resolved.add(resolved_id)
        if isinstance(reference, str):
            resolved.add(reference)
    return resolved


def _object_by_id(goal: dict[str, Any]) -> dict[str, Any]:
    return {
        str(row.get("id")): row
        for row in (goal.get("objects", []) if isinstance(goal.get("objects"), list) else [])
        if isinstance(row, dict) and row.get("id")
    }


def _route_relation_context(specification: dict[str, Any] | None, graph: dict[str, Any] | None) -> dict[str, Any]:
    resolved = _resolved_object_ids(specification)
    selected_edges = _selected_edge_ids(specification)
    for goal in _semantic_goals_v2(specification):
        objects = _object_by_id(goal)
        for relation in goal.get("relation_intents", []) if isinstance(goal.get("relation_intents"), list) else []:
            if not isinstance(relation, dict) or relation.get("relation") != "route_to":
                continue
            source_key = str(relation.get("source") or "")
            target_key = str(relation.get("target") or "")
            source_object = objects.get(source_key)
            target_object = objects.get(target_key)
            source_unresolved = isinstance(source_object, dict) and source_object.get("type") == "unresolved_reference"
            target_unresolved = isinstance(target_object, dict) and target_object.get("type") == "unresolved_reference"
            source_edge_id = next((edge_id for edge_id in selected_edges if _edge_by_id(graph, edge_id)), None)
            known_subject = _edge_display_by_id(graph, source_edge_id) if source_edge_id else ""
            known_target = ""
            target_reference = target_object.get("reference") if isinstance(target_object, dict) else None
            if target_unresolved:
                known_target = ""
            elif isinstance(target_reference, str) and target_reference in resolved and target_reference in _nodes_by_id(graph):
                known_target = _node_display(graph, target_reference)
            elif isinstance(target_reference, str) and target_reference.strip():
                node_id = next((node_id for node_id, node in _nodes_by_id(graph).items() if str(node.get("label") or node_id) == target_reference), "")
                known_target = _node_display(graph, node_id) if node_id else target_reference
            return {
                "relation": relation,
                "source": None if source_unresolved else source_object,
                "target": None if target_unresolved else target_object,
                "known_subject": known_subject,
                "known_target": known_target,
                "source_unresolved": source_unresolved,
                "target_unresolved": target_unresolved,
            }
    return {}


def _unresolved_semantic_dimensions(specification: dict[str, Any] | None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for goal in _semantic_goals_v2(specification):
        goal_id = str(goal.get("id") or "")
        objects = _object_by_id(goal)
        for unresolved in goal.get("unresolved_dimensions", []) if isinstance(goal.get("unresolved_dimensions"), list) else []:
            if not isinstance(unresolved, dict):
                continue
            row = dict(unresolved)
            row["goal_id"] = goal_id
            object_id = str(row.get("object") or "")
            if object_id and object_id in objects:
                row["object_ref"] = objects[object_id]
            rows.append(row)
    return rows


def _semantic_dimension_prompt(dimension: str, object_ref: dict[str, Any] | None, fallback: str) -> str:
    object_type = str(object_ref.get("type") or "") if isinstance(object_ref, dict) else ""
    if dimension == "transformation":
        if object_type == "edge_reference":
            return "\u4f60\u60f3\u600e\u4e48\u4fee\u6539\u8fd9\u6761\u8fde\u63a5\uff1f\u53ef\u4ee5\u8bf4\u660e\u8981\u6539\u63a5\u5230\u54ea\u4e2a\u8282\u70b9\u3001\u4fee\u6539\u6807\u7b7e\uff0c\u6216\u5220\u9664\u5b83\u3002"
        if object_type == "existing_reference":
            return "\u4f60\u60f3\u600e\u4e48\u4fee\u6539\u8fd9\u4e2a\u8282\u70b9\uff1f\u53ef\u4ee5\u8bf4\u660e\u8981\u6539\u540d\u3001\u5220\u9664\uff0c\u6216\u8c03\u6574\u5b83\u4e0e\u5176\u4ed6\u8282\u70b9\u7684\u5173\u7cfb\u3002"
        return "\u4f60\u60f3\u600e\u4e48\u4fee\u6539\u8fd9\u4e2a\u5bf9\u8c61\uff1f\u8bf7\u8865\u5145\u5177\u4f53\u7684\u4fee\u6539\u65b9\u5f0f\u3002"
    if dimension in {"parameter", "value"}:
        return "\u8bf7\u8865\u5145\u8fd9\u6b21\u4fee\u6539\u9700\u8981\u7684\u5177\u4f53\u503c\u3002"
    if dimension == "reference":
        return fallback or ZH_UNKNOWN_CLARIFICATION
    if dimension == "constraint":
        return "\u8bf7\u8865\u5145\u8fd9\u6b21\u4fee\u6539\u9700\u8981\u6ee1\u8db3\u7684\u7ea6\u675f\u3002"
    return fallback or ZH_UNKNOWN_CLARIFICATION


def _interaction_from_feedback(
    status: str,
    specification: dict[str, Any] | None,
    graph: dict[str, Any] | None,
    *,
    candidates: list[dict[str, Any]],
    question: str,
    route_confirmation_question: str,
    executed: list[dict[str, Any]],
    incomplete: list[dict[str, Any]],
    error: str | None,
) -> dict[str, Any]:
    if status == "success":
        return {"type": "success", "prompt": ""}
    if status == "partial_success":
        return {"type": "partial_completion", "completed": executed, "remaining": incomplete}
    if route_confirmation_question:
        return {"type": "confirm_change", "prompt": route_confirmation_question}
    if candidates:
        first = candidates[0]
        rows = first.get("candidates", []) if isinstance(first.get("candidates"), list) else []
        candidate_type = str(first.get("type") or "object")
        noun = "\u8fde\u63a5" if candidate_type == "edge" else "\u8282\u70b9"
        prompt = f"\u6211\u627e\u5230 {len(rows)} \u4e2a\u53ef\u80fd\u7684{noun}\uff0c\u4f60\u60f3\u4fee\u6539\u54ea\u4e00\u4e2a\uff1f"
        return {"type": "choose_candidate", "candidate_type": candidate_type, "prompt": prompt, "candidates": rows}
    issue_kinds = {str(issue.get("kind") or "") for issue in _grounding_issues(specification)}
    route_context = _route_relation_context(specification, graph)
    unresolved_dimensions = _unresolved_semantic_dimensions(specification)
    transformation = next(
        (
            row
            for row in unresolved_dimensions
            if str(row.get("dimension") or "") == "transformation"
        ),
        None,
    )
    if isinstance(transformation, dict):
        prompt = _semantic_dimension_prompt(
            "transformation",
            transformation.get("object_ref") if isinstance(transformation.get("object_ref"), dict) else None,
            question,
        )
        return {
            "type": "provide_missing_slot",
            "slot": "transformation",
            "dimension": "transformation",
            "prompt": prompt,
        }
    generic_unresolved = next((row for row in unresolved_dimensions if isinstance(row, dict)), None)
    if isinstance(generic_unresolved, dict):
        dimension = str(generic_unresolved.get("dimension") or "unknown")
        prompt = _semantic_dimension_prompt(
            dimension,
            generic_unresolved.get("object_ref") if isinstance(generic_unresolved.get("object_ref"), dict) else None,
            question,
        )
        return {
            "type": "provide_missing_slot",
            "slot": dimension,
            "dimension": dimension,
            "prompt": prompt,
        }
    if issue_kinds & {"missing_route_edge_binding", "missing_edge_reference", "missing_branch_edge_binding"}:
        known_target = str(route_context.get("known_target") or "")
        prompt = (
            f"\u4f60\u60f3\u628a\u54ea\u6761\u5206\u652f\u6216\u8fde\u63a5\u6539\u5230\u201c{known_target}\u201d\uff1f"
            if known_target
            else question or ZH_MISSING_EDGE_REASON
        )
        return {"type": "provide_missing_slot", "slot": "subject_edge", "known_target": known_target, "prompt": prompt}
    if issue_kinds & {"missing_route_target_binding"}:
        known_subject = str(route_context.get("known_subject") or "")
        prompt = (
            f"\u4f60\u5e0c\u671b\u628a\u201c{known_subject}\u201d\u8fd9\u6761\u8fde\u63a5\u6539\u5230\u54ea\u4e2a\u8282\u70b9\uff1f"
            if known_subject
            else "\u4f60\u5e0c\u671b\u628a\u8fd9\u6761\u8fde\u63a5\u6539\u5230\u54ea\u4e2a\u8282\u70b9\uff1f"
        )
        return {"type": "provide_missing_slot", "slot": "route_target", "known_subject": known_subject, "prompt": prompt}
    if "unknown_relation_object" in issue_kinds and route_context:
        relation = route_context.get("relation") if isinstance(route_context.get("relation"), dict) else {}
        source_key = str(relation.get("source") or "")
        target_key = str(relation.get("target") or "")
        if not route_context.get("source") and route_context.get("target"):
            known_target = str(route_context.get("known_target") or "")
            prompt = (
                f"\u4f60\u60f3\u628a\u54ea\u6761\u5206\u652f\u6216\u8fde\u63a5\u6539\u5230\u201c{known_target}\u201d\uff1f"
                if known_target
                else "\u4f60\u60f3\u4fee\u6539\u54ea\u6761\u5206\u652f\u6216\u8fde\u63a5\uff1f"
            )
            return {"type": "provide_missing_slot", "slot": "subject_edge", "known_target": known_target, "prompt": prompt}
        if route_context.get("source") and not route_context.get("target"):
            known_subject = str(route_context.get("known_subject") or "")
            prompt = (
                f"\u4f60\u5e0c\u671b\u628a\u201c{known_subject}\u201d\u8fd9\u6761\u8fde\u63a5\u6539\u5230\u54ea\u4e2a\u8282\u70b9\uff1f"
                if known_subject
                else "\u4f60\u5e0c\u671b\u628a\u8fd9\u6761\u8fde\u63a5\u6539\u5230\u54ea\u4e2a\u8282\u70b9\uff1f"
            )
            return {"type": "provide_missing_slot", "slot": "route_target", "known_subject": known_subject, "prompt": prompt}
        if source_key and target_key:
            return {"type": "provide_missing_slot", "slot": "relation_target", "prompt": question or ZH_UNKNOWN_CLARIFICATION}
    if "missing_semantic_role" in issue_kinds and route_context:
        if route_context.get("source_unresolved") and route_context.get("target"):
            known_target = str(route_context.get("known_target") or "")
            prompt = (
                f"\u4f60\u60f3\u628a\u54ea\u6761\u5206\u652f\u6216\u8fde\u63a5\u6539\u5230\u201c{known_target}\u201d\uff1f"
                if known_target
                else "\u4f60\u60f3\u4fee\u6539\u54ea\u6761\u5206\u652f\u6216\u8fde\u63a5\uff1f"
            )
            return {"type": "provide_missing_slot", "slot": "subject_edge", "known_target": known_target, "prompt": prompt}
        if route_context.get("source") and route_context.get("target_unresolved"):
            known_subject = str(route_context.get("known_subject") or "")
            prompt = (
                f"\u4f60\u5e0c\u671b\u628a\u201c{known_subject}\u201d\u8fd9\u6761\u8fde\u63a5\u6539\u5230\u54ea\u4e2a\u8282\u70b9\uff1f"
                if known_subject
                else "\u4f60\u5e0c\u671b\u628a\u8fd9\u6761\u8fde\u63a5\u6539\u5230\u54ea\u4e2a\u8282\u70b9\uff1f"
            )
            return {"type": "provide_missing_slot", "slot": "route_target", "known_subject": known_subject, "prompt": prompt}
    if status == "needs_clarification":
        return {"type": "provide_missing_slot", "slot": "unknown", "prompt": question or ZH_UNKNOWN_CLARIFICATION}
    if status in {"execution_error", "planning_error", "interpretation_error"} or error:
        prompt = "\u8fd9\u6b21\u4fee\u6539\u6267\u884c\u6ca1\u6709\u6210\u529f\uff0c\u8bf7\u91cd\u8bd5\u6216\u6362\u4e00\u79cd\u8868\u8fbe\u3002"
        return {"type": "retry_after_internal_failure", "prompt": prompt}
    return {"type": "unsupported_goal", "prompt": question or ZH_UNKNOWN_UNSUPPORTED}


def _message_from_interaction(interaction: dict[str, Any]) -> str | None:
    interaction_type = interaction.get("type")
    if interaction_type in {"provide_missing_slot", "confirm_change", "retry_after_internal_failure", "unsupported_goal"}:
        return str(interaction.get("prompt") or "").strip() or None
    if interaction_type == "choose_candidate":
        lines = [str(interaction.get("prompt") or "").strip()]
        rows = interaction.get("candidates") if isinstance(interaction.get("candidates"), list) else []
        for index, candidate in enumerate(rows[:5], start=1):
            lines.append(f"{index}. {candidate.get('label')}")
        if len(rows) > 5:
            lines.append(f"\u8fd8\u6709 {len(rows) - 5} \u4e2a\u5019\u9009\u3002")
        return "\n".join(line for line in lines if line).strip()
    return None


def derive_interaction_need(
    status: str,
    specification: dict[str, Any] | None = None,
    graph: dict[str, Any] | None = None,
    clarification_message: str | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    """Derive structured interaction facts before user-facing wording."""
    partial = _partial_execution(specification)
    subgoals = partial.get("subgoals") if isinstance(partial.get("subgoals"), list) else []
    diff = _trace_graph_diff(specification)
    can_show_executed = status in {"success", "partial_success"} and (not diff or _diff_has_changes(diff))
    executed_rows = [
        row
        for row in subgoals
        if can_show_executed and isinstance(row, dict) and row.get("status") == "executed"
    ]
    diff_summaries = _structural_relation_summaries(specification, graph, diff)
    for summary in _readable_diff_summaries(diff, graph):
        if summary not in diff_summaries:
            diff_summaries.append(summary)
    if can_show_executed:
        for summary in _preserved_edge_summaries(specification, graph, diff):
            if summary not in diff_summaries:
                diff_summaries.append(summary)
    if can_show_executed and diff_summaries:
        executed = [{"summary": summary} for summary in diff_summaries]
    else:
        executed = [{"summary": _readable_description(row)} for row in executed_rows]
    incomplete = [
        {
            "summary": _readable_description(row),
            "reason": _readable_reason(row.get("reason")),
            "status": _readable_incomplete_status(row.get("status")),
        }
        for row in subgoals
        if isinstance(row, dict) and row.get("status") != "executed"
    ]
    candidates = extract_user_candidates(specification, graph)
    route_confirmation_question = ""
    layout_collision_question = ""
    issue_question = ""
    for issue in _grounding_issues(specification):
        if issue.get("kind") == "route_same_source_target_confirmation":
            route_confirmation_question = _same_target_confirmation_question(issue, graph)
            break
        if issue.get("kind") == "layout_collision":
            layout_collision_question = _layout_collision_question(issue, graph)
        if not issue_question:
            issue_question = _readable_issue_question(issue, graph)
    question = route_confirmation_question or layout_collision_question or issue_question or _readable_question(clarification_message, fallback_for_unsafe_lowering=True)
    if not question and candidates:
        first = candidates[0]
        if str(first.get("kind", "")).startswith("ambiguous"):
            question = f"我找到了多个可能的“{first.get('reference') or '对象'}”，请从候选中选择一个。"
        elif str(first.get("kind", "")).startswith("missing"):
            question = f"当前图中没有找到“{first.get('reference') or '目标对象'}”对应的对象。"
        elif str(first.get("kind", "")).startswith("stale"):
            question = "已选择的候选对象已经不存在，请重新指定。"
    if not question and status == "unsupported":
        question = "当前请求还不能安全转为可执行的流程修改。"
    if not question and error:
        question = _readable_question(error, fallback_for_unsafe_lowering=True)

    interaction = _interaction_from_feedback(
        status,
        specification,
        graph,
        candidates=candidates,
        question=question,
        route_confirmation_question=route_confirmation_question,
        executed=executed,
        incomplete=incomplete,
        error=error,
    )
    return {
        "status": status,
        "interaction": interaction,
        "candidates": candidates,
        "question": question,
        "route_confirmation_question": route_confirmation_question,
        "executed": executed,
        "incomplete": incomplete,
    }


def build_user_feedback(
    status: str,
    specification: dict[str, Any] | None = None,
    graph: dict[str, Any] | None = None,
    clarification_message: str | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    interaction_need = derive_interaction_need(status, specification, graph, clarification_message, error)
    executed = interaction_need["executed"]
    incomplete = interaction_need["incomplete"]
    candidates = interaction_need["candidates"]
    question = interaction_need["question"]
    route_confirmation_question = interaction_need["route_confirmation_question"]
    interaction = interaction_need["interaction"]

    interaction_message = _message_from_interaction(interaction)
    if interaction_message and status != "partial_success":
        if interaction.get("type") == "confirm_change":
            interaction_message = f"\u8fd9\u4e2a\u4fee\u6539\u9700\u8981\u4f60\u786e\u8ba4\u3002\n{interaction_message}"
        return {
            "status": status,
            "message": interaction_message,
            "executed": executed if status == "success" else [],
            "incomplete": incomplete,
            "interaction": interaction,
            "clarification": {
                "question": str(interaction.get("prompt") or question or ""),
                "candidates": candidates,
            },
        }

    lines: list[str] = []
    if status == "success":
        lines.append("\u5df2\u5b8c\u6210\u4fee\u6539\uff0c\u6d41\u7a0b\u56fe\u5df2\u66f4\u65b0\u3002")
    elif status == "partial_success":
        lines.append("\u5df2\u5b8c\u6210\u90e8\u5206\u4fee\u6539\u3002")
    elif status == "needs_clarification":
        if route_confirmation_question:
            lines.append("\u8fd9\u4e2a\u4fee\u6539\u9700\u8981\u4f60\u786e\u8ba4\u3002")
        elif candidates:
            lines.append("\u6211\u627e\u5230\u4e86\u591a\u4e2a\u53ef\u80fd\u7684\u5bf9\u8c61\uff0c\u9700\u8981\u4f60\u786e\u8ba4\u3002")
        else:
            lines.append("\u6211\u8fd8\u4e0d\u80fd\u786e\u5b9a\u8981\u4fee\u6539\u7684\u5bf9\u8c61\u3002")
    elif status == "unsupported":
        lines.append("\u8fd9\u6b21\u8bf7\u6c42\u6682\u65f6\u4e0d\u80fd\u5b89\u5168\u6267\u884c\u3002")
    else:
        lines.append(question or "\u8bf7\u6c42\u672a\u5b8c\u6210\u3002")
    if executed:
        lines.append("")
        lines.append("\u5df2\u5b8c\u6210\uff1a")
        lines.extend(f"- {row['summary']}" for row in executed)
    if incomplete:
        lines.append("")
        lines.append("\u672a\u5b8c\u6210\uff1a")
        lines.extend(f"- {row['summary']}：{row['reason'] or row['status']}" for row in incomplete)
    if question:
        lines.append("")
        lines.append("\u9700\u8981\u4f60\u786e\u8ba4\uff1a")
        lines.append(question)
    if candidates:
        lines.append("")
        lines.append("\u5019\u9009\uff1a")
        for group in candidates:
            rows = group.get("candidates", [])
            visible_rows = rows[:5] if isinstance(rows, list) else []
            for index, candidate in enumerate(visible_rows, start=1):
                lines.append(f"{index}. {candidate.get('label')}")
            if isinstance(rows, list) and len(rows) > len(visible_rows):
                lines.append(f"\u8fd8\u6709 {len(rows) - len(visible_rows)} \u4e2a\u5019\u9009\u3002")


    return {
        "status": status,
        "message": "\n".join(lines).strip(),
        "executed": executed,
        "incomplete": incomplete,
        "interaction": interaction,
        "clarification": {
            "question": question,
            "candidates": candidates,
        },
    }
