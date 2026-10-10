from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


MIN_CONFIDENT_SCORE = 0.30
AMBIGUITY_MARGIN = 0.08


@dataclass(frozen=True)
class ReferenceCandidate:
    id: str
    score: float
    evidence: list[str]


def _text(value: Any) -> str:
    return str(value or "").strip().lower()


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _field_values(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, (int, float)):
        return [str(value)]
    if isinstance(value, list):
        result: list[str] = []
        for item in value:
            result.extend(_field_values(item))
        return result
    if isinstance(value, dict):
        result = []
        for item in value.values():
            result.extend(_field_values(item))
        return result
    return []


def _cjk_chars(text: str) -> set[str]:
    return {char for char in text if "\u4e00" <= char <= "\u9fff"}


def _word_tokens(text: str) -> set[str]:
    return {token for token in re.findall(r"[a-z0-9]+", text.lower()) if token}


def _ngrams(text: str, size: int = 2) -> set[str]:
    compact = re.sub(r"\s+", "", text.lower())
    if len(compact) < size:
        return {compact} if compact else set()
    return {compact[index : index + size] for index in range(len(compact) - size + 1)}


def _dice(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return (2.0 * len(left & right)) / (len(left) + len(right))


def semantic_text_score(reference: str, candidate_text: str) -> float:
    reference = _text(reference)
    candidate_text = _text(candidate_text)
    if not reference or not candidate_text:
        return 0.0
    if reference == candidate_text:
        return 1.0
    if reference in candidate_text or candidate_text in reference:
        return 0.95
    cjk_score = _dice(_cjk_chars(reference), _cjk_chars(candidate_text))
    word_score = _dice(_word_tokens(reference), _word_tokens(candidate_text))
    ngram_score = _dice(_ngrams(reference), _ngrams(candidate_text))
    return max(cjk_score, word_score, ngram_score)


def _node_lookup(graph_data: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        node["id"]: node
        for node in graph_data.get("nodes", [])
        if isinstance(node, dict) and isinstance(node.get("id"), str)
    }


def _edge_lookup(graph_data: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        edge["id"]: edge
        for edge in graph_data.get("edges", [])
        if isinstance(edge, dict) and isinstance(edge.get("id"), str)
    }


def _neighbor_labels(graph_data: dict[str, Any], node_id: str, nodes: dict[str, dict[str, Any]]) -> list[str]:
    labels: list[str] = []
    for edge in graph_data.get("edges", []):
        if not isinstance(edge, dict):
            continue
        source = edge.get("source")
        target = edge.get("target")
        other_id = None
        if source == node_id and isinstance(target, str):
            other_id = target
        elif target == node_id and isinstance(source, str):
            other_id = source
        if other_id and isinstance(nodes.get(other_id), dict):
            labels.append(str(nodes[other_id].get("label", "")))
    return labels


def _node_candidate_texts(graph_data: dict[str, Any], node: dict[str, Any], nodes: dict[str, dict[str, Any]]) -> list[str]:
    del graph_data, nodes
    fields: list[str] = []
    for key in ("id", "label", "type", "semantic_type", "description"):
        fields.extend(_field_values(node.get(key)))
    fields.extend(_field_values(node.get("aliases")))
    fields.extend(_field_values(node.get("metadata")))
    fields.extend(_field_values(node.get("properties")))
    return [field for field in fields if field.strip()]


def _condition_tokens(expression: str) -> list[str]:
    return [token for token in re.findall(r"[\u4e00-\u9fffa-zA-Z0-9_]+", expression) if token]


def _edge_candidate_texts(edge: dict[str, Any], nodes: dict[str, dict[str, Any]]) -> list[str]:
    fields: list[str] = []
    for key in ("id", "label", "branch_label", "type", "description"):
        fields.extend(_field_values(edge.get(key)))
    condition = edge.get("condition")
    if isinstance(condition, dict):
        expression = condition.get("expression")
        fields.extend(_field_values(expression))
        if isinstance(expression, str):
            fields.extend(_condition_tokens(expression))
    fields.extend(_field_values(edge.get("aliases")))
    fields.extend(_field_values(edge.get("metadata")))
    fields.extend(_field_values(edge.get("properties")))

    source = edge.get("source")
    target = edge.get("target")
    source_label = ""
    target_label = ""
    if isinstance(source, str) and isinstance(nodes.get(source), dict):
        source_label = str(nodes[source].get("label", ""))
        fields.append(source_label)
    if isinstance(target, str) and isinstance(nodes.get(target), dict):
        target_label = str(nodes[target].get("label", ""))
        fields.append(target_label)
    if source_label and target_label:
        fields.append(" ".join([source_label, target_label]))
    route_label = str(edge.get("branch_label") or edge.get("label") or "").strip()
    if route_label and (source_label or target_label):
        fields.append(" ".join(part for part in (route_label, source_label, target_label) if part))
    return [field for field in fields if field.strip()]


def _rank(reference: str, rows: list[tuple[str, list[str]]]) -> list[ReferenceCandidate]:
    ranked: list[ReferenceCandidate] = []
    for row_id, texts in rows:
        scored = [(semantic_text_score(reference, text), text) for text in texts]
        scored = [(score, text) for score, text in scored if score > 0]
        if not scored:
            continue
        score = max(score for score, _text_value in scored)
        evidence = sorted({text for text_score, text in scored if text_score == score})
        ranked.append(ReferenceCandidate(id=row_id, score=score, evidence=evidence))
    return sorted(ranked, key=lambda row: (-row.score, row.id))


def _decide(ranked: list[ReferenceCandidate]) -> tuple[str | None, list[str], list[ReferenceCandidate]]:
    if not ranked or ranked[0].score < MIN_CONFIDENT_SCORE:
        return None, [], ranked
    top_score = ranked[0].score
    close = [row for row in ranked if top_score - row.score <= AMBIGUITY_MARGIN and row.score >= MIN_CONFIDENT_SCORE]
    if len(close) == 1:
        return close[0].id, [], ranked
    return None, sorted(row.id for row in close), ranked


def resolve_semantic_node_reference(graph_data: dict[str, Any], reference: str) -> tuple[str | None, list[str], dict[str, Any]]:
    nodes = _node_lookup(graph_data)
    rows = [
        (node_id, _node_candidate_texts(graph_data, node, nodes))
        for node_id, node in nodes.items()
    ]
    ranked = _rank(reference, rows)
    node_id, ambiguous, _ranked = _decide(ranked)
    metadata: dict[str, Any] = {
        "resolution_type": "semantic_similarity",
        "semantic_candidates": [
            {"id": row.id, "score": round(row.score, 4), "evidence": row.evidence}
            for row in ranked[:5]
        ],
    }
    return node_id, ambiguous, metadata


def resolve_semantic_edge_reference(graph_data: dict[str, Any], reference: str) -> tuple[str | None, list[str], dict[str, Any]]:
    edges = _edge_lookup(graph_data)
    nodes = _node_lookup(graph_data)
    rows = [
        (edge_id, _edge_candidate_texts(edge, nodes))
        for edge_id, edge in edges.items()
    ]
    ranked = _rank(reference, rows)
    edge_id, ambiguous, _ranked = _decide(ranked)
    metadata: dict[str, Any] = {
        "resolution_type": "semantic_similarity",
        "semantic_candidates": [
            {"id": row.id, "score": round(row.score, 4), "evidence": row.evidence}
            for row in ranked[:5]
        ],
    }
    return edge_id, ambiguous, metadata


def resolve_strict_semantic_node_reference(graph_data: dict[str, Any], reference: str) -> tuple[str | None, list[str], dict[str, Any]]:
    nodes = _node_lookup(graph_data)
    rows = [
        (node_id, _node_candidate_texts(graph_data, node, nodes))
        for node_id, node in nodes.items()
    ]
    ranked = _rank(reference, rows)
    exact = [row for row in ranked if row.score >= 1.0]
    metadata: dict[str, Any] = {
        "resolution_type": "strict_semantic_similarity",
        "semantic_candidates": [
            {"id": row.id, "score": round(row.score, 4), "evidence": row.evidence}
            for row in ranked[:5]
        ],
    }
    if len(exact) == 1:
        return exact[0].id, [], metadata
    if len(exact) > 1:
        return None, sorted(row.id for row in exact), metadata
    return None, [], metadata


def resolve_strict_semantic_edge_reference(graph_data: dict[str, Any], reference: str) -> tuple[str | None, list[str], dict[str, Any]]:
    edges = _edge_lookup(graph_data)
    nodes = _node_lookup(graph_data)
    rows = [
        (edge_id, _edge_candidate_texts(edge, nodes))
        for edge_id, edge in edges.items()
    ]
    ranked = _rank(reference, rows)
    exact = [row for row in ranked if row.score >= 1.0]
    metadata: dict[str, Any] = {
        "resolution_type": "strict_semantic_similarity",
        "semantic_candidates": [
            {"id": row.id, "score": round(row.score, 4), "evidence": row.evidence}
            for row in ranked[:5]
        ],
    }
    if len(exact) == 1:
        return exact[0].id, [], metadata
    if len(exact) > 1:
        return None, sorted(row.id for row in exact), metadata
    return None, [], metadata
