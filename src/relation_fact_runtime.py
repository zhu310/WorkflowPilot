from __future__ import annotations

from copy import deepcopy
from typing import Any, Callable

from .canonical_goal_state import normalize_relation_fact


PHASE1B_RELATION_GROUPER_SYSTEM = """Return one JSON object for a shadow-only Relation Grouper.

You receive:
- raw_utterance
- base_references
- base_relation_candidates
- optional active_prior_relation_refs / active_prior_relations

Task:
- Detect only explicit grouped semantic relations that are already present in the base semantic content.
- Output grouped semantic content, not operations.
- Do not invent refs.
- Do not output relation_id, delta op, correction target, properties, group_semantics, graph ids, or grounding.

Return JSON only. Output exactly:
{
  "relation_groups": [
    {
      "kind": "connection" | "precedence",
      "members": [
        {"source": "semantic ref id", "target": "semantic ref id"}
      ]
    }
  ]
}

Rules:
- members.length must be >= 2; each relation group must have at least 2 members.
- Use only refs visible in base_references or active prior relation context.
- A grouped relation is not the same as several unrelated binary facts.
- If no grouped relation is explicitly present, return {"relation_groups": []}.
"""


def attach_managed_relation_grouper(
    artifact: dict[str, Any],
    runner: Callable[[str, dict[str, Any]], Any],
    raw_utterance: str,
    semantic_proposal: dict[str, Any],
    *,
    planning_goal_state: dict[str, Any] | None = None,
    system_prompt: str = PHASE1B_RELATION_GROUPER_SYSTEM,
    parsed_json: Callable[[Any], Any] | None = None,
    raw_model_trace: Callable[[Any, Any], dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    payload = relation_grouper_payload(raw_utterance, semantic_proposal, planning_goal_state=planning_goal_state)
    if not _has_groupable_relation_candidates(payload):
        return []
    artifact["relation_grouper_input"] = payload
    try:
        grouper_result = runner(system_prompt, payload)
        parsed_grouper = parsed_json(grouper_result) if parsed_json is not None else _default_parsed_json(grouper_result)
        artifact["raw_relation_grouper"] = (
            raw_model_trace(grouper_result, parsed_grouper)
            if raw_model_trace is not None
            else _default_raw_model_trace(grouper_result, parsed_grouper)
        )
        artifact["shadow_relation_grouper"] = build_shadow_relation_group_artifact(
            parsed_grouper,
            semantic_proposal,
            planning_goal_state=planning_goal_state,
        )
        return artifact["shadow_relation_grouper"].get("managed_relation_fact_proposals", []) if artifact["shadow_relation_grouper"].get("ok") else []
    except Exception as exc:
        artifact["raw_relation_grouper"] = {"parsed": None, "parse_error": str(exc), "normalized_from_fence": False}
        artifact["shadow_relation_grouper"] = {
            "ok": False,
            "relation_groups": [],
            "validation": {"ok": False, "errors": [{"kind": "relation_grouper_shadow_failure", "message": str(exc)}]},
            "managed_relation_fact_proposals": [],
        }
        return []


def relation_grouper_payload(raw_utterance: str, semantic_proposal: dict[str, Any], *, planning_goal_state: dict[str, Any] | None = None) -> dict[str, Any]:
    facts = semantic_proposal.get("desired_facts") if isinstance(semantic_proposal.get("desired_facts"), list) else []
    payload = {
        "raw_utterance": raw_utterance,
        "base_references": deepcopy(semantic_proposal.get("references", [])) if isinstance(semantic_proposal.get("references"), list) else [],
        "base_relation_candidates": [
            deepcopy(fact)
            for fact in facts
            if isinstance(fact, dict) and str(fact.get("predicate") or "").startswith("relation.")
        ],
    }
    prior_context = relation_grouper_prior_context(semantic_proposal, planning_goal_state or {})
    if prior_context["active_prior_relation_refs"] or prior_context["active_prior_relations"]:
        payload.update(prior_context)
    return payload


def _has_groupable_relation_candidates(payload: dict[str, Any]) -> bool:
    candidates = payload.get("base_relation_candidates") if isinstance(payload.get("base_relation_candidates"), list) else []
    groupable_count = 0
    for candidate in candidates:
        if isinstance(candidate, dict) and candidate.get("predicate") == "relation.before":
            groupable_count += 1
    return groupable_count >= 2


def build_shadow_relation_group_artifact(parsed: Any, semantic_proposal: dict[str, Any], *, planning_goal_state: dict[str, Any] | None = None) -> dict[str, Any]:
    normalized = normalize_relation_group_proposal(parsed)
    validation = validate_relation_group_proposal(normalized, semantic_proposal, planning_goal_state=planning_goal_state)
    usable = usable_relation_group_proposal(normalized, semantic_proposal, planning_goal_state=planning_goal_state)
    relation_facts = relation_groups_to_managed_relation_fact_proposals(usable, semantic_proposal)
    ok = validation["ok"] or bool(usable["relation_groups"])
    return {
        "ok": ok,
        "relation_groups": normalized["relation_groups"],
        "validation": validation,
        "managed_relation_fact_proposals": relation_facts,
    }


def normalize_relation_group_proposal(parsed: Any) -> dict[str, Any]:
    row = parsed if isinstance(parsed, dict) else {}
    groups = []
    for group in row.get("relation_groups", []) if isinstance(row.get("relation_groups"), list) else []:
        if not isinstance(group, dict):
            continue
        members = []
        for member in group.get("members", []) if isinstance(group.get("members"), list) else []:
            if not isinstance(member, dict):
                continue
            members.append({"source": str(member.get("source") or "").strip(), "target": str(member.get("target") or "").strip()})
        groups.append({"kind": str(group.get("kind") or "").strip(), "members": members})
    return {"relation_groups": groups}


def validate_relation_group_proposal(proposal: dict[str, Any], semantic_proposal: dict[str, Any], *, planning_goal_state: dict[str, Any] | None = None) -> dict[str, Any]:
    allowed_refs = relation_grouper_allowed_refs(semantic_proposal, planning_goal_state or {})
    errors: list[dict[str, Any]] = []
    for group_index, group in enumerate(proposal.get("relation_groups", [])):
        kind = group.get("kind")
        if kind not in {"connection", "precedence"}:
            errors.append({"kind": "invalid_kind", "group_index": group_index, "value": kind})
        members = group.get("members") if isinstance(group.get("members"), list) else []
        if len(members) < 2:
            errors.append({"kind": "too_few_members", "group_index": group_index, "count": len(members)})
        for member_index, member in enumerate(members):
            source = member.get("source")
            target = member.get("target")
            if source not in allowed_refs:
                errors.append({"kind": "invalid_ref", "group_index": group_index, "member_index": member_index, "field": "source", "ref_id": source})
            if target not in allowed_refs:
                errors.append({"kind": "invalid_ref", "group_index": group_index, "member_index": member_index, "field": "target", "ref_id": target})
    return {"ok": not errors, "errors": errors}


def usable_relation_group_proposal(proposal: dict[str, Any], semantic_proposal: dict[str, Any], *, planning_goal_state: dict[str, Any] | None = None) -> dict[str, Any]:
    groups = []
    for group in proposal.get("relation_groups", []):
        if not isinstance(group, dict):
            continue
        single = {"relation_groups": [group]}
        if validate_relation_group_proposal(single, semantic_proposal, planning_goal_state=planning_goal_state)["ok"]:
            groups.append(group)
    return {"relation_groups": groups}


def relation_grouper_allowed_refs(semantic_proposal: dict[str, Any], planning_goal_state: dict[str, Any] | None = None) -> set[str]:
    allowed_refs = {
        row.get("ref_id")
        for row in semantic_proposal.get("references", [])
        if isinstance(row, dict) and isinstance(row.get("ref_id"), str) and row.get("ref_id")
    }
    goal_state = planning_goal_state if isinstance(planning_goal_state, dict) else {}
    active_refs = _active_goal_state_ref_ids(goal_state)
    allowed_refs.update(_active_managed_relation_ref_ids(goal_state) & active_refs)
    candidate_refs = set()
    for fact in semantic_proposal.get("desired_facts", []):
        if isinstance(fact, dict) and str(fact.get("predicate") or "").startswith("relation."):
            candidate_refs.update(_reference_operand_ids(fact.get("operands")))
    allowed_refs.update(candidate_refs & active_refs)
    return allowed_refs


def relation_grouper_prior_context(semantic_proposal: dict[str, Any], planning_goal_state: dict[str, Any] | None = None) -> dict[str, Any]:
    goal_state = planning_goal_state if isinstance(planning_goal_state, dict) else {}
    prior_ref_ids = relation_grouper_allowed_refs(semantic_proposal, goal_state) - {
        row.get("ref_id")
        for row in semantic_proposal.get("references", [])
        if isinstance(row, dict) and isinstance(row.get("ref_id"), str) and row.get("ref_id")
    }
    refs_by_id = {
        row.get("ref_id"): row
        for row in goal_state.get("references", [])
        if isinstance(row, dict) and isinstance(row.get("ref_id"), str)
    }
    active_prior_relation_refs = []
    for ref_id in sorted(prior_ref_ids):
        ref = refs_by_id.get(ref_id)
        if not isinstance(ref, dict):
            continue
        active_prior_relation_refs.append(
            {
                "ref_id": ref_id,
                "entity_type": ref.get("entity_type"),
                "mention_text": _reference_mention_text(ref),
            }
        )
    active_prior_relations = []
    for relation_fact in _managed_relation_facts(goal_state):
        normalized = normalize_relation_fact(relation_fact)
        members = [
            {"source": member["source_ref"], "target": member["target_ref"]}
            for member in normalized["members"]
            if member["source_ref"] in prior_ref_ids or member["target_ref"] in prior_ref_ids
        ]
        if members:
            active_prior_relations.append({"kind": normalized["kind"], "members": members})
    return {
        "active_prior_relation_refs": active_prior_relation_refs,
        "active_prior_relations": active_prior_relations,
    }


def relation_groups_to_managed_relation_fact_proposals(proposal: dict[str, Any], semantic_proposal: dict[str, Any]) -> list[dict[str, Any]]:
    provenance_refs = _relation_group_provenance_refs(semantic_proposal)
    relation_facts = []
    for group in proposal.get("relation_groups", []):
        relation_facts.append(
            normalize_relation_fact(
                {
                    "kind": group["kind"],
                    "members": [
                        {
                            "source_ref": member["source"],
                            "target_ref": member["target"],
                            "properties": {},
                            "provenance_refs": provenance_refs,
                        }
                        for member in group["members"]
                    ],
                    "group_semantics": {},
                    "provenance_refs": provenance_refs,
                }
            )
        )
    return relation_facts


def _reference_operand_ids(operands: Any) -> set[str]:
    refs: set[str] = set()
    if isinstance(operands, dict):
        for value in operands.values():
            if isinstance(value, dict) and value.get("operand_type") == "reference" and isinstance(value.get("ref_id"), str):
                refs.add(value["ref_id"])
    return refs


def _managed_relation_facts(goal_state: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        row
        for row in goal_state.get("relation_facts", [])
        if isinstance(row, dict)
        and isinstance(row.get("relation_id"), str)
        and row.get("relation_id")
        and not str(row.get("relation_id")).startswith("rf:")
    ]


def _active_goal_state_ref_ids(goal_state: dict[str, Any]) -> set[str]:
    return {
        row.get("ref_id")
        for row in goal_state.get("references", [])
        if isinstance(row, dict) and isinstance(row.get("ref_id"), str) and row.get("ref_id")
    }


def _active_managed_relation_ref_ids(goal_state: dict[str, Any]) -> set[str]:
    ref_ids: set[str] = set()
    for relation_fact in _managed_relation_facts(goal_state):
        normalized = normalize_relation_fact(relation_fact)
        for member in normalized["members"]:
            if member["source_ref"]:
                ref_ids.add(member["source_ref"])
            if member["target_ref"]:
                ref_ids.add(member["target_ref"])
    return ref_ids


def _reference_mention_text(reference: dict[str, Any]) -> str | None:
    mention = reference.get("mention")
    if isinstance(mention, dict) and isinstance(mention.get("text"), str):
        return mention["text"]
    return None


def _relation_group_provenance_refs(semantic_proposal: dict[str, Any]) -> list[str]:
    refs: list[str] = []
    for field in ("references", "desired_facts"):
        for row in semantic_proposal.get(field, []):
            if not isinstance(row, dict):
                continue
            for ref in row.get("provenance_refs", []) if isinstance(row.get("provenance_refs"), list) else []:
                if isinstance(ref, str) and ref not in refs:
                    refs.append(ref)
    return refs


def _default_parsed_json(result: Any) -> Any:
    return getattr(result, "parsed_json", result if isinstance(result, dict) else None)


def _default_raw_model_trace(result: Any, parsed: Any) -> dict[str, Any]:
    return {
        "raw_response": getattr(result, "raw_response", ""),
        "parsed": parsed,
        "parse_error": getattr(result, "parse_error", None),
        "normalized_from_fence": bool(getattr(result, "normalized_from_fence", False)),
    }
