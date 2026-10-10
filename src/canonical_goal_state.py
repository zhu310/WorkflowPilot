from __future__ import annotations

import json
from copy import deepcopy
from typing import Any


CANONICAL_GOAL_STATE_REVISION = "phase1a.shadow.v1"
RELATION_FACT_KINDS = {"connection", "precedence"}

PREDICATE_SIGNATURES: dict[str, dict[str, Any]] = {
    "entity.exists": {
        "predicate_id": "entity.exists",
        "role_names": ["subject"],
        "role_types": {"subject": ["node"]},
        "anchor_roles": ["subject"],
        "objective_roles": [],
        "symmetry": "none",
        "normalization_policy": {"operand_order": "preserve", "fact_key": ["predicate", "subject"]},
        "explicit_empty_policy": "not_applicable",
        "semantic_expressible": True,
        "model_facing_desired_fact": True,
        "lowering_supported": True,
    },
    "entity.absent": {
        "predicate_id": "entity.absent",
        "role_names": ["subject"],
        "role_types": {"subject": ["node"]},
        "anchor_roles": ["subject"],
        "objective_roles": [],
        "symmetry": "none",
        "normalization_policy": {"operand_order": "preserve", "fact_key": ["predicate", "subject"]},
        "explicit_empty_policy": "not_applicable",
        "semantic_expressible": True,
        "model_facing_desired_fact": True,
        "lowering_supported": True,
    },
    "property.equals": {
        "predicate_id": "property.equals",
        "role_names": ["subject", "property", "value"],
        "role_types": {"subject": ["node", "edge"], "property": ["property_name"], "value": ["literal"]},
        "anchor_roles": ["subject", "property"],
        "objective_roles": ["value"],
        "symmetry": "none",
        "normalization_policy": {"operand_order": "preserve", "fact_key": ["predicate", "subject", "property"]},
        "explicit_empty_policy": "requires_structured_empty_intent",
        "semantic_expressible": True,
        "model_facing_desired_fact": True,
        "lowering_supported": True,
    },
    "relation.before": {
        "predicate_id": "relation.before",
        "role_names": ["subject", "anchor"],
        "role_types": {"subject": ["node"], "anchor": ["node"]},
        "anchor_roles": ["subject", "anchor"],
        "objective_roles": [],
        "symmetry": "none",
        "normalization_policy": {"operand_order": "preserve", "fact_key": ["predicate", "subject", "anchor"]},
        "explicit_empty_policy": "not_applicable",
        "semantic_expressible": True,
        "model_facing_desired_fact": True,
        "lowering_supported": True,
    },
    "relation.after": {
        "predicate_id": "relation.after",
        "role_names": ["subject", "anchor"],
        "role_types": {"subject": ["node"], "anchor": ["node"]},
        "anchor_roles": ["subject", "anchor"],
        "objective_roles": [],
        "symmetry": "none",
        "normalization_policy": {"operand_order": "preserve", "fact_key": ["predicate", "subject", "anchor"]},
        "explicit_empty_policy": "not_applicable",
        "semantic_expressible": True,
        "model_facing_desired_fact": False,
        "canonicalization_policy": {"canonical_predicate": "relation.before", "operand_role_map": {"subject": "anchor", "anchor": "subject"}},
        "lowering_supported": True,
    },
    "relation.right_of": {
        "predicate_id": "relation.right_of",
        "role_names": ["target", "anchor"],
        "role_types": {"target": ["node"], "anchor": ["node"]},
        "anchor_roles": ["target", "anchor"],
        "objective_roles": [],
        "symmetry": "none",
        "normalization_policy": {"operand_order": "preserve", "fact_key": ["predicate", "target", "anchor"]},
        "explicit_empty_policy": "not_applicable",
        "semantic_expressible": True,
        "model_facing_desired_fact": True,
        "lowering_supported": True,
    },
    "relation.route_to": {
        "predicate_id": "relation.route_to",
        "role_names": ["subject", "target"],
        "role_types": {"subject": ["edge"], "target": ["node"]},
        "anchor_roles": ["subject"],
        "objective_roles": ["target"],
        "symmetry": "none",
        "normalization_policy": {"operand_order": "preserve", "fact_key": ["predicate", "subject"]},
        "explicit_empty_policy": "not_applicable",
        "semantic_expressible": True,
        "model_facing_desired_fact": True,
        "lowering_supported": True,
    },
    "relation.parallel_with": {
        "predicate_id": "relation.parallel_with",
        "role_names": ["left", "right"],
        "role_types": {"left": ["node"], "right": ["node"]},
        "anchor_roles": ["left", "right"],
        "objective_roles": [],
        "symmetry": "unordered",
        "normalization_policy": {"operand_order": "sort_by_reference_id", "fact_key": ["predicate", "left", "right"]},
        "explicit_empty_policy": "not_applicable",
        "semantic_expressible": True,
        "model_facing_desired_fact": True,
        "lowering_supported": True,
    },
    "relation.connected_to": {
        "predicate_id": "relation.connected_to",
        "role_names": ["source", "target"],
        "role_types": {"source": ["node"], "target": ["node"]},
        "anchor_roles": ["source", "target"],
        "objective_roles": [],
        "symmetry": "none",
        "normalization_policy": {"operand_order": "preserve", "fact_key": ["predicate", "source", "target"]},
        "explicit_empty_policy": "not_applicable",
        "semantic_expressible": True,
        "model_facing_desired_fact": False,
        "lowering_supported": False,
    },
    "relation.joins_before": {
        "predicate_id": "relation.joins_before",
        "role_names": ["source", "target"],
        "role_types": {"source": ["node"], "target": ["node"]},
        "anchor_roles": ["source", "target"],
        "objective_roles": [],
        "symmetry": "none",
        "normalization_policy": {"operand_order": "preserve", "fact_key": ["predicate", "source", "target"]},
        "explicit_empty_policy": "not_applicable",
        "semantic_expressible": True,
        "model_facing_desired_fact": True,
        "lowering_supported": False,
    },
}

CONSTRAINT_SIGNATURES: dict[str, dict[str, Any]] = {
    "preserve.entity": {
        "constraint_signature_id": "preserve.entity",
        "role_names": ["target"],
        "role_types": {"target": ["node", "edge", "workflow", "unknown"]},
        "role_cardinality": {"target": "single"},
        "graph_dependency": "optional",
        "validator_key": "preserve_entity",
        "normalization_policy": {"operand_order": "preserve"},
        "semantic_expressible": True,
        "validation_supported": False,
        "lowering_supported": False,
    },
    "preserve.join": {
        "constraint_signature_id": "preserve.join",
        "role_names": ["target"],
        "role_types": {"target": ["node", "workflow", "unknown"]},
        "role_cardinality": {"target": "single"},
        "graph_dependency": "required_when_target_specific",
        "validator_key": "preserve_join",
        "normalization_policy": {"operand_order": "preserve"},
        "semantic_expressible": True,
        "validation_supported": "limited_text_hint_in_ordering_lowering",
        "lowering_supported": False,
    },
    "limit.scope": {
        "constraint_signature_id": "limit.scope",
        "role_names": ["target"],
        "role_types": {"target": ["node", "edge", "workflow", "unknown"]},
        "role_cardinality": {"target": "single"},
        "graph_dependency": "optional",
        "validator_key": "limit_scope",
        "normalization_policy": {"operand_order": "preserve"},
        "semantic_expressible": True,
        "validation_supported": False,
        "lowering_supported": False,
    },
}

_RELATION_PREDICATES = {
    "before": "relation.before",
    "after": "relation.after",
    "right_of": "relation.right_of",
    "route_to": "relation.route_to",
    "parallel_with": "relation.parallel_with",
    "connected_to": "relation.connected_to",
    "joins_before": "relation.joins_before",
}


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str) and item.strip()]


def normalize_relation_member(member: dict[str, Any]) -> dict[str, Any]:
    properties = member.get("properties")
    return {
        "source_ref": str(member.get("source_ref") or "").strip(),
        "target_ref": str(member.get("target_ref") or "").strip(),
        "properties": deepcopy(properties) if isinstance(properties, dict) else {},
        "provenance_refs": _string_list(member.get("provenance_refs")),
    }


def normalize_relation_fact(relation_fact: dict[str, Any]) -> dict[str, Any]:
    members = relation_fact.get("members") if isinstance(relation_fact.get("members"), list) else []
    group_semantics = relation_fact.get("group_semantics")
    return {
        "relation_id": str(relation_fact.get("relation_id") or "").strip(),
        "kind": str(relation_fact.get("kind") or "").strip(),
        "members": [
            normalize_relation_member(member)
            for member in members
            if isinstance(member, dict)
        ],
        "group_semantics": deepcopy(group_semantics) if isinstance(group_semantics, dict) else {},
        "provenance_refs": _string_list(relation_fact.get("provenance_refs")),
    }


def relation_fact_endpoint_views(relation_fact: dict[str, Any]) -> dict[str, Any]:
    normalized = normalize_relation_fact(relation_fact)

    def unique(values: list[str]) -> list[str]:
        seen: set[str] = set()
        result: list[str] = []
        for value in values:
            if value and value not in seen:
                seen.add(value)
                result.append(value)
        return result

    sources = unique([member["source_ref"] for member in normalized["members"]])
    targets = unique([member["target_ref"] for member in normalized["members"]])
    return {"sources": sources, "targets": targets, "cardinality": (len(sources), len(targets))}


def validate_relation_fact(relation_fact: Any) -> list[dict[str, Any]]:
    errors: list[dict[str, Any]] = []
    if not isinstance(relation_fact, dict):
        return [{"kind": "invalid_relation_fact", "message": "relation fact must be an object"}]
    forbidden_endpoint_fields = {"sources", "targets", "cardinality"}
    for field in sorted(forbidden_endpoint_fields & set(relation_fact)):
        errors.append({"kind": "derived_relation_fact_field", "field": field})
    relation_id = relation_fact.get("relation_id")
    if not isinstance(relation_id, str) or not relation_id.strip():
        errors.append({"kind": "missing_relation_id"})
    kind = relation_fact.get("kind")
    if not isinstance(kind, str) or kind.strip() not in RELATION_FACT_KINDS:
        errors.append({"kind": "invalid_relation_kind", "relation_kind": kind})
    members = relation_fact.get("members")
    if not isinstance(members, list) or not members:
        errors.append({"kind": "empty_relation_members"})
        members = []
    group_semantics = relation_fact.get("group_semantics", {})
    if not isinstance(group_semantics, dict):
        errors.append({"kind": "invalid_group_semantics"})
    provenance_refs = relation_fact.get("provenance_refs", [])
    if "provenance_refs" in relation_fact and (
        not isinstance(provenance_refs, list)
        or not all(isinstance(item, str) and item.strip() for item in provenance_refs)
    ):
        errors.append({"kind": "invalid_relation_provenance_refs"})
    for index, member in enumerate(members):
        location = f"members[{index}]"
        if not isinstance(member, dict):
            errors.append({"kind": "invalid_relation_member", "location": location})
            continue
        if not isinstance(member.get("source_ref"), str) or not member.get("source_ref", "").strip():
            errors.append({"kind": "missing_member_source_ref", "location": location})
        if not isinstance(member.get("target_ref"), str) or not member.get("target_ref", "").strip():
            errors.append({"kind": "missing_member_target_ref", "location": location})
        if "properties" in member and not isinstance(member.get("properties"), dict):
            errors.append({"kind": "invalid_member_properties", "location": location})
        member_provenance_refs = member.get("provenance_refs", [])
        if "provenance_refs" in member and (
            not isinstance(member_provenance_refs, list)
            or not all(isinstance(item, str) and item.strip() for item in member_provenance_refs)
        ):
            errors.append({"kind": "invalid_member_provenance_refs", "location": location})
    return errors


def model_facing_predicate_signatures() -> dict[str, dict[str, Any]]:
    return {
        predicate_id: signature
        for predicate_id, signature in PREDICATE_SIGNATURES.items()
        if signature.get("model_facing_desired_fact", signature.get("semantic_expressible") is True) is True
    }


def canonicalize_desired_fact(fact: dict[str, Any]) -> dict[str, Any]:
    canonical = deepcopy(fact)
    predicate = str(canonical.get("predicate") or "")
    signature = PREDICATE_SIGNATURES.get(predicate)
    policy = signature.get("canonicalization_policy") if isinstance(signature, dict) else None
    if not isinstance(policy, dict):
        return canonical
    canonical_predicate = str(policy.get("canonical_predicate") or predicate)
    role_map = policy.get("operand_role_map") if isinstance(policy.get("operand_role_map"), dict) else {}
    operands = canonical.get("operands") if isinstance(canonical.get("operands"), dict) else {}
    canonical["predicate"] = canonical_predicate
    canonical["predicate_signature_id"] = canonical_predicate
    canonical["operands"] = {
        canonical_role: deepcopy(operands.get(source_role))
        for canonical_role, source_role in role_map.items()
        if source_role in operands
    }
    return canonical


def predicate_fact_key(fact: dict[str, Any]) -> str:
    fact = canonicalize_desired_fact(fact)
    signature = PREDICATE_SIGNATURES.get(str(fact.get("predicate") or ""))
    if not signature:
        return canonical_json({"predicate": fact.get("predicate"), "operands": fact.get("operands", {})})
    operands = fact.get("operands") if isinstance(fact.get("operands"), dict) else {}
    key_roles = signature.get("normalization_policy", {}).get("fact_key", [])
    return canonical_json(
        {
            "predicate": fact.get("predicate"),
            "anchors": {
                role: operands.get(role)
                for role in key_roles
                if role != "predicate"
            },
        }
    )


def build_canonical_shadow_artifact(
    semantic_goal: dict[str, Any],
    *,
    instruction: str | None = None,
    source_ref: str = "production_semantic_goal_v2",
) -> dict[str, Any]:
    try:
        goal_state, warnings, evidence_bundle = _semantic_goal_to_canonical_goal_state_with_evidence(
            semantic_goal,
            instruction=instruction,
            source_ref=source_ref,
        )
        return {
            "enabled": True,
            "ok": True,
            "revision": CANONICAL_GOAL_STATE_REVISION,
            "goal_state": goal_state,
            "evidence_bundle": evidence_bundle,
            "warnings": warnings,
            "representation_errors": [],
            "comparison": _comparison_summary(goal_state, semantic_goal),
            "validation": _validate_goal_state(goal_state, warnings),
        }
    except Exception as exc:  # pragma: no cover - caller also protects fail-open behavior.
        return {
            "enabled": True,
            "ok": False,
            "revision": CANONICAL_GOAL_STATE_REVISION,
            "failure_stage": "canonical_goal_state",
            "error": str(exc),
        }


def semantic_goal_to_canonical_goal_state(
    semantic_goal: dict[str, Any],
    *,
    instruction: str | None = None,
    source_ref: str = "production_semantic_goal_v2",
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    goal_state, warnings, _evidence_bundle = _semantic_goal_to_canonical_goal_state_with_evidence(
        semantic_goal,
        instruction=instruction,
        source_ref=source_ref,
    )
    return goal_state, warnings


def _semantic_goal_to_canonical_goal_state_with_evidence(
    semantic_goal: dict[str, Any],
    *,
    instruction: str | None = None,
    source_ref: str = "production_semantic_goal_v2",
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    goals = semantic_goal.get("semantic_goals_v2") if isinstance(semantic_goal.get("semantic_goals_v2"), list) else []
    builder = _CanonicalGoalBuilder(instruction=instruction, source_ref=source_ref)
    if not goals:
        builder.add_open_statement("empty_semantic_goal", _goal_text(semantic_goal))
    for goal in goals:
        if not isinstance(goal, dict):
            continue
        builder.index_goal_objects(goal)
    for goal in goals:
        if not isinstance(goal, dict):
            continue
        builder.add_goal(goal)
    return builder.finish(), builder.warnings, builder.evidence_bundle()


class _CanonicalGoalBuilder:
    def __init__(self, *, instruction: str | None, source_ref: str) -> None:
        self.instruction = instruction if isinstance(instruction, str) else ""
        self.source_ref = source_ref
        self.provenance: list[dict[str, Any]] = []
        self.references: list[dict[str, Any]] = []
        self.desired_facts: list[dict[str, Any]] = []
        self.relation_facts: list[dict[str, Any]] = []
        self.constraints: list[dict[str, Any]] = []
        self.open_statements: list[dict[str, Any]] = []
        self.ref_by_object_key: dict[str, str] = {}
        self.object_by_key: dict[str, dict[str, Any]] = {}
        self.warnings: list[dict[str, Any]] = []
        self._prov_seq = 0
        self._ref_seq = 0
        self._fact_seq = 0
        self._relation_fact_seq = 0
        self._constraint_seq = 0
        self._statement_seq = 0
        self.default_provenance_id = self._add_provenance("semantic_goal_adapter", source_ref, self._utterance_span())

    def finish(self) -> dict[str, Any]:
        goal_state = {
            "goal_state_id": "gs1",
            "revision": CANONICAL_GOAL_STATE_REVISION,
            "references": self.references,
            "desired_facts": self.desired_facts,
            "constraints": self.constraints,
            "open_statements": self.open_statements,
        }
        if self.relation_facts:
            goal_state["relation_facts"] = self.relation_facts
        return goal_state

    def evidence_bundle(self) -> list[dict[str, Any]]:
        return deepcopy(self.provenance)

    def index_goal_objects(self, goal: dict[str, Any]) -> None:
        for item in _list_field(goal, "objects"):
            if isinstance(item, dict):
                self._ensure_reference(item, entity_type_hint=None)

    def add_goal(self, goal: dict[str, Any]) -> None:
        objects = {
            str(item.get("id") or ""): item
            for item in _list_field(goal, "objects")
            if isinstance(item, dict) and str(item.get("id") or "")
        }
        for state in _list_field(goal, "object_desired_states"):
            if isinstance(state, dict):
                self._add_state(state, objects)
        for relation in _list_field(goal, "relation_intents"):
            if isinstance(relation, dict):
                self._add_relation(relation, objects)
        for relation_fact in _list_field(goal, "relation_facts"):
            if isinstance(relation_fact, dict):
                self._add_grouped_relation_fact(relation_fact, objects)
        for constraint in _list_field(goal, "semantic_constraints"):
            if isinstance(constraint, dict):
                self._add_constraint(constraint, objects)
        for unresolved in _list_field(goal, "unresolved_dimensions"):
            if isinstance(unresolved, dict):
                self.add_open_statement("unresolved_semantic_dimension", _compact_dict(unresolved))
        text = str(goal.get("text") or "").strip()
        if text and not (self.desired_facts or self.constraints or self.open_statements):
            self.add_open_statement("semantic_goal_text", text)

    def _add_state(self, state: dict[str, Any], objects: dict[str, dict[str, Any]]) -> None:
        object_id = str(state.get("object") or "").strip()
        item = objects.get(object_id)
        ref_id = self._ensure_reference(item, entity_type_hint="node") if item else self._implicit_reference("unknown", object_id)
        state_name = str(state.get("state") or "").strip()
        if state_name == "preserved":
            self._append_constraint("preserve.entity", {"target": _reference_operand(ref_id)})
            return
        if state_name == "exists":
            self._append_fact("entity.exists", {"subject": _reference_operand(ref_id)})
            return
        if state_name == "absent":
            self._append_fact("entity.absent", {"subject": _reference_operand(ref_id)})
            return
        if state_name == "property_equals":
            prop = str(state.get("property") or "").strip()
            value = state.get("value")
            if _is_structured_explicit_empty(value) or state.get("explicit_empty") is True:
                entity_type = "edge" if item and item.get("type") == "edge_reference" else "node"
                ref_id = self._ensure_reference(item, entity_type_hint=entity_type) if item else ref_id
                self._append_fact(
                    "property.equals",
                    {
                        "subject": _reference_operand(ref_id),
                        "property": _literal_operand("property_name", prop),
                        "value": _explicit_empty_operand(),
                    },
                )
                return
            if value == "":
                self.warnings.append(
                    {
                        "kind": "ambiguous_empty_semantics",
                        "message": "production Semantic Goal v2 does not distinguish explicit empty from missing value",
                        "object": object_id,
                        "property": prop,
                    }
                )
                self.add_open_statement("unverified_empty_property_value", _compact_dict(state))
                return
            if not prop or not isinstance(value, str):
                self.add_open_statement("unmappable_property_equals", _compact_dict(state))
                return
            entity_type = "edge" if item and item.get("type") == "edge_reference" else "node"
            ref_id = self._ensure_reference(item, entity_type_hint=entity_type) if item else ref_id
            self._append_fact(
                "property.equals",
                {
                    "subject": _reference_operand(ref_id),
                    "property": _literal_operand("property_name", prop),
                    "value": _literal_operand("string", value),
                },
            )
            return
        self.add_open_statement("unsupported_object_state", _compact_dict(state))

    def _add_relation(self, relation: dict[str, Any], objects: dict[str, dict[str, Any]]) -> None:
        relation_name = str(relation.get("relation") or "").strip()
        predicate = _RELATION_PREDICATES.get(relation_name)
        if predicate is None:
            self.add_open_statement("unsupported_relation_intent", _compact_dict(relation))
            return
        source_id = str(relation.get("source") or "").strip()
        target_id = str(relation.get("target") or "").strip()
        source_type, target_type = _relation_operand_types(relation_name)
        source_ref = self._ensure_reference(objects.get(source_id), entity_type_hint=source_type) if source_id in objects else self._implicit_reference(source_type, source_id)
        target_ref = self._ensure_reference(objects.get(target_id), entity_type_hint=target_type) if target_id in objects else self._implicit_reference(target_type, target_id)
        if relation_name in {"before", "after"}:
            operands = {"subject": _reference_operand(source_ref), "anchor": _reference_operand(target_ref)}
        elif relation_name == "right_of":
            operands = {"target": _reference_operand(source_ref), "anchor": _reference_operand(target_ref)}
        elif relation_name == "route_to":
            operands = {"subject": _reference_operand(source_ref), "target": _reference_operand(target_ref)}
        elif relation_name == "parallel_with":
            left, right = sorted([source_ref, target_ref])
            operands = {"left": _reference_operand(left), "right": _reference_operand(right)}
        else:
            operands = {"source": _reference_operand(source_ref), "target": _reference_operand(target_ref)}
        fact = self._append_fact(predicate, operands)
        if relation_name in {"before", "after"} and fact is not None:
            self._append_shadow_precedence_relation_fact(fact)

    def _add_grouped_relation_fact(self, relation_fact: dict[str, Any], objects: dict[str, dict[str, Any]]) -> None:
        kind = str(relation_fact.get("kind") or "").strip()
        if kind not in RELATION_FACT_KINDS:
            self.add_open_statement("unsupported_grouped_relation_kind", _compact_dict(relation_fact))
            return
        members: list[dict[str, Any]] = []
        for member in _list_field(relation_fact, "members"):
            if not isinstance(member, dict):
                continue
            source_id = str(member.get("source") or "").strip()
            target_id = str(member.get("target") or "").strip()
            if source_id not in objects or target_id not in objects:
                self.add_open_statement("unmappable_grouped_relation_member", _compact_dict(member))
                continue
            source_ref = self._ensure_reference(objects.get(source_id), entity_type_hint="node")
            target_ref = self._ensure_reference(objects.get(target_id), entity_type_hint="node")
            properties = member.get("properties")
            members.append(
                {
                    "source_ref": source_ref,
                    "target_ref": target_ref,
                    "properties": deepcopy(properties) if isinstance(properties, dict) else {},
                    "provenance_refs": [self.default_provenance_id],
                }
            )
        if not members:
            return
        self._relation_fact_seq += 1
        group_semantics = relation_fact.get("group_semantics")
        self.relation_facts.append(
            normalize_relation_fact(
                {
                    "relation_id": f"rg{self._relation_fact_seq}",
                    "kind": kind,
                    "members": members,
                    "group_semantics": deepcopy(group_semantics) if isinstance(group_semantics, dict) else {},
                    "provenance_refs": [self.default_provenance_id],
                }
            )
        )

    def _add_constraint(self, constraint: dict[str, Any], objects: dict[str, dict[str, Any]]) -> None:
        constraint_type = str(constraint.get("type") or "").strip()
        target = str(constraint.get("target") or "").strip()
        signature_id = constraint_type if constraint_type in CONSTRAINT_SIGNATURES else ""
        if not signature_id and constraint_type == "preserve" and target in objects:
            signature_id = "preserve.entity"
        if not signature_id:
            self.add_open_statement("unstructured_semantic_constraint", _compact_dict(constraint))
            return
        target_ref = self._ensure_reference(objects.get(target), entity_type_hint=None) if target in objects else self._implicit_reference("unknown", target)
        self._append_constraint(signature_id, {"target": _reference_operand(target_ref)})

    def _append_fact(self, predicate: str, operands: dict[str, Any]) -> dict[str, Any] | None:
        canonical = canonicalize_desired_fact({"predicate": predicate, "operands": operands})
        predicate = str(canonical.get("predicate") or predicate)
        operands = canonical.get("operands") if isinstance(canonical.get("operands"), dict) else operands
        signature = PREDICATE_SIGNATURES.get(predicate)
        if signature is None:
            self.add_open_statement("unregistered_predicate", {"predicate": predicate, "operands": operands})
            return None
        self._fact_seq += 1
        fact = {
            "fact_id": f"f{self._fact_seq}",
            "predicate": predicate,
            "predicate_signature_id": signature["predicate_id"],
            "operands": operands,
            "provenance_refs": [self.default_provenance_id],
        }
        fact["fact_key"] = predicate_fact_key(fact)
        self.desired_facts.append(fact)
        return fact

    def _append_shadow_precedence_relation_fact(self, fact: dict[str, Any]) -> None:
        if fact.get("predicate") != "relation.before":
            return
        operands = fact.get("operands") if isinstance(fact.get("operands"), dict) else {}
        subject = operands.get("subject") if isinstance(operands.get("subject"), dict) else {}
        anchor = operands.get("anchor") if isinstance(operands.get("anchor"), dict) else {}
        source_ref = subject.get("ref_id")
        target_ref = anchor.get("ref_id")
        if not (isinstance(source_ref, str) and source_ref and isinstance(target_ref, str) and target_ref):
            return
        provenance_refs = _string_list(fact.get("provenance_refs"))
        relation_fact = normalize_relation_fact(
            {
                "relation_id": f"rf:{fact['fact_id']}",
                "kind": "precedence",
                "members": [
                    {
                        "source_ref": source_ref,
                        "target_ref": target_ref,
                        "properties": {},
                        "provenance_refs": provenance_refs,
                    }
                ],
                "group_semantics": {},
                "provenance_refs": provenance_refs,
            }
        )
        self.relation_facts.append(relation_fact)

    def _append_constraint(self, signature_id: str, operands: dict[str, Any]) -> None:
        signature = CONSTRAINT_SIGNATURES.get(signature_id)
        if signature is None:
            self.add_open_statement("unregistered_constraint", {"constraint_predicate": signature_id, "operands": operands})
            return
        self._constraint_seq += 1
        self.constraints.append(
            {
                "constraint_id": f"c{self._constraint_seq}",
                "constraint_predicate": signature_id,
                "constraint_signature_id": signature["constraint_signature_id"],
                "operands": operands,
                "provenance_refs": [self.default_provenance_id],
            }
        )

    def add_open_statement(self, statement_type: str, content: Any) -> None:
        self._statement_seq += 1
        self.open_statements.append(
            {
                "statement_id": f"s{self._statement_seq}",
                "statement_type": statement_type,
                "content": content,
                "provenance_refs": [self.default_provenance_id],
            }
        )

    def _ensure_reference(self, item: dict[str, Any] | None, *, entity_type_hint: str | None) -> str:
        if not isinstance(item, dict):
            return self._implicit_reference(entity_type_hint or "unknown", "")
        object_key = str(item.get("id") or canonical_json(item))
        if object_key in self.ref_by_object_key:
            ref_id = self.ref_by_object_key[object_key]
            if entity_type_hint:
                self._refine_reference_type(ref_id, entity_type_hint)
            return ref_id
        object_type = str(item.get("type") or "").strip()
        entity_type = entity_type_hint or _entity_type_from_object_type(object_type)
        text = _object_text(item)
        mention = None if object_type == "unresolved_reference" else self._mention(text)
        self._ref_seq += 1
        ref_id = f"r{self._ref_seq}"
        self.ref_by_object_key[object_key] = ref_id
        self.object_by_key[object_key] = deepcopy(item)
        self.references.append(
            {
                "ref_id": ref_id,
                "entity_type": entity_type,
                "cardinality": "single",
                "mention": mention,
                "provenance_refs": [self.default_provenance_id],
            }
        )
        return ref_id

    def _implicit_reference(self, entity_type: str, source_id: str) -> str:
        key = f"implicit:{entity_type}:{source_id}"
        if key in self.ref_by_object_key:
            return self.ref_by_object_key[key]
        self._ref_seq += 1
        ref_id = f"r{self._ref_seq}"
        self.ref_by_object_key[key] = ref_id
        self.references.append(
            {
                "ref_id": ref_id,
                "entity_type": entity_type if entity_type in {"node", "edge", "workflow", "unknown"} else "unknown",
                "cardinality": "single",
                "mention": None,
                "provenance_refs": [self.default_provenance_id],
            }
        )
        return ref_id

    def _refine_reference_type(self, ref_id: str, entity_type: str) -> None:
        if entity_type not in {"node", "edge", "workflow", "unknown"}:
            return
        for ref in self.references:
            if ref.get("ref_id") == ref_id and ref.get("entity_type") == "unknown":
                ref["entity_type"] = entity_type
                return

    def _add_provenance(self, source_type: str, source_ref: str, evidence_span: dict[str, Any] | None) -> str:
        self._prov_seq += 1
        provenance_id = f"p{self._prov_seq}"
        self.provenance.append(
            {
                "provenance_id": provenance_id,
                "source_type": source_type,
                "source_ref": source_ref,
                "evidence_span": evidence_span,
                "graph_version": None,
            }
        )
        return provenance_id

    def _utterance_span(self) -> dict[str, Any] | None:
        if not self.instruction:
            return None
        return {"utterance_id": "u1", "start": 0, "end": len(self.instruction), "text": self.instruction}

    def _mention(self, text: str) -> dict[str, Any] | None:
        if not text:
            return None
        span = None
        if self.instruction:
            index = self.instruction.find(text)
            if index >= 0:
                span = {"utterance_id": "u1", "start": index, "end": index + len(text), "text": text}
        return {"text": text, "span": span}


def _entity_type_from_object_type(object_type: str) -> str:
    if object_type == "edge_reference":
        return "edge"
    if object_type in {"existing_reference", "new_entity"}:
        return "node"
    return "unknown"


def _relation_operand_types(relation: str) -> tuple[str, str]:
    if relation == "route_to":
        return "edge", "node"
    return "node", "node"


def _object_text(item: dict[str, Any]) -> str:
    for key in ("reference", "label", "text"):
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _reference_operand(ref_id: str) -> dict[str, str]:
    return {"operand_type": "reference", "ref_id": ref_id}


def _literal_operand(literal_type: str, value: str) -> dict[str, str]:
    return {"operand_type": "literal", "literal_type": literal_type, "value": value}


def _explicit_empty_operand() -> dict[str, str]:
    return {"operand_type": "explicit_empty"}


def _is_structured_explicit_empty(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    marker = str(value.get("operand_type") or value.get("type") or value.get("kind") or "").strip()
    return marker == "explicit_empty"


def _compact_dict(value: dict[str, Any]) -> dict[str, Any]:
    return {str(key): deepcopy(child) for key, child in value.items() if child not in (None, "", [], {})}


def _list_field(value: dict[str, Any], key: str) -> list[Any]:
    child = value.get(key)
    return child if isinstance(child, list) else []


def _goal_text(value: dict[str, Any]) -> str:
    clarification = value.get("clarification") if isinstance(value.get("clarification"), dict) else {}
    for key in ("question", "reason", "message"):
        text = clarification.get(key)
        if isinstance(text, str) and text.strip():
            return text.strip()
    return ""


def _comparison_summary(goal_state: dict[str, Any], semantic_goal: dict[str, Any]) -> dict[str, Any]:
    goals = semantic_goal.get("semantic_goals_v2") if isinstance(semantic_goal.get("semantic_goals_v2"), list) else []
    return {
        "source_goal_count": len(goals),
        "reference_count": len(goal_state.get("references", [])),
        "desired_fact_count": len(goal_state.get("desired_facts", [])),
        "constraint_count": len(goal_state.get("constraints", [])),
        "open_statement_count": len(goal_state.get("open_statements", [])),
        "fact_predicates": [fact.get("predicate") for fact in goal_state.get("desired_facts", []) if isinstance(fact, dict)],
    }


def _validate_goal_state(goal_state: dict[str, Any], warnings: list[dict[str, Any]]) -> dict[str, Any]:
    errors: list[dict[str, Any]] = []
    forbidden_top_level = {"provenance", "provenance_refs", "status", "graph_version", "bindings", "candidates", "pending", "confirmation"}
    for key in forbidden_top_level:
        if key in goal_state:
            errors.append({"kind": "forbidden_top_level_field", "field": key})
    for ref in _list_field(goal_state, "references"):
        if not isinstance(ref, dict):
            continue
        if "semantic_role" in ref or "binding" in ref or "candidate_id" in ref:
            errors.append({"kind": "reference_runtime_leakage", "ref_id": ref.get("ref_id")})
    for fact in _list_field(goal_state, "desired_facts"):
        if not isinstance(fact, dict):
            continue
        predicate = fact.get("predicate")
        if predicate not in PREDICATE_SIGNATURES:
            errors.append({"kind": "unknown_predicate", "predicate": predicate})
    for index, relation_fact in enumerate(_list_field(goal_state, "relation_facts")):
        for error in validate_relation_fact(relation_fact):
            errors.append({"kind": "invalid_relation_fact", "index": index, "error": error})
    return {"ok": not errors, "errors": errors, "warnings": warnings}
