from __future__ import annotations

import json
from typing import Any

from .canonical_goal_state import CONSTRAINT_SIGNATURES, model_facing_predicate_signatures


REFERENCE_ENTITY_TYPES = ("node", "edge", "workflow", "unknown")
REFERENCE_CARDINALITIES = ("single", "set")
SEMANTIC_DELTA_OPS = ("ADD", "CORRECT", "RETRACT")
SEMANTIC_ELEMENT_KINDS = ("reference", "desired_fact", "constraint", "open_statement")
TURN_RELATIONS = ("CONTINUE", "CORRECT", "CANCEL", "NEW_TASK")
STAGE_TYPES = ("GROUNDING_CANDIDATE", "PREFLIGHT_CONFIRMATION")
STAGE_RESULTS = ("resolved", "unresolved", "remain_pending", "cancelled", "authorized", "declined")

STAGE_RESULT_EQUIVALENCE = {
    "unresolved": {"unresolved", "remain_pending"},
}

CANONICAL_DISPLAY_PROPERTY = "display_label"
PROPERTY_ALIASES_BY_ENTITY_TYPE = {
    "node": {
        "display_label": CANONICAL_DISPLAY_PROPERTY,
        "label": CANONICAL_DISPLAY_PROPERTY,
        "name": CANONICAL_DISPLAY_PROPERTY,
    },
    "edge": {
        "display_label": CANONICAL_DISPLAY_PROPERTY,
        "branch_label": CANONICAL_DISPLAY_PROPERTY,
        "label": CANONICAL_DISPLAY_PROPERTY,
    },
}


def canonical_property_name(entity_type: str | None, property_name: Any) -> str:
    """Return the shadow semantic identity for a display property literal."""
    raw = str(property_name or "").strip()
    return PROPERTY_ALIASES_BY_ENTITY_TYPE.get(str(entity_type or ""), {}).get(raw, raw)


def canonical_open_statement_type(_statement_type: Any) -> str:
    """Open statement subtypes are model wording, not separate semantic truth."""
    return "open"


def build_model_facing_semantic_grammar() -> str:
    """Render the frozen shadow semantic contract into deterministic prompt text."""
    contract = {
        "reference": {
            "required_fields": {
                "ref_id": "string, optional for ADD because the merge layer can assign it",
                "entity_type": list(REFERENCE_ENTITY_TYPES),
                "cardinality": list(REFERENCE_CARDINALITIES),
                "mention": "null for implicit references, otherwise {text, span}",
                "provenance_refs": "array of evidence ids, optional",
            },
            "forbidden_fields": ["binding", "candidate_id", "node_id", "edge_id", "semantic_role"],
            "implicit_reference": {"mention": None, "cardinality": "single or set from user meaning"},
        },
        "operand_objects": {
            "reference": {"operand_type": "reference", "ref_id": "<ref_id>"},
            "literal": {"operand_type": "literal", "literal_type": "<string|property_name|...>", "value": "<value>"},
            "explicit_empty": {"operand_type": "explicit_empty"},
        },
        "canonical_property_identity": {
            "canonical_user_visible_display_property": CANONICAL_DISPLAY_PROPERTY,
            "aliases_by_entity_type": PROPERTY_ALIASES_BY_ENTITY_TYPE,
            "policy": "Use display_label for user-visible names/labels. Runtime lowering may map it to implementation fields.",
        },
        "desired_fact_value_shape": {
            "predicate": "one exact predicate ID from predicates[].predicate",
            "predicate_signature_id": "same value as predicate, optional but recommended",
            "operands": "object whose keys exactly match required_operand_roles",
        },
        "constraint_value_shape": {
            "constraint_predicate": "one exact constraint ID from constraints[].constraint_predicate",
            "constraint_signature_id": "same value as constraint_predicate, optional but recommended",
            "operands": "object whose keys exactly match required_operand_roles",
        },
        "predicates": [_predicate_contract(predicate_id, signature) for predicate_id, signature in sorted(model_facing_predicate_signatures().items())],
        "constraints": [_constraint_contract(signature_id, signature) for signature_id, signature in sorted(CONSTRAINT_SIGNATURES.items())],
        "turn_relation": list(TURN_RELATIONS),
        "stage_continuation": {"stage_types": list(STAGE_TYPES), "results": list(STAGE_RESULTS)},
    }
    return (
        "Frozen semantic contract grammar. Use these exact IDs and shapes only.\n"
        "Do not invent predicates, entity types, cardinalities, operand role names, or delta ops.\n"
        "This grammar is generated from the frozen registries used by the validator.\n"
        f"{json.dumps(contract, ensure_ascii=False, indent=2, sort_keys=True)}\n"
        "Canonical desired_fact examples by schema shape only:\n"
        "- Output desired facts as {\"predicate\":\"relation.route_to\",\"predicate_signature_id\":\"relation.route_to\",\"operands\":{\"subject\":{\"operand_type\":\"reference\",\"ref_id\":\"...\"},\"target\":{\"operand_type\":\"reference\",\"ref_id\":\"...\"}}}.\n"
        "- Output properties as {\"predicate\":\"property.equals\",\"predicate_signature_id\":\"property.equals\",\"operands\":{\"subject\":{\"operand_type\":\"reference\",\"ref_id\":\"...\"},\"property\":{\"operand_type\":\"literal\",\"literal_type\":\"property_name\",\"value\":\"...\"},\"value\":{\"operand_type\":\"literal\",\"literal_type\":\"string\",\"value\":\"...\"}}}.\n"
        "- Output removals as {\"predicate\":\"entity.absent\",\"predicate_signature_id\":\"entity.absent\",\"operands\":{\"subject\":{\"operand_type\":\"reference\",\"ref_id\":\"...\"}}}.\n"
        "- relation.before uses operands subject=<reference to node>, anchor=<reference to node>.\n"
        "- relation.parallel_with uses operands left=<reference to node>, right=<reference to node>.\n"
        "Do not output desired_fact.value.predicate_id without desired_fact.value.predicate; predicate_id is registry metadata, while predicate is the output field name.\n"
        "For user-visible node or edge names/labels, prefer property value display_label; label/name/branch_label are normalized as aliases only for proven display-property identity.\n"
        "ExplicitEmpty is valid only as the property.equals value operand when the user explicitly commits to an empty value. Do not convert vague changes, preserve/show/hide preferences, or missing values into ExplicitEmpty.\n"
        "Selection is not intent. When the user already expresses a semantic relation or property change but omits a participant, selection and graph facts may constrain the implicit Reference entity type and scope.\n"
        "When a predicate role has exactly one allowed entity type in the registry, references used in that role should use that entity type; do not label an omitted role as a different object type.\n"
        "When the utterance expresses a broad change request but no registered desired_fact or constraint is yet justified, preserve it as an open_statement instead of returning an empty semantic proposal.\n"
        "When a candidate or confirmation reply contains residual semantic content, consume only the stage-answer span and represent the residual content with the same grammar.\n"
        "If the user is not answering the current waiting stage and instead expresses an independent goal, turn_relation.relation is NEW_TASK.\n"
    )


def stage_result_matches(expected: str, actual: str | None) -> bool:
    if actual is None:
        return False
    if actual == expected:
        return True
    return actual in STAGE_RESULT_EQUIVALENCE.get(expected, set())


def _predicate_contract(predicate_id: str, signature: dict[str, Any]) -> dict[str, Any]:
    return {
        "predicate": predicate_id,
        "required_operand_roles": list(signature.get("role_names", [])),
        "role_types": signature.get("role_types", {}),
        "symmetry": signature.get("symmetry"),
        "fact_key": signature.get("normalization_policy", {}).get("fact_key", []),
        "explicit_empty_policy": signature.get("explicit_empty_policy"),
    }


def _constraint_contract(signature_id: str, signature: dict[str, Any]) -> dict[str, Any]:
    return {
        "constraint_predicate": signature_id,
        "required_operand_roles": list(signature.get("role_names", [])),
        "role_types": signature.get("role_types", {}),
        "role_cardinality": signature.get("role_cardinality", {}),
    }
