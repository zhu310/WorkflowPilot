from __future__ import annotations

from src.phase1b_shadow import (
    PHASE1B_EXTRACTION_SYSTEM,
    PHASE1B_MODEL_GRAMMAR,
    PHASE1B_RELATION_GROUPER_SYSTEM,
    build_shadow_relation_group_artifact,
    empty_goal_state,
    relation_grouper_allowed_refs,
    relation_grouper_payload,
    run_phase1b_shadow,
    semantic_equivalence_key,
)


def ref(ref_id: str, text: str, *, start: int = 0) -> dict:
    return {
        "ref_id": ref_id,
        "entity_type": "node",
        "cardinality": "single",
        "mention": {"text": text, "span": {"utterance_id": "u1", "start": start, "end": start + len(text)}},
        "provenance_refs": ["p1"],
    }


def desired_before(source_ref: str, target_ref: str) -> dict:
    return {
        "predicate": "relation.before",
        "predicate_signature_id": "relation.before",
        "operands": {
            "subject": {"operand_type": "reference", "ref_id": source_ref},
            "anchor": {"operand_type": "reference", "ref_id": target_ref},
        },
        "provenance_refs": ["p1"],
    }


def desired_route_to(source_ref: str, target_ref: str) -> dict:
    return {
        "predicate": "relation.route_to",
        "predicate_signature_id": "relation.route_to",
        "operands": {
            "subject": {"operand_type": "reference", "ref_id": source_ref},
            "target": {"operand_type": "reference", "ref_id": target_ref},
        },
        "provenance_refs": ["p1"],
    }


def extraction(*, refs: list[dict], desired_facts: list[dict] | None = None, relation: str = "CONTINUE") -> dict:
    return {
        "turn_relation": {"relation": relation, "task_context_ref": None, "evidence_refs": ["p1"], "reason_codes": []},
        "semantic_proposal": {
            "references": refs,
            "desired_facts": desired_facts or [],
            "constraints": [],
            "open_statements": [],
        },
    }


def model_for(base_payload: dict, grouper_payload: dict):
    def model(system: str, _model_payload: dict):
        if "discourse reference continuity" in system:
            return {"discourse_reference_continuity": {"status": "none", "ref_ids": [], "evidence": ""}}
        if "Relation Grouper" in system:
            return grouper_payload
        return base_payload

    return model


def relation_fact(members: list[tuple[str, str]], *, relation_id: str) -> dict:
    return {
        "relation_id": relation_id,
        "kind": "precedence",
        "members": [
            {"source_ref": source, "target_ref": target, "properties": {}, "provenance_refs": ["p1"]}
            for source, target in members
        ],
        "group_semantics": {},
        "provenance_refs": ["p1"],
    }


def test_base_model_contract_hides_relation_fact_exposure() -> None:
    assert '"relation_facts": []' not in PHASE1B_EXTRACTION_SYSTEM
    assert "relation_fact_proposal_shape" not in PHASE1B_MODEL_GRAMMAR
    assert "Grouped relation_fact examples" not in PHASE1B_MODEL_GRAMMAR
    assert "semantic_delta" not in PHASE1B_EXTRACTION_SYSTEM
    assert "target_id" not in PHASE1B_EXTRACTION_SYSTEM


def test_shadow_grouper_contract_is_minimal_and_separate() -> None:
    assert "relation_groups" in PHASE1B_RELATION_GROUPER_SYSTEM
    assert "at least 2 members" in PHASE1B_RELATION_GROUPER_SYSTEM
    assert "Do not output relation_id" in PHASE1B_RELATION_GROUPER_SYSTEM
    assert "properties" in PHASE1B_RELATION_GROUPER_SYSTEM
    assert "Return JSON only" in PHASE1B_RELATION_GROUPER_SYSTEM


def test_shadow_grouper_payload_uses_only_base_refs_and_relation_candidates() -> None:
    proposal = {
        "references": [ref("r1", "A"), ref("r2", "B")],
        "desired_facts": [desired_before("r1", "r2"), {"predicate": "entity.exists", "operands": {}}],
    }

    payload = relation_grouper_payload("A before B", proposal)

    assert payload["raw_utterance"] == "A before B"
    assert payload["base_references"] == proposal["references"]
    assert payload["base_relation_candidates"] == [proposal["desired_facts"][0]]
    assert "graph_context" not in payload
    assert "selection" not in payload


def test_shadow_grouper_payload_can_include_minimal_active_prior_relation_context() -> None:
    proposal = {
        "references": [ref("r4", "D")],
        "desired_facts": [desired_before("r1", "r4"), desired_before("r2", "r4")],
    }
    goal_state = empty_goal_state()
    goal_state["references"] = [ref("r1", "A"), ref("r2", "B"), ref("r3", "C")]
    goal_state["relation_facts"] = [relation_fact([("r1", "r3"), ("r2", "r3")], relation_id="rg1")]

    payload = relation_grouper_payload("No, A and B before D.", proposal, planning_goal_state=goal_state)

    assert payload["base_references"] == proposal["references"]
    assert sorted(row["ref_id"] for row in payload["active_prior_relation_refs"]) == ["r1", "r2", "r3"]
    assert payload["active_prior_relations"] == [
        {"kind": "precedence", "members": [{"source": "r1", "target": "r3"}, {"source": "r2", "target": "r3"}]}
    ]
    assert "relation_id" not in payload["active_prior_relations"][0]
    assert "graph_context" not in payload
    assert "selection" not in payload


def test_shadow_grouper_allowed_refs_are_current_plus_active_prior_only() -> None:
    proposal = {
        "references": [ref("r4", "D")],
        "desired_facts": [desired_before("r1", "r4"), desired_before("r_stale", "r4")],
    }
    goal_state = empty_goal_state()
    goal_state["references"] = [ref("r1", "A"), ref("r2", "B"), ref("r3", "C"), ref("r_unused", "Unused")]
    goal_state["relation_facts"] = [relation_fact([("r1", "r3"), ("r2", "r3")], relation_id="rg1")]

    assert relation_grouper_allowed_refs(proposal, goal_state) == {"r1", "r2", "r3", "r4"}


def test_relation_group_proposal_adapts_to_one_shadow_relation_fact() -> None:
    proposal = {
        "references": [ref("r1", "A"), ref("r2", "B"), ref("r3", "C")],
        "desired_facts": [desired_before("r1", "r3"), desired_before("r2", "r3")],
    }
    parsed = {"relation_groups": [{"kind": "precedence", "members": [{"source": "r1", "target": "r3"}, {"source": "r2", "target": "r3"}]}]}

    artifact = build_shadow_relation_group_artifact(parsed, proposal)

    assert artifact["ok"] is True
    assert artifact["relation_groups"] == parsed["relation_groups"]
    assert len(artifact["managed_relation_fact_proposals"]) == 1
    relation = artifact["managed_relation_fact_proposals"][0]
    assert relation["relation_id"] == ""
    assert relation["kind"] == "precedence"
    assert [(member["source_ref"], member["target_ref"], member["properties"]) for member in relation["members"]] == [
        ("r1", "r3", {}),
        ("r2", "r3", {}),
    ]
    assert relation["group_semantics"] == {}


def test_relation_group_proposal_rejects_one_member_and_unknown_refs() -> None:
    proposal = {"references": [ref("r1", "A"), ref("r2", "B")], "desired_facts": []}
    one_member = {"relation_groups": [{"kind": "precedence", "members": [{"source": "r1", "target": "r2"}]}]}
    unknown = {"relation_groups": [{"kind": "precedence", "members": [{"source": "r1", "target": "r2"}, {"source": "r_missing", "target": "r2"}]}]}

    assert build_shadow_relation_group_artifact(one_member, proposal)["ok"] is False
    assert build_shadow_relation_group_artifact(unknown, proposal)["ok"] is False


def test_relation_group_proposal_accepts_prior_refs_from_active_relation_context() -> None:
    proposal = {
        "references": [ref("r4", "D")],
        "desired_facts": [desired_before("r1", "r4"), desired_before("r2", "r4")],
    }
    goal_state = empty_goal_state()
    goal_state["references"] = [ref("r1", "A"), ref("r2", "B"), ref("r3", "C")]
    goal_state["relation_facts"] = [relation_fact([("r1", "r3"), ("r2", "r3")], relation_id="rg1")]
    parsed = {"relation_groups": [{"kind": "precedence", "members": [{"source": "r1", "target": "r4"}, {"source": "r2", "target": "r4"}]}]}

    artifact = build_shadow_relation_group_artifact(parsed, proposal, planning_goal_state=goal_state)

    assert artifact["ok"] is True
    assert artifact["managed_relation_fact_proposals"][0]["relation_id"] == ""
    assert [(member["source_ref"], member["target_ref"]) for member in artifact["managed_relation_fact_proposals"][0]["members"]] == [
        ("r1", "r4"),
        ("r2", "r4"),
    ]


def test_relation_group_proposal_rejects_stale_and_invented_prior_refs_after_new_task_reset() -> None:
    proposal = {
        "references": [ref("r4", "D")],
        "desired_facts": [desired_before("r1", "r4"), desired_before("r_invented", "r4")],
    }
    parsed = {
        "relation_groups": [
            {"kind": "precedence", "members": [{"source": "r1", "target": "r4"}, {"source": "r_invented", "target": "r4"}]}
        ]
    }

    artifact = build_shadow_relation_group_artifact(parsed, proposal, planning_goal_state=empty_goal_state())

    assert artifact["ok"] is False
    invalid_refs = {error["ref_id"] for error in artifact["validation"]["errors"] if error["kind"] == "invalid_ref"}
    assert invalid_refs == {"r1", "r_invented"}


def test_shadow_grouper_enters_managed_lifecycle_without_srg_identity() -> None:
    base = extraction(
        refs=[ref("r1", "A"), ref("r2", "B"), ref("r3", "C")],
        desired_facts=[desired_before("r1", "r3"), desired_before("r2", "r3")],
    )
    grouped = {"relation_groups": [{"kind": "precedence", "members": [{"source": "r1", "target": "r3"}, {"source": "r2", "target": "r3"}]}]}

    run = run_phase1b_shadow("A and B before C", model_runner=model_for(base, grouped))

    assert run.ok
    assert run.artifact["shadow_relation_grouper"]["ok"] is True
    assert run.artifact["shadow_relation_grouper"]["managed_relation_fact_proposals"][0]["relation_id"] == ""
    assert run.artifact["after_goal_state"]["relation_facts"][0]["relation_id"] == "rg1"
    assert [(member["source_ref"], member["target_ref"]) for member in run.artifact["after_goal_state"]["relation_facts"][0]["members"]] == [
        ("r1", "r3"),
        ("r2", "r3"),
    ]


def test_shadow_grouper_uses_valid_group_when_invalid_singleton_is_also_present() -> None:
    base = extraction(
        refs=[ref("r1", "A"), ref("r2", "B"), ref("r3", "C"), ref("r4", "D")],
        desired_facts=[desired_before("r1", "r3"), desired_before("r2", "r3"), desired_before("r3", "r4")],
    )
    grouped = {
        "relation_groups": [
            {"kind": "precedence", "members": [{"source": "r1", "target": "r3"}, {"source": "r2", "target": "r3"}]},
            {"kind": "precedence", "members": [{"source": "r3", "target": "r4"}]},
        ]
    }

    run = run_phase1b_shadow("A and B before C, then C before D", model_runner=model_for(base, grouped))

    assert run.ok
    assert run.artifact["shadow_relation_grouper"]["validation"]["ok"] is False
    assert [error["kind"] for error in run.artifact["shadow_relation_grouper"]["validation"]["errors"]] == ["too_few_members"]
    assert [(member["source_ref"], member["target_ref"]) for member in run.artifact["after_goal_state"]["relation_facts"][0]["members"]] == [
        ("r1", "r3"),
        ("r2", "r3"),
    ]


def test_shadow_grouper_does_not_run_for_route_to_compatibility_path() -> None:
    base = extraction(refs=[ref("r1", "failure edge"), ref("r2", "Review")], desired_facts=[desired_route_to("r1", "r2")])
    systems: list[str] = []

    def model(system: str, _payload: dict):
        systems.append(system)
        if "discourse reference continuity" in system:
            return {"discourse_reference_continuity": {"status": "none", "ref_ids": [], "evidence": ""}}
        if "Relation Grouper" in system:
            raise AssertionError("route_to must remain on the compatibility path")
        return base

    run = run_phase1b_shadow("route failure edge to Review", model_runner=model)

    assert run.ok
    assert all("Relation Grouper" not in system for system in systems)
    assert run.artifact["after_goal_state"].get("relation_facts") in (None, [])


def test_shadow_grouper_correct_target_uses_prior_sources_and_preserves_relation_id() -> None:
    first = extraction(
        refs=[ref("r1", "A"), ref("r2", "B"), ref("r3", "C")],
        desired_facts=[desired_before("r1", "r3"), desired_before("r2", "r3")],
    )
    grouped_first = {"relation_groups": [{"kind": "precedence", "members": [{"source": "r1", "target": "r3"}, {"source": "r2", "target": "r3"}]}]}
    before = run_phase1b_shadow("A and B before C", model_runner=model_for(first, grouped_first)).artifact["after_goal_state"]
    second = extraction(
        refs=[ref("r4", "D")],
        desired_facts=[desired_before("r1", "r4"), desired_before("r2", "r4")],
        relation="CORRECT",
    )
    grouped_second = {"relation_groups": [{"kind": "precedence", "members": [{"source": "r1", "target": "r4"}, {"source": "r2", "target": "r4"}]}]}

    run = run_phase1b_shadow("No, A and B before D", current_goal_state=before, model_runner=model_for(second, grouped_second))

    assert run.ok
    assert run.artifact["shadow_relation_grouper"]["ok"] is True
    assert [row["ref_id"] for row in run.artifact["relation_grouper_input"]["active_prior_relation_refs"]] == ["r1", "r2", "r3"]
    assert run.artifact["semantic_delta"]["operations"][-1]["op"] == "CORRECT"
    assert run.artifact["semantic_delta"]["operations"][-1]["target_id"] == "rg1"
    assert run.artifact["after_goal_state"]["relation_facts"][0]["relation_id"] == "rg1"
    assert [(member["source_ref"], member["target_ref"]) for member in run.artifact["after_goal_state"]["relation_facts"][0]["members"]] == [
        ("r1", "r4"),
        ("r2", "r4"),
    ]


def test_shadow_grouper_correct_member_set_uses_current_source_and_prior_target() -> None:
    first = extraction(
        refs=[ref("r1", "A"), ref("r2", "B"), ref("r3", "C")],
        desired_facts=[desired_before("r1", "r3"), desired_before("r2", "r3")],
    )
    grouped_first = {"relation_groups": [{"kind": "precedence", "members": [{"source": "r1", "target": "r3"}, {"source": "r2", "target": "r3"}]}]}
    before = run_phase1b_shadow("A and B before C", model_runner=model_for(first, grouped_first)).artifact["after_goal_state"]
    second = extraction(
        refs=[ref("r4", "D")],
        desired_facts=[desired_before("r1", "r3"), desired_before("r4", "r3")],
        relation="CORRECT",
    )
    grouped_second = {"relation_groups": [{"kind": "precedence", "members": [{"source": "r1", "target": "r3"}, {"source": "r4", "target": "r3"}]}]}

    run = run_phase1b_shadow("No, A and D before C", current_goal_state=before, model_runner=model_for(second, grouped_second))

    assert run.ok
    assert run.artifact["semantic_delta"]["operations"][-1]["target_id"] == "rg1"
    assert [(member["source_ref"], member["target_ref"]) for member in run.artifact["after_goal_state"]["relation_facts"][0]["members"]] == [
        ("r1", "r3"),
        ("r4", "r3"),
    ]


def test_shadow_grouper_all_prior_ref_repeat_dedupes_without_duplicate() -> None:
    first = extraction(
        refs=[ref("r1", "A"), ref("r2", "B"), ref("r3", "C")],
        desired_facts=[desired_before("r1", "r3"), desired_before("r2", "r3")],
    )
    grouped_first = {"relation_groups": [{"kind": "precedence", "members": [{"source": "r1", "target": "r3"}, {"source": "r2", "target": "r3"}]}]}
    before = run_phase1b_shadow("A and B before C", model_runner=model_for(first, grouped_first)).artifact["after_goal_state"]
    second = extraction(
        refs=[],
        desired_facts=[desired_before("r1", "r3"), desired_before("r2", "r3")],
    )
    grouped_second = {"relation_groups": [{"kind": "precedence", "members": [{"source": "r1", "target": "r3"}, {"source": "r2", "target": "r3"}]}]}

    run = run_phase1b_shadow("Keep that grouped relation", current_goal_state=before, model_runner=model_for(second, grouped_second))

    assert run.ok
    assert [row["relation_id"] for row in run.artifact["after_goal_state"]["relation_facts"]] == ["rg1"]
    assert all(op["element_kind"] != "relation_fact" for op in run.artifact["semantic_delta"]["operations"])


def test_shadow_grouper_ambiguous_relation_correction_does_not_fuzzy_choose() -> None:
    before = empty_goal_state()
    before["references"] = [ref("r1", "A"), ref("r2", "B"), ref("r3", "C"), ref("r4", "D"), ref("r5", "E")]
    before["relation_facts"] = [
        relation_fact([("r1", "r3"), ("r2", "r3")], relation_id="rg1"),
        relation_fact([("r1", "r4"), ("r2", "r4")], relation_id="rg2"),
    ]
    second = extraction(
        refs=[ref("r5", "E")],
        desired_facts=[desired_before("r1", "r5"), desired_before("r2", "r5")],
        relation="CORRECT",
    )
    grouped_second = {"relation_groups": [{"kind": "precedence", "members": [{"source": "r1", "target": "r5"}, {"source": "r2", "target": "r5"}]}]}

    run = run_phase1b_shadow("No, A and B before E", current_goal_state=before, model_runner=model_for(second, grouped_second))

    assert not run.ok
    assert run.artifact["shadow_relation_grouper"]["ok"] is True
    assert run.artifact["delta_planner"]["errors"][0]["message"] == "CORRECT relation_fact target is ambiguous"


def test_legacy_shadow_relation_facts_do_not_enter_semantic_equivalence_as_grouped_truth() -> None:
    goal_state = empty_goal_state()
    goal_state["references"] = [ref("r1", "A"), ref("r2", "B")]
    goal_state["relation_facts"] = [relation_fact([("r1", "r2")], relation_id="rf:f1")]

    assert "relation_facts" not in semantic_equivalence_key(goal_state)
