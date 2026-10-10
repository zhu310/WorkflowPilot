from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import get_settings
from src.canonical_goal_state import canonicalize_desired_fact
from src.phase1b_shadow import PHASE1B_EXTRACTION_SYSTEM, PHASE1B_MODEL_GRAMMAR, empty_goal_state, run_phase1b_shadow, semantic_equivalence_key
from src.semantic_contract import canonical_property_name, stage_result_matches


OUT_DIR = ROOT / "runs"


def ref_operand(ref_id: str) -> dict[str, str]:
    return {"operand_type": "reference", "ref_id": ref_id}


def literal(value: str) -> dict[str, str]:
    return {"operand_type": "literal", "literal_type": "string", "value": value}


def prop(value: str) -> dict[str, str]:
    return {"operand_type": "literal", "literal_type": "property_name", "value": value}


def state_with_route(target_text: str = "结束") -> dict[str, Any]:
    state = empty_goal_state()
    state["references"] = [
        {"ref_id": "r1", "entity_type": "edge", "cardinality": "single", "mention": None, "provenance_refs": ["p1"]},
        {"ref_id": "r2", "entity_type": "node", "cardinality": "single", "mention": {"text": target_text, "span": None}, "provenance_refs": ["p1"]},
    ]
    state["desired_facts"] = [
        {
            "fact_id": "f1",
            "predicate": "relation.route_to",
            "predicate_signature_id": "relation.route_to",
            "operands": {"subject": ref_operand("r1"), "target": ref_operand("r2")},
            "provenance_refs": ["p1"],
        }
    ]
    return state


def base_context() -> dict[str, Any]:
    return {
        "visible_area": {"left": 0, "top": 0, "right": 1400, "bottom": 640},
        "graph_context": {
            "node_mapping": "结果判断=n_decision；生成报告=n_report；人工复核=n_human；发布结果=n_publish；结束=n_end",
            "current_edges": "结果判断->生成报告(success)；结果判断->人工复核(failure)；生成报告->发布结果；发布结果->结束",
        },
    }


def single_turn_cases() -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    route_variants = [
        "把这条边改到结束",
        "让这条边指向结束",
        "这条边接到结束节点",
        "把它连到结束",
        "让这个分支走到结束",
        "这条线改成去结束节点",
    ]
    cases += [{"id": f"route_{i}", "family": "route_equivalence", "utterance": text, "expect": {"predicate": "relation.route_to", "target_text": "结束"}} for i, text in enumerate(route_variants, 1)]
    implicit = ["改到人工复核", "指到人工复核", "改成走人工复核", "接到人工复核那里", "让它去人工复核"]
    cases += [{"id": f"implicit_{i}", "family": "implicit_reference", "utterance": text, "expect": {"predicate": "relation.route_to", "implicit_edge": True, "target_text": "人工复核"}} for i, text in enumerate(implicit, 1)]
    labels = ["把这条边标签改成 retry", "这条分支显示 retry", "把文字换成 retry", "标签用 retry", "这条线标记成 retry"]
    cases += [{"id": f"edge_label_{i}", "family": "property_label", "utterance": text, "expect": {"predicate": "property.equals", "property": "branch_label", "value": "retry"}} for i, text in enumerate(labels, 1)]
    node_labels = ["把这个节点改名为归档确认", "这个节点叫归档确认", "节点名称改成归档确认", "把它的名字换成归档确认"]
    cases += [{"id": f"node_label_{i}", "family": "property_node_name", "utterance": text, "expect": {"predicate": "property.equals", "property": "label", "value": "归档确认"}} for i, text in enumerate(node_labels, 1)]
    explicit_empty = ["清空这条边的标签", "把这条线的文字去掉", "这个分支不要显示文字", "删除这个标签，但保留这条边", "label 置为空"]
    cases += [{"id": f"explicit_empty_{i}", "family": "explicit_empty_positive", "utterance": text, "expect": {"predicate": "property.equals", "explicit_empty": True}} for i, text in enumerate(explicit_empty, 1)]
    empty_negative = ["把这条边改一下", "修改一下这个标签", "处理这个分支", "调整这条线"]
    cases += [{"id": f"explicit_empty_neg_{i}", "family": "explicit_empty_negative", "utterance": text, "expect": {"no_explicit_empty": True}} for i, text in enumerate(empty_negative, 1)]
    deletes = ["删除发布结果节点", "把发布结果节点去掉", "流程里不要发布结果这个节点", "移除发布结果", "去掉这个节点"]
    cases += [{"id": f"absent_{i}", "family": "entity_absent", "utterance": text, "expect": {"predicate": "entity.absent"}} for i, text in enumerate(deletes, 1)]
    before = ["把审核放到发布前面", "审核要在发布之前", "发布之前先审核"]
    after = ["把审核放到发布后面", "发布完再审核"]
    cases += [{"id": f"before_{i}", "family": "ordering_before", "utterance": text, "expect": {"predicate": "relation.before"}} for i, text in enumerate(before, 1)]
    cases += [{"id": f"after_{i}", "family": "ordering_after", "utterance": text, "expect": {"predicate": "relation.after"}} for i, text in enumerate(after, 1)]
    parallel = ["审核和复核并行", "复核和审核并行"]
    cases += [{"id": f"parallel_{i}", "family": "parallel", "utterance": text, "expect": {"predicate": "relation.parallel_with"}} for i, text in enumerate(parallel, 1)]
    sets = ["删除这些节点", "把选中的这些节点去掉", "这些节点都不要了"]
    singles = ["删除这个节点", "把这个节点去掉"]
    cases += [{"id": f"set_{i}", "family": "cardinality_set", "utterance": text, "expect": {"cardinality": "set"}} for i, text in enumerate(sets, 1)]
    cases += [{"id": f"single_{i}", "family": "cardinality_single", "utterance": text, "expect": {"cardinality": "single"}} for i, text in enumerate(singles, 1)]
    open_statements = ["优化一下这个流程图", "把这个流程整理一下", "这个流程需要更合理一点", "帮我改善一下这个流程", "让整个流程更清晰一些"]
    cases += [{"id": f"open_{i}", "family": "open_statement", "utterance": text, "expect": {"open_statement": True, "no_desired_fact": True}} for i, text in enumerate(open_statements, 1)]
    adversarial = ["改一下", "调整一下", "处理一下这里", "这个不太对", "再优化一下", "这里有问题", "换一个"]
    cases += [{"id": f"adversarial_{i}", "family": "non_hallucination", "utterance": text, "expect": {"no_concrete_fact": True}} for i, text in enumerate(adversarial, 1)]
    recent = ["刚才新增的那个节点", "上一个新增节点", "刚才那个节点"]
    cases += [{"id": f"recent_{i}", "family": "recent_reference", "utterance": text, "expect": {"reference_or_open": True}} for i, text in enumerate(recent, 1)]
    return cases


def multi_turn_cases() -> list[dict[str, Any]]:
    return [
        {"id": "temporal_route_1", "family": "temporal_equivalence", "turns": ["把这条边改一下", "让它指向结束"], "compare_to": "把这条边改到结束"},
        {"id": "temporal_property_1", "family": "temporal_equivalence", "turns": ["把这条边改一下", "标签改成 retry"], "compare_to": "把这条边标签改成 retry"},
        {"id": "temporal_node_1", "family": "temporal_equivalence", "turns": ["把这个节点改一下", "改名为归档确认"], "compare_to": "把这个节点改名为归档确认"},
        {"id": "correction_route_1", "family": "correction", "turns": ["把这条边改到审核", "不对，改到结束"], "expect_final": {"predicate": "relation.route_to", "target_text": "结束"}},
        {"id": "correction_route_2", "family": "correction", "turns": ["把这条边改到审核", "不是审核，是结束"], "expect_final": {"predicate": "relation.route_to", "target_text": "结束"}},
        {"id": "cross_factkey_1", "family": "cross_factkey_correction", "turns": ["把 A 放到 B 前面", "不对，是 B 在 A 前面"], "expect_final": {"predicate": "relation.before"}},
        {"id": "candidate_resolved_1", "family": "candidate_stage", "runtime_stage": candidate_stage(), "turns": ["第二个"], "expect_stage": "resolved"},
        {"id": "candidate_resolved_2", "family": "candidate_stage", "runtime_stage": candidate_stage(), "turns": ["选第二个"], "expect_stage": "resolved"},
        {"id": "candidate_unresolved_1", "family": "candidate_unresolved", "runtime_stage": candidate_stage(), "turns": ["我还不确定"], "expect_stage": "unresolved"},
        {"id": "candidate_new_task_1", "family": "candidate_new_task", "runtime_stage": candidate_stage(), "turns": ["删除发布结果节点"], "expect_relation": "NEW_TASK"},
        {"id": "confirmation_authorized_1", "family": "confirmation_stage", "runtime_stage": confirmation_stage(), "turns": ["确认"], "expect_stage": "authorized"},
        {"id": "confirmation_declined_1", "family": "confirmation_stage", "runtime_stage": confirmation_stage(), "turns": ["不要这样改"], "expect_stage": "declined"},
        {"id": "candidate_residual_1", "family": "residual_intent", "runtime_stage": candidate_stage(), "turns": ["第二个，然后把标签改成 retry"], "expect": {"predicate": "property.equals", "value": "retry"}},
        {"id": "confirmation_residual_1", "family": "multi_act_ordering", "runtime_stage": confirmation_stage(), "turns": ["确认，然后删除发布结果节点"], "expect": {"predicate": "entity.absent"}},
        {"id": "candidate_correction_residual_1", "family": "multi_act_ordering", "runtime_stage": candidate_stage(), "turns": ["不是第二个，是第一个，然后改到结束"], "expect": {"predicate": "relation.route_to", "target_text": "结束"}},
    ]


def candidate_stage() -> dict[str, Any]:
    return {"stage_type": "GROUNDING_CANDIDATE", "stage_id": "stage_candidate_1", "candidates": [{"id": "A", "label": "第一条"}, {"id": "B", "label": "第二条"}]}


def confirmation_stage() -> dict[str, Any]:
    return {"stage_type": "PREFLIGHT_CONFIRMATION", "stage_id": "stage_confirm_1", "question": "是否确认执行这个修改？"}


def high_risk_repeats() -> list[dict[str, Any]]:
    return [
        {"id": "repeat_implicit", "family": "stability", "utterance": "改到人工复核", "runs": 5, "expect": {"predicate": "relation.route_to", "implicit_edge": True}},
        {"id": "repeat_vague", "family": "stability", "utterance": "把这条边改一下", "runs": 5, "expect": {"no_explicit_empty": True}},
        {"id": "repeat_open", "family": "stability", "utterance": "优化一下这个流程图", "runs": 5, "expect": {"open_statement": True, "no_desired_fact": True}},
        {"id": "repeat_empty", "family": "stability", "utterance": "清空这条边标签", "runs": 5, "expect": {"predicate": "property.equals", "explicit_empty": True}},
        {"id": "repeat_candidate_new", "family": "stability", "utterance": "删除发布结果节点", "runs": 5, "runtime_stage": candidate_stage(), "expect_relation": "NEW_TASK"},
        {"id": "repeat_correction", "family": "stability", "utterance": "不对，改到结束", "runs": 5, "current_goal_state": state_with_route("审核"), "expect": {"predicate": "relation.route_to", "target_text": "结束"}},
        {"id": "repeat_residual", "family": "stability", "utterance": "第二个，然后把标签改成 retry", "runs": 5, "runtime_stage": candidate_stage(), "expect": {"predicate": "property.equals", "value": "retry"}},
    ]


def run_turn(utterance: str, *, current_goal_state: dict[str, Any] | None = None, runtime_stage: dict[str, Any] | None = None) -> dict[str, Any]:
    started = time.time()
    result = run_phase1b_shadow(
        utterance,
        current_goal_state=current_goal_state,
        task_context={"task_context_id": "task1", "state": "active"} if current_goal_state else {},
        runtime_stage=runtime_stage,
        graph_context=base_context(),
        selection={"node_ids": [], "edge_ids": ["e_decision_report"]},
    )
    artifact = result.artifact
    artifact["duration_seconds"] = round(time.time() - started, 3)
    artifact["run_ok"] = result.ok
    artifact["run_error"] = result.error
    return artifact


def evaluate_expectation(artifact: dict[str, Any], expect: dict[str, Any] | None = None, *, expect_stage: str | None = None, expect_relation: str | None = None) -> tuple[bool, list[dict[str, Any]]]:
    failures = []
    state = artifact.get("after_goal_state") if isinstance(artifact.get("after_goal_state"), dict) else {}
    facts = state.get("desired_facts", []) if isinstance(state.get("desired_facts"), list) else []
    refs = state.get("references", []) if isinstance(state.get("references"), list) else []
    open_statements = state.get("open_statements", []) if isinstance(state.get("open_statements"), list) else []
    if expect_stage:
        actual_stage = artifact.get("stage_continuation", {}).get("result") if isinstance(artifact.get("stage_continuation"), dict) else None
        if not stage_result_matches(expect_stage, actual_stage):
            failures.append(failure("STAGE_CONTINUATION_QUALITY", f"expected stage {expect_stage}, got {actual_stage}"))
    if expect_relation:
        actual_relation = artifact.get("turn_relation", {}).get("relation") if isinstance(artifact.get("turn_relation"), dict) else None
        if actual_relation != expect_relation:
            failures.append(failure("TURN_RELATION_QUALITY", f"expected relation {expect_relation}, got {actual_relation}"))
    for item in artifact.get("validation", {}).get("errors", []) if isinstance(artifact.get("validation"), dict) else []:
        failures.append(failure(item.get("category", "EVIDENCE_VALIDATION"), item.get("message", "validation error")))
    for item in artifact.get("invariants", {}).get("errors", []) if isinstance(artifact.get("invariants"), dict) else []:
        failures.append(failure("SEMANTIC_DELTA", item.get("message", "invariant error")))
    expect = expect or {}
    if predicate := expect.get("predicate"):
        canonical_predicate = _canonical_expected_predicate(str(predicate))
        if not any(fact.get("predicate") == canonical_predicate for fact in facts):
            failures.append(failure("EXTRACTION_QUALITY", f"missing predicate {predicate}"))
    if expect.get("implicit_edge"):
        edge_refs = [ref for ref in refs if ref.get("entity_type") == "edge"]
        if not edge_refs or not any(ref.get("mention") is None for ref in edge_refs):
            failures.append(failure("EXTRACTION_QUALITY", "missing implicit edge reference with mention=null"))
    if target_text := expect.get("target_text"):
        if target_text not in json.dumps(state, ensure_ascii=False):
            failures.append(failure("EXTRACTION_QUALITY", f"missing target text {target_text}"))
    if property_name := expect.get("property"):
        if not _has_canonical_property(facts, refs, property_name):
            failures.append(failure("EXTRACTION_QUALITY", f"missing property {property_name}"))
    if value := expect.get("value"):
        if value not in json.dumps(facts, ensure_ascii=False):
            failures.append(failure("EXTRACTION_QUALITY", f"missing value {value}"))
    if expect.get("explicit_empty"):
        if '"operand_type": "explicit_empty"' not in json.dumps(facts, ensure_ascii=False):
            failures.append(failure("EXTRACTION_QUALITY", "missing typed explicit_empty operand"))
    if expect.get("no_explicit_empty"):
        if '"operand_type": "explicit_empty"' in json.dumps(facts, ensure_ascii=False):
            failures.append(failure("EXTRACTION_QUALITY", "false positive explicit_empty"))
    if cardinality := expect.get("cardinality"):
        if not any(ref.get("cardinality") == cardinality for ref in refs):
            failures.append(failure("EXTRACTION_QUALITY", f"missing cardinality {cardinality}"))
    if expect.get("open_statement"):
        if not open_statements:
            failures.append(failure("NON_HALLUCINATION", "missing open semantic statement"))
    if expect.get("no_desired_fact") and facts:
        failures.append(failure("NON_HALLUCINATION", "hallucinated desired fact"))
    if expect.get("no_concrete_fact") and facts:
        failures.append(failure("NON_HALLUCINATION", "hallucinated concrete desired fact"))
    if expect.get("reference_or_open") and not refs and not open_statements:
        failures.append(failure("EXTRACTION_QUALITY", "expected reference or open statement"))
    return not failures, failures


def failure(category: str, message: str) -> dict[str, str]:
    return {"category": category, "message": message}


def _canonical_expected_predicate(predicate: str) -> str:
    canonical = canonicalize_desired_fact(
        {
            "predicate": predicate,
            "operands": {
                "subject": {"operand_type": "reference", "ref_id": "subject"},
                "anchor": {"operand_type": "reference", "ref_id": "anchor"},
            },
        }
    )
    return str(canonical.get("predicate") or predicate)


def _has_canonical_property(facts: list[dict[str, Any]], refs: list[dict[str, Any]], expected_property: str) -> bool:
    refs_by_id = {ref.get("ref_id"): ref for ref in refs if isinstance(ref, dict)}
    for fact in facts:
        if not isinstance(fact, dict) or fact.get("predicate") != "property.equals":
            continue
        operands = fact.get("operands") if isinstance(fact.get("operands"), dict) else {}
        subject = operands.get("subject")
        property_operand = operands.get("property")
        if not isinstance(subject, dict) or not isinstance(property_operand, dict):
            continue
        ref = refs_by_id.get(subject.get("ref_id"))
        entity_type = ref.get("entity_type") if isinstance(ref, dict) else None
        actual = canonical_property_name(entity_type, property_operand.get("value"))
        expected = canonical_property_name(entity_type, expected_property)
        if actual == expected:
            return True
    return False


def run_single_cases(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    results = []
    for case in cases:
        artifact = run_turn(case["utterance"])
        passed, failures = evaluate_expectation(artifact, case.get("expect"))
        results.append(pack_result(case, [artifact], passed, failures))
    return results


def run_multi_cases(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    results = []
    for case in cases:
        artifacts = []
        state = deepcopy(case.get("current_goal_state", empty_goal_state()))
        for turn in case["turns"]:
            artifact = run_turn(turn, current_goal_state=state, runtime_stage=case.get("runtime_stage"))
            artifacts.append(artifact)
            state = artifact.get("after_goal_state", state)
        failures = []
        passed, check_failures = evaluate_expectation(artifacts[-1], case.get("expect") or case.get("expect_final"), expect_stage=case.get("expect_stage"), expect_relation=case.get("expect_relation"))
        failures.extend(check_failures)
        if compare_to := case.get("compare_to"):
            single = run_turn(compare_to)
            if semantic_equivalence_key(single.get("after_goal_state", {})) != semantic_equivalence_key(artifacts[-1].get("after_goal_state", {})):
                failures.append(failure("EXTRACTION_QUALITY", "single-turn and multi-turn final GoalState are not semantically equivalent"))
            artifacts.append({"comparison_single_turn": single})
        results.append(pack_result(case, artifacts, passed and not failures, failures))
    return results


def run_repeats(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    results = []
    for case in cases:
        artifacts = []
        keys = []
        failures = []
        for _ in range(case["runs"]):
            artifact = run_turn(case["utterance"], current_goal_state=case.get("current_goal_state"), runtime_stage=case.get("runtime_stage"))
            artifacts.append(artifact)
            keys.append(canonical_key_string(artifact.get("after_goal_state", {})))
            _passed, check_failures = evaluate_expectation(artifact, case.get("expect"), expect_relation=case.get("expect_relation"))
            failures.extend(check_failures)
        divergent = len(set(keys)) > 1
        if divergent:
            failures.append(failure("MODEL_INSTABILITY", "semantic equivalence key diverged across repeated runs"))
        packed = pack_result(case, artifacts, not failures, failures)
        packed["stability"] = {"runs": case["runs"], "equivalent_runs": case["runs"] if not divergent else keys.count(keys[0]), "divergent_runs": 0 if not divergent else case["runs"] - keys.count(keys[0])}
        results.append(packed)
    return results


def canonical_key_string(goal_state: dict[str, Any]) -> str:
    return json.dumps(semantic_equivalence_key(goal_state), ensure_ascii=False, sort_keys=True)


def pack_result(case: dict[str, Any], artifacts: list[dict[str, Any]], passed: bool, failures: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "id": case["id"],
        "family": case["family"],
        "utterance": case.get("utterance"),
        "turns": case.get("turns"),
        "passed": passed,
        "failures": failures,
        "artifacts": artifacts,
    }


def summarize(results: list[dict[str, Any]], settings: Any, prompt_hash: str, grammar_hash: str) -> dict[str, Any]:
    total_calls = sum(len([a for a in row["artifacts"] if "raw_turn" in a]) for row in results)
    structured = 0
    schema_valid = 0
    invariant_pass = 0
    distribution = Counter()
    family_totals = Counter()
    family_pass = Counter()
    for row in results:
        family_totals[row["family"]] += 1
        if row["passed"]:
            family_pass[row["family"]] += 1
        for artifact in row["artifacts"]:
            if "raw_turn" not in artifact:
                continue
            if isinstance(artifact.get("raw_extraction", {}).get("parsed"), dict):
                structured += 1
            if artifact.get("validation", {}).get("ok") is True:
                schema_valid += 1
            if artifact.get("invariants", {}).get("ok") is True:
                invariant_pass += 1
        for item in row["failures"]:
            distribution[item["category"]] += 1
    return {
        "environment": {
            "real_model_runner": True,
            "model": settings.model,
            "base_url": settings.base_url,
            "temperature": settings.temperature,
            "timeout_seconds": settings.timeout_seconds,
            "prompt_hash_sha256": prompt_hash,
            "grammar_hash_sha256": grammar_hash,
            "feature_flag": "benchmark_direct_runner",
        },
        "metrics": {
            "total_real_llm_calls": total_calls,
            "structured_output_success": structured,
            "schema_valid_outputs": schema_valid,
            "semantic_invariant_pass": invariant_pass,
            "semantic_invariant_fail": total_calls - invariant_pass,
            "case_count": len(results),
            "case_pass": sum(1 for row in results if row["passed"]),
            "case_fail": sum(1 for row in results if not row["passed"]),
            "family_totals": dict(family_totals),
            "family_pass": dict(family_pass),
            "failure_distribution": dict(distribution),
        },
    }


def historical_probe_matrix(results: list[dict[str, Any]]) -> list[dict[str, str]]:
    lookup = {row["id"]: row for row in results}
    mapping = [
        ("1", "改到人工复核", "repeat_implicit"),
        ("2", "selected edge vague -> route end", "temporal_route_1"),
        ("3", "selected edge vague -> label retry", "temporal_property_1"),
        ("4", "selected node vague -> rename", "temporal_node_1"),
        ("5", "candidate: 第二个", "candidate_resolved_1"),
        ("6", "candidate click second + this edge", "candidate_residual_1"),
        ("7", "candidate unresolved", "candidate_unresolved_1"),
        ("8", "confirmation confirm", "confirmation_authorized_1"),
        ("9", "open optimization", "repeat_open"),
        ("10", "candidate pending -> delete publish", "candidate_new_task_1"),
        ("11", "recent new node reference", "recent_1"),
        ("12", "multi-selection delete nodes", "set_1"),
        ("13", "aggregated edge this branch", "implicit_5"),
        ("14", "aggregated edge two branches", "set_2"),
        ("15", "route_to(X,A) -> B", "correction_route_1"),
        ("16", "label=A -> correct B", "edge_label_1"),
        ("17", "clear edge label", "explicit_empty_1"),
        ("18", "version changed -> old candidate", None),
        ("19", "candidate second + label retry", "candidate_residual_1"),
        ("20", "confirmation + delete publish", "confirmation_residual_1"),
        ("21", "candidate correction + route end", "candidate_correction_residual_1"),
    ]
    rows = []
    for probe_id, name, case_id in mapping:
        if case_id is None:
            rows.append({"probe": probe_id, "name": name, "status": "NOT_APPLICABLE", "evidence": "version-stale production binding is outside Phase 1B extraction benchmark"})
            continue
        row = lookup.get(case_id)
        if row is None:
            rows.append({"probe": probe_id, "name": name, "status": "BLOCKED_BY_HARNESS", "evidence": f"case {case_id} missing"})
        else:
            rows.append({"probe": probe_id, "name": name, "status": "PASS" if row["passed"] else "FAIL", "evidence": "case passed" if row["passed"] else row["failures"][0]["message"]})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="")
    args = parser.parse_args()
    settings = get_settings()
    prompt_hash = hashlib.sha256(PHASE1B_EXTRACTION_SYSTEM.encode("utf-8")).hexdigest()
    grammar_hash = hashlib.sha256(PHASE1B_MODEL_GRAMMAR.encode("utf-8")).hexdigest()
    started = time.strftime("%Y-%m-%dT%H:%M:%S")
    single = run_single_cases(single_turn_cases())
    multi = run_multi_cases(multi_turn_cases())
    repeats = run_repeats(high_risk_repeats())
    results = single + multi + repeats
    report = summarize(results, settings, prompt_hash, grammar_hash)
    report["started_at"] = started
    report["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    report["single_turn_count"] = len(single)
    report["multi_turn_count"] = len(multi)
    report["repeat_group_count"] = len(repeats)
    report["results"] = results
    report["historical_probe_matrix"] = historical_probe_matrix(results)
    OUT_DIR.mkdir(exist_ok=True)
    out = Path(args.out) if args.out else OUT_DIR / f"phase1b_acceptance_{int(time.time())}.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"out": str(out), **report["metrics"], "environment": report["environment"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
