import os

import pytest

from src.semantic_goal import validate_semantic_goal_spec
from src.workflow_ir_runtime import SEMANTIC_GOAL_UNDERSTANDING_SYSTEM, _complete_json
from semantic_goal_contract_helpers import semantic_goal_contract_violations


NATURAL_LANGUAGE_CASES = [
    "增加人工审核节点",
    "删除库存检查",
    "把审批节点名称改成人工审核",
    "把审核移动到风控后面",
    "失败流程转人工审核",
    "异常流程接人工处理",
    "增加人工审核节点，并让失败分支走人工审核",
    "让风险检查和审计并行执行",
    "把这个节点移动到右边",
    "优化一下这个流程",
    "给我几个优化方案",
    "不改变已有审批节点，只把失败分支改成人工审核",
]


def test_semantic_goal_prompt_contract_static() -> None:
    prompt = SEMANTIC_GOAL_UNDERSTANDING_SYSTEM

    assert "Semantic Goal v2 primitives" in prompt
    assert "object_desired_states" in prompt
    assert "relation_intents" in prompt
    assert "semantic_constraints" in prompt
    assert "unresolved_reference" in prompt
    assert "Clarification is not the same as no semantic goal" in prompt
    assert "Omission of an unresolved role is not omission of the entire goal" in prompt
    assert "Use only exact object phrases" in prompt
    assert "Graph labels are context, not mention text" in prompt
    assert "Reference.text must come from the user utterance" in prompt
    assert "existing\" means the user is referring to an entity that already belongs to the current workflow" in prompt
    assert "it does not mean text should be a graph label" in prompt
    assert "先核价再收款" in prompt
    assert "价格校验" in prompt
    assert "支付确认" in prompt
    assert "text=\"核价\"" in prompt
    assert "text=\"收款\"" in prompt
    assert "Apply the exact-phrase rule to route targets too" in prompt
    assert "similar existing graph label" in prompt
    assert "edge_reference.text must be an edge-local evidence phrase that actually appears in the current utterance" in prompt
    assert "Do not concatenate a source node phrase with an outcome phrase" in prompt
    assert "Semantic Grounding combines route relation evidence, source-node hints, and graph facts" in prompt
    assert "recent transaction" in prompt
    assert "keep that referent phrase unchanged as an existing_reference object" in prompt
    assert "Do not resolve ambiguity" in prompt
    assert "Do not output operation kinds" in prompt
    assert "Chained ordered relations" in prompt
    assert "resolve the omitted participant from discourse structure" in prompt
    assert "A before B and B before C" in prompt
    assert "A before B and A before C" in prompt
    assert "unresolved_reference for the missing role" in prompt
    for forbidden in [
        "node_id",
        "edge_id",
        "workflow_ir",
        "control_flow_regions",
        "mutation steps",
        "planner steps",
    ]:
        assert forbidden in prompt
    for operation_kind in ["layout", "ordering", "route", "branch_property", "delete", "parallel", "rename", "insert"]:
        assert operation_kind in prompt


def test_semantic_goal_prompt_does_not_reintroduce_route_phrase_synthesis() -> None:
    prompt = SEMANTIC_GOAL_UNDERSTANDING_SYSTEM

    assert "source node wording plus outcome/condition wording" not in prompt
    assert "Risk Check fails" not in prompt


def test_reference_phrase_contract_rejects_graph_label_substitution_for_colloquial_phrase() -> None:
    graph_label_substitution = {
        "status": "resolved",
        "semantic_goals": [
            {
                "goal_category": "relation_goal",
                "text": "核价 before 收款",
                "objects": [
                    {"id": "o1", "type": "existing_reference", "text": "价格校验", "role": "object"},
                    {"id": "o2", "type": "existing_reference", "text": "支付确认", "role": "object"},
                ],
                "relation_intents": [{"source": "o1", "relation": "before", "target": "o2"}],
            }
        ],
        "clarification": {"needed": False, "question": "", "reason": ""},
    }
    source_phrase_preserved = {
        **graph_label_substitution,
        "semantic_goals": [
            {
                **graph_label_substitution["semantic_goals"][0],
                "objects": [
                    {"id": "o1", "type": "existing_reference", "text": "核价", "role": "object"},
                    {"id": "o2", "type": "existing_reference", "text": "收款", "role": "object"},
                ],
            }
        ],
    }

    invalid = validate_semantic_goal_spec(graph_label_substitution, source_instruction="先核价再收款")
    valid = validate_semantic_goal_spec(source_phrase_preserved, source_instruction="先核价再收款")

    assert "semantic_goal_contract_violation" in {issue["kind"] for issue in invalid["issues"]}
    assert invalid["ready_for_grounding"] is False
    assert valid["ready_for_grounding"] is True


def test_route_edge_reference_contract_rejects_synthesized_source_outcome_phrase() -> None:
    synthesized = {
        "status": "resolved",
        "semantic_goals": [
            {
                "goal_category": "relation_goal",
                "text": "route failure",
                "objects": [
                    {"id": "node", "type": "existing_reference", "text": "风控检查", "role": "object"},
                    {"id": "route", "type": "edge_reference", "text": "风控检查失败", "role": "source"},
                    {"id": "target", "type": "existing_reference", "text": "拒绝", "role": "target"},
                ],
                "relation_intents": [{"source": "route", "relation": "route_to", "target": "target"}],
            }
        ],
        "clarification": {"needed": False, "question": "", "reason": ""},
    }
    edge_local_phrase = {
        **synthesized,
        "semantic_goals": [
            {
                **synthesized["semantic_goals"][0],
                "objects": [
                    {"id": "node", "type": "existing_reference", "text": "风控检查", "role": "object"},
                    {"id": "route", "type": "edge_reference", "text": "失败时", "role": "source"},
                    {"id": "target", "type": "existing_reference", "text": "拒绝", "role": "target"},
                ],
            }
        ],
    }

    instruction = "风控检查改名为风险复核，失败时走拒绝"
    invalid = validate_semantic_goal_spec(synthesized, source_instruction=instruction)
    valid = validate_semantic_goal_spec(edge_local_phrase, source_instruction=instruction)

    assert "semantic_goal_contract_violation" in {issue["kind"] for issue in invalid["issues"]}
    assert invalid["ready_for_grounding"] is False
    assert valid["ready_for_grounding"] is True


def test_semantic_goal_contract_rejects_workflow_ir_and_ids() -> None:
    invalid = {
        "status": "resolved",
        "workflow_ir": {"properties": [{"node_id": "n1", "changes": {"x": 100}}]},
        "semantic_goals_v2": [
            {
                "id": "g1",
                "goal_category": "relation_goal",
                "text": "move",
                "objects": [{"id": "o1", "type": "existing_reference", "text": "审核"}],
                "relation_intents": [{"source": "o1", "relation": "right_of", "target": "o2"}],
            }
        ],
    }

    result = validate_semantic_goal_spec(invalid, source_instruction="把审核移动到右边")

    kinds = {issue["kind"] for issue in result["issues"]}
    assert "semantic_goal_contract_violation" in kinds
    assert not result["ready_for_grounding"]
    helper_violations = semantic_goal_contract_violations(invalid)
    assert {row["kind"] for row in helper_violations} >= {"forbidden_id_key", "forbidden_workflow_key"}


def test_contract_helper_does_not_false_positive_short_keys_inside_text_values() -> None:
    valid = {
        "status": "resolved",
        "semantic_goals": [
            {
                "goal_category": "object_state_goal",
                "text": "Rename text exists property_equals safely.",
                "objects": [{"id": "o1", "type": "existing_reference", "text": "审批节点", "role": "object"}],
                "object_desired_states": [
                    {"object": "o1", "state": "property_equals", "property": "display_label", "value": "人工审核"}
                ],
                "relation_intents": [],
                "semantic_constraints": [],
            }
        ],
        "clarification": {"needed": False, "question": "", "reason": ""},
    }

    assert semantic_goal_contract_violations(valid) == []


@pytest.mark.skipif(os.getenv("RUN_REAL_LLM_TESTS") != "1", reason="set RUN_REAL_LLM_TESTS=1 to run real LLM prompt regression")
@pytest.mark.parametrize("instruction", NATURAL_LANGUAGE_CASES)
def test_optional_real_llm_semantic_goal_prompt_regression(instruction: str) -> None:
    graph = {
        "nodes": [
            {"id": "approval", "type": "task", "label": "审批节点"},
            {"id": "risk", "type": "task", "label": "风控"},
            {"id": "manual", "type": "task", "label": "人工审核"},
        ],
        "edges": [
            {"id": "edge_failure", "source": "approval", "target": "manual", "label": "failure", "branch_label": "失败"}
        ],
    }
    result = _complete_json(
        SEMANTIC_GOAL_UNDERSTANDING_SYSTEM,
        {
            "instruction": instruction,
            "graph": graph,
            "selection": {"node_ids": [], "edge_ids": []},
            "request_context": {"conversation_history": []},
        },
    )

    assert result.parse_error is None
    assert isinstance(result.parsed_json, dict)
    validation = validate_semantic_goal_spec(result.parsed_json, source_instruction=instruction)
    assert validation["valid"] is True
    assert semantic_goal_contract_violations(result.parsed_json) == []
