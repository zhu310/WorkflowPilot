# WorkflowPilot / AI-Flow-Canvas

WorkflowPilot is a natural-language workflow graph editing system. It does not let the LLM rewrite the graph or emit graph mutations directly. The current production path is:

```text
User Instruction
-> Semantic Goal Understanding
-> Semantic Goal Validation
-> Semantic Grounding
-> Goal Lowering
-> Workflow IR
-> Compiler
-> Planner
-> Executor
-> Versioned Graph
```

The main architecture upgrade in this repository is the move from `LLM -> Workflow IR` to `Semantic Goal + Grounding + deterministic Lowering`. The LLM expresses user goals with Semantic Goal v2 primitives; the program binds references to graph objects, lowers executable goals to the existing Workflow IR, plans deterministic edits, and records versioned transactions.

## Current Verified Backend Capabilities

The mocked Semantic Goal and backend regression suite currently covers:

- Semantic Goal v2 primitives: object references, object desired states, relation intents, semantic constraints.
- Context-aware grounding: selection, recent transaction diff, conversation-history fallback, and node/edge semantic reference candidate generation.
- Core editing closure for exists, node property changes, absent/delete, route-to existing or newly-created targets, simple safe `parallel_with`, and partial execution.
- Partial Execution Contract: supported subgoals can commit as `partial_success`; unsupported subgoals remain visible in trace/specification instead of becoming false success.
- User-facing feedback and session-local clarification continuation for ambiguous candidates.

Current local regression baseline: `222+` pytest cases for mocked Semantic Goal and deterministic backend behavior. Real LLM prompt stability is covered by static prompt contract tests by default and optional real-LLM regression when `RUN_REAL_LLM_TESTS=1` is set.

## Important Boundaries

Implemented, but bounded:

- Demo selection is sent to the backend and can ground phrases such as “这个节点” or “这个分支”.
- Recent successful transaction diffs can ground “刚才那个节点”.
- Clarification continuation is session-local and candidate-based; it is not long-term memory or a general Agent loop.
- Simple parallel lowering is supported only when graph facts make the boundary deterministic.

Not fully implemented:

- Embedding/vector retrieval.
- Industry ontology or long-term alias database.
- Long-term conversation memory.
- Real LogicFlow viewport-based `visible_area` grounding.
- Complex parallel boundary inference.
- Automatic proposal -> confirmation -> execution for abstract optimization requests.
- Guaranteed real LLM stability without running optional real-LLM regression.

## Repository Layout

```text
src/                       core implementation
tests/                     unit, contract, and deterministic regression tests
demo/                      LogicFlow demo frontend and API server
experiments/               optional acceptance and regression runners
docs/                      architecture notes, evidence, and historical analysis
```

## Setup

1. Install Python 3.12+.
2. Install Python dependencies:

```powershell
pip install -r requirements.txt
```

3. Copy `.env.example` to `.env` and set `DEEPSEEK_API_KEY` locally for real LLM demo usage. Do not commit `.env`.

## Verification

Default deterministic regression:

```powershell
python -m pytest -q
python -m compileall src tests
```

Optional real LLM prompt regression:

```powershell
$env:RUN_REAL_LLM_TESTS="1"
python -m pytest tests/test_llm_semantic_goal_prompt_regression.py -q
```

Optional historical acceptance runners remain under `experiments/`. Treat their results as evidence for the specific runner conditions, not as a claim that arbitrary workflow edits are supported.
