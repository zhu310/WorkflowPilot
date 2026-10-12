# WorkflowPilot / AI-Flow-Canvas

WorkflowPilot is a natural-language workflow canvas editing prototype. It targets complex workflow diagrams used in approval, data-processing, and operations scenarios, where manual node/edge editing is inefficient and direct LLM-generated graph mutations can easily create wrong edges, missing links, or unintended changes.

The key design choice is that the LLM does not rewrite the graph directly. It only helps express the user's editing intent as semantic goals. The program then grounds those goals to concrete graph objects, lowers them into Workflow IR, plans constrained graph changes, and executes the final mutations deterministically.

## Architecture

```text
User instruction
-> Semantic Goal understanding
-> Grounding to graph objects
-> Lowering
-> Workflow IR
-> Compiler
-> Constraint Planner
-> Deterministic Executor
-> Transaction / Versioned graph state
```

This separation keeps the LLM away from direct business-state mutation. The backend owns object binding, constraint planning, mutation authorization, execution, and graph validation.

## Implemented Capabilities

- Natural-language edits for workflow nodes, edges, routing targets, branch labels, and branch conditions.
- Semantic Goal primitives for object references, object desired states, relation intents, and semantic constraints.
- Grounding based on the current graph, frontend selection, recent successful transaction diffs, and candidate generation for ambiguous node/edge references.
- Clarification flow for ambiguous or missing references, with session-local pending clarification state.
- Workflow IR lowering and compilation into required relations, forbidden relations, property changes, edge property changes, and preservation-related constraints.
- Constraint planning with per-step mutation authorization before deterministic execution.
- Session-local transaction and version state, including graph snapshots, undo, redo, rollback, and recent-diff-based context reference such as "the node just added".
- LogicFlow demo frontend connected to the semantic editing backend.
- Pytest coverage for grounding, lowering, planning, execution, user feedback, versioning, and deterministic regression cases.

## Validation Boundary

The system performs machine-checkable validation, including:

- invalid or ambiguous object references;
- invalid Workflow IR structure;
- non-deterministic structural boundaries;
- conflicting required/forbidden relation constraints;
- unauthorized mutation steps;
- invalid final graph structure, such as missing node endpoints or duplicate identifiers.

It does not claim general natural-language semantic equivalence checking. In other words, the system can verify whether the generated IR, constraints, plan, and final graph satisfy supported machine-checkable conditions, but it does not prove that an arbitrary natural-language request has been fully and semantically completed.

## Supported And Bounded Scenarios

Supported examples include:

- rename or update a selected node;
- insert a new node before or after an existing node;
- route a selected or referenced branch to another node;
- update branch labels and branch conditions;
- transform a simple deterministic sequence into a parallel structure;
- continue editing with references such as "this node" or "the node just added";
- ask for clarification when a reference has multiple candidates.

Known boundaries:

- no long-term memory or persistent alias database;
- no embedding or vector retrieval layer;
- no full industry ontology;
- no guaranteed real-LLM stability unless optional real-LLM regression is run;
- no general-purpose agent loop for abstract optimization requests;
- complex parallel boundary inference is only supported when graph facts make the boundary deterministic;
- real LogicFlow viewport-based `visible_area` grounding is not fully implemented.

## Repository Layout

```text
src/                       Core Python backend
tests/                     Unit, contract, and regression tests
demo/                      LogicFlow demo frontend and API server
experiments/               Optional acceptance and regression runners
docs/                      Architecture notes, evidence, and project writeups
fixtures/                  Sample workflow graphs and regression inputs
```

## Setup

Requirements:

- Python 3.12+
- Node.js 18+ for the LogicFlow demo frontend

Install Python dependencies:

```powershell
pip install -r requirements.txt
```

For real LLM usage, copy the environment template and configure your API key:

```powershell
copy .env.example .env
```

Set `DEEPSEEK_API_KEY` in `.env`. Do not commit `.env`.

## Run The Demo

Build the frontend:

```powershell
cd demo\frontend
npm install
npm run build
cd ..\..
```

Start the demo server:

```powershell
$env:AI_EDIT_PIPELINE="semantic"
python demo\server.py
```

Open:

```text
http://127.0.0.1:8080
```

## Example Prompts

The demo supports prompts such as:

- Put the knowledge retrieval node to the right of the parsing task.
- Add a fact-checking step before publishing the result.
- Rename the failure branch to manual review.
- Route the failure flow to manual handling.
- Add a notification node while keeping the exception path unchanged.
- Move the node just added after risk control.

Chinese prompts are also supported when the configured LLM can understand them, for example:

- 把知识检索节点放到解析任务右侧。
- 在发布结果前增加事实校验。
- 把失败分支改名为人工审核。
- 失败流程转人工复核。
- 把刚才新增的节点放到风控后面。

## Verification

Run the deterministic backend regression suite:

```powershell
python -m pytest -q
python -m compileall src tests
```

Run focused checks for versioning and recent-transaction grounding:

```powershell
python -m pytest tests\test_versioning.py tests\test_recent_transaction_grounding.py -q
```

Optional real-LLM prompt regression:

```powershell
$env:RUN_REAL_LLM_TESTS="1"
python -m pytest tests\test_llm_semantic_goal_prompt_regression.py -q
```

Historical acceptance runners remain under `experiments/`. Their outputs should be treated as evidence for the specific runner conditions, not as a claim that arbitrary workflow edits are supported.

## Project Notes

For resume or interview discussion, the most important point is the architecture:

> The LLM does not directly output graph mutations. It outputs semantic editing intent, and the program handles object grounding, IR lowering, constraint planning, deterministic execution, transactions, and validation.

This is the main difference between WorkflowPilot and a simple "LLM draws a flowchart" demo.
