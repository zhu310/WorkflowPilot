# WorkflowPilot

WorkflowPilot：基于大模型的智能流程图编辑系统

WorkflowPilot: An LLM-powered workflow graph editing system.

WorkflowPilot turns natural-language editing requests into safe, structured workflow graph transactions. It accepts a current graph plus request context, grounds user intent to graph objects, builds deterministic obligations and target state, uses Workflow IR only where semantic structure is still needed, plans minimal changes, executes them, validates the final graph, and records versioned results.

## Core Pipeline

```text
User Instruction
↓
LLM Understanding
↓
Grounding
↓
Desired State
↓
Constraint Validation
↓
Workflow IR
↓
Planner
↓
Executor
↓
Graph Validation
```

The LLM is not trusted to directly mutate the graph. Confirmed graph facts are lowered deterministically, ambiguous requests enter clarification, and unauthorized or unintended changes are rejected.

## Verified Results

| Acceptance area | Result |
|---|---:|
| Formal Regression | 200 / 200 correct |
| Large Graph Acceptance | 360 nodes / 399 edges |
| Multi-turn Acceptance | 4 / 4 cases |
| Deterministic Stress | 5100 / 5100 checks |
| False success | 0 |
| Unauthorized mutation | 0 |
| Unintended change | 0 |

Detailed evidence is in `docs/evidence/`, especially `docs/evidence/capability_matrix.md`.

## What It Supports

- Explicit edits: rename, move, insert, delete, edge redirect.
- Complex topology edits: move-after, serial-to-parallel, retry/exception/failure combinations, labeled flow.
- Scope handling: explicit object, selected nodes/edges, visible area, stage/group/subprocess, and global canvas.
- Clarification: ambiguous object, branch, scope, and historical reference cases.
- Multi-turn editing: versioned graph continuation, rollback/redo, and recent transaction mutation reference grounding.
- Conditional branches: branch labels and condition metadata are preserved through local edits and active redirects.
- Visualization: before/after graph views, changed objects, preserved objects, transaction steps, and trace summary.

## Repository Layout

```text
src/                       core implementation
tests/                     unit and deterministic tests
experiments/acceptance/    acceptance runners for final capabilities
experiments/regression/    formal and safety regression runners
docs/evidence/             final evidence summaries and capability matrix
docs/demo/                 before/after visualization demo
docs/archive/              historical design and diagnosis notes
```

## Setup

1. Install Python 3.12+.
2. Install test dependencies:

```powershell
pip install -r requirements.txt
```

3. Copy `.env.example` to `.env` and set `DEEPSEEK_API_KEY` locally. Do not commit `.env`.

## Verification

```powershell
python -m pytest
python -m experiments.deterministic_stress
python -m experiments.stage3_condition_branch_acceptance
python -m experiments.stage3_node_type_catalog_acceptance
python -m experiments.stage3_large_graph_acceptance
```

The full formal regression can be run with:

```powershell
python -m experiments.workflow_ir_formal_regression
```

Raw run artifacts are intentionally excluded from the recommended GitHub upload. Curated summaries live in `docs/evidence/`.
