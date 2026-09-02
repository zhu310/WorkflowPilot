# WorkflowPilot Capability Matrix

This matrix summarizes the frozen acceptance evidence for the current GitHub release candidate.

WorkflowPilot：基于大模型的智能流程图编辑系统

WorkflowPilot: An LLM-powered workflow graph editing system.

## Verified Capabilities

| Capability | Status | Evidence |
|---|---|---|
| Structured graph input and structured output | VERIFIED | `docs/evidence/formal_200_summary.json` |
| Program-readable transaction result, constraints, planner steps, and final graph | VERIFIED | `docs/evidence/formal_200_summary.json` |
| Single-node property edit / rename | VERIFIED | `docs/evidence/formal_200_summary.json`, `docs/evidence/node_type_catalog_summary.json` |
| Node insert with relation repair | VERIFIED | `docs/evidence/formal_200_summary.json` |
| Node delete | VERIFIED | `docs/evidence/formal_200_summary.json` |
| Edge create/delete/redirect | VERIFIED | `docs/evidence/formal_200_summary.json` |
| Labeled relation edit | VERIFIED | `docs/evidence/formal_200_summary.json`, `docs/evidence/large_graph_summary.json` |
| Conditional branch preservation and active redirect metadata preservation | VERIFIED | `docs/evidence/condition_branch_summary.json` |
| Serial-to-parallel topology rewrite | VERIFIED | `docs/evidence/formal_200_summary.json` |
| Retry / exception / failure side relation preservation | VERIFIED | `docs/evidence/formal_200_summary.json`, `docs/evidence/condition_branch_summary.json` |
| Mixed structural, relation, and property edits | VERIFIED | `docs/evidence/formal_200_summary.json` |
| Explicit node scope | VERIFIED | `docs/evidence/formal_200_summary.json`, `docs/evidence/large_graph_summary.json` |
| Selected-node scope | VERIFIED | `docs/evidence/multiturn_summary.json`, `experiments/runs/stage3-selected-layout-acceptance-20260902-135734/summary.json` |
| Selected-edge scope | VERIFIED | `docs/evidence/condition_branch_summary.json` |
| Visible-area scope | VERIFIED | `docs/evidence/large_graph_summary.json` |
| Stage / group / subprocess scope grounding and preservation | VERIFIED | `docs/evidence/large_graph_summary.json` |
| Global full-canvas scope | VERIFIED | `docs/evidence/large_graph_summary.json` |
| Ambiguity and clarification lifecycle | VERIFIED | `docs/evidence/large_graph_summary.json`, `docs/evidence/condition_branch_summary.json`, `docs/evidence/abstract_intent_summary.json`, `docs/evidence/multiturn_summary.json` |
| Abstract intent without false success | VERIFIED | `docs/evidence/abstract_intent_summary.json` |
| Abstract plus explicit mixed request | VERIFIED | `docs/evidence/abstract_intent_summary.json` |
| Multi-turn edit based on latest graph state | VERIFIED | `docs/evidence/multiturn_summary.json` |
| Recent transaction mutation reference grounding | VERIFIED | `docs/evidence/multiturn_summary.json` |
| Rollback / redo snapshot consistency | VERIFIED | `docs/evidence/multiturn_summary.json` |
| 300+ complex graph acceptance | VERIFIED | `docs/evidence/large_graph_summary.json` |
| Node type catalog ingestion and preservation | VERIFIED | `docs/evidence/node_type_catalog_summary.json` |
| Invalid graph prevention | VERIFIED | `docs/evidence/formal_200_summary.json`, `docs/evidence/deterministic_stress_summary.json` |
| Unauthorized and unintended mutation prevention | VERIFIED | `docs/evidence/formal_200_summary.json`, `docs/evidence/deterministic_stress_summary.json` |
| Before/after visualization and changed/preserved summary | VERIFIED | `docs/evidence/visualization_diff_demo_manifest.json`, `docs/demo/visualization_diff_demo/index.html` |

## Core Safety Baseline

| Metric | Current evidence |
|---|---|
| Formal regression | 200 / 200 correct |
| Safe rejection | 0 |
| Incorrect execution | 0 |
| Invalid graph | 0 |
| Unauthorized mutation | 0 |
| Unintended change | 0 |
| Deterministic stress | Passed |

See `docs/evidence/formal_200_summary.json` and `docs/evidence/deterministic_stress_summary.json`.

## Current Boundaries

| Boundary | Status | Notes |
|---|---|---|
| Selected multi-node align | Not implemented | Non-blocking layout enhancement. Group move and selected-only preservation are verified. |
| Selected multi-node distribute | Not implemented | Non-blocking layout enhancement. |
| Group create/delete/add/remove/resize | Not implemented | Current requirement coverage is group/stage/subprocess reading, scope grounding, and preservation. |
| Condition expression editing | Not implemented | Conditional branch preservation and redirect metadata preservation are verified. |
| Branch label editing | Not implemented | Branch labels are preserved through unrelated edits and conditional redirects. |
| Dedicated loop analysis / unreachable-node repair proposal | Not implemented | Advanced graph analysis, not required for the current release acceptance baseline. |
| 1000+ node graph | Untested | Bonus-scale scenario. Current verified large graph is 360 nodes / 399 edges. |
| Token / model cost reporting | Partial | Evidence schema has cost fields, but current summaries do not include concrete cost accounting. |
| Systematic latency benchmark | Partial | Large graph acceptance records per-case latency, but no broad latency distribution benchmark is included. |

## Non-Blocking Enhancements

- Precise layout alignment and distribution.
- Group CRUD and group resizing.
- Condition expression and branch label active editing.
- Dedicated graph analysis for unreachable nodes, loops, and repair suggestions.
- 1000+ node stress acceptance.
- Richer performance and cost dashboards.

These items are useful future work, but they are not blockers for the current acceptance baseline.
