# AI-Flow-Canvas Autumn 2026 Baseline

This document records the frozen autumn recruitment baseline for AI-Flow-Canvas as of 2026-10-10. It is a project handoff and interview-preparation note, not a claim that the system is a complete workflow engine.

## A. Project Positioning

AI-Flow-Canvas is a natural-language workflow editing system. A user describes a desired change to an existing business flow graph, and the system turns that instruction into a controlled graph edit.

The core problem is not drawing nodes and edges. The hard part is preserving business graph correctness while interpreting natural language. A large language model can understand user intent, but it should not directly mutate the final business graph. The production path therefore separates language understanding from deterministic graph mutation and validation.

## B. Current Architecture

The current semantic pipeline is:

Natural language -> Semantic Goal -> Grounding -> RelationFact / Goal Lowering -> Workflow IR -> Compiler -> Constraints -> Planner / Executor -> Validation -> Product Feedback.

Layer responsibilities:

- Semantic Goal: captures user meaning as structured goals, references, relation intents, property goals, object states, and preserve constraints.
- Grounding: binds user references to concrete Business Graph nodes or edges, while preserving ambiguity when a reference cannot be safely resolved.
- RelationFact / Grouper: preserves grouped relation semantics when the user expresses a group of related structural facts. Grouping must not change the user goal.
- Goal Lowering: converts grounded semantic goals into Workflow IR without reinterpreting natural language.
- Workflow IR: represents target structural intent, such as sequence, parallel regions, auxiliary relations, entities, and preserve obligations.
- Compiler: turns IR and current graph facts into deterministic constraints. It must not treat arbitrary graph adjacency as edit authorization.
- Constraints: describe required relations, forbidden relations, entity changes, property changes, and preservation obligations.
- Planner / Executor: apply authorized graph mutations.
- Validation: checks both structural validity and whether user goals and preserve obligations are actually satisfied.
- Product Feedback: explains user-visible results without leaking internal ids or listing unrelated internal preservation constraints.

## C. Verified Capabilities

These capabilities were verified through real browser natural-language testing, not just isolated unit fixtures:

1. Short local reorder
   - User goal: move "工具调用" before "知识检索".
   - Result: local order changed while unrelated graph structure was preserved.
   - Main modules: Semantic Goal, Grounding, Goal Lowering, Workflow IR compiler, Product Feedback.

2. Full local reorder with context
   - User goal: move "工具调用" before "知识检索", while both remain between "解析任务" and "结果判断".
   - Result: graph changed from `解析任务 -> 知识检索 -> 工具调用 -> 结果判断` to `解析任务 -> 工具调用 -> 知识检索 -> 结果判断`.
   - Evidence: real browser execution and semantic entry checks passed.
   - Main modules: RelationFact grouping, grouped precedence lowering, compiler local sequence rewrite, Product Feedback.

3. Basic parallel conversion
   - User goal: turn "知识检索" and "工具调用" into parallel steps.
   - Result: both branches start after "解析任务" and rejoin at "结果判断"; join mode is set to `all`.
   - Main modules: parallel lowering, boundary-constrained compiler conversion, validation.

4. Complex insertion with explicit preserve
   - User goal: insert "复核确认" between "人工复核" and "发布结果" while preserving two failure branches and the direct "生成报告 -> 发布结果" path.
   - Result: inserted node and preserved specified business edges without creating an unauthorized sibling edge.
   - Main modules: edge reference grounding, preserve constraint propagation, scoped compiler ownership, planner/executor, feedback.

5. Multi-turn reference and rename
   - User goal: refer to the recently added node and rename it.
   - Result: recent transaction context resolved the intended node and renamed it without altering topology.
   - Main modules: conversation context, semantic grounding, property update path.

6. Delete inserted node and restore explicit direct connection
   - User goal: delete the inserted confirmation node and reconnect "人工复核" directly to "发布结果" while preserving other business paths.
   - Result: restored the original topology with a new edge id; historical edge id identity is not required.
   - Main modules: absent entity handling, node-to-node direct connection, preserve validation.

## D. Core Design Tradeoffs

The system separates LLM understanding from deterministic graph mutation because a valid natural-language interpretation is not the same as a safe graph edit.

Grounding is required because user phrases are not graph ids. It resolves references to current graph objects, detects ambiguity, and protects against editing the wrong node or edge.

RelationFact grouping preserves relation group identity when a user expresses grouped structure. It must not invent new user goals, and deterministic code must not infer grouping only from graph topology.

Compiler boundary ownership is intentionally constrained. Earlier loose incoming/outgoing rewiring could delete unrelated business edges, such as a direct `report -> publish` path, and could create unlabeled sibling edges. The compiler may rewrite local structure only when the IR and graph facts provide enough evidence that an existing relation belongs to the target replacement.

`before` is not always a direct edge. It is a precedence relation. Some edits require direct topology changes, but the system must not globally equate every before relation with an immediate Business Graph edge.

Validation checks more than graph shape. It must verify user goals, preservation obligations, and absence of unauthorized edits. A structurally valid graph can still be wrong if it drops a protected business edge or reports an unfinished target as success.

## E. Historical Severe Issues And Fixes

- Unauthorized edge loss: earlier structural editing could delete `report -> publish` while inserting a node after a merge point. The fix constrained compiler boundary ownership and added preserve validation.
- Extra unlabeled sibling edge: a loose boundary rule could add an unlabeled `report -> human` while a labeled failure edge already existed. The fix preserved edge identity, label, condition, and property semantics instead of endpoint-only merging.
- Over-constrained compiler: after tightening boundaries, basic reorder and parallel edits were blocked. Later work restored safe local sequence and parallel conversions without returning to global topology rewiring.
- Grouped vs ungrouped lowering inconsistency: grouped RelationFacts and legacy binary relation paths could produce unequal IR. The current baseline preserves grouped relation semantics through lowering while keeping compatibility paths.
- Plain local sequence swap: simple reorder and full local reorder now work when the graph provides a clear local chain and preservation constraints do not conflict.
- Unsupported reported as clarification: structural compiler limits were sometimes surfaced as vague "needs clarification". Feedback now distinguishes user ambiguity from system unsupported boundaries more carefully.
- Feedback summarized context instead of actual change: grouped reorder feedback could say "解析任务 before 工具调用", which was already true before the transaction. Feedback now prefers ordering relations that became true in the transaction.

## F. Test And Acceptance Record

Known recent records:

- Real browser checks: the six capability scenarios above passed.
- Real semantic entry checks after feedback fix: short reorder, full reorder, basic parallel, and parse/retrieval parallel returned success.
- Feedback-related tests after the final feedback fix: `17 passed`.
- `python -m compileall src`: passed.
- Full Semantic Acceptance run: `runs/semantic_acceptance/20261009_181125`, result `28/40`.
- Previous comparable Semantic Acceptance baseline: `22/40`.
- Reported delta for the 20261009_181125 run: 6 FAIL -> PASS, 0 PASS -> FAIL.

This baseline does not claim full acceptance. The 40-case suite still has 12 non-pass cases.

## G. Current Capability Boundaries

The following are known limits and should not be oversold:

- Complex ordering involving success/failure edges, branch labels, conditions, or other business metadata is not fully supported.
- `test_contiguous_ordering_shadow_preserves_failure_branch_metadata` remains a known failing deterministic test.
- Complex relocate operations and nested structural edits are not fully closed.
- Explicit edge deletion is not treated as a complete product capability.
- The system supports selected grounded route/branch edits, but arbitrary route semantics and vague branch references can still require clarification.
- The current workflow editor is not a general-purpose commercial workflow engine. It is a research/prototype system with a correctness-focused semantic editing pipeline.

## H. Interview Narrative

AI-Flow-Canvas demonstrates how to build a natural-language editor for business workflow graphs without letting the LLM directly mutate production state.

The key technical story:

- The LLM extracts user intent into semantic goals.
- Grounding maps user phrases to real graph nodes and edges.
- A deterministic lowering/compiler pipeline turns goals into authorized graph constraints.
- Planner and executor apply only validated mutations.
- Validation and feedback ensure the final graph actually matches the user goal and that protected business relations remain intact.

The most important engineering challenge was correctness under structural edits. A visually successful edit can still be wrong if it silently deletes a non-target business edge. The project therefore invests heavily in edge identity, preserve constraints, scoped compiler ownership, and final graph validation.

The strongest verified demos for autumn recruitment are:

- reorder two steps,
- convert two serial steps into parallel branches,
- insert a node after a merge while preserving failure and direct paths,
- rename a recently added node across turns,
- delete that node and restore a direct connection.

The honest limitation is that complex structural rewrites involving branch metadata and arbitrary relocate semantics remain bounded. This is intentional for the frozen version: correctness and safe stopping are preferred over guessing edits that might corrupt the business graph.
