# AI-Flow-Canvas LogicFlow Demo

This demo connects a LogicFlow canvas to the WorkflowPilot semantic editing backend.

The frontend sends the current graph, user instruction, and selection state to the backend. The backend runs the Semantic Goal pipeline, commits successful edits into a session-local version state, and returns the final graph, transaction diff, and user-facing feedback.

## Run

Build the frontend:

```powershell
cd demo\frontend
npm install
npm run build
cd ..\..
```

Start the backend server:

```powershell
$env:AI_EDIT_PIPELINE="semantic"
python demo\server.py
```

Open:

```text
http://127.0.0.1:8080
```

For real LLM usage, configure `.env` before starting the server:

```powershell
copy .env.example .env
```

Set `DEEPSEEK_API_KEY` in `.env`. Do not commit `.env`.

## Demo Prompts

English:

- Put the knowledge retrieval node to the right of the parsing task.
- Add a fact-checking step before publishing the result.
- Rename the failure branch to manual review.
- Route the failure flow to manual handling.
- Add a notification node while keeping the exception path unchanged.
- Move the node just added after risk control.

Chinese:

- 把知识检索节点放到解析任务右侧。
- 在发布结果前增加事实校验。
- 把失败分支改名为人工审核。
- 失败流程转人工复核。
- 增加通知节点，同时保留异常处理。
- 把刚才新增的节点放到风控后面。

## API Flow

```text
LogicFlow graphData
  -> /api/session/new
  -> /api/ai-edit
  -> Semantic Goal understanding
  -> Grounding
  -> Lowering
  -> Workflow IR
  -> Compiler / Constraint Planner / Executor
  -> committed version, final graph, diff, and user_feedback
  -> LogicFlow render
```

## Product Boundary

The demo exposes user-facing statuses including `success`, `partial_success`, `needs_clarification`, `unsupported`, and `execution_error`.

Ambiguous candidates are stored as session-local pending clarification. The next user turn can select a candidate with natural phrases such as "choose the first one", "选第一个", or by naming the visible candidate.

This is not a long-term memory system or a full agent loop. Current limitations include real viewport-based `visible_area` grounding, complex parallel boundary inference, embedding/vector retrieval, and automatic proposal confirmation for abstract optimization requests.
