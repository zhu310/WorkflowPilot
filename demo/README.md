# AI-Flow-Canvas LogicFlow Demo

This demo connects a LogicFlow canvas to the current Semantic Goal editing backend.

## Run

```powershell
cd demo/frontend
npm install
npm run build
cd ..\..
$env:AI_EDIT_PIPELINE="semantic"
python demo\server.py
```

Open:

```text
http://127.0.0.1:8080
```

The server can call the real LLM pipeline. Configure `.env` before starting:

```powershell
copy .env.example .env
# Set DEEPSEEK_API_KEY in .env
```

## Demo Prompts

- 把知识检索节点放到解析任务右侧
- 在发布结果前增加事实校验
- 失败分支改名为人工审核
- 失败流程转人工复核
- 增加通知节点，同时保留异常处理

## Current Architecture

```text
LogicFlow graphData
  -> /api/session/new
  -> /api/ai-edit
  -> Semantic Goal Understanding
  -> Semantic Grounding
  -> Goal Lowering
  -> existing Workflow IR / Compiler / Planner / Executor
  -> committed version, final graph, diff, and user_feedback
  -> LogicFlow render
```

## Product Boundary

The demo now exposes user-facing `success`, `partial_success`, `needs_clarification`, `unsupported`, ambiguous, and missing-object feedback. Ambiguous candidates are stored as a session-local pending clarification; the next user turn can select a candidate with natural phrases such as \u201c\u9009\u7b2c\u4e00\u4e2a\u201d or by naming the visible candidate.

This is not a long-term memory system or a full Agent loop. Current limitations include real viewport-based `visible_area`, complex parallel boundary inference, embedding/vector retrieval, and automatic proposal confirmation for abstract optimization requests.
