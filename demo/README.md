# AI-Flow-Canvas LogicFlow Demo

This demo connects a LogicFlow canvas to the WorkflowPilot editing backend.

## Run

```powershell
cd demo/frontend
npm install
npm run build
cd ..\..
python demo\server.py
```

Open:

```text
http://127.0.0.1:8080
```

The server calls the real WorkflowPilot LLM pipeline. Configure `.env` before starting:

```powershell
copy .env.example .env
# Set DEEPSEEK_API_KEY in .env
python demo\server.py
```

The page also exposes undo, redo, multi-turn version history, and a 360-node large graph test entry.

## Demo Prompts

- 把 AI 初审改名为智能材料初审
- 把知识检索和工具调用改成并行执行，最后汇总到结果判断
- 在生成报告前增加事实校验
- 失败分支统一进入人工复核

## Architecture

```text
LogicFlow graphData
  -> /api/session/new
  -> /api/ai-edit
  -> run_versioned_transaction
  -> run_workflow_ir_transaction
  -> Workflow IR / constraints / planner steps
  -> committed version, final graph, and diff
  -> LogicFlow render
```
