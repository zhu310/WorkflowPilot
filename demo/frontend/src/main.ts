import LogicFlow from '@logicflow/core';
import '@logicflow/core/lib/style/index.css';
import './style.css';

type WorkflowNode = {
  id: string;
  type: string;
  label: string;
  x: number;
  y: number;
  width?: number;
  height?: number;
  join?: { mode: string };
};

type WorkflowEdge = {
  id: string;
  source: string;
  target: string;
  label?: string;
};

type WorkflowGraph = {
  nodes: WorkflowNode[];
  edges: WorkflowEdge[];
};

type EditResult = {
  status: string;
  error?: string | null;
  clarification_message?: string;
  steps: Record<string, unknown>[];
  final_graph?: WorkflowGraph | null;
  graph?: WorkflowGraph;
  diff?: Record<string, unknown>;
  specification?: Record<string, unknown>;
  session_id?: string;
  current_version_id?: string;
  transaction?: Record<string, unknown>;
  transaction_diff?: Record<string, unknown> | null;
  large_graph?: Record<string, unknown>;
  mutation_authorization?: Record<string, unknown>[] | null;
};

type DiagnosisResult = {
  status: string;
  error?: string;
  diagnosis?: {
    analysis: string;
    options: Array<{
      id: string;
      title: string;
      description: string;
      tradeoff: string;
    }>;
    recommended_option_id: string;
    recommended_instruction: string;
    option_instructions?: Record<string, string>;
    question: string;
    requires_confirmation: boolean;
  };
};

const initialGraph: WorkflowGraph = {
  nodes: [
    { id: 'n_start', type: 'start', label: '用户需求', x: 110, y: 260, width: 96, height: 96 },
    { id: 'n_submit', type: 'task', label: '解析任务', x: 280, y: 260, width: 140, height: 72 },
    { id: 'n_retrieval', type: 'retrieval', label: '知识检索', x: 470, y: 180, width: 140, height: 72 },
    { id: 'n_tool', type: 'tool_call', label: '工具调用', x: 470, y: 340, width: 140, height: 72 },
    { id: 'n_approval_decision', type: 'decision', label: '结果判断', x: 680, y: 260, width: 104, height: 104 },
    { id: 'n_report', type: 'llm_reasoning', label: '生成报告', x: 890, y: 220, width: 140, height: 72 },
    { id: 'n_publish', type: 'output', label: '发布结果', x: 1090, y: 220, width: 140, height: 72 },
    { id: 'n_human', type: 'human_review', label: '人工复核', x: 890, y: 420, width: 140, height: 72 },
    { id: 'n_end', type: 'end', label: '结束', x: 1280, y: 260, width: 96, height: 96 }
  ],
  edges: [
    { id: 'e_start_submit', source: 'n_start', target: 'n_submit', label: '' },
    { id: 'e_submit_retrieval', source: 'n_submit', target: 'n_retrieval', label: '' },
    { id: 'e_retrieval_tool', source: 'n_retrieval', target: 'n_tool', label: '' },
    { id: 'e_tool_decision', source: 'n_tool', target: 'n_approval_decision', label: '' },
    { id: 'e_decision_report', source: 'n_approval_decision', target: 'n_report', label: 'success' },
    { id: 'e_decision_human', source: 'n_approval_decision', target: 'n_human', label: 'failure' },
    { id: 'e_report_publish', source: 'n_report', target: 'n_publish', label: '' },
    { id: 'e_report_human', source: 'n_report', target: 'n_human', label: 'failure' },
    { id: 'e_human_publish', source: 'n_human', target: 'n_publish', label: '' },
    { id: 'e_publish_end', source: 'n_publish', target: 'n_end', label: '' }
  ]
};

const nodeTypeMap: Record<string, string> = {
  start: 'circle',
  end: 'circle',
  decision: 'diamond'
};

const typeLabel: Record<string, string> = {
  start: '入口',
  task: '任务',
  retrieval: '检索',
  tool_call: '工具',
  decision: '判断',
  llm_reasoning: 'LLM',
  output: '输出',
  human_review: '人工',
  checkpoint: '校验',
  end: '结束'
};

const root = document.querySelector<HTMLDivElement>('#root');
if (!root) throw new Error('missing root');

root.innerHTML = `
  <main class="app">
    <section class="canvas-panel">
      <div class="topbar">
        <div>
          <h1>AI-Flow-Canvas</h1>
          <p>自然语言修改 AI Agent 工作流</p>
        </div>
        <div class="top-actions">
          <button id="zoomOut" title="缩小画布">-</button>
          <button id="zoomIn" title="放大画布">+</button>
          <button id="fitView" title="适配当前流程图">适配</button>
          <button id="reset">重置</button>
        </div>
      </div>
      <div id="graph"></div>
    </section>
    <aside class="assistant-panel">
      <div class="assistant-head">
        <div>
          <h2>AI 流程助手</h2>
          <p id="session">正在连接真实 LLM</p>
        </div>
        <span id="versionBadge">v1</span>
      </div>
      <div id="messages" class="messages" aria-live="polite"></div>
      <div class="toolbar">
        <button id="undo">撤销</button>
        <button id="redo">重做</button>
        <button id="large" title="运行工程侧 360 节点压力测试">大图实测</button>
      </div>
      <form id="chatForm" class="composer">
        <textarea id="instruction" placeholder="直接描述你想怎么改流程，例如：发布结果必须来自人工复核的输出，不要让生成报告直接发布。"></textarea>
        <button class="primary" id="run" type="submit">发送</button>
      </form>
    </aside>
  </main>
`;

const graphEl = document.querySelector<HTMLDivElement>('#graph')!;
const instructionEl = document.querySelector<HTMLTextAreaElement>('#instruction')!;
const chatForm = document.querySelector<HTMLFormElement>('#chatForm')!;
const messagesEl = document.querySelector<HTMLDivElement>('#messages')!;
const sessionEl = document.querySelector<HTMLParagraphElement>('#session')!;
const versionBadgeEl = document.querySelector<HTMLSpanElement>('#versionBadge')!;
let workflowGraph: WorkflowGraph = structuredClone(initialGraph);
let sessionId = '';
let currentVersionId = '';
let quickSelectionOverride: { node_ids: string[]; edge_ids: string[] } | null = null;
let pendingAssistantEl: HTMLDivElement | null = null;
let conversationHistory: Array<{ role: 'user' | 'assistant'; content: string }> = [];
let lastEditFocus: Record<string, unknown> | null = null;

const lf = new LogicFlow({
  container: graphEl,
  grid: true,
  keyboard: { enabled: true },
  stopScrollGraph: true,
  stopZoomGraph: true
});

function workflowToLogicFlow(graph: WorkflowGraph) {
  return {
    nodes: graph.nodes.map((node) => ({
      id: node.id,
      type: nodeTypeMap[node.type] ?? 'rect',
      x: node.x,
      y: node.y,
      properties: {
        workflowType: node.type,
        typeLabel: typeLabel[node.type] ?? node.type,
        join: node.join,
        width: node.width,
        height: node.height
      },
      ...(node.width ? { width: node.width } : {}),
      ...(node.height ? { height: node.height } : {}),
      text: node.label,
    })),
    edges: graph.edges.map((edge) => ({
      id: edge.id,
      type: 'polyline',
      sourceNodeId: edge.source,
      targetNodeId: edge.target,
      text: edge.label ?? '',
      properties: {
        label: edge.label ?? ''
      }
    }))
  };
}

function logicFlowToWorkflow(): WorkflowGraph {
  const data = lf.getGraphData() as any;
  return {
    nodes: data.nodes.map((node: any) => ({
      id: node.id,
      type: node.properties?.workflowType ?? 'task',
      label: typeof node.text === 'string' ? node.text : node.text?.value ?? '',
      x: node.x,
      y: node.y,
      ...(node.properties?.width ? { width: node.properties.width } : {}),
      ...(node.properties?.height ? { height: node.properties.height } : {}),
      ...(node.properties?.join ? { join: node.properties.join } : {})
    })),
    edges: data.edges.map((edge: any) => ({
      id: edge.id,
      source: edge.sourceNodeId,
      target: edge.targetNodeId,
      label: edge.properties?.label ?? (typeof edge.text === 'string' ? edge.text : edge.text?.value ?? '')
    })),
    ...Object.fromEntries(
      Object.entries(workflowGraph).filter(([key]) => !['nodes', 'edges'].includes(key))
    )
  };
}

function getSelection() {
  const selected = lf.getSelectElements(false) as any;
  const actual = {
    node_ids: selected.nodes.map((node: any) => node.id),
    edge_ids: selected.edges.map((edge: any) => edge.id)
  };
  if (actual.node_ids.length || actual.edge_ids.length) {
    return actual;
  }
  return quickSelectionOverride ?? actual;
}

function render(graph: WorkflowGraph) {
  workflowGraph = structuredClone(graph);
  lf.render(workflowToLogicFlow(workflowGraph));
  lf.fitView(36, 36);
  updateSelection();
}

function pretty(value: unknown) {
  return JSON.stringify(value, null, 2);
}

function updateSelection() {
  getSelection();
}

lf.on('selection:selected,node:click,edge:click,blank:click', updateSelection);

graphEl.addEventListener(
  'wheel',
  (event) => {
    event.preventDefault();
    lf.translate(-event.deltaX, -event.deltaY);
  },
  { passive: false }
);

function appendMessage(role: 'user' | 'assistant' | 'system', text: string) {
  const row = document.createElement('div');
  row.className = `message ${role}`;
  row.textContent = text;
  messagesEl.append(row);
  messagesEl.scrollTop = messagesEl.scrollHeight;
  return row;
}

function rememberMessage(role: 'user' | 'assistant', text: string) {
  conversationHistory.push({ role, content: text });
}

function appendBusinessMessage(role: 'user' | 'assistant', text: string) {
  const row = appendMessage(role, text);
  rememberMessage(role, text);
  return row;
}

function completeAssistantBusinessMessage(row: HTMLDivElement, text: string) {
  row.textContent = text;
  rememberMessage('assistant', text);
}

function historySnapshot() {
  return conversationHistory.map((message) => ({ ...message }));
}

function idsFromDiff(result: EditResult) {
  const diff = (result.diff ?? {}) as any;
  const nodeIds = new Set<string>();
  const edgeRefs: Array<{ source: string; target: string; label?: string }> = [];
  const collectEdge = (edge: any) => {
    if (!edge || typeof edge !== 'object') return;
    if (typeof edge.source === 'string') nodeIds.add(edge.source);
    if (typeof edge.target === 'string') nodeIds.add(edge.target);
    if (typeof edge.source === 'string' && typeof edge.target === 'string') {
      edgeRefs.push({ source: edge.source, target: edge.target, label: edge.label ?? '' });
    }
  };
  (diff.added_nodes ?? []).forEach((node: any) => typeof node?.id === 'string' && nodeIds.add(node.id));
  (diff.removed_nodes ?? []).forEach((node: any) => typeof node?.id === 'string' && nodeIds.add(node.id));
  (diff.changed_nodes ?? []).forEach((node: any) => typeof node?.id === 'string' && nodeIds.add(node.id));
  (diff.added_edges ?? []).forEach(collectEdge);
  (diff.removed_edges ?? []).forEach(collectEdge);
  return { node_ids: Array.from(nodeIds), edge_refs: edgeRefs };
}

function asksForAdvice(userText: string) {
  const compact = userText.replace(/\s+/g, '');
  const asksToInspect =
    compact.includes('检查') ||
    compact.includes('看一下') ||
    compact.includes('分析') ||
    compact.includes('评估');
  const mentionsLogic =
    compact.includes('逻辑') ||
    compact.includes('混乱') ||
    compact.includes('有点乱') ||
    compact.includes('乱');
  return (
    compact.includes('几种') ||
    compact.includes('哪一种') ||
    compact.includes('为什么') ||
    compact.includes('為什麼') ||
    compact.includes('推荐') ||
    compact.includes('改法') ||
    compact.includes('符合逻辑') ||
    compact.includes('逻辑有问题') ||
    compact.includes('逻辑混乱') ||
    compact.includes('逻辑不对') ||
    compact.includes('有点乱') ||
    compact.includes('有错误') ||
    compact.includes('哪里有问题') ||
    compact.includes('这里有问题') ||
    compact.includes('有没有问题') ||
    compact.includes('有什么问题') ||
    compact.includes('是不是有问题') ||
    compact.includes('是不是混乱') ||
    compact.includes('是否合理') ||
    compact.includes('合理吗') ||
    compact.includes('哪里不合理') ||
    compact.includes('需要优化') ||
    compact.includes('怎么优化') ||
    (asksToInspect && mentionsLogic)
  );
}

function asksForEdit(userText: string) {
  const compact = userText.replace(/\s+/g, '');
  return (
    /^把/u.test(compact) ||
    compact.includes('修改') ||
    compact.includes('改成') ||
    compact.includes('改为') ||
    compact.includes('删除') ||
    compact.includes('新增') ||
    compact.includes('插入') ||
    compact.includes('连接') ||
    compact.includes('直接改') ||
    compact.includes('帮我改') ||
    compact.includes('就按') ||
    compact.includes('按照')
  );
}

function summarizeDiagnosis(result: DiagnosisResult) {
  const diagnosis = result.diagnosis;
  if (!diagnosis) {
    return result.error ? `诊断失败：${result.error}` : '暂时无法完成诊断。';
  }
  const lines = [diagnosis.analysis];
  if (diagnosis.options.length) {
    lines.push('', '可选改法：');
    diagnosis.options.forEach((option, index) => {
      const suffix = option.tradeoff ? `（${option.tradeoff}）` : '';
      lines.push(`${index + 1}. ${option.title}：${option.description}${suffix}`);
    });
  }
  if (diagnosis.question) {
    lines.push('', diagnosis.question);
  }
  return lines.join('\n');
}

async function requestDiagnosis(userText: string, history: Array<{ role: 'user' | 'assistant'; content: string }>) {
  return (await fetch('/api/diagnose', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      session_id: sessionId,
      graph: workflowGraph,
      message: userText,
      conversation_history: history,
      focus_context: lastEditFocus ?? {}
    })
  }).then((response) => response.json())) as DiagnosisResult;
}

function friendlyBackendError(result: EditResult) {
  const raw = `${result.clarification_message || result.error || result.status || ''}`;
  if (raw.includes('underspecified abstract goal')) {
    return '这个问题更像是在让 AI 判断流程逻辑，而不是一个已经确定的修改指令。我会先给出可选改法，请你确认后再执行。';
  }
  if (raw.includes('required structural participants not fully realized')) {
    return '暂时不能安全落图：后端没有完整实现这次结构修改目标。请把希望形成的最终关系说得更具体，或者让我先给出几种可选改法。';
  }
  if (raw.includes('parallel structure lacks unambiguous boundary')) {
    return '暂时不能安全改成并行：当前描述没有形成唯一的并行入口、并行分支和汇聚节点。请说明哪些节点作为并行分支，以及它们完成后汇聚到哪个节点。';
  }
  if (raw.includes('unsupported edit specification')) {
    return '暂时不能安全落图：当前描述还不能转换成可靠的流程修改方案。';
  }
  return raw || '暂时不能安全修改。';
}

function summarizeResult(result: EditResult) {
  if (result.status === 'success') {
    const added = Array.isArray((result.diff as any)?.added_edges) ? (result.diff as any).added_edges.length : 0;
    const removed = Array.isArray((result.diff as any)?.removed_edges) ? (result.diff as any).removed_edges.length : 0;
    const changedNodes = Array.isArray((result.diff as any)?.changed_nodes) ? (result.diff as any).changed_nodes.length : 0;
    const parts = [];
    if (added) parts.push(`新增 ${added} 条连线`);
    if (removed) parts.push(`删除 ${removed} 条连线`);
    if (changedNodes) parts.push(`更新 ${changedNodes} 个节点`);
    return parts.length ? `已完成修改：${parts.join('，')}。` : '已完成修改，流程图已更新。';
  }
  if (result.status === 'needs_clarification') {
    return friendlyBackendError(result) || '我需要再确认一下你的修改意图。';
  }
  if (result.status === 'llm_error') {
    return '真实 LLM 调用失败：请检查 API Key、模型服务余额或网络连接后重试。';
  }
  return friendlyBackendError(result);
}

function graphContext() {
  const nodes = workflowGraph.nodes.map((node) => `${node.label}=${node.id}`).join('；');
  const edges = workflowGraph.edges
    .map((edge) => {
      const source = workflowGraph.nodes.find((node) => node.id === edge.source)?.label ?? edge.source;
      const target = workflowGraph.nodes.find((node) => node.id === edge.target)?.label ?? edge.target;
      return `${source}->${target}${edge.label ? `(${edge.label})` : ''}`;
    })
    .join('；');
  return { node_mapping: nodes, current_edges: edges };
}

document.querySelector<HTMLButtonElement>('#reset')!.addEventListener('click', () => {
  sessionId = '';
  currentVersionId = '';
  conversationHistory = [];
  lastEditFocus = null;
  messagesEl.textContent = '';
  appendMessage('assistant', '已重置为初始流程。你可以直接描述希望怎么改，我会在后台调用真实 LLM 并更新左侧画布。');
  render(initialGraph);
  void createSession(initialGraph);
});

chatForm.addEventListener('submit', async (event) => {
  event.preventDefault();
  const userText = instructionEl.value.trim();
  if (!userText) return;
  if (!sessionId) {
    await createSession(workflowGraph);
  }
  const requestHistory = historySnapshot();
  appendBusinessMessage('user', userText);
  instructionEl.value = '';
  if (asksForAdvice(userText) && !asksForEdit(userText)) {
    const pending = appendMessage('assistant', '正在检查当前流程结构...');
    const diagnosis = await requestDiagnosis(userText, requestHistory);
    const summary = summarizeDiagnosis(diagnosis);
    completeAssistantBusinessMessage(pending, summary);
    return;
  }
  pendingAssistantEl = appendMessage('assistant', '正在理解你的修改意图...');
  await new Promise((resolve) => setTimeout(resolve, 180));
  if (pendingAssistantEl) pendingAssistantEl.textContent = '正在调用真实 LLM 生成 Workflow IR...';
  const result = (await fetch('/api/ai-edit', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      session_id: sessionId,
      instruction: userText,
      selection: getSelection(),
      request_context: {
        visible_area: { left: 0, top: 0, right: 1400, bottom: 640 },
        original_user_message: userText,
        conversation_history: requestHistory,
        graph_context: graphContext()
      }
    })
  }).then((response) => response.json())) as EditResult;

  applyServerState(result);
  if (pendingAssistantEl) {
    const summary = summarizeResult(result);
    completeAssistantBusinessMessage(pendingAssistantEl, summary);
    pendingAssistantEl = null;
  }
  if (result.status === 'success') {
    lastEditFocus = {
      last_user_instruction: userText,
      current_version_id: result.current_version_id,
      changed_area: idsFromDiff(result),
      diff: result.diff ?? {}
    };
  }
});

document.querySelector<HTMLButtonElement>('#zoomOut')!.addEventListener('click', () => {
  lf.zoom(false);
});

document.querySelector<HTMLButtonElement>('#zoomIn')!.addEventListener('click', () => {
  lf.zoom(true);
});

document.querySelector<HTMLButtonElement>('#fitView')!.addEventListener('click', () => {
  lf.fitView(36, 36);
});

document.querySelector<HTMLButtonElement>('#undo')!.addEventListener('click', async () => {
  const result = (await fetch('/api/undo', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ session_id: sessionId })
  }).then((response) => response.json())) as EditResult;
  applyServerState(result);
  appendMessage('assistant', result.status === 'success' ? '已撤销上一步修改。' : '当前没有可以撤销的修改。');
});

document.querySelector<HTMLButtonElement>('#redo')!.addEventListener('click', async () => {
  const result = (await fetch('/api/redo', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ session_id: sessionId })
  }).then((response) => response.json())) as EditResult;
  applyServerState(result);
  appendMessage('assistant', result.status === 'success' ? '已重做刚才撤销的修改。' : '当前没有可以重做的修改。');
});

document.querySelector<HTMLButtonElement>('#large')!.addEventListener('click', async () => {
  const pending = appendMessage('assistant', '正在运行 360 节点真实大图实测，这会稍微久一点...');
  const result = (await fetch('/api/large-graph-test', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ limit: 5 })
  }).then((response) => response.json())) as EditResult;
  applyServerState(result);
  const cases = Array.isArray((result.large_graph as any)?.cases) ? (result.large_graph as any).cases : [];
  const passed = cases.filter((row: any) => row.status === 'success' && row.valid_final_graph).length;
  const clarification = cases.filter((row: any) => row.status === 'needs_clarification').length;
  pending.textContent = `大图实测完成：${passed} 个用例成功落图，${clarification} 个用例需要澄清；当前图规模 ${((result.large_graph as any)?.node_count ?? workflowGraph.nodes.length)} 个节点。`;
});

async function createSession(graph: WorkflowGraph) {
  const result = (await fetch('/api/session/new', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ graph })
  }).then((response) => response.json())) as EditResult;
  applyServerState(result);
}

function applyServerState(result: EditResult) {
  if (result.session_id) sessionId = result.session_id;
  if (result.current_version_id) currentVersionId = result.current_version_id;
  if (result.graph) render(result.graph);
  else if (result.final_graph) render(result.final_graph);
  quickSelectionOverride = null;
  versionBadgeEl.textContent = currentVersionId || 'v1';
  sessionEl.textContent = `${workflowGraph.nodes.length} 个节点，${workflowGraph.edges.length} 条连线`;
}

render(initialGraph);
void createSession(initialGraph);
appendMessage('assistant', '你好，我可以帮你直接修改左侧流程图。请用业务语言描述目标，例如：发布结果必须来自人工复核的输出，不要让生成报告直接发布。');
