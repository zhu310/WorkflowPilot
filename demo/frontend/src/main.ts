import LogicFlow from '@logicflow/core';
import { Dagre, ElkLayout } from '@logicflow/layout';
import ELK from 'elkjs/lib/elk.bundled.js';
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
  [key: string]: unknown;
};

type WorkflowEdge = {
  id: string;
  source: string;
  target: string;
  label?: string;
  branch_label?: string;
  condition?: { expression?: string; [key: string]: unknown };
  metadata?: Record<string, unknown>;
  properties?: Record<string, unknown>;
  [key: string]: unknown;
};

type WorkflowGraph = {
  nodes: WorkflowNode[];
  edges: WorkflowEdge[];
  [key: string]: unknown;
};

type EditResult = {
  status: string;
  error?: string | null;
  clarification_message?: string;
  message?: string;
  pending_id?: string | null;
  user_feedback?: {
    status: string;
    message: string;
    interaction?: { type?: string; prompt?: string; slot?: string; candidate_type?: string };
    executed: Array<{ summary: string }>;
    incomplete: Array<{ summary: string; reason: string; status: string }>;
    clarification: {
      question: string;
      candidates: Array<{
        reference: string;
        type: string;
        candidates: Array<{ id: string; type: string; label: string; reason: string }>;
      }>;
    };
  };
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
    { id: 'n_retrieval', type: 'retrieval', label: '知识检索', x: 470, y: 150, width: 140, height: 72 },
    { id: 'n_tool', type: 'tool_call', label: '工具调用', x: 470, y: 390, width: 140, height: 72 },
    { id: 'n_approval_decision', type: 'decision', label: '结果判断', x: 700, y: 270, width: 104, height: 104 },
    { id: 'n_report', type: 'llm_reasoning', label: '生成报告', x: 960, y: 120, width: 140, height: 72 },
    { id: 'n_publish', type: 'output', label: '发布结果', x: 1240, y: 120, width: 140, height: 72 },
    { id: 'n_human', type: 'human_review', label: '人工复核', x: 960, y: 520, width: 140, height: 72 },
    { id: 'n_end', type: 'end', label: '结束', x: 1500, y: 270, width: 96, height: 96 }
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
          <button id="fitView" title="整理当前流程图布局">整理布局</button>
          <button id="reset">重置</button>
        </div>
      </div>
      <div id="graph"></div>
    </section>
    <aside class="assistant-panel">
      <div class="assistant-head">
        <div>
          <h2>AI 流程助手</h2>
          <p id="session">正在连接会话</p>
        </div>
        <span id="versionBadge">v1</span>
      </div>
      <div id="messages" class="messages" aria-live="polite"></div>
      <div class="toolbar">
        <button id="undo">撤销</button>
        <button id="redo">重做</button>
        <button id="large" title="运行工程侧 360 节点压力测试">大图实测</button>
      </div>
      <div class="selection-debug">
        <div class="selection-title-row">
          <strong>当前选中</strong>
          <button id="clearSelection" type="button">清空选择</button>
        </div>
        <div id="currentSelectionSummary" class="selection-summary">当前未选择节点或分支。</div>
        <details class="selection-details">
          <summary>调试信息</summary>
          <div>
            <strong>Current selection</strong>
            <pre id="currentSelectionDebug">{"node_ids":[],"edge_ids":[]}</pre>
          </div>
          <div>
            <strong>Last send selection</strong>
            <pre id="lastSendSelectionDebug">{"node_ids":[],"edge_ids":[]}</pre>
          </div>
          <div>
            <strong>Last send readable</strong>
            <div id="lastSendSelectionSummary" class="selection-summary compact">尚未发送选择。</div>
          </div>
        </details>
      </div>
      <form id="chatForm" class="composer">
        <textarea id="instruction" placeholder="直接描述你想怎么改流程，例如：发布结果必须来自人工复核的输出，不要让生成报告直接发布。"></textarea>
        <button class="primary" id="run" type="submit">发送</button>
      </form>
    </aside>
  </main>
`;

const instructionEl = document.querySelector<HTMLTextAreaElement>('#instruction')!;
const chatForm = document.querySelector<HTMLFormElement>('#chatForm')!;
const messagesEl = document.querySelector<HTMLDivElement>('#messages')!;
const graphEl = document.querySelector<HTMLDivElement>('#graph')!;
const sessionEl = document.querySelector<HTMLParagraphElement>('#session')!;
const versionBadgeEl = document.querySelector<HTMLSpanElement>('#versionBadge')!;
const clearSelectionEl = document.querySelector<HTMLButtonElement>('#clearSelection')!;
const currentSelectionSummaryEl = document.querySelector<HTMLDivElement>('#currentSelectionSummary')!;
const lastSendSelectionSummaryEl = document.querySelector<HTMLDivElement>('#lastSendSelectionSummary')!;
const currentSelectionDebugEl = document.querySelector<HTMLPreElement>('#currentSelectionDebug')!;
const lastSendSelectionDebugEl = document.querySelector<HTMLPreElement>('#lastSendSelectionDebug')!;
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
  plugins: [Dagre, ElkLayout],
  pluginsOptions: {
    dagre: {
      rankdir: 'LR',
      nodesep: 80,
      ranksep: 220,
      edgesep: 48,
      isDefaultAnchor: false
    },
    elkLayout: {
      rankdir: 'LR',
      nodesep: 80,
      ranksep: 220,
      edgesep: 48,
      isDefaultAnchor: false
    }
  },
  stopScrollGraph: true,
  stopZoomGraph: true
});

type SelectionSnapshot = { node_ids: string[]; edge_ids: string[] };
type DisplayEdgeInfo = {
  id: string;
  source: string;
  target: string;
  label: string;
  underlying_edge_ids: string[];
};
type EdgePoint = { x: number; y: number };
type AnchorSide = 'left' | 'right' | 'top' | 'bottom';
type GeometryEngine = 'native_elk' | 'dagre_legacy';
type CanonicalShapeKind = 'rect' | 'diamond' | 'ellipse';
type CanonicalShapeGeometry = {
  kind: CanonicalShapeKind;
  width: number;
  height: number;
  rx: number;
  ry: number;
  radius?: number;
};
type DisplayEdgeProjection = {
  id: string;
  type: 'polyline';
  sourceNodeId: string;
  targetNodeId: string;
  sourceAnchorId?: string;
  targetAnchorId?: string;
  startPoint?: EdgePoint;
  endPoint?: EdgePoint;
  pointsList?: EdgePoint[];
  text: string;
  properties: Record<string, unknown>;
};
type LogicFlowNodeProjection = {
  id: string;
  type: string;
  x: number;
  y: number;
  anchorsOffset?: Array<{ id: string; x: number; y: number }>;
  properties: Record<string, unknown>;
  width?: number;
  height?: number;
  text: string;
};
type LogicFlowGraphProjection = {
  nodes: LogicFlowNodeProjection[];
  edges: DisplayEdgeProjection[];
};
type ElkPortProjection = {
  id: string;
  x?: number;
  y?: number;
  width?: number;
  height?: number;
  layoutOptions?: Record<string, string>;
};
type ElkNodeProjection = {
  id: string;
  x?: number;
  y?: number;
  width: number;
  height: number;
  ports?: ElkPortProjection[];
  layoutOptions?: Record<string, string>;
};
type ElkEdgeSectionProjection = {
  id?: string;
  startPoint: EdgePoint;
  endPoint: EdgePoint;
  bendPoints?: EdgePoint[];
  incomingSections?: string[];
  outgoingSections?: string[];
};
type ElkEdgeProjection = {
  id: string;
  sources: string[];
  targets: string[];
  sourcePort?: string;
  targetPort?: string;
  sections?: ElkEdgeSectionProjection[];
};
type ElkGraphProjection = {
  id: string;
  layoutOptions: Record<string, string>;
  children: ElkNodeProjection[];
  edges: ElkEdgeProjection[];
};
type DisplayGeometryBBox = {
  left: number;
  right: number;
  top: number;
  bottom: number;
  width: number;
  height: number;
  centerX: number;
  centerY: number;
};

let frontendSelectionState: SelectionSnapshot = { node_ids: [], edge_ids: [] };
let displayEdgeLookup = new Map<string, DisplayEdgeInfo>();
const nativeElk = new ELK();
const DISPLAY_VIEWPORT_PADDING = 48;
const DISPLAY_BBOX_LABEL_PADDING = 64;
const READABLE_MIN_ZOOM = 0.42;
const READABLE_MAX_ZOOM = 1;

function emptySelection(): SelectionSnapshot {
  return { node_ids: [], edge_ids: [] };
}

function hasSelection(selection: SelectionSnapshot) {
  return selection.node_ids.length > 0 || selection.edge_ids.length > 0;
}

function uniqueIds(values: string[]) {
  return Array.from(new Set(values.filter(Boolean)));
}

function eventDataId(payload?: unknown) {
  const raw = payload as any;
  const id = raw?.data?.id ?? raw?.id;
  return typeof id === 'string' ? id : '';
}

function toggleIdList(ids: string[], id: string) {
  if (!id) return ids;
  if (ids.includes(id)) return ids.filter((item) => item !== id);
  return uniqueIds([...ids, id]);
}

function toggleEdgeIds(ids: string[], toggledIds: string[]) {
  const targets = uniqueIds(toggledIds);
  if (!targets.length) return ids;
  const current = new Set(ids);
  const allSelected = targets.every((id) => current.has(id));
  targets.forEach((id) => {
    if (allSelected) current.delete(id);
    else current.add(id);
  });
  return Array.from(current);
}

function displayEdgeIdsForEvent(id: string) {
  return displayEdgeLookup.get(id)?.underlying_edge_ids ?? [id];
}

function userVisibleEdgeLabel(edge: WorkflowEdge) {
  const branchLabel = typeof edge.branch_label === 'string' ? edge.branch_label.trim() : '';
  if (branchLabel) return branchLabel;
  const label = typeof edge.label === 'string' ? edge.label.trim() : '';
  if (label) return label;
  const condition = edge.condition;
  if (condition && typeof condition === 'object') {
    for (const key of ['label', 'summary', 'description', 'display_label']) {
      const value = condition[key];
      if (typeof value === 'string' && value.trim()) return value.trim();
    }
  }
  return '';
}

function combinedEdgeLabel(edges: WorkflowEdge[]) {
  const labels = edges.map(userVisibleEdgeLabel).filter(Boolean);
  return Array.from(new Set(labels)).join(' / ');
}

function displayEdgeId(source: string, target: string, edges: WorkflowEdge[], reservedIds: Set<string>, usedDisplayIds: Set<string>) {
  if (edges.length === 1) return edges[0].id;
  const sourceKey = encodeURIComponent(source);
  const targetKey = encodeURIComponent(target);
  const edgeKey = edges.map((edge) => encodeURIComponent(edge.id)).join('__');
  const base = `__afc_display_edge__${sourceKey}__${targetKey}__${edgeKey}`;
  let candidate = base;
  let suffix = 2;
  while (reservedIds.has(candidate) || usedDisplayIds.has(candidate)) {
    candidate = `${base}__${suffix}`;
    suffix += 1;
  }
  return candidate;
}

function groupEdgesByEndpoints(edges: WorkflowEdge[]) {
  const groups = new Map<string, WorkflowEdge[]>();
  edges.forEach((edge) => {
    const key = `${edge.source}\u0000${edge.target}`;
    const group = groups.get(key) ?? [];
    group.push(edge);
    groups.set(key, group);
  });
  return Array.from(groups.values());
}

function assertDisplayProjectionCoverage(businessEdges: WorkflowEdge[], displayEdges: Array<{ id: string; properties: Record<string, unknown> }>) {
  const businessEdgeIds = businessEdges.map((edge) => edge.id);
  const uniqueBusinessIds = new Set(businessEdgeIds);
  if (uniqueBusinessIds.size !== businessEdgeIds.length) {
    throw new Error('AI_DISPLAY_PROJECTION_DUPLICATE_BUSINESS_EDGE_ID');
  }
  const owners = new Map<string, string>();
  displayEdges.forEach((edge) => {
    const underlyingIds = edge.properties.underlying_edge_ids;
    if (!Array.isArray(underlyingIds) || underlyingIds.length === 0) {
      throw new Error(`AI_DISPLAY_PROJECTION_MISSING_UNDERLYING_EDGES:${edge.id}`);
    }
    underlyingIds.forEach((rawId) => {
      if (typeof rawId !== 'string' || !uniqueBusinessIds.has(rawId)) {
        throw new Error(`AI_DISPLAY_PROJECTION_UNKNOWN_UNDERLYING_EDGE:${edge.id}`);
      }
      const previousOwner = owners.get(rawId);
      if (previousOwner) {
        throw new Error(`AI_DISPLAY_PROJECTION_DUPLICATE_UNDERLYING_EDGE:${rawId}:${previousOwner}:${edge.id}`);
      }
      owners.set(rawId, edge.id);
    });
  });
  if (owners.size !== uniqueBusinessIds.size) {
    const missing = businessEdgeIds.filter((id) => !owners.has(id));
    throw new Error(`AI_DISPLAY_PROJECTION_MISSING_BUSINESS_EDGE:${missing.join(',')}`);
  }
}

function routePointsFromDisplayEdge(edge: any): EdgePoint[] {
  const rawPoints = Array.isArray(edge?.pointsList) ? edge.pointsList : [];
  const points = rawPoints
    .map((point: any) => ({ x: Number(point?.x), y: Number(point?.y) }))
    .filter((point: EdgePoint) => Number.isFinite(point.x) && Number.isFinite(point.y));
  if (points.length >= 2) return points;
  const start = edge?.startPoint;
  const end = edge?.endPoint;
  if (Number.isFinite(start?.x) && Number.isFinite(start?.y) && Number.isFinite(end?.x) && Number.isFinite(end?.y)) {
    return [
      { x: Number(start.x), y: Number(start.y) },
      { x: Number(end.x), y: Number(end.y) }
    ];
  }
  return [];
}

function routeLength(points: EdgePoint[]) {
  return points.slice(1).reduce((total, point, index) => {
    const previous = points[index];
    return total + Math.abs(point.x - previous.x) + Math.abs(point.y - previous.y);
  }, 0);
}

function collinearSegmentOverlap(a1: number, a2: number, b1: number, b2: number) {
  const minA = Math.min(a1, a2);
  const maxA = Math.max(a1, a2);
  const minB = Math.min(b1, b2);
  const maxB = Math.max(b1, b2);
  return Math.max(0, Math.min(maxA, maxB) - Math.max(minA, minB));
}

function independentRouteOverlap(a: EdgePoint[], b: EdgePoint[]) {
  let overlapLength = 0;
  for (let ai = 1; ai < a.length; ai += 1) {
    const a0 = a[ai - 1];
    const a1 = a[ai];
    for (let bi = 1; bi < b.length; bi += 1) {
      const b0 = b[bi - 1];
      const b1 = b[bi];
      const aHorizontal = a0.y === a1.y;
      const bHorizontal = b0.y === b1.y;
      const aVertical = a0.x === a1.x;
      const bVertical = b0.x === b1.x;
      if (aHorizontal && bHorizontal && a0.y === b0.y) {
        overlapLength += collinearSegmentOverlap(a0.x, a1.x, b0.x, b1.x);
      } else if (aVertical && bVertical && a0.x === b0.x) {
        overlapLength += collinearSegmentOverlap(a0.y, a1.y, b0.y, b1.y);
      }
    }
  }
  const shortest = Math.min(routeLength(a), routeLength(b));
  return {
    overlapLength,
    overlapRatio: shortest > 0 ? overlapLength / shortest : 0
  };
}

function anchorId(nodeId: string, side: AnchorSide, index: number) {
  return `${nodeId}:${side}:${index}`;
}

function oppositeSide(side: AnchorSide): AnchorSide {
  if (side === 'left') return 'right';
  if (side === 'right') return 'left';
  if (side === 'top') return 'bottom';
  return 'top';
}

function canonicalShapeKind(node: WorkflowNode | LogicFlowNodeProjection): CanonicalShapeKind {
  const workflowType =
    typeof (node as LogicFlowNodeProjection).properties?.workflowType === 'string'
      ? String((node as LogicFlowNodeProjection).properties.workflowType)
      : String((node as WorkflowNode).type ?? '');
  if (workflowType === 'decision') return 'diamond';
  if (workflowType === 'start' || workflowType === 'end') return 'ellipse';
  return 'rect';
}

function numberFrom(value: unknown, fallback: number) {
  const parsed = Number(value);
  return Number.isFinite(parsed) && parsed > 0 ? parsed : fallback;
}

function canonicalShapeGeometry(node: WorkflowNode | LogicFlowNodeProjection): CanonicalShapeGeometry {
  const kind = canonicalShapeKind(node);
  const properties = (node as LogicFlowNodeProjection).properties ?? {};
  const workflowType = typeof properties.workflowType === 'string' ? properties.workflowType : (node as WorkflowNode).type;
  const defaultWidth = workflowType === 'start' || workflowType === 'end' ? 96 : workflowType === 'decision' ? 104 : 140;
  const defaultHeight = workflowType === 'start' || workflowType === 'end' ? 96 : workflowType === 'decision' ? 104 : 72;
  const width = numberFrom((node as WorkflowNode).width ?? properties.width, defaultWidth);
  const height = numberFrom((node as WorkflowNode).height ?? properties.height, defaultHeight);
  if (kind === 'diamond') {
    return { kind, width, height, rx: width / 2, ry: height / 2, radius: 0 };
  }
  if (kind === 'ellipse') {
    return { kind, width, height, rx: width / 2, ry: height / 2 };
  }
  return { kind, width, height, rx: width / 2, ry: height / 2 };
}

function sideForRelativeGeometry(source: WorkflowNode, target: WorkflowNode): AnchorSide {
  const dx = target.x - source.x;
  const dy = target.y - source.y;
  if (Math.abs(dy) > Math.abs(dx) * 1.15) {
    return dy >= 0 ? 'bottom' : 'top';
  }
  return dx >= 0 ? 'right' : 'left';
}

function oppositeNodeSortValue(side: AnchorSide, node: WorkflowNode) {
  return side === 'left' || side === 'right' ? node.y : node.x;
}

function displayEdgeSortKey(edge: DisplayEdgeProjection) {
  return `${edge.sourceNodeId}\u0000${edge.targetNodeId}\u0000${edge.id}`;
}

function distributedPortCoordinate(side: AnchorSide, index: number, count: number, geometry: CanonicalShapeGeometry) {
  const { width, height } = geometry;
  const span = side === 'left' || side === 'right' ? height : width;
  const usable = Math.max(0, span - 20);
  return count <= 1 ? 0 : -usable / 2 + (usable * index) / (count - 1);
}

function shapeAwareBoundaryPoint(side: AnchorSide, index: number, count: number, geometry: CanonicalShapeGeometry) {
  const { width, height, rx, ry, kind } = geometry;
  const stable = distributedPortCoordinate(side, index, count, geometry);
  if (kind === 'rect') {
    if (side === 'left') return { x: -width / 2, y: stable };
    if (side === 'right') return { x: width / 2, y: stable };
    if (side === 'top') return { x: stable, y: -height / 2 };
    return { x: stable, y: height / 2 };
  }
  if (kind === 'diamond') {
    if (side === 'left' || side === 'right') {
      const y = Math.max(-ry, Math.min(ry, stable));
      const x = rx * Math.max(0, 1 - Math.abs(y) / ry);
      return { x: side === 'right' ? x : -x, y };
    }
    const x = Math.max(-rx, Math.min(rx, stable));
    const y = ry * Math.max(0, 1 - Math.abs(x) / rx);
    return { x, y: side === 'bottom' ? y : -y };
  }
  if (side === 'left' || side === 'right') {
    const y = Math.max(-ry, Math.min(ry, stable));
    const x = rx * Math.sqrt(Math.max(0, 1 - (y * y) / (ry * ry)));
    return { x: side === 'right' ? x : -x, y };
  }
  const x = Math.max(-rx, Math.min(rx, stable));
  const y = ry * Math.sqrt(Math.max(0, 1 - (x * x) / (rx * rx)));
  return { x, y: side === 'bottom' ? y : -y };
}

function assignStablePorts(nodes: WorkflowNode[], displayEdges: DisplayEdgeProjection[]) {
  const nodeById = new Map(nodes.map((node) => [node.id, node]));
  const demands = new Map<string, Map<AnchorSide, Array<{ edge: DisplayEdgeProjection; opposite: WorkflowNode }>>>();
  const ensure = (nodeId: string, side: AnchorSide) => {
    let nodeDemands = demands.get(nodeId);
    if (!nodeDemands) {
      nodeDemands = new Map();
      demands.set(nodeId, nodeDemands);
    }
    const sideDemands = nodeDemands.get(side) ?? [];
    nodeDemands.set(side, sideDemands);
    return sideDemands;
  };
  displayEdges.forEach((edge) => {
    const source = nodeById.get(edge.sourceNodeId);
    const target = nodeById.get(edge.targetNodeId);
    if (!source || !target) return;
    const sourceSide = sideForRelativeGeometry(source, target);
    const targetSide = oppositeSide(sourceSide);
    ensure(source.id, sourceSide).push({ edge, opposite: target });
    ensure(target.id, targetSide).push({ edge, opposite: source });
  });
  const anchorOffsetsByNode = new Map<string, Array<{ id: string; x: number; y: number }>>();
  demands.forEach((nodeDemands, nodeId) => {
    const node = nodeById.get(nodeId);
    if (!node) return;
    const geometry = canonicalShapeGeometry(node);
    const offsets: Array<{ id: string; x: number; y: number }> = [];
    nodeDemands.forEach((items, side) => {
      const sorted = [...items].sort((a, b) => {
        const byGeometry = oppositeNodeSortValue(side, a.opposite) - oppositeNodeSortValue(side, b.opposite);
        return byGeometry || displayEdgeSortKey(a.edge).localeCompare(displayEdgeSortKey(b.edge));
      });
      sorted.forEach((item, index) => {
        const id = anchorId(nodeId, side, index);
        const offset = shapeAwareBoundaryPoint(side, index, sorted.length, geometry);
        offsets.push({ id, ...offset });
        if (item.edge.sourceNodeId === nodeId) {
          item.edge.sourceAnchorId = id;
        } else {
          item.edge.targetAnchorId = id;
        }
      });
    });
    anchorOffsetsByNode.set(nodeId, offsets);
  });
  return anchorOffsetsByNode;
}

function elkSide(side: AnchorSide) {
  if (side === 'left') return 'WEST';
  if (side === 'right') return 'EAST';
  if (side === 'top') return 'NORTH';
  return 'SOUTH';
}

function sideFromAnchorId(anchor: string | undefined): AnchorSide | null {
  const side = anchor?.split(':')[1];
  return side === 'left' || side === 'right' || side === 'top' || side === 'bottom' ? side : null;
}

function nativeElkPortFromAnchor(anchor: { id: string; x: number; y: number }, node: LogicFlowNodeProjection): ElkPortProjection {
  const portSize = 1;
  const side = sideFromAnchorId(anchor.id);
  const geometry = canonicalShapeGeometry(node);
  const centerX = geometry.width / 2 + anchor.x;
  const centerY = geometry.height / 2 + anchor.y;
  return {
    id: anchor.id,
    width: portSize,
    height: portSize,
    ...(side
      ? { layoutOptions: { 'elk.port.side': elkSide(side) } }
      : {
          x: centerX - portSize / 2,
          y: centerY - portSize / 2
        })
  };
}

function nativeElkOptions(): Record<string, string> {
  return {
    'elk.algorithm': 'layered',
    'elk.direction': 'RIGHT',
    'elk.edgeRouting': 'ORTHOGONAL',
    'elk.separateConnectedComponents': 'true',
    'elk.spacing.nodeNode': '80',
    'elk.spacing.edgeEdge': '48',
    'elk.spacing.edgeNode': '36',
    'elk.spacing.componentComponent': '120',
    'elk.layered.spacing.nodeNodeBetweenLayers': '120',
    'elk.layered.spacing.edgeNodeBetweenLayers': '48',
    'elk.layered.spacing.edgeEdgeBetweenLayers': '48',
    'elk.layered.crossingMinimization.strategy': 'LAYER_SWEEP',
    'elk.layered.nodePlacement.strategy': 'NETWORK_SIMPLEX',
    'elk.layered.nodePlacement.favorStraightEdges': 'true',
    'elk.layered.cycleBreaking.strategy': 'GREEDY'
  };
}

function buildNativeElkGraph(graphData: LogicFlowGraphProjection): ElkGraphProjection {
  const children = graphData.nodes.map((node) => {
    const geometry = canonicalShapeGeometry(node);
    const { width, height } = geometry;
    const anchors = Array.isArray(node.anchorsOffset) ? node.anchorsOffset : [];
    return {
      id: node.id,
      width,
      height,
      ports: anchors.map((anchor) => nativeElkPortFromAnchor(anchor, { ...node, width, height })),
      layoutOptions: {
        'elk.portConstraints': 'FIXED_SIDE'
      }
    };
  });
  const edges = graphData.edges.map((edge) => ({
    id: edge.id,
    sources: [edge.sourceAnchorId ?? edge.sourceNodeId],
    targets: [edge.targetAnchorId ?? edge.targetNodeId],
    ...(edge.sourceAnchorId ? { sourcePort: edge.sourceAnchorId } : {}),
    ...(edge.targetAnchorId ? { targetPort: edge.targetAnchorId } : {})
  }));
  return {
    id: 'ai-flow-canvas-display-graph',
    layoutOptions: nativeElkOptions(),
    children,
    edges
  };
}

function pointKey(point: EdgePoint) {
  return `${Math.round(point.x * 1000) / 1000}:${Math.round(point.y * 1000) / 1000}`;
}

function appendUniquePoint(points: EdgePoint[], point: EdgePoint) {
  if (!Number.isFinite(point.x) || !Number.isFinite(point.y)) return;
  const previous = points[points.length - 1];
  if (previous && pointKey(previous) === pointKey(point)) return;
  points.push({ x: point.x, y: point.y });
}

function sectionPoints(section: ElkEdgeSectionProjection) {
  return [section.startPoint, ...(section.bendPoints ?? []), section.endPoint];
}

function flattenElkSections(edgeId: string, sections: ElkEdgeSectionProjection[] | undefined) {
  if (!Array.isArray(sections) || sections.length === 0) {
    throw new Error(`AI_NATIVE_ELK_EDGE_WITHOUT_SECTION:${edgeId}`);
  }
  if (sections.length === 1) return sectionPoints(sections[0]);

  const byId = new Map(sections.filter((section) => section.id).map((section) => [section.id as string, section]));
  const incomingIds = new Set<string>();
  sections.forEach((section) => (section.incomingSections ?? []).forEach((id) => incomingIds.add(id)));
  const start = sections.find((section) => section.id && !incomingIds.has(section.id)) ?? sections[0];
  const ordered: ElkEdgeSectionProjection[] = [];
  const seen = new Set<string>();
  const visit = (section: ElkEdgeSectionProjection) => {
    const id = section.id ?? `${ordered.length}`;
    if (seen.has(id)) return;
    seen.add(id);
    ordered.push(section);
    (section.outgoingSections ?? []).forEach((nextId) => {
      const next = byId.get(nextId);
      if (next) visit(next);
    });
  };
  visit(start);
  sections.forEach((section) => {
    const id = section.id ?? '';
    if (!id || !seen.has(id)) ordered.push(section);
  });
  console.warn('AI_NATIVE_ELK_MULTI_SECTION_EDGE_FLATTENED', { edgeId, sectionCount: sections.length });
  const points: EdgePoint[] = [];
  ordered.forEach((section) => sectionPoints(section).forEach((point) => appendUniquePoint(points, point)));
  return points;
}

function anchorsOffsetFromElkPorts(node: ElkNodeProjection) {
  const width = Number(node.width) || 140;
  const height = Number(node.height) || 72;
  return (node.ports ?? [])
    .filter((port) => typeof port.id === 'string' && Number.isFinite(port.x) && Number.isFinite(port.y))
    .map((port) => {
      const portWidth = Number(port.width ?? 0);
      const portHeight = Number(port.height ?? 0);
      const endpointX = Number(port.x) + portWidth / 2;
      const endpointY = Number(port.y) + portHeight / 2;
      return {
        id: port.id,
        x: endpointX - width / 2,
        y: endpointY - height / 2
      };
    });
}

function shapeBoundaryHorizontalX(geometry: CanonicalShapeGeometry, y: number) {
  const { width, rx, ry, kind } = geometry;
  const safeY = Math.max(-ry, Math.min(ry, y));
  if (kind === 'diamond') return rx * Math.max(0, 1 - Math.abs(safeY) / ry);
  if (kind === 'ellipse') return rx * Math.sqrt(Math.max(0, 1 - (safeY * safeY) / (ry * ry)));
  return width / 2;
}

function shapeBoundaryVerticalY(geometry: CanonicalShapeGeometry, x: number) {
  const { height, rx, ry, kind } = geometry;
  const safeX = Math.max(-rx, Math.min(rx, x));
  if (kind === 'diamond') return ry * Math.max(0, 1 - Math.abs(safeX) / rx);
  if (kind === 'ellipse') return ry * Math.sqrt(Math.max(0, 1 - (safeX * safeX) / (rx * rx)));
  return height / 2;
}

function projectEndpointToShapeBoundary(
  node: LogicFlowNodeProjection | undefined,
  endpoint: EdgePoint | undefined,
  adjacent: EdgePoint | undefined
) {
  if (!node || !endpoint || !adjacent) return null;
  const nodeX = Number(node.x);
  const nodeY = Number(node.y);
  if (![nodeX, nodeY, endpoint.x, endpoint.y, adjacent.x, adjacent.y].every(Number.isFinite)) return null;
  const geometry = canonicalShapeGeometry(node);
  const dx = endpoint.x - adjacent.x;
  const dy = endpoint.y - adjacent.y;
  if (Math.abs(dx) >= Math.abs(dy)) {
    const localY = Math.max(-geometry.ry, Math.min(geometry.ry, endpoint.y - nodeY));
    const side = endpoint.x >= nodeX ? 1 : -1;
    return {
      x: nodeX + side * shapeBoundaryHorizontalX(geometry, localY),
      y: nodeY + localY
    };
  }
  const localX = Math.max(-geometry.rx, Math.min(geometry.rx, endpoint.x - nodeX));
  const side = endpoint.y >= nodeY ? 1 : -1;
  return {
    x: nodeX + localX,
    y: nodeY + side * shapeBoundaryVerticalY(geometry, localX)
  };
}

function clampRouteEndpointsToCanonicalAnchors(
  edge: DisplayEdgeProjection,
  pointsList: EdgePoint[],
  laidOutNodesById: Map<string, LogicFlowNodeProjection>
) {
  if (pointsList.length < 2) return pointsList;
  const sourcePoint = projectEndpointToShapeBoundary(laidOutNodesById.get(edge.sourceNodeId), pointsList[0], pointsList[1]);
  const targetPoint = projectEndpointToShapeBoundary(
    laidOutNodesById.get(edge.targetNodeId),
    pointsList[pointsList.length - 1],
    pointsList[pointsList.length - 2]
  );
  return pointsList.map((point, index) => {
    if (index === 0 && sourcePoint) return sourcePoint;
    if (index === pointsList.length - 1 && targetPoint) return targetPoint;
    return point;
  });
}

function applyNativeElkLayoutToGraphData(graphData: LogicFlowGraphProjection, elkGraph: ElkGraphProjection): LogicFlowGraphProjection {
  const nodesById = new Map((elkGraph.children ?? []).map((node) => [node.id, node]));
  const edgesById = new Map((elkGraph.edges ?? []).map((edge) => [edge.id, edge]));
  const laidOutNodes = graphData.nodes.map((node) => {
      const elkNode = nodesById.get(node.id);
      if (!elkNode || !Number.isFinite(elkNode.x) || !Number.isFinite(elkNode.y)) return node;
      const width = Number(elkNode.width) || Number(node.width ?? node.properties?.width) || 140;
      const height = Number(elkNode.height) || Number(node.height ?? node.properties?.height) || 72;
      const canonicalAnchors = Array.isArray(node.anchorsOffset) ? node.anchorsOffset : anchorsOffsetFromElkPorts(elkNode);
      return {
        ...node,
        x: Number(elkNode.x) + width / 2,
        y: Number(elkNode.y) + height / 2,
        width,
        height,
        anchorsOffset: canonicalAnchors,
        properties: {
          ...node.properties,
          width,
          height
        }
      };
    });
  const laidOutNodesById = new Map(laidOutNodes.map((node) => [node.id, node]));
  return {
    nodes: laidOutNodes,
    edges: graphData.edges.map((edge) => {
      const elkEdge = edgesById.get(edge.id);
      const pointsList = clampRouteEndpointsToCanonicalAnchors(
        edge,
        flattenElkSections(edge.id, elkEdge?.sections),
        laidOutNodesById
      );
      return {
        ...edge,
        startPoint: pointsList[0],
        endPoint: pointsList[pointsList.length - 1],
        pointsList
      };
    })
  };
}

function validateGeometryContract() {
  const data = lf.getGraphData() as any;
  const edges = Array.isArray(data?.edges) ? data.edges : [];
  edges.forEach((edge: any) => {
    const sourceModel = typeof edge?.sourceNodeId === 'string' ? lf.getNodeModelById(edge.sourceNodeId) as any : undefined;
    const targetModel = typeof edge?.targetNodeId === 'string' ? lf.getNodeModelById(edge.targetNodeId) as any : undefined;
    const sourceAnchor = sourceModel?.getAnchorInfo?.(edge?.sourceAnchorId);
    const targetAnchor = targetModel?.getAnchorInfo?.(edge?.targetAnchorId);
    const start = edge?.startPoint;
    const end = edge?.endPoint;
    if (
      sourceAnchor &&
      start &&
      (Math.abs(Number(start.x) - Number(sourceAnchor.x)) > 0.1 || Math.abs(Number(start.y) - Number(sourceAnchor.y)) > 0.1)
    ) {
      console.warn('AI_GEOMETRY_SOURCE_ANCHOR_MISMATCH', { edgeId: edge.id, sourceAnchor, start });
    }
    if (
      targetAnchor &&
      end &&
      (Math.abs(Number(end.x) - Number(targetAnchor.x)) > 0.1 || Math.abs(Number(end.y) - Number(targetAnchor.y)) > 0.1)
    ) {
      console.warn('AI_GEOMETRY_TARGET_ANCHOR_MISMATCH', { edgeId: edge.id, targetAnchor, end });
    }
  });
  for (let index = 0; index < edges.length; index += 1) {
    for (let other = index + 1; other < edges.length; other += 1) {
      const edge = edges[index];
      const otherEdge = edges[other];
      if (edge?.id === otherEdge?.id) continue;
      const overlap = independentRouteOverlap(routePointsFromDisplayEdge(edge), routePointsFromDisplayEdge(otherEdge));
      if (overlap.overlapLength > 48 && overlap.overlapRatio > 0.24) {
        console.warn('AI_GEOMETRY_INDEPENDENT_EDGE_OVERLAP', { edgeId: edge.id, otherEdgeId: otherEdge.id, ...overlap });
      }
    }
  }
}

function applySelectionHighlight() {
  const selectedNodeIds = new Set(frontendSelectionState.node_ids);
  const selectedEdgeIds = new Set(frontendSelectionState.edge_ids);
  try {
    const data = lf.getGraphData() as any;
    const nodes = Array.isArray(data?.nodes) ? data.nodes : [];
    const edges = Array.isArray(data?.edges) ? data.edges : [];
    nodes.forEach((node: any) => {
      if (typeof node?.id !== 'string') return;
      const model = lf.getNodeModelById(node.id) as any;
      if (!model?.updateStyles) return;
      model.updateStyles(
        selectedNodeIds.has(node.id)
          ? {
              stroke: '#2563eb',
              strokeWidth: 3,
              fill: '#eff6ff'
            }
          : {}
      );
    });
    edges.forEach((edge: any) => {
      if (typeof edge?.id !== 'string') return;
      const underlying = displayEdgeLookup.get(edge.id)?.underlying_edge_ids ?? [edge.id];
      const isSelected = underlying.some((edgeId) => selectedEdgeIds.has(edgeId));
      const model = (lf as any).getEdgeModelById?.(edge.id);
      if (!model?.updateStyles) return;
      model.updateStyles(
        isSelected
          ? {
              stroke: '#2563eb',
              strokeWidth: 3,
              strokeDasharray: '8 4'
            }
          : {}
      );
    });
  } catch (error) {
    console.warn('AI_SELECTION_HIGHLIGHT_FAILED', error);
  }
}

function setFrontendSelectionFromEvent(kind: 'node' | 'edge' | 'blank', payload?: unknown) {
  if (kind === 'blank') {
    return;
  }
  const id = eventDataId(payload);
  if (!id) return;
  if (kind === 'node') {
    frontendSelectionState = {
      node_ids: toggleIdList(frontendSelectionState.node_ids, id),
      edge_ids: frontendSelectionState.edge_ids
    };
    applySelectionHighlight();
    return;
  }
  frontendSelectionState = {
    node_ids: frontendSelectionState.node_ids,
    edge_ids: toggleEdgeIds(frontendSelectionState.edge_ids, displayEdgeIdsForEvent(id))
  };
  applySelectionHighlight();
}

function workflowToLogicFlow(graph: WorkflowGraph): LogicFlowGraphProjection {
  const nextDisplayEdges = new Map<string, DisplayEdgeInfo>();
  const businessEdgeIds = new Set(graph.edges.map((edge) => edge.id));
  const usedDisplayIds = new Set<string>();
  const displayEdges: DisplayEdgeProjection[] = groupEdgesByEndpoints(graph.edges).map((edges) => {
    const source = edges[0].source;
    const target = edges[0].target;
    const id = displayEdgeId(source, target, edges, businessEdgeIds, usedDisplayIds);
    usedDisplayIds.add(id);
    const label = combinedEdgeLabel(edges);
    const underlying_edge_ids = edges.map((edge) => edge.id);
    nextDisplayEdges.set(id, {
      id,
      source,
      target,
      label,
      underlying_edge_ids
    });
    return {
      id,
      type: 'polyline',
      sourceNodeId: source,
      targetNodeId: target,
      sourceAnchorId: undefined,
      targetAnchorId: undefined,
      text: label,
      properties: {
        label,
        displayEdge: edges.length > 1,
        underlying_edge_ids,
        underlying_edges: edges.map((edge, index) => ({
          id: edge.id,
          label: userVisibleEdgeLabel(edge),
          branch_label: edge.branch_label ?? '',
          source: edge.source,
          target: edge.target
        }))
      }
    };
  });
  const anchorOffsetsByNode = assignStablePorts(graph.nodes, displayEdges);
  assertDisplayProjectionCoverage(graph.edges, displayEdges);
  displayEdgeLookup = nextDisplayEdges;
  return {
    nodes: graph.nodes.map((node) => {
      const geometry = canonicalShapeGeometry(node);
      const shapeProperties =
        geometry.kind === 'diamond'
          ? { rx: geometry.rx, ry: geometry.ry, style: { radius: geometry.radius ?? 0 } }
          : geometry.kind === 'ellipse' && geometry.rx === geometry.ry
            ? { r: geometry.rx }
            : {};
      return {
        id: node.id,
        type: nodeTypeMap[node.type] ?? 'rect',
        x: node.x,
        y: node.y,
        anchorsOffset: anchorOffsetsByNode.get(node.id) ?? [],
        properties: {
          workflowType: node.type,
          typeLabel: typeLabel[node.type] ?? node.type,
          join: node.join,
          width: geometry.width,
          height: geometry.height,
          shape: geometry.kind,
          ...shapeProperties
        },
        width: geometry.width,
        height: geometry.height,
        text: node.label
      };
    }),
    edges: displayEdges
  };
}

function readLogicFlowSelection(): SelectionSnapshot {
  const selected = lf.getSelectElements(false) as any;
  return {
    node_ids: Array.isArray(selected?.nodes) ? selected.nodes.map((node: any) => node.id).filter(Boolean) : [],
    edge_ids: Array.isArray(selected?.edges) ? selected.edges.map((edge: any) => edge.id).filter(Boolean) : []
  };
}

function writeSelectionDebug(current: SelectionSnapshot, lastSend?: SelectionSnapshot) {
  writeSelectionSummary(currentSelectionSummaryEl, current);
  currentSelectionDebugEl.textContent = pretty(current);
  if (lastSend) {
    writeSelectionSummary(lastSendSelectionSummaryEl, lastSend);
    lastSendSelectionDebugEl.textContent = pretty(lastSend);
  }
}

function nodeLabel(nodeId: string) {
  return workflowGraph.nodes.find((node) => node.id === nodeId)?.label ?? nodeId;
}

function edgeSummary(edgeId: string) {
  const edge = workflowGraph.edges.find((item) => item.id === edgeId);
  if (!edge) return edgeId;
  const source = nodeLabel(edge.source);
  const target = nodeLabel(edge.target);
  const label = edge.label?.trim();
  return `${label ? `${label}: ` : ''}${source} -> ${target}`;
}

function summarizeSelectedEdges(edgeIds: string[]) {
  const selected = new Set(edgeIds);
  return groupEdgesByEndpoints(workflowGraph.edges.filter((edge) => selected.has(edge.id))).map((edges) => {
    const source = nodeLabel(edges[0].source);
    const target = nodeLabel(edges[0].target);
    const label = combinedEdgeLabel(edges);
    return `${label ? `${label}: ` : ''}${source} -> ${target}`;
  });
}

function appendSelectionGroup(container: HTMLElement, title: string, items: string[]) {
  if (!items.length) return;
  const heading = document.createElement('div');
  heading.className = 'selection-group-title';
  heading.textContent = title;
  container.append(heading);
  const list = document.createElement('ul');
  list.className = 'selection-list';
  items.forEach((item) => {
    const li = document.createElement('li');
    li.textContent = item;
    list.append(li);
  });
  container.append(list);
}

function writeSelectionSummary(container: HTMLElement, selection: SelectionSnapshot) {
  container.textContent = '';
  if (!hasSelection(selection)) {
    container.textContent = '当前未选择节点或分支。';
    return;
  }
  appendSelectionGroup(container, '节点', selection.node_ids.map(nodeLabel));
  appendSelectionGroup(container, '分支', summarizeSelectedEdges(selection.edge_ids));
}

function getSelection(): SelectionSnapshot {
  if (hasSelection(frontendSelectionState)) {
    return frontendSelectionState;
  }
  return quickSelectionOverride ?? frontendSelectionState;
}

type LayoutReason = 'initial' | 'server-state' | 'reset' | 'manual' | 'session-create';
const GEOMETRY_ENGINE_STORAGE_KEY = 'aiFlowCanvas.geometryEngine';
const DEFAULT_GEOMETRY_ENGINE: GeometryEngine = 'native_elk';

const officialLayoutOptions = {
  rankdir: 'LR' as const,
  nodesep: 160,
  ranksep: 80,
  edgesep: 64,
  isDefaultAnchor: false
};

let layoutRunId = 0;
let pendingOfficialLayout: { id: number; reason: LayoutReason } | null = null;

function readGeometryEngine(): GeometryEngine {
  try {
    const fromStorage = window.localStorage.getItem(GEOMETRY_ENGINE_STORAGE_KEY);
    if (fromStorage === 'native_elk' || fromStorage === 'dagre_legacy') return fromStorage;
  } catch {
    // Ignore storage access failures and use the production default.
  }
  return DEFAULT_GEOMETRY_ENGINE;
}

function mergeDisplayNodePositionsIntoWorkflow() {
  const data = lf.getGraphData() as any;
  const displayNodes = new Map<string, any>(
    (Array.isArray(data?.nodes) ? data.nodes : [])
      .filter((node: any) => typeof node?.id === 'string')
      .map((node: any) => [node.id, node])
  );
  workflowGraph = {
    ...workflowGraph,
    nodes: workflowGraph.nodes.map((node) => {
      const displayNode = displayNodes.get(node.id);
      if (!displayNode || typeof displayNode.x !== 'number' || typeof displayNode.y !== 'number') {
        return node;
      }
      return {
        ...node,
        x: displayNode.x,
        y: displayNode.y,
        ...(displayNode.properties?.width ? { width: displayNode.properties.width } : {}),
        ...(displayNode.properties?.height ? { height: displayNode.properties.height } : {})
      };
    }),
    edges: workflowGraph.edges.map((edge) => ({ ...edge }))
  };
}

function syncDisplayGraphAfterLayout(reason: LayoutReason, runId: number) {
  if (runId !== layoutRunId) return;
  mergeDisplayNodePositionsIntoWorkflow();
  validateGeometryContract();
  applySelectionHighlight();
  fitReadableView();
  updateSelection(`official-layout:${reason}:synced`);
}

function completeOfficialLayout(runId: number, reason: LayoutReason) {
  if (!pendingOfficialLayout || pendingOfficialLayout.id !== runId) return;
  pendingOfficialLayout = null;
  window.requestAnimationFrame(() => syncDisplayGraphAfterLayout(reason, runId));
}

function expandBBoxWithPoint(bounds: { left: number; right: number; top: number; bottom: number }, point: EdgePoint) {
  if (!Number.isFinite(point.x) || !Number.isFinite(point.y)) return;
  bounds.left = Math.min(bounds.left, point.x);
  bounds.right = Math.max(bounds.right, point.x);
  bounds.top = Math.min(bounds.top, point.y);
  bounds.bottom = Math.max(bounds.bottom, point.y);
}

function bboxFromBounds(bounds: { left: number; right: number; top: number; bottom: number }): DisplayGeometryBBox | null {
  if (![bounds.left, bounds.right, bounds.top, bounds.bottom].every(Number.isFinite)) return null;
  const width = bounds.right - bounds.left;
  const height = bounds.bottom - bounds.top;
  if (width <= 0 || height <= 0) return null;
  return {
    ...bounds,
    width,
    height,
    centerX: bounds.left + width / 2,
    centerY: bounds.top + height / 2
  };
}

function computeDisplayGeometryBBox(): DisplayGeometryBBox | null {
  const graphData = lf.getGraphData() as any;
  const bounds = {
    left: Number.POSITIVE_INFINITY,
    right: Number.NEGATIVE_INFINITY,
    top: Number.POSITIVE_INFINITY,
    bottom: Number.NEGATIVE_INFINITY
  };
  (Array.isArray(graphData?.nodes) ? graphData.nodes : []).forEach((node: LogicFlowNodeProjection) => {
    if (!Number.isFinite(node?.x) || !Number.isFinite(node?.y)) return;
    const geometry = canonicalShapeGeometry(node);
    expandBBoxWithPoint(bounds, { x: node.x - geometry.width / 2, y: node.y - geometry.height / 2 });
    expandBBoxWithPoint(bounds, { x: node.x + geometry.width / 2, y: node.y + geometry.height / 2 });
  });
  (Array.isArray(graphData?.edges) ? graphData.edges : []).forEach((edge: DisplayEdgeProjection) => {
    routePointsFromDisplayEdge(edge).forEach((point) => expandBBoxWithPoint(bounds, point));
  });
  const bbox = bboxFromBounds(bounds);
  if (!bbox) return null;
  return {
    left: bbox.left - DISPLAY_BBOX_LABEL_PADDING,
    right: bbox.right + DISPLAY_BBOX_LABEL_PADDING,
    top: bbox.top - DISPLAY_BBOX_LABEL_PADDING,
    bottom: bbox.bottom + DISPLAY_BBOX_LABEL_PADDING,
    width: bbox.width + DISPLAY_BBOX_LABEL_PADDING * 2,
    height: bbox.height + DISPLAY_BBOX_LABEL_PADDING * 2,
    centerX: bbox.centerX,
    centerY: bbox.centerY
  };
}

function clampZoom(zoom: number) {
  if (!Number.isFinite(zoom) || zoom <= 0) return READABLE_MIN_ZOOM;
  return Math.max(READABLE_MIN_ZOOM, Math.min(READABLE_MAX_ZOOM, zoom));
}

function fitReadableView() {
  const bbox = computeDisplayGeometryBBox();
  const canvasWidth = graphEl.clientWidth;
  const canvasHeight = graphEl.clientHeight;
  if (!bbox || canvasWidth <= 0 || canvasHeight <= 0) {
    lf.fitView(DISPLAY_VIEWPORT_PADDING, DISPLAY_VIEWPORT_PADDING);
    return;
  }
  const availableWidth = Math.max(1, canvasWidth - DISPLAY_VIEWPORT_PADDING * 2);
  const availableHeight = Math.max(1, canvasHeight - DISPLAY_VIEWPORT_PADDING * 2);
  const fitZoom = Math.min(availableWidth / bbox.width, availableHeight / bbox.height);
  const nextZoom = clampZoom(fitZoom);
  lf.zoom(nextZoom, [canvasWidth / 2, canvasHeight / 2]);
  lf.focusOn({ x: bbox.centerX, y: bbox.centerY });
}

const rawRenderRawData = (lf as any).renderRawData?.bind(lf);
if (rawRenderRawData) {
  (lf as any).renderRawData = (graphData: unknown) => {
    const result = rawRenderRawData(graphData);
    const pending = pendingOfficialLayout;
    if (pending) {
      completeOfficialLayout(pending.id, pending.reason);
    }
    return result;
  };
}

function runLegacyOfficialLayout(reason: LayoutReason, runId: number) {
  const extensions = (lf as any).extension ?? {};
  pendingOfficialLayout = { id: runId, reason };
  if (extensions.dagre?.layout) {
    extensions.dagre.layout(officialLayoutOptions);
    return true;
  }
  if (extensions.elkLayout?.layout) {
    extensions.elkLayout.layout(officialLayoutOptions);
    return true;
  }
  return false;
}

async function runNativeElkLayout(reason: LayoutReason, runId: number, graphData: LogicFlowGraphProjection) {
  try {
    const elkInput = buildNativeElkGraph(graphData);
    const elkOutput = (await nativeElk.layout(elkInput as any)) as ElkGraphProjection;
    if (runId !== layoutRunId) return;
    const nextGraphData = applyNativeElkLayoutToGraphData(graphData, elkOutput);
    pendingOfficialLayout = { id: runId, reason };
    const renderRawData = (lf as any).renderRawData?.bind(lf);
    renderRawData?.(nextGraphData);
    if (!renderRawData) completeOfficialLayout(runId, reason);
  } catch (error) {
    if (runId !== layoutRunId) return;
    pendingOfficialLayout = null;
    console.warn('AI_NATIVE_ELK_LAYOUT_FAILED', { reason, error });
    const fallbackRunId = ++layoutRunId;
    try {
      if (!runLegacyOfficialLayout(reason, fallbackRunId)) {
        pendingOfficialLayout = null;
        console.warn('AI_FLOW_LAYOUT_UNAVAILABLE', { reason });
        fitReadableView();
      }
    } catch (fallbackError) {
      pendingOfficialLayout = null;
      console.warn('AI_FLOW_LAYOUT_FAILED', { reason, error: fallbackError });
      fitReadableView();
    }
  }
}

function runOfficialLayout(reason: LayoutReason, graphData?: LogicFlowGraphProjection) {
  const runId = ++layoutRunId;
  const engine = readGeometryEngine();
  if (engine === 'native_elk') {
    const displayGraphData = graphData ?? (lf.getGraphData() as LogicFlowGraphProjection);
    void runNativeElkLayout(reason, runId, displayGraphData);
    return;
  }
  try {
    if (!runLegacyOfficialLayout(reason, runId)) {
      pendingOfficialLayout = null;
      console.warn('AI_FLOW_LAYOUT_UNAVAILABLE', { reason });
      fitReadableView();
    }
  } catch (error) {
    pendingOfficialLayout = null;
    console.warn('AI_FLOW_LAYOUT_FAILED', { reason, error });
    fitReadableView();
  }
}

function render(graph: WorkflowGraph, reason: LayoutReason = 'server-state') {
  workflowGraph = structuredClone(graph);
  frontendSelectionState = emptySelection();
  const graphData = workflowToLogicFlow(workflowGraph);
  lf.render(graphData);
  runOfficialLayout(reason, graphData);
  updateSelection();
}

function pretty(value: unknown) {
  return JSON.stringify(value, null, 2);
}

function updateSelection(eventName = 'selection:update', payload?: unknown) {
  const raw = readLogicFlowSelection();
  const effective = getSelection();
  writeSelectionDebug(effective);
  console.log('LF_SELECTION_EVENT', {
    eventName,
    raw,
    effective,
    frontendSelectionState,
    quickSelectionOverride,
    payload
  });
}

const selectionEventHandlers: Record<string, (payload?: unknown) => void> = {
  'node:click': (payload) => setFrontendSelectionFromEvent('node', payload),
  'edge:click': (payload) => setFrontendSelectionFromEvent('edge', payload),
  'blank:click': (payload) => setFrontendSelectionFromEvent('blank', payload),
  'graph:render': () => {
    if (pendingOfficialLayout) return;
    frontendSelectionState = emptySelection();
  }
};

[
  'selection:selected',
  'selection:unselected',
  'selection:select',
  'selection:mousedown',
  'selection:dragstart',
  'selection:drop',
  'node:click',
  'edge:click',
  'blank:click',
  'graph:render'
].forEach((eventName) => {
  lf.on(eventName, (payload: unknown) => {
    selectionEventHandlers[eventName]?.(payload);
    updateSelection(eventName, payload);
    window.setTimeout(() => updateSelection(`${eventName}:afterTick`, payload), 0);
  });
});

instructionEl.addEventListener('focus', () => {
  updateSelection('input:focus');
  window.setTimeout(() => updateSelection('input:focus:afterTick'), 0);
});

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
  const asksToInspect = compact.includes('检查') || compact.includes('看一下') || compact.includes('分析') || compact.includes('评估');
  const mentionsLogic = compact.includes('逻辑') || compact.includes('混乱') || compact.includes('有点乱') || compact.includes('乱');
  return (
    compact.includes('几种') ||
    compact.includes('哪一种') ||
    compact.includes('为什么') ||
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
  if (result.status === 'needs_clarification') {
    return '\u6211\u8fd8\u9700\u8981\u4f60\u8865\u5145\u66f4\u660e\u786e\u7684\u4fee\u6539\u5bf9\u8c61\u6216\u76ee\u6807\u3002';
  }
  if (result.status === 'execution_error') {
    return '\u8fd9\u6b21\u4fee\u6539\u6ca1\u6709\u6210\u529f\u5e94\u7528\uff0c\u6d41\u7a0b\u56fe\u4fdd\u6301\u4e0d\u53d8\u3002\u8bf7\u6362\u4e00\u79cd\u66f4\u660e\u786e\u7684\u8bf4\u6cd5\u91cd\u8bd5\u3002';
  }
  return '\u8fd9\u6b21\u4fee\u6539\u6ca1\u6709\u5b8c\u6210\uff0c\u8bf7\u6362\u4e00\u79cd\u66f4\u660e\u786e\u7684\u8bf4\u6cd5\u91cd\u8bd5\u3002';
}

function summarizeFeedback(result: EditResult) {
  const feedback = result.user_feedback;
  if (!feedback?.message) return '';
  const lines = [feedback.message];
  const hasVisibleCandidates = Boolean(
    feedback.clarification?.candidates?.some((group) => Array.isArray(group.candidates) && group.candidates.length > 0)
  );
  if (result.pending_id && hasVisibleCandidates) {
    lines.push('');
    lines.push('\u4f60\u53ef\u4ee5\u76f4\u63a5\u56de\u590d\u201c\u7b2c\u4e00\u4e2a\u201d\u6216\u201c\u7b2c\u4e8c\u4e2a\u201d\u7ee7\u7eed\u4e0a\u4e00\u8f6e\u4fee\u6539\u3002');
  }
  return lines.join('\n');
}

function summarizeResult(result: EditResult) {
  const feedbackSummary = summarizeFeedback(result);
  if (feedbackSummary) return feedbackSummary;
  if (result.status === 'success' || result.status === 'partial_success') {
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
    return '\u8fd9\u6b21\u8bf7\u6c42\u6ca1\u6709\u6210\u529f\u63d0\u4ea4\uff0c\u8bf7\u7a0d\u540e\u91cd\u8bd5\uff0c\u6216\u6362\u4e00\u79cd\u66f4\u660e\u786e\u7684\u8bf4\u6cd5\u3002';
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
  appendMessage('assistant', '\u5df2\u91cd\u7f6e\u4e3a\u521d\u59cb\u6d41\u7a0b\u3002\u4f60\u53ef\u4ee5\u76f4\u63a5\u63cf\u8ff0\u5e0c\u671b\u600e\u4e48\u6539\uff0c\u6211\u4f1a\u66f4\u65b0\u5de6\u4fa7\u753b\u5e03\u3002');
  render(initialGraph, 'reset');
  void createSession(initialGraph);
});

clearSelectionEl.addEventListener('click', () => {
  frontendSelectionState = emptySelection();
  applySelectionHighlight();
  updateSelection('selection:clear-button');
});

document.querySelector<HTMLButtonElement>('#run')!.addEventListener('mousedown', () => {
  updateSelection('send:mousedown');
  window.setTimeout(() => updateSelection('send:mousedown:afterTick'), 0);
});

document.querySelector<HTMLButtonElement>('#run')!.addEventListener('click', () => {
  updateSelection('send:click');
  window.setTimeout(() => updateSelection('send:click:afterTick'), 0);
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
  if (pendingAssistantEl) pendingAssistantEl.textContent = '\u6b63\u5728\u7406\u89e3\u4f60\u7684\u4fee\u6539\u610f\u56fe...';
  const selectionBeforeSend = getSelection();
  writeSelectionDebug(selectionBeforeSend, selectionBeforeSend);
  const requestBody = {
    session_id: sessionId,
    instruction: userText,
    selection: selectionBeforeSend,
    request_context: {
      visible_area: { left: 0, top: 0, right: 1400, bottom: 640 },
      original_user_message: userText,
      conversation_history: requestHistory,
      graph_context: graphContext()
    }
  };
  console.log('AI_EDIT_SELECTION_BEFORE_SEND', selectionBeforeSend);
  console.log('AI_EDIT_REQUEST_BODY', requestBody);
  const result = (await fetch('/api/ai-edit', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(requestBody)
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
  runOfficialLayout('manual');
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
    body: JSON.stringify({ limit: 8 })
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

render(initialGraph, 'initial');
void createSession(initialGraph);
appendMessage('assistant', '你好，我可以帮你直接修改左侧流程图。请先在画布上选择节点或分支，也可以直接用业务语言描述希望流程图满足的目标。');
