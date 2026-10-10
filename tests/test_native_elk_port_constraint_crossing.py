from __future__ import annotations

import json
import subprocess
import textwrap
from pathlib import Path


FRONTEND_DIR = Path("demo/frontend")


GEOMETRY_SCRIPT = r"""
const ELK = require('elkjs/lib/elk.bundled.js');
const opts = {
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
const postEditNodes = [
  ['n_start','start',96,96,110,260], ['n_submit','task',140,72,280,260],
  ['n_retrieval','retrieval',140,72,470,150], ['n_tool','tool_call',140,72,470,390],
  ['n_decision','decision',104,104,700,270], ['n_report','llm_reasoning',140,72,960,120],
  ['n_publish','output',140,72,1240,120], ['n_human','human_review',140,72,960,520],
  ['n_end','end',96,96,1500,270]
].map(([id,type,width,height,x,y]) => ({ id, type, width, height, x, y }));
const postEditEdges = [
  ['e_start_submit','n_start','n_submit'], ['e_submit_retrieval','n_submit','n_retrieval'],
  ['e_retrieval_tool','n_retrieval','n_tool'], ['e_tool_decision','n_tool','n_decision'],
  ['e_decision_report','n_decision','n_report'], ['e_decision_human','n_decision','n_human'],
  ['e_report_publish','n_report','n_publish'], ['e_report_end','n_report','n_end'],
  ['e_human_publish','n_human','n_publish'], ['e_publish_end','n_publish','n_end']
].map(([id, source, target]) => ({ id, source, target }));
const initialEdges = postEditEdges.map((edge) =>
  edge.id === 'e_report_end' ? { id: 'e_report_human', source: 'n_report', target: 'n_human' } : edge
);
const fanNodes = [
  ['a','start',96,96,0,0], ['b','task',140,72,200,-140], ['c','task',140,72,200,0],
  ['d','task',140,72,200,140], ['finish','end',96,96,440,0]
].map(([id,type,width,height,x,y]) => ({ id, type, width, height, x, y }));
const fanEdges = [
  ['ab','a','b'], ['ac','a','c'], ['ad','a','d'], ['bf','b','finish'], ['cf','c','finish'], ['df','d','finish']
].map(([id, source, target]) => ({ id, source, target }));
function kind(n) { return n.type === 'decision' ? 'diamond' : (n.type === 'start' || n.type === 'end') ? 'ellipse' : 'rect'; }
function geom(n) { return { kind: kind(n), width: n.width, height: n.height, rx: n.width / 2, ry: n.height / 2 }; }
function side(s, t) {
  const dx = t.x - s.x, dy = t.y - s.y;
  return Math.abs(dy) > Math.abs(dx) * 1.15 ? (dy >= 0 ? 'bottom' : 'top') : (dx >= 0 ? 'right' : 'left');
}
function opp(s) { return s === 'left' ? 'right' : s === 'right' ? 'left' : s === 'top' ? 'bottom' : 'top'; }
function elkSide(s) { return s === 'left' ? 'WEST' : s === 'right' ? 'EAST' : s === 'top' ? 'NORTH' : 'SOUTH'; }
function dist(sd, i, count, g) {
  const span = sd === 'left' || sd === 'right' ? g.height : g.width;
  const usable = Math.max(0, span - 20);
  return count <= 1 ? 0 : -usable / 2 + usable * i / (count - 1);
}
function boundaryPoint(sd, i, count, g) {
  const stable = dist(sd, i, count, g), { width, height, rx, ry, kind } = g;
  if (kind === 'rect') {
    if (sd === 'left') return { x: -width / 2, y: stable };
    if (sd === 'right') return { x: width / 2, y: stable };
    if (sd === 'top') return { x: stable, y: -height / 2 };
    return { x: stable, y: height / 2 };
  }
  if (kind === 'diamond') {
    if (sd === 'left' || sd === 'right') {
      const y = Math.max(-ry, Math.min(ry, stable));
      const x = rx * Math.max(0, 1 - Math.abs(y) / ry);
      return { x: sd === 'right' ? x : -x, y };
    }
    const x = Math.max(-rx, Math.min(rx, stable));
    const y = ry * Math.max(0, 1 - Math.abs(x) / rx);
    return { x, y: sd === 'bottom' ? y : -y };
  }
  if (sd === 'left' || sd === 'right') {
    const y = Math.max(-ry, Math.min(ry, stable));
    const x = rx * Math.sqrt(Math.max(0, 1 - (y * y) / (ry * ry)));
    return { x: sd === 'right' ? x : -x, y };
  }
  const x = Math.max(-rx, Math.min(rx, stable));
  const y = ry * Math.sqrt(Math.max(0, 1 - (x * x) / (rx * rx)));
  return { x, y: sd === 'bottom' ? y : -y };
}
function assign(nodes, edges) {
  const nm = new Map(nodes.map((n) => [n.id, n]));
  const display = edges.map((e) => ({ ...e, sourceNodeId: e.source, targetNodeId: e.target }));
  const demands = new Map();
  function ensure(id, sd) {
    let m = demands.get(id); if (!m) { m = new Map(); demands.set(id, m); }
    let a = m.get(sd); if (!a) { a = []; m.set(sd, a); }
    return a;
  }
  for (const edge of display) {
    const s = nm.get(edge.source), t = nm.get(edge.target), ss = side(s, t), ts = opp(ss);
    ensure(s.id, ss).push({ edge, opposite: t });
    ensure(t.id, ts).push({ edge, opposite: s });
  }
  const anchors = new Map();
  for (const [nodeId, sides] of demands) {
    const node = nm.get(nodeId), g = geom(node), nodeAnchors = [];
    for (const [sd, items] of sides) {
      items.sort((a, b) => ((sd === 'left' || sd === 'right' ? a.opposite.y - b.opposite.y : a.opposite.x - b.opposite.x) || a.edge.id.localeCompare(b.edge.id)));
      items.forEach((item, index) => {
        const id = `${nodeId}:${sd}:${index}`, p = boundaryPoint(sd, index, items.length, g);
        nodeAnchors.push({ id, side: sd, index, ...p });
        if (item.edge.source === nodeId) item.edge.sourceAnchorId = id; else item.edge.targetAnchorId = id;
      });
    }
    anchors.set(nodeId, nodeAnchors);
  }
  return { display, anchors };
}
function port(anchor, node, variant) {
  const p = { id: anchor.id, width: 1, height: 1 };
  if (variant === 'FIXED_POS') {
    const g = geom(node);
    p.x = g.width / 2 + anchor.x - 0.5;
    p.y = g.height / 2 + anchor.y - 0.5;
  } else {
    p.layoutOptions = { 'elk.port.side': elkSide(anchor.side) };
  }
  return p;
}
function points(section) { return [section.startPoint, ...(section.bendPoints || []), section.endPoint]; }
function boundaryX(g, y) {
  const safeY = Math.max(-g.ry, Math.min(g.ry, y));
  if (g.kind === 'diamond') return g.rx * Math.max(0, 1 - Math.abs(safeY) / g.ry);
  if (g.kind === 'ellipse') return g.rx * Math.sqrt(Math.max(0, 1 - (safeY * safeY) / (g.ry * g.ry)));
  return g.width / 2;
}
function boundaryY(g, x) {
  const safeX = Math.max(-g.rx, Math.min(g.rx, x));
  if (g.kind === 'diamond') return g.ry * Math.max(0, 1 - Math.abs(safeX) / g.rx);
  if (g.kind === 'ellipse') return g.ry * Math.sqrt(Math.max(0, 1 - (safeX * safeX) / (g.rx * g.rx)));
  return g.height / 2;
}
function project(node, endpoint, adjacent) {
  const g = geom(node), dx = endpoint.x - adjacent.x, dy = endpoint.y - adjacent.y;
  if (Math.abs(dx) >= Math.abs(dy)) {
    const localY = Math.max(-g.ry, Math.min(g.ry, endpoint.y - node.cy));
    const side = endpoint.x >= node.cx ? 1 : -1;
    return { x: node.cx + side * boundaryX(g, localY), y: node.cy + localY };
  }
  const localX = Math.max(-g.rx, Math.min(g.rx, endpoint.x - node.cx));
  const side = endpoint.y >= node.cy ? 1 : -1;
  return { x: node.cx + localX, y: node.cy + side * boundaryY(g, localX) };
}
function between(a, b, c) {
  return Math.min(a.x, b.x) <= c.x && c.x <= Math.max(a.x, b.x) && Math.min(a.y, b.y) <= c.y && c.y <= Math.max(a.y, b.y);
}
function segmentsCross(a, b, c, d) {
  const eps = 1e-6, cross = (p, q, r) => (q.x - p.x) * (r.y - p.y) - (q.y - p.y) * (r.x - p.x);
  const c1 = cross(a, b, c), c2 = cross(a, b, d), c3 = cross(c, d, a), c4 = cross(c, d, b);
  if (Math.abs(c1) < eps && between(a, b, c)) return true;
  if (Math.abs(c2) < eps && between(a, b, d)) return true;
  if (Math.abs(c3) < eps && between(c, d, a)) return true;
  if (Math.abs(c4) < eps && between(c, d, b)) return true;
  return (c1 > eps) !== (c2 > eps) && (c3 > eps) !== (c4 > eps);
}
function collinearOverlap(a, b, c, d) {
  if (a.y === b.y && c.y === d.y && a.y === c.y) {
    return Math.max(0, Math.min(Math.max(a.x, b.x), Math.max(c.x, d.x)) - Math.max(Math.min(a.x, b.x), Math.min(c.x, d.x)));
  }
  if (a.x === b.x && c.x === d.x && a.x === c.x) {
    return Math.max(0, Math.min(Math.max(a.y, b.y), Math.max(c.y, d.y)) - Math.max(Math.min(a.y, b.y), Math.min(c.y, d.y)));
  }
  return 0;
}
function shapeError(node, point) {
  const g = geom(node), x = point.x - node.cx, y = point.y - node.cy;
  if (g.kind === 'diamond') return Math.abs(Math.abs(x) / g.rx + Math.abs(y) / g.ry - 1);
  if (g.kind === 'ellipse') return Math.abs((x * x) / (g.rx * g.rx) + (y * y) / (g.ry * g.ry) - 1);
  return Math.min(Math.abs(Math.abs(x) - g.rx), Math.abs(Math.abs(y) - g.ry));
}
function segmentCrossesNodeBBox(a, b, node) {
  const left = node.x, right = node.x + node.width, top = node.y, bottom = node.y + node.height;
  if (a.x === b.x) {
    if (a.x <= left || a.x >= right) return false;
    return Math.max(Math.min(a.y, b.y), top) < Math.min(Math.max(a.y, b.y), bottom);
  }
  if (a.y === b.y) {
    if (a.y <= top || a.y >= bottom) return false;
    return Math.max(Math.min(a.x, b.x), left) < Math.min(Math.max(a.x, b.x), right);
  }
  return false;
}
function metrics(routes, laidNodes) {
  const crossings = [];
  const longOverlaps = [];
  const nodeEdgeIntersections = [];
  for (let i = 0; i < routes.length; i++) for (let j = i + 1; j < routes.length; j++) {
    const a = routes[i], b = routes[j];
    if (a.source === b.source || a.source === b.target || a.target === b.source || a.target === b.target) continue;
    let overlap = 0;
    for (let ai = 1; ai < a.points.length; ai++) for (let bi = 1; bi < b.points.length; bi++) {
      if (segmentsCross(a.points[ai - 1], a.points[ai], b.points[bi - 1], b.points[bi])) crossings.push([a.id, b.id]);
      overlap += collinearOverlap(a.points[ai - 1], a.points[ai], b.points[bi - 1], b.points[bi]);
    }
    if (overlap > 48) longOverlaps.push([a.id, b.id, overlap]);
  }
  const nodeById = new Map(laidNodes.map((node) => [node.id, node]));
  for (const route of routes) {
    for (const node of laidNodes) {
      if (node.id === route.source || node.id === route.target) continue;
      for (let index = 1; index < route.points.length; index++) {
        if (segmentCrossesNodeBBox(route.points[index - 1], route.points[index], node)) {
          nodeEdgeIntersections.push([route.id, node.id]);
        }
      }
    }
  }
  const anchorErrors = routes.flatMap((route) => {
    const source = nodeById.get(route.source), target = nodeById.get(route.target);
    return [shapeError(source, route.points[0]), shapeError(target, route.points[route.points.length - 1])];
  });
  const bbox = {
    width: Math.max(...laidNodes.map((n) => n.x + n.width)) - Math.min(...laidNodes.map((n) => n.x)),
    height: Math.max(...laidNodes.map((n) => n.y + n.height)) - Math.min(...laidNodes.map((n) => n.y))
  };
  return { crossings, nodeEdgeIntersections, longOverlaps, maxAnchorError: Math.max(...anchorErrors), bbox };
}
async function run(name, nodes, edges, variant) {
  const { display, anchors } = assign(nodes, edges);
  const input = {
    id: 'g',
    layoutOptions: opts,
    children: nodes.map((n) => ({
      id: n.id,
      width: n.width,
      height: n.height,
      ports: (anchors.get(n.id) || []).map((a) => port(a, n, variant)),
      layoutOptions: { 'elk.portConstraints': variant }
    })),
    edges: display.map((e) => ({ id: e.id, sources: [e.sourceAnchorId], targets: [e.targetAnchorId], sourcePort: e.sourceAnchorId, targetPort: e.targetAnchorId }))
  };
  const out = await new ELK().layout(input);
  const laid = new Map(out.children.map((n) => [n.id, { ...nodes.find((row) => row.id === n.id), ...n, cx: n.x + n.width / 2, cy: n.y + n.height / 2 }]));
  const rawRoutes = out.edges.map((e) => {
    const d = display.find((row) => row.id === e.id);
    return { id: e.id, source: d.source, target: d.target, points: points(e.sections[0]) };
  });
  const mappedRoutes = out.edges.map((e) => {
    const d = display.find((row) => row.id === e.id);
    const p = points(e.sections[0]);
    const source = laid.get(d.source), target = laid.get(d.target);
    p[0] = project(source, p[0], p[1]);
    p[p.length - 1] = project(target, p[p.length - 1], p[p.length - 2]);
    return { id: e.id, source: d.source, target: d.target, points: p };
  });
  return { name, variant, raw: metrics(rawRoutes, Array.from(laid.values())), mapped: metrics(mappedRoutes, Array.from(laid.values())) };
}
(async () => {
  const result = [
    await run('post-edit', postEditNodes, postEditEdges, 'FIXED_POS'),
    await run('post-edit', postEditNodes, postEditEdges, 'FIXED_SIDE'),
    await run('initial', postEditNodes, initialEdges, 'FIXED_SIDE'),
    await run('fan-in-out', fanNodes, fanEdges, 'FIXED_SIDE'),
    await run('post-edit-repeat', postEditNodes, postEditEdges, 'FIXED_SIDE')
  ];
  console.log(JSON.stringify(result));
})();
"""


def geometry_matrix() -> list[dict]:
    process = subprocess.run(
        ["node", "-e", textwrap.dedent(GEOMETRY_SCRIPT)],
        cwd=FRONTEND_DIR,
        check=True,
        text=True,
        capture_output=True,
    )
    return json.loads(process.stdout)


def by_name_variant(rows: list[dict], name: str, variant: str) -> dict:
    return next(row for row in rows if row["name"] == name and row["variant"] == variant)


def test_fixed_pos_baseline_reproduces_post_edit_crossing() -> None:
    rows = geometry_matrix()
    baseline = by_name_variant(rows, "post-edit", "FIXED_POS")

    assert len(baseline["raw"]["crossings"]) >= 1


def test_fixed_side_removes_raw_and_mapped_post_edit_crossing() -> None:
    rows = geometry_matrix()
    fixed_side = by_name_variant(rows, "post-edit", "FIXED_SIDE")

    assert fixed_side["raw"]["crossings"] == []
    assert fixed_side["mapped"]["crossings"] == []
    assert fixed_side["mapped"]["nodeEdgeIntersections"] == []
    assert fixed_side["mapped"]["longOverlaps"] == []
    assert fixed_side["mapped"]["maxAnchorError"] < 1e-6


def test_fixed_side_regresses_initial_and_fan_in_out_without_crossing() -> None:
    rows = geometry_matrix()

    for name in ["initial", "fan-in-out"]:
        row = by_name_variant(rows, name, "FIXED_SIDE")
        assert row["raw"]["crossings"] == []
        assert row["mapped"]["crossings"] == []
        assert row["mapped"]["nodeEdgeIntersections"] == []
        assert row["mapped"]["longOverlaps"] == []
        assert row["mapped"]["maxAnchorError"] < 1e-6
        assert row["mapped"]["bbox"]["width"] > 0
        assert row["mapped"]["bbox"]["height"] > 0


def test_fixed_side_layout_is_deterministic_for_crossing_fixture() -> None:
    rows = geometry_matrix()
    first = by_name_variant(rows, "post-edit", "FIXED_SIDE")
    repeat = by_name_variant(rows, "post-edit-repeat", "FIXED_SIDE")

    assert first["raw"] == repeat["raw"]
    assert first["mapped"] == repeat["mapped"]
