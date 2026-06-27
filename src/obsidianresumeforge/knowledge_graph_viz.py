"""
Generate an interactive vis.js HTML graph from a KGStore.

Output: {vault_path}/KnowledgeGraph/graph.html — open in any browser.

Features:
  - Hover detail panel with metric bars
  - Click-to-focus: highlights neighbourhood, dims others
  - Search bar: highlight + zoom matching nodes
  - Filter panel: node type and keyword decision checkboxes
  - Pathfinding: BFS shortest path between two selected nodes
  - Legend for node types and edge styles
  - Node size = degree centrality + PageRank contribution
  - Edge style = confidence: solid (REUSE/structural), dashed (ADAPT), dotted (NEW)
  - Dark aesthetic theme (#0d1117 GitHub dark)
  - Physics toggle and Reset View controls
"""
from __future__ import annotations

import json
import math
import pathlib
import logging
import random

logger = logging.getLogger(__name__)

_OUTPUT_FILENAME = "graph.html"

# ── colour palette ──────────────────────────────────────────────────────────
_COLORS = {
    "run_low":      "#c0392b",
    "run_mid":      "#e67e22",
    "run_high":     "#27ae60",
    "role":         "#00b4d8",
    "company":      "#9b5de5",
    "kw_reuse":     "#06d6a0",
    "kw_adapt":     "#ffd166",
    "kw_new":       "#ef476f",
    "kw_none":      "#6c757d",
    "bullet":       "#8d99ae",
    "bullet_promo": "#f9c74f",
}

_EDGE_COLORS = {
    "for_role":     "#00b4d8",
    "at_company":   "#9b5de5",
    "used_keyword": "#555577",
    "used_bullet":  "#444455",
    "backed_by":    "#333344",
}


# ── metrics ──────────────────────────────────────────────────────────────────

def _compute_metrics(nodes: dict, edges: list) -> dict[str, dict]:
    """Return {node_id: {degree, degree_centrality, pagerank}} for all nodes."""
    n = len(nodes)
    if n == 0:
        return {}

    # Build adjacency (undirected degree)
    degree: dict[str, int] = {nid: 0 for nid in nodes}
    adj: dict[str, list[str]] = {nid: [] for nid in nodes}
    for e in edges:
        src, dst = e["src"], e["dst"]
        if src in nodes and dst in nodes:
            degree[src] += 1
            degree[dst] += 1
            adj[src].append(dst)
            adj[dst].append(src)

    denom = max(n - 1, 1)
    dc = {nid: degree[nid] / denom for nid in nodes}

    # PageRank — 20-iteration power method
    d = 0.85
    pr = {nid: 1.0 / n for nid in nodes}
    for _ in range(20):
        new_pr: dict[str, float] = {}
        for nid in nodes:
            rank_sum = 0.0
            for nb in adj[nid]:
                if degree[nb] > 0:
                    rank_sum += pr[nb] / degree[nb]
            new_pr[nid] = (1 - d) / n + d * rank_sum
        pr = new_pr

    return {
        nid: {
            "degree": degree[nid],
            "degree_centrality": round(dc[nid], 4),
            "pagerank": round(pr[nid], 6),
        }
        for nid in nodes
    }


def _node_size(dc: float, pr: float) -> int:
    raw = 12 + dc * 28 + pr * 400
    return max(10, min(40, int(raw)))


def _score_to_color(score: float) -> str:
    """Interpolate red→amber→green hex by composite_score 0→1."""
    score = max(0.0, min(1.0, score))
    hue = score * 120  # 0=red, 120=green
    # Convert HSL(hue, 70%, 45%) → hex
    h = hue / 60
    c = 0.7 * (1 - abs(2 * 0.45 - 1))  # chroma
    x = c * (1 - abs(h % 2 - 1))
    if h < 1:
        r1, g1, b1 = c, x, 0
    elif h < 2:
        r1, g1, b1 = x, c, 0
    elif h < 3:
        r1, g1, b1 = 0, c, x
    else:
        r1, g1, b1 = 0, x, c
    m = 0.45 - c / 2
    r, g, b = int((r1 + m) * 255), int((g1 + m) * 255), int((b1 + m) * 255)
    return f"#{r:02x}{g:02x}{b:02x}"


# ── Pre-computed layout ───────────────────────────────────────────────────────

_TIER_Y = {"run": 0, "role": -320, "company": -320, "keyword": 320, "bullet": 640}
_TIER_XSCALE = {"run": 300, "role": 250, "company": 250, "keyword": 600, "bullet": 700}


def _compute_layout(nodes: dict) -> dict[str, tuple[float, float]]:
    """Assign x,y positions by node type tier so the graph starts laid out."""
    groups: dict[str, list[str]] = {}
    for nid, attrs in nodes.items():
        t = attrs.get("type", "unknown")
        groups.setdefault(t, []).append(nid)

    positions: dict[str, tuple[float, float]] = {}
    rng = random.Random(42)

    for ntype, nids in groups.items():
        y_base = _TIER_Y.get(ntype, 0)
        x_scale = _TIER_XSCALE.get(ntype, 400)
        n = len(nids)
        for i, nid in enumerate(nids):
            if n == 1:
                x = 0.0
            else:
                x = (i / (n - 1) - 0.5) * 2 * x_scale
            # Small jitter so overlapping nodes spread slightly
            jitter_x = rng.uniform(-20, 20)
            jitter_y = rng.uniform(-20, 20)
            positions[nid] = (round(x + jitter_x, 1), round(y_base + jitter_y, 1))

    return positions


# ── vis.js data builders ─────────────────────────────────────────────────────

def _build_vis_nodes(nodes: dict, metrics: dict, layout: dict[str, tuple[float, float]] | None = None) -> list[dict]:
    vis_nodes = []
    for node_id, attrs in nodes.items():
        m = metrics.get(node_id, {"degree": 0, "degree_centrality": 0.0, "pagerank": 1.0 / max(len(nodes), 1)})
        ntype = attrs.get("type", "")
        size = _node_size(m["degree_centrality"], m["pagerank"])

        if ntype == "run":
            score = attrs.get("composite_score", 0.0)
            color = _score_to_color(score)
            label = attrs.get("run_id", node_id)
            detail = {
                "type": "Run",
                "id": node_id,
                "role": attrs.get("role", "—"),
                "company": attrs.get("company", "—"),
                "score": score,
                "passed": attrs.get("passed", False),
                "date": attrs.get("date", "—"),
            }

        elif ntype == "role":
            color = _COLORS["role"]
            run_count = attrs.get("run_count", 1)
            label = attrs.get("display", attrs.get("name", node_id))
            detail = {"type": "Role", "id": node_id, "run_count": run_count}

        elif ntype == "company":
            color = _COLORS["company"]
            run_count = attrs.get("run_count", 1)
            label = attrs.get("name", node_id)
            detail = {"type": "Company", "id": node_id, "run_count": run_count}

        elif ntype == "keyword":
            last = attrs.get("last_decision", "")
            color = _COLORS.get(f"kw_{last.lower()}", _COLORS["kw_none"])
            label = attrs.get("text", node_id)
            detail = {
                "type": "Keyword",
                "id": node_id,
                "text": attrs.get("text", ""),
                "domain": attrs.get("domain", ""),
                "last_decision": last or "—",
                "reuse_count": attrs.get("reuse_count", 0),
                "adapt_count": attrs.get("adapt_count", 0),
                "new_count": attrs.get("new_count", 0),
                "run_count": attrs.get("run_count", 0),
            }

        elif ntype == "bullet":
            promoted = attrs.get("promoted", False)
            color = _COLORS["bullet_promo"] if promoted else _COLORS["bullet"]
            label = "★" if promoted else "•"
            text = attrs.get("text", "")
            detail = {
                "type": "Bullet",
                "id": node_id,
                "promoted": promoted,
                "text": text,
                "source_file": attrs.get("source_file", "—"),
                "use_count": attrs.get("use_count", 1),
                "avg_source_score": attrs.get("avg_source_score", 0.0),
            }

        else:
            color = "#888888"
            label = node_id
            detail = {"type": ntype or "unknown", "id": node_id}

        node_dict: dict = {
            "id": node_id,
            "label": label,
            "color": {"background": color, "border": color, "highlight": {"background": "#ffffff", "border": color}},
            "size": size,
            "font": {"color": "#e0e0e0", "size": 12},
            "_type": ntype,
            "_attrs": attrs,
            "_metrics": m,
            "_detail": detail,
        }
        if layout and node_id in layout:
            x, y = layout[node_id]
            node_dict["x"] = x
            node_dict["y"] = y
        vis_nodes.append(node_dict)

    return vis_nodes


def _build_vis_edges(edges: list, nodes: dict) -> list[dict]:
    vis_edges = []
    for i, edge in enumerate(edges):
        src, dst, rel = edge["src"], edge["dst"], edge["rel"]
        if src not in nodes or dst not in nodes:
            continue

        eattrs = edge.get("attrs", {})
        color = _EDGE_COLORS.get(rel, "#555555")
        width = 1

        # Edge dashes: solid=REUSE/structural, dashed=ADAPT, dotted=NEW
        decision = eattrs.get("decision", "")
        if rel == "used_keyword":
            priority = eattrs.get("priority", "Medium")
            width = {"Critical": 3, "High": 2, "Medium": 1, "Low": 1}.get(priority, 1)
            color = _COLORS.get(f"kw_{decision.lower()}", "#666666")
            if decision == "ADAPT":
                dashes = [8, 4]
            elif decision == "NEW":
                dashes = [3, 3]
            else:
                dashes = False
            label_str = f"{decision} | {priority}"
        elif rel == "backed_by":
            dashes = [4, 4]
            label_str = f"backed_by | {eattrs.get('run_id', '—')}"
        else:
            dashes = False
            label_str = rel

        vis_edges.append({
            "id": i,
            "from": src,
            "to": dst,
            "color": {"color": color, "highlight": "#ffffff", "opacity": 0.8},
            "width": width,
            "dashes": dashes,
            "_rel": rel,
            "_decision": decision,
            "_label": label_str,
        })

    return vis_edges


def _build_stats(nodes: dict, metrics: dict) -> dict:
    type_counts: dict[str, int] = {}
    for attrs in nodes.values():
        t = attrs.get("type", "unknown")
        type_counts[t] = type_counts.get(t, 0) + 1

    top_by_dc = sorted(metrics.items(), key=lambda x: x[1]["degree_centrality"], reverse=True)[:5]
    top_by_pr = sorted(metrics.items(), key=lambda x: x[1]["pagerank"], reverse=True)[:5]

    return {
        "node_count": len(nodes),
        "type_counts": type_counts,
        "top_degree_centrality": [{"id": nid, "dc": round(m["degree_centrality"], 3)} for nid, m in top_by_dc],
        "top_pagerank": [{"id": nid, "pr": round(m["pagerank"], 5)} for nid, m in top_by_pr],
    }


# ── HTML template ─────────────────────────────────────────────────────────────

def _render_html(vis_nodes: list, vis_edges: list, stats: dict) -> str:
    nodes_json = json.dumps(vis_nodes, ensure_ascii=False)
    edges_json = json.dumps(vis_edges, ensure_ascii=False)
    stats_json = json.dumps(stats, ensure_ascii=False)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>ObsidianResumeForge — Knowledge Graph</title>
<script src="https://unpkg.com/vis-network@9.1.9/standalone/umd/vis-network.min.js"></script>
<style>
  *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}

  :root {{
    --bg: #0d1117;
    --surface: #161b22;
    --surface2: #21262d;
    --border: #30363d;
    --text: #c9d1d9;
    --text-muted: #8b949e;
    --accent: #58a6ff;
    --green: #3fb950;
    --yellow: #d29922;
    --red: #f85149;
    --purple: #bc8cff;
    --teal: #39d353;
  }}

  body {{
    background: var(--bg);
    color: var(--text);
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', monospace;
    font-size: 13px;
    display: flex;
    flex-direction: column;
    height: 100vh;
    overflow: hidden;
  }}

  /* ── Top bar ── */
  #topbar {{
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 8px 12px;
    background: var(--surface);
    border-bottom: 1px solid var(--border);
    flex-shrink: 0;
  }}
  #topbar h1 {{
    font-size: 13px;
    font-weight: 600;
    color: var(--accent);
    white-space: nowrap;
    margin-right: 8px;
  }}
  #search-input {{
    flex: 1;
    max-width: 280px;
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 6px;
    color: var(--text);
    padding: 5px 10px;
    font-size: 12px;
    outline: none;
  }}
  #search-input:focus {{ border-color: var(--accent); }}
  .btn {{
    padding: 5px 10px;
    border-radius: 6px;
    border: 1px solid var(--border);
    background: var(--surface2);
    color: var(--text);
    cursor: pointer;
    font-size: 12px;
    white-space: nowrap;
    transition: border-color 0.15s;
  }}
  .btn:hover {{ border-color: var(--accent); color: var(--accent); }}
  .btn.active {{ background: var(--accent); color: #0d1117; border-color: var(--accent); }}
  #path-status {{
    font-size: 11px;
    color: var(--text-muted);
    flex: 1;
    text-align: right;
  }}

  /* ── Main layout ── */
  #main {{
    display: flex;
    flex: 1;
    overflow: hidden;
  }}

  /* ── Left sidebar ── */
  #sidebar {{
    width: 210px;
    flex-shrink: 0;
    background: var(--surface);
    border-right: 1px solid var(--border);
    overflow-y: auto;
    padding: 10px;
    display: flex;
    flex-direction: column;
    gap: 12px;
  }}
  .panel-title {{
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: var(--text-muted);
    margin-bottom: 6px;
  }}

  /* Legend */
  .legend-item {{
    display: flex;
    align-items: center;
    gap: 7px;
    margin-bottom: 5px;
    font-size: 12px;
  }}
  .legend-dot {{
    width: 12px;
    height: 12px;
    border-radius: 50%;
    flex-shrink: 0;
  }}
  .legend-line {{
    width: 24px;
    height: 2px;
    flex-shrink: 0;
  }}

  /* Filters */
  .filter-group {{ margin-bottom: 8px; }}
  .filter-group label {{
    display: flex;
    align-items: center;
    gap: 6px;
    font-size: 12px;
    cursor: pointer;
    padding: 2px 0;
    color: var(--text);
  }}
  .filter-group label:hover {{ color: var(--accent); }}
  input[type=checkbox] {{ accent-color: var(--accent); }}

  /* Stats */
  .stat-row {{
    display: flex;
    justify-content: space-between;
    font-size: 11px;
    margin-bottom: 3px;
  }}
  .stat-row .val {{ color: var(--accent); font-weight: 600; }}
  .top-list {{ font-size: 11px; color: var(--text-muted); }}
  .top-list li {{
    list-style: none;
    display: flex;
    justify-content: space-between;
    padding: 1px 0;
  }}
  .top-list li span {{ color: var(--text); }}

  /* ── Graph canvas ── */
  #network-wrap {{
    flex: 1;
    position: relative;
    overflow: hidden;
  }}
  #network {{ width: 100%; height: 100%; }}

  /* ── Right detail panel ── */
  #detail-panel {{
    width: 240px;
    flex-shrink: 0;
    background: var(--surface);
    border-left: 1px solid var(--border);
    overflow-y: auto;
    padding: 10px;
    display: none;
  }}
  #detail-panel.visible {{ display: block; }}
  #detail-close {{
    float: right;
    cursor: pointer;
    color: var(--text-muted);
    font-size: 16px;
    line-height: 1;
  }}
  #detail-close:hover {{ color: var(--text); }}
  #detail-title {{
    font-size: 13px;
    font-weight: 600;
    margin-bottom: 10px;
    color: var(--accent);
    word-break: break-all;
  }}
  .detail-row {{
    display: flex;
    justify-content: space-between;
    font-size: 12px;
    padding: 3px 0;
    border-bottom: 1px solid var(--border);
  }}
  .detail-row .dkey {{ color: var(--text-muted); }}
  .detail-row .dval {{ color: var(--text); max-width: 140px; text-align: right; word-break: break-word; }}
  .metric-bar-wrap {{
    margin: 8px 0 4px;
  }}
  .metric-label {{
    font-size: 11px;
    color: var(--text-muted);
    display: flex;
    justify-content: space-between;
    margin-bottom: 2px;
  }}
  .metric-bar {{
    height: 4px;
    background: var(--border);
    border-radius: 2px;
    overflow: hidden;
    margin-bottom: 6px;
  }}
  .metric-bar-fill {{ height: 100%; border-radius: 2px; }}
  #detail-text {{
    font-size: 11px;
    color: var(--text-muted);
    margin-top: 8px;
    line-height: 1.5;
    word-break: break-word;
  }}

  /* ── Bottom pathfinding bar ── */
  #pathbar {{
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 7px 12px;
    background: var(--surface);
    border-top: 1px solid var(--border);
    flex-shrink: 0;
    font-size: 12px;
  }}
  #pathbar .label {{ color: var(--text-muted); white-space: nowrap; }}
  #path-from, #path-to {{
    flex: 1;
    max-width: 200px;
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 6px;
    color: var(--text);
    padding: 4px 8px;
    font-size: 11px;
    outline: none;
  }}
  #path-result {{
    flex: 1;
    font-size: 11px;
    color: var(--text-muted);
    overflow: hidden;
    white-space: nowrap;
    text-overflow: ellipsis;
  }}
</style>
</head>
<body>

<!-- Top bar -->
<div id="topbar">
  <h1>KG</h1>
  <input id="search-input" type="text" placeholder="Search nodes…" />
  <button class="btn" id="btn-reset">Reset View</button>
  <button class="btn" id="btn-physics">Physics: OFF</button>
  <button class="btn" id="btn-clear-focus">Clear Focus</button>
  <span id="path-status"></span>
</div>

<!-- Main -->
<div id="main">

  <!-- Left sidebar -->
  <div id="sidebar">

    <!-- Legend -->
    <div>
      <div class="panel-title">Legend — Nodes</div>
      <div class="legend-item"><div class="legend-dot" style="background:#58a6ff"></div> Run (low score)</div>
      <div class="legend-item"><div class="legend-dot" style="background:#27ae60"></div> Run (high score)</div>
      <div class="legend-item"><div class="legend-dot" style="background:#00b4d8"></div> Role</div>
      <div class="legend-item"><div class="legend-dot" style="background:#9b5de5"></div> Company</div>
      <div class="legend-item"><div class="legend-dot" style="background:#06d6a0"></div> Keyword REUSE</div>
      <div class="legend-item"><div class="legend-dot" style="background:#ffd166"></div> Keyword ADAPT</div>
      <div class="legend-item"><div class="legend-dot" style="background:#ef476f"></div> Keyword NEW</div>
      <div class="legend-item"><div class="legend-dot" style="background:#f9c74f"></div> Bullet ★ promoted</div>
      <div class="legend-item"><div class="legend-dot" style="background:#8d99ae"></div> Bullet</div>
    </div>

    <div>
      <div class="panel-title">Legend — Edges</div>
      <div class="legend-item">
        <div class="legend-line" style="background:#06d6a0"></div> REUSE (solid)
      </div>
      <div class="legend-item">
        <div class="legend-line" style="background:#ffd166; background: repeating-linear-gradient(90deg,#ffd166 0,#ffd166 8px,transparent 8px,transparent 12px)"></div> ADAPT (dashed)
      </div>
      <div class="legend-item">
        <div class="legend-line" style="background: repeating-linear-gradient(90deg,#ef476f 0,#ef476f 3px,transparent 3px,transparent 6px)"></div> NEW (dotted)
      </div>
      <div class="legend-item">
        <div class="legend-line" style="background:#00b4d8"></div> Structural
      </div>
    </div>

    <!-- Filters -->
    <div>
      <div class="panel-title">Filter — Node Type</div>
      <div class="filter-group" id="type-filters">
        <label><input type="checkbox" checked value="run"> Run</label>
        <label><input type="checkbox" checked value="role"> Role</label>
        <label><input type="checkbox" checked value="company"> Company</label>
        <label><input type="checkbox" checked value="keyword"> Keyword</label>
        <label><input type="checkbox" checked value="bullet"> Bullet</label>
      </div>
    </div>

    <div>
      <div class="panel-title">Filter — Decision</div>
      <div class="filter-group" id="decision-filters">
        <label><input type="checkbox" checked value="REUSE"> REUSE</label>
        <label><input type="checkbox" checked value="ADAPT"> ADAPT</label>
        <label><input type="checkbox" checked value="NEW"> NEW</label>
        <label><input type="checkbox" checked value=""> (n/a)</label>
      </div>
    </div>

    <!-- Stats -->
    <div id="stats-panel">
      <div class="panel-title">Stats</div>
    </div>

  </div>

  <!-- Graph canvas -->
  <div id="network-wrap">
    <div id="network"></div>
  </div>

  <!-- Right detail panel -->
  <div id="detail-panel">
    <span id="detail-close">✕</span>
    <div id="detail-title">—</div>
    <div id="detail-body"></div>
    <div id="detail-text"></div>
  </div>

</div>

<!-- Bottom pathfinding bar -->
<div id="pathbar">
  <span class="label">Path:</span>
  <input id="path-from" type="text" placeholder="From node ID…" />
  <span class="label">→</span>
  <input id="path-to" type="text" placeholder="To node ID…" />
  <button class="btn" id="btn-find-path">Find</button>
  <button class="btn" id="btn-clear-path">Clear</button>
  <span id="path-result">Click two nodes to set endpoints, then Find</span>
</div>

<script>
// ── Data ─────────────────────────────────────────────────────────────────────
const RAW_NODES = {nodes_json};
const RAW_EDGES = {edges_json};
const STATS = {stats_json};

// ── Build DataSets ───────────────────────────────────────────────────────────
const allNodeData = new Map();
RAW_NODES.forEach(n => allNodeData.set(n.id, n));

const nodesDS = new vis.DataSet(RAW_NODES);
const edgesDS = new vis.DataSet(RAW_EDGES);

// ── Network ──────────────────────────────────────────────────────────────────
const container = document.getElementById('network');
const options = {{
  nodes: {{
    shape: 'dot',
    borderWidth: 1.5,
    chosen: {{
      node: (values, id, selected, hovering) => {{
        values.shadowSize = 8;
        values.shadowColor = 'rgba(88,166,255,0.5)';
      }}
    }}
  }},
  edges: {{
    smooth: {{ type: 'continuous', roundness: 0.2 }},
    arrows: {{ to: {{ enabled: false }} }},
    hoverWidth: 0.5,
  }},
  physics: {{
    enabled: false,
  }},
  interaction: {{
    hover: true,
    tooltipDelay: 100,
    hideEdgesOnDrag: true,
  }},
}};

const network = new vis.Network(container, {{ nodes: nodesDS, edges: edgesDS }}, options);

// ── Colour snapshots for dim/highlight ───────────────────────────────────────
const originalNodeColors = new Map();
RAW_NODES.forEach(n => originalNodeColors.set(n.id, n.color));
const originalEdgeColors = new Map();
RAW_EDGES.forEach(e => originalEdgeColors.set(e.id, e.color));

// ── State ────────────────────────────────────────────────────────────────────
let physicsOn = false;
let focusedNode = null;
let pathNodeA = null, pathNodeB = null;
let pathHighlighted = [];

// ── Adjacency (for BFS + neighbourhood) ─────────────────────────────────────
const adjacency = new Map();
RAW_NODES.forEach(n => adjacency.set(n.id, new Set()));
RAW_EDGES.forEach(e => {{
  if (adjacency.has(e.from)) adjacency.get(e.from).add(e.to);
  if (adjacency.has(e.to))   adjacency.get(e.to).add(e.from);
}});

// ── Helpers ──────────────────────────────────────────────────────────────────
function dimAll() {{
  const dimNode = (id) => {{
    const orig = originalNodeColors.get(id);
    const bg = orig?.background || orig || '#333';
    nodesDS.update({{ id, color: {{ background: bg + '33', border: bg + '33', highlight: orig }} }});
  }};
  const dimEdge = (id) => {{
    const orig = originalEdgeColors.get(id);
    edgesDS.update({{ id, color: {{ color: (orig?.color || '#555') + '22', opacity: 0.1 }} }});
  }};
  nodesDS.getIds().forEach(dimNode);
  edgesDS.getIds().forEach(dimEdge);
}}

function restoreAll() {{
  const updates = [];
  originalNodeColors.forEach((color, id) => updates.push({{ id, color }}));
  nodesDS.update(updates);
  const eupdates = [];
  originalEdgeColors.forEach((color, id) => eupdates.push({{ id, color }}));
  edgesDS.update(eupdates);
}}

function highlightNeighbourhood(nodeId) {{
  dimAll();
  const neighbours = adjacency.get(nodeId) || new Set();
  const allFocused = new Set([nodeId, ...neighbours]);

  // Restore focused nodes
  allFocused.forEach(id => {{
    const orig = originalNodeColors.get(id);
    if (orig) nodesDS.update({{ id, color: orig }});
  }});

  // Restore connecting edges
  RAW_EDGES.forEach(e => {{
    if (allFocused.has(e.from) && allFocused.has(e.to)) {{
      const orig = originalEdgeColors.get(e.id);
      edgesDS.update({{ id: e.id, color: orig || e.color }});
    }}
  }});
}}

// ── Hover tooltip (detail panel) ─────────────────────────────────────────────
network.on('hoverNode', (params) => {{
  const nd = allNodeData.get(params.node);
  if (!nd) return;
  showDetail(nd);
}});

function showDetail(nd) {{
  const panel = document.getElementById('detail-panel');
  const title = document.getElementById('detail-title');
  const body = document.getElementById('detail-body');
  const textEl = document.getElementById('detail-text');
  const d = nd._detail;
  const m = nd._metrics;

  panel.classList.add('visible');
  title.textContent = nd.label || nd.id;

  let rows = '';
  const row = (k, v) => `<div class="detail-row"><span class="dkey">${{k}}</span><span class="dval">${{v}}</span></div>`;

  rows += row('Type', d.type);

  if (d.type === 'Run') {{
    rows += row('Role', d.role);
    rows += row('Company', d.company);
    rows += row('Date', d.date);
    rows += row('Score', d.score.toFixed(2) + (d.passed ? ' ✓' : ' ✗'));
  }} else if (d.type === 'Role' || d.type === 'Company') {{
    rows += row('Runs', d.run_count);
  }} else if (d.type === 'Keyword') {{
    rows += row('Domain', d.domain);
    rows += row('Last decision', d.last_decision);
    rows += row('REUSE / ADAPT / NEW', `${{d.reuse_count}} / ${{d.adapt_count}} / ${{d.new_count}}`);
    rows += row('Total runs', d.run_count);
  }} else if (d.type === 'Bullet') {{
    rows += row('Source', d.source_file);
    rows += row('Used', d.use_count + '×');
    rows += row('Avg score', d.avg_source_score.toFixed(2));
    rows += row('Promoted', d.promoted ? '★ yes' : 'no');
  }}

  rows += row('Degree', m.degree);
  rows += row('Degree centrality', m.degree_centrality.toFixed(3));
  rows += row('PageRank', m.pagerank.toFixed(5));

  body.innerHTML = rows;

  // Metric bars
  const dcPct = Math.round(m.degree_centrality * 100);
  const prPct = Math.min(100, Math.round(m.pagerank * 50000));
  body.innerHTML += `
    <div class="metric-bar-wrap">
      <div class="metric-label"><span>Degree centrality</span><span>${{dcPct}}%</span></div>
      <div class="metric-bar"><div class="metric-bar-fill" style="width:${{dcPct}}%;background:#58a6ff"></div></div>
      <div class="metric-label"><span>PageRank (rel)</span><span>${{prPct}}%</span></div>
      <div class="metric-bar"><div class="metric-bar-fill" style="width:${{prPct}}%;background:#3fb950"></div></div>
    </div>`;

  if (d.type === 'Bullet' && d.text) {{
    textEl.textContent = d.text.length > 200 ? d.text.slice(0, 200) + '…' : d.text;
  }} else {{
    textEl.textContent = '';
  }}
}}

// ── Click: focus + pathfinding endpoint selection ────────────────────────────
network.on('click', (params) => {{
  if (params.nodes.length === 0) {{
    // Clicked canvas — but don't clear focus automatically
    return;
  }}
  const nodeId = params.nodes[0];
  const nd = allNodeData.get(nodeId);
  if (nd) showDetail(nd);

  // Focus/neighbourhood highlight
  if (focusedNode === nodeId) {{
    clearFocus();
  }} else {{
    focusedNode = nodeId;
    highlightNeighbourhood(nodeId);
  }}

  // Pathfinding endpoint selection
  if (pathNodeA === null) {{
    pathNodeA = nodeId;
    document.getElementById('path-from').value = nodeId;
    document.getElementById('path-result').textContent = `A: ${{nodeId}} — now click or type B`;
  }} else if (pathNodeB === null && nodeId !== pathNodeA) {{
    pathNodeB = nodeId;
    document.getElementById('path-to').value = nodeId;
    document.getElementById('path-result').textContent = `A→B set — click Find`;
  }}
}});

function clearFocus() {{
  focusedNode = null;
  restoreAll();
  if (pathHighlighted.length === 0) {{
    document.getElementById('path-status').textContent = '';
  }}
}}

document.getElementById('btn-clear-focus').addEventListener('click', () => {{
  clearFocus();
}});

// ── Search ───────────────────────────────────────────────────────────────────
document.getElementById('search-input').addEventListener('input', (e) => {{
  const q = e.target.value.trim().toLowerCase();
  if (!q) {{ restoreAll(); return; }}

  const matched = [];
  RAW_NODES.forEach(n => {{
    const label = (n.label || '').toLowerCase();
    const id = n.id.toLowerCase();
    if (label.includes(q) || id.includes(q)) matched.push(n.id);
  }});

  if (matched.length === 0) {{ dimAll(); return; }}

  dimAll();
  matched.forEach(id => {{
    const orig = originalNodeColors.get(id);
    if (orig) nodesDS.update({{ id, color: orig }});
  }});

  network.selectNodes(matched);
  network.fit({{ nodes: matched, animation: {{ duration: 400, easingFunction: 'easeInOutQuad' }} }});
}});

// ── Physics toggle ───────────────────────────────────────────────────────────
document.getElementById('btn-physics').addEventListener('click', () => {{
  physicsOn = !physicsOn;
  if (physicsOn) {{
    network.setOptions({{ physics: {{
      enabled: true,
      forceAtlas2Based: {{ gravitationalConstant: -50, centralGravity: 0.01, springLength: 120, springConstant: 0.08, damping: 0.4, avoidOverlap: 0.3 }},
      solver: 'forceAtlas2Based',
      stabilization: {{ enabled: false }},
    }} }});
  }} else {{
    network.setOptions({{ physics: {{ enabled: false }} }});
  }}
  document.getElementById('btn-physics').textContent = physicsOn ? 'Physics: ON' : 'Physics: OFF';
}});

// ── Reset View ───────────────────────────────────────────────────────────────
document.getElementById('btn-reset').addEventListener('click', () => {{
  network.fit({{ animation: {{ duration: 500, easingFunction: 'easeInOutQuad' }} }});
  clearFocus();
}});

// ── Filters (use hidden property — avoids DataSet rebuild) ───────────────────
function applyFilters() {{
  const activeTypes = new Set(
    [...document.querySelectorAll('#type-filters input:checked')].map(i => i.value)
  );
  const activeDecisions = new Set(
    [...document.querySelectorAll('#decision-filters input:checked')].map(i => i.value)
  );

  const nodeUpdates = [];
  RAW_NODES.forEach(n => {{
    const t = n._type || '';
    let visible = activeTypes.has(t);
    if (visible && t === 'keyword') {{
      const last = n._attrs?.last_decision || '';
      if (!activeDecisions.has(last)) visible = false;
    }}
    nodeUpdates.push({{ id: n.id, hidden: !visible }});
  }});
  nodesDS.update(nodeUpdates);

  const edgeUpdates = [];
  const visibleIds = new Set(nodeUpdates.filter(u => !u.hidden).map(u => u.id));
  RAW_EDGES.forEach(e => {{
    edgeUpdates.push({{ id: e.id, hidden: !(visibleIds.has(e.from) && visibleIds.has(e.to)) }});
  }});
  edgesDS.update(edgeUpdates);
}}

document.querySelectorAll('#type-filters input, #decision-filters input').forEach(cb => {{
  cb.addEventListener('change', applyFilters);
}});

// ── BFS Pathfinding ──────────────────────────────────────────────────────────
function bfs(startId, endId) {{
  if (startId === endId) return [startId];
  const visited = new Set([startId]);
  const queue = [[startId]];
  while (queue.length > 0) {{
    const path = queue.shift();
    const node = path[path.length - 1];
    const nbrs = adjacency.get(node) || new Set();
    for (const nb of nbrs) {{
      if (!visited.has(nb)) {{
        const newPath = [...path, nb];
        if (nb === endId) return newPath;
        visited.add(nb);
        queue.push(newPath);
      }}
    }}
  }}
  return null;
}}

function highlightPath(path) {{
  // Clear previous path highlight
  clearPathHighlight();

  const pathSet = new Set(path);
  dimAll();

  // Highlight path nodes
  path.forEach(id => {{
    const orig = originalNodeColors.get(id);
    if (orig) nodesDS.update({{ id, color: orig }});
  }});

  // Highlight path edges
  const pathEdgeIds = [];
  for (let i = 0; i < path.length - 1; i++) {{
    const a = path[i], b = path[i + 1];
    RAW_EDGES.forEach(e => {{
      if ((e.from === a && e.to === b) || (e.from === b && e.to === a)) {{
        const orig = originalEdgeColors.get(e.id);
        edgesDS.update({{ id: e.id, color: {{ color: '#58a6ff', opacity: 1.0 }} }});
        pathEdgeIds.push(e.id);
      }}
    }});
  }}
  pathHighlighted = pathEdgeIds;

  network.fit({{ nodes: path, animation: {{ duration: 500, easingFunction: 'easeInOutQuad' }} }});
}}

function clearPathHighlight() {{
  pathHighlighted.forEach(id => {{
    const orig = originalEdgeColors.get(id);
    if (orig) edgesDS.update({{ id, color: orig }});
  }});
  pathHighlighted = [];
}}

document.getElementById('btn-find-path').addEventListener('click', () => {{
  const fromVal = document.getElementById('path-from').value.trim() || pathNodeA;
  const toVal = document.getElementById('path-to').value.trim() || pathNodeB;
  if (!fromVal || !toVal) {{
    document.getElementById('path-result').textContent = 'Set both endpoints first.';
    return;
  }}
  if (!adjacency.has(fromVal) || !adjacency.has(toVal)) {{
    document.getElementById('path-result').textContent = 'One or both node IDs not found.';
    return;
  }}
  const path = bfs(fromVal, toVal);
  if (!path) {{
    document.getElementById('path-result').textContent = `No path from ${{fromVal}} to ${{toVal}}`;
    clearPathHighlight();
    return;
  }}
  document.getElementById('path-result').textContent =
    `Path (${{path.length - 1}} hops): ${{path.join(' → ')}}`;
  highlightPath(path);
}});

document.getElementById('btn-clear-path').addEventListener('click', () => {{
  pathNodeA = null; pathNodeB = null;
  document.getElementById('path-from').value = '';
  document.getElementById('path-to').value = '';
  document.getElementById('path-result').textContent = 'Click two nodes to set endpoints, then Find';
  clearPathHighlight();
  restoreAll();
}});

// ── Close detail panel ───────────────────────────────────────────────────────
document.getElementById('detail-close').addEventListener('click', () => {{
  document.getElementById('detail-panel').classList.remove('visible');
}});

// ── Populate stats panel ─────────────────────────────────────────────────────
(function populateStats() {{
  const panel = document.getElementById('stats-panel');
  let html = '<div class="panel-title">Stats</div>';
  html += `<div class="stat-row"><span>Total nodes</span><span class="val">${{STATS.node_count}}</span></div>`;
  Object.entries(STATS.type_counts).forEach(([t, c]) => {{
    html += `<div class="stat-row"><span>${{t}}</span><span class="val">${{c}}</span></div>`;
  }});
  html += '<div style="margin-top:8px;"><div class="panel-title">Top Degree Centrality</div>';
  html += '<ul class="top-list">';
  STATS.top_degree_centrality.forEach(item => {{
    const label = allNodeData.get(item.id)?._detail?.text || allNodeData.get(item.id)?.label || item.id;
    html += `<li><span>${{label.length > 16 ? label.slice(0, 14) + '…' : label}}</span><span>${{item.dc}}</span></li>`;
  }});
  html += '</ul></div>';
  html += '<div style="margin-top:4px;"><div class="panel-title">Top PageRank</div>';
  html += '<ul class="top-list">';
  STATS.top_pagerank.forEach(item => {{
    const label = allNodeData.get(item.id)?._detail?.text || allNodeData.get(item.id)?.label || item.id;
    html += `<li><span>${{label.length > 16 ? label.slice(0, 14) + '…' : label}}</span><span>${{item.pr}}</span></li>`;
  }});
  html += '</ul></div>';
  panel.innerHTML = html;
}})();
</script>
</body>
</html>"""


# ── Public API ───────────────────────────────────────────────────────────────

def generate_html(store, vault_path: str) -> pathlib.Path:
    """Build and save graph.html from the KGStore using vis.js. Returns the output path."""
    nodes: dict = store._data["nodes"]
    edges: list = store._data["edges"]

    metrics = _compute_metrics(nodes, edges)
    layout = _compute_layout(nodes)
    vis_nodes = _build_vis_nodes(nodes, metrics, layout)
    vis_edges = _build_vis_edges(edges, nodes)
    stats = _build_stats(nodes, metrics)

    html = _render_html(vis_nodes, vis_edges, stats)

    out_path = pathlib.Path(vault_path) / "KnowledgeGraph" / _OUTPUT_FILENAME
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    logger.info("KG visualization saved: %s", out_path)
    return out_path
