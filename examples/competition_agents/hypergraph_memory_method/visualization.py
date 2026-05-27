from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

from .builder import ensure_hypergraph_view
from .index import query_nodes
from .scoring import node_kind


NODE_LABELS = {
    "memory": "记忆记录",
    "task": "任务",
    "group": "连续任务组",
    "turn": "轮次",
    "dataset": "数据集",
    "agent": "智能体",
    "action": "动作",
    "message": "消息",
    "tag": "标签",
    "keyword": "关键词",
    "expected_reuse": "期望复用",
    "artifact": "产物类型",
    "state": "状态角色",
    "state_ref": "状态引用",
    "memory_ref": "记忆引用",
    "gap": "认知缺口",
    "supporting_fact": "支撑事实",
    "evidence": "证据片段",
    "entity": "实体",
    "metric": "指标",
    "outcome": "结果",
    "execution": "执行字段",
    "tool": "工具",
    "capability": "能力",
    "intent": "查询意图",
    "strategy": "协作策略",
    "experience": "可复用经验",
    "reuse_benefit": "复用收益",
    "archive_scope": "归档范围",
    "lifecycle": "生命周期",
    "hyperedge": "超边",
    "query": "查询节点",
}

EDGE_LABELS = {
    "memory_event": "主记忆事件：把一条记忆的关键节点连在一起",
    "task_context": "任务上下文：任务组、轮次、数据集和期望复用关系",
    "agent_message": "智能体消息：谁把什么状态/记忆引用传给谁",
    "state_flow": "状态流：plan、evidence、answer 等状态引用",
    "memory_reuse": "记忆复用：当前记忆复用了哪些历史记忆",
    "cognitive_gap": "认知缺口：当时智能体缺什么、需要什么",
    "evidence": "证据：本轮保存的证据片段",
    "supporting_fact": "支撑事实：多跳问答中的证据链实体和事实",
    "outcome": "运行结果：检索模式、执行模式、通信指标等",
    "tool_execution": "工具执行：未来 executor 的工具调用轨迹",
    "sandbox_outcome": "沙箱结果：未来 sandbox 的执行结果",
    "strategy": "协作策略：本轮采用的协作方式",
    "experience": "经验：从本轮轨迹蒸馏出的可复用经验",
    "reuse_benefit": "复用收益：命中、通信、latent 传递等收益信息",
    "archive_scope": "归档范围：这条记忆适合在哪个范围复用",
}

NODE_COLORS = {
    "memory": "#2e7d32",
    "task": "#1565c0",
    "group": "#1976d2",
    "turn": "#42a5f5",
    "dataset": "#90caf9",
    "agent": "#6a1b9a",
    "action": "#8e24aa",
    "message": "#ab47bc",
    "tag": "#546e7a",
    "keyword": "#90a4ae",
    "expected_reuse": "#ef6c00",
    "artifact": "#f57c00",
    "state": "#00838f",
    "state_ref": "#00acc1",
    "memory_ref": "#558b2f",
    "gap": "#c62828",
    "supporting_fact": "#e65100",
    "evidence": "#ff8f00",
    "entity": "#fb8c00",
    "metric": "#5d4037",
    "outcome": "#795548",
    "execution": "#6d4c41",
    "tool": "#3e2723",
    "capability": "#7b1fa2",
    "intent": "#283593",
    "strategy": "#ad8b00",
    "experience": "#c0a000",
    "reuse_benefit": "#d4a017",
    "archive_scope": "#616161",
    "lifecycle": "#757575",
    "hyperedge": "#212121",
    "query": "#d81b60",
}


def _split_node(node_id: str) -> tuple[str, str]:
    kind, _, value = node_id.partition(":")
    return kind or "unknown", value or node_id


def chinese_node_description(node_id: str) -> str:
    kind, value = _split_node(node_id)
    kind_label = NODE_LABELS.get(kind, kind)
    value_label = value.replace("-", " ")
    return f"{kind_label}：{value_label}"


def _short(value: Any, limit: int = 140) -> str:
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str)
    return text if len(text) <= limit else text[:limit] + "..."


def _node_payload(node_id: str, *, highlight: bool = False) -> dict[str, Any]:
    kind, value = _split_node(node_id)
    return {
        "id": node_id,
        "kind": kind,
        "value": value,
        "label": value[:42],
        "cn_label": NODE_LABELS.get(kind, kind),
        "description": chinese_node_description(node_id),
        "color": NODE_COLORS.get(kind, "#607d8b"),
        "highlight": highlight,
    }


def build_memory_graph(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Convert real memory records into a bipartite hypergraph visualization model."""

    nodes: dict[str, dict[str, Any]] = {}
    links: list[dict[str, Any]] = []
    hyperedges: list[dict[str, Any]] = []

    for record in records:
        memory_id = record.get("memory_id", "")
        hypergraph = ensure_hypergraph_view(record)
        for hyperedge in _all_hyperedges(hypergraph):
            edge_id = str(hyperedge.get("hyperedge_id", ""))
            edge_type = str(hyperedge.get("type", "hyperedge"))
            if not edge_id:
                continue
            nodes.setdefault(
                edge_id,
                {
                    "id": edge_id,
                    "kind": "hyperedge",
                    "value": edge_id,
                    "label": edge_id.replace("hyperedge:", ""),
                    "cn_label": "超边",
                    "description": EDGE_LABELS.get(edge_type, f"超边：{edge_type}"),
                    "edge_type": edge_type,
                    "color": NODE_COLORS["hyperedge"],
                    "shape": "diamond",
                },
            )
            hyperedges.append(
                {
                    "id": edge_id,
                    "type": edge_type,
                    "description": EDGE_LABELS.get(edge_type, edge_type),
                    "memory_id": memory_id,
                    "nodes": list(hyperedge.get("nodes", [])),
                }
            )
            for raw_node in hyperedge.get("nodes", []):
                node = str(raw_node)
                nodes.setdefault(node, _node_payload(node))
                links.append({"source": node, "target": edge_id, "edge_type": edge_type})

    return {
        "nodes": list(nodes.values()),
        "links": _dedupe_links(links),
        "hyperedges": hyperedges,
    }


def build_retrieval_graph(records: list[dict[str, Any]], query: str, hits: list[dict[str, Any]]) -> dict[str, Any]:
    """Build the real retrieval path graph for one query and its memory hits."""

    base_graph = build_memory_graph(records)
    query_node_ids = sorted(query_nodes(query))
    hit_memory_ids = {hit.get("memory_id") for hit in hits}
    matched_nodes = {node for hit in hits for node in hit.get("matched_nodes", [])}
    matched_edges = {edge for hit in hits for edge in hit.get("matched_hyperedges", [])}

    nodes_by_id = {node["id"]: dict(node) for node in base_graph["nodes"]}
    links = list(base_graph["links"])

    for node_id in query_node_ids:
        nodes_by_id.setdefault(node_id, _node_payload(node_id))
        nodes_by_id[node_id]["is_query_node"] = True
        if node_id in matched_nodes:
            nodes_by_id[node_id]["highlight"] = True

    for node_id, node in nodes_by_id.items():
        kind = node.get("kind") or node_kind(node_id)
        if node_id in matched_nodes:
            node["highlight"] = True
        if node_id in matched_edges:
            node["highlight"] = True
        if kind == "memory" and node_id.replace("memory:", "") in hit_memory_ids:
            node["highlight"] = True
            node["is_hit_memory"] = True

    for hit in hits:
        for path in hit.get("reasoning_path", []):
            query_node = str(path.get("query_node", ""))
            hyperedge_id = str(path.get("hyperedge_id", ""))
            memory_id = str(path.get("memory_id", ""))
            memory_node = f"memory:{memory_id}"
            if query_node and hyperedge_id:
                links.append(
                    {
                        "source": query_node,
                        "target": hyperedge_id,
                        "edge_type": "retrieval_path",
                        "highlight": True,
                    }
                )
            if hyperedge_id and memory_id:
                links.append(
                    {
                        "source": hyperedge_id,
                        "target": memory_node,
                        "edge_type": "retrieval_path",
                        "highlight": True,
                    }
                )

    return {
        "query": query,
        "query_nodes": query_node_ids,
        "hits": _summarize_hits(hits),
        "nodes": list(nodes_by_id.values()),
        "links": _dedupe_links(links),
        "hyperedges": base_graph["hyperedges"],
    }


def export_visualization(
    *,
    records: list[dict[str, Any]],
    retrievals: list[dict[str, Any]],
    out_dir: Path,
    title: str,
) -> dict[str, Path]:
    """Write JSON and HTML visualization files from real run data."""

    out_dir.mkdir(parents=True, exist_ok=True)
    graph = build_memory_graph(records)
    retrieval_graphs = [
        build_retrieval_graph(records, retrieval["query"], retrieval["hits"])
        for retrieval in retrievals
        if retrieval.get("hits")
    ]
    payload = {
        "title": title,
        "memory_count": len(records),
        "memory_graph": graph,
        "retrieval_graphs": retrieval_graphs,
        "records": [_summarize_record(record) for record in records],
    }

    json_path = out_dir / "memory_graph.json"
    html_path = out_dir / "memory_graph.html"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    html_path.write_text(render_html(payload), encoding="utf-8")
    return {"json": json_path, "html": html_path}


def render_html(payload: dict[str, Any]) -> str:
    data = json.dumps(payload, ensure_ascii=False)
    title = html.escape(str(payload.get("title", "Hypergraph Memory Visualization")))
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <title>{title}</title>
  <style>
    body {{ margin: 0; font-family: Arial, "Microsoft YaHei", sans-serif; background: #f7f8fa; color: #1f2933; }}
    header {{ padding: 16px 22px; background: #111827; color: white; }}
    header h1 {{ margin: 0 0 6px; font-size: 20px; }}
    header p {{ margin: 0; color: #d1d5db; font-size: 13px; }}
    main {{ display: grid; grid-template-columns: 330px 1fr; height: calc(100vh - 72px); }}
    aside {{ overflow: auto; padding: 14px; border-right: 1px solid #d7dde5; background: white; }}
    section {{ overflow: auto; padding: 16px; }}
    .panel {{ border: 1px solid #d7dde5; background: white; border-radius: 6px; padding: 12px; margin-bottom: 12px; }}
    .panel h2 {{ margin: 0 0 10px; font-size: 15px; }}
    .btn {{ display: block; width: 100%; text-align: left; border: 1px solid #c9d2df; background: #fff; border-radius: 5px; padding: 8px; margin: 6px 0; cursor: pointer; }}
    .btn.active {{ border-color: #2563eb; background: #eff6ff; }}
    .legend {{ display: grid; grid-template-columns: 14px 1fr; gap: 6px 8px; font-size: 12px; }}
    .swatch {{ width: 12px; height: 12px; border-radius: 2px; margin-top: 2px; }}
    .meta {{ font-size: 12px; line-height: 1.6; color: #4b5563; }}
    .canvas-wrap {{ background: white; border: 1px solid #d7dde5; border-radius: 6px; overflow: auto; }}
    svg {{ display: block; min-width: 100%; background: #ffffff; }}
    .link {{ stroke: #a7b0bd; stroke-width: 1.1; opacity: 0.75; }}
    .link.highlight {{ stroke: #dc2626; stroke-width: 2.4; opacity: 1; }}
    .node circle {{ stroke: #fff; stroke-width: 1.5; }}
    .node polygon {{ stroke: #fff; stroke-width: 1.5; }}
    .node.highlight circle, .node.highlight polygon {{ stroke: #dc2626; stroke-width: 3; }}
    .node text {{ font-size: 10px; fill: #111827; paint-order: stroke; stroke: white; stroke-width: 3px; stroke-linejoin: round; }}
    .tooltip {{ position: fixed; pointer-events: none; background: #111827; color: white; padding: 8px 10px; border-radius: 5px; font-size: 12px; max-width: 360px; display: none; z-index: 10; }}
    pre {{ white-space: pre-wrap; word-break: break-word; font-size: 12px; background: #f3f4f6; border-radius: 5px; padding: 8px; }}
  </style>
</head>
<body>
  <header>
    <h1>{title}</h1>
    <p>真实运行数据生成：圆点是普通节点，黑色菱形是超边；红色描边表示检索命中的节点或路径。</p>
  </header>
  <main>
    <aside>
      <div class="panel">
        <h2>视图</h2>
        <button class="btn active" id="memoryBtn">整体记忆超图</button>
        <div id="retrievalButtons"></div>
      </div>
      <div class="panel">
        <h2>图例</h2>
        <div class="legend" id="legend"></div>
      </div>
      <div class="panel">
        <h2>当前视图说明</h2>
        <div class="meta" id="viewMeta"></div>
      </div>
      <div class="panel">
        <h2>命中结果</h2>
        <pre id="hitInfo">整体图无检索命中。</pre>
      </div>
    </aside>
    <section>
      <div class="canvas-wrap">
        <svg id="graph" width="1400" height="920"></svg>
      </div>
    </section>
  </main>
  <div class="tooltip" id="tooltip"></div>
  <script>
    const DATA = {data};
    const COLORS = {json.dumps(NODE_COLORS, ensure_ascii=False)};
    const LABELS = {json.dumps(NODE_LABELS, ensure_ascii=False)};
    const svg = document.getElementById("graph");
    const tooltip = document.getElementById("tooltip");
    const retrievalButtons = document.getElementById("retrievalButtons");
    const viewMeta = document.getElementById("viewMeta");
    const hitInfo = document.getElementById("hitInfo");
    const memoryBtn = document.getElementById("memoryBtn");
    let currentButton = memoryBtn;

    function setActive(btn) {{
      currentButton.classList.remove("active");
      btn.classList.add("active");
      currentButton = btn;
    }}

    const EDGE_COLUMNS = {{
      memory_event: 150,
      task_context: 300,
      agent_message: 480,
      state_flow: 660,
      evidence: 840,
      supporting_fact: 940,
      cognitive_gap: 1100,
      outcome: 1240,
      strategy: 1400,
      experience: 1560,
      reuse_benefit: 1720,
      archive_scope: 1880,
      tool_execution: 1240,
      sandbox_outcome: 1240,
      retrieval_path: 1000,
    }};
    const KIND_COLUMNS = {{
      memory: 70,
      task: 230,
      group: 230,
      turn: 230,
      dataset: 230,
      expected_reuse: 300,
      agent: 420,
      action: 540,
      message: 480,
      capability: 560,
      state: 620,
      state_ref: 710,
      evidence: 820,
      supporting_fact: 910,
      entity: 1010,
      gap: 1100,
      metric: 1210,
      outcome: 1270,
      execution: 1270,
      tool: 1270,
      strategy: 1380,
      experience: 1550,
      reuse_benefit: 1710,
      archive_scope: 1860,
      lifecycle: 1960,
      artifact: 2050,
      tag: 2180,
      keyword: 2320,
      intent: 90,
      memory_ref: 170,
    }};

    function memoryIdFromText(text) {{
      const m = String(text || "").match(/mem-\\d+/);
      return m ? m[0] : "";
    }}

    function edgeMemoryId(edgeId) {{
      return memoryIdFromText(edgeId);
    }}

    function buildEdgeMap(nodes) {{
      const map = new Map();
      nodes.forEach(n => {{
        if (n.kind === "hyperedge") map.set(n.id, n);
      }});
      return map;
    }}

    function inferNodeMemory(node, links, edgeMap) {{
      if (node.kind === "memory") return node.value || memoryIdFromText(node.id);
      const counts = new Map();
      links.forEach(l => {{
        let edge = null;
        if (l.source === node.id && edgeMap.has(l.target)) edge = edgeMap.get(l.target);
        if (l.target === node.id && edgeMap.has(l.source)) edge = edgeMap.get(l.source);
        if (!edge) return;
        const mid = edgeMemoryId(edge.id);
        if (!mid) return;
        counts.set(mid, (counts.get(mid) || 0) + 1);
      }});
      let best = "";
      let bestCount = -1;
      counts.forEach((count, mid) => {{
        if (count > bestCount) {{
          best = mid;
          bestCount = count;
        }}
      }});
      return best;
    }}

    function spreadBucket(items, baseY, step, maxPerColumn = 10) {{
      items.forEach((n, i) => {{
        const band = Math.floor(i / maxPerColumn);
        const offset = i % maxPerColumn;
        n.y = baseY + offset * step;
        n.x += band * 46;
      }});
    }}

    function layout(nodes, links) {{
      const width = 2500;
      const edgeMap = buildEdgeMap(nodes);
      const memoryIds = Array.from(new Set(nodes
        .filter(n => n.kind === "memory")
        .map(n => n.value || memoryIdFromText(n.id))
        .filter(Boolean)
      )).sort();
      const rowHeight = 360;
      const rowBase = new Map();
      memoryIds.forEach((mid, idx) => rowBase.set(mid, 95 + idx * rowHeight));
      const fallbackBase = 95 + memoryIds.length * rowHeight;

      const buckets = new Map();
      function addBucket(key, node) {{
        if (!buckets.has(key)) buckets.set(key, []);
        buckets.get(key).push(node);
      }}

      nodes.forEach(n => {{
        if (n.kind === "hyperedge") {{
          const mid = edgeMemoryId(n.id);
          const baseY = rowBase.get(mid) || fallbackBase;
          const edgeType = n.edge_type || "memory_event";
          n.x = EDGE_COLUMNS[edgeType] || 1200;
          addBucket(`${{mid}}:edge:${{edgeType}}`, n);
          n.rowBase = baseY;
          return;
        }}
        const mid = inferNodeMemory(n, links, edgeMap);
        const baseY = rowBase.get(mid) || fallbackBase;
        n.x = KIND_COLUMNS[n.kind] || 2260;
        if (n.is_query_node) n.x = Math.min(n.x, 120);
        addBucket(`${{mid || "shared"}}:node:${{n.kind}}:${{n.x}}`, n);
        n.rowBase = baseY;
      }});

      buckets.forEach(items => {{
        const baseY = items[0].rowBase;
        const isEdge = items[0].kind === "hyperedge";
        spreadBucket(items, baseY + (isEdge ? 0 : 38), isEdge ? 42 : 28, isEdge ? 8 : 9);
      }});

      svg.setAttribute("width", width);
      svg.setAttribute("height", Math.max(920, fallbackBase + 260));
    }}

    function render(graph, mode) {{
      svg.innerHTML = "";
      const nodes = graph.nodes.map(n => ({{...n}}));
      const rawLinks = graph.links || [];
      const nodeById = new Map(nodes.map(n => [n.id, n]));
      const links = rawLinks.filter(l => {{
        if (l.highlight) return true;
        if (l.edge_type !== "memory_event") return true;
        const source = nodeById.get(l.source);
        const target = nodeById.get(l.target);
        return (source && source.kind === "memory") || (target && target.kind === "memory");
      }});
      layout(nodes, links);

      const linkLayer = document.createElementNS("http://www.w3.org/2000/svg", "g");
      svg.appendChild(linkLayer);
      links.forEach(l => {{
        const s = nodeById.get(l.source);
        const t = nodeById.get(l.target);
        if (!s || !t) return;
        const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
        line.setAttribute("x1", s.x);
        line.setAttribute("y1", s.y);
        line.setAttribute("x2", t.x);
        line.setAttribute("y2", t.y);
        line.setAttribute("class", "link" + (l.highlight ? " highlight" : ""));
        if (l.edge_type === "memory_event") {{
          line.setAttribute("stroke-dasharray", "4 4");
          line.setAttribute("opacity", "0.28");
        }}
        linkLayer.appendChild(line);
      }});

      const nodeLayer = document.createElementNS("http://www.w3.org/2000/svg", "g");
      svg.appendChild(nodeLayer);
      nodes.forEach(n => {{
        const g = document.createElementNS("http://www.w3.org/2000/svg", "g");
        g.setAttribute("class", "node" + (n.highlight ? " highlight" : ""));
        g.setAttribute("transform", `translate(${{n.x}},${{n.y}})`);
        const color = n.color || COLORS[n.kind] || "#607d8b";
        if (n.kind === "hyperedge") {{
          const p = document.createElementNS("http://www.w3.org/2000/svg", "polygon");
          p.setAttribute("points", "0,-13 18,0 0,13 -18,0");
          p.setAttribute("fill", color);
          g.appendChild(p);
        }} else {{
          const c = document.createElementNS("http://www.w3.org/2000/svg", "circle");
          c.setAttribute("r", n.is_hit_memory ? 13 : 9);
          c.setAttribute("fill", color);
          g.appendChild(c);
        }}
        const text = document.createElementNS("http://www.w3.org/2000/svg", "text");
        text.setAttribute("x", 16);
        text.setAttribute("y", 4);
        text.textContent = `${{n.cn_label || LABELS[n.kind] || n.kind}}：${{n.label || n.value || n.id}}`;
        g.appendChild(text);
        g.addEventListener("mousemove", ev => {{
          tooltip.style.display = "block";
          tooltip.style.left = (ev.clientX + 12) + "px";
          tooltip.style.top = (ev.clientY + 12) + "px";
          tooltip.innerHTML = `<b>${{escapeHtml(n.id)}}</b><br>${{escapeHtml(n.description || "")}}`;
        }});
        g.addEventListener("mouseleave", () => tooltip.style.display = "none");
        nodeLayer.appendChild(g);
      }});

      if (mode === "memory") {{
        viewMeta.innerHTML = `记忆数：${{DATA.memory_count}}<br>节点数：${{graph.nodes.length}}<br>连线数：${{graph.links.length}}`;
        hitInfo.textContent = "整体图展示所有真实写入的 memory 和它们的超边结构。";
      }} else {{
        viewMeta.innerHTML = `查询：${{escapeHtml(graph.query)}}<br>查询节点数：${{graph.query_nodes.length}}<br>命中数：${{graph.hits.length}}`;
        hitInfo.textContent = JSON.stringify(graph.hits, null, 2);
      }}
    }}

    function escapeHtml(s) {{
      return String(s).replace(/[&<>"']/g, c => ({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[c]));
    }}

    function init() {{
      const legendKinds = ["memory", "hyperedge", "task", "agent", "state", "evidence", "supporting_fact", "gap", "strategy", "experience", "reuse_benefit", "keyword"];
      const legend = document.getElementById("legend");
      legendKinds.forEach(k => {{
        const s = document.createElement("span");
        s.className = "swatch";
        s.style.background = COLORS[k] || "#607d8b";
        const label = document.createElement("span");
        label.textContent = LABELS[k] || k;
        legend.appendChild(s);
        legend.appendChild(label);
      }});
      DATA.retrieval_graphs.forEach((g, idx) => {{
        const btn = document.createElement("button");
        btn.className = "btn";
        btn.textContent = `检索视图 ${{idx + 1}}：${{g.query.slice(0, 48)}}`;
        btn.onclick = () => {{ setActive(btn); render(g, "retrieval"); }};
        retrievalButtons.appendChild(btn);
      }});
      memoryBtn.onclick = () => {{ setActive(memoryBtn); render(DATA.memory_graph, "memory"); }};
      render(DATA.memory_graph, "memory");
    }}
    init();
  </script>
</body>
</html>
"""


def _all_hyperedges(hypergraph: dict[str, Any]) -> list[dict[str, Any]]:
    edges = []
    seen = set()
    event = hypergraph.get("event_hyperedge")
    if isinstance(event, dict):
        edges.append(event)
        seen.add(event.get("hyperedge_id"))
    for edge in hypergraph.get("hyperedges", []):
        if not isinstance(edge, dict) or edge.get("hyperedge_id") in seen:
            continue
        edges.append(edge)
        seen.add(edge.get("hyperedge_id"))
    return edges


def _dedupe_links(links: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped = {}
    for link in links:
        key = (link.get("source"), link.get("target"), link.get("edge_type"), bool(link.get("highlight")))
        deduped[key] = link
    return list(deduped.values())


def _summarize_hits(hits: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "memory_id": hit.get("memory_id"),
            "summary": _short(hit.get("summary", ""), 220),
            "retrieval_score": hit.get("retrieval_score"),
            "hypergraph_score": hit.get("hypergraph_score"),
            "embedding_score": hit.get("embedding_score"),
            "matched_nodes": hit.get("matched_nodes", [])[:20],
            "matched_hyperedges": hit.get("matched_hyperedges", [])[:20],
            "reasoning_path": hit.get("reasoning_path", [])[:12],
        }
        for hit in hits
    ]


def _summarize_record(record: dict[str, Any]) -> dict[str, Any]:
    hypergraph = ensure_hypergraph_view(record)
    return {
        "memory_id": record.get("memory_id"),
        "summary": _short(record.get("summary", ""), 220),
        "task_topic": _short(record.get("task_topic", ""), 220),
        "tags": record.get("tags", []),
        "hyperedge_count": len(_all_hyperedges(hypergraph)),
        "distilled_experience": hypergraph.get("distilled_experience", {}),
    }
