/* app.js — context-optimizer demo visualization */

// ── State ────────────────────────────────────────────────────────────────────
const state = {
  treeData: null,       // {nodes, edges, depth} from /api/tree
  nodeMap: new Map(),   // id → node datum
  status: null,
  activeFile: null,
  simulation: null,
  svg: null,
  nodeEls: null,
  linkEls: null,
};

// ── Suggestions ───────────────────────────────────────────────────────────────
//
// Four scenario groups, each targeting a specific Arbor capability:
//
//  DOMAIN ROUTING   — queries that span multiple indexed domains.
//                     Arbor should route each to the correct subsystem
//                     without cross-contamination.
//
//  MULTI-GRANULARITY — high-level questions answered from cluster summaries;
//                      low-level questions that need a specific raw block.
//
//  ITERATIVE EXPANSION — queries whose vocabulary doesn't appear verbatim in
//                      any single L1 summary, triggering the second-pass
//                      expansion to top-8 blocks.
//
//  DJANGO CROSS-SYSTEM — queries that stress-test routing across Django's
//                      many overlapping subsystems (ORM, auth, HTTP, cache,
//                      templates all use words like "session", "query",
//                      "cache", "permission" with different meanings).

// Scenario 1 — Domain routing (requests library, P&P, prose)
const SUGGESTIONS_ROUTING = [
  "Describe the character of Mr Darcy",
  "How does session-level authentication work in requests?",
  "How does the cookie jar persist state across requests?",
  "What is RAG and how does it work?",
  "Explain the compress then retrieve architecture",
];

// Scenario 2 — Multi-granularity retrieval
const SUGGESTIONS_GRANULARITY = [
  "What HTTP features does the requests library provide?",
  "How is the Arbor index structured at a high level?",
  "How does the HTTPAdapter handle connection pooling?",
  "How are HTTP retries and backoff configured?",
  "How does requests handle redirects?",
];

// Scenario 3 — Iterative expansion
const SUGGESTIONS_EXPANSION = [
  "Is Mr Darcy considered proud or humble by the other characters?",
  "Describe the relationship between Jane Bennet and Mr Bingley",
  "How is trust established between parties in requests?",
];

// Scenario 4 — Django cross-system routing (requires django-src corpus)
// Each query uses vocabulary that appears in multiple Django subsystems.
// The point of the demo: Arbor routes "session" to auth vs HTTP vs ORM
// depending on what the query is actually asking about.
const SUGGESTIONS_DJANGO = [
  // Routing — each should land on a specific Django subsystem
  "How does Django authenticate a user?",
  "How does Django's ORM build a SQL query from a QuerySet?",
  "How does Django's template engine render a variable?",
  "How does Django route an incoming HTTP request to a view?",
  "How does Django's cache framework store and retrieve values?",
  "How are Django signals dispatched to receivers?",
  // Cross-system stress tests — same word, different systems
  "How is a session managed in Django?",       // auth session vs HTTP session
  "How does Django handle permissions?",       // auth perms vs model perms vs view perms
  "What does a Django middleware do?",
  "How does Django handle database connections?",
  // Multi-granularity within Django
  "What is the Django ORM at a high level?",   // cluster summary sufficient
  "How does QuerySet.filter() translate to SQL WHERE clauses?",  // raw block needed
  "What authentication backends does Django support?",
  "How does Django's session middleware read the session cookie?",
];

// All suggestions merged; the UI renders them as clickable chips.
const SUGGESTIONS = [
  ...SUGGESTIONS_ROUTING,
  ...SUGGESTIONS_GRANULARITY,
  ...SUGGESTIONS_EXPANSION,
  ...SUGGESTIONS_DJANGO,
];

// ── Init ──────────────────────────────────────────────────────────────────────
document.addEventListener("DOMContentLoaded", async () => {
  // File panel collapse toggle
  document.getElementById("file-panel-toggle").addEventListener("click", () => {
    document.getElementById("file-panel").classList.toggle("collapsed");
  });

  renderSuggestions();

  // Retry status until ready — cold-start can take 60-90s while ChromaDB
  // and the sentence-transformer model load (40 attempts × 2500ms = 100s).
  for (let attempt = 0; attempt < 40; attempt++) {
    await checkStatus();
    if (state.status?.ready) break;
    await new Promise(r => setTimeout(r, 2500));
  }

  if (state.status?.ready) {
    await loadFiles();
    await buildGraph();

    // Re-layout (NOT rebuild) when the container resizes — avoids
    // discarding block children that were added during a query.
    new ResizeObserver(() => {
      if (state.hierData && state.g) {
        const container = document.getElementById("trie-container");
        state.W = container.clientWidth;
        state.H = container.clientHeight;
        document.getElementById("trie-svg").setAttribute("viewBox", `0 0 ${state.W} ${state.H}`);
        redrawTree(false);
      }
    }).observe(document.getElementById("trie-container"));
  }

  document.getElementById("query-btn").addEventListener("click", handleQuery);
  document.getElementById("query-input").addEventListener("keydown", (e) => {
    if (e.key === "Enter") handleQuery();
  });
});

// ── Status ────────────────────────────────────────────────────────────────────
async function checkStatus() {
  const overlay = document.getElementById("not-ready-overlay");
  try {
    const r = await fetch("/api/status");
    state.status = await r.json();
    const dot = document.getElementById("status-dot");
    const txt = document.getElementById("status-text");
    if (state.status.ready) {
      dot.classList.add("ready");
      txt.textContent =
        `${state.status.block_count} blocks \u00b7 ${state.status.cluster_count} clusters \u00b7 depth ${state.status.depth}`;
      if (overlay) overlay.style.display = "none";
    } else {
      txt.textContent = state.status.message || "Index not ready";
      if (overlay) overlay.style.display = "flex";
    }
  } catch {
    document.getElementById("status-text").textContent = "Server error";
    if (overlay) overlay.style.display = "flex";
  }
}

// ── Suggestions ───────────────────────────────────────────────────────────────
function renderSuggestions() {
  const container = document.getElementById("suggestions");
  SUGGESTIONS.forEach((s) => {
    const chip = document.createElement("span");
    chip.className = "chip";
    chip.textContent = s;
    chip.addEventListener("click", () => {
      document.getElementById("query-input").value = s;
      handleQuery();
    });
    container.appendChild(chip);
  });
}

// ── File tree ─────────────────────────────────────────────────────────────────
async function loadFiles() {
  try {
    const r = await fetch("/api/files");
    const data = await r.json();
    renderFileTree(data.tree);
  } catch (e) {
    console.error("loadFiles:", e);
  }
}

function renderFileTree(tree) {
  const container = document.getElementById("file-tree");
  container.innerHTML = "";

  for (const [dir, files] of Object.entries(tree)) {
    const parts = dir.replace(/\\/g, "/").split("/");
    const label = parts.slice(-2).join("/");

    // Wrapper
    const group = document.createElement("div");
    group.className = "dir-group";

    // Collapsible header
    const dirEl = document.createElement("div");
    dirEl.className = "dir-node";
    dirEl.innerHTML = `<span class="dir-arrow">▾</span><span>${label}</span><span style="margin-left:auto;font-size:10px;color:var(--text-dim)">${files.length}</span>`;

    // File list
    const filesEl = document.createElement("div");
    filesEl.className = "dir-files";

    files.forEach((filePath) => {
      const name = filePath.replace(/\\/g, "/").split("/").pop();
      const ext = name.split(".").pop() || "";
      const el = document.createElement("div");
      el.className = "file-node";
      el.dataset.path = filePath;
      el.innerHTML = `${name.replace("." + ext, "")}<span class="file-ext">.${ext}</span>`;
      el.addEventListener("click", () => openFile(filePath, el));
      filesEl.appendChild(el);
    });

    // Toggle collapse on dir click
    dirEl.addEventListener("click", () => {
      const collapsed = filesEl.classList.toggle("collapsed");
      dirEl.classList.toggle("collapsed", collapsed);
    });

    group.appendChild(dirEl);
    group.appendChild(filesEl);
    container.appendChild(group);
  }
}

async function openFile(path, el) {
  // Deactivate previous
  document.querySelectorAll(".file-node.active").forEach((n) =>
    n.classList.remove("active")
  );
  el.classList.add("active");
  state.activeFile = path;

  const placeholder = document.getElementById("viewer-placeholder");
  const content = document.getElementById("viewer-content");
  placeholder.style.display = "none";

  try {
    const r = await fetch(`/api/file?path=${encodeURIComponent(path)}`);
    const data = await r.json();
    content.textContent = data.content;
  } catch {
    content.textContent = "Error loading file.";
  }
}

// ── D3 force graph ────────────────────────────────────────────────────────────
async function buildGraph() {
  try {
    const r = await fetch("/api/tree");
    state.treeData = await r.json();
    renderGraph(state.treeData);
  } catch (e) {
    console.error("buildGraph:", e);
  }
}

function renderGraph({ nodes }) {
  const container = document.getElementById("trie-container");
  const W = container.clientWidth;
  const H = container.clientHeight;

  document.getElementById("trie-svg").innerHTML = "";
  const svg = d3.select("#trie-svg").attr("viewBox", `0 0 ${W} ${H}`);
  state.svg = svg;
  state.W = W;
  state.H = H;

  const g = svg.append("g").attr("class", "graph-root");
  svg.call(d3.zoom().scaleExtent([0.15, 5]).on("zoom", e => g.attr("transform", e.transform)));
  state.g = g;

  // Build a virtual-root hierarchy: root → clusters (no blocks yet)
  const clusterNodes = nodes.filter(n => n.type === "cluster");
  state.hierData = {
    id: "__root__", type: "root", label: "Index",
    children: clusterNodes.map(c => ({ ...c, children: [] })),
  };
  state.hierRoot = d3.hierarchy(state.hierData);
  state.nodeMap.clear();

  // Fixed node-size tree so spacing is predictable regardless of count
  state.treeLayout = d3.tree().nodeSize([90, 100]);

  redrawTree(false);
}

// Recompute layout and transition all nodes/links to new positions.
function redrawTree(animate) {
  const g = state.g;
  if (!g || !state.hierData) return;

  state.hierRoot = d3.hierarchy(state.hierData);
  state.treeLayout(state.hierRoot);

  const descendants = state.hierRoot.descendants();
  const links = state.hierRoot.links();

  // Centre the tree in the viewport
  const xs = descendants.map(d => d.x);
  const ys = descendants.map(d => d.y);
  const ox = state.W / 2 - (Math.min(...xs) + Math.max(...xs)) / 2;
  const oy = 50 - Math.min(...ys);          // 50px top padding
  const dur = animate ? 480 : 0;

  // ── Links ──────────────────────────────────────────────────────────────────
  const lSel = g.selectAll(".tree-link")
    .data(links, d => `${d.source.data.id}~${d.target.data.id}`);

  lSel.enter().append("path").attr("class", "tree-link")
    .attr("opacity", 0)
    .attr("d", d => d3.linkVertical()({
      source: [d.source.x + ox, d.source.y + oy],
      target: [d.target.x + ox, d.target.y + oy],
    }))
    .transition().duration(dur).attr("opacity", 1);

  lSel.transition().duration(dur)
    .attr("opacity", 1)
    .attr("d", d => d3.linkVertical()({
      source: [d.source.x + ox, d.source.y + oy],
      target: [d.target.x + ox, d.target.y + oy],
    }));

  lSel.exit().transition().duration(dur).attr("opacity", 0).remove();

  // ── Nodes ──────────────────────────────────────────────────────────────────
  const nodeR = d => d.data.type === "cluster" ? 28 : d.data.type === "block" ? 14 : 10;

  const nSel = g.selectAll(".tree-node")
    .data(descendants, d => d.data.id);

  const entered = nSel.enter().append("g")
    .attr("class", d => `tree-node node-${d.data.type}`)
    .attr("transform", d => `translate(${d.x + ox},${d.y + oy})`)
    .attr("opacity", 0);

  entered.append("circle").attr("r", d => d.data.type === "block" ? 0 : nodeR(d));

  // Primary label
  entered.append("text").attr("dy", "0.35em")
    .text(d => {
      if (d.data.type === "root")    return "Index";
      if (d.data.type === "cluster") return d.data.label || d.data.id.slice(-8);
      // Block: short filename without extension
      const src = (d.data.source_file || d.data.block_id || "").replace(/\\/g, "/");
      return src.split("/").pop().replace(/\.[^.]+$/, "").slice(0, 10);
    });

  // Distance badge (blocks only)
  entered.filter(d => d.data.type === "block")
    .append("text").attr("class", "block-score").attr("dy", "2.2em")
    .text(d => d.data.distance != null ? `d=${d.data.distance.toFixed(2)}` : "");

  // Events (non-root nodes)
  entered.filter(d => d.data.type !== "root")
    .style("cursor", "pointer")
    .on("click", (event, d) => {
      event.stopPropagation();
      if (d.data.type === "block") fetchBlock(d.data.block_id || d.data.id);
      showTooltip(event, d.data);
    })
    .on("mousemove", (event, d) => showTooltip(event, d.data))
    .on("mouseleave", hideTooltip);

  // Animate block circles in from r=0
  entered.filter(d => d.data.type === "block")
    .select("circle").transition().duration(dur).attr("r", nodeR);

  entered.transition().duration(dur).attr("opacity", 1);

  // Move existing nodes to new positions
  nSel.transition().duration(dur)
    .attr("transform", d => `translate(${d.x + ox},${d.y + oy})`)
    .attr("opacity", 1);

  nSel.exit().transition().duration(dur).attr("opacity", 0).remove();

  // Rebuild node map
  state.nodeMap.clear();
  descendants.forEach(d => state.nodeMap.set(d.data.id, d.data));
}

// ── Tooltip ───────────────────────────────────────────────────────────────────
function showTooltip(event, d) {
  const tt = document.getElementById("tooltip");
  const id = (d.id || d.block_id || "").slice(-16);
  tt.innerHTML = `<strong>${id}</strong><br>${(d.summary || "").slice(0, 180)}…`;
  tt.style.left = event.clientX + 12 + "px";
  tt.style.top = event.clientY - 8 + "px";
  tt.classList.add("visible");
}
function hideTooltip() {
  document.getElementById("tooltip").classList.remove("visible");
}

// ── Query ─────────────────────────────────────────────────────────────────────
async function handleQuery() {
  const input = document.getElementById("query-input");
  const btn = document.getElementById("query-btn");
  const q = input.value.trim();
  if (!q || !state.status?.ready) return;

  btn.disabled = true;
  btn.innerHTML = '<span class="spinner"></span>';

  resetHighlights();

  try {
    const r = await fetch("/api/query", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        query: q,
        top_clusters: 4,
        top_blocks_per_cluster: 4,
      }),
    });
    const data = await r.json();
    await replayQueryResult(data);
  } catch (e) {
    console.error("Query failed:", e);
  } finally {
    btn.disabled = false;
    btn.textContent = "Ask";
  }
}

// ── Query result replay (animated) ────────────────────────────────────────────
async function replayQueryResult(data) {
  const { cluster_hits, steps, answer, fetched_blocks, latency_ms } = data;

  // Step 1 — flash searched clusters
  await sleep(200);
  const searchedIds = new Set(cluster_hits.map(c => c.cluster_id));
  highlightNodes(searchedIds, "searched");

  await sleep(600);

  // Step 2 — expand clusters: add block children to tree
  for (const ch of cluster_hits) {
    await expandCluster(ch);
    await sleep(350);
  }

  await sleep(400);

  // Step 3 — highlight fetched raw blocks (reasoning agent)
  const fetchedIds = new Set((fetched_blocks || []).map(b => b.block_id));
  if (fetchedIds.size > 0) {
    highlightNodes(fetchedIds, "fetched");
    await sleep(400);
  }

  showAnswer(answer, steps, latency_ms);
}

function highlightNodes(ids, cls) {
  if (!state.g) return;
  state.g.selectAll(".tree-node")
    .classed("searched", d => ids.has(d.data.id) && cls === "searched")
    .classed("expanded", d => ids.has(d.data.id) && cls === "expanded")
    .classed("matched",  d => ids.has(d.data.id) && cls === "matched")
    .classed("fetched",  d => ids.has(d.data.id) && cls === "fetched");
}

async function expandCluster(ch) {
  if (!state.hierData || !state.g) return;

  // Add block children to this cluster in the hierarchy data
  const clusterData = state.hierData.children.find(c => c.id === ch.cluster_id);
  if (clusterData) {
    clusterData.children = ch.block_hits.map(bh => ({
      ...bh,
      id: bh.block_id,
      type: "block",
      label: bh.block_id.slice(-8),
    }));
  }

  // Re-layout the whole tree with the new children, then highlight
  redrawTree(true);
  await sleep(120);

  highlightNodes(new Set([ch.cluster_id]), "expanded");
  if (ch.block_hits.length > 0) {
    highlightNodes(new Set([ch.block_hits[0].block_id]), "matched");
  }

  appendTraversalStep(ch);
}

async function fetchBlock(blockId) {
  try {
    const r = await fetch(`/api/block?block_id=${encodeURIComponent(blockId)}`);
    const data = await r.json();
    document.getElementById("traversal-log").style.display = "none";
    document.getElementById("viewer-header").textContent = "File Viewer";
    const content = document.getElementById("viewer-content");
    const placeholder = document.getElementById("viewer-placeholder");
    placeholder.style.display = "none";
    content.textContent = `[Block: ${blockId}]\n\n${data.raw_text}`;
  } catch {
    /* ignore */
  }
}

function appendTraversalStep(ch) {
  const log = document.getElementById("traversal-log");
  const placeholder = document.getElementById("viewer-placeholder");
  const content = document.getElementById("viewer-content");

  placeholder.style.display = "none";
  content.textContent = "";
  log.style.display = "block";
  document.getElementById("viewer-header").textContent = "Traversal Log";

  const shortId = ch.cluster_id.replace("cluster_", "").slice(-8);
  const superSum = (ch.super_summary || "").trim();
  const bestId = ch.block_hits.length > 0 ? ch.block_hits[0].block_id : null;

  const el = document.createElement("div");
  el.innerHTML = `
    <div class="trace-cluster-header">
      <span class="trace-cluster-id">${shortId}</span>
      <span class="trace-dist">cluster dist ${ch.distance?.toFixed(3)}</span>
    </div>
    <div class="trace-super-summary">${superSum.slice(0, 250) || "(no cluster summary)"}</div>
    ${ch.block_hits.map((bh, i) => {
      const fname = (bh.source_file || bh.block_id).split(/[\\/]/).pop();
      const isBest = bh.block_id === bestId;
      return `<div class="trace-block${isBest ? " best" : ""}">
        <div class="trace-block-header">
          <span class="trace-block-id">${fname || bh.block_id.slice(-14)}</span>
          ${isBest ? '<span class="trace-best-badge">best match</span>' : ""}
          <span class="trace-dist" style="margin-left:auto">dist ${bh.distance?.toFixed(3)}</span>
        </div>
        <div class="trace-block-summary">${(bh.summary || "").slice(0, 200)}…</div>
      </div>`;
    }).join("")}
  `;
  log.appendChild(el);
  log.scrollTop = log.scrollHeight;
}

function resetHighlights() {
  // Strip block children so the tree reverts to clusters-only
  if (state.hierData) {
    state.hierData.children.forEach(c => { c.children = []; });
    redrawTree(false);
  }
  document.getElementById("answer-panel").classList.remove("visible");
  const log = document.getElementById("traversal-log");
  log.innerHTML = "";
  log.style.display = "none";
  document.getElementById("viewer-header").textContent = "File Viewer";
  document.getElementById("viewer-placeholder").style.display = "";
  document.getElementById("viewer-content").textContent = "";
}


// ── Answer panel ──────────────────────────────────────────────────────────────
function showAnswer(answer, steps, latencyMs) {
  const panel = document.getElementById("answer-panel");
  panel.classList.add("visible");

  document.getElementById("answer-text").textContent = answer;

  const stepsList = document.getElementById("steps-list");
  stepsList.innerHTML = "";

  (steps || []).forEach((s) => {
    const badge = document.createElement("span");
    badge.className = "step-badge step-" + s.action.split("_")[0];
    badge.textContent =
      s.action + (s.target_id ? `(${s.target_id.slice(-8)})` : "");
    stepsList.appendChild(badge);
  });

  if (latencyMs) {
    const lbl = document.createElement("span");
    lbl.style.cssText = "font-size:11px;color:var(--text-dim);margin-left:auto";
    lbl.textContent = `${latencyMs.toFixed(0)} ms`;
    stepsList.appendChild(lbl);
  }
}

// ── Utils ─────────────────────────────────────────────────────────────────────
function sleep(ms) {
  return new Promise((r) => setTimeout(r, ms));
}
