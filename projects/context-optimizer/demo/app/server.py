"""
Demo web server — FastAPI backend for the context-optimizer visualization portal.

Endpoints
---------
GET  /                  Serve the single-page app (index.html)
GET  /api/status        Index stats: block_count, cluster_count, depth, ready
GET  /api/tree          Full tree structure as {nodes, edges} for D3
GET  /api/files         Indexed file listing from FileRegistry
GET  /api/file          Raw content of a specific file by path
POST /api/query         Run a query; returns answer + cluster_hits + steps
"""
from __future__ import annotations

# ── Dev-mode bootstrap ─────────────────────────────────────────────────────────
# Wire src/ as context_optimizer if running from the repo without pip install.
# This is a no-op when the package is already installed.
try:
    import context_optimizer  # noqa: F401
except ImportError:
    import importlib.util as _ilu, types as _types
    from pathlib import Path as _Path
    _src = (_Path(__file__).parent.parent.parent / "src").resolve()
    _pkg = _types.ModuleType("context_optimizer")
    _pkg.__path__ = [str(_src)]
    _pkg.__package__ = "context_optimizer"
    _pkg.__file__ = str(_src / "__init__.py")
    import sys as _sys; _sys.modules["context_optimizer"] = _pkg
    _spec = _ilu.spec_from_file_location("context_optimizer", _src / "__init__.py", submodule_search_locations=[str(_src)])
    _spec.loader.exec_module(_pkg)
    del _ilu, _types, _Path, _src, _pkg, _sys, _spec
# ─────────────────────────────────────────────────────────────────────────────

import os
import re as _re
import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# ── Index paths ───────────────────────────────────────────────────────────────

_HERE = Path(__file__).parent
_INDEX_DIR = Path(
    os.environ.get("DEMO_INDEX_DIR", str(_HERE.parent / ".index"))
).resolve()
_REASONING_MODEL = os.environ.get("DEMO_REASONING_MODEL", "")

# ── Lazy-loaded index components ──────────────────────────────────────────────

_tree: Any = None
_block_index: Any = None
_reasoning_llm: Any = None
_depth: int = 2


def _load_index() -> bool:
    """Load the index lazily on first request. Returns True if successful."""
    global _tree, _block_index, _file_registry, _reasoning_llm, _depth

    if _tree is not None:
        return True

    if not _INDEX_DIR.exists():
        return False

    try:
        from context_optimizer.raw_index import BlockIndex
        from context_optimizer.tree_index import TreeIndex

        _block_index = BlockIndex(str(_INDEX_DIR / "blocks.db"))
        _tree = TreeIndex(
            collection_name="demo_index",
            persist_directory=str(_INDEX_DIR),
            block_index=_block_index,
        )
        _depth = _tree._depth

        # Optional reasoning LLM (Ollama)
        if _REASONING_MODEL:
            try:
                from context_optimizer.providers.ollama import build as _ollama_build

                _reasoning_llm = _ollama_build(
                    model=_REASONING_MODEL,
                    base_url=os.environ.get(
                        "OLLAMA_BASE_URL", "http://localhost:11434"
                    ),
                    temperature=0.0,
                )
            except Exception as exc:
                print(f"[server] Reasoning LLM unavailable: {exc}")

        return True
    except Exception as exc:
        print(f"[server] Failed to load index: {exc}")
        return False


# ── Sentence-level relevance helper ──────────────────────────────────────────

_STOP = frozenset(
    "a an the and or but in on at to for of with by from is are was were be been "
    "have has had do does did that this it he she they we you i his her its their "
    "who what which how when where will would could should may might then than".split()
)


def _relevant_excerpt(text: str, query: str, max_chars: int = 500) -> str:
    """
    Return the most query-relevant passage from *text*.

    Splits into candidate segments (paragraphs, then sentences within them),
    scores each by TF-IDF word overlap against the query keywords, and
    assembles the top-scoring non-overlapping segments up to *max_chars*.
    Preserves document order for readability.
    """
    # Tokenise query into meaningful keywords
    q_words = {
        w.lower().rstrip("?.,!;:")
        for w in query.split()
        if w.lower().rstrip("?.,!;:") not in _STOP and len(w) > 2
    }
    if not q_words:
        return text.strip()[:max_chars]

    # Split into paragraphs first, then into sentences within each paragraph
    paragraphs = [p.strip() for p in _re.split(r'\n{2,}', text.strip()) if p.strip()]
    candidates: list[str] = []
    for para in paragraphs:
        # If paragraph is long, split further into sentences
        if len(para) > 300:
            sents = [s.strip() for s in _re.split(r'(?<=[.!?])\s+', para) if s.strip() and len(s.strip()) > 20]
            candidates.extend(sents)
        else:
            candidates.append(para)

    if not candidates:
        return text.strip()[:max_chars]

    def _score(seg: str) -> float:
        words = {w.lower().rstrip(".,;:!?\"'") for w in seg.split() if len(w) > 2}
        matches = q_words & words
        if not matches:
            return 0.0
        # Bonus for exact phrase fragments
        seg_lower = seg.lower()
        phrase_bonus = sum(1.5 for w in q_words if w in seg_lower)
        return len(matches) + phrase_bonus

    scored = sorted(enumerate(candidates), key=lambda x: _score(x[1]), reverse=True)

    # Greedily pick top segments up to max_chars, then restore document order
    chosen_indices: list[int] = []
    chars_used = 0
    for idx, seg in scored:
        if chars_used + len(seg) > max_chars and chosen_indices:
            break
        chosen_indices.append(idx)
        chars_used += len(seg) + 4  # account for separator
        if chars_used >= max_chars:
            break

    if not chosen_indices:
        chosen_indices = [scored[0][0]] if scored else [0]

    chosen_indices.sort()  # restore document order
    parts = [candidates[i] for i in chosen_indices]

    # Join consecutive segments cleanly; use "…" between non-consecutive ones
    result_parts: list[str] = [parts[0]]
    for prev_i, curr_i, seg in zip(chosen_indices, chosen_indices[1:], parts[1:]):
        sep = " " if curr_i == prev_i + 1 else " … "
        result_parts.append(sep + seg)

    return "".join(result_parts).strip()


# ── App ────────────────────────────────────────────────────────────────────────

app = FastAPI(title="Context Optimizer Demo")
app.mount("/static", StaticFiles(directory=str(_HERE / "static")), name="static")


@app.get("/", include_in_schema=False)
def root() -> FileResponse:
    return FileResponse(str(_HERE / "static" / "index.html"))


# ── /api/status ───────────────────────────────────────────────────────────────


@app.get("/api/status")
def status() -> dict:
    ready = _load_index()
    if not ready:
        return {
            "ready": False,
            "message": f"Index not found at {_INDEX_DIR}. Run setup_demo.py first.",
        }
    return {
        "ready": True,
        "block_count": _tree.block_count(),
        "cluster_count": _tree.cluster_count(),
        "depth": _depth,
        "index_dir": str(_INDEX_DIR),
        "reasoning_model": _REASONING_MODEL or None,
    }


# ── /api/tree ─────────────────────────────────────────────────────────────────


@app.get("/api/tree")
def tree_data() -> dict:
    """Return all cluster and block nodes + edges for D3 rendering."""
    if not _load_index():
        raise HTTPException(503, "Index not ready — run setup_demo.py first")

    # Use the top-level collection (L{depth}) as cluster nodes so depth=3
    # indexes show L3 super-clusters rather than the intermediate L2 clusters.
    top_lvl = _tree._depth
    top_coll = _tree._levels[top_lvl]
    clusters_raw = top_coll.get(include=["documents", "metadatas"])
    cluster_ids: list[str] = clusters_raw.get("ids", [])
    cluster_docs: list[str] = clusters_raw.get("documents", [])
    cluster_metas: list[dict] = clusters_raw.get("metadatas", [{}] * len(cluster_ids))

    # Fetch all L1 block nodes
    l1 = _tree._levels[1]
    blocks_raw = l1.get(include=["documents", "metadatas"])
    block_ids: list[str] = blocks_raw.get("ids", [])
    block_docs: list[str] = blocks_raw.get("documents", [])
    block_metas: list[dict] = blocks_raw.get("metadatas", [{}] * len(block_ids))

    nodes: list[dict] = []
    edges: list[dict] = []

    # Build block lookup: block_id → metadata
    block_meta_map = {
        bid: {"summary": doc[:200], "meta": meta}
        for bid, doc, meta in zip(block_ids, block_docs, block_metas)
    }

    for cid, cdoc, cmeta in zip(cluster_ids, cluster_docs, cluster_metas):
        raw_ids: str = cmeta.get("child_ids", cmeta.get("child_block_ids", ""))
        children = [c for c in raw_ids.split(",") if c]

        nodes.append(
            {
                "id": cid,
                "type": "cluster",
                "label": cid[-8:],  # short label
                "summary": cdoc[:300],
                "child_count": len(children),
            }
        )
        for bid in children:
            if bid in block_meta_map:
                info = block_meta_map[bid]
                nodes.append(
                    {
                        "id": bid,
                        "type": "block",
                        "label": bid[-8:],
                        "summary": info["summary"],
                        "source_file": str(info["meta"].get("source_file", "")),
                        "parent_cluster": cid,
                    }
                )
                edges.append({"source": cid, "target": bid})

    return {"nodes": nodes, "edges": edges, "depth": _depth}


# ── /api/files ────────────────────────────────────────────────────────────────


@app.get("/api/files")
def list_files() -> dict:
    """Return a tree-friendly listing of all indexed files, derived from L1 block metadata."""
    if not _load_index():
        raise HTTPException(503, "Index not ready — run setup_demo.py first")

    # Derive unique source files from ChromaDB L1 block metadata.
    # FileRegistry is only populated by the watcher; for demo builds we
    # read directly from the L1 collection which ingest_directory always fills.
    l1 = _tree._levels[1]
    all_meta = l1.get(include=["metadatas"])
    raw_paths = sorted(
        {m.get("source_file", "") for m in (all_meta.get("metadatas") or [])}
        - {""}  # remove empty strings
    )

    # Build a simple nested structure: {directory: [files]}
    tree: dict[str, list[str]] = {}
    for p in raw_paths:
        parent = str(Path(p).parent)
        tree.setdefault(parent, []).append(p)

    return {"files": raw_paths, "tree": tree}


# ── /api/file ─────────────────────────────────────────────────────────────────


@app.get("/api/file")
def get_file_content(path: str = Query(..., description="Absolute path to file")) -> dict:
    """Return the raw text content of an indexed file."""
    if not _load_index():
        raise HTTPException(503, "Index not ready")

    # Security: only serve files that appear in L1 metadata (closed set from index)
    l1 = _tree._levels[1]
    all_meta = l1.get(include=["metadatas"])
    indexed = {m.get("source_file", "") for m in (all_meta.get("metadatas") or [])}
    if path not in indexed:
        raise HTTPException(404, f"File '{path}' not in index")

    try:
        text = Path(path).read_text(encoding="utf-8", errors="replace")
        return {"path": path, "content": text, "size": len(text)}
    except OSError as exc:
        raise HTTPException(500, f"Cannot read file: {exc}") from exc


# ── /api/query ────────────────────────────────────────────────────────────────


class QueryRequest(BaseModel):
    query: str
    top_clusters: int = 4
    top_blocks_per_cluster: int = 4
    max_rounds: int = 3
    gap: float = 2.0   # wide gap: explore all clusters so the best L1 block is always found


@app.post("/api/query")
def run_query(req: QueryRequest) -> dict:
    """
    Run a query against the tree index.

    Returns cluster_hits (for trie visualization) plus an optional
    synthesized answer from the reasoning LLM.
    """
    if not _load_index():
        raise HTTPException(503, "Index not ready — run setup_demo.py first")

    t0 = time.perf_counter()

    # ── Step 1: hierarchical search ───────────────────────────────────────────
    cluster_hits = _tree.search(
        req.query,
        top_clusters=req.top_clusters,
        top_blocks_per_cluster=req.top_blocks_per_cluster,
        gap=req.gap,
    )

    # Serialize cluster hits for the frontend
    hit_data: list[dict] = []
    for ch in cluster_hits:
        hit_data.append(
            {
                "cluster_id": ch.cluster_id,
                "super_summary": ch.super_summary,
                "distance": round(ch.distance, 4),
                "child_block_ids": ch.child_block_ids,
                "block_hits": [
                    {
                        "block_id": bh.block_id,
                        "summary": bh.summary[:500],
                        "distance": round(bh.distance, 4),
                        "cluster_id": bh.cluster_id,
                    }
                    for bh in ch.block_hits
                ],
            }
        )

    steps: list[dict] = [
        {"action": "search_clusters", "target_id": "", "detail": f"top {req.top_clusters} clusters"}
    ]

    # ── Step 2: reasoning agent (optional) ────────────────────────────────────
    answer = ""
    fetched_blocks: list[dict] = []

    if _reasoning_llm is not None:
        from context_optimizer.tree_reasoner import TreeReasoningAgent

        agent = TreeReasoningAgent(
            tree=_tree,
            llm=_reasoning_llm,
            top_clusters=req.top_clusters,
            top_blocks_per_cluster=req.top_blocks_per_cluster,
            max_rounds=req.max_rounds,
        )
        result = agent.reason(req.query)
        answer = result.answer

        for s in result.steps:
            steps.append(
                {
                    "action": s.action,
                    "target_id": s.target_id,
                    "latency_ms": round(s.latency_ms, 1),
                }
            )
            if s.action == "fetch_raw_block" and s.target_id:
                raw = _tree.get_raw_block(s.target_id)
                if raw:
                    fetched_blocks.append(
                        {
                            "block_id": s.target_id,
                            "raw_text": raw[:2000],
                            "truncated": len(raw) > 2000,
                        }
                    )
    else:
        # Retrieval-only mode with iterative expansion.
        #
        # Round 1 fetches top-3 L1 blocks and scores how well each extracted
        # excerpt overlaps with the query keywords.  If the best score is below
        # the relevance threshold we haven't found enough — expand to top-8 and
        # try again.  This is a lightweight stand-in for the LLM-driven
        # "I need more context" loop that TreeReasoningAgent provides when Ollama
        # is available (set DEMO_REASONING_MODEL to enable that path).

        _RELEVANCE_THRESHOLD = 0.25  # at least 25 % of query keywords must appear

        def _score_excerpt(text: str) -> float:
            q_words = {
                w.lower().rstrip("?.,!;:")
                for w in req.query.split()
                if w.lower().rstrip("?.,!;:") not in _STOP and len(w) > 2
            }
            if not q_words:
                return 1.0
            t_words = {w.lower().rstrip(".,;:!?\"'") for w in text.split() if len(w) > 2}
            return len(q_words & t_words) / len(q_words)

        candidates: list[tuple[str, float, str]] = []  # (excerpt, score, source_file)
        seen_bids: set[str] = set()

        for top_k, round_label in ((3, "initial"), (8, "expanded")):
            l1_hits = _tree._query_level(1, req.query, None, top_k)
            new_hits = [(bid, s, d) for bid, s, d in l1_hits if bid not in seen_bids]

            for bid, _summary, _dist in new_hits:
                seen_bids.add(bid)
                raw = _tree.get_raw_block(bid)
                if not raw:
                    continue
                meta_res = _tree._levels[1].get(ids=[bid], include=["metadatas"])
                src = meta_res["metadatas"][0].get("source_file", "") if meta_res["ids"] else ""
                excerpt = _relevant_excerpt(raw, req.query, max_chars=500)
                score = _score_excerpt(excerpt)
                candidates.append((excerpt, score, src))

            best = max((sc for _, sc, _ in candidates), default=0.0)
            if best >= _RELEVANCE_THRESHOLD:
                steps.append({
                    "action": "retrieval_sufficient",
                    "target_id": "",
                    "detail": f"{round_label} retrieval: best_score={best:.2f} (threshold={_RELEVANCE_THRESHOLD})",
                })
                break
            else:
                steps.append({
                    "action": "expand_retrieval",
                    "target_id": "",
                    "detail": f"{round_label} retrieval insufficient (best_score={best:.2f}), expanding…",
                })

        # Sort by score, keep top-3 most relevant excerpts, restore document feel
        candidates.sort(key=lambda x: x[1], reverse=True)
        top3 = candidates[:3]

        if top3:
            fnames = list(dict.fromkeys(
                Path(src).name for _, _, src in top3 if src
            ))
            sources = ", ".join(fnames) if fnames else "unknown"
            answer = f"[Retrieved from: {sources}]\n\n" + "\n\n---\n\n".join(e for e, _, _ in top3)
        else:
            answer = "No relevant content found."

    elapsed_ms = (time.perf_counter() - t0) * 1000

    return {
        "query": req.query,
        "answer": answer,
        "cluster_hits": hit_data,
        "steps": steps,
        "fetched_blocks": fetched_blocks,
        "latency_ms": round(elapsed_ms, 1),
        "reasoning_model": _REASONING_MODEL or None,
    }


# ── /api/debug ───────────────────────────────────────────────────────────────


@app.get("/api/debug")
def debug_index(query: str = "Who are the main characters in Pride and Prejudice?") -> dict:
    """Diagnostic: run an L1 direct query and report what get_raw_block returns."""
    if not _load_index():
        raise HTTPException(503, "Index not ready")
    l1_coll = _tree._levels[1]
    l1_error = None
    l1_results = []
    try:
        res = l1_coll.query(query_texts=[query], n_results=3, include=["documents", "distances"])
        for j in range(len(res["ids"][0])):
            bid = res["ids"][0][j]
            raw = _tree.get_raw_block(bid)
            ptr = _block_index.get_meta(bid) if _block_index else None
            l1_results.append({
                "block_id": bid,
                "distance": round(res["distances"][0][j], 4),
                "summary_snippet": (res["documents"][0][j] or "")[:120],
                "in_block_index": ptr is not None,
                "file_path": ptr.file_path if ptr else None,
                "raw_text_ok": raw is not None,
            })
    except Exception as exc:
        l1_error = str(exc)
    return {
        "query": query,
        "l1_count": l1_coll.count(),
        "l1_error": l1_error,
        "hits": l1_results,
    }


@app.post("/api/repair")
def repair_l1_index() -> dict:
    """
    Repair a broken L1 HNSW index.

    ChromaDB sometimes fails to persist the HNSW binary files for a
    collection that was populated but never queried.  This endpoint reads
    all L1 entries via metadata-only GET (which bypasses HNSW), then
    deletes and recreates the collection to force a clean HNSW build.
    """
    if not _load_index():
        raise HTTPException(503, "Index not ready")

    l1_coll = _tree._levels[1]
    coll_name = l1_coll.name

    # Read all entries via metadata (doesn't touch HNSW)
    all_data = l1_coll.get(include=["documents", "metadatas"])
    ids = all_data["ids"]
    docs = all_data["documents"]
    metas = all_data["metadatas"]
    if not ids:
        return {"status": "error", "detail": "L1 is empty — nothing to repair"}

    # Delete the broken collection and recreate it with the same embedding fn
    emb_fn = l1_coll._embedding_function
    _tree._client.delete_collection(coll_name)
    new_coll = _tree._client.create_collection(
        name=coll_name,
        embedding_function=emb_fn,
        metadata={"hnsw:space": "cosine"},
    )

    # Re-add in batches of 100
    batch = 100
    for start in range(0, len(ids), batch):
        new_coll.add(
            ids=ids[start:start + batch],
            documents=docs[start:start + batch],
            metadatas=metas[start:start + batch],
        )

    # Force a dummy query so ChromaDB flushes the HNSW segment to disk
    new_coll.query(query_texts=["dummy warmup query"], n_results=1)

    # Swap into the live tree so the server uses the repaired collection
    _tree._levels[1] = new_coll
    return {"status": "ok", "repaired_entries": len(ids)}


# ── /api/block ────────────────────────────────────────────────────────────────


@app.get("/api/block")
def get_block(block_id: str = Query(...)) -> dict:
    """Fetch the raw text of a specific block by ID."""
    if not _load_index():
        raise HTTPException(503, "Index not ready")
    raw = _tree.get_raw_block(block_id)
    if raw is None:
        raise HTTPException(404, f"Block '{block_id}' not found")
    return {"block_id": block_id, "raw_text": raw}
