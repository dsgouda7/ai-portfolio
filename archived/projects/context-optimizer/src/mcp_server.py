"""
context-optimizer MCP server.

Exposes the Tree-of-Summaries index as three MCP tools:

  search_clusters(query, top_k=3)
      Entry-point: cosine-searches the top-level cluster summaries and
      returns them with their child block previews.  The calling agent
      uses this to decide which subtree is relevant — without reading
      raw content.

  expand_cluster(cluster_id)
      Drills one level deeper: returns the individual L1 block summaries
      inside a cluster.  Use when a cluster looks relevant but lacks detail.

  fetch_raw_block(block_id)
      Returns the full raw text of a specific block from disk via the
      BlockIndex file pointer.  Use only when a summary is insufficient
      and exact quotes/numbers/names are needed.

Security model
--------------
* The agent only sees what it explicitly asks for — raw corpus data is
  never pushed unprompted.
* ``block_id`` and ``cluster_id`` are validated against an allow-list
  (IDs that exist in the BlockIndex/ChromaDB collection) before any
  file I/O occurs.  Path-traversal is impossible: IDs are opaque hashes,
  not filesystem paths.
* In stdio transport mode (default) the server has no network exposure;
  the MCP client spawns it as a subprocess over stdin/stdout.

Usage
-----
Spawned automatically by the MCP client after `context-optimizer install-mcp`.
Can also be run directly::

    context-optimizer-mcp --index ~/.co/index

Or via Python::

    python -m context_optimizer.mcp_server --index ~/.co/index
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

# ── ID validation ─────────────────────────────────────────────────────────────
# Block and cluster IDs are hex/alphanumeric strings produced by the indexer.
# Reject anything that looks like a path or shell injection.
_ID_RE = re.compile(r"^[a-zA-Z0-9_\-]{1,256}$")


def _validate_id(value: str, label: str) -> None:
    if not _ID_RE.match(value):
        raise ValueError(
            f"Invalid {label} '{value}': must match ^[a-zA-Z0-9_\\-]{{1,256}}$"
        )


# ── Lazy index loader ─────────────────────────────────────────────────────────

_tree: Any = None
_block_index: Any = None
_index_dir: Path | None = None


def _load_index(index_dir: Path) -> None:
    global _tree, _block_index, _index_dir

    if not index_dir.exists():
        raise RuntimeError(
            f"Index directory '{index_dir}' not found. "
            "Run `context-optimizer build --corpus <path>` first."
        )

    from context_optimizer.raw_index import BlockIndex
    from context_optimizer.tree_index import TreeIndex

    _index_dir = index_dir
    _block_index = BlockIndex(str(index_dir / "blocks.db"))
    _tree = TreeIndex(
        collection_name="app_index",
        persist_directory=str(index_dir),
        block_index=_block_index,
    )


# ── Tool implementations ──────────────────────────────────────────────────────


def _search_clusters(query: str, top_k: int = 3) -> str:
    """
    Cosine-search top-level cluster summaries for *query*.

    Returns a JSON array of cluster objects, each with:
      - cluster_id, super_summary, distance
      - blocks: [{block_id, summary, distance}]

    The agent uses this as its entry point — it reads the super-summaries
    to decide which cluster is relevant, then optionally drills down.
    """
    if _tree is None:
        return json.dumps({"error": "Index not loaded"})

    cluster_hits = _tree.search(
        query,
        top_clusters=max(1, top_k),
        top_blocks_per_cluster=3,
    )

    result = []
    for ch in cluster_hits:
        result.append(
            {
                "cluster_id": ch.cluster_id,
                "super_summary": ch.super_summary,
                "distance": round(ch.distance, 4),
                "blocks": [
                    {
                        "block_id": bh.block_id,
                        "summary": bh.summary[:400],
                        "distance": round(bh.distance, 4),
                    }
                    for bh in ch.block_hits
                ],
            }
        )
    return json.dumps(result, ensure_ascii=False)


def _expand_cluster(cluster_id: str) -> str:
    """
    Return all L1 block summaries inside *cluster_id*.

    Use when a cluster super-summary looks relevant but you need more
    detail before deciding whether to fetch raw blocks.
    """
    _validate_id(cluster_id, "cluster_id")
    if _tree is None:
        return json.dumps({"error": "Index not loaded"})

    block_hits = _tree.expand_cluster(cluster_id)
    result = [
        {
            "block_id": bh.block_id,
            "summary": bh.summary,
            "cluster_id": bh.cluster_id,
            "distance": round(bh.distance, 4),
        }
        for bh in block_hits
    ]
    return json.dumps(result, ensure_ascii=False)


def _fetch_raw_block(block_id: str) -> str:
    """
    Return the full raw text for *block_id* from the BlockIndex.

    The block ID must exist in the index — arbitrary file paths are
    not accepted.  Returns {"error": ...} if the ID is unknown.
    """
    _validate_id(block_id, "block_id")
    if _block_index is None:
        return json.dumps({"error": "Index not loaded"})

    raw = _block_index.get_block_text(block_id)
    if raw is None:
        return json.dumps({"error": f"Block '{block_id}' not found in index"})

    return json.dumps(
        {"block_id": block_id, "raw_text": raw}, ensure_ascii=False
    )


# ── MCP server ────────────────────────────────────────────────────────────────


async def _run_server(index_dir: Path) -> None:
    _load_index(index_dir)

    try:
        from mcp.server import Server  # type: ignore[import]
        from mcp.server.stdio import stdio_server  # type: ignore[import]
        from mcp.types import (  # type: ignore[import]
            TextContent,
            Tool,
        )
    except ImportError:
        print(
            "[context-optimizer-mcp] ERROR: 'mcp' package not installed.\n"
            "Install with:  pip install 'context-optimizer[mcp]'",
            file=sys.stderr,
        )
        sys.exit(1)

    server = Server("context-optimizer")

    @server.list_tools()
    async def _list_tools() -> list[Tool]:
        return [
            Tool(
                name="search_clusters",
                description=(
                    "Search the corpus index for a query. Returns cluster summaries "
                    "and a preview of their block summaries. Start here for every query."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The question or search phrase",
                        },
                        "top_k": {
                            "type": "integer",
                            "description": "Number of clusters to return (default 3)",
                            "default": 3,
                        },
                    },
                    "required": ["query"],
                },
            ),
            Tool(
                name="expand_cluster",
                description=(
                    "Expand a cluster to see all its individual block summaries. "
                    "Use when a cluster looks relevant but its super-summary lacks detail."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "cluster_id": {
                            "type": "string",
                            "description": "The cluster_id returned by search_clusters",
                        }
                    },
                    "required": ["cluster_id"],
                },
            ),
            Tool(
                name="fetch_raw_block",
                description=(
                    "Fetch the full raw text of a specific block from disk. "
                    "Use only when summaries are insufficient and you need exact "
                    "quotes, numbers, or names."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "block_id": {
                            "type": "string",
                            "description": "The block_id returned by search_clusters or expand_cluster",
                        }
                    },
                    "required": ["block_id"],
                },
            ),
        ]

    @server.call_tool()
    async def _call_tool(name: str, arguments: dict | None) -> list[TextContent]:
        arguments = arguments or {}
        try:
            if name == "search_clusters":
                result = _search_clusters(
                    query=arguments["query"],
                    top_k=int(arguments.get("top_k", 3)),
                )
            elif name == "expand_cluster":
                result = _expand_cluster(cluster_id=arguments["cluster_id"])
            elif name == "fetch_raw_block":
                result = _fetch_raw_block(block_id=arguments["block_id"])
            else:
                result = json.dumps({"error": f"Unknown tool: {name}"})
        except Exception as exc:  # noqa: BLE001
            result = json.dumps({"error": str(exc)})

        return [TextContent(type="text", text=result)]

    async with stdio_server() as streams:
        await server.run(
            streams[0],
            streams[1],
            server.create_initialization_options(),
        )


# ── Entry points ──────────────────────────────────────────────────────────────


def serve_stdio() -> None:
    """
    Registered as ``context-optimizer-mcp`` by pyproject.toml.

    MCP clients (VS Code, Claude Desktop, Cursor) spawn this binary
    directly over stdin/stdout — no HTTP, no auth tokens needed.
    """
    parser = argparse.ArgumentParser(
        prog="context-optimizer-mcp",
        description="context-optimizer MCP server (stdio transport)",
    )
    parser.add_argument(
        "--index",
        default=os.getenv("CONTEXT_OPTIMIZER_INDEX_DIR", str(Path.home() / ".co" / "index")),
        metavar="DIR",
        help="Path to the built index directory (default: ~/.co/index)",
    )
    args = parser.parse_args()

    # Load .env so API keys / model overrides are available
    try:
        from dotenv import load_dotenv  # type: ignore[import]
        load_dotenv()
    except ImportError:
        pass

    asyncio.run(_run_server(Path(args.index).expanduser().resolve()))


if __name__ == "__main__":
    serve_stdio()
