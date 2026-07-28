"""
TreeIndex -- N-level hierarchical summary index for large-corpus retrieval.

Architecture
------------
Level 0 (raw)
    Raw bytes on disk, accessed via BlockIndex file pointers.

Level 1 (block summaries)
    One semantic-core summary per block, stored in ChromaDB {name}_L1.

Level 2..N (cluster super-summaries)
    Each level clusters the level below it (cluster_size entries per node).
    depth=2 -> L1 blocks + L2 clusters  (default, same as before)
    depth=3 -> L1 + L2 + L3 super-clusters (better for 400 MB+ corpora)

Query path
----------
1. Cosine search L{depth} -> top-k super-clusters
2. Drill down one level at a time via expand_cluster()
3. Reasoning agent: expand_cluster | fetch_raw_block | answer
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from context_optimizer.compressor import CompressedChunk
    from context_optimizer.raw_index import BlockIndex


def _auto_tree_depth(
    cluster_size: int,
    top_k: int = 0,
    max_depth: int = 4,
    n_blocks: int = 0,
    corpus_bytes: int = 0,
    block_bytes: int = 1,
) -> int:
    """
    Compute the minimum depth so the top level has ~cluster_size entries.

    d = max(2, min(ceil(log(n/k) / log(k)) + 1, max_depth))

    Args *n_blocks* (preferred, known after Pass 1) or *corpus_bytes* +
    *block_bytes* for a pre-ingestion estimate.
    """
    target = top_k if top_k > 0 else cluster_size
    actual_n = n_blocks if n_blocks > 0 else max(1, math.ceil(corpus_bytes / block_bytes))
    if actual_n <= cluster_size:
        return 2
    raw = math.log(actual_n / target) / math.log(cluster_size)
    return max(2, min(int(math.ceil(raw)) + 1, max_depth))


@dataclass
class BlockHit:
    block_id: str
    summary: str
    cluster_id: str
    distance: float


@dataclass
class ClusterHit:
    cluster_id: str
    super_summary: str
    distance: float
    child_block_ids: list[str] = field(default_factory=list)
    block_hits: list[BlockHit] = field(default_factory=list)


class TreeIndex:
    """N-level hierarchical summary index (configurable depth)."""

    _CLUSTER_PROMPT = (
        "Produce a single compressed semantic core covering ALL block summaries below. "
        "Same format as inputs: noun phrases, key verbs, proper nouns, dates, numbers "
        "-- zero filler words.\n\n"
        "This entry allows a query to decide whether ANY of the {n} blocks are relevant.\n\n"
        "Aim for 150-200 tokens. Semicolons between phrases. No prose, no JSON.\n\n"
        "Block summaries:\n{summaries}\n\nCompressed cluster core:"
    )

    def __init__(
        self,
        collection_name: str = "tree",
        persist_directory: str = "./tree_db",
        block_index: "BlockIndex | None" = None,
        embedding_backend: str | None = None,
        embedding_model_name: str | None = None,
        depth: int = 2,
    ) -> None:
        self.collection_name = collection_name
        self._block_index = block_index
        self._depth = max(1, depth)

        import chromadb
        from chromadb.config import Settings
        from context_optimizer.cached_retriever import CachedChromaRetriever

        Path(persist_directory).mkdir(parents=True, exist_ok=True)
        self._persist_dir = persist_directory
        self._client = chromadb.PersistentClient(
            path=str(persist_directory),
            settings=Settings(anonymized_telemetry=False, allow_reset=True),
        )

        _dummy = CachedChromaRetriever(
            collection_name="tree_init_dummy",
            persist_directory=persist_directory,
            embedding_backend=embedding_backend,
            embedding_model_name=embedding_model_name,
            cache_size=1,
        )
        emb_fn = _dummy.chroma_embedding_fn

        # Auto-detect the actual depth from existing collections so that a
        # pre-built index with depth=3 is loaded correctly even when the
        # caller passes depth=2 (the default).
        existing_names = {c.name for c in self._client.list_collections()}
        detected = 1
        for probe in range(1, 10):
            if f"{collection_name}_L{probe}" in existing_names:
                detected = probe
            else:
                break
        if detected > self._depth:
            self._depth = detected

        self._levels: dict[int, Any] = {}
        for lvl in range(1, self._depth + 1):
            self._levels[lvl] = self._client.get_or_create_collection(
                name=f"{collection_name}_L{lvl}",
                embedding_function=emb_fn,
                metadata={"hnsw:space": "cosine"},
            )

        counts = "  ".join(
            f"L{lvl}={self._levels[lvl].count()}" for lvl in range(1, self._depth + 1)
        )
        print(f"[TreeIndex] {counts}  ({collection_name})")

    # Backward-compat properties

    @property
    def _l1(self) -> Any:
        return self._levels[1]

    @property
    def _l2(self) -> Any:
        return self._levels.get(2, self._levels[1])

    # Build

    def build_from_chunks(
        self,
        chunks: "list[CompressedChunk]",
        cluster_size: int = 4,
        llm: Any | None = None,
        label: str = "",
    ) -> None:
        """
        Populate L1 from chunks, then build L2..L{depth} super-summaries.

        Each level above L1 clusters the level below it.  With depth=2 this
        is identical to the original behaviour.  With depth=3 an L3 level
        provides a coarser entry-point that scales better for large corpora.
        """
        _pfx = f"[{label}] " if label else ""

        # L1: populate from compressed blocks
        l1 = self._levels[1]
        if l1.count() == 0:
            print(f"{_pfx}[TreeIndex] Adding {len(chunks)} L1 block summaries ...")
            ids, docs, metas = [], [], []
            for c in chunks:
                ids.append(c.chunk_id)
                docs.append(c.compressed_summary or c.raw_text[:800])
                metas.append(
                    {
                        "source_file": str(c.metadata.get("source_file", "")),
                        "byte_start": int(c.metadata.get("byte_start", 0)),
                        "byte_end": int(c.metadata.get("byte_end", 0)),
                        "block_idx": int(c.metadata.get("block_idx", 0)),
                    }
                )
            l1.add(ids=ids, documents=docs, metadatas=metas)
            # Force a dummy query so ChromaDB flushes the HNSW segment files to
            # disk immediately.  Without this, PersistentClient may defer the
            # flush and the HNSW index files are absent on the next server start,
            # causing "Error creating hnsw segment reader: Nothing found on disk".
            try:
                l1.query(query_texts=["warmup"], n_results=1)
            except Exception:
                pass
            print(f"{_pfx}[TreeIndex] L1 ready - {l1.count()} entries")
        else:
            print(
                f"{_pfx}[TreeIndex] L1 already populated ({l1.count()} entries), skipping"
            )

        # L2..LN: iteratively cluster the level below
        prev_ids = [c.chunk_id for c in chunks]
        prev_docs = [c.compressed_summary or c.raw_text[:400] for c in chunks]

        for lvl in range(2, self._depth + 1):
            curr = self._levels[lvl]
            if curr.count() > 0:
                print(
                    f"{_pfx}[TreeIndex] Pass {lvl}: L{lvl} already populated ({curr.count()} entries), skipping"
                )
                existing = curr.get(include=["documents"])
                prev_ids = existing["ids"]
                prev_docs = existing["documents"]
                continue

            n_input = len(prev_ids)
            n_clusters = max(1, (n_input + cluster_size - 1) // cluster_size)
            print(
                f"{_pfx}[TreeIndex] Pass {lvl}: {n_input} L{lvl-1} nodes "
                f"-> {n_clusters} L{lvl} clusters  "
                f"(cluster_size={cluster_size}, LLM calls={n_clusters}, "
                f"same model as Pass 1) ..."
            )
            new_ids, new_docs, new_metas = [], [], []
            for ci in range(n_clusters):
                start = ci * cluster_size
                end = min(start + cluster_size, len(prev_ids))
                child_ids = prev_ids[start:end]
                child_sums = prev_docs[start:end]
                cluster_id = f"cluster_L{lvl}_{ci:04d}"
                t0 = time.perf_counter()

                if llm is not None:
                    prompt = self._CLUSTER_PROMPT.format(
                        n=len(child_ids),
                        summaries="\n---\n".join(child_sums),
                    )
                    try:
                        resp = llm.invoke(prompt)
                        super_summary = (
                            resp.content if hasattr(resp, "content") else str(resp)
                        ).strip()[:800]
                    except Exception as exc:
                        print(
                            f"{_pfx}[TreeIndex] WARNING L{lvl} cluster {ci} error: {exc}"
                        )
                        super_summary = " ; ".join(child_sums)[:800]
                else:
                    super_summary = " ; ".join(child_sums)[:1200]

                elapsed = time.perf_counter() - t0
                print(
                    f"{_pfx}[TreeIndex] L{lvl} cluster {ci+1}/{n_clusters} "
                    f"({len(child_ids)} items -> {len(super_summary)//4} tok)  "
                    f"{elapsed:.1f}s"
                )
                new_ids.append(cluster_id)
                new_docs.append(super_summary)
                new_metas.append(
                    {
                        "child_ids": ",".join(child_ids),
                        "item_start": start,
                        "item_end": end,
                        "level_below": lvl - 1,
                    }
                )

            curr.add(ids=new_ids, documents=new_docs, metadatas=new_metas)
            print(f"{_pfx}[TreeIndex] L{lvl} ready - {curr.count()} entries")
            prev_ids = new_ids
            prev_docs = new_docs

    # Search

    def _query_level(
        self,
        level: int,
        query: str,
        child_ids: list[str] | None,
        top_k: int,
    ) -> list[tuple[str, str, float]]:
        """Query one ChromaDB level; return (id, doc, distance) triples."""
        coll = self._levels[level]
        n = min(top_k, coll.count())
        if n == 0:
            return []
        try:
            res = coll.query(
                query_texts=[query],
                n_results=n,
                include=["documents", "distances"],
            )
            out = []
            for j in range(len(res["ids"][0])):
                rid = res["ids"][0][j]
                if child_ids is None or rid in child_ids:
                    out.append((rid, res["documents"][0][j], res["distances"][0][j]))
            # If child_ids were specified but the query returned none of them
            # (all top-k slots taken by blocks belonging to other clusters),
            # fall back to fetching those ids directly so block_hits is never
            # silently empty when we know the children exist.
            if child_ids and not out:
                got = coll.get(ids=child_ids[:top_k], include=["documents"])
                return [(i, d, 1.0) for i, d in zip(got["ids"], got["documents"])]
            return out
        except Exception:
            if child_ids:
                got = coll.get(ids=child_ids[:top_k], include=["documents"])
                return [(i, d, 0.5) for i, d in zip(got["ids"], got["documents"])]
            return []

    def search(
        self,
        query: str,
        top_clusters: int = 2,
        top_blocks_per_cluster: int = 3,
        gap: float = 0.03,
    ) -> list[ClusterHit]:
        """
        Hierarchical search from L{depth} down to L{depth-1}.

        depth=2: block_hits are L1 block summaries (identical to original).
        depth>=3: block_hits are L{depth-1} cluster summaries; use
                  expand_cluster() to drill one level deeper.

        *gap* is an absolute cosine-distance gap.  After retrieving
        ``top_clusters`` candidates the method prunes any cluster whose
        distance exceeds ``best_distance + gap``, keeping at least one
        result.  The same threshold (widened by 2×) is applied to blocks
        within each selected cluster.  Fallback blocks (distance == 1.0,
        returned when the filter yields nothing) are excluded unless they
        are the only hits available.
        """
        top_lvl = self._depth
        top_hits = self._query_level(top_lvl, query, None, top_clusters)
        if not top_hits:
            return []

        # ── Cluster-level gap pruning ─────────────────────────────────────────
        if gap > 0 and len(top_hits) > 1:
            best_dist = top_hits[0][2]
            cutoff = best_dist + gap
            pruned = [h for h in top_hits if h[2] <= cutoff]
            top_hits = pruned if pruned else top_hits[:1]

        child_lvl = max(1, top_lvl - 1)
        cluster_hits: list[ClusterHit] = []

        for cluster_id, super_sum, dist in top_hits:
            result = self._levels[top_lvl].get(ids=[cluster_id], include=["metadatas"])
            meta = result["metadatas"][0] if result["ids"] else {}
            raw_ids = meta.get("child_ids", meta.get("child_block_ids", ""))
            child_ids = [c for c in raw_ids.split(",") if c]

            child_hits_raw = self._query_level(
                child_lvl, query, child_ids, top_blocks_per_cluster
            )

            # ── Block-level gap pruning ───────────────────────────────────────
            # Separate real semantic hits from fallback hits (distance == 1.0).
            real = [h for h in child_hits_raw if h[2] < 0.999]
            if real and gap > 0:
                best_block_dist = real[0][2]
                block_cutoff = best_block_dist + gap * 2
                filtered = [h for h in real if h[2] <= block_cutoff]
                child_hits_raw = filtered if filtered else real[:1]
            elif real:
                child_hits_raw = real
            # else: no real hits — keep the fallback entries as-is

            block_hits = [
                BlockHit(block_id=bid, summary=doc, cluster_id=cluster_id, distance=d)
                for bid, doc, d in child_hits_raw
            ]
            cluster_hits.append(
                ClusterHit(
                    cluster_id=cluster_id,
                    super_summary=super_sum,
                    distance=dist,
                    child_block_ids=child_ids,
                    block_hits=block_hits,
                )
            )

        return cluster_hits

    def expand_cluster(
        self,
        cluster_id: str,
        query: str = "",
        top_k: int = 5,
    ) -> list[BlockHit]:
        """
        Drill one level below cluster_id.

        L3 cluster -> returns L2 summaries.
        L2 cluster -> returns L1 block summaries.
        The reasoning agent calls this iteratively to navigate toward raw blocks.
        """
        for lvl in range(self._depth, 1, -1):
            result = self._levels[lvl].get(ids=[cluster_id], include=["metadatas"])
            if not result["ids"]:
                continue
            meta = result["metadatas"][0]
            raw_ids = meta.get("child_ids", meta.get("child_block_ids", ""))
            child_ids = [c for c in raw_ids.split(",") if c]
            hits_raw = self._query_level(lvl - 1, query or cluster_id, child_ids, top_k)
            return [
                BlockHit(block_id=bid, summary=doc, cluster_id=cluster_id, distance=d)
                for bid, doc, d in hits_raw
            ]
        return []

    def get_raw_block(self, block_id: str) -> str | None:
        if self._block_index is None:
            return None
        return self._block_index.get_text(block_id)

    def get_block_summary(self, block_id: str) -> str | None:
        result = self._levels[1].get(ids=[block_id])
        if result["ids"]:
            return result["documents"][0]
        return None

    def cluster_count(self) -> int:
        """Total entries across all levels above L1."""
        return sum(self._levels[lvl].count() for lvl in range(2, self._depth + 1))

    def block_count(self) -> int:
        return self._levels[1].count()

    def depth(self) -> int:
        return self._depth

    def __repr__(self) -> str:
        counts = "  ".join(
            f"L{lvl}={self._levels[lvl].count()}" for lvl in range(1, self._depth + 1)
        )
        return f"TreeIndex({self.collection_name!r}, {counts})"
