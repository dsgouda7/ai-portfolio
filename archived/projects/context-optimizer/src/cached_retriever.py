"""
Semantic Cache with Local Embeddings

Two-tier retrieval:
1. Semantic Cache (in-memory): Fast similarity check on recent queries
2. ChromaDB (persistent): Fallback for cache misses

Uses local sentence-transformers model (no API costs, ~50ms CPU).
"""

from __future__ import annotations

import os
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np

# Local embedding model
try:
    from sentence_transformers import SentenceTransformer

    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print(
        "[WARNING] sentence-transformers not installed. Install with: pip install sentence-transformers"
    )

import chromadb
from chromadb.config import Settings

from .compressor import CompressedChunk, split_into_sub_chunks


class _OllamaEmbedder:
    """
    Thin wrapper around Ollama's /api/embeddings endpoint that matches
    the SentenceTransformer.encode() interface expected by SemanticCache.
    """

    def __init__(
        self, model: str = "nomic-embed-text", base_url: str = "http://localhost:11434"
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")

    def encode(
        self,
        texts: str | list[str],
        convert_to_numpy: bool = True,
        show_progress_bar: bool = False,
    ) -> "np.ndarray":
        import json
        import urllib.request

        single = isinstance(texts, str)
        if single:
            texts = [texts]
        embeddings = []
        for text in texts:
            payload = json.dumps({"model": self.model, "prompt": text}).encode()
            req = urllib.request.Request(
                f"{self.base_url}/api/embeddings",
                data=payload,
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req) as resp:
                data = json.loads(resp.read())
            embeddings.append(data["embedding"])
        arr = np.array(embeddings, dtype=np.float32)
        return arr[0] if single else arr


class SemanticCache:
    """
    In-memory semantic cache for query results.

    Caches query embeddings + retrieved chunks.
    On new query, checks if similar query exists in cache (cosine similarity).

    Much faster than vector DB for repetitive queries (1-2ms vs 10-50ms).
    """

    def __init__(
        self,
        embedding_model: SentenceTransformer,
        max_size: int = 1000,
        similarity_threshold: float = 0.85,
    ):
        """
        Initialize semantic cache

        Args:
            embedding_model: Local sentence-transformers model
            max_size: Max cached queries (LRU eviction)
            similarity_threshold: Min cosine similarity to consider a cache hit (0.85 = 85%)
        """
        self.embedding_model = embedding_model
        self.max_size = max_size
        self.similarity_threshold = similarity_threshold

        # LRU cache: query_text -> (query_embedding, retrieved_chunks, timestamp)
        self.cache: OrderedDict[str, tuple[np.ndarray, list[dict], float]] = (
            OrderedDict()
        )

        self.hits = 0
        self.misses = 0

    def _embed_query(self, query: str) -> np.ndarray:
        """Embed query using local model"""
        return self.embedding_model.encode(
            query, convert_to_numpy=True, show_progress_bar=False
        )

    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors"""
        return float(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))

    def get(self, query: str) -> list[dict] | None:
        """
        Check cache for semantically similar query.

        Fast path: exact string match (O(1), no embedding).
        Slow path: cosine similarity over all cached embeddings (O(N)).

        Returns:
            Cached results if similar query found, None otherwise
        """
        if not self.cache:
            self.misses += 1
            return None

        # Fast path: exact match (no embedding needed — handles repeat queries in < 1ms)
        if query in self.cache:
            self.hits += 1
            self.cache.move_to_end(query)
            return self.cache[query][1]

        # Slow path: semantic similarity
        query_embedding = self._embed_query(query)

        best_match = None
        best_similarity = 0.0

        for cached_query, (cached_embedding, cached_results, _) in self.cache.items():
            similarity = self._cosine_similarity(query_embedding, cached_embedding)

            if similarity > best_similarity:
                best_similarity = similarity
                best_match = (cached_query, cached_results)

        # Cache hit if similarity above threshold
        if best_similarity >= self.similarity_threshold:
            self.hits += 1
            # Move to end (LRU)
            self.cache.move_to_end(best_match[0])
            return best_match[1]

        self.misses += 1
        return None

    def put(self, query: str, results: list[dict]) -> None:
        """Add query results to cache"""

        # Embed query
        query_embedding = self._embed_query(query)

        # Add to cache
        self.cache[query] = (query_embedding, results, time.time())
        self.cache.move_to_end(query)

        # Evict oldest if over capacity (LRU)
        if len(self.cache) > self.max_size:
            self.cache.popitem(last=False)

    def clear(self) -> None:
        """Clear entire cache"""
        self.cache.clear()
        self.hits = 0
        self.misses = 0

    def get_stats(self) -> dict:
        """Get cache statistics"""
        total_queries = self.hits + self.misses
        hit_rate = (self.hits / total_queries * 100) if total_queries > 0 else 0.0

        return {
            "cache_size": len(self.cache),
            "max_size": self.max_size,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate_pct": hit_rate,
            "similarity_threshold": self.similarity_threshold,
        }


class CachedChromaRetriever:
    """
    Two-tier retrieval: Semantic Cache → ChromaDB

    1. Check semantic cache (1-2ms, in-memory)
    2. On miss, query ChromaDB (10-50ms, persistent)
    3. Cache result for future queries

    Best of both worlds: fast + scalable.
    """

    def __init__(
        self,
        collection_name: str = "compressed_chunks",
        persist_directory: str = "./chroma_db",
        embedding_model_name: str | None = None,
        cache_size: int = 1000,
        cache_threshold: float = 0.85,
        embedding_backend: str | None = None,
        raw_index: "Any | None" = None,
    ):
        """
        Initialize cached retriever with pluggable embedding backend.

        Args:
            collection_name:    ChromaDB collection name
            persist_directory:  Where to store ChromaDB
            embedding_model_name: Model name (overrides env default)
            cache_size:         Max cached queries
            cache_threshold:    Semantic similarity threshold for cache hit
            embedding_backend:  "sentence-transformers" (default) | "ollama"
                                Overrides CONTEXT_OPTIMIZER_EMBEDDING_BACKEND env var.
            raw_index:          Optional :class:`~context_optimizer.raw_index.RawIndex`
                                instance.  When provided, :meth:`get_chunk_by_id`
                                uses an O(1) primary-key lookup instead of a
                                ChromaDB ``.get()`` call (~5 ms → ~0.1 ms).
                                Also eliminates the 4 000-char truncation on
                                ``raw_text`` stored in ChromaDB metadata.

        Environment variables (fallbacks):
            CONTEXT_OPTIMIZER_EMBEDDING_BACKEND  sentence-transformers | ollama
            CONTEXT_OPTIMIZER_EMBEDDING_MODEL    model name
            OLLAMA_BASE_URL                      http://localhost:11434
        """
        self._raw_index = raw_index
        backend = (
            embedding_backend
            or os.getenv("CONTEXT_OPTIMIZER_EMBEDDING_BACKEND", "sentence-transformers")
        ).lower()

        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)

        # ── Embedding model ────────────────────────────────────────────────
        if backend == "ollama":
            ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
            model_name = embedding_model_name or os.getenv(
                "CONTEXT_OPTIMIZER_EMBEDDING_MODEL", "nomic-embed-text"
            )
            print(f"[CachedRetriever] Embedding backend : Ollama ({ollama_url})")
            print(f"[CachedRetriever] Embedding model   : {model_name}")
            self.embedding_model = _OllamaEmbedder(
                model=model_name, base_url=ollama_url
            )
            from chromadb.utils import embedding_functions

            self.chroma_embedding_fn = embedding_functions.OllamaEmbeddingFunction(
                url=f"{ollama_url}/api/embeddings",
                model_name=model_name,
            )
        else:
            if not SENTENCE_TRANSFORMERS_AVAILABLE:
                raise ImportError(
                    "sentence-transformers required. "
                    "Install with: pip install sentence-transformers  "
                    "OR set CONTEXT_OPTIMIZER_EMBEDDING_BACKEND=ollama"
                )
            model_name = embedding_model_name or os.getenv(
                "CONTEXT_OPTIMIZER_EMBEDDING_MODEL", "all-MiniLM-L6-v2"
            )
            print(f"[CachedRetriever] Embedding backend : sentence-transformers")
            print(f"[CachedRetriever] Embedding model   : {model_name}")
            self.embedding_model = SentenceTransformer(model_name)
            from chromadb.utils import embedding_functions

            self.chroma_embedding_fn = (
                embedding_functions.SentenceTransformerEmbeddingFunction(
                    model_name=model_name
                )
            )

        # ── ChromaDB client ────────────────────────────────────────────────
        self.client = chromadb.PersistentClient(
            path=str(self.persist_directory),
            settings=Settings(anonymized_telemetry=False, allow_reset=True),
        )

        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.chroma_embedding_fn,
            metadata={"hnsw:space": "cosine"},
        )

        # ── Child index (parent-child multi-vector retrieval) ──────────────
        # Raw sub-chunks (200-token windows) indexed with exact vocabulary.
        # When a specific keyword is "blurred" by the LLM summariser, the
        # child index still captures it so the parent summary can be returned.
        # Name is derived from the parent collection to stay grouped on disk.
        self.children_collection = self.client.get_or_create_collection(
            name=f"{collection_name}__children",
            embedding_function=self.chroma_embedding_fn,
            metadata={"hnsw:space": "cosine"},
        )

        # ── Semantic cache ─────────────────────────────────────────────────
        self.cache = SemanticCache(
            embedding_model=self.embedding_model,
            max_size=cache_size,
            similarity_threshold=cache_threshold,
        )

        print(
            f"[CachedRetriever] Collection : '{collection_name}'  ({self.collection.count():,} chunks)"
        )
        print(
            f"[CachedRetriever] Children   : '{collection_name}__children'  "
            f"({self.children_collection.count():,} sub-chunks)"
        )
        print(f"[CachedRetriever] Storage    : {self.persist_directory}")
        print(
            f"[CachedRetriever] Cache      : {cache_size} queries  threshold={cache_threshold*100:.0f}%"
        )

    def add_chunks(self, chunks: list[CompressedChunk], batch_size: int = 100) -> None:
        """Add compressed chunks to ChromaDB"""
        if not chunks:
            return

        print(f"[CachedRetriever] Adding {len(chunks):,} chunks...")

        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]

            ids = [chunk.chunk_id for chunk in batch]
            documents = [chunk.compressed_summary for chunk in batch]

            def _safe_meta(v):
                """ChromaDB 0.5+ only accepts str/int/float/bool metadata values."""
                if isinstance(v, (str, int, float, bool)):
                    return v
                if isinstance(v, (list, tuple)):
                    return ",".join(str(x) for x in v)
                return str(v) if v is not None else ""

            metadatas = [
                {
                    "entities": ",".join(chunk.entities),
                    "keywords": ",".join(chunk.keywords),
                    "original_tokens": chunk.original_tokens,
                    "compressed_tokens": chunk.compressed_tokens,
                    "compression_ratio": float(chunk.compression_ratio),
                    # Pointer model: raw text stored but NOT indexed in the vector space.
                    # Fetched on demand via get_chunk_by_id(); keeps the index 10x smaller.
                    "raw_text": chunk.raw_text[:4000],
                    **{k: _safe_meta(v) for k, v in chunk.metadata.items()},
                }
                for chunk in batch
            ]

            self.collection.add(ids=ids, documents=documents, metadatas=metadatas)

            if (i + batch_size) % 500 == 0:
                print(
                    f"  [Progress] {min(i + batch_size, len(chunks)):,}/{len(chunks):,}"
                )

        print(f"[CachedRetriever] [OK] Added {len(chunks):,} chunks")

    def search(
        self,
        query: str,
        top_k: int = 5,
        where_filter: dict[str, Any] | None = None,
        use_cache: bool = True,
    ) -> list[dict]:
        """
        Search with semantic cache + ChromaDB fallback

        Args:
            query: Search query
            top_k: Number of results
            where_filter: Metadata filters
            use_cache: Whether to use cache (default: True)

        Returns:
            List of results with timing info
        """
        start_time = time.time()

        # Check cache first
        if use_cache:
            cached_results = self.cache.get(query)
            if cached_results is not None:
                latency = (time.time() - start_time) * 1000
                print(f"[CachedRetriever] Cache HIT ({latency:.2f}ms)")
                return cached_results

        # Cache miss - query ChromaDB
        results = self.collection.query(
            query_texts=[query], n_results=top_k, where=where_filter
        )

        # Format results
        hits = []
        for i in range(len(results["ids"][0])):
            meta = results["metadatas"][0][i]
            hits.append(
                {
                    "chunk_id": results["ids"][0][i],
                    "compressed_summary": results["documents"][0][i],
                    # raw_text bubbled to top level for direct access by reasoner
                    "raw_text": meta.get("raw_text", ""),
                    "metadata": meta,
                    "distance": (
                        results["distances"][0][i] if "distances" in results else None
                    ),
                    "entities": (
                        meta["entities"].split(",") if meta.get("entities") else []
                    ),
                    "keywords": (
                        meta["keywords"].split(",") if meta.get("keywords") else []
                    ),
                }
            )

        # Cache results
        if use_cache:
            self.cache.put(query, hits)

        latency = (time.time() - start_time) * 1000
        print(f"[CachedRetriever] Cache MISS ({latency:.2f}ms, queried ChromaDB)")

        return hits

    def get_stats(self) -> dict:
        """Get retriever + cache statistics"""
        cache_stats = self.cache.get_stats()

        return {
            "collection_name": self.collection.name,
            "total_chunks": self.collection.count(),
            "child_chunks": self.children_collection.count(),
            "storage_path": str(self.persist_directory),
            "embedding_model": self.embedding_model.get_sentence_embedding_dimension(),
            "cache": cache_stats,
        }

    # ── Parent-child multi-vector retrieval ───────────────────────────────────

    def add_raw_sub_chunks(
        self,
        chunks: list[CompressedChunk],
        sub_chunk_tokens: int = 200,
        batch_size: int = 100,
    ) -> int:
        """
        Index raw sub-chunks for parent-child (multi-vector) retrieval.

        Splits each ``CompressedChunk.raw_text`` into smaller windows of
        *sub_chunk_tokens* tokens and stores them in the children collection
        with exact source vocabulary.  Granular keywords that an LLM summariser
        might drop (e.g. ``"pocket watch"``, ``"gold chain"``) are preserved in
        the child embeddings.

        On retrieval via :meth:`search_with_child_index`, the best-matching
        child's ``parent_chunk_id`` is used to look up and return the parent
        compressed summary — giving the reasoning model coherent context while
        ensuring specific keyword queries still hit the right chunk.

        Parameters
        ----------
        chunks:
            Compressed chunks (parent level); their ``raw_text`` is split.
        sub_chunk_tokens:
            Target token budget per sub-chunk (default 200 ≈ 800 chars).
        batch_size:
            ChromaDB write batch size.

        Returns
        -------
        int
            Total number of sub-chunks indexed.
        """
        all_ids: list[str] = []
        all_docs: list[str] = []
        all_meta: list[dict] = []

        for chunk in chunks:
            sub_texts = split_into_sub_chunks(
                chunk.raw_text, sub_chunk_tokens=sub_chunk_tokens
            )
            for sub_idx, sub_text in enumerate(sub_texts):
                sub_id = f"{chunk.chunk_id}__child_{sub_idx:04d}"
                all_ids.append(sub_id)
                all_docs.append(sub_text)
                all_meta.append(
                    {
                        "parent_chunk_id": chunk.chunk_id,
                        "sub_chunk_idx": sub_idx,
                        "parent_entities": ",".join(chunk.entities),
                    }
                )

        if not all_ids:
            return 0

        for i in range(0, len(all_ids), batch_size):
            self.children_collection.add(
                ids=all_ids[i : i + batch_size],
                documents=all_docs[i : i + batch_size],
                metadatas=all_meta[i : i + batch_size],
            )

        print(
            f"[CachedRetriever] Child index: added {len(all_ids):,} sub-chunks "
            f"from {len(chunks):,} parent chunks"
        )
        return len(all_ids)

    def search_with_child_index(
        self,
        query: str,
        top_k: int = 5,
        use_cache: bool = True,
    ) -> list[dict]:
        """
        Parent-child multi-vector retrieval.

        Searches raw sub-chunks first (preserving exact vocabulary), then maps
        each matched child back to its parent compressed summary.  Falls back
        to :meth:`search` if the children collection is empty.

        Algorithm
        ---------
        1. Embed the query and search ``{collection}__children`` (raw sub-chunks).
        2. Map each child hit to its ``parent_chunk_id``.
        3. Deduplicate by ``parent_chunk_id``; keep best child distance per parent.
        4. Fetch parent summaries from the main collection.

        When a query targets a specific detail that was scrubbed by the LLM
        (e.g. ``"pocket watch"``) the child index hits the precise 200-token
        window.  The parent summary is then returned so the reasoning model
        has coherent context while the retrieval is driven by exact keywords.

        Parameters
        ----------
        query:
            User query text.
        top_k:
            How many parent results to return.
        use_cache:
            Whether to use the semantic cache (default True).

        Returns
        -------
        list[dict]
            Same format as :meth:`search`, sourced via child→parent lookup.
        """
        if self.children_collection.count() == 0:
            return self.search(query, top_k=top_k, use_cache=use_cache)

        cache_key = f"__child__{query}"
        if use_cache:
            cached = self.cache.get(cache_key)
            if cached is not None:
                return cached

        # Fetch more children to allow deduplication by parent
        n_children = min(top_k * 5, self.children_collection.count())
        child_results = self.children_collection.query(
            query_texts=[query],
            n_results=n_children,
        )

        # Deduplicate: best (lowest) distance per parent_chunk_id
        seen_parents: dict[str, float] = {}
        parent_order: list[str] = []
        for i in range(len(child_results["ids"][0])):
            parent_id = child_results["metadatas"][0][i]["parent_chunk_id"]
            dist = (
                child_results["distances"][0][i]
                if "distances" in child_results
                else 0.0
            )
            if parent_id not in seen_parents:
                seen_parents[parent_id] = dist
                parent_order.append(parent_id)
            else:
                seen_parents[parent_id] = min(seen_parents[parent_id], dist)

        parent_order.sort(key=lambda pid: seen_parents[pid])
        top_parent_ids = parent_order[:top_k]

        # Fetch parent summaries from main collection
        hits: list[dict] = []
        for parent_id in top_parent_ids:
            result = self.collection.get(ids=[parent_id])
            if not result["ids"]:
                continue
            meta = result["metadatas"][0]
            hits.append(
                {
                    "chunk_id": parent_id,
                    "compressed_summary": result["documents"][0],
                    "raw_text": meta.get("raw_text", ""),
                    "metadata": meta,
                    "distance": seen_parents[parent_id],
                    "entities": (
                        meta.get("entities", "").split(",")
                        if meta.get("entities")
                        else []
                    ),
                    "keywords": (
                        meta.get("keywords", "").split(",")
                        if meta.get("keywords")
                        else []
                    ),
                }
            )

        if use_cache:
            self.cache.put(cache_key, hits)

        return hits

    def get_chunk_by_id(self, chunk_id: str | list) -> dict | None:
        """
        Retrieve a specific chunk by ID, including raw_text.

        ``chunk_id`` may be a list if the reasoning LLM returned multiple IDs
        in a single JSON response; the first element is used in that case.

        Uses a two-step lookup strategy:

        1. **RawIndex fast path** (~0.1 ms) — if a
           :class:`~context_optimizer.raw_index.RawIndex` was provided at
           construction time, do an O(1) primary-key SQLite lookup.  This
           also returns the *full*, un-truncated raw text.
        2. **ChromaDB fallback** (~5 ms) — fetch from ChromaDB metadata.
           Raw text is truncated to 4 000 chars in this path.

        Parameters
        ----------
        chunk_id:
            Chunk identifier, e.g. ``"chunk_000042"``.

        Returns
        -------
        dict | None
            A dict with at least ``chunk_id``, ``raw_text``,
            ``compressed_summary``, ``entities``, and ``keywords`` keys.
            Returns ``None`` if the chunk is not found.
        """
        # Normalise: LLM sometimes returns a list of IDs; use the first.
        if isinstance(chunk_id, list):
            chunk_id = chunk_id[0] if chunk_id else None
        if not chunk_id:
            return None

        # ── Fast path: RawIndex primary-key lookup ────────────────────────
        if self._raw_index is not None:
            raw = self._raw_index.get(chunk_id)
            if raw is not None:
                # Raw text found; fetch compressed summary from ChromaDB for completeness
                chroma_result = self.collection.get(ids=[chunk_id])
                if chroma_result["ids"]:
                    meta = chroma_result["metadatas"][0]
                    return {
                        "chunk_id": chunk_id,
                        "compressed_summary": chroma_result["documents"][0],
                        "metadata": meta,
                        "raw_text": raw,  # full, un-truncated
                        "entities": (
                            meta.get("entities", "").split(",")
                            if meta.get("entities")
                            else []
                        ),
                        "keywords": (
                            meta.get("keywords", "").split(",")
                            if meta.get("keywords")
                            else []
                        ),
                        "source": "raw_index",
                    }
                # RawIndex hit but not in ChromaDB yet — return partial record
                return {
                    "chunk_id": chunk_id,
                    "compressed_summary": "",
                    "metadata": {},
                    "raw_text": raw,
                    "entities": [],
                    "keywords": [],
                    "source": "raw_index",
                }

        # ── Fallback: ChromaDB metadata (raw_text truncated at 4 000 chars) ─
        results = self.collection.get(ids=[chunk_id])

        if not results["ids"]:
            return None

        return {
            "chunk_id": results["ids"][0],
            "compressed_summary": results["documents"][0],
            "metadata": results["metadatas"][0],
            "raw_text": results["metadatas"][0].get("raw_text", ""),
            "entities": (
                results["metadatas"][0]["entities"].split(",")
                if results["metadatas"][0].get("entities")
                else []
            ),
            "keywords": (
                results["metadatas"][0]["keywords"].split(",")
                if results["metadatas"][0].get("keywords")
                else []
            ),
            "source": "chromadb",
        }

    def clear_cache(self) -> None:
        """Clear semantic cache"""
        self.cache.clear()
        print("[CachedRetriever] Cache cleared")
