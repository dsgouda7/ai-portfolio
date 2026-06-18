"""
Retrieval backends for experiments.

`SimpleRetriever` remains the lightweight TF-IDF baseline used by OOTB RAG.
`SemanticVectorRetriever` adds semantic chunking plus vector retrieval for
Pipe C's MCP server. It prefers Chroma for local persistence when available,
but can fall back to an in-memory vector index for deterministic mock runs.
"""
from __future__ import annotations

import hashlib
import math
import os
import re
from dataclasses import dataclass
from collections import Counter

try:
    from langchain_chroma import Chroma
except ImportError:  # pragma: no cover
    Chroma = None

try:
    from langchain_ollama import OllamaEmbeddings
except ImportError:  # pragma: no cover
    OllamaEmbeddings = None


def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    left_norm = math.sqrt(sum(value * value for value in left)) or 1.0
    right_norm = math.sqrt(sum(value * value for value in right)) or 1.0
    return sum(l * r for l, r in zip(left, right)) / (left_norm * right_norm)


@dataclass
class SemanticChunk:
    chunk_id: str
    text: str
    summary: str
    metadata: dict[str, str]
    token_count: int


@dataclass
class RetrievalHit:
    chunk_id: str
    text: str
    summary: str
    metadata: dict[str, str]
    vector_score: float
    lexical_score: float

    @property
    def combined_score(self) -> float:
        return (self.vector_score * 0.7) + (self.lexical_score * 0.3)


class HashingEmbeddings:
    """Deterministic local embedding fallback for mock/offline runs."""

    def __init__(self, dimensions: int = 256):
        self.dimensions = dimensions

    @staticmethod
    def _tokens(text: str) -> list[str]:
        return re.findall(r"[a-zA-Z0-9_.:-]+", text.lower())

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        for token in self._tokens(text):
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            bucket = int.from_bytes(digest[:4], "big") % self.dimensions
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[bucket] += sign
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [value / norm for value in vector]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


def build_embeddings(provider: str = "mock") -> tuple[object, str]:
    """Return the configured embedding backend plus a descriptive name."""
    selected = os.getenv("CONTEXT_OPTIMIZER_EMBEDDING_PROVIDER", "").lower()
    if not selected:
        selected = "ollama" if provider == "ollama" else "hash"

    if selected == "ollama" and OllamaEmbeddings is not None:
        model_name = os.getenv("CONTEXT_OPTIMIZER_EMBEDDING_MODEL", "nomic-embed-text")
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        return OllamaEmbeddings(model=model_name, base_url=base_url), f"ollama:{model_name}"

    return HashingEmbeddings(), "hashing-local"


class _InMemoryVectorStore:
    def __init__(self, embeddings: HashingEmbeddings | object, chunks: list[SemanticChunk]):
        self._embeddings = embeddings
        self._chunks = chunks
        self._vectors = self._embed_documents([chunk.text for chunk in chunks])

    def _embed_documents(self, texts: list[str]) -> list[list[float]]:
        embed = getattr(self._embeddings, "embed_documents")
        return embed(texts)

    def _embed_query(self, text: str) -> list[float]:
        embed = getattr(self._embeddings, "embed_query")
        return embed(text)

    def similarity_search(self, query: str, k: int = 5) -> list[tuple[SemanticChunk, float]]:
        query_vector = self._embed_query(query)
        scored = [
            (_cosine_similarity(query_vector, vector), chunk)
            for chunk, vector in zip(self._chunks, self._vectors)
        ]
        scored.sort(key=lambda item: item[0], reverse=True)
        return [(chunk, score) for score, chunk in scored[:k]]


class SemanticVectorRetriever:
    """Semantic chunker + vector retriever with configurable backend."""

    def __init__(
        self,
        corpus: list[str],
        chunk_token_target: int = 160,
        backend: str | None = None,
        persist_directory: str | None = None,
        embeddings: object | None = None,
        provider: str = "mock",
    ):
        self._raw = corpus
        self._chunks = self._build_semantic_chunks(corpus, chunk_token_target)
        self._backend_name = (backend or os.getenv("CONTEXT_OPTIMIZER_VECTOR_BACKEND", "chroma")).lower()
        self._embeddings, self._embedding_name = (embeddings, "custom") if embeddings is not None else build_embeddings(provider)
        self._lexical = SimpleRetriever([chunk.text for chunk in self._chunks], chunk_size=1, overlap=0)
        self._persist_directory = persist_directory or os.getenv(
            "CONTEXT_OPTIMIZER_VECTOR_DIR",
            ".vectorstore/context-optimizer",
        )
        self._vector_store = self._build_vector_store()

    def _build_vector_store(self) -> object:
        if self._backend_name != "chroma" or Chroma is None:
            self._backend_name = "memory"
            return _InMemoryVectorStore(self._embeddings, self._chunks)

        texts = [chunk.text for chunk in self._chunks]
        metadatas = [
            {
                **chunk.metadata,
                "chunk_id": chunk.chunk_id,
                "summary": chunk.summary,
                "token_count": str(chunk.token_count),
            }
            for chunk in self._chunks
        ]
        ids = [chunk.chunk_id for chunk in self._chunks]
        return Chroma.from_texts(
            texts=texts,
            embedding=self._embeddings,
            metadatas=metadatas,
            ids=ids,
            persist_directory=self._persist_directory,
            collection_name="context_optimizer_semantic_chunks",
        )

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return re.findall(r"[a-zA-Z0-9_.:-]+", text.lower())

    @classmethod
    def _build_semantic_chunks(cls, corpus: list[str], chunk_token_target: int) -> list[SemanticChunk]:
        chunks: list[SemanticChunk] = []
        buffer: list[str] = []
        buffer_tokens = 0
        chunk_index = 0

        for line_index, line in enumerate(corpus):
            line_tokens = _estimate_tokens(line)
            if buffer and buffer_tokens + line_tokens > chunk_token_target:
                chunks.append(cls._make_chunk(chunk_index, buffer))
                chunk_index += 1
                buffer = []
                buffer_tokens = 0

            buffer.append(f"[{line_index:04d}] {line}")
            buffer_tokens += line_tokens

        if buffer:
            chunks.append(cls._make_chunk(chunk_index, buffer))

        for index, chunk in enumerate(chunks):
            prev_chunk_id = chunks[index - 1].chunk_id if index > 0 else ""
            next_chunk_id = chunks[index + 1].chunk_id if index + 1 < len(chunks) else ""
            chunk.metadata["prev_chunk_id"] = prev_chunk_id
            chunk.metadata["next_chunk_id"] = next_chunk_id

        return chunks

    @staticmethod
    def _make_chunk(chunk_index: int, lines: list[str]) -> SemanticChunk:
        text = "\n".join(lines)
        line_numbers = []
        services = set()
        severities = set()
        first_payload = ""
        last_payload = ""
        for line in lines:
            match = re.match(r"\[(\d{4})\]\s+(.*)", line)
            if not match:
                continue
            line_numbers.append(match.group(1))
            rest = match.group(2)
            if not first_payload:
                first_payload = rest
            last_payload = rest
            parts = rest.split()
            if len(parts) >= 3:
                severities.add(parts[1])
                services.add(parts[2])
        line_start = line_numbers[0] if line_numbers else f"{chunk_index:04d}"
        line_end = line_numbers[-1] if line_numbers else line_start
        boundary_state = SemanticVectorRetriever._infer_boundary_state(first_payload, last_payload)
        summary = (
            f"lines {line_start}-{line_end}; services={','.join(sorted(services)) or 'unknown'}; "
            f"severities={','.join(sorted(severities)) or 'mixed'}; "
            f"boundary={boundary_state['boundary_reason']}"
        )
        metadata = {
            "line_start": line_start,
            "line_end": line_end,
            "services": ",".join(sorted(services)),
            "severities": ",".join(sorted(severities)),
            "boundary_preserved": boundary_state["boundary_preserved"],
            "boundary_reason": boundary_state["boundary_reason"],
            "needs_prev_chunk": boundary_state["needs_prev_chunk"],
            "needs_next_chunk": boundary_state["needs_next_chunk"],
            "boundary_contract": "preserve original span; do not merge facts across chunk edges",
        }
        return SemanticChunk(
            chunk_id=f"chunk-{chunk_index:04d}",
            text=text,
            summary=summary,
            metadata=metadata,
            token_count=_estimate_tokens(text),
        )

    @staticmethod
    def _infer_boundary_state(first_payload: str, last_payload: str) -> dict[str, str]:
        first = first_payload.lower()
        last = last_payload.lower()

        starts_mid_stack = any(marker in first for marker in [" at ", "system.timeoutexception", "innerexception"])
        ends_mid_stack = any(marker in last for marker in [" at ", "system.timeoutexception", "innerexception"])
        ends_with_open_cause = any(marker in last for marker in ["retries=", "operation=", "upstream:"])

        needs_prev = starts_mid_stack
        needs_next = ends_mid_stack or ends_with_open_cause

        if not needs_prev and not needs_next:
            reason = "closed_span"
        elif needs_prev and needs_next:
            reason = "mid_sequence"
        elif needs_prev:
            reason = "continues_from_previous"
        else:
            reason = "continues_into_next"

        return {
            "boundary_preserved": "true",
            "boundary_reason": reason,
            "needs_prev_chunk": "true" if needs_prev else "false",
            "needs_next_chunk": "true" if needs_next else "false",
        }

    def _lexical_scores(self, query: str) -> dict[str, float]:
        query_tokens = self._tokenize(query)
        scores: dict[str, float] = {}
        for chunk in self._chunks:
            scores[chunk.chunk_id] = self._lexical._tfidf_score(query_tokens, chunk.text)
        max_score = max(scores.values(), default=0.0) or 1.0
        return {chunk_id: score / max_score for chunk_id, score in scores.items()}

    def retrieve(
        self,
        query: str,
        k: int = 5,
        filters: dict[str, str] | None = None,
    ) -> list[RetrievalHit]:
        filters = filters or {}
        lexical_scores = self._lexical_scores(query)

        if self._backend_name == "chroma" and Chroma is not None:
            where = self._build_chroma_filter(filters)
            results = self._vector_store.similarity_search_with_relevance_scores(
                query=query,
                k=max(k * 2, 8),
                filter=where or None,
            )
            hits: list[RetrievalHit] = []
            for document, score in results:
                metadata = {key: str(value) for key, value in (document.metadata or {}).items()}
                chunk_id = metadata.get("chunk_id", "unknown")
                hits.append(
                    RetrievalHit(
                        chunk_id=chunk_id,
                        text=document.page_content,
                        summary=metadata.get("summary", document.page_content.splitlines()[0]),
                        metadata=metadata,
                        vector_score=float(score),
                        lexical_score=lexical_scores.get(chunk_id, 0.0),
                    )
                )
        else:
            results = self._vector_store.similarity_search(query=query, k=max(k * 2, 8))
            hits = [
                RetrievalHit(
                    chunk_id=chunk.chunk_id,
                    text=chunk.text,
                    summary=chunk.summary,
                    metadata=chunk.metadata,
                    vector_score=score,
                    lexical_score=lexical_scores.get(chunk.chunk_id, 0.0),
                )
                for chunk, score in results
            ]

        filtered = [hit for hit in hits if self._matches_filters(hit, filters)]
        filtered.sort(key=lambda hit: hit.combined_score, reverse=True)
        return filtered[:k]

    @staticmethod
    def _build_chroma_filter(filters: dict[str, str]) -> dict[str, str]:
        where: dict[str, str] = {}
        if filters.get("service"):
            where["services"] = filters["service"]
        if filters.get("severity"):
            where["severities"] = filters["severity"]
        return where

    @staticmethod
    def _matches_filters(hit: RetrievalHit, filters: dict[str, str]) -> bool:
        if not filters:
            return True
        service = filters.get("service")
        severity = filters.get("severity")
        if service and service not in hit.metadata.get("services", ""):
            return False
        if severity and severity not in hit.metadata.get("severities", ""):
            return False
        return True

    @property
    def backend_name(self) -> str:
        return self._backend_name

    @property
    def embedding_name(self) -> str:
        return self._embedding_name

    @property
    def chunk_count(self) -> int:
        return len(self._chunks)

    @property
    def line_count(self) -> int:
        return len(self._raw)


class SimpleRetriever:
    """TF-IDF retriever over a list of raw log lines. CPU-safe, zero deps."""

    def __init__(self, corpus: list[str], chunk_size: int = 20, overlap: int = 4):
        self._raw = corpus
        self._chunks: list[str] = self._build_chunks(corpus, chunk_size, overlap)
        self._idf: dict[str, float] = self._compute_idf()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_chunks(self, corpus: list[str], size: int, overlap: int) -> list[str]:
        step = max(1, size - overlap)
        chunks: list[str] = []
        for i in range(0, len(corpus), step):
            chunk = "\n".join(corpus[i : i + size])
            if chunk.strip():
                chunks.append(chunk)
        return chunks

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return re.findall(r"[a-zA-Z0-9]+", text.lower())

    def _compute_idf(self) -> dict[str, float]:
        n = len(self._chunks)
        df: Counter = Counter()
        for chunk in self._chunks:
            df.update(set(self._tokenize(chunk)))
        return {t: math.log((n + 1) / (df[t] + 1)) for t in df}

    def _tfidf_score(self, query_tokens: list[str], chunk: str) -> float:
        chunk_tokens = self._tokenize(chunk)
        tf = Counter(chunk_tokens)
        total = len(chunk_tokens) or 1
        return sum((tf[t] / total) * self._idf.get(t, 0.0) for t in query_tokens)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def retrieve(self, query: str, k: int = 5) -> list[str]:
        """Return top-k chunks most relevant to query (TF-IDF scored)."""
        q_tokens = self._tokenize(query)
        scored = sorted(
            [(self._tfidf_score(q_tokens, c), c) for c in self._chunks],
            key=lambda x: x[0],
            reverse=True,
        )
        return [c for _, c in scored[:k]]

    def retrieve_by_keyword(self, keyword: str, lines_context: int = 5) -> list[str]:
        """Return log lines containing keyword with surrounding context."""
        kw = keyword.lower()
        all_lines = self._raw
        results: list[str] = []
        seen_starts: set[int] = set()
        for i, line in enumerate(all_lines):
            if kw in line.lower():
                start = max(0, i - lines_context)
                if start in seen_starts:
                    continue
                seen_starts.add(start)
                end = min(len(all_lines), i + lines_context + 1)
                window = "\n".join(
                    f"{'>' if j == i else ' '} [{j:04d}] {all_lines[j]}"
                    for j in range(start, end)
                )
                results.append(window)
                if len(results) >= 6:
                    break
        return results

    @property
    def chunk_count(self) -> int:
        return len(self._chunks)

    @property
    def line_count(self) -> int:
        return len(self._raw)
