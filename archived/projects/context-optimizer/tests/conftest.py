"""
Shared test fixtures and helpers for context-optimizer tests.

All fixtures are designed to run without external services:
- No Ollama / LLM calls (all LLMs are mocked)
- No sentence-transformers model downloads (fake embedder used)
- ChromaDB runs in-memory via EphemeralClient
"""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

# ── Fake embedding model (SentenceTransformer-compatible) ────────────────────


class FakeEmbedder:
    """
    Deterministic embedder using SHA-256 so that *different* strings produce
    *different* (near-orthogonal) unit vectors.  This prevents false cache
    hits in SemanticCache tests that use a high similarity threshold.

    Uses 16 float32 values derived from the first 32 bytes of the hash,
    which is the ``DIM`` exposed by ``get_sentence_embedding_dimension()``.
    """

    DIM = 16

    def encode(
        self,
        text: str | list[str],
        convert_to_numpy: bool = True,
        show_progress_bar: bool = False,
    ) -> np.ndarray:
        single = isinstance(text, str)
        texts = [text] if single else text
        result = []
        for t in texts:
            digest = hashlib.sha256(t.encode()).digest()  # 32 bytes
            # Convert pairs of bytes to uint16, then to float32
            raw = np.array(
                [int.from_bytes(digest[i : i + 2], "big") for i in range(0, 32, 2)],
                dtype=np.float32,
            )
            norm = np.linalg.norm(raw)
            result.append(raw / norm if norm > 0 else raw)
        arr = np.array(result, dtype=np.float32)
        return arr[0] if single else arr

    # Satisfy SentenceTransformer.get_sentence_embedding_dimension()
    def get_sentence_embedding_dimension(self) -> int:
        return self.DIM


# ── Fake ChromaDB embedding function ─────────────────────────────────────────


class FakeChromaEmbeddingFn:
    """Callable that produces fixed-size list-of-list embeddings for ChromaDB.

    ChromaDB >= 0.5 requires ``.name()`` and ``.embed_query()`` methods.
    Returning ``"default"`` from ``name()`` causes ChromaDB to skip the
    conflict-check validation, which is the correct behaviour for test doubles.
    """

    DIM = 16

    def _encode_batch(self, input: list[str]) -> list[list[float]]:
        embedder = FakeEmbedder()
        return [embedder.encode(t).tolist() for t in input]

    def __call__(self, input: list[str]) -> list[list[float]]:
        return self._encode_batch(input)

    def embed_query(self, input: list[str]) -> list[list[float]]:
        """Called by ChromaDB when embedding query texts."""
        return self._encode_batch(input)

    def embed_documents(self, input: list[str]) -> list[list[float]]:
        """Called by ChromaDB when embedding document texts."""
        return self._encode_batch(input)

    # ChromaDB 0.5+ requires this; returning "default" bypasses conflict checks.
    def name(self) -> str:
        return "default"


# ── Mock LLM ─────────────────────────────────────────────────────────────────


class MockLLMResponse:
    """Mimics a LangChain message object with a `.content` attribute."""

    def __init__(self, content: str) -> None:
        self.content = content


class MockLLM:
    """
    Minimal LLM mock that returns deterministic JSON-formatted compression output.

    Parameters
    ----------
    fail:
        If True, raises RuntimeError on every call (tests the fallback path).
    bad_json:
        If True, returns non-JSON text (tests the JSON parse-error fallback).
    """

    def __init__(self, fail: bool = False, bad_json: bool = False) -> None:
        self._fail = fail
        self._bad_json = bad_json
        self.call_count = 0

    def invoke(self, prompt: str) -> MockLLMResponse:
        self.call_count += 1
        if self._fail:
            raise RuntimeError("Simulated LLM failure")
        if self._bad_json:
            return MockLLMResponse("not valid json at all")
        # Return valid compression JSON
        return MockLLMResponse(
            '{"summary": "Mock compressed summary.", '
            '"entities": ["entity_a", "entity_b"], '
            '"keywords": ["kw1", "kw2"], '
            '"has_code": false, "has_math": false, "section": "test"}'
        )


# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture()
def fake_embedder() -> FakeEmbedder:
    return FakeEmbedder()


@pytest.fixture()
def mock_llm() -> MockLLM:
    return MockLLM()


@pytest.fixture()
def mock_llm_fail() -> MockLLM:
    return MockLLM(fail=True)


@pytest.fixture()
def mock_llm_bad_json() -> MockLLM:
    return MockLLM(bad_json=True)
