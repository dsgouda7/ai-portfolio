"""Unit tests for vectorization pipeline."""

import pytest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from embeddings.embedding_manager import EmbeddingManager
from langchain_core.documents import Document


@pytest.fixture(autouse=True)
def stub_huggingface_embeddings(monkeypatch):
    monkeypatch.setattr(
        "embeddings.embedding_manager.HuggingFaceEmbeddings",
        lambda **kwargs: kwargs,
    )


def test_embedding_manager_initialization():
    """Test EmbeddingManager can be instantiated."""
    manager = EmbeddingManager(
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        device="cpu",
        chunk_size=512,
        chunk_overlap=64
    )
    assert manager.embedding_model == "sentence-transformers/all-MiniLM-L6-v2"
    assert manager.device == "cpu"
    assert manager.chunk_size == 512


def test_document_splitting():
    """Test document splitting functionality."""
    manager = EmbeddingManager(
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        device="cpu",
        chunk_size=100,
        chunk_overlap=20
    )

    # Create a long document
    long_text = "This is a test sentence. " * 50
    docs = [Document(page_content=long_text, metadata={"id": "1"})]

    chunks = manager.split_documents(docs)

    assert len(chunks) > 1
    assert all(isinstance(chunk, Document) for chunk in chunks)
    assert all(len(chunk.page_content) <= 120 for chunk in chunks)  # Some tolerance


@pytest.mark.skipif(not Path("../../data/delta_lake/documents").exists(),
                    reason="Delta table does not exist")
def test_load_from_delta():
    """Test loading documents from Delta Lake (if table exists)."""
    manager = EmbeddingManager(
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        device="cpu"
    )

    documents = manager.load_from_delta("../../data/delta_lake")
    assert len(documents) > 0
    assert all(isinstance(doc, Document) for doc in documents)
