"""
Tests for embeddings module
"""

import pytest
import numpy as np
from src.embeddings import EmbeddingManager


@pytest.fixture
def embedding_manager():
    """Create embedding manager for testing."""
    return EmbeddingManager(
        model_name="all-MiniLM-L6-v2",
        persist_directory="./data/test_chroma_db",
        collection_name="test_collection"
    )


def test_generate_embeddings(embedding_manager):
    """Test embedding generation."""
    texts = [
        "What is the price of a pepperoni pizza?",
        "Do you offer gluten-free options?"
    ]
    
    embeddings = embedding_manager.generate_embeddings(texts)
    
    assert isinstance(embeddings, np.ndarray)
    assert embeddings.shape[0] == len(texts)
    assert embeddings.shape[1] > 0  # Has embedding dimensions


def test_add_and_query_documents(embedding_manager):
    """Test adding documents and querying."""
    # Clear collection first
    embedding_manager.clear_collection()
    
    # Add test documents
    documents = [
        "Margherita pizza costs $12.99 with tomato sauce and mozzarella.",
        "Pepperoni pizza is $14.99 with pepperoni and cheese.",
        "We deliver from 11 AM to 11 PM every day."
    ]
    
    embedding_manager.add_documents(documents)
    
    # Query
    results = embedding_manager.query("What time do you deliver?", n_results=2)
    
    assert len(results['documents'][0]) == 2
    assert len(results['distances'][0]) == 2


def test_retrieve_context(embedding_manager):
    """Test context retrieval."""
    # Clear and add documents
    embedding_manager.clear_collection()
    
    documents = [
        "Margherita pizza: $12.99",
        "Pepperoni pizza: $14.99",
        "Delivery hours: 11 AM - 11 PM"
    ]
    
    embedding_manager.add_documents(documents)
    
    # Retrieve context
    context_docs = embedding_manager.retrieve_context(
        "pizza prices",
        top_k=2,
        similarity_threshold=0.0
    )
    
    assert len(context_docs) <= 2
    assert all('text' in doc for doc in context_docs)
    assert all('similarity' in doc for doc in context_docs)


def test_collection_stats(embedding_manager):
    """Test collection statistics."""
    stats = embedding_manager.get_collection_stats()
    
    assert 'name' in stats
    assert 'count' in stats
    assert 'model' in stats
    assert stats['model'] == "all-MiniLM-L6-v2"


def test_clear_collection(embedding_manager):
    """Test clearing collection."""
    # Add documents
    documents = ["Test document 1", "Test document 2"]
    embedding_manager.add_documents(documents)
    
    # Clear
    embedding_manager.clear_collection()
    
    # Verify empty
    stats = embedding_manager.get_collection_stats()
    assert stats['count'] == 0
