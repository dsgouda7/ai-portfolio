"""
Tests for RAG pipeline
"""

import pytest
from unittest.mock import Mock, patch
from src.rag import RAGPipeline
from src.embeddings import EmbeddingManager
from src.utils import load_config


@pytest.fixture
def mock_embedding_manager():
    """Create mock embedding manager."""
    manager = Mock(spec=EmbeddingManager)
    
    # Mock retrieve_context to return sample documents
    manager.retrieve_context.return_value = [
        {
            'text': 'Margherita pizza costs $12.99',
            'similarity': 0.85,
            'metadata': {},
            'id': 'doc_1'
        },
        {
            'text': 'We deliver from 11 AM to 11 PM',
            'similarity': 0.75,
            'metadata': {},
            'id': 'doc_2'
        }
    ]
    
    return manager


@pytest.fixture
def rag_pipeline(mock_embedding_manager):
    """Create RAG pipeline for testing."""
    config = load_config()
    return RAGPipeline(mock_embedding_manager, config)


def test_retrieve(rag_pipeline, mock_embedding_manager):
    """Test document retrieval."""
    query = "What is the price of margherita pizza?"
    
    documents = rag_pipeline.retrieve(query)
    
    assert len(documents) == 2
    assert documents[0]['similarity'] == 0.85
    mock_embedding_manager.retrieve_context.assert_called_once()


def test_augment_prompt(rag_pipeline):
    """Test prompt augmentation."""
    query = "What time do you deliver?"
    context_docs = [
        {'text': 'We deliver from 11 AM to 11 PM', 'similarity': 0.9}
    ]
    conversation_history = [
        {'role': 'user', 'content': 'Hello'},
        {'role': 'assistant', 'content': 'Hi! How can I help?'}
    ]
    
    system_prompt, user_prompt = rag_pipeline.augment_prompt(
        query,
        context_docs,
        conversation_history
    )
    
    assert isinstance(system_prompt, str)
    assert isinstance(user_prompt, str)
    assert 'PizzaBot' in system_prompt
    assert query in user_prompt
    assert 'We deliver from 11 AM to 11 PM' in user_prompt


def test_generate_fallback(rag_pipeline):
    """Test fallback response generation."""
    query = "Tell me about your pizzas"
    context_docs = [
        {'text': 'We offer many pizza varieties', 'similarity': 0.8}
    ]
    
    response = rag_pipeline._generate_fallback_response(query, context_docs)
    
    assert isinstance(response, str)
    assert len(response) > 0


def test_query_pipeline(rag_pipeline, mock_embedding_manager):
    """Test complete RAG pipeline."""
    query = "What is the price of pepperoni pizza?"
    
    result = rag_pipeline.query(query)
    
    assert 'response' in result
    assert 'context_docs' in result
    assert 'pipeline_time' in result
    assert isinstance(result['response'], str)
    assert len(result['context_docs']) > 0


def test_query_with_conversation_history(rag_pipeline):
    """Test query with conversation history."""
    query = "What about the pepperoni?"
    history = [
        {'role': 'user', 'content': 'Show me your menu'},
        {'role': 'assistant', 'content': 'Here are our pizzas...'}
    ]
    
    result = rag_pipeline.query(query, conversation_history=history)
    
    assert 'response' in result
    assert 'context_docs' in result


def test_empty_context_handling(rag_pipeline, mock_embedding_manager):
    """Test handling of empty context."""
    # Mock empty retrieval
    mock_embedding_manager.retrieve_context.return_value = []
    
    query = "Random question"
    result = rag_pipeline.query(query)
    
    assert 'response' in result
    assert isinstance(result['response'], str)
