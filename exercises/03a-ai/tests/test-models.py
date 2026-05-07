"""
Tests for chatbot models
"""

import pytest
from unittest.mock import Mock
from src.models import ChatbotEngine, IntentDetector, OrderValidator
from src.rag import RAGPipeline
from src.utils import load_config


@pytest.fixture
def config():
    """Load test configuration."""
    return load_config()


@pytest.fixture
def intent_detector():
    """Create intent detector."""
    return IntentDetector(confidence_threshold=0.6)


@pytest.fixture
def order_validator(config):
    """Create order validator."""
    return OrderValidator(config)


@pytest.fixture
def mock_rag_pipeline():
    """Create mock RAG pipeline."""
    pipeline = Mock(spec=RAGPipeline)
    pipeline.query.return_value = {
        'response': 'Sure! I can help you order a pizza.',
        'context_docs': [],
        'tokens_used': 50
    }
    return pipeline


@pytest.fixture
def chatbot(mock_rag_pipeline, config):
    """Create chatbot engine."""
    return ChatbotEngine(mock_rag_pipeline, config)


def test_intent_detection_order(intent_detector):
    """Test order intent detection."""
    message = "I want to order a pepperoni pizza"
    result = intent_detector.detect(message)
    
    assert result['intent'] == 'order_pizza'
    assert result['confidence'] > 0


def test_intent_detection_menu(intent_detector):
    """Test menu inquiry intent detection."""
    message = "What's on your menu?"
    result = intent_detector.detect(message)
    
    assert result['intent'] == 'check_menu'


def test_intent_detection_question(intent_detector):
    """Test question intent detection."""
    message = "What are your delivery hours?"
    result = intent_detector.detect(message)
    
    assert result['intent'] == 'ask_question'


def test_intent_detection_greeting(intent_detector):
    """Test greeting intent detection."""
    message = "Hello!"
    result = intent_detector.detect(message)
    
    assert result['intent'] == 'general_chat'


def test_order_validation_valid(order_validator):
    """Test valid order validation."""
    order_items = [
        {'name': 'Margherita', 'quantity': 2, 'price': 12.99}
    ]
    delivery_address = "123 Main St, City, State 12345"
    
    result = order_validator.validate(order_items, delivery_address)
    
    assert result['valid'] is True
    assert len(result['errors']) == 0


def test_order_validation_no_items(order_validator):
    """Test validation with no items."""
    result = order_validator.validate([], "123 Main St")
    
    assert result['valid'] is False
    assert 'No items in order' in result['errors']


def test_order_validation_too_many_pizzas(order_validator):
    """Test validation with too many pizzas."""
    order_items = [
        {'name': 'Margherita', 'quantity': 6, 'price': 12.99}
    ]
    
    result = order_validator.validate(order_items, "123 Main St")
    
    assert result['valid'] is False
    assert any('Maximum' in error for error in result['errors'])


def test_order_validation_no_address(order_validator):
    """Test validation without delivery address."""
    order_items = [
        {'name': 'Margherita', 'quantity': 1, 'price': 12.99}
    ]
    
    result = order_validator.validate(order_items, None)
    
    assert result['valid'] is False
    assert 'Delivery address required' in result['errors']


def test_chatbot_process_message(chatbot):
    """Test message processing."""
    session_id = "test_session_1"
    message = "I want to order a pizza"
    
    result = chatbot.process_message(session_id, message)
    
    assert 'response' in result
    assert 'intent' in result
    assert 'session_id' in result
    assert result['session_id'] == session_id


def test_chatbot_session_creation(chatbot):
    """Test session creation."""
    session_id = "test_session_2"
    message = "Hello"
    
    chatbot.process_message(session_id, message)
    
    session = chatbot.get_session(session_id)
    assert session is not None
    assert 'messages' in session
    assert len(session['messages']) == 2  # User + assistant


def test_chatbot_conversation_history(chatbot):
    """Test conversation history tracking."""
    session_id = "test_session_3"
    
    chatbot.process_message(session_id, "Hello")
    chatbot.process_message(session_id, "What's on the menu?")
    chatbot.process_message(session_id, "I'll take a pepperoni")
    
    session = chatbot.get_session(session_id)
    assert len(session['messages']) == 6  # 3 exchanges


def test_chatbot_clear_session(chatbot):
    """Test session clearing."""
    session_id = "test_session_4"
    
    chatbot.process_message(session_id, "Hello")
    chatbot.clear_session(session_id)
    
    session = chatbot.get_session(session_id)
    assert session is None


def test_chatbot_order_handling(chatbot, mock_rag_pipeline):
    """Test order handling flow."""
    session_id = "test_session_5"
    message = "I want to order 2 margherita pizzas"
    
    result = chatbot.process_message(session_id, message)
    
    assert result['intent'] == 'order_pizza'
    assert 'address' in result['response'].lower()


def test_chatbot_cancellation(chatbot):
    """Test order cancellation."""
    session_id = "test_session_6"
    
    # Place order
    chatbot.process_message(session_id, "I want a pizza")
    
    # Cancel
    result = chatbot.process_message(session_id, "Cancel my order")
    
    assert 'cancel' in result['response'].lower()
