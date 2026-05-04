"""
Tests for Flask API
"""

import pytest
import json
from src.api import app


@pytest.fixture
def client():
    """Create test client."""
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client


def test_health_endpoint(client):
    """Test health check endpoint."""
    response = client.get('/health')
    
    assert response.status_code == 200
    data = json.loads(response.data)
    assert data['status'] == 'healthy'


def test_chat_endpoint(client):
    """Test chat endpoint."""
    payload = {
        'message': 'I want to order a pizza',
        'session_id': 'test_session_api'
    }
    
    response = client.post(
        '/chat',
        data=json.dumps(payload),
        content_type='application/json'
    )
    
    assert response.status_code == 200
    data = json.loads(response.data)
    assert 'response' in data
    assert 'session_id' in data
    assert 'intent' in data


def test_chat_endpoint_missing_message(client):
    """Test chat endpoint with missing message."""
    payload = {}
    
    response = client.post(
        '/chat',
        data=json.dumps(payload),
        content_type='application/json'
    )
    
    assert response.status_code == 400
    data = json.loads(response.data)
    assert 'error' in data


def test_chat_endpoint_auto_session(client):
    """Test chat endpoint creates session automatically."""
    payload = {
        'message': 'Hello'
    }
    
    response = client.post(
        '/chat',
        data=json.dumps(payload),
        content_type='application/json'
    )
    
    assert response.status_code == 200
    data = json.loads(response.data)
    assert 'session_id' in data
    assert len(data['session_id']) > 0


def test_menu_endpoint(client):
    """Test menu endpoint."""
    response = client.get('/menu')
    
    assert response.status_code == 200
    data = json.loads(response.data)
    assert 'menu' in data
    assert 'timestamp' in data


def test_session_get_endpoint(client):
    """Test get session endpoint."""
    # First create a session
    chat_payload = {
        'message': 'Hello',
        'session_id': 'test_get_session'
    }
    client.post(
        '/chat',
        data=json.dumps(chat_payload),
        content_type='application/json'
    )
    
    # Now get the session
    response = client.get('/session/test_get_session')
    
    assert response.status_code == 200
    data = json.loads(response.data)
    assert data['session_id'] == 'test_get_session'
    assert 'message_count' in data


def test_session_get_not_found(client):
    """Test get non-existent session."""
    response = client.get('/session/nonexistent_session')
    
    assert response.status_code == 404
    data = json.loads(response.data)
    assert 'error' in data


def test_session_delete_endpoint(client):
    """Test delete session endpoint."""
    # Create session
    chat_payload = {
        'message': 'Hello',
        'session_id': 'test_delete_session'
    }
    client.post(
        '/chat',
        data=json.dumps(chat_payload),
        content_type='application/json'
    )
    
    # Delete session
    response = client.delete('/session/test_delete_session')
    
    assert response.status_code == 200
    data = json.loads(response.data)
    assert 'message' in data


def test_stats_endpoint(client):
    """Test stats endpoint."""
    response = client.get('/stats')
    
    assert response.status_code == 200
    data = json.loads(response.data)
    assert 'vector_db' in data
    assert 'active_sessions' in data
    assert 'config' in data


def test_metrics_endpoint(client):
    """Test Prometheus metrics endpoint."""
    response = client.get('/metrics')
    
    assert response.status_code == 200
    assert b'pizzabot' in response.data


def test_chat_with_context(client):
    """Test chat with context inclusion."""
    payload = {
        'message': 'What pizzas do you have?',
        'session_id': 'test_context',
        'include_context': True
    }
    
    response = client.post(
        '/chat',
        data=json.dumps(payload),
        content_type='application/json'
    )
    
    assert response.status_code == 200
    data = json.loads(response.data)
    assert 'context_docs' in data


def test_conversation_flow(client):
    """Test multi-turn conversation."""
    session_id = 'test_conversation_flow'
    
    # Turn 1: Greeting
    response1 = client.post(
        '/chat',
        data=json.dumps({
            'message': 'Hello',
            'session_id': session_id
        }),
        content_type='application/json'
    )
    assert response1.status_code == 200
    
    # Turn 2: Menu inquiry
    response2 = client.post(
        '/chat',
        data=json.dumps({
            'message': 'What pizzas do you have?',
            'session_id': session_id
        }),
        content_type='application/json'
    )
    assert response2.status_code == 200
    
    # Turn 3: Order
    response3 = client.post(
        '/chat',
        data=json.dumps({
            'message': 'I want to order a pepperoni pizza',
            'session_id': session_id
        }),
        content_type='application/json'
    )
    assert response3.status_code == 200
    
    # Check session has all messages
    session_response = client.get(f'/session/{session_id}')
    session_data = json.loads(session_response.data)
    assert session_data['message_count'] >= 6  # At least 3 exchanges
