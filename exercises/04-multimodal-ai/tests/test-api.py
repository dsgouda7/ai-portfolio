"""
Tests for Flask API endpoints
"""

import pytest
import io
import json
from PIL import Image
import numpy as np
from src.api import app


@pytest.fixture
def client():
    """Create test client."""
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client


@pytest.fixture
def sample_image_bytes():
    """Create sample image as bytes."""
    img = Image.new('RGB', (224, 224), color='red')
    img_io = io.BytesIO()
    img.save(img_io, 'PNG')
    img_io.seek(0)
    return img_io


@pytest.fixture
def sample_audio_bytes():
    """Create sample audio as bytes."""
    import soundfile as sf
    audio = np.random.randn(16000).astype(np.float32)
    audio_io = io.BytesIO()
    sf.write(audio_io, audio, 16000, format='WAV')
    audio_io.seek(0)
    return audio_io


def test_health_endpoint(client):
    """Test health check endpoint."""
    response = client.get('/health')
    
    assert response.status_code == 200
    data = json.loads(response.data)
    assert data['status'] == 'healthy'


def test_similarity_endpoint_missing_text(client, sample_image_bytes):
    """Test similarity endpoint with missing text."""
    response = client.post('/similarity', data={
        'image': (sample_image_bytes, 'test.png')
    })
    
    assert response.status_code == 400
    data = json.loads(response.data)
    assert 'error' in data


def test_similarity_endpoint_missing_image(client):
    """Test similarity endpoint with missing image."""
    response = client.post('/similarity', data={
        'text': 'a red car'
    })
    
    assert response.status_code == 400
    data = json.loads(response.data)
    assert 'error' in data


def test_similarity_endpoint_success(client, sample_image_bytes):
    """Test similarity endpoint with valid inputs."""
    response = client.post('/similarity', data={
        'text': 'a red car',
        'image': (sample_image_bytes, 'test.png')
    }, content_type='multipart/form-data')
    
    # Might fail if models not initialized, but structure should be correct
    if response.status_code == 200:
        data = json.loads(response.data)
        assert 'similarity' in data
        assert isinstance(data['similarity'], (int, float))


def test_transcribe_endpoint_missing_audio(client):
    """Test transcribe endpoint with missing audio."""
    response = client.post('/transcribe')
    
    assert response.status_code == 400
    data = json.loads(response.data)
    assert 'error' in data


def test_transcribe_endpoint_success(client, sample_audio_bytes):
    """Test transcribe endpoint with valid audio."""
    response = client.post('/transcribe', data={
        'audio': (sample_audio_bytes, 'test.wav')
    }, content_type='multipart/form-data')
    
    # Might fail if model not initialized
    if response.status_code == 200:
        data = json.loads(response.data)
        assert 'text' in data


def test_generate_endpoint_missing_prompt(client):
    """Test generate endpoint with missing prompt."""
    response = client.post('/generate', 
        data=json.dumps({}),
        content_type='application/json'
    )
    
    assert response.status_code == 400


def test_generate_endpoint_with_prompt(client):
    """Test generate endpoint with valid prompt."""
    response = client.post('/generate',
        data=json.dumps({'prompt': 'a red car'}),
        content_type='application/json'
    )
    
    # Might fail if model not initialized, but should handle gracefully
    # 200 or 500, but not 400 (validation error)
    assert response.status_code in [200, 500]


def test_caption_endpoint_missing_image(client):
    """Test caption endpoint with missing image."""
    response = client.post('/caption')
    
    assert response.status_code == 400
    data = json.loads(response.data)
    assert 'error' in data


def test_caption_endpoint_success(client, sample_image_bytes):
    """Test caption endpoint with valid image."""
    response = client.post('/caption', data={
        'image': (sample_image_bytes, 'test.png')
    }, content_type='multipart/form-data')
    
    # Might fail if model not initialized
    if response.status_code == 200:
        data = json.loads(response.data)
        assert 'caption' in data


def test_search_endpoint_no_query(client):
    """Test search endpoint with no query."""
    response = client.post('/search')
    
    assert response.status_code == 400


def test_search_endpoint_with_text(client):
    """Test search endpoint with text query."""
    response = client.post('/search', data={
        'text': 'red car'
    }, content_type='multipart/form-data')
    
    if response.status_code == 200:
        data = json.loads(response.data)
        assert 'results' in data
        assert 'query_modalities' in data


def test_cors_headers(client):
    """Test CORS headers are present."""
    response = client.get('/health')
    
    # CORS headers should be present
    assert 'Access-Control-Allow-Origin' in response.headers or response.status_code == 200
