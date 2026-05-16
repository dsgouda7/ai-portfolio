"""
Tests for CLIP model
"""

import pytest
import numpy as np
from PIL import Image
from src.models.clip import CLIPModel


@pytest.fixture
def clip_model(device):
    """Initialize CLIP model for testing."""
    return CLIPModel(device=device, cache_dir="./models/clip")


def test_clip_initialization(clip_model):
    """Test CLIP model initialization."""
    assert clip_model is not None
    assert clip_model.model is not None
    assert clip_model.processor is not None


def test_compute_similarity(clip_model, sample_text, sample_image):
    """Test text-image similarity computation."""
    similarity = clip_model.compute_similarity(sample_text, sample_image)
    
    assert isinstance(similarity, float)
    assert 0.0 <= similarity <= 1.0


def test_zero_shot_classification(clip_model, sample_image):
    """Test zero-shot image classification."""
    labels = ["car", "dog", "cat", "tree", "building"]
    results = clip_model.zero_shot_classify(sample_image, labels)
    
    assert isinstance(results, dict)
    assert len(results) == len(labels)
    assert all(label in results for label in labels)
    
    # Check probabilities sum to ~1
    assert abs(sum(results.values()) - 1.0) < 0.01


def test_get_image_embedding(clip_model, sample_image):
    """Test image embedding extraction."""
    embedding = clip_model.get_image_embedding(sample_image)
    
    assert isinstance(embedding, np.ndarray)
    assert len(embedding.shape) == 1  # 1D vector
    assert embedding.shape[0] > 0  # Non-empty


def test_get_text_embedding(clip_model, sample_text):
    """Test text embedding extraction."""
    embedding = clip_model.get_text_embedding(sample_text)
    
    assert isinstance(embedding, np.ndarray)
    assert len(embedding.shape) == 1  # 1D vector
    assert embedding.shape[0] > 0  # Non-empty


def test_search_images(clip_model, sample_image, tmp_path):
    """Test image search functionality."""
    # Create multiple test images
    image_paths = []
    for i in range(3):
        img_path = tmp_path / f"image_{i}.png"
        sample_image.save(img_path)
        image_paths.append(str(img_path))
    
    # Search
    query = "a red car"
    results = clip_model.search_images(query, image_paths, top_k=2)
    
    assert len(results) == 2
    assert all(isinstance(r, tuple) and len(r) == 2 for r in results)
    assert all(isinstance(r[1], float) for r in results)


def test_similarity_with_multiple_texts(clip_model, sample_image):
    """Test similarity with multiple text queries."""
    texts = ["a car", "a dog", "a building"]
    similarities = clip_model.compute_similarity(texts, sample_image)
    
    assert isinstance(similarities, np.ndarray)
    assert similarities.shape[0] == len(texts)


def test_similarity_symmetric(clip_model, sample_text, sample_image):
    """Test that similarity is symmetric."""
    sim1 = clip_model.compute_similarity(sample_text, sample_image)
    sim2 = clip_model.compute_similarity(sample_text, sample_image)
    
    assert abs(sim1 - sim2) < 1e-5  # Should be identical
