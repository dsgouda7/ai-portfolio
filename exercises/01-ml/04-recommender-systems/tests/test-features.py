"""Tests for feature engineering"""

import pytest
import pandas as pd
import numpy as np

from src.features import EmbeddingGenerator


class TestEmbeddingGenerator:
    """Test suite for EmbeddingGenerator class."""
    
    def test_init_default(self):
        """Test initialization with default parameters."""
        generator = EmbeddingGenerator()
        
        assert generator.embedding_dim == 50
        assert generator.user_embeddings is None
        assert generator.item_embeddings is None
        assert not generator._fitted
    
    def test_init_custom_dim(self):
        """Test initialization with custom embedding dimension."""
        generator = EmbeddingGenerator(embedding_dim=100)
        
        assert generator.embedding_dim == 100
    
    def test_init_invalid_dim(self):
        """Test that invalid embedding_dim raises ValueError."""
        with pytest.raises(ValueError):
            EmbeddingGenerator(embedding_dim=0)
        
        with pytest.raises(ValueError):
            EmbeddingGenerator(embedding_dim=-10)
    
    def test_fit_transform(self):
        """Test embedding generation."""
        # Create synthetic ratings
        ratings = pd.DataFrame({
            'user_id': [0, 0, 1, 1, 2, 2],
            'item_id': [0, 1, 1, 2, 0, 2],
            'rating': [5, 4, 3, 5, 4, 2]
        })
        
        generator = EmbeddingGenerator(embedding_dim=10)
        user_emb, item_emb = generator.fit_transform(ratings)
        
        # Check shapes
        n_users = ratings['user_id'].nunique()
        n_items = ratings['item_id'].nunique()
        
        assert user_emb.shape == (n_users, 10)
        assert item_emb.shape == (n_items, 10)
        
        # Check fitted flag
        assert generator._fitted
    
    def test_get_user_embedding(self):
        """Test retrieving user embedding."""
        ratings = pd.DataFrame({
            'user_id': [0, 1, 2],
            'item_id': [0, 1, 2],
            'rating': [5, 4, 3]
        })
        
        generator = EmbeddingGenerator(embedding_dim=10)
        generator.fit_transform(ratings)
        
        # Get embedding for user 0
        emb = generator.get_user_embedding(0)
        
        assert emb.shape == (10,)
        assert isinstance(emb, np.ndarray)
    
    def test_get_user_embedding_not_fitted(self):
        """Test that getting embedding before fit raises RuntimeError."""
        generator = EmbeddingGenerator()
        
        with pytest.raises(RuntimeError):
            generator.get_user_embedding(0)
    
    def test_get_user_embedding_invalid_id(self):
        """Test that invalid user_id raises KeyError."""
        ratings = pd.DataFrame({
            'user_id': [0, 1],
            'item_id': [0, 1],
            'rating': [5, 4]
        })
        
        generator = EmbeddingGenerator()
        generator.fit_transform(ratings)
        
        with pytest.raises(KeyError):
            generator.get_user_embedding(999)
    
    def test_get_item_embedding(self):
        """Test retrieving item embedding."""
        ratings = pd.DataFrame({
            'user_id': [0, 1, 2],
            'item_id': [0, 1, 2],
            'rating': [5, 4, 3]
        })
        
        generator = EmbeddingGenerator(embedding_dim=10)
        generator.fit_transform(ratings)
        
        # Get embedding for item 0
        emb = generator.get_item_embedding(0)
        
        assert emb.shape == (10,)
        assert isinstance(emb, np.ndarray)
    
    def test_compute_similarity_cosine(self):
        """Test cosine similarity computation."""
        generator = EmbeddingGenerator()
        
        emb1 = np.array([1.0, 0.0, 0.0])
        emb2 = np.array([1.0, 0.0, 0.0])
        
        similarity = generator.compute_similarity(emb1, emb2, metric='cosine')
        
        # Identical vectors should have similarity = 1.0
        assert np.isclose(similarity, 1.0)
    
    def test_compute_similarity_euclidean(self):
        """Test euclidean similarity computation."""
        generator = EmbeddingGenerator()
        
        emb1 = np.array([0.0, 0.0, 0.0])
        emb2 = np.array([0.0, 0.0, 0.0])
        
        similarity = generator.compute_similarity(emb1, emb2, metric='euclidean')
        
        # Identical vectors should have high similarity
        assert similarity > 0.9
    
    def test_compute_similarity_invalid_metric(self):
        """Test that invalid metric raises ValueError."""
        generator = EmbeddingGenerator()
        
        emb1 = np.array([1.0, 0.0])
        emb2 = np.array([0.0, 1.0])
        
        with pytest.raises(ValueError):
            generator.compute_similarity(emb1, emb2, metric='invalid')
