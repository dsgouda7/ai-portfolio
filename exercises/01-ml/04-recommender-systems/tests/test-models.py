"""Tests for model training and inference"""

import pytest
import pandas as pd
import numpy as np

from src.models import ModelRegistry
from src.data import load_and_split_ratings


class TestModelRegistry:
    """Test suite for ModelRegistry class."""
    
    def test_init(self):
        """Test initialization."""
        registry = ModelRegistry()
        
        assert isinstance(registry.models, dict)
        assert len(registry.models) == 0
        assert registry.best_model_name is None
    
    @pytest.mark.slow
    def test_train_matrix_factorization(self):
        """Test matrix factorization training."""
        # Load small dataset
        train, _, _, ui_train, _, _ = load_and_split_ratings(random_state=42)
        
        registry = ModelRegistry()
        metrics = registry.train_matrix_factorization(
            train, ui_train, n_factors=10
        )
        
        # Check metrics
        assert 'rmse' in metrics
        assert 'mae' in metrics
        assert metrics['rmse'] > 0
        assert metrics['mae'] > 0
        
        # Check model stored
        assert "matrix_factorization" in registry.models
    
    def test_train_matrix_factorization_invalid_factors(self):
        """Test that invalid n_factors raises ValueError."""
        registry = ModelRegistry()
        
        train = pd.DataFrame({
            'user_id': [0, 1],
            'item_id': [0, 1],
            'rating': [5, 4]
        })
        ui_train = train.pivot_table(
            index='user_id', columns='item_id', values='rating', fill_value=0
        )
        
        with pytest.raises(ValueError):
            registry.train_matrix_factorization(train, ui_train, n_factors=0)
    
    def test_predict_rating(self):
        """Test rating prediction."""
        # Create simple test data
        train = pd.DataFrame({
            'user_id': [0, 0, 1, 1, 2, 2],
            'item_id': [0, 1, 1, 2, 0, 2],
            'rating': [5, 4, 3, 5, 4, 2]
        })
        ui_train = train.pivot_table(
            index='user_id', columns='item_id', values='rating', fill_value=0
        )
        
        registry = ModelRegistry()
        registry.train_matrix_factorization(train, ui_train, n_factors=2)
        
        # Predict rating
        rating = registry.predict_rating(user_id=0, item_id=0)
        
        # Check output
        assert isinstance(rating, float)
        assert 1 <= rating <= 5  # Rating should be in valid range
    
    def test_predict_rating_model_not_trained(self):
        """Test that predicting before training raises RuntimeError."""
        registry = ModelRegistry()
        
        with pytest.raises(RuntimeError):
            registry.predict_rating(user_id=0, item_id=0)
    
    def test_recommend_items(self):
        """Test item recommendation."""
        # Create simple test data
        train = pd.DataFrame({
            'user_id': [0, 0, 1, 1, 2, 2],
            'item_id': [0, 1, 1, 2, 0, 2],
            'rating': [5, 4, 3, 5, 4, 2]
        })
        ui_train = train.pivot_table(
            index='user_id', columns='item_id', values='rating', fill_value=0
        )
        
        registry = ModelRegistry()
        registry.train_matrix_factorization(train, ui_train, n_factors=2)
        
        # Get recommendations
        recommendations = registry.recommend_items(user_id=0, k=2)
        
        # Check output
        assert isinstance(recommendations, list)
        assert len(recommendations) == 2
        assert all(isinstance(item_id, int) for item_id, _ in recommendations)
        assert all(isinstance(score, float) for _, score in recommendations)
    
    def test_recommend_items_exclude_seen(self):
        """Test recommendation with seen items exclusion."""
        train = pd.DataFrame({
            'user_id': [0, 0, 1, 1, 2, 2],
            'item_id': [0, 1, 1, 2, 0, 2],
            'rating': [5, 4, 3, 5, 4, 2]
        })
        ui_train = train.pivot_table(
            index='user_id', columns='item_id', values='rating', fill_value=0
        )
        
        registry = ModelRegistry()
        registry.train_matrix_factorization(train, ui_train, n_factors=2)
        
        # Get recommendations excluding item 0
        seen_items = {0}
        recommendations = registry.recommend_items(
            user_id=0, k=2, exclude_seen=True, seen_items=seen_items
        )
        
        # Check that item 0 is not recommended
        recommended_items = [item_id for item_id, _ in recommendations]
        assert 0 not in recommended_items


class TestModelPersistence:
    """Test suite for model saving and loading."""
    
    def test_save_and_load(self, tmp_path):
        """Test model persistence."""
        # Train a simple model
        train = pd.DataFrame({
            'user_id': [0, 1, 2],
            'item_id': [0, 1, 2],
            'rating': [5, 4, 3]
        })
        ui_train = train.pivot_table(
            index='user_id', columns='item_id', values='rating', fill_value=0
        )
        
        registry = ModelRegistry()
        registry.train_matrix_factorization(train, ui_train, n_factors=2)
        
        # Save model
        model_path = tmp_path / "test_model.pkl"
        registry.save("matrix_factorization", str(model_path))
        
        # Load model
        new_registry = ModelRegistry()
        new_registry.load("matrix_factorization", str(model_path))
        
        # Check model loaded
        assert "matrix_factorization" in new_registry.models
    
    def test_save_nonexistent_model(self, tmp_path):
        """Test that saving non-existent model raises ValueError."""
        registry = ModelRegistry()
        
        with pytest.raises(ValueError):
            registry.save("nonexistent", str(tmp_path / "model.pkl"))
    
    def test_load_nonexistent_file(self):
        """Test that loading non-existent file raises FileNotFoundError."""
        registry = ModelRegistry()
        
        with pytest.raises(FileNotFoundError):
            registry.load("model", "nonexistent_path.pkl")
