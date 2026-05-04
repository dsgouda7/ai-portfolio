"""Tests for feature engineering"""

import pytest
import numpy as np

from src.features import FeatureEngineer
from src.data import load_and_split


@pytest.fixture
def sample_data():
    """Fixture providing sample train/test data."""
    X_train, _, X_test, y_train, _, y_test = load_and_split(random_state=42)
    
    # Use smaller subset for faster tests
    X_train_small = X_train[:20]
    X_test_small = X_test[:10]
    y_train_small = y_train[:20]
    
    return X_train_small, X_test_small, y_train_small


class TestFeatureEngineer:
    """Test suite for FeatureEngineer class."""
    
    def test_init_default(self):
        """Test initialization with default parameters."""
        engineer = FeatureEngineer()
        
        assert engineer.hog_orientations == 9
        assert engineer.scale_features == True
        assert engineer._fitted == False
    
    def test_fit_transform_with_hog(self, sample_data):
        """Test fit_transform with HOG features."""
        X_train, _, _ = sample_data
        engineer = FeatureEngineer(hog_orientations=9, pca_components=20)
        
        X_transformed = engineer.fit_transform(X_train)
        
        # Should have PCA-reduced features
        assert X_transformed.shape[1] == 20
        assert X_transformed.shape[0] == X_train.shape[0]
    
    def test_transform_before_fit(self, sample_data):
        """Test that transform before fit raises error."""
        X_train, _, _ = sample_data
        engineer = FeatureEngineer()
        
        with pytest.raises(RuntimeError, match="not fitted"):
            engineer.transform(X_train)
    
    def test_fit_transform_then_transform(self, sample_data):
        """Test that transform works after fit_transform."""
        X_train, X_test, _ = sample_data
        engineer = FeatureEngineer(pca_components=20)
        
        X_train_transformed = engineer.fit_transform(X_train)
        X_test_transformed = engineer.transform(X_test)
        
        # Should have same number of features
        assert X_train_transformed.shape[1] == X_test_transformed.shape[1]
