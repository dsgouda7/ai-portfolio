"""Tests for feature engineering"""

import pytest
import numpy as np

from src.features import FeatureEngineer


class TestFeatureEngineer:
    """Test suite for FeatureEngineer class."""
    
    def test_feature_engineer_init(self):
        """Test FeatureEngineer initialization."""
        fe = FeatureEngineer(scale_features=True, pca_components=10)
        
        assert fe.scale_features is True
        assert fe.pca_components == 10
        assert fe.scaler is not None
        assert fe.pca is not None
    
    def test_feature_engineer_fit_transform(self):
        """Test fit_transform with scaling only."""
        X = np.random.randn(100, 20)
        fe = FeatureEngineer(scale_features=True, pca_components=None)
        
        X_transformed = fe.fit_transform(X)
        
        # Check shape unchanged
        assert X_transformed.shape == X.shape
        
        # Check standardization (mean ~0, std ~1)
        assert np.abs(X_transformed.mean()) < 0.1
        assert np.abs(X_transformed.std() - 1.0) < 0.1
    
    def test_feature_engineer_with_pca(self):
        """Test feature engineering with PCA."""
        X = np.random.randn(100, 20)
        fe = FeatureEngineer(scale_features=True, pca_components=10)
        
        X_transformed = fe.fit_transform(X)
        
        # Check dimensionality reduction
        assert X_transformed.shape == (100, 10)
        assert fe.n_features_in_ == 20
        assert fe.n_features_out_ == 10
    
    def test_feature_engineer_transform_consistency(self):
        """Test that transform produces consistent results."""
        X_train = np.random.randn(100, 20)
        X_test = np.random.randn(50, 20)
        
        fe = FeatureEngineer(scale_features=True, pca_components=10)
        fe.fit(X_train)
        
        X_test_transformed1 = fe.transform(X_test)
        X_test_transformed2 = fe.transform(X_test)
        
        # Should be identical
        np.testing.assert_array_equal(X_test_transformed1, X_test_transformed2)
    
    def test_feature_engineer_transform_before_fit(self):
        """Test that transform before fit raises ValueError."""
        X = np.random.randn(100, 20)
        fe = FeatureEngineer(scale_features=True)
        
        with pytest.raises(ValueError, match="not fitted"):
            fe.transform(X)
    
    def test_feature_engineer_wrong_shape(self):
        """Test that wrong input shape raises ValueError."""
        X_train = np.random.randn(100, 20)
        X_test = np.random.randn(50, 15)  # Wrong feature count
        
        fe = FeatureEngineer(scale_features=True)
        fe.fit(X_train)
        
        with pytest.raises(ValueError, match="features"):
            fe.transform(X_test)
    
    def test_feature_engineer_no_scaling(self):
        """Test feature engineer without scaling."""
        X = np.random.randn(100, 20)
        fe = FeatureEngineer(scale_features=False, pca_components=None)
        
        X_transformed = fe.fit_transform(X)
        
        # Should be unchanged
        np.testing.assert_array_equal(X, X_transformed)
    
    def test_feature_engineer_get_feature_names(self):
        """Test get_feature_names method."""
        X = np.random.randn(100, 20)
        fe = FeatureEngineer(scale_features=True, pca_components=10)
        fe.fit(X)
        
        feature_names = fe.get_feature_names()
        
        assert len(feature_names) == 10
        assert all(name.startswith("pca_") for name in feature_names)
