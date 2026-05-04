"""Tests for feature engineering for FraudShield"""

import pytest
import pandas as pd
import numpy as np

from src.features import FeatureEngineer


class TestFeatureEngineer:
    """Test suite for FeatureEngineer class."""
    
    def test_initialization_default(self):
        """Test initialization with default parameters."""
        engineer = FeatureEngineer()
        
        assert engineer.scale_features is True
        assert engineer.n_components_pca is None
        assert engineer._fitted is False
    
    def test_initialization_custom(self):
        """Test initialization with custom parameters."""
        engineer = FeatureEngineer(scale_features=True, n_components_pca=5)
        
        assert engineer.scale_features is True
        assert engineer.n_components_pca == 5
    
    def test_initialization_invalid_pca(self):
        """Test initialization fails with invalid PCA components."""
        with pytest.raises(ValueError):
            FeatureEngineer(n_components_pca=0)
    
    def test_fit_transform_scaling_only(self):
        """Test fit_transform with scaling only."""
        X = pd.DataFrame(np.random.randn(100, 10) * 10 + 5)
        engineer = FeatureEngineer(scale_features=True)
        
        X_transformed = engineer.fit_transform(X)
        
        # Check output type
        assert isinstance(X_transformed, pd.DataFrame)
        assert X_transformed.shape == X.shape
        
        # Check standardization (mean ~0, std ~1)
        assert abs(X_transformed.mean().mean()) < 0.1
        assert abs(X_transformed.std().mean() - 1.0) < 0.1
        
        # Check fitted flag
        assert engineer._fitted is True
    
    def test_fit_transform_no_scaling(self):
        """Test fit_transform without scaling."""
        X = pd.DataFrame(np.random.randn(100, 10) * 10 + 5)
        engineer = FeatureEngineer(scale_features=False)
        
        X_transformed = engineer.fit_transform(X)
        
        # Should be unchanged
        pd.testing.assert_frame_equal(X_transformed, X)
    
    def test_fit_transform_with_pca(self):
        """Test fit_transform with PCA."""
        X = pd.DataFrame(np.random.randn(100, 10))
        engineer = FeatureEngineer(scale_features=True, n_components_pca=5)
        
        X_transformed = engineer.fit_transform(X)
        
        # Check dimensionality reduction
        assert X_transformed.shape == (100, 5)
        assert list(X_transformed.columns) == [f'pc_{i+1}' for i in range(5)]
    
    def test_transform_before_fit_raises(self):
        """Test transform before fit raises error."""
        X = pd.DataFrame(np.random.randn(100, 10))
        engineer = FeatureEngineer()
        
        with pytest.raises(RuntimeError, match="not fitted"):
            engineer.transform(X)
    
    def test_transform_after_fit(self):
        """Test transform on new data after fit."""
        X_train = pd.DataFrame(np.random.randn(100, 10) * 10 + 5)
        X_test = pd.DataFrame(np.random.randn(50, 10) * 10 + 5)
        
        engineer = FeatureEngineer(scale_features=True)
        engineer.fit_transform(X_train)
        
        X_test_transformed = engineer.transform(X_test)
        
        # Check output
        assert isinstance(X_test_transformed, pd.DataFrame)
        assert X_test_transformed.shape == X_test.shape
    
    def test_transform_feature_mismatch_raises(self):
        """Test transform with wrong features raises error."""
        X_train = pd.DataFrame(np.random.randn(100, 10))
        X_test = pd.DataFrame(np.random.randn(50, 8))  # Wrong number of features
        
        engineer = FeatureEngineer()
        engineer.fit_transform(X_train)
        
        with pytest.raises(ValueError, match="Feature mismatch"):
            engineer.transform(X_test)
    
    def test_validate_input_nan_raises(self):
        """Test that NaN values raise error."""
        X = pd.DataFrame([[1, 2], [np.nan, 4]])
        engineer = FeatureEngineer()
        
        with pytest.raises(ValueError, match="NaN"):
            engineer.fit_transform(X)
    
    def test_validate_input_inf_raises(self):
        """Test that infinite values raise error."""
        X = pd.DataFrame([[1, 2], [np.inf, 4]])
        engineer = FeatureEngineer()
        
        with pytest.raises(ValueError, match="infinite"):
            engineer.fit_transform(X)
    
    def test_validate_input_empty_raises(self):
        """Test that empty dataframe raises error."""
        X = pd.DataFrame()
        engineer = FeatureEngineer()
        
        with pytest.raises(ValueError, match="empty"):
            engineer.fit_transform(X)
    
    def test_get_feature_importance_no_pca(self):
        """Test get_feature_importance with no PCA."""
        X = pd.DataFrame(np.random.randn(100, 10))
        engineer = FeatureEngineer(scale_features=True)
        engineer.fit_transform(X)
        
        importance = engineer.get_feature_importance()
        assert importance is None
    
    def test_get_feature_importance_with_pca(self):
        """Test get_feature_importance with PCA."""
        X = pd.DataFrame(np.random.randn(100, 10))
        engineer = FeatureEngineer(scale_features=True, n_components_pca=5)
        engineer.fit_transform(X)
        
        importance = engineer.get_feature_importance()
        assert importance is not None
        assert len(importance) == 5
        assert all(0 <= v <= 1 for v in importance.values)
        assert importance.sum() <= 1.0
    
    def test_pca_components_exceeds_features(self):
        """Test PCA with more components than features."""
        X = pd.DataFrame(np.random.randn(100, 5))
        engineer = FeatureEngineer(scale_features=True, n_components_pca=10)
        
        # Should warn and reduce to n_features
        X_transformed = engineer.fit_transform(X)
        assert X_transformed.shape[1] == 5
