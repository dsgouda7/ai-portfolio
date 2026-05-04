"""Tests for feature engineering"""

import pytest
import pandas as pd
import numpy as np

from src.features import FeatureEngineer
from src.data import load_and_split


@pytest.fixture
def sample_data():
    """Fixture providing sample train/test data."""
    X_train, _, X_test, y_train, _, y_test = load_and_split(random_state=42)
    
    # Use smaller subset for faster tests
    X_train_small = X_train.head(100)
    X_test_small = X_test.head(50)
    y_train_small = y_train.head(100)
    
    return X_train_small, X_test_small, y_train_small


class TestFeatureEngineer:
    """Test suite for FeatureEngineer class."""
    
    def test_init_default(self):
        """Test initialization with default parameters."""
        engineer = FeatureEngineer()
        
        assert engineer.polynomial_degree == 1
        assert engineer.scale_features == True
        assert engineer.vif_threshold is None
        assert engineer._fitted == False
    
    def test_init_custom(self):
        """Test initialization with custom parameters."""
        engineer = FeatureEngineer(
            polynomial_degree=2,
            scale_features=True,
            vif_threshold=10.0
        )
        
        assert engineer.polynomial_degree == 2
        assert engineer.vif_threshold == 10.0
    
    def test_init_invalid_degree(self):
        """Test that invalid polynomial degree raises error."""
        with pytest.raises(ValueError):
            FeatureEngineer(polynomial_degree=0)
    
    def test_fit_transform_no_expansion(self, sample_data):
        """Test fit_transform without polynomial expansion."""
        X_train, _, _ = sample_data
        engineer = FeatureEngineer(polynomial_degree=1, scale_features=True)
        
        X_transformed = engineer.fit_transform(X_train)
        
        # Should have same number of features
        assert X_transformed.shape[1] == X_train.shape[1]
        assert X_transformed.shape[0] == X_train.shape[0]
        
        # Should be standardized (mean ≈ 0, std ≈ 1)
        assert np.abs(X_transformed.mean().mean()) < 0.1
        assert np.abs(X_transformed.std().mean() - 1.0) < 0.1
    
    def test_fit_transform_with_expansion(self, sample_data):
        """Test fit_transform with polynomial expansion."""
        X_train, _, _ = sample_data
        engineer = FeatureEngineer(polynomial_degree=2, scale_features=True)
        
        X_transformed = engineer.fit_transform(X_train)
        
        # Should have more features (8 + 8 + 28 interactions for degree 2)
        assert X_transformed.shape[1] > X_train.shape[1]
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
        engineer = FeatureEngineer(polynomial_degree=2)
        
        X_train_transformed = engineer.fit_transform(X_train)
        X_test_transformed = engineer.transform(X_test)
        
        # Should have same number of features
        assert X_train_transformed.shape[1] == X_test_transformed.shape[1]
    
    def test_fit_transform_nan_values(self, sample_data):
        """Test that NaN values raise error."""
        X_train, _, _ = sample_data
        X_train_nan = X_train.copy()
        X_train_nan.iloc[0, 0] = np.nan
        
        engineer = FeatureEngineer()
        
        with pytest.raises(ValueError, match="NaN"):
            engineer.fit_transform(X_train_nan)
    
    def test_fit_transform_inf_values(self, sample_data):
        """Test that infinite values raise error."""
        X_train, _, _ = sample_data
        X_train_inf = X_train.copy()
        X_train_inf.iloc[0, 0] = np.inf
        
        engineer = FeatureEngineer()
        
        with pytest.raises(ValueError, match="infinite"):
            engineer.fit_transform(X_train_inf)
    
    def test_transform_nan_values(self, sample_data):
        """Test that transform rejects NaN values."""
        X_train, X_test, _ = sample_data
        engineer = FeatureEngineer()
        engineer.fit_transform(X_train)
        
        X_test_nan = X_test.copy()
        X_test_nan.iloc[0, 0] = np.nan
        
        with pytest.raises(ValueError, match="NaN"):
            engineer.transform(X_test_nan)
    
    def test_get_feature_importance_mapping_before_fit(self):
        """Test that getting mapping before fit raises error."""
        engineer = FeatureEngineer()
        
        with pytest.raises(RuntimeError, match="not fitted"):
            engineer.get_feature_importance_mapping()
    
    def test_get_feature_importance_mapping_after_fit(self, sample_data):
        """Test getting feature importance mapping after fit."""
        X_train, _, _ = sample_data
        engineer = FeatureEngineer(scale_features=True)
        engineer.fit_transform(X_train)
        
        mapping = engineer.get_feature_importance_mapping()
        
        assert "feature_names" in mapping
        assert "means" in mapping
        assert "stds" in mapping
        assert len(mapping["feature_names"]) == len(mapping["means"])
    
    def test_no_scaling(self, sample_data):
        """Test feature engineering without scaling."""
        X_train, _, _ = sample_data
        engineer = FeatureEngineer(scale_features=False)
        
        X_transformed = engineer.fit_transform(X_train)
        
        # Should NOT be standardized
        # Original data has non-zero means
        assert np.abs(X_transformed.mean().mean()) > 0.1
    
    def test_reproducibility(self, sample_data):
        """Test that same data produces same transformation."""
        X_train, _, _ = sample_data
        
        engineer1 = FeatureEngineer(polynomial_degree=2)
        X_transformed1 = engineer1.fit_transform(X_train)
        
        engineer2 = FeatureEngineer(polynomial_degree=2)
        X_transformed2 = engineer2.fit_transform(X_train)
        
        pd.testing.assert_frame_equal(X_transformed1, X_transformed2)
