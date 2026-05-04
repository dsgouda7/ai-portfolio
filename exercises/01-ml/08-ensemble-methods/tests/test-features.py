"""Tests for feature engineering"""

import pytest
import numpy as np
import pandas as pd

from src.features import FeatureEngineer


def test_feature_engineer_init():
    """Test FeatureEngineer initialization."""
    engineer = FeatureEngineer(scale_features=True, feature_selection=True, top_k_features=5)
    
    assert engineer.scale_features is True
    assert engineer.feature_selection is True
    assert engineer.top_k_features == 5
    assert engineer._fitted is False


def test_fit_transform_basic(sample_splits):
    """Test basic fit_transform."""
    X_train, _, y_train, _ = sample_splits
    
    engineer = FeatureEngineer(scale_features=True, feature_selection=False)
    X_transformed = engineer.fit_transform(X_train, y_train)
    
    # Check shape unchanged (no feature selection)
    assert X_transformed.shape == X_train.shape
    
    # Check standardization (mean ~0, std ~1)
    assert np.abs(X_transformed.mean().mean()) < 0.1
    assert np.abs(X_transformed.std().mean() - 1.0) < 0.1


def test_feature_selection(sample_splits):
    """Test feature selection via mutual information."""
    X_train, _, y_train, _ = sample_splits
    
    top_k = 5
    engineer = FeatureEngineer(scale_features=True, feature_selection=True, top_k_features=top_k)
    X_transformed = engineer.fit_transform(X_train, y_train)
    
    # Check that features were selected
    assert X_transformed.shape[1] == top_k
    assert len(engineer.selected_feature_scores) == top_k


def test_transform_after_fit(sample_splits):
    """Test transform on new data after fit_transform."""
    X_train, X_test, y_train, _ = sample_splits
    
    engineer = FeatureEngineer(scale_features=True, feature_selection=True, top_k_features=5)
    X_train_transformed = engineer.fit_transform(X_train, y_train)
    X_test_transformed = engineer.transform(X_test)
    
    # Check same features selected
    assert X_train_transformed.shape[1] == X_test_transformed.shape[1]
    assert list(X_train_transformed.columns) == list(X_test_transformed.columns)


def test_transform_before_fit(sample_data):
    """Test that transform before fit raises error."""
    X, _ = sample_data
    
    engineer = FeatureEngineer()
    
    with pytest.raises(RuntimeError, match="Pipeline not fitted"):
        engineer.transform(X)


def test_no_scaling(sample_splits):
    """Test with scaling disabled."""
    X_train, _, y_train, _ = sample_splits
    
    engineer = FeatureEngineer(scale_features=False, feature_selection=False)
    X_transformed = engineer.fit_transform(X_train, y_train)
    
    # Check that data is unchanged
    pd.testing.assert_frame_equal(X_transformed, X_train)


def test_invalid_top_k():
    """Test that invalid top_k raises ValueError."""
    with pytest.raises(ValueError, match="top_k_features must be >= 1"):
        FeatureEngineer(top_k_features=0)
    
    with pytest.raises(ValueError):
        FeatureEngineer(top_k_features=-5)


def test_missing_values_validation(sample_splits):
    """Test that missing values raise ValueError."""
    X_train, _, y_train, _ = sample_splits
    
    # Introduce NaN
    X_train_nan = X_train.copy()
    X_train_nan.iloc[0, 0] = np.nan
    
    engineer = FeatureEngineer()
    
    with pytest.raises(ValueError, match="contains NaN"):
        engineer.fit_transform(X_train_nan, y_train)


def test_infinite_values_validation(sample_splits):
    """Test that infinite values raise ValueError."""
    X_train, _, y_train, _ = sample_splits
    
    # Introduce inf
    X_train_inf = X_train.copy()
    X_train_inf.iloc[0, 0] = np.inf
    
    engineer = FeatureEngineer()
    
    with pytest.raises(ValueError, match="contains infinite"):
        engineer.fit_transform(X_train_inf, y_train)


def test_reproducibility(sample_splits):
    """Test that same data produces same transformation."""
    X_train, _, y_train, _ = sample_splits
    
    engineer1 = FeatureEngineer(scale_features=True, feature_selection=True, top_k_features=5)
    engineer2 = FeatureEngineer(scale_features=True, feature_selection=True, top_k_features=5)
    
    X_transformed1 = engineer1.fit_transform(X_train, y_train)
    X_transformed2 = engineer2.fit_transform(X_train, y_train)
    
    # Check that transformations are identical
    pd.testing.assert_frame_equal(X_transformed1, X_transformed2)


def test_feature_names_preserved(sample_splits):
    """Test that feature names are tracked correctly."""
    X_train, _, y_train, _ = sample_splits
    
    engineer = FeatureEngineer(scale_features=True, feature_selection=True, top_k_features=5)
    X_transformed = engineer.fit_transform(X_train, y_train)
    
    assert engineer.feature_names == list(X_transformed.columns)
    assert len(engineer.feature_names) == 5


def test_feature_selection_without_labels(sample_data):
    """Test that feature selection without labels raises error."""
    X, _ = sample_data
    
    engineer = FeatureEngineer(feature_selection=True, top_k_features=5)
    
    with pytest.raises(ValueError, match="y is required"):
        engineer.fit_transform(X, y=None)
