"""Tests for data loading and splitting"""

import pytest
import numpy as np
import pandas as pd

from src.data import load_and_split


def test_load_and_split_basic():
    """Test basic data loading and splitting."""
    X_train, X_val, X_test, y_train, y_val, y_test = load_and_split(
        dataset='make_classification',
        test_size=0.2,
        val_size=0.1,
        random_state=42,
        n_samples=1000
    )
    
    # Check types
    assert isinstance(X_train, pd.DataFrame)
    assert isinstance(y_train, pd.Series)
    
    # Check sizes
    total_samples = len(X_train) + len(X_val) + len(X_test)
    assert total_samples == 1000
    
    # Check proportions (approximately)
    assert 0.65 <= len(X_train) / total_samples <= 0.75
    assert 0.08 <= len(X_val) / total_samples <= 0.12
    assert 0.18 <= len(X_test) / total_samples <= 0.22


def test_stratified_splits():
    """Test that splits preserve class distribution."""
    X_train, X_val, X_test, y_train, y_val, y_test = load_and_split(
        dataset='make_classification',
        random_state=42,
        n_samples=1000
    )
    
    # Get class ratios
    train_ratio = y_train.mean()
    val_ratio = y_val.mean()
    test_ratio = y_test.mean()
    
    # Check that ratios are similar (within 10%)
    assert 0.9 * train_ratio <= val_ratio <= 1.1 * train_ratio
    assert 0.9 * train_ratio <= test_ratio <= 1.1 * train_ratio


def test_no_data_leakage():
    """Test that train/val/test sets have no overlapping indices."""
    X_train, X_val, X_test, y_train, y_val, y_test = load_and_split(
        random_state=42,
        n_samples=1000
    )
    
    train_idx = set(X_train.index)
    val_idx = set(X_val.index)
    test_idx = set(X_test.index)
    
    # Check no overlap
    assert len(train_idx & val_idx) == 0
    assert len(train_idx & test_idx) == 0
    assert len(val_idx & test_idx) == 0


def test_invalid_split_sizes():
    """Test that invalid split sizes raise ValueError."""
    # test_size + val_size >= 1.0
    with pytest.raises(ValueError, match="test_size \\+ val_size must be < 1.0"):
        load_and_split(test_size=0.6, val_size=0.5)
    
    # test_size out of range
    with pytest.raises(ValueError):
        load_and_split(test_size=1.5)
    
    # val_size out of range
    with pytest.raises(ValueError):
        load_and_split(val_size=-0.1)


def test_reproducibility():
    """Test that same random_state produces same splits."""
    result1 = load_and_split(random_state=42, n_samples=100)
    result2 = load_and_split(random_state=42, n_samples=100)
    
    X_train1, _, _, y_train1, _, _ = result1
    X_train2, _, _, y_train2, _, _ = result2
    
    # Check that data is identical
    pd.testing.assert_frame_equal(X_train1, X_train2)
    pd.testing.assert_series_equal(y_train1, y_train2)


def test_feature_count():
    """Test that generated data has correct number of features."""
    n_features = 15
    X_train, _, _, _, _, _ = load_and_split(
        n_features=n_features,
        n_samples=100,
        random_state=42
    )
    
    assert X_train.shape[1] == n_features


def test_no_missing_values():
    """Test that data has no missing or infinite values."""
    X_train, X_val, X_test, y_train, y_val, y_test = load_and_split(
        random_state=42,
        n_samples=100
    )
    
    # Check for NaN
    assert not X_train.isnull().any().any()
    assert not X_val.isnull().any().any()
    assert not X_test.isnull().any().any()
    assert not y_train.isnull().any()
    assert not y_val.isnull().any()
    assert not y_test.isnull().any()
    
    # Check for inf
    assert not np.isinf(X_train.values).any()
    assert not np.isinf(X_val.values).any()
    assert not np.isinf(X_test.values).any()


def test_class_distribution():
    """Test that class distribution is as expected."""
    X_train, _, _, y_train, _, _ = load_and_split(
        random_state=42,
        n_samples=1000
    )
    
    # Check that we have binary classification
    assert set(y_train.unique()) == {0, 1}
    
    # Check imbalance (should be roughly 70/30)
    class_counts = y_train.value_counts()
    ratio = class_counts[0] / class_counts[1]
    assert 1.5 <= ratio <= 3.0  # Allow some variance
