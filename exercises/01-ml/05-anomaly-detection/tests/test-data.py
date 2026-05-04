"""Tests for data loading and splitting for FraudShield"""

import pytest
import pandas as pd
import numpy as np

from src.data import load_and_split, get_dataset_info, _validate_splits


class TestLoadAndSplit:
    """Test suite for load_and_split function."""
    
    def test_load_and_split_default(self):
        """Test loading with default parameters."""
        X_train, X_val, X_test, y_train, y_val, y_test = load_and_split(
            n_samples=1000, n_features=10
        )
        
        # Check types
        assert isinstance(X_train, pd.DataFrame)
        assert isinstance(y_train, pd.Series)
        
        # Check shapes
        assert X_train.shape[0] == len(y_train)
        assert X_val.shape[0] == len(y_val)
        assert X_test.shape[0] == len(y_test)
        
        # Check feature count
        assert X_train.shape[1] == X_val.shape[1] == X_test.shape[1] == 10
        
        # Check no missing values
        assert not X_train.isnull().any().any()
        assert not y_train.isnull().any()
        
        # Check binary labels
        assert set(y_train.unique()).issubset({0, 1})
    
    def test_load_and_split_custom_contamination(self):
        """Test loading with custom contamination rate."""
        contamination = 0.15
        X_train, X_val, X_test, y_train, y_val, y_test = load_and_split(
            n_samples=1000,
            n_features=10,
            contamination=contamination
        )
        
        total_samples = len(y_train) + len(y_val) + len(y_test)
        total_anomalies = (y_train == 1).sum() + (y_val == 1).sum() + (y_test == 1).sum()
        
        actual_contamination = total_anomalies / total_samples
        
        # Should be close to target contamination (within 5%)
        assert abs(actual_contamination - contamination) < 0.05
    
    def test_load_and_split_reproducibility(self):
        """Test that same random_state produces same splits."""
        X_train1, _, _, y_train1, _, _ = load_and_split(
            n_samples=500, n_features=10, random_state=42
        )
        X_train2, _, _, y_train2, _, _ = load_and_split(
            n_samples=500, n_features=10, random_state=42
        )
        
        # Should be identical
        pd.testing.assert_frame_equal(X_train1, X_train2)
        pd.testing.assert_series_equal(y_train1, y_train2)
    
    def test_load_and_split_no_data_leakage(self):
        """Test that there's no overlap between train/val/test sets."""
        X_train, X_val, X_test, _, _, _ = load_and_split(
            n_samples=1000, n_features=10
        )
        
        train_idx = set(X_train.index)
        val_idx = set(X_val.index)
        test_idx = set(X_test.index)
        
        # No overlap
        assert len(train_idx & val_idx) == 0
        assert len(train_idx & test_idx) == 0
        assert len(val_idx & test_idx) == 0
    
    def test_load_and_split_both_classes_present(self):
        """Test that both classes are present in all splits."""
        X_train, X_val, X_test, y_train, y_val, y_test = load_and_split(
            n_samples=1000, n_features=10, contamination=0.1
        )
        
        # Both classes in all splits
        assert len(y_train.unique()) == 2
        assert len(y_val.unique()) == 2
        assert len(y_test.unique()) == 2
    
    def test_load_and_split_invalid_contamination(self):
        """Test that invalid contamination raises ValueError."""
        with pytest.raises(ValueError):
            load_and_split(n_samples=1000, contamination=-0.1)
        
        with pytest.raises(ValueError):
            load_and_split(n_samples=1000, contamination=1.5)
    
    def test_load_and_split_invalid_n_samples(self):
        """Test that invalid n_samples raises ValueError."""
        with pytest.raises(ValueError):
            load_and_split(n_samples=50)  # Too few samples
    
    def test_load_and_split_invalid_n_features(self):
        """Test that invalid n_features raises ValueError."""
        with pytest.raises(ValueError):
            load_and_split(n_samples=1000, n_features=1)  # Too few features


class TestGetDatasetInfo:
    """Test suite for get_dataset_info function."""
    
    def test_get_dataset_info(self):
        """Test getting dataset info."""
        info = get_dataset_info()
        
        assert 'n_features' in info
        assert 'feature_names' in info
        assert 'target_name' in info
        assert 'classes' in info
        
        assert info['target_name'] == 'is_anomaly'
        assert info['classes'] == ['normal', 'anomaly']


class TestValidateSplits:
    """Test suite for _validate_splits function."""
    
    def test_validate_splits_valid(self):
        """Test validation with valid splits."""
        X_train, X_val, X_test, y_train, y_val, y_test = load_and_split(
            n_samples=1000, n_features=10
        )
        
        # Should not raise
        _validate_splits(X_train, X_val, X_test, y_train, y_val, y_test)
    
    def test_validate_splits_missing_values(self):
        """Test validation fails with missing values."""
        X_train = pd.DataFrame([[1, 2], [np.nan, 4]])
        X_val = pd.DataFrame([[1, 2]])
        X_test = pd.DataFrame([[1, 2]])
        y_train = pd.Series([0, 1])
        y_val = pd.Series([0])
        y_test = pd.Series([1])
        
        with pytest.raises(ValueError, match="Missing values"):
            _validate_splits(X_train, X_val, X_test, y_train, y_val, y_test)
    
    def test_validate_splits_length_mismatch(self):
        """Test validation fails with length mismatch."""
        X_train = pd.DataFrame([[1, 2], [3, 4]])
        X_val = pd.DataFrame([[1, 2]])
        X_test = pd.DataFrame([[1, 2]])
        y_train = pd.Series([0, 1, 0])  # Length mismatch
        y_val = pd.Series([0])
        y_test = pd.Series([1])
        
        with pytest.raises(ValueError, match="length mismatch"):
            _validate_splits(X_train, X_val, X_test, y_train, y_val, y_test)
