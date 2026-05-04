"""Tests for data loading and splitting"""

import pytest
import numpy as np

from src.data import load_and_split, _validate_splits


class TestLoadAndSplit:
    """Test suite for load_and_split function."""
    
    def test_load_and_split_default(self):
        """Test loading with default parameters."""
        X_train, X_val, X_test, y_train, y_val, y_test = load_and_split()
        
        # Check types
        assert isinstance(X_train, np.ndarray)
        assert isinstance(y_train, np.ndarray)
        
        # Check shapes
        assert X_train.shape[0] == len(y_train)
        assert X_val.shape[0] == len(y_val)
        assert X_test.shape[0] == len(y_test)
        
        # Check feature count
        assert X_train.shape[1] == X_val.shape[1] == X_test.shape[1]
        
        # Check no missing values
        assert not np.isnan(X_train).any()
        assert not np.isnan(y_train).any()
    
    def test_load_and_split_custom_params(self):
        """Test loading with custom parameters."""
        X_train, X_val, X_test, y_train, y_val, y_test = load_and_split(
            n_samples=1000,
            n_features=10,
            n_classes=5,
            test_size=0.3,
            val_size=0.15
        )
        
        # Check total samples
        total = len(X_train) + len(X_val) + len(X_test)
        assert total == 1000
        
        # Check feature count
        assert X_train.shape[1] == 10
        
        # Check class count
        assert len(np.unique(y_train)) <= 5
    
    def test_load_and_split_reproducibility(self):
        """Test that same random_state produces same splits."""
        X_train1, _, _, y_train1, _, _ = load_and_split(
            n_samples=1000, random_state=42
        )
        X_train2, _, _, y_train2, _, _ = load_and_split(
            n_samples=1000, random_state=42
        )
        
        # Should be identical
        np.testing.assert_array_equal(X_train1, X_train2)
        np.testing.assert_array_equal(y_train1, y_train2)
    
    def test_load_and_split_invalid_test_size(self):
        """Test that invalid test_size raises ValueError."""
        with pytest.raises(ValueError):
            load_and_split(test_size=-0.1)
        
        with pytest.raises(ValueError):
            load_and_split(test_size=1.5)
    
    def test_load_and_split_invalid_sizes(self):
        """Test that test_size + val_size >= 1.0 raises ValueError."""
        with pytest.raises(ValueError):
            load_and_split(test_size=0.7, val_size=0.4)


class TestValidateSplits:
    """Test suite for _validate_splits function."""
    
    def test_validate_splits_valid(self):
        """Test validation passes for valid splits."""
        X_train = np.random.randn(100, 10)
        X_val = np.random.randn(20, 10)
        X_test = np.random.randn(30, 10)
        y_train = np.random.randint(0, 5, 100)
        y_val = np.random.randint(0, 5, 20)
        y_test = np.random.randint(0, 5, 30)
        
        # Should not raise
        _validate_splits(X_train, X_val, X_test, y_train, y_val, y_test)
    
    def test_validate_splits_nan_values(self):
        """Test validation fails for NaN values."""
        X_train = np.random.randn(100, 10)
        X_train[0, 0] = np.nan
        X_val = np.random.randn(20, 10)
        X_test = np.random.randn(30, 10)
        y_train = np.random.randint(0, 5, 100)
        y_val = np.random.randint(0, 5, 20)
        y_test = np.random.randint(0, 5, 30)
        
        with pytest.raises(ValueError, match="Missing values"):
            _validate_splits(X_train, X_val, X_test, y_train, y_val, y_test)
    
    def test_validate_splits_feature_mismatch(self):
        """Test validation fails for feature count mismatch."""
        X_train = np.random.randn(100, 10)
        X_val = np.random.randn(20, 8)  # Wrong feature count
        X_test = np.random.randn(30, 10)
        y_train = np.random.randint(0, 5, 100)
        y_val = np.random.randint(0, 5, 20)
        y_test = np.random.randint(0, 5, 30)
        
        with pytest.raises(ValueError, match="mismatch"):
            _validate_splits(X_train, X_val, X_test, y_train, y_val, y_test)
