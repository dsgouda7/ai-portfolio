"""Tests for data loading and splitting"""

import pytest
import pandas as pd
import numpy as np

from src.data import load_and_split, load_dataset_info, _validate_splits


class TestLoadAndSplit:
    """Test suite for load_and_split function."""
    
    def test_load_and_split_default(self):
        """Test loading with default parameters."""
        X_train, X_val, X_test, y_train, y_val, y_test = load_and_split()
        
        # Check types
        assert isinstance(X_train, pd.DataFrame)
        assert isinstance(y_train, pd.Series)
        
        # Check shapes
        assert X_train.shape[0] == len(y_train)
        assert X_val.shape[0] == len(y_val)
        assert X_test.shape[0] == len(y_test)
        
        # Check feature count
        assert X_train.shape[1] == X_val.shape[1] == X_test.shape[1]
        
        # Check no missing values
        assert not X_train.isnull().any().any()
        assert not y_train.isnull().any()
    
    def test_load_and_split_custom_sizes(self):
        """Test loading with custom split sizes."""
        X_train, X_val, X_test, y_train, y_val, y_test = load_and_split(
            test_size=0.3,
            val_size=0.15
        )
        
        total_samples = len(X_train) + len(X_val) + len(X_test)
        
        # Test size should be approximately 30%
        test_ratio = len(X_test) / total_samples
        assert 0.28 <= test_ratio <= 0.32
        
        # Val size should be approximately 15% of non-test data
        val_ratio = len(X_val) / total_samples
        assert 0.13 <= val_ratio <= 0.17
    
    def test_load_and_split_reproducibility(self):
        """Test that same random_state produces same splits."""
        X_train1, _, _, y_train1, _, _ = load_and_split(random_state=42)
        X_train2, _, _, y_train2, _, _ = load_and_split(random_state=42)
        
        # Should be identical
        pd.testing.assert_frame_equal(X_train1, X_train2)
        pd.testing.assert_series_equal(y_train1, y_train2)
    
    def test_load_and_split_different_seeds(self):
        """Test that different random_state produces different splits."""
        X_train1, _, _, _, _, _ = load_and_split(random_state=42)
        X_train2, _, _, _, _, _ = load_and_split(random_state=123)
        
        # Should be different (at least some rows)
        assert not X_train1.equals(X_train2)
    
    def test_load_and_split_no_data_leakage(self):
        """Test that there's no overlap between train/val/test sets."""
        X_train, X_val, X_test, _, _, _ = load_and_split()
        
        train_idx = set(X_train.index)
        val_idx = set(X_val.index)
        test_idx = set(X_test.index)
        
        # No overlap
        assert len(train_idx & val_idx) == 0
        assert len(train_idx & test_idx) == 0
        assert len(val_idx & test_idx) == 0
    
    def test_load_and_split_invalid_test_size(self):
        """Test that invalid test_size raises ValueError."""
        with pytest.raises(ValueError):
            load_and_split(test_size=-0.1)
        
        with pytest.raises(ValueError):
            load_and_split(test_size=1.5)
    
    def test_load_and_split_invalid_val_size(self):
        """Test that invalid val_size raises ValueError."""
        with pytest.raises(ValueError):
            load_and_split(val_size=-0.1)
        
        with pytest.raises(ValueError):
            load_and_split(val_size=1.5)
    
    def test_load_and_split_sizes_too_large(self):
        """Test that test_size + val_size >= 1.0 raises ValueError."""
        with pytest.raises(ValueError):
            load_and_split(test_size=0.6, val_size=0.5)


class TestValidateSplits:
    """Test suite for _validate_splits function."""
    
    def test_validate_splits_valid(self):
        """Test that valid splits pass validation."""
        X_train, X_val, X_test, y_train, y_val, y_test = load_and_split()
        
        # Should not raise
        _validate_splits(X_train, X_val, X_test, y_train, y_val, y_test)
    
    def test_validate_splits_missing_values(self):
        """Test that splits with NaN fail validation."""
        X_train, X_val, X_test, y_train, y_val, y_test = load_and_split()
        
        # Add NaN to train set
        X_train_nan = X_train.copy()
        X_train_nan.iloc[0, 0] = np.nan
        
        with pytest.raises(ValueError, match="Missing values"):
            _validate_splits(X_train_nan, X_val, X_test, y_train, y_val, y_test)
    
    def test_validate_splits_length_mismatch(self):
        """Test that X and y length mismatch fails validation."""
        X_train, X_val, X_test, y_train, y_val, y_test = load_and_split()
        
        # Remove one row from y_train
        y_train_short = y_train.iloc[:-1]
        
        with pytest.raises(ValueError, match="length mismatch"):
            _validate_splits(X_train, X_val, X_test, y_train_short, y_val, y_test)
    
    def test_validate_splits_feature_mismatch(self):
        """Test that different feature counts fail validation."""
        X_train, X_val, X_test, y_train, y_val, y_test = load_and_split()
        
        # Remove one column from val set
        X_val_fewer = X_val.drop(columns=[X_val.columns[0]])
        
        with pytest.raises(ValueError, match="features mismatch"):
            _validate_splits(X_train, X_val_fewer, X_test, y_train, y_val, y_test)


class TestLoadDatasetInfo:
    """Test suite for load_dataset_info function."""
    
    def test_load_dataset_info(self):
        """Test that dataset info is loaded correctly."""
        info = load_dataset_info()
        
        # Check keys
        assert "feature_names" in info
        assert "target_name" in info
        assert "n_samples" in info
        assert "n_features" in info
        
        # Check values
        assert len(info["feature_names"]) == 8
        assert info["n_features"] == 8
        assert info["n_samples"] > 0
        assert "MedInc" in info["feature_names"]
