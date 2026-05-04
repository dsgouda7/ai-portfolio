"""Tests for data loading and splitting"""

import pytest
import numpy as np

from src.data import load_and_split, load_dataset_info, _validate_splits


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
    
    def test_load_and_split_reproducibility(self):
        """Test that same random_state produces same splits."""
        X_train1, _, _, y_train1, _, _ = load_and_split(random_state=42)
        X_train2, _, _, y_train2, _, _ = load_and_split(random_state=42)
        
        # Should be identical
        np.testing.assert_array_equal(X_train1, X_train2)
        np.testing.assert_array_equal(y_train1, y_train2)
    
    def test_load_and_split_invalid_test_size(self):
        """Test that invalid test_size raises ValueError."""
        with pytest.raises(ValueError):
            load_and_split(test_size=-0.1)
        
        with pytest.raises(ValueError):
            load_and_split(test_size=1.5)


class TestLoadDatasetInfo:
    """Test suite for load_dataset_info function."""
    
    def test_load_dataset_info(self):
        """Test that dataset info is loaded correctly."""
        info = load_dataset_info()
        
        # Check keys
        assert "n_samples" in info
        assert "n_features" in info
        assert "n_classes" in info
        
        # Check values
        assert info["n_features"] == 4096  # 64x64 images
        assert info["n_classes"] == 40  # Olivetti faces has 40 subjects
