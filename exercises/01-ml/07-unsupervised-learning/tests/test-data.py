"""Tests for data loading module"""

import pytest
import pandas as pd
from src.data import load_and_prepare


class TestLoadAndPrepare:
    """Test data loading and preparation."""
    
    def test_load_iris(self):
        """Test loading Iris dataset."""
        X, y_true = load_and_prepare(dataset="iris")
        
        assert isinstance(X, pd.DataFrame)
        assert len(X) == 150
        assert X.shape[1] == 4
        assert y_true is not None
        assert len(y_true.unique()) == 3
    
    def test_load_wine(self):
        """Test loading Wine dataset."""
        X, y_true = load_and_prepare(dataset="wine")
        
        assert isinstance(X, pd.DataFrame)
        assert len(X) == 178
        assert X.shape[1] == 13
        assert y_true is not None
        assert len(y_true.unique()) == 3
    
    def test_load_make_blobs(self):
        """Test generating synthetic clustered data."""
        X, y_true = load_and_prepare(
            dataset="make_blobs",
            n_samples=500,
            n_features=2,
            centers=4
        )
        
        assert isinstance(X, pd.DataFrame)
        assert len(X) == 500
        assert X.shape[1] == 2
        assert y_true is not None
        assert len(y_true.unique()) == 4
    
    def test_no_ground_truth(self):
        """Test loading without ground truth labels."""
        X, y_true = load_and_prepare(dataset="iris", include_ground_truth=False)
        
        assert isinstance(X, pd.DataFrame)
        assert y_true is None
    
    def test_invalid_dataset(self):
        """Test error handling for invalid dataset."""
        with pytest.raises(ValueError, match="dataset must be one of"):
            load_and_prepare(dataset="invalid")
    
    def test_reproducibility(self):
        """Test that same random_state gives same data."""
        X1, y1 = load_and_prepare(dataset="make_blobs", random_state=42)
        X2, y2 = load_and_prepare(dataset="make_blobs", random_state=42)
        
        pd.testing.assert_frame_equal(X1, X2)
        pd.testing.assert_series_equal(y1, y2)
    
    def test_no_missing_values(self):
        """Test that loaded data has no missing values."""
        X, y_true = load_and_prepare(dataset="iris")
        
        assert not X.isnull().any().any()
        assert not y_true.isnull().any()
    
    def test_no_infinite_values(self):
        """Test that loaded data has no infinite values."""
        import numpy as np
        X, y_true = load_and_prepare(dataset="iris")
        
        assert not np.isinf(X.values).any()
