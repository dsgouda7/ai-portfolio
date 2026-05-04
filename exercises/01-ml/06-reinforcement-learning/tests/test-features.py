"""Tests for state preprocessing"""

import pytest
import numpy as np

from src.features import StatePreprocessor


class TestStatePreprocessor:
    """Test suite for StatePreprocessor."""
    
    def test_preprocessor_initialization(self):
        """Test preprocessor initializes correctly."""
        preprocessor = StatePreprocessor(clip_range=10.0)
        
        assert preprocessor.clip_range == 10.0
        assert not preprocessor.is_fitted
    
    def test_preprocessor_fit(self):
        """Test fitting scaler on training data."""
        preprocessor = StatePreprocessor()
        
        # Generate random training states
        states = np.random.randn(100, 4)
        preprocessor.fit(states)
        
        assert preprocessor.is_fitted
        assert preprocessor.scaler.mean_ is not None
        assert preprocessor.scaler.scale_ is not None
    
    def test_preprocessor_transform_single_state(self):
        """Test transforming single state."""
        preprocessor = StatePreprocessor()
        
        states = np.random.randn(100, 4)
        preprocessor.fit(states)
        
        single_state = np.array([0.1, 0.2, -0.3, 0.4])
        normalized = preprocessor.transform(single_state)
        
        assert normalized.shape == (4,)
        assert isinstance(normalized, np.ndarray)
    
    def test_preprocessor_transform_batch(self):
        """Test transforming batch of states."""
        preprocessor = StatePreprocessor()
        
        states = np.random.randn(100, 4)
        preprocessor.fit(states)
        
        test_states = np.random.randn(20, 4)
        normalized = preprocessor.transform(test_states)
        
        assert normalized.shape == (20, 4)
        assert isinstance(normalized, np.ndarray)
    
    def test_preprocessor_fit_transform(self):
        """Test fit and transform in one step."""
        preprocessor = StatePreprocessor()
        
        states = np.random.randn(100, 4)
        normalized = preprocessor.fit_transform(states)
        
        assert preprocessor.is_fitted
        assert normalized.shape == states.shape
        
        # Check normalization (mean ~0, std ~1)
        assert np.abs(np.mean(normalized)) < 0.1
        assert np.abs(np.std(normalized) - 1.0) < 0.1
    
    def test_preprocessor_clipping(self):
        """Test that outliers are clipped."""
        preprocessor = StatePreprocessor(clip_range=2.0)
        
        states = np.random.randn(100, 4)
        preprocessor.fit(states)
        
        # Create outlier state
        outlier = np.array([100.0, 100.0, 100.0, 100.0])
        normalized = preprocessor.transform(outlier, clip=True)
        
        # Should be clipped to [-2, 2]
        assert np.all(normalized >= -2.0)
        assert np.all(normalized <= 2.0)
    
    def test_preprocessor_no_clipping(self):
        """Test transform without clipping."""
        preprocessor = StatePreprocessor(clip_range=2.0)
        
        states = np.random.randn(100, 4)
        preprocessor.fit(states)
        
        outlier = np.array([100.0, 100.0, 100.0, 100.0])
        normalized = preprocessor.transform(outlier, clip=False)
        
        # Should have large values (not clipped)
        assert np.any(np.abs(normalized) > 2.0)
    
    def test_preprocessor_inverse_transform(self):
        """Test inverse transformation."""
        preprocessor = StatePreprocessor()
        
        states = np.random.randn(100, 4)
        preprocessor.fit(states)
        
        original = np.array([0.1, 0.2, -0.3, 0.4])
        normalized = preprocessor.transform(original, clip=False)
        recovered = preprocessor.inverse_transform(normalized)
        
        # Should recover original (approximately)
        np.testing.assert_array_almost_equal(original, recovered, decimal=5)
    
    def test_preprocessor_not_fitted_error(self):
        """Test that transform raises error if not fitted."""
        preprocessor = StatePreprocessor()
        
        state = np.array([0.1, 0.2, -0.3, 0.4])
        
        with pytest.raises(RuntimeError):
            preprocessor.transform(state)
    
    def test_preprocessor_invalid_state_shape(self):
        """Test that invalid state shape raises error."""
        preprocessor = StatePreprocessor()
        
        states = np.random.randn(100, 4)
        preprocessor.fit(states)
        
        # 3D array should fail
        invalid_state = np.random.randn(2, 3, 4)
        
        with pytest.raises(ValueError):
            preprocessor.transform(invalid_state)
    
    def test_preprocessor_get_params(self):
        """Test getting scaler parameters."""
        preprocessor = StatePreprocessor()
        
        states = np.random.randn(100, 4)
        preprocessor.fit(states)
        
        params = preprocessor.get_params()
        
        assert "mean" in params
        assert "scale" in params
        assert "clip_range" in params
        assert len(params["mean"]) == 4
        assert len(params["scale"]) == 4
