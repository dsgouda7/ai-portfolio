"""Tests for feature engineering module"""

import pytest
import pandas as pd
import numpy as np
from src.features import FeatureEngineer
from src.data import load_and_prepare


class TestFeatureEngineer:
    """Test feature engineering pipeline."""
    
    @pytest.fixture
    def sample_data(self):
        """Fixture providing sample data."""
        X, _ = load_and_prepare(dataset="iris")
        return X
    
    def test_initialization(self):
        """Test feature engineer initialization."""
        engineer = FeatureEngineer(
            scale_features=True,
            n_components_pca=2
        )
        
        assert engineer.scale_features is True
        assert engineer.n_components_pca == 2
        assert engineer._fitted is False
    
    def test_fit_transform_scaling_only(self, sample_data):
        """Test scaling without PCA."""
        engineer = FeatureEngineer(scale_features=True, n_components_pca=None)
        X_transformed = engineer.fit_transform(sample_data)
        
        assert isinstance(X_transformed, pd.DataFrame)
        assert X_transformed.shape == sample_data.shape
        assert engineer._fitted is True
        
        # Check standardization (mean ~0, std ~1)
        assert np.abs(X_transformed.mean().mean()) < 0.1
        assert np.abs(X_transformed.std().mean() - 1.0) < 0.1
    
    def test_fit_transform_with_pca(self, sample_data):
        """Test scaling + PCA."""
        engineer = FeatureEngineer(scale_features=True, n_components_pca=2)
        X_transformed = engineer.fit_transform(sample_data)
        
        assert isinstance(X_transformed, pd.DataFrame)
        assert X_transformed.shape == (len(sample_data), 2)
        assert list(X_transformed.columns) == ["PC1", "PC2"]
        assert engineer.pca is not None
    
    def test_transform_without_fit(self, sample_data):
        """Test that transform fails without fit."""
        engineer = FeatureEngineer()
        
        with pytest.raises(RuntimeError, match="not fitted"):
            engineer.transform(sample_data)
    
    def test_transform_after_fit(self, sample_data):
        """Test transform on new data after fit."""
        engineer = FeatureEngineer(scale_features=True, n_components_pca=2)
        engineer.fit_transform(sample_data)
        
        # Transform new data
        X_new = sample_data.iloc[:10]
        X_transformed = engineer.transform(X_new)
        
        assert isinstance(X_transformed, pd.DataFrame)
        assert X_transformed.shape == (10, 2)
    
    def test_umap_transformation(self, sample_data):
        """Test UMAP dimensionality reduction."""
        engineer = FeatureEngineer()
        X_scaled = engineer.fit_transform(sample_data)
        
        # Apply UMAP
        X_umap = engineer.fit_transform_umap(X_scaled)
        
        assert isinstance(X_umap, pd.DataFrame)
        assert X_umap.shape == (len(sample_data), 2)
        assert list(X_umap.columns) == ["UMAP1", "UMAP2"]
        assert engineer.umap_reducer is not None
    
    def test_umap_without_fit(self, sample_data):
        """Test that UMAP transform fails without fit."""
        engineer = FeatureEngineer()
        X_scaled = engineer.fit_transform(sample_data)
        
        with pytest.raises(RuntimeError, match="UMAP not fitted"):
            engineer.transform_umap(X_scaled)
    
    def test_invalid_pca_components(self):
        """Test error for invalid PCA components."""
        with pytest.raises(ValueError, match="n_components_pca must be >= 1"):
            FeatureEngineer(n_components_pca=0)
    
    def test_invalid_umap_neighbors(self):
        """Test error for invalid UMAP neighbors."""
        with pytest.raises(ValueError, match="n_neighbors_umap must be >= 2"):
            FeatureEngineer(n_neighbors_umap=1)
    
    def test_missing_values_validation(self, sample_data):
        """Test that missing values are detected."""
        engineer = FeatureEngineer()
        
        # Introduce NaN
        X_bad = sample_data.copy()
        X_bad.iloc[0, 0] = np.nan
        
        with pytest.raises(ValueError, match="NaN values"):
            engineer.fit_transform(X_bad)
    
    def test_infinite_values_validation(self, sample_data):
        """Test that infinite values are detected."""
        engineer = FeatureEngineer()
        
        # Introduce inf
        X_bad = sample_data.copy()
        X_bad.iloc[0, 0] = np.inf
        
        with pytest.raises(ValueError, match="infinite values"):
            engineer.fit_transform(X_bad)
    
    def test_pca_variance_retention(self, sample_data):
        """Test that PCA retains variance information."""
        engineer = FeatureEngineer(scale_features=True, n_components_pca=2)
        engineer.fit_transform(sample_data)
        
        explained_var = engineer.pca.explained_variance_ratio_.sum()
        assert 0.5 < explained_var < 1.0  # Should retain significant variance
