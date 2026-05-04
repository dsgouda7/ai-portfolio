"""Feature engineering for FaceAI

Provides: HOG features, PCA dimensionality reduction, scaling
"""

import logging
from typing import Optional, Tuple

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from skimage.feature import hog


logger = logging.getLogger("faceai")


class FeatureEngineer:
    """Feature engineering pipeline with HOG features and dimensionality reduction.
    
    Pipeline stages:
    1. HOG (Histogram of Oriented Gradients) feature extraction
    2. PCA dimensionality reduction (optional)
    3. Standardization (zero mean, unit variance)
    
    Attributes:
        scaler: StandardScaler transformer
        pca: PCA transformer
        feature_dim: Dimension of engineered features
    
    Example:
        >>> engineer = FeatureEngineer(hog_orientations=9, pca_components=50)
        >>> X_train_transformed = engineer.fit_transform(X_train)
        >>> X_test_transformed = engineer.transform(X_test)
    """
    
    def __init__(
        self,
        hog_orientations: int = 9,
        hog_pixels_per_cell: Tuple[int, int] = (8, 8),
        hog_cells_per_block: Tuple[int, int] = (2, 2),
        pca_components: Optional[int] = None,
        scale_features: bool = True
    ):
        """Initialize feature engineer.
        
        Args:
            hog_orientations: Number of orientation bins for HOG
            hog_pixels_per_cell: Size of cell for HOG
            hog_cells_per_block: Number of cells per block for HOG
            pca_components: Number of PCA components (None = no PCA)
            scale_features: Whether to standardize features
        
        Raises:
            ValueError: If parameters are invalid
        """
        if hog_orientations < 1:
            raise ValueError(f"hog_orientations must be >= 1, got {hog_orientations}")
        
        self.hog_orientations = hog_orientations
        self.hog_pixels_per_cell = hog_pixels_per_cell
        self.hog_cells_per_block = hog_cells_per_block
        self.pca_components = pca_components
        self.scale_features = scale_features
        
        self.scaler = None
        self.pca = None
        self.feature_dim = None
        self._fitted = False
        self._original_image_shape = None
        
        logger.info(
            f"Initialized FeatureEngineer (orientations={hog_orientations}, "
            f"pca={pca_components}, scale={scale_features})"
        )
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit feature engineering pipeline and transform data.
        
        Args:
            X: Input images (flattened or 2D arrays)
        
        Returns:
            Transformed features
        
        Raises:
            ValueError: If X contains NaN or inf values
        """
        self._validate_input(X, "fit_transform")
        
        # Store original shape for transformation
        if len(X.shape) == 2 and X.shape[1] == 4096:  # 64x64 flattened
            self._original_image_shape = (64, 64)
        else:
            self._original_image_shape = (64, 64)  # Default for Olivetti
        
        logger.info(f"Extracting HOG features from {len(X)} images")
        
        # Stage 1: HOG feature extraction
        X_hog = self._extract_hog_features(X)
        logger.info(f"HOG features extracted: {X_hog.shape[1]} dimensions")
        
        # Stage 2: PCA dimensionality reduction
        if self.pca_components is not None:
            logger.info(f"Applying PCA (components={self.pca_components})")
            self.pca = PCA(n_components=self.pca_components, random_state=42)
            X_hog = self.pca.fit_transform(X_hog)
            
            explained_var = self.pca.explained_variance_ratio_.sum()
            logger.info(
                f"PCA reduced to {self.pca_components} components "
                f"(explained variance: {explained_var:.2%})"
            )
        
        # Stage 3: Standardization
        if self.scale_features:
            logger.info("Standardizing features")
            self.scaler = StandardScaler()
            X_hog = self.scaler.fit_transform(X_hog)
        
        self.feature_dim = X_hog.shape[1]
        self._fitted = True
        
        logger.info(f"Feature engineering complete: {self.feature_dim} features")
        
        return X_hog
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform data using fitted pipeline.
        
        Args:
            X: Input images
        
        Returns:
            Transformed features
        
        Raises:
            RuntimeError: If pipeline not fitted
            ValueError: If X contains NaN/inf or wrong shape
        """
        if not self._fitted:
            raise RuntimeError("Pipeline not fitted. Call fit_transform() first.")
        
        self._validate_input(X, "transform")
        
        # Stage 1: HOG feature extraction
        X_hog = self._extract_hog_features(X)
        
        # Stage 2: PCA transformation
        if self.pca is not None:
            X_hog = self.pca.transform(X_hog)
        
        # Stage 3: Standardization
        if self.scaler is not None:
            X_hog = self.scaler.transform(X_hog)
        
        return X_hog
    
    def _extract_hog_features(self, X: np.ndarray) -> np.ndarray:
        """Extract HOG features from images.
        
        Args:
            X: Input images (flattened or 2D)
        
        Returns:
            HOG feature array
        """
        hog_features = []
        
        for i in range(len(X)):
            # Reshape flattened image if needed
            if len(X[i].shape) == 1:
                img = X[i].reshape(self._original_image_shape)
            else:
                img = X[i]
            
            try:
                # Extract HOG features
                features = hog(
                    img,
                    orientations=self.hog_orientations,
                    pixels_per_cell=self.hog_pixels_per_cell,
                    cells_per_block=self.hog_cells_per_block,
                    feature_vector=True
                )
                hog_features.append(features)
                
            except Exception as e:
                logger.error(f"HOG extraction failed for image {i}: {e}")
                raise RuntimeError(f"HOG extraction failed: {e}") from e
        
        return np.array(hog_features)
    
    def _validate_input(self, X: np.ndarray, context: str) -> None:
        """Validate input array.
        
        Args:
            X: Input to validate
            context: Context string for error messages
        
        Raises:
            ValueError: If validation fails
        """
        if not isinstance(X, np.ndarray):
            raise ValueError(f"{context}: X must be a numpy array")
        
        if X.size == 0:
            raise ValueError(f"{context}: X is empty")
        
        if np.isnan(X).any():
            raise ValueError(f"{context}: X contains NaN values")
        
        if np.isinf(X).any():
            raise ValueError(f"{context}: X contains infinite values")
    
    def get_feature_info(self) -> dict:
        """Get information about extracted features.
        
        Returns:
            Dictionary with feature statistics and parameters
        """
        if not self._fitted:
            raise RuntimeError("Pipeline not fitted. Call fit_transform() first.")
        
        info = {
            "feature_dim": self.feature_dim,
            "hog_orientations": self.hog_orientations,
            "hog_pixels_per_cell": self.hog_pixels_per_cell,
            "hog_cells_per_block": self.hog_cells_per_block,
            "pca_components": self.pca_components,
        }
        
        if self.pca is not None:
            info["explained_variance_ratio"] = self.pca.explained_variance_ratio_.tolist()
            info["cumulative_variance"] = self.pca.explained_variance_ratio_.cumsum().tolist()
        
        if self.scaler is not None:
            info["feature_means"] = self.scaler.mean_.tolist()
            info["feature_stds"] = self.scaler.scale_.tolist()
        
        return info
