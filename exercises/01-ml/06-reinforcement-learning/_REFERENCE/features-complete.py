"""State preprocessing for AgentAI

Provides: State normalization, feature scaling, preprocessing pipeline
"""

import logging
from typing import Optional

import numpy as np
from sklearn.preprocessing import StandardScaler

from src.utils import validate_positive


logger = logging.getLogger("agentai")


class StatePreprocessor:
    """Preprocess states for policy networks.
    
    Provides:
    - State normalization (zero mean, unit variance)
    - Clipping to prevent outliers
    - Batch processing support
    
    Attributes:
        scaler: Fitted StandardScaler
        clip_range: Range to clip normalized values
        is_fitted: Whether scaler has been fitted
    
    Example:
        >>> preprocessor = StatePreprocessor()
        >>> preprocessor.fit(training_states)
        >>> normalized_state = preprocessor.transform(state)
    """
    
    def __init__(self, clip_range: float = 10.0):
        """Initialize state preprocessor.
        
        Args:
            clip_range: Clip normalized values to [-clip_range, +clip_range]
        
        Raises:
            ValueError: If clip_range is not positive
        """
        validate_positive(clip_range, "clip_range")
        
        self.scaler = StandardScaler()
        self.clip_range = clip_range
        self.is_fitted = False
        
        logger.info(f"Initialized StatePreprocessor (clip_range={clip_range})")
    
    def fit(self, states: np.ndarray) -> "StatePreprocessor":
        """Fit scaler on training states.
        
        Args:
            states: Array of states (n_samples, n_features)
        
        Returns:
            Self for method chaining
        
        Raises:
            ValueError: If states array is invalid
        
        Example:
            >>> preprocessor.fit(training_states)
        """
        if states.ndim != 2:
            raise ValueError(f"States must be 2D array, got shape {states.shape}")
        
        if len(states) == 0:
            raise ValueError("Cannot fit on empty states array")
        
        logger.info(f"Fitting scaler on {len(states)} states")
        
        try:
            self.scaler.fit(states)
            self.is_fitted = True
            
            logger.info(
                f"Scaler fitted - Mean: {self.scaler.mean_}, "
                f"Std: {self.scaler.scale_}"
            )
            
            return self
            
        except Exception as e:
            logger.error(f"Scaler fitting failed: {e}")
            raise RuntimeError(f"Failed to fit scaler: {e}") from e
    
    def transform(
        self,
        states: np.ndarray,
        clip: bool = True
    ) -> np.ndarray:
        """Transform states using fitted scaler.
        
        Args:
            states: States to transform (can be single state or batch)
            clip: Whether to clip values to [-clip_range, +clip_range]
        
        Returns:
            Normalized states
        
        Raises:
            RuntimeError: If scaler not fitted
            ValueError: If states shape is invalid
        
        Example:
            >>> normalized = preprocessor.transform(state)
        """
        if not self.is_fitted:
            raise RuntimeError("Scaler not fitted. Call fit() first.")
        
        # Handle single state (1D array)
        single_state = False
        if states.ndim == 1:
            states = states.reshape(1, -1)
            single_state = True
        
        if states.ndim != 2:
            raise ValueError(f"States must be 1D or 2D array, got shape {states.shape}")
        
        try:
            # Normalize
            normalized = self.scaler.transform(states)
            
            # Clip outliers
            if clip:
                normalized = np.clip(normalized, -self.clip_range, self.clip_range)
            
            # Return single state if input was single
            if single_state:
                normalized = normalized[0]
            
            return normalized
            
        except Exception as e:
            logger.error(f"State transformation failed: {e}")
            raise RuntimeError(f"Failed to transform states: {e}") from e
    
    def fit_transform(
        self,
        states: np.ndarray,
        clip: bool = True
    ) -> np.ndarray:
        """Fit scaler and transform states in one step.
        
        Args:
            states: States to fit and transform
            clip: Whether to clip values
        
        Returns:
            Normalized states
        
        Example:
            >>> normalized = preprocessor.fit_transform(training_states)
        """
        self.fit(states)
        return self.transform(states, clip=clip)
    
    def inverse_transform(self, normalized_states: np.ndarray) -> np.ndarray:
        """Convert normalized states back to original scale.
        
        Args:
            normalized_states: Normalized states
        
        Returns:
            States in original scale
        
        Raises:
            RuntimeError: If scaler not fitted
        
        Example:
            >>> original = preprocessor.inverse_transform(normalized)
        """
        if not self.is_fitted:
            raise RuntimeError("Scaler not fitted. Call fit() first.")
        
        # Handle single state
        single_state = False
        if normalized_states.ndim == 1:
            normalized_states = normalized_states.reshape(1, -1)
            single_state = True
        
        try:
            original = self.scaler.inverse_transform(normalized_states)
            
            if single_state:
                original = original[0]
            
            return original
            
        except Exception as e:
            logger.error(f"Inverse transformation failed: {e}")
            raise RuntimeError(f"Failed to inverse transform: {e}") from e
    
    def get_params(self) -> dict:
        """Get scaler parameters.
        
        Returns:
            Dictionary with mean and scale
        
        Raises:
            RuntimeError: If scaler not fitted
        """
        if not self.is_fitted:
            raise RuntimeError("Scaler not fitted. Call fit() first.")
        
        return {
            "mean": self.scaler.mean_.tolist(),
            "scale": self.scaler.scale_.tolist(),
            "clip_range": self.clip_range,
        }
