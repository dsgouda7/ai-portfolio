"""State preprocessing and feature engineering for AgentAI

This module provides:
- StatePreprocessor for normalization and scaling (with TODOs)
- Feature engineering for state representation
- Immediate feedback during preprocessing

Learning objectives:
1. Implement state normalization with StandardScaler
2. Handle continuous vs discrete state spaces
3. Apply clipping to prevent outliers
4. See preprocessing statistics immediately
"""

import logging
from typing import Optional

import numpy as np
from rich.console import Console
from sklearn.preprocessing import StandardScaler

from src.utils import validate_positive

logger = logging.getLogger("agentai")
console = Console()


class StatePreprocessor:
    """Preprocess states for policy networks.
    
    Provides:
    - State normalization (zero mean, unit variance)
    - Clipping to prevent outliers
    - Batch processing support
    
    Why normalization matters:
    - Neural networks train faster with normalized inputs
    - Prevents large state values from dominating gradients
    - Improves numerical stability
    
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
        """
        TODO: Fit scaler on training states
        """
        # TODO: Your implementation here
        raise NotImplementedError("Implement scaler fitting")
    
    def transform(
        self,
        states: np.ndarray,
        clip: bool = True
    ) -> np.ndarray:
        """
        TODO: Transform states using fitted scaler
        """
        # TODO: Your implementation here
        raise NotImplementedError("Implement state transformation")
    
    def fit_transform(
        self,
        states: np.ndarray,
        clip: bool = True
    ) -> np.ndarray:
        """
        TODO: Fit scaler and transform states in one step
        """
        # TODO: Your implementation here
        raise NotImplementedError("Implement fit_transform")
    
    def inverse_transform(self, normalized_states: np.ndarray) -> np.ndarray:
        """
        TODO: Convert normalized states back to original scale
        """
        # TODO: Your implementation here
        raise NotImplementedError("Implement inverse transform")
    
    def get_params(self) -> dict:
        """Get scaler parameters.
        
        Returns:
            Dictionary with mean and std
        
        Example:
            >>> params = preprocessor.get_params()
            {"mean": [0.02, 0.01, -0.03, 0.00], "std": [0.52, 0.89, 0.21, 0.95]}
        """
        if not self.is_fitted:
            raise RuntimeError("Scaler not fitted. Call fit() first.")
        
        return {
            "mean": self.scaler.mean_.tolist(),
            "std": self.scaler.scale_.tolist(),
            "clip_range": self.clip_range
        }


class StateEncoder:
    """Encode categorical or discrete state features.
    
    For environments with discrete actions or mixed state spaces.
    (Optional - not needed for CartPole, but useful for other environments)
    """
    
    def __init__(self, n_categories: int):
        """Initialize state encoder.
        
        Args:
            n_categories: Number of discrete categories
        """
        self.n_categories = n_categories
    
    def one_hot_encode(self, state: int) -> np.ndarray:
        """
        TODO (Optional): One-hot encode discrete state
        """
        # TODO: Your implementation here (optional)
        raise NotImplementedError("Implement one-hot encoding")


class FeatureEngineer:
    """Engineer features for RL state representation.
    
    Provides:
    - Polynomial features (e.g., x² for nonlinear relationships)
    - State history (previous N states for temporal context)
    - Domain-specific features (e.g., angular velocity for pendulum)
    
    (Optional - for advanced exercises)
    """
    
    def __init__(self, polynomial_degree: int = 1, history_length: int = 1):
        """Initialize feature engineer.
        
        Args:
            polynomial_degree: Degree for polynomial features
            history_length: Number of previous states to include
        """
        self.polynomial_degree = polynomial_degree
        self.history_length = history_length
        self.state_history = []
    
    def add_polynomial_features(self, state: np.ndarray) -> np.ndarray:
        """
        TODO (Optional): Add polynomial features
        """
        # TODO: Your implementation here (optional)
        raise NotImplementedError("Implement polynomial features")
    
    def add_state_history(self, state: np.ndarray) -> np.ndarray:
        """
        TODO (Optional): Add previous states as features
        """
        # TODO: Your implementation here (optional)
        raise NotImplementedError("Implement state history")
