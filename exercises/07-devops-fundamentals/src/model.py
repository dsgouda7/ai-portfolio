"""Simple ML model for demonstration purposes."""

import joblib
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from typing import Tuple
import logging

logger = logging.getLogger(__name__)


class MLModel:
    """Simple ML model wrapper for predictions."""
    
    def __init__(self, model_path: str = None):
        """Initialize model.
        
        Args:
            model_path: Path to saved model file
        """
        self.model = None
        self.scaler = StandardScaler()
        
        if model_path and Path(model_path).exists():
            self.load(model_path)
        else:
            logger.info("No model loaded, creating new model")
            self._create_default_model()
    
    def _create_default_model(self):
        """Create default model for demonstration."""
        self.model = Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42
            ))
        ])
    
    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train the model.
        
        Args:
            X: Training features
            y: Training labels
        """
        logger.info(f"Training model with {len(X)} samples")
        self.model.fit(X, y)
        logger.info("Model training complete")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions.
        
        Args:
            X: Input features
            
        Returns:
            Predictions array
        """
        if self.model is None:
            raise ValueError("Model not trained or loaded")
        
        return self.model.predict(X)
    
    def save(self, path: str) -> None:
        """Save model to disk.
        
        Args:
            path: Path to save model
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, path)
        logger.info(f"Model saved to {path}")
    
    def load(self, path: str) -> None:
        """Load model from disk.
        
        Args:
            path: Path to model file
        """
        self.model = joblib.load(path)
        logger.info(f"Model loaded from {path}")


def create_dummy_data(n_samples: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
    """Create dummy data for demonstration.
    
    Args:
        n_samples: Number of samples
        
    Returns:
        Tuple of (X, y)
    """
    np.random.seed(42)
    X = np.random.randn(n_samples, 5)
    y = X[:, 0] * 2 + X[:, 1] * 3 - X[:, 2] + np.random.randn(n_samples) * 0.5
    return X, y
