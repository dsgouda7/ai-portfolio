"""Model training and registry for FraudShield

Provides: ModelRegistry for training IsolationForest, OneClassSVM, Autoencoder
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from src.utils import timer, validate_positive


logger = logging.getLogger("fraudshield")


class ModelRegistry:
    """Registry for training and managing multiple anomaly detection models.
    
    Supported models:
    - Isolation Forest (tree-based outlier detection)
    - One-Class SVM (density-based outlier detection)
    - Autoencoder (neural network reconstruction error)
    
    Attributes:
        models: Dictionary of trained models
        best_model_name: Name of best performing model
        validation_scores: Validation scores for each model
    
    Example:
        >>> registry = ModelRegistry()
        >>> registry.train_isolation_forest(X_train, y_train, contamination=0.1)
        >>> registry.train_one_class_svm(X_train, y_train, nu=0.1)
        >>> predictions = registry.predict(X_test, "isolation_forest")
    """
    
    def __init__(self):
        """Initialize empty model registry."""
        self.models = {}
        self.best_model_name = None
        self.validation_scores = {}
        
        logger.info("Initialized ModelRegistry")
    
    @timer
    def train_isolation_forest(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        contamination: float = 0.1,
        n_estimators: int = 100,
        max_samples: str = 'auto',
        random_state: int = 42
    ) -> Dict[str, float]:
        """Train Isolation Forest model.
        
        Isolation Forest builds an ensemble of trees and identifies anomalies
        as points that require fewer splits to isolate.
        
        Args:
            X: Training features
            y: Training labels (0=normal, 1=anomaly)
            contamination: Expected proportion of outliers
            n_estimators: Number of trees in the forest
            max_samples: Number of samples to draw for each tree
            random_state: Random seed
        
        Returns:
            Dictionary with training metrics (precision, recall, f1, accuracy)
        
        Example:
            >>> metrics = registry.train_isolation_forest(X_train, y_train)
            >>> print(f"F1 Score: {metrics['f1']:.3f}")
        """
        validate_positive(contamination, "contamination")
        validate_positive(n_estimators, "n_estimators")
        
        logger.info(
            f"Training Isolation Forest (contamination={contamination}, "
            f"n_estimators={n_estimators})"
        )
        
        try:
            model = IsolationForest(
                contamination=contamination,
                n_estimators=n_estimators,
                max_samples=max_samples,
                random_state=random_state,
                n_jobs=-1
            )
            
            # Fit model (unsupervised, but we have labels for validation)
            model.fit(X)
            
            # Predict: -1 for anomalies, 1 for normal
            # Convert to 0/1 for consistency with labels
            y_pred_raw = model.predict(X)
            y_pred = (y_pred_raw == -1).astype(int)
            
            # Store model
            self.models["isolation_forest"] = model
            
            # Compute metrics
            metrics = self._compute_metrics(y, y_pred)
            self.validation_scores["isolation_forest"] = metrics["f1"]
            
            logger.info(
                f"Isolation Forest trained - Precision: {metrics['precision']:.3f}, "
                f"Recall: {metrics['recall']:.3f}, F1: {metrics['f1']:.3f}"
            )
            
            return metrics
            
        except Exception as e:
            logger.error(f"Isolation Forest training failed: {e}")
            raise RuntimeError(f"Isolation Forest training failed: {e}") from e
    
    @timer
    def train_one_class_svm(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        kernel: str = 'rbf',
        gamma: str = 'scale',
        nu: float = 0.1
    ) -> Dict[str, float]:
        """Train One-Class SVM model.
        
        One-Class SVM learns a decision boundary around normal data points
        using kernel methods.
        
        Args:
            X: Training features
            y: Training labels (0=normal, 1=anomaly)
            kernel: Kernel type ('linear', 'poly', 'rbf', 'sigmoid')
            gamma: Kernel coefficient
            nu: Upper bound on fraction of outliers (should match contamination)
        
        Returns:
            Dictionary with training metrics
        
        Example:
            >>> metrics = registry.train_one_class_svm(X_train, y_train, nu=0.1)
        """
        validate_positive(nu, "nu")
        
        if nu > 1.0:
            raise ValueError(f"nu must be <= 1.0, got {nu}")
        
        logger.info(f"Training One-Class SVM (kernel={kernel}, nu={nu})")
        
        try:
            model = OneClassSVM(
                kernel=kernel,
                gamma=gamma,
                nu=nu
            )
            
            # Fit model
            model.fit(X)
            
            # Predict: -1 for anomalies, 1 for normal
            y_pred_raw = model.predict(X)
            y_pred = (y_pred_raw == -1).astype(int)
            
            # Store model
            self.models["one_class_svm"] = model
            
            # Compute metrics
            metrics = self._compute_metrics(y, y_pred)
            self.validation_scores["one_class_svm"] = metrics["f1"]
            
            logger.info(
                f"One-Class SVM trained - Precision: {metrics['precision']:.3f}, "
                f"Recall: {metrics['recall']:.3f}, F1: {metrics['f1']:.3f}"
            )
            
            return metrics
            
        except Exception as e:
            logger.error(f"One-Class SVM training failed: {e}")
            raise RuntimeError(f"One-Class SVM training failed: {e}") from e
    
    @timer
    def train_autoencoder(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        encoding_dim: int = 10,
        epochs: int = 50,
        batch_size: int = 32,
        learning_rate: float = 0.001,
        contamination_threshold: float = 0.1
    ) -> Dict[str, float]:
        """Train Autoencoder model for anomaly detection.
        
        Autoencoder learns to reconstruct normal data. Anomalies have
        higher reconstruction errors.
        
        Args:
            X: Training features
            y: Training labels (0=normal, 1=anomaly)
            encoding_dim: Dimension of encoded representation
            epochs: Number of training epochs
            batch_size: Batch size for training
            learning_rate: Learning rate for optimizer
            contamination_threshold: Percentile for reconstruction error threshold
        
        Returns:
            Dictionary with training metrics
        
        Example:
            >>> metrics = registry.train_autoencoder(X_train, y_train)
        """
        validate_positive(encoding_dim, "encoding_dim")
        validate_positive(epochs, "epochs")
        validate_positive(batch_size, "batch_size")
        validate_positive(learning_rate, "learning_rate")
        
        logger.info(
            f"Training Autoencoder (encoding_dim={encoding_dim}, epochs={epochs})"
        )
        
        try:
            import tensorflow as tf
            from tensorflow import keras
            
            # Suppress TF warnings
            tf.get_logger().setLevel('ERROR')
            
            input_dim = X.shape[1]
            
            # Build autoencoder
            encoder = keras.Sequential([
                keras.layers.Dense(encoding_dim * 2, activation='relu', input_shape=(input_dim,)),
                keras.layers.Dense(encoding_dim, activation='relu')
            ])
            
            decoder = keras.Sequential([
                keras.layers.Dense(encoding_dim * 2, activation='relu', input_shape=(encoding_dim,)),
                keras.layers.Dense(input_dim, activation='linear')
            ])
            
            autoencoder = keras.Sequential([encoder, decoder])
            
            autoencoder.compile(
                optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                loss='mse'
            )
            
            # Train only on normal samples for better anomaly detection
            X_normal = X[y == 0]
            
            history = autoencoder.fit(
                X_normal.values,
                X_normal.values,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=0.1,
                verbose=0
            )
            
            # Compute reconstruction errors on all training data
            X_reconstructed = autoencoder.predict(X.values, verbose=0)
            reconstruction_errors = np.mean(np.square(X.values - X_reconstructed), axis=1)
            
            # Set threshold at specified percentile
            threshold = np.percentile(reconstruction_errors, (1 - contamination_threshold) * 100)
            
            # Predict: 1 if reconstruction error > threshold
            y_pred = (reconstruction_errors > threshold).astype(int)
            
            # Store model and threshold
            self.models["autoencoder"] = {
                "model": autoencoder,
                "threshold": threshold
            }
            
            # Compute metrics
            metrics = self._compute_metrics(y, y_pred)
            self.validation_scores["autoencoder"] = metrics["f1"]
            
            final_loss = history.history['loss'][-1]
            logger.info(
                f"Autoencoder trained - Loss: {final_loss:.4f}, "
                f"Threshold: {threshold:.4f}, F1: {metrics['f1']:.3f}"
            )
            
            return metrics
            
        except ImportError:
            logger.error("TensorFlow not installed. Install with: pip install tensorflow")
            raise RuntimeError("TensorFlow required for autoencoder")
        except Exception as e:
            logger.error(f"Autoencoder training failed: {e}")
            raise RuntimeError(f"Autoencoder training failed: {e}") from e
    
    def predict(self, X: pd.DataFrame, model_name: str) -> np.ndarray:
        """Make predictions using specified model.
        
        Args:
            X: Features to predict on
            model_name: Name of model to use
        
        Returns:
            Binary predictions (0=normal, 1=anomaly)
        
        Raises:
            ValueError: If model not found
        
        Example:
            >>> predictions = registry.predict(X_test, "isolation_forest")
        """
        if model_name not in self.models:
            raise ValueError(
                f"Model '{model_name}' not found. Available: {list(self.models.keys())}"
            )
        
        model = self.models[model_name]
        
        if model_name == "autoencoder":
            # Special handling for autoencoder
            autoencoder = model["model"]
            threshold = model["threshold"]
            
            X_reconstructed = autoencoder.predict(X.values, verbose=0)
            reconstruction_errors = np.mean(np.square(X.values - X_reconstructed), axis=1)
            predictions = (reconstruction_errors > threshold).astype(int)
        else:
            # IsolationForest and OneClassSVM return -1/1
            predictions_raw = model.predict(X)
            predictions = (predictions_raw == -1).astype(int)
        
        return predictions
    
    def predict_scores(self, X: pd.DataFrame, model_name: str) -> np.ndarray:
        """Get anomaly scores (higher = more anomalous).
        
        Args:
            X: Features to score
            model_name: Name of model to use
        
        Returns:
            Anomaly scores
        
        Example:
            >>> scores = registry.predict_scores(X_test, "isolation_forest")
        """
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found")
        
        model = self.models[model_name]
        
        if model_name == "autoencoder":
            autoencoder = model["model"]
            X_reconstructed = autoencoder.predict(X.values, verbose=0)
            scores = np.mean(np.square(X.values - X_reconstructed), axis=1)
        else:
            # IsolationForest and OneClassSVM have decision_function
            # More negative = more anomalous, so negate
            scores = -model.decision_function(X)
        
        return scores
    
    def save_model(self, model_name: str, path: str) -> None:
        """Save trained model to disk.
        
        Args:
            model_name: Name of model to save
            path: File path to save to
        
        Example:
            >>> registry.save_model("isolation_forest", "models/best_model.pkl")
        """
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found")
        
        model_path = Path(path)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        
        joblib.dump(self.models[model_name], model_path)
        logger.info(f"Model '{model_name}' saved to {model_path}")
    
    def load_model(self, model_name: str, path: str) -> None:
        """Load trained model from disk.
        
        Args:
            model_name: Name to assign to loaded model
            path: File path to load from
        
        Example:
            >>> registry.load_model("isolation_forest", "models/best_model.pkl")
        """
        model_path = Path(path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        self.models[model_name] = joblib.load(model_path)
        logger.info(f"Model '{model_name}' loaded from {model_path}")
    
    def _compute_metrics(self, y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
        """Compute binary classification metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
        
        Returns:
            Dictionary with precision, recall, f1, accuracy
        """
        return {
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
            "accuracy": float(accuracy_score(y_true, y_pred)),
        }
    
    def select_best_model(self) -> str:
        """Select best model based on validation F1 score.
        
        Returns:
            Name of best model
        
        Example:
            >>> best = registry.select_best_model()
            >>> print(f"Best model: {best}")
        """
        if not self.validation_scores:
            raise RuntimeError("No models trained yet")
        
        self.best_model_name = max(self.validation_scores, key=self.validation_scores.get)
        logger.info(
            f"Best model: {self.best_model_name} "
            f"(F1: {self.validation_scores[self.best_model_name]:.3f})"
        )
        
        return self.best_model_name
