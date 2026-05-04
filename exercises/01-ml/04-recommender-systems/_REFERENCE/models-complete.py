"""Model training and registry for FlixAI

Provides: ModelRegistry for training Matrix Factorization and Neural CF
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import numpy as np
import pandas as pd
from scipy.sparse.linalg import svds

from src.utils import timer, validate_positive


logger = logging.getLogger("flixai")


class ModelRegistry:
    """Registry for training and managing recommender models.
    
    Supported models:
    - Matrix Factorization (SVD)
    - Neural Collaborative Filtering (TensorFlow)
    - Item-based Collaborative Filtering
    
    Attributes:
        models: Dictionary of trained models
        best_model_name: Name of best performing model
        user_factors: User latent factors (matrix factorization)
        item_factors: Item latent factors (matrix factorization)
    
    Example:
        >>> registry = ModelRegistry()
        >>> registry.train_matrix_factorization(ratings_train, n_factors=50)
        >>> predictions = registry.predict(user_id=123, k=10)
    """
    
    def __init__(self):
        """Initialize empty model registry."""
        self.models = {}
        self.best_model_name = None
        self.user_factors = None
        self.item_factors = None
        self.user_bias = None
        self.item_bias = None
        self.global_mean = None
        
        logger.info("Initialized ModelRegistry")
    
    @timer
    def train_matrix_factorization(
        self,
        ratings: pd.DataFrame,
        user_item_matrix: pd.DataFrame,
        n_factors: int = 50,
        regularization: float = 0.1
    ) -> Dict[str, float]:
        """Train matrix factorization model using SVD.
        
        Decompose user-item matrix into user and item latent factors:
        R ≈ U @ Σ @ V^T
        
        Args:
            ratings: DataFrame with user_id, item_id, rating columns
            user_item_matrix: User-item matrix (users x items)
            n_factors: Number of latent factors
            regularization: Regularization parameter (not used in SVD, for compatibility)
        
        Returns:
            Dictionary with training metrics
        
        Raises:
            ValueError: If n_factors is not positive
            RuntimeError: If training fails
        
        Example:
            >>> metrics = registry.train_matrix_factorization(
            ...     train_ratings, user_item_train, n_factors=50
            ... )
            >>> print(f"Train RMSE: {metrics['rmse']:.3f}")
        """
        validate_positive(n_factors, "n_factors")
        
        logger.info(f"Training Matrix Factorization (n_factors={n_factors})")
        
        try:
            # Compute global mean and center the matrix
            self.global_mean = user_item_matrix.values[user_item_matrix.values > 0].mean()
            
            # Center ratings
            centered_matrix = user_item_matrix.values.copy()
            centered_matrix[centered_matrix > 0] -= self.global_mean
            
            # Apply SVD
            logger.info("Computing SVD decomposition...")
            U, sigma, Vt = svds(centered_matrix, k=n_factors)
            
            # Store factors
            self.user_factors = U
            self.item_factors = Vt.T
            sigma_diag = np.diag(sigma)
            
            # Store full model
            self.models["matrix_factorization"] = {
                'user_factors': self.user_factors,
                'item_factors': self.item_factors,
                'sigma': sigma_diag,
                'global_mean': self.global_mean,
                'n_factors': n_factors
            }
            
            # Compute training metrics
            predictions = self._predict_all(user_item_matrix)
            actual = user_item_matrix.values[user_item_matrix.values > 0]
            predicted = predictions[user_item_matrix.values > 0]
            
            rmse = np.sqrt(np.mean((actual - predicted) ** 2))
            mae = np.mean(np.abs(actual - predicted))
            
            metrics = {
                "rmse": float(rmse),
                "mae": float(mae),
                "n_factors": n_factors
            }
            
            logger.info(f"Matrix Factorization trained - RMSE: {rmse:.3f}, MAE: {mae:.3f}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Matrix Factorization training failed: {e}")
            raise RuntimeError(f"Matrix Factorization training failed: {e}") from e
    
    def _predict_all(self, user_item_matrix: pd.DataFrame) -> np.ndarray:
        """Predict all ratings using matrix factorization.
        
        Args:
            user_item_matrix: User-item matrix
        
        Returns:
            Predicted ratings matrix
        """
        model = self.models.get("matrix_factorization")
        if model is None:
            raise RuntimeError("Matrix factorization model not trained")
        
        predictions = (
            self.user_factors @ model['sigma'] @ self.item_factors.T
            + model['global_mean']
        )
        
        return predictions
    
    @timer
    def train_neural_cf(
        self,
        ratings: pd.DataFrame,
        embedding_dim: int = 50,
        hidden_layers: list = [128, 64, 32],
        dropout_rate: float = 0.2,
        learning_rate: float = 0.001,
        batch_size: int = 256,
        n_epochs: int = 10
    ) -> Dict[str, float]:
        """Train neural collaborative filtering model.
        
        Uses TensorFlow to learn user and item embeddings through
        multi-layer perceptron.
        
        Args:
            ratings: DataFrame with user_id, item_id, rating columns
            embedding_dim: Dimension of user/item embeddings
            hidden_layers: List of hidden layer sizes
            dropout_rate: Dropout rate for regularization
            learning_rate: Learning rate for optimizer
            batch_size: Batch size for training
            n_epochs: Number of training epochs
        
        Returns:
            Dictionary with training metrics
        
        Example:
            >>> metrics = registry.train_neural_cf(
            ...     train_ratings,
            ...     embedding_dim=50,
            ...     n_epochs=10
            ... )
        """
        try:
            import tensorflow as tf
        except ImportError:
            logger.error("TensorFlow not installed. Install with: pip install tensorflow")
            raise RuntimeError("TensorFlow required for neural CF")
        
        logger.info(
            f"Training Neural CF (embedding_dim={embedding_dim}, "
            f"layers={hidden_layers}, epochs={n_epochs})"
        )
        
        try:
            # Prepare data
            n_users = ratings['user_id'].max() + 1
            n_items = ratings['item_id'].max() + 1
            
            user_ids = ratings['user_id'].values
            item_ids = ratings['item_id'].values
            ratings_values = ratings['rating'].values
            
            # Build model
            user_input = tf.keras.Input(shape=(1,), name='user_input')
            item_input = tf.keras.Input(shape=(1,), name='item_input')
            
            # Embeddings
            user_embedding = tf.keras.layers.Embedding(
                n_users, embedding_dim, name='user_embedding'
            )(user_input)
            item_embedding = tf.keras.layers.Embedding(
                n_items, embedding_dim, name='item_embedding'
            )(item_input)
            
            # Flatten
            user_vec = tf.keras.layers.Flatten()(user_embedding)
            item_vec = tf.keras.layers.Flatten()(item_embedding)
            
            # Concatenate
            concat = tf.keras.layers.Concatenate()([user_vec, item_vec])
            
            # Hidden layers
            x = concat
            for units in hidden_layers:
                x = tf.keras.layers.Dense(units, activation='relu')(x)
                x = tf.keras.layers.Dropout(dropout_rate)(x)
            
            # Output
            output = tf.keras.layers.Dense(1, activation='linear', name='output')(x)
            
            # Compile model
            model = tf.keras.Model(inputs=[user_input, item_input], outputs=output)
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate),
                loss='mse',
                metrics=['mae']
            )
            
            logger.info(f"Model architecture: {model.count_params()} parameters")
            
            # Train model
            history = model.fit(
                [user_ids, item_ids],
                ratings_values,
                batch_size=batch_size,
                epochs=n_epochs,
                validation_split=0.1,
                verbose=0
            )
            
            # Store model
            self.models["neural_cf"] = model
            
            # Get final metrics
            final_loss = history.history['loss'][-1]
            final_mae = history.history['mae'][-1]
            
            metrics = {
                "rmse": float(np.sqrt(final_loss)),
                "mae": float(final_mae),
                "n_epochs": n_epochs,
                "n_params": model.count_params()
            }
            
            logger.info(f"Neural CF trained - RMSE: {metrics['rmse']:.3f}, MAE: {metrics['mae']:.3f}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Neural CF training failed: {e}")
            raise RuntimeError(f"Neural CF training failed: {e}") from e
    
    def predict_rating(
        self,
        user_id: int,
        item_id: int,
        model_name: str = "matrix_factorization"
    ) -> float:
        """Predict rating for a user-item pair.
        
        Args:
            user_id: User ID
            item_id: Item ID
            model_name: Name of model to use
        
        Returns:
            Predicted rating
        
        Example:
            >>> rating = registry.predict_rating(user_id=123, item_id=456)
        """
        if model_name == "matrix_factorization":
            if self.user_factors is None:
                raise RuntimeError("Matrix factorization model not trained")
            
            prediction = (
                np.dot(self.user_factors[user_id], self.item_factors[item_id])
                + self.global_mean
            )
            return float(np.clip(prediction, 1, 5))
        
        elif model_name == "neural_cf":
            model = self.models.get("neural_cf")
            if model is None:
                raise RuntimeError("Neural CF model not trained")
            
            prediction = model.predict(
                [np.array([user_id]), np.array([item_id])],
                verbose=0
            )[0][0]
            return float(np.clip(prediction, 1, 5))
        
        else:
            raise ValueError(f"Unknown model: {model_name}")
    
    def recommend_items(
        self,
        user_id: int,
        k: int = 10,
        exclude_seen: bool = True,
        seen_items: Optional[set] = None,
        model_name: str = "matrix_factorization"
    ) -> list:
        """Recommend top-k items for a user.
        
        Args:
            user_id: User ID
            k: Number of recommendations
            exclude_seen: Whether to exclude already-seen items
            seen_items: Set of item IDs already seen by user
            model_name: Name of model to use
        
        Returns:
            List of (item_id, predicted_rating) tuples
        
        Example:
            >>> recommendations = registry.recommend_items(user_id=123, k=10)
            >>> print(f"Top item: {recommendations[0][0]} (score: {recommendations[0][1]:.2f})")
        """
        if model_name == "matrix_factorization":
            if self.user_factors is None:
                raise RuntimeError("Matrix factorization model not trained")
            
            # Compute scores for all items
            user_vector = self.user_factors[user_id]
            scores = self.item_factors @ user_vector + self.global_mean
            
            # Exclude seen items
            if exclude_seen and seen_items:
                for item_id in seen_items:
                    scores[item_id] = -np.inf
            
            # Get top-k
            top_k_indices = np.argsort(scores)[::-1][:k]
            recommendations = [(int(idx), float(scores[idx])) for idx in top_k_indices]
            
            return recommendations
        
        else:
            raise ValueError(f"Recommendation not implemented for model: {model_name}")
    
    def save(self, model_name: str, path: str) -> None:
        """Save trained model to disk.
        
        Args:
            model_name: Name of model to save
            path: Path to save model
        
        Example:
            >>> registry.save("matrix_factorization", "models/mf_model.pkl")
        """
        model_path = Path(path)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        
        model = self.models.get(model_name)
        if model is None:
            raise ValueError(f"Model {model_name} not found in registry")
        
        joblib.dump(model, model_path)
        logger.info(f"Model '{model_name}' saved to {model_path}")
    
    def load(self, model_name: str, path: str) -> None:
        """Load trained model from disk.
        
        Args:
            model_name: Name to assign to loaded model
            path: Path to model file
        
        Example:
            >>> registry.load("matrix_factorization", "models/mf_model.pkl")
        """
        model_path = Path(path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        self.models[model_name] = joblib.load(model_path)
        logger.info(f"Model '{model_name}' loaded from {model_path}")
