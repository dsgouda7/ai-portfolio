"""Model training and registry for UnifiedAI

Provides: ModelRegistry for training dense and CNN neural networks with TensorFlow/Keras
"""

import logging
from pathlib import Path
from typing import Any, Dict, List

import joblib
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks

from src.utils import timer, validate_positive


logger = logging.getLogger("unifiedai")


class ModelRegistry:
    """Registry for training and managing multiple neural network models.
    
    Supported models:
    - Dense Neural Network (fully connected)
    - 1D CNN (convolutional for feature vectors)
    
    Attributes:
        models: Dictionary of trained models
        best_model_name: Name of best performing model
        histories: Training histories for each model
    
    Example:
        >>> registry = ModelRegistry()
        >>> registry.train_dense_nn(X_train, y_train, architecture=[128, 64, 32])
        >>> predictions = registry.predict(X_test, "dense_nn")
    """
    
    def __init__(self):
        """Initialize empty model registry."""
        self.models = {}
        self.best_model_name = None
        self.histories = {}
        
        # Set TensorFlow logging level
        tf.get_logger().setLevel('ERROR')
        
        logger.info("Initialized ModelRegistry")
    
    @timer
    def train_dense_nn(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        architecture: List[int] = [128, 64, 32],
        activation: str = 'relu',
        dropout: float = 0.3,
        learning_rate: float = 0.001,
        batch_size: int = 32,
        epochs: int = 50,
        early_stopping_patience: int = 5
    ) -> Dict[str, Any]:
        """Train dense neural network model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            architecture: List of hidden layer sizes
            activation: Activation function
            dropout: Dropout rate
            learning_rate: Learning rate for Adam optimizer
            batch_size: Batch size
            epochs: Maximum epochs
            early_stopping_patience: Early stopping patience
        
        Returns:
            Dictionary with training metrics
        
        Example:
            >>> metrics = registry.train_dense_nn(X_train, y_train, X_val, y_val)
        """
        validate_positive(learning_rate, "learning_rate")
        validate_positive(batch_size, "batch_size")
        
        logger.info(
            f"Training Dense NN (architecture={architecture}, "
            f"dropout={dropout}, lr={learning_rate})"
        )
        
        try:
            # Build model
            n_features = X_train.shape[1]
            n_classes = len(np.unique(y_train))
            
            model = keras.Sequential(name="dense_nn")
            
            # Input layer
            model.add(layers.Input(shape=(n_features,)))
            
            # Hidden layers
            for i, units in enumerate(architecture):
                model.add(layers.Dense(units, activation=activation, name=f"dense_{i+1}"))
                model.add(layers.Dropout(dropout, name=f"dropout_{i+1}"))
            
            # Output layer
            model.add(layers.Dense(n_classes, activation='softmax', name="output"))
            
            # Compile model
            model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
            
            logger.info(f"Model architecture: {n_features} → {' → '.join(map(str, architecture))} → {n_classes}")
            
            # Callbacks
            early_stop = callbacks.EarlyStopping(
                monitor='val_loss',
                patience=early_stopping_patience,
                restore_best_weights=True,
                verbose=0
            )
            
            # Train model
            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                batch_size=batch_size,
                epochs=epochs,
                callbacks=[early_stop],
                verbose=0
            )
            
            # Store model and history
            self.models["dense_nn"] = model
            self.histories["dense_nn"] = history.history
            
            # Training metrics
            train_loss = history.history['loss'][-1]
            train_acc = history.history['accuracy'][-1]
            val_loss = history.history['val_loss'][-1]
            val_acc = history.history['val_accuracy'][-1]
            
            metrics = {
                "train_loss": float(train_loss),
                "train_accuracy": float(train_acc),
                "val_loss": float(val_loss),
                "val_accuracy": float(val_acc),
                "epochs_trained": len(history.history['loss'])
            }
            
            logger.info(
                f"Dense NN trained in {metrics['epochs_trained']} epochs - "
                f"Val Acc: {val_acc:.3f}, Val Loss: {val_loss:.3f}"
            )
            
            return metrics
            
        except Exception as e:
            logger.error(f"Dense NN training failed: {e}")
            raise RuntimeError(f"Dense NN training failed: {e}") from e
    
    @timer
    def train_cnn_1d(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        filters: List[int] = [64, 32, 16],
        kernel_size: int = 3,
        pool_size: int = 2,
        dropout: float = 0.3,
        learning_rate: float = 0.001,
        batch_size: int = 32,
        epochs: int = 50,
        early_stopping_patience: int = 5
    ) -> Dict[str, Any]:
        """Train 1D CNN model (treats feature vector as sequence).
        
        Args:
            X_train, y_train, X_val, y_val: Training and validation data
            filters: List of filter counts per conv layer
            kernel_size: Convolution kernel size
            pool_size: MaxPooling pool size
            dropout: Dropout rate
            learning_rate: Learning rate
            batch_size: Batch size
            epochs: Maximum epochs
            early_stopping_patience: Early stopping patience
        
        Returns:
            Dictionary with training metrics
        """
        validate_positive(learning_rate, "learning_rate")
        validate_positive(batch_size, "batch_size")
        
        logger.info(f"Training 1D CNN (filters={filters}, kernel={kernel_size})")
        
        try:
            # Reshape for CNN: (samples, features, 1)
            X_train_reshaped = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
            X_val_reshaped = X_val.reshape(X_val.shape[0], X_val.shape[1], 1)
            
            n_features = X_train.shape[1]
            n_classes = len(np.unique(y_train))
            
            model = keras.Sequential(name="cnn_1d")
            
            # Input layer
            model.add(layers.Input(shape=(n_features, 1)))
            
            # Conv blocks
            for i, num_filters in enumerate(filters):
                model.add(layers.Conv1D(
                    num_filters,
                    kernel_size,
                    activation='relu',
                    padding='same',
                    name=f"conv_{i+1}"
                ))
                model.add(layers.MaxPooling1D(pool_size, name=f"pool_{i+1}"))
                model.add(layers.Dropout(dropout, name=f"dropout_{i+1}"))
            
            # Flatten and dense layers
            model.add(layers.GlobalAveragePooling1D(name="global_pool"))
            model.add(layers.Dense(64, activation='relu', name="dense"))
            model.add(layers.Dropout(dropout, name="dropout_final"))
            model.add(layers.Dense(n_classes, activation='softmax', name="output"))
            
            # Compile
            model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
            
            logger.info("1D CNN architecture built")
            
            # Callbacks
            early_stop = callbacks.EarlyStopping(
                monitor='val_loss',
                patience=early_stopping_patience,
                restore_best_weights=True,
                verbose=0
            )
            
            # Train
            history = model.fit(
                X_train_reshaped, y_train,
                validation_data=(X_val_reshaped, y_val),
                batch_size=batch_size,
                epochs=epochs,
                callbacks=[early_stop],
                verbose=0
            )
            
            # Store model and history
            self.models["cnn_1d"] = model
            self.histories["cnn_1d"] = history.history
            
            # Metrics
            train_loss = history.history['loss'][-1]
            train_acc = history.history['accuracy'][-1]
            val_loss = history.history['val_loss'][-1]
            val_acc = history.history['val_accuracy'][-1]
            
            metrics = {
                "train_loss": float(train_loss),
                "train_accuracy": float(train_acc),
                "val_loss": float(val_loss),
                "val_accuracy": float(val_acc),
                "epochs_trained": len(history.history['loss'])
            }
            
            logger.info(
                f"1D CNN trained in {metrics['epochs_trained']} epochs - "
                f"Val Acc: {val_acc:.3f}, Val Loss: {val_loss:.3f}"
            )
            
            return metrics
            
        except Exception as e:
            logger.error(f"1D CNN training failed: {e}")
            raise RuntimeError(f"1D CNN training failed: {e}") from e
    
    def predict(self, X: np.ndarray, model_name: str) -> np.ndarray:
        """Make predictions with specified model.
        
        Args:
            X: Features
            model_name: Name of model to use
        
        Returns:
            Predicted class labels
        
        Raises:
            ValueError: If model not found
        """
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found. Available: {list(self.models.keys())}")
        
        model = self.models[model_name]
        
        # Reshape for CNN if needed
        if model_name == "cnn_1d":
            X = X.reshape(X.shape[0], X.shape[1], 1)
        
        predictions = model.predict(X, verbose=0)
        return np.argmax(predictions, axis=1)
    
    def predict_proba(self, X: np.ndarray, model_name: str) -> np.ndarray:
        """Get prediction probabilities.
        
        Args:
            X: Features
            model_name: Name of model
        
        Returns:
            Class probabilities
        """
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found")
        
        model = self.models[model_name]
        
        if model_name == "cnn_1d":
            X = X.reshape(X.shape[0], X.shape[1], 1)
        
        return model.predict(X, verbose=0)
    
    def save_model(self, model_name: str, path: Path) -> None:
        """Save model to disk.
        
        Args:
            model_name: Name of model to save
            path: Save path
        """
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found")
        
        model = self.models[model_name]
        model.save(path)
        logger.info(f"Model '{model_name}' saved to {path}")
    
    def load_model(self, model_name: str, path: Path) -> None:
        """Load model from disk.
        
        Args:
            model_name: Name to assign to loaded model
            path: Path to model file
        """
        model = keras.models.load_model(path)
        self.models[model_name] = model
        logger.info(f"Model '{model_name}' loaded from {path}")
