"""Tests for ensemble model training"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path

from src.models import ModelRegistry


def test_model_registry_init():
    """Test ModelRegistry initialization."""
    registry = ModelRegistry()
    
    assert len(registry.base_models) == 0
    assert len(registry.ensemble_models) == 0
    assert registry.best_model_name is None


def test_train_base_models(sample_splits, temp_config):
    """Test training base models."""
    import yaml
    
    X_train, _, y_train, _ = sample_splits
    
    with open(temp_config) as f:
        config = yaml.safe_load(f)
    
    registry = ModelRegistry()
    metrics = registry.train_base_models(X_train, y_train, config, cv_folds=3)
    
    # Check that models were trained
    assert 'logistic_regression' in registry.base_models
    assert 'logistic_regression' in metrics
    
    # Check metrics structure
    assert 'cv_accuracy' in metrics['logistic_regression']
    assert 'cv_std' in metrics['logistic_regression']


@pytest.mark.slow
def test_train_multiple_base_models(sample_splits):
    """Test training multiple base models."""
    X_train, _, y_train, _ = sample_splits
    
    config = {
        'base_models': {
            'xgboost': {
                'n_estimators': 10,
                'max_depth': 3,
                'learning_rate': 0.1,
                'random_state': 42
            },
            'logistic_regression': {
                'C': 1.0,
                'max_iter': 100,
                'random_state': 42
            }
        }
    }
    
    registry = ModelRegistry()
    metrics = registry.train_base_models(X_train, y_train, config, cv_folds=2)
    
    # Check that both models were trained
    assert len(registry.base_models) == 2
    assert 'xgboost' in registry.base_models
    assert 'logistic_regression' in registry.base_models


@pytest.mark.slow
def test_train_voting_ensemble(sample_splits):
    """Test training voting ensemble."""
    X_train, _, y_train, _ = sample_splits
    
    config = {
        'base_models': {
            'logistic_regression': {
                'C': 1.0,
                'max_iter': 100,
                'random_state': 42
            }
        }
    }
    
    registry = ModelRegistry()
    registry.train_base_models(X_train, y_train, config, cv_folds=2)
    
    # Train voting ensemble
    metrics = registry.train_voting_ensemble(X_train, y_train, strategy='soft', cv_folds=2)
    
    # Check ensemble was created
    assert 'voting' in registry.ensemble_models
    assert 'cv_accuracy' in metrics
    assert 'n_estimators' in metrics


@pytest.mark.slow
def test_train_stacking_ensemble(sample_splits):
    """Test training stacking ensemble."""
    X_train, _, y_train, _ = sample_splits
    
    config = {
        'base_models': {
            'logistic_regression': {
                'C': 1.0,
                'max_iter': 100,
                'random_state': 42
            }
        }
    }
    
    registry = ModelRegistry()
    registry.train_base_models(X_train, y_train, config, cv_folds=2)
    
    # Train stacking ensemble
    metrics = registry.train_stacking_ensemble(
        X_train, y_train,
        meta_learner='logistic_regression',
        cv_folds=2
    )
    
    # Check ensemble was created
    assert 'stacking' in registry.ensemble_models
    assert 'cv_accuracy' in metrics
    assert 'meta_learner' in metrics


def test_predict_base_model(sample_splits):
    """Test prediction with base model."""
    X_train, X_test, y_train, _ = sample_splits
    
    config = {
        'base_models': {
            'logistic_regression': {
                'C': 1.0,
                'max_iter': 100,
                'random_state': 42
            }
        }
    }
    
    registry = ModelRegistry()
    registry.train_base_models(X_train, y_train, config, cv_folds=2)
    
    # Make predictions
    predictions = registry.predict(X_test, 'logistic_regression')
    
    # Check predictions
    assert len(predictions) == len(X_test)
    assert set(predictions).issubset({0, 1})


def test_predict_with_probabilities(sample_splits):
    """Test prediction with probabilities."""
    X_train, X_test, y_train, _ = sample_splits
    
    config = {
        'base_models': {
            'logistic_regression': {
                'C': 1.0,
                'max_iter': 100,
                'random_state': 42
            }
        }
    }
    
    registry = ModelRegistry()
    registry.train_base_models(X_train, y_train, config, cv_folds=2)
    
    # Get probabilities
    probabilities = registry.predict(X_test, 'logistic_regression', return_proba=True)
    
    # Check probabilities
    assert probabilities.shape == (len(X_test), 2)
    assert np.all((probabilities >= 0) & (probabilities <= 1))
    assert np.allclose(probabilities.sum(axis=1), 1.0)


def test_predict_all_base_models(sample_splits):
    """Test getting predictions from all base models."""
    X_train, X_test, y_train, _ = sample_splits
    
    config = {
        'base_models': {
            'logistic_regression': {
                'C': 1.0,
                'max_iter': 100,
                'random_state': 42
            }
        }
    }
    
    registry = ModelRegistry()
    registry.train_base_models(X_train, y_train, config, cv_folds=2)
    
    # Get all predictions
    all_predictions = registry.predict_all_base_models(X_test)
    
    # Check predictions
    assert 'logistic_regression' in all_predictions
    assert len(all_predictions['logistic_regression']) == len(X_test)


def test_predict_nonexistent_model(sample_splits):
    """Test that predicting with nonexistent model raises error."""
    X_train, X_test, y_train, _ = sample_splits
    
    registry = ModelRegistry()
    
    with pytest.raises(ValueError, match="Model .* not found"):
        registry.predict(X_test, 'nonexistent_model')


def test_voting_without_base_models(sample_splits):
    """Test that voting ensemble requires base models."""
    X_train, _, y_train, _ = sample_splits
    
    registry = ModelRegistry()
    
    with pytest.raises(RuntimeError, match="Must train base models first"):
        registry.train_voting_ensemble(X_train, y_train)


def test_stacking_without_base_models(sample_splits):
    """Test that stacking ensemble requires base models."""
    X_train, _, y_train, _ = sample_splits
    
    registry = ModelRegistry()
    
    with pytest.raises(RuntimeError, match="Must train base models first"):
        registry.train_stacking_ensemble(X_train, y_train)


@pytest.mark.slow
def test_save_and_load_models(sample_splits, tmp_path):
    """Test saving and loading models."""
    X_train, _, y_train, _ = sample_splits
    
    config = {
        'base_models': {
            'logistic_regression': {
                'C': 1.0,
                'max_iter': 100,
                'random_state': 42
            }
        }
    }
    
    # Train and save
    registry = ModelRegistry()
    registry.train_base_models(X_train, y_train, config, cv_folds=2)
    registry.save(tmp_path)
    
    # Check file exists
    assert (tmp_path / "logistic_regression.pkl").exists()
    
    # Load model
    loaded_model = registry.load(tmp_path, "logistic_regression")
    assert loaded_model is not None


def test_cv_scores_tracking(sample_splits):
    """Test that CV scores are tracked correctly."""
    X_train, _, y_train, _ = sample_splits
    
    config = {
        'base_models': {
            'logistic_regression': {
                'C': 1.0,
                'max_iter': 100,
                'random_state': 42
            }
        }
    }
    
    registry = ModelRegistry()
    registry.train_base_models(X_train, y_train, config, cv_folds=3)
    
    # Check CV scores tracked
    assert 'logistic_regression' in registry.cv_scores
    assert 0.0 <= registry.cv_scores['logistic_regression'] <= 1.0
