"""Model training and registry for EnsembleAI

Provides: ModelRegistry for training ensemble methods (voting, stacking, boosting)
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional, List

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

from src.utils import timer, validate_positive


logger = logging.getLogger("ensembleai")


class ModelRegistry:
    """Registry for training and managing ensemble classification models.
    
    Supported base models:
    - XGBoost (gradient boosting)
    - LightGBM (gradient boosting)
    - CatBoost (gradient boosting)
    - Random Forest
    - Logistic Regression
    
    Supported ensemble methods:
    - Voting (soft/hard voting)
    - Stacking (meta-learner)
    - Boosting cascade (sequential)
    
    Attributes:
        base_models: Dictionary of trained base models
        ensemble_models: Dictionary of trained ensemble models
        best_model_name: Name of best performing model
        cv_scores: Cross-validation scores for each model
    
    Example:
        >>> registry = ModelRegistry()
        >>> registry.train_base_models(X_train, y_train)
        >>> registry.train_voting_ensemble(X_train, y_train)
        >>> predictions = registry.predict(X_test, "voting")
    """
    
    def __init__(self):
        """Initialize empty model registry."""
        self.base_models = {}
        self.ensemble_models = {}
        self.best_model_name = None
        self.cv_scores = {}
        
        logger.info("Initialized ModelRegistry for EnsembleAI")
    
    @timer
    def train_base_models(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        config: Dict[str, Any],
        cv_folds: int = 5
    ) -> Dict[str, Dict[str, float]]:
        """Train all base models configured in config.
        
        Args:
            X: Training features
            y: Training labels
            config: Configuration dict with base_models section
            cv_folds: Number of cross-validation folds
        
        Returns:
            Dictionary of metrics for each base model
        
        Example:
            >>> metrics = registry.train_base_models(X_train, y_train, config)
        """
        logger.info("Training base models")
        all_metrics = {}
        
        # XGBoost
        if 'xgboost' in config.get('base_models', {}):
            xgb_params = config['base_models']['xgboost']
            metrics = self._train_xgboost(X, y, xgb_params, cv_folds)
            all_metrics['xgboost'] = metrics
        
        # LightGBM
        if 'lightgbm' in config.get('base_models', {}):
            lgbm_params = config['base_models']['lightgbm']
            metrics = self._train_lightgbm(X, y, lgbm_params, cv_folds)
            all_metrics['lightgbm'] = metrics
        
        # CatBoost
        if 'catboost' in config.get('base_models', {}):
            cb_params = config['base_models']['catboost']
            metrics = self._train_catboost(X, y, cb_params, cv_folds)
            all_metrics['catboost'] = metrics
        
        # Random Forest
        if 'random_forest' in config.get('base_models', {}):
            rf_params = config['base_models']['random_forest']
            metrics = self._train_random_forest(X, y, rf_params, cv_folds)
            all_metrics['random_forest'] = metrics
        
        # Logistic Regression
        if 'logistic_regression' in config.get('base_models', {}):
            lr_params = config['base_models']['logistic_regression']
            metrics = self._train_logistic_regression(X, y, lr_params, cv_folds)
            all_metrics['logistic_regression'] = metrics
        
        logger.info(f"Base models trained: {list(all_metrics.keys())}")
        return all_metrics
    
    def _train_xgboost(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        params: Dict[str, Any],
        cv_folds: int
    ) -> Dict[str, float]:
        """Train XGBoost classifier."""
        logger.info("Training XGBoost")
        
        model = XGBClassifier(**params)
        model.fit(X, y)
        
        # Cross-validation
        cv_scores = cross_val_score(model, X, y, cv=cv_folds, scoring='accuracy')
        cv_acc = cv_scores.mean()
        
        self.base_models['xgboost'] = model
        self.cv_scores['xgboost'] = cv_acc
        
        logger.info(f"XGBoost trained - CV Accuracy: {cv_acc:.4f}")
        return {'cv_accuracy': cv_acc, 'cv_std': cv_scores.std()}
    
    def _train_lightgbm(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        params: Dict[str, Any],
        cv_folds: int
    ) -> Dict[str, float]:
        """Train LightGBM classifier."""
        logger.info("Training LightGBM")
        
        model = LGBMClassifier(**params)
        model.fit(X, y)
        
        cv_scores = cross_val_score(model, X, y, cv=cv_folds, scoring='accuracy')
        cv_acc = cv_scores.mean()
        
        self.base_models['lightgbm'] = model
        self.cv_scores['lightgbm'] = cv_acc
        
        logger.info(f"LightGBM trained - CV Accuracy: {cv_acc:.4f}")
        return {'cv_accuracy': cv_acc, 'cv_std': cv_scores.std()}
    
    def _train_catboost(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        params: Dict[str, Any],
        cv_folds: int
    ) -> Dict[str, float]:
        """Train CatBoost classifier."""
        logger.info("Training CatBoost")
        
        model = CatBoostClassifier(**params)
        model.fit(X, y)
        
        cv_scores = cross_val_score(model, X, y, cv=cv_folds, scoring='accuracy')
        cv_acc = cv_scores.mean()
        
        self.base_models['catboost'] = model
        self.cv_scores['catboost'] = cv_acc
        
        logger.info(f"CatBoost trained - CV Accuracy: {cv_acc:.4f}")
        return {'cv_accuracy': cv_acc, 'cv_std': cv_scores.std()}
    
    def _train_random_forest(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        params: Dict[str, Any],
        cv_folds: int
    ) -> Dict[str, float]:
        """Train Random Forest classifier."""
        logger.info("Training Random Forest")
        
        model = RandomForestClassifier(**params)
        model.fit(X, y)
        
        cv_scores = cross_val_score(model, X, y, cv=cv_folds, scoring='accuracy')
        cv_acc = cv_scores.mean()
        
        self.base_models['random_forest'] = model
        self.cv_scores['random_forest'] = cv_acc
        
        logger.info(f"Random Forest trained - CV Accuracy: {cv_acc:.4f}")
        return {'cv_accuracy': cv_acc, 'cv_std': cv_scores.std()}
    
    def _train_logistic_regression(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        params: Dict[str, Any],
        cv_folds: int
    ) -> Dict[str, float]:
        """Train Logistic Regression classifier."""
        logger.info("Training Logistic Regression")
        
        model = LogisticRegression(**params)
        model.fit(X, y)
        
        cv_scores = cross_val_score(model, X, y, cv=cv_folds, scoring='accuracy')
        cv_acc = cv_scores.mean()
        
        self.base_models['logistic_regression'] = model
        self.cv_scores['logistic_regression'] = cv_acc
        
        logger.info(f"Logistic Regression trained - CV Accuracy: {cv_acc:.4f}")
        return {'cv_accuracy': cv_acc, 'cv_std': cv_scores.std()}
    
    @timer
    def train_voting_ensemble(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        strategy: str = 'soft',
        weights: Optional[List[float]] = None,
        cv_folds: int = 5
    ) -> Dict[str, float]:
        """Train voting ensemble from base models.
        
        Args:
            X: Training features
            y: Training labels
            strategy: 'soft' (probability voting) or 'hard' (majority voting)
            weights: Optional weights for each base model (None = equal weights)
            cv_folds: Number of cross-validation folds
        
        Returns:
            Dictionary with training metrics
        
        Example:
            >>> metrics = registry.train_voting_ensemble(X_train, y_train, strategy='soft')
        """
        if not self.base_models:
            raise RuntimeError("Must train base models first. Call train_base_models().")
        
        logger.info(f"Training Voting Ensemble ({strategy} voting)")
        
        # Create estimators list
        estimators = [(name, model) for name, model in self.base_models.items()]
        
        # Train voting classifier
        voting_clf = VotingClassifier(
            estimators=estimators,
            voting=strategy,
            weights=weights
        )
        voting_clf.fit(X, y)
        
        # Cross-validation
        cv_scores = cross_val_score(voting_clf, X, y, cv=cv_folds, scoring='accuracy')
        cv_acc = cv_scores.mean()
        
        self.ensemble_models['voting'] = voting_clf
        self.cv_scores['voting'] = cv_acc
        
        logger.info(f"Voting Ensemble trained - CV Accuracy: {cv_acc:.4f}")
        logger.info(f"Base models used: {list(self.base_models.keys())}")
        
        return {'cv_accuracy': cv_acc, 'cv_std': cv_scores.std(), 'n_estimators': len(estimators)}
    
    @timer
    def train_stacking_ensemble(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        meta_learner: str = 'logistic_regression',
        cv_folds: int = 5
    ) -> Dict[str, float]:
        """Train stacking ensemble with meta-learner.
        
        Args:
            X: Training features
            y: Training labels
            meta_learner: Meta-learner type ('logistic_regression', 'random_forest', 'xgboost')
            cv_folds: Number of cross-validation folds for stacking
        
        Returns:
            Dictionary with training metrics
        
        Example:
            >>> metrics = registry.train_stacking_ensemble(X_train, y_train)
        """
        if not self.base_models:
            raise RuntimeError("Must train base models first. Call train_base_models().")
        
        logger.info(f"Training Stacking Ensemble (meta-learner={meta_learner})")
        
        # Create estimators list
        estimators = [(name, model) for name, model in self.base_models.items()]
        
        # Create meta-learner
        if meta_learner == 'logistic_regression':
            final_estimator = LogisticRegression(random_state=42)
        elif meta_learner == 'random_forest':
            final_estimator = RandomForestClassifier(n_estimators=50, random_state=42)
        elif meta_learner == 'xgboost':
            final_estimator = XGBClassifier(n_estimators=50, random_state=42)
        else:
            raise ValueError(f"Unknown meta-learner: {meta_learner}")
        
        # Train stacking classifier
        stacking_clf = StackingClassifier(
            estimators=estimators,
            final_estimator=final_estimator,
            cv=cv_folds
        )
        stacking_clf.fit(X, y)
        
        # Evaluate
        cv_scores = cross_val_score(stacking_clf, X, y, cv=cv_folds, scoring='accuracy')
        cv_acc = cv_scores.mean()
        
        self.ensemble_models['stacking'] = stacking_clf
        self.cv_scores['stacking'] = cv_acc
        
        logger.info(f"Stacking Ensemble trained - CV Accuracy: {cv_acc:.4f}")
        logger.info(f"Base models: {list(self.base_models.keys())}, Meta-learner: {meta_learner}")
        
        return {
            'cv_accuracy': cv_acc,
            'cv_std': cv_scores.std(),
            'n_base_models': len(estimators),
            'meta_learner': meta_learner
        }
    
    def predict(
        self,
        X: pd.DataFrame,
        model_name: str,
        return_proba: bool = False
    ) -> np.ndarray:
        """Make predictions using specified model.
        
        Args:
            X: Features to predict
            model_name: Name of model to use
            return_proba: Whether to return probabilities
        
        Returns:
            Predictions (labels or probabilities)
        
        Raises:
            ValueError: If model not found
        """
        # Check base models
        if model_name in self.base_models:
            model = self.base_models[model_name]
        # Check ensemble models
        elif model_name in self.ensemble_models:
            model = self.ensemble_models[model_name]
        else:
            raise ValueError(f"Model '{model_name}' not found")
        
        if return_proba:
            return model.predict_proba(X)
        else:
            return model.predict(X)
    
    def predict_all_base_models(
        self,
        X: pd.DataFrame,
        return_proba: bool = False
    ) -> Dict[str, np.ndarray]:
        """Get predictions from all base models.
        
        Args:
            X: Features to predict
            return_proba: Whether to return probabilities
        
        Returns:
            Dictionary mapping model_name -> predictions
        """
        predictions = {}
        for name, model in self.base_models.items():
            if return_proba:
                predictions[name] = model.predict_proba(X)
            else:
                predictions[name] = model.predict(X)
        
        return predictions
    
    def save(self, path: Path) -> None:
        """Save all models to disk.
        
        Args:
            path: Directory path to save models
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save base models
        for name, model in self.base_models.items():
            model_path = path / f"{name}.pkl"
            joblib.dump(model, model_path)
            logger.info(f"Saved {name} to {model_path}")
        
        # Save ensemble models
        for name, model in self.ensemble_models.items():
            model_path = path / f"{name}_ensemble.pkl"
            joblib.dump(model, model_path)
            logger.info(f"Saved {name} ensemble to {model_path}")
        
        logger.info(f"All models saved to {path}")
    
    def load(self, path: Path, model_name: str) -> Any:
        """Load a specific model from disk.
        
        Args:
            path: Directory path containing models
            model_name: Name of model to load
        
        Returns:
            Loaded model
        """
        path = Path(path)
        model_path = path / f"{model_name}.pkl"
        
        if not model_path.exists():
            model_path = path / f"{model_name}_ensemble.pkl"
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        model = joblib.load(model_path)
        logger.info(f"Loaded {model_name} from {model_path}")
        
        return model
