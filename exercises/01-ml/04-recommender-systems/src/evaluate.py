"""Model evaluation and diagnostics for FlixAI

Provides: RecommenderEvaluator for comprehensive model assessment
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error


logger = logging.getLogger("flixai")


class RecommenderEvaluator:
    """Automated recommender system evaluation with ranking metrics.
    
    Provides:
    - Rating prediction metrics (RMSE, MAE)
    - Ranking metrics (Precision@K, Recall@K, NDCG)
    - Coverage analysis
    - Diversity metrics
    
    Attributes:
        metrics: Dictionary of computed metrics
    
    Example:
        >>> evaluator = RecommenderEvaluator()
        >>> metrics = evaluator.evaluate(model, test_ratings)
        >>> print(f"RMSE: {metrics['rmse']:.3f}, Precision@10: {metrics['precision@10']:.3f}")
    """
    
    def __init__(self):
        """Initialize evaluator."""
        self.metrics = {}
        
        # Set plotting style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (10, 6)
        
        logger.info("Initialized RecommenderEvaluator")
    
    def evaluate(
        self,
        model,
        test_ratings: pd.DataFrame,
        k_values: List[int] = [5, 10, 20],
        set_name: str = "test"
    ) -> Dict[str, float]:
        """Evaluate recommender model on test set.
        
        Args:
            model: Trained model with predict_rating() and recommend_items() methods
            test_ratings: DataFrame with user_id, item_id, rating columns
            k_values: List of K values for ranking metrics
            set_name: Name of dataset (for logging)
        
        Returns:
            Dictionary with rmse, mae, precision@k, recall@k, ndcg
        
        Example:
            >>> metrics = evaluator.evaluate(model, test_ratings, k_values=[5, 10])
        """
        logger.info(f"Evaluating model on {set_name} set ({len(test_ratings)} ratings)")
        
        # Rating prediction metrics
        predictions = []
        actuals = []
        
        for _, row in test_ratings.iterrows():
            try:
                pred = model.predict_rating(
                    user_id=int(row['user_id']),
                    item_id=int(row['item_id'])
                )
                predictions.append(pred)
                actuals.append(row['rating'])
            except Exception as e:
                logger.warning(f"Failed to predict for user {row['user_id']}, item {row['item_id']}: {e}")
                continue
        
        # Compute rating metrics
        rmse = np.sqrt(mean_squared_error(actuals, predictions))
        mae = mean_absolute_error(actuals, predictions)
        
        metrics = {
            "rmse": float(rmse),
            "mae": float(mae),
        }
        
        # Ranking metrics for each K
        for k in k_values:
            precision_k, recall_k, ndcg_k = self._compute_ranking_metrics(
                model, test_ratings, k
            )
            metrics[f"precision@{k}"] = precision_k
            metrics[f"recall@{k}"] = recall_k
            metrics[f"ndcg@{k}"] = ndcg_k
        
        # Store results
        self.metrics[set_name] = metrics
        
        logger.info(
            f"{set_name.capitalize()} metrics - "
            f"RMSE: {rmse:.3f}, MAE: {mae:.3f}, "
            f"Precision@10: {metrics.get('precision@10', 0):.3f}"
        )
        
        return metrics
    
    def _compute_ranking_metrics(
        self,
        model,
        test_ratings: pd.DataFrame,
        k: int
    ) -> tuple:
        """Compute ranking metrics for top-k recommendations.
        
        Args:
            model: Trained model
            test_ratings: Test ratings
            k: Number of recommendations
        
        Returns:
            Tuple of (precision@k, recall@k, ndcg@k)
        """
        # Group by user
        user_items = test_ratings.groupby('user_id')['item_id'].apply(set).to_dict()
        
        precisions = []
        recalls = []
        ndcgs = []
        
        for user_id, relevant_items in user_items.items():
            try:
                # Get recommendations
                recommendations = model.recommend_items(
                    user_id=user_id,
                    k=k,
                    exclude_seen=False
                )
                recommended_items = {item_id for item_id, _ in recommendations}
                
                # Precision@K
                hits = len(recommended_items & relevant_items)
                precision = hits / k if k > 0 else 0
                precisions.append(precision)
                
                # Recall@K
                recall = hits / len(relevant_items) if len(relevant_items) > 0 else 0
                recalls.append(recall)
                
                # NDCG@K
                ndcg = self._compute_ndcg(recommendations, relevant_items, k)
                ndcgs.append(ndcg)
                
            except Exception as e:
                logger.warning(f"Failed to compute ranking metrics for user {user_id}: {e}")
                continue
        
        avg_precision = np.mean(precisions) if precisions else 0.0
        avg_recall = np.mean(recalls) if recalls else 0.0
        avg_ndcg = np.mean(ndcgs) if ndcgs else 0.0
        
        return float(avg_precision), float(avg_recall), float(avg_ndcg)
    
    def _compute_ndcg(
        self,
        recommendations: List[tuple],
        relevant_items: set,
        k: int
    ) -> float:
        """Compute Normalized Discounted Cumulative Gain.
        
        Args:
            recommendations: List of (item_id, score) tuples
            relevant_items: Set of relevant item IDs
            k: Number of recommendations
        
        Returns:
            NDCG@K score
        """
        # DCG
        dcg = 0.0
        for i, (item_id, score) in enumerate(recommendations[:k]):
            if item_id in relevant_items:
                dcg += 1.0 / np.log2(i + 2)  # i+2 because i starts at 0
        
        # IDCG (ideal DCG)
        idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(relevant_items), k)))
        
        # NDCG
        ndcg = dcg / idcg if idcg > 0 else 0.0
        
        return ndcg
    
    def plot_metrics_comparison(
        self,
        save_path: Optional[str] = None
    ) -> None:
        """Plot comparison of metrics across different K values.
        
        Args:
            save_path: Path to save plot (if None, displays only)
        
        Example:
            >>> evaluator.plot_metrics_comparison(save_path="plots/metrics.png")
        """
        if not self.metrics:
            raise RuntimeError("No metrics available. Call evaluate() first.")
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        # Extract K values and metrics
        k_values = []
        precisions = []
        recalls = []
        ndcgs = []
        
        for metric_name, value in self.metrics.get('test', {}).items():
            if '@' in metric_name:
                k = int(metric_name.split('@')[1])
                k_values.append(k)
                
                if 'precision' in metric_name:
                    precisions.append(value)
                elif 'recall' in metric_name:
                    recalls.append(value)
                elif 'ndcg' in metric_name:
                    ndcgs.append(value)
        
        # Sort by K
        sorted_indices = np.argsort(k_values)
        k_values = [k_values[i] for i in sorted_indices]
        precisions = [precisions[i] for i in sorted_indices]
        recalls = [recalls[i] for i in sorted_indices]
        ndcgs = [ndcgs[i] for i in sorted_indices]
        
        # Precision@K
        axes[0].plot(k_values, precisions, marker='o', linewidth=2)
        axes[0].set_xlabel('K')
        axes[0].set_ylabel('Precision@K')
        axes[0].set_title('Precision vs K')
        axes[0].grid(True, alpha=0.3)
        
        # Recall@K
        axes[1].plot(k_values, recalls, marker='o', linewidth=2, color='orange')
        axes[1].set_xlabel('K')
        axes[1].set_ylabel('Recall@K')
        axes[1].set_title('Recall vs K')
        axes[1].grid(True, alpha=0.3)
        
        # NDCG@K
        axes[2].plot(k_values, ndcgs, marker='o', linewidth=2, color='green')
        axes[2].set_xlabel('K')
        axes[2].set_ylabel('NDCG@K')
        axes[2].set_title('NDCG vs K')
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Metrics plot saved to {save_path}")
        else:
            plt.show()
    
    def compute_coverage(
        self,
        model,
        all_items: set,
        n_users: int = 100,
        k: int = 10
    ) -> float:
        """Compute recommendation coverage.
        
        Coverage = fraction of items that appear in at least one recommendation.
        
        Args:
            model: Trained model
            all_items: Set of all item IDs
            n_users: Number of users to sample
            k: Number of recommendations per user
        
        Returns:
            Coverage score (0.0 to 1.0)
        
        Example:
            >>> coverage = evaluator.compute_coverage(model, all_items, n_users=100)
            >>> print(f"Coverage: {coverage:.2%}")
        """
        recommended_items = set()
        
        for user_id in range(n_users):
            try:
                recommendations = model.recommend_items(user_id=user_id, k=k)
                for item_id, _ in recommendations:
                    recommended_items.add(item_id)
            except Exception:
                continue
        
        coverage = len(recommended_items) / len(all_items) if all_items else 0.0
        
        logger.info(f"Coverage: {coverage:.2%} ({len(recommended_items)}/{len(all_items)} items)")
        
        return float(coverage)
