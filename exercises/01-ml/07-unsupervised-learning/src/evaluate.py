"""Model evaluation and diagnostics for SegmentAI

Provides: AutoEvaluator for comprehensive clustering assessment
"""

import logging
from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    silhouette_score,
    silhouette_samples,
    davies_bouldin_score,
    calinski_harabasz_score
)
from scipy.cluster.hierarchy import dendrogram, linkage


logger = logging.getLogger("segmentai")


class AutoEvaluator:
    """Automated clustering evaluation with visualizations and diagnostics.
    
    Provides:
    - Unsupervised metrics (silhouette, Davies-Bouldin, Calinski-Harabasz)
    - Elbow method plots
    - Silhouette analysis
    - Dendrogram visualization
    - Cluster distribution analysis
    
    Attributes:
        metrics: Dictionary of computed metrics
        cluster_labels: Current cluster assignments
    
    Example:
        >>> evaluator = AutoEvaluator()
        >>> metrics = evaluator.evaluate(X, labels)
        >>> evaluator.plot_silhouette(X, labels)
        >>> evaluator.plot_elbow(X, k_range=[2, 3, 4, 5])
    """
    
    def __init__(self):
        """Initialize evaluator."""
        self.metrics = {}
        self.cluster_labels = None
        
        # Set plotting style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (12, 6)
        
        logger.info("Initialized AutoEvaluator")
    
    def evaluate(
        self,
        X: pd.DataFrame,
        labels: np.ndarray,
        y_true: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """Compute comprehensive clustering metrics.
        
        Args:
            X: Features
            labels: Predicted cluster labels
            y_true: Optional ground truth labels (for reference only)
        
        Returns:
            Dictionary with metrics
        
        Example:
            >>> metrics = evaluator.evaluate(X, labels)
            >>> print(f"Silhouette: {metrics['silhouette_score']:.3f}")
        """
        self.cluster_labels = labels
        
        # Count clusters (excluding noise -1)
        unique_labels = set(labels)
        n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
        n_noise = list(labels).count(-1)
        
        logger.info(f"Evaluating clustering: {n_clusters} clusters, {n_noise} noise points")
        
        metrics = {
            "n_clusters": n_clusters,
            "n_noise": n_noise,
            "noise_ratio": n_noise / len(X) if len(X) > 0 else 0.0
        }
        
        # Can only compute metrics if we have 2+ clusters and non-noise points
        if n_clusters >= 2 and n_noise < len(X):
            # Filter out noise points
            mask = labels != -1
            X_filtered = X.values[mask]
            labels_filtered = labels[mask]
            
            # Silhouette score (-1 to 1, higher is better)
            try:
                metrics["silhouette_score"] = silhouette_score(X_filtered, labels_filtered)
            except:
                metrics["silhouette_score"] = -1.0
            
            # Davies-Bouldin index (lower is better, 0 is best)
            try:
                metrics["davies_bouldin_index"] = davies_bouldin_score(X_filtered, labels_filtered)
            except:
                metrics["davies_bouldin_index"] = float('inf')
            
            # Calinski-Harabasz score (higher is better)
            try:
                metrics["calinski_harabasz_score"] = calinski_harabasz_score(X_filtered, labels_filtered)
            except:
                metrics["calinski_harabasz_score"] = 0.0
        else:
            logger.warning("Cannot compute metrics: need 2+ clusters and <100% noise")
            metrics["silhouette_score"] = -1.0
            metrics["davies_bouldin_index"] = float('inf')
            metrics["calinski_harabasz_score"] = 0.0
        
        # Cluster size distribution
        cluster_sizes = pd.Series(labels).value_counts().to_dict()
        metrics["cluster_sizes"] = cluster_sizes
        
        self.metrics = metrics
        
        logger.info(
            f"Metrics - Silhouette: {metrics['silhouette_score']:.3f}, "
            f"Davies-Bouldin: {metrics['davies_bouldin_index']:.3f}, "
            f"Calinski-Harabasz: {metrics['calinski_harabasz_score']:.1f}"
        )
        
        return metrics
    
    def plot_silhouette(
        self,
        X: pd.DataFrame,
        labels: np.ndarray,
        save_path: Optional[str] = None
    ) -> None:
        """Plot silhouette analysis for each cluster.
        
        Args:
            X: Features
            labels: Cluster labels
            save_path: Optional path to save figure
        
        Example:
            >>> evaluator.plot_silhouette(X, labels, "plots/silhouette.png")
        """
        # Filter out noise
        mask = labels != -1
        X_filtered = X.values[mask]
        labels_filtered = labels[mask]
        
        n_clusters = len(set(labels_filtered))
        
        if n_clusters < 2:
            logger.warning("Cannot plot silhouette: need 2+ clusters")
            return
        
        # Compute silhouette scores
        silhouette_avg = silhouette_score(X_filtered, labels_filtered)
        sample_silhouette_values = silhouette_samples(X_filtered, labels_filtered)
        
        # Create plot
        fig, ax = plt.subplots()
        y_lower = 10
        
        for i in sorted(set(labels_filtered)):
            # Get silhouette values for this cluster
            ith_cluster_silhouette_values = sample_silhouette_values[labels_filtered == i]
            ith_cluster_silhouette_values.sort()
            
            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i
            
            color = plt.cm.Spectral(float(i) / n_clusters)
            ax.fill_betweenx(
                np.arange(y_lower, y_upper),
                0,
                ith_cluster_silhouette_values,
                facecolor=color,
                edgecolor=color,
                alpha=0.7
            )
            
            # Label cluster
            ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
            y_lower = y_upper + 10
        
        ax.set_title(f"Silhouette Analysis (avg={silhouette_avg:.3f})")
        ax.set_xlabel("Silhouette Coefficient")
        ax.set_ylabel("Cluster")
        ax.axvline(x=silhouette_avg, color="red", linestyle="--", label="Average")
        ax.legend()
        
        plt.tight_layout()
        
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved silhouette plot to {save_path}")
        
        plt.close()
    
    def plot_elbow(
        self,
        X: pd.DataFrame,
        k_range: list,
        save_path: Optional[str] = None
    ) -> Dict[int, float]:
        """Plot elbow method to find optimal number of clusters.
        
        Args:
            X: Features
            k_range: List of k values to try
            save_path: Optional path to save figure
        
        Returns:
            Dictionary mapping k to inertia
        
        Example:
            >>> inertias = evaluator.plot_elbow(X, k_range=[2, 3, 4, 5, 6, 7])
        """
        from sklearn.cluster import KMeans
        
        logger.info(f"Computing elbow method for k={k_range}")
        
        inertias = {}
        silhouette_scores = {}
        
        for k in k_range:
            if k < 2 or k >= len(X):
                continue
            
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(X)
            
            inertias[k] = kmeans.inertia_
            silhouette_scores[k] = silhouette_score(X, labels)
        
        # Create dual-axis plot
        fig, ax1 = plt.subplots(figsize=(10, 5))
        
        # Plot inertia (elbow)
        color = 'tab:blue'
        ax1.set_xlabel('Number of Clusters (k)')
        ax1.set_ylabel('Inertia (Within-Cluster Sum of Squares)', color=color)
        ax1.plot(list(inertias.keys()), list(inertias.values()), 
                'o-', color=color, linewidth=2, markersize=8)
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.grid(True, alpha=0.3)
        
        # Plot silhouette score on second axis
        ax2 = ax1.twinx()
        color = 'tab:red'
        ax2.set_ylabel('Silhouette Score', color=color)
        ax2.plot(list(silhouette_scores.keys()), list(silhouette_scores.values()),
                's-', color=color, linewidth=2, markersize=8)
        ax2.tick_params(axis='y', labelcolor=color)
        ax2.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Target (0.5)')
        ax2.legend()
        
        plt.title('Elbow Method & Silhouette Score for Optimal k')
        plt.tight_layout()
        
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved elbow plot to {save_path}")
        
        plt.close()
        
        return inertias
    
    def plot_dendrogram(
        self,
        X: pd.DataFrame,
        method: str = 'ward',
        save_path: Optional[str] = None
    ) -> None:
        """Plot dendrogram for hierarchical clustering.
        
        Args:
            X: Features
            method: Linkage method (ward, complete, average, single)
            save_path: Optional path to save figure
        
        Example:
            >>> evaluator.plot_dendrogram(X, method='ward', save_path="plots/dendrogram.png")
        """
        logger.info(f"Computing dendrogram (method={method})")
        
        # Compute linkage
        Z = linkage(X, method=method)
        
        # Create plot
        plt.figure(figsize=(12, 6))
        dendrogram(Z, truncate_mode='lastp', p=30, leaf_font_size=10)
        plt.title(f'Hierarchical Clustering Dendrogram ({method})')
        plt.xlabel('Sample Index or (Cluster Size)')
        plt.ylabel('Distance')
        plt.tight_layout()
        
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved dendrogram to {save_path}")
        
        plt.close()
    
    def plot_clusters_2d(
        self,
        X: pd.DataFrame,
        labels: np.ndarray,
        title: str = "Cluster Visualization",
        save_path: Optional[str] = None
    ) -> None:
        """Plot 2D cluster visualization.
        
        Args:
            X: Features (should be 2D, e.g., after PCA/UMAP)
            labels: Cluster labels
            title: Plot title
            save_path: Optional path to save figure
        
        Example:
            >>> evaluator.plot_clusters_2d(X_umap, labels, "UMAP Clusters")
        """
        if X.shape[1] != 2:
            logger.warning(f"X has {X.shape[1]} dimensions, expected 2 for visualization")
            return
        
        plt.figure(figsize=(10, 8))
        
        # Plot each cluster
        unique_labels = set(labels)
        colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
        
        for label, color in zip(unique_labels, colors):
            if label == -1:
                # Noise points in black
                color = 'k'
                marker = 'x'
                label_name = 'Noise'
            else:
                marker = 'o'
                label_name = f'Cluster {label}'
            
            mask = labels == label
            plt.scatter(
                X.iloc[mask, 0],
                X.iloc[mask, 1],
                c=[color],
                marker=marker,
                s=50,
                alpha=0.7,
                edgecolors='k',
                linewidths=0.5,
                label=label_name
            )
        
        plt.xlabel(X.columns[0])
        plt.ylabel(X.columns[1])
        plt.title(title)
        plt.legend()
        plt.tight_layout()
        
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved cluster plot to {save_path}")
        
        plt.close()
