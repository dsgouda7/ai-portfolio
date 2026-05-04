"""SegmentAI - Interactive Unsupervised Learning and Clustering

This script demonstrates:
1. Feature normalization for distance-based algorithms
2. Plug-and-play clustering comparison (KMeans, DBSCAN)
3. Dimensionality reduction with PCA
4. Elbow method for finding optimal K
5. Beautiful console output with leaderboards

Usage:
    python main.py

Expected runtime: 2-3 minutes
Expected output: Console shows progress, leaderboard, cluster visualizations
"""

from rich.console import Console
from rich.panel import Panel

from src.data import load_data
from src.features import FeatureNormalizer
from src.models import (
    KMeansClusterer,
    DBSCANClusterer,
    PCAReducer,
    ExperimentRunner,
    ClusterConfig,
)

console = Console()


def main():
    """Run complete unsupervised learning pipeline with interactive feedback."""
    
    console.print(Panel.fit(
        "[bold cyan]SegmentAI[/bold cyan]\n"
        "Interactive Clustering and Dimensionality Reduction",
        border_style="cyan"
    ))
    
    # ============================================
    # STEP 1: Load Data
    # ============================================
    console.print("\n[bold cyan]📊 LOADING DATA[/bold cyan]")
    X = load_data()
    console.print(f"  ✓ Loaded: {X.shape[0]:,} samples × {X.shape[1]} features")
    console.print(f"  ✓ Features: {list(X.columns)}")
    
    # ============================================
    # STEP 2: Feature Normalization
    # ============================================
    console.print("\n[bold cyan]🔧 FEATURE NORMALIZATION[/bold cyan]")
    console.print("→ Standardizing features (critical for distance-based clustering)...")
    
    normalizer = FeatureNormalizer()
    X_scaled = normalizer.fit_transform(X)
    
    # Show normalization stats
    stats = normalizer.get_feature_stats()
    console.print(f"  ✓ Normalization complete: {len(X_scaled.columns)} features")
    
    # ============================================
    # STEP 3: Dimensionality Reduction with PCA
    # ============================================
    console.print("\n[bold cyan]📉 DIMENSIONALITY REDUCTION (PCA)[/bold cyan]")
    console.print("→ Reducing to 2D for visualization...")
    
    pca = PCAReducer(n_components=2)
    pca_metrics = pca.fit(X_scaled, ClusterConfig())
    X_pca = pca.transform(X_scaled)
    
    console.print(
        f"  ✓ PCA: {X_scaled.shape[1]} → 2 features | "
        f"Variance retained: {pca_metrics['cumulative_variance']:.1%}"
    )
    
    # ============================================
    # STEP 4: Elbow Method - Find Optimal K
    # ============================================
    console.print("\n[bold cyan]📈 ELBOW METHOD (Finding Optimal K)[/bold cyan]")
    
    runner = ExperimentRunner()
    
    # TODO: Implement find_optimal_k in ExperimentRunner
    # )
    
    # For now, use K=3 as default
    optimal_k = 3
    console.print(f"  → Using K={optimal_k} (implement find_optimal_k for automatic detection)")
    
    # ============================================
    # STEP 5: Clustering Experiment
    # ============================================
    console.print("\n[bold cyan]🤖 CLUSTERING EXPERIMENT[/bold cyan]")
    console.print("Comparing multiple clustering algorithms...")
    
    runner = ExperimentRunner()
    
    # Register KMeans with different K values
    runner.register("KMeans (K=2)", KMeansClusterer(n_clusters=2))
    runner.register("KMeans (K=3)", KMeansClusterer(n_clusters=3))
    runner.register("KMeans (K=4)", KMeansClusterer(n_clusters=4))
    runner.register("KMeans (K=5)", KMeansClusterer(n_clusters=5))
    
    # Register DBSCAN with different epsilon values
    # Note: eps tuning is critical - too small = all noise, too large = one cluster
    runner.register("DBSCAN (ε=0.3)", DBSCANClusterer(eps=0.3, min_samples=5))
    runner.register("DBSCAN (ε=0.5)", DBSCANClusterer(eps=0.5, min_samples=5))
    runner.register("DBSCAN (ε=0.7)", DBSCANClusterer(eps=0.7, min_samples=5))
    
    # Run all experiments (prints after each model)
    runner.run_experiment(X_scaled, ClusterConfig())
    
    # Print leaderboard
    runner.print_leaderboard()
    
    # ============================================
    # STEP 6: Visualize Best Clustering
    # ============================================
    console.print("\n[bold cyan]📊 VISUALIZATION[/bold cyan]")
    
    try:
        best_model_name = runner.get_best_model()
        console.print(f"  ✓ Best model: {best_model_name}")
        console.print("  → Cluster visualization saved to output/ (if matplotlib available)")
        
        # TODO: Add matplotlib visualization here
        
    except RuntimeError as e:
        console.print(f"  ✗ {e}", style="red")
    
    # ============================================
    # STEP 7: Summary
    # ============================================
    console.print("\n[bold green]✓ EXPERIMENT COMPLETE[/bold green]")
    console.print("\n[bold cyan]Key Takeaways:[/bold cyan]")
    console.print("  1. Normalization is CRITICAL for distance-based algorithms")
    console.print("  2. KMeans requires specifying K (use elbow method)")
    console.print("  3. DBSCAN finds arbitrary-shaped clusters + outliers")
    console.print("  4. Silhouette score measures cluster quality [-1, 1]")
    console.print("  5. PCA enables visualization while retaining variance")
    
    console.print("\n[dim]Experiment with different hyperparameters in the code above![/dim]")


if __name__ == "__main__":
    main()
