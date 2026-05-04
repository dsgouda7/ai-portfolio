"""FlixAI - Interactive Recommender System Training and Experimentation

This script demonstrates:
1. User-item matrix construction with immediate feedback
2. Plug-and-play recommender comparison (Collaborative Filtering, Matrix Factorization)
3. Recommendation quality evaluation (Precision@k, Recall@k)
4. Beautiful console output with leaderboards

Usage:
    python main.py

Expected runtime: 3-5 minutes (multiple models)
Expected output: Console shows progress, leaderboard, sample recommendations
"""

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from src.data import load_movielens
from src.features import (
    build_user_item_matrix,
    split_ratings,
    compute_rating_stats,
)
from src.models import (
    CollaborativeFilteringRecommender,
    MatrixFactorizationRecommender,
    ExperimentRunner,
    ModelConfig,
)

console = Console()


def main():
    """Run complete recommender system pipeline with interactive feedback."""
    
    console.print(Panel.fit(
        "[bold cyan]FlixAI[/bold cyan]\n"
        "Interactive Recommender System Training",
        border_style="cyan"
    ))
    
    # ============================================
    # STEP 1: Load MovieLens Data
    # ============================================
    console.print("\n[bold cyan]📊 LOADING DATA[/bold cyan]")
    ratings = load_movielens()
    console.print(f"  ✓ Loaded {len(ratings):,} ratings")
    
    # Compute and display statistics
    stats = compute_rating_stats(ratings)
    
    # ============================================
    # STEP 2: Split Data
    # ============================================
    console.print("\n[bold cyan]✂️  SPLITTING DATA[/bold cyan]")
    train_ratings, test_ratings = split_ratings(ratings, test_size=0.2, random_state=42)
    
    # ============================================
    # STEP 3: Build User-Item Matrix
    # ============================================
    console.print("\n[bold cyan]🔧 BUILDING USER-ITEM MATRIX[/bold cyan]")
    console.print("→ Creating user-item rating matrix...")
    
    user_item_train = build_user_item_matrix(train_ratings)
    user_item_test = build_user_item_matrix(test_ratings)
    
    console.print(f"  ✓ Train matrix: {user_item_train.shape[0]:,} users × {user_item_train.shape[1]:,} items")
    console.print(f"  ✓ Test matrix: {user_item_test.shape[0]:,} users × {user_item_test.shape[1]:,} items")
    
    # ============================================
    # STEP 4: Model Training (Plug-and-Play Comparison)
    # ============================================
    console.print("\n[bold cyan]🤖 MODEL TRAINING[/bold cyan]")
    console.print("Comparing multiple recommender algorithms...")
    
    runner = ExperimentRunner()
    
    # TODO: Register Collaborative Filtering with different k values
    # Try k=10, 20, 30 to see impact of neighborhood size
    # Hint: runner.register("CF (k=20)", CollaborativeFilteringRecommender(n_neighbors=20))
    runner.register("CF (k=10)", CollaborativeFilteringRecommender(n_neighbors=10))
    runner.register("CF (k=20)", CollaborativeFilteringRecommender(n_neighbors=20))
    runner.register("CF (k=30)", CollaborativeFilteringRecommender(n_neighbors=30))
    
    # TODO: Register Matrix Factorization with different numbers of factors
    # Try k=20, 50, 100 to see impact of latent dimensionality
    # Experiment: Which wins? More factors or fewer?
    runner.register("MF (k=20)", MatrixFactorizationRecommender(n_factors=20))
    runner.register("MF (k=50)", MatrixFactorizationRecommender(n_factors=50))
    runner.register("MF (k=100)", MatrixFactorizationRecommender(n_factors=100))
    
    # Run all experiments (prints after each model)
    runner.run_experiment(train_ratings, user_item_train, ModelConfig())
    
    # Print leaderboard
    runner.print_leaderboard()
    
    # ============================================
    # STEP 5: Evaluate Recommendation Quality
    # ============================================
    console.print("\n[bold cyan]📈 RECOMMENDATION QUALITY[/bold cyan]")
    console.print("→ Evaluating top-10 recommendations on test set...")
    
    # TODO: Evaluate recommendations using precision@k and recall@k
    # metrics = runner.evaluate_recommendations(test_ratings, k=10)
    
    # ============================================
    # STEP 6: Generate Sample Recommendations
    # ============================================
    console.print("\n[bold cyan]🎬 SAMPLE RECOMMENDATIONS[/bold cyan]")
    
    best_model = runner.get_best_model()
    console.print(f"→ Generating recommendations using {best_model.name}...")
    
    # TODO: Generate recommendations for sample users
    # Show top-5 recommendations for 3 different users
    # sample_users = [0, 10, 25]
    # 
    # for user_id in sample_users:
    #     recommendations = best_model.recommend(user_id, k=5)
    #     
    #     console.print(f"\n[cyan]User {user_id}:[/cyan]")
    #     for item_id, predicted_rating in recommendations:
    #         console.print(f"  → Item {item_id}: {predicted_rating:.2f} ⭐")
    
    # ============================================
    # STEP 7: Save Best Model
    # ============================================
    console.print("\n[bold cyan]💾 SAVING MODEL[/bold cyan]")
    
    # TODO: Save best model
    # best_model.save("models/best_recommender.pkl")
    # console.print("  ✓ Model saved to models/best_recommender.pkl", style="green")
    
    console.print("\n[bold green]✨ Training complete![/bold green]\n")
    
    # ============================================
    # NEXT STEPS (Optional)
    # ============================================
    console.print("[dim]Next steps:")
    console.print("  1. Try different k values for CF and MF")
    console.print("  2. Implement item-based CF (find similar items)")
    console.print("  3. Add content-based features (movie genres, actors)")
    console.print("  4. Evaluate with NDCG metric for ranking quality")
    console.print("  5. Handle cold start (new users/items)[/dim]")


if __name__ == "__main__":
    main()
