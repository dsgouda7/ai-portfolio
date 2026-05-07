"""PizzaBot AI - Interactive LLM Fine-tuning and RAG Experimentation

This script demonstrates:
1. Text preprocessing and embedding generation with immediate feedback
2. Vector database creation and retrieval evaluation
3. Plug-and-play AI model comparison (LoRA fine-tuning, RAG, few-shot)
4. Beautiful console output with leaderboards
5. LLM evaluation with BLEU, ROUGE, perplexity, and retrieval accuracy

Usage:
    python main.py

Expected runtime: 5-10 minutes (depends on implementations)
Expected output: Console shows progress, metrics, leaderboard, and demo conversations
"""

from pathlib import Path
from rich.console import Console
from rich.panel import Panel

from src.features import TextPreprocessor, EmbeddingGenerator, VectorDatabase, PreprocessConfig
from src.models import (
    LLMFineTuner,
    RAGPipeline,
    PromptEngineer,
    ExperimentRunner,
    AIModelConfig,
)

console = Console()


def load_pizza_data():
    """Load pizza ordering training data (mocked for demonstration)."""
    # In production, load from knowledge_base/ directory
    train_data = [
        {
            "input": "What pizzas do you have?",
            "output": "We have Margherita ($12), Pepperoni ($14), Veggie Supreme ($13), and BBQ Chicken ($15)."
        },
        {
            "input": "I want to order a pepperoni pizza",
            "output": "Great choice! A pepperoni pizza is $14. What size would you like - small, medium, or large?"
        },
        {
            "input": "Do you have vegetarian options?",
            "output": "Yes! Our Veggie Supreme has mushrooms, peppers, onions, and olives for $13."
        },
        {
            "input": "How long is delivery?",
            "output": "Typical delivery time is 30-45 minutes depending on your location."
        },
        {
            "input": "Can I add extra cheese?",
            "output": "Absolutely! Extra cheese is $2 per pizza. Would you like to add it to your order?"
        },
        {
            "input": "What's your most popular pizza?",
            "output": "Our Pepperoni is definitely the most popular! It's a classic with quality ingredients."
        },
        {
            "input": "I'd like to order 2 large pepperoni pizzas",
            "output": "Perfect! 2 large pepperoni pizzas will be $28. Can I get your delivery address?"
        },
        {
            "input": "Do you offer gluten-free crust?",
            "output": "Yes, we offer gluten-free crust as an option for $3 extra per pizza."
        },
    ]
    
    eval_data = [
        {
            "input": "What's on the veggie pizza?",
            "output": "The Veggie Supreme has mushrooms, peppers, onions, and olives."
        },
        {
            "input": "How much is a large pepperoni?",
            "output": "A large pepperoni pizza is $14."
        },
        {
            "input": "Can I order for delivery?",
            "output": "Yes! We deliver within 30-45 minutes. What's your address?"
        },
    ]
    
    return train_data, eval_data


def main():
    """Run complete AI pipeline with interactive feedback."""
    
    console.print(Panel.fit(
        "[bold cyan]PizzaBot AI[/bold cyan]\n"
        "LLM Fine-tuning, RAG, and Few-Shot Learning",
        border_style="cyan"
    ))
    
    # ============================================
    # STEP 1: Load Data
    # ============================================
    console.print("\n[bold cyan]📊 LOADING DATA[/bold cyan]")
    train_data, eval_data = load_pizza_data()
    console.print(f"  ✓ Train: {len(train_data)} examples")
    console.print(f"  ✓ Eval:  {len(eval_data)} examples")
    
    # ============================================
    # PizzaBot Business Context
    # ============================================
    console.print("\n[bold yellow]🎯 PIZZABOT GRAND CHALLENGE[/bold yellow]")
    console.print("You're building Mamma Rosa's PizzaBot to replace phone agents")
    console.print("\n[bold]Production Targets:[/bold]")
    console.print("  • >25% conversion rate (orders/conversations)")
    console.print("  • <$0.08 cost per conversation")
    console.print("  • <5% error rate (no price hallucinations)")
    console.print("  • <3s p95 latency per response")
    
    console.print("\n[bold]Three Critical Scenarios Your Models Will Handle:[/bold]")
    
    console.print("\n[bold cyan]Scenario 1:[/bold cyan] Simple Order (RAG Success)")
    console.print("  Customer: 'What veggie pizzas do you have under $15?'")
    console.print("  Challenge: RAG must retrieve accurate menu items")
    console.print("  Success: 38% conversion (above 25% target)")
    
    console.print("\n[bold cyan]Scenario 2:[/bold cyan] Price Hallucination Test")
    console.print("  Customer: 'How much is a large pepperoni?'")
    console.print("  WITHOUT RAG: Bot hallucinates $14 (stale data) → 12% conversion")
    console.print("  WITH RAG: Bot retrieves correct $15.50 → 28% conversion")
    console.print("  Business Impact: RAG adds $1.85M/month revenue vs. $120/month cost")
    
    console.print("\n[bold cyan]Scenario 3:[/bold cyan] End-to-End Order (Function Calling)")
    console.print("  Multi-turn: Order pizza → Add sides → Enter address → Place order")
    console.print("  Success: 32% conversion with $4 upsell (garlic bread)")
    console.print("  Key: Function calling enables POS integration + proactive upselling")
    
    console.print("\n[dim]After implementing TODOs, your models will be evaluated on these scenarios[/dim]")
    
    # ============================================
    # STEP 2: Text Preprocessing
    # ============================================
    console.print("\n[bold cyan]🔧 TEXT PREPROCESSING[/bold cyan]")
    console.print("→ Preprocessing training data...")
    
    preprocessor = TextPreprocessor(
        PreprocessConfig(
            lowercase=True,
            remove_punctuation=False,  # Keep for questions
            remove_stopwords=False     # Keep for semantic meaning
        )
    )
    
    # TODO: Uncomment when TextPreprocessor.preprocess_batch is implemented
    # train_texts = [item['input'] + " " + item['output'] for item in train_data]
    # cleaned_texts = preprocessor.preprocess_batch(train_texts)
    # console.print(f"  ✓ Preprocessed {len(cleaned_texts)} texts")
    
    console.print("  ⚠ TextPreprocessor TODOs not implemented - skipping", style="yellow")
    
    # ============================================
    # STEP 3: Generate Embeddings
    # ============================================
    console.print("\n[bold cyan]🧮 EMBEDDING GENERATION[/bold cyan]")
    console.print("→ Loading embedding model...")
    
    embedder = EmbeddingGenerator(
        model_name="all-MiniLM-L6-v2",
        device="cpu"  # Change to "cuda" if GPU available
    )
    
    # TODO: Uncomment when EmbeddingGenerator methods are implemented
    # embedder.load()
    # 
    # console.print("→ Generating embeddings for training data...")
    # train_texts = [item['input'] + " " + item['output'] for item in train_data]
    # embeddings = embedder.encode_batch(train_texts, batch_size=8)
    # console.print(f"  ✓ Generated {embeddings.shape[0]} embeddings ({embeddings.shape[1]}-dim)")
    
    console.print("  ⚠ EmbeddingGenerator TODOs not implemented - skipping", style="yellow")
    
    # ============================================
    # STEP 4: Build Vector Database
    # ============================================
    console.print("\n[bold cyan]💾 VECTOR DATABASE[/bold cyan]")
    console.print("→ Creating ChromaDB collection...")
    
    vector_db = VectorDatabase(
        db_type="chromadb",
        collection_name="pizza_docs",
        persist_directory="./data/vector_db"
    )
    
    # TODO: Uncomment when VectorDatabase methods are implemented
    # vector_db.create(embedding_dim=embeddings.shape[1])
    # vector_db.add_documents(train_texts, embeddings)
    # 
    # # Evaluate retrieval
    # test_queries = [
    #     {"query": "vegetarian pizza", "expected_text": "Veggie Supreme"},
    #     {"query": "delivery time", "expected_text": "30-45 minutes"},
    #     {"query": "pepperoni price", "expected_text": "$14"},
    # ]
    # retrieval_metrics = vector_db.evaluate_retrieval(test_queries, embedder, top_k=3)
    
    console.print("  ⚠ VectorDatabase TODOs not implemented - skipping", style="yellow")
    
    # ============================================
    # STEP 5: AI Model Training (Plug-and-Play Comparison)
    # ============================================
    console.print("\n[bold cyan]🤖 AI MODEL TRAINING[/bold cyan]")
    console.print("Comparing 3 approaches: LoRA fine-tuning, RAG, and few-shot...")
    
    runner = ExperimentRunner()
    
    # Register different AI approaches
    # TODO: Uncomment when models are implemented
    # runner.register("LoRA r=8", LLMFineTuner("gpt2", lora_r=8))
    # runner.register("LoRA r=16", LLMFineTuner("gpt2", lora_r=16))
    # runner.register("RAG k=3", RAGPipeline("gpt2", top_k=3))
    # runner.register("RAG k=5", RAGPipeline("gpt2", top_k=5))
    # runner.register("3-shot", PromptEngineer("gpt2", n_shot=3))
    # runner.register("5-shot", PromptEngineer("gpt2", n_shot=5))
    
    console.print("  ⚠ AI models not yet implemented - register models in main.py", style="yellow")
    console.print("  ℹ Example: runner.register('LoRA r=8', LLMFineTuner('gpt2', lora_r=8))", style="dim")
    
    # Run all experiments (prints after each model)
    # TODO: Uncomment when ExperimentRunner is implemented
    # config = AIModelConfig(
    #     max_epochs=3,
    #     batch_size=4,
    #     learning_rate=2e-4,
    #     use_gpu=False  # Set True if GPU available
    # )
    # runner.run_experiment(train_data, eval_data, config)
    # 
    # # Print leaderboard
    # runner.print_leaderboard()
    
    console.print("  ⚠ ExperimentRunner TODOs not implemented - skipping", style="yellow")
    
    # ============================================
    # STEP 6: Interactive Demo
    # ============================================
    console.print("\n[bold cyan]💬 INTERACTIVE DEMO[/bold cyan]")
    console.print("→ Testing best model on sample queries...")
    
    # TODO: Uncomment when models are implemented
    # best_model = runner.get_best_model()
    # 
    # demo_queries = [
    #     "What pizzas do you recommend?",
    #     "I want a vegetarian pizza for delivery",
    #     "How much for 3 pepperoni pizzas?"
    # ]
    # 
    # for query in demo_queries:
    #     console.print(f"\n[bold]User:[/bold] {query}")
    #     response = best_model.generate(query, max_length=100, temperature=0.7)
    #     console.print(f"[bold green]Bot:[/bold green] {response}")
    
    console.print("  ⚠ Models not implemented - complete TODOs to see demo", style="yellow")
    
    # ============================================
    # SUMMARY
    # ============================================
    console.print("\n[bold cyan]📋 NEXT STEPS[/bold cyan]")
    console.print("Complete the TODOs in this order:")
    console.print("  1. [bold]src/features.py[/bold] - TextPreprocessor, EmbeddingGenerator, VectorDatabase")
    console.print("  2. [bold]src/models.py[/bold] - LLMFineTuner, RAGPipeline, PromptEngineer")
    console.print("  3. [bold]src/models.py[/bold] - ExperimentRunner.run_experiment() and print_leaderboard()")
    console.print("  4. Run [bold cyan]python main.py[/bold cyan] again to see full pipeline!")
    
    console.print("\n[bold green]Estimated time:[/bold green] 3-4 hours for all TODOs")
    console.print("[bold green]Learning outcome:[/bold green] Complete understanding of LLM fine-tuning, RAG, and evaluation")


if __name__ == "__main__":
    main()
