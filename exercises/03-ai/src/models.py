"""AI model training with LLM fine-tuning and RAG pipeline

This module provides:
- Abstract AIModel interface for plug-and-play LLM systems
- LLMFineTuner: Fine-tune models with LoRA/QLoRA (with TODOs)
- RAGPipeline: Retrieval-Augmented Generation system (with TODOs)
- PromptEngineer: Few-shot prompt optimization (with TODOs)
- ExperimentRunner: Compare different AI approaches
- Immediate feedback with rich console output

Learning objectives:
1. Implement LLM fine-tuning with parameter-efficient methods (LoRA/QLoRA)
2. Build RAG pipeline with vector search and context injection
3. Engineer effective prompts with few-shot examples
4. Compare AI approaches using plug-and-play registry pattern
5. Evaluate with perplexity, BLEU, ROUGE, and retrieval accuracy
6. See results immediately after each model trains
"""

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from rich.console import Console
from rich.table import Table

logger = logging.getLogger("pizzabot")
console = Console()


@dataclass
class AIModelConfig:
    """Configuration for AI model training."""
    max_epochs: int = 3
    batch_size: int = 8
    learning_rate: float = 2e-4
    random_state: int = 42
    verbose: bool = True
    max_length: int = 512
    use_gpu: bool = True


class AIModel(ABC):
    """Abstract base class for all AI models.

    Provides common interface for plug-and-play experimentation.
    Subclasses implement train() and generate() methods.
    """

    def __init__(self, name: str):
        """Initialize AI model with name for display."""
        self.name = name
        self.model = None
        self.tokenizer = None
        self.metrics = {}

    @abstractmethod
    def train(
        self,
        train_data: List[Dict[str, str]],
        eval_data: Optional[List[Dict[str, str]]],
        config: AIModelConfig
    ) -> Dict[str, float]:
        """Train model and return metrics with immediate console feedback.

        Args:
            train_data: Training examples [{"input": "...", "output": "..."}]
            eval_data: Validation examples (optional)
            config: Training configuration

        Returns:
            Dictionary with metrics: {"perplexity": float, "train_loss": float, ...}
        """
        pass

    @abstractmethod
    def generate(
        self,
        prompt: str,
        max_length: int = 256,
        temperature: float = 0.7
    ) -> str:
        """Generate text from prompt.

        Args:
            prompt: Input prompt
            max_length: Maximum tokens to generate
            temperature: Sampling temperature (higher = more creative)

        Returns:
            Generated text
        """
        pass

    def save(self, path: str) -> None:
        """Save trained model to disk."""
        if self.model is None:
            raise ValueError("Cannot save untrained model")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        # Model-specific save logic implemented in subclasses
        logger.info(f"Saved {self.name} to {path}")


class LLMFineTuner(AIModel):
    """LLM fine-tuning with LoRA (Low-Rank Adaptation).

    LoRA adds trainable low-rank matrices to frozen LLM weights:
    - Only 0.1% of parameters trained → faster, less memory
    - Quality comparable to full fine-tuning
    - Useful for domain adaptation (pizza ordering, medical, legal, etc.)

    QLoRA variant uses 4-bit quantization for even lower memory.
    """

    def __init__(
        self,
        base_model: str = "gpt2",
        lora_r: int = 8,
        lora_alpha: int = 32,
        use_qlora: bool = False
    ):
        """Initialize LLM fine-tuner.

        Args:
            base_model: Base model name (gpt2, llama2, mistral, etc.)
            lora_r: LoRA rank (higher = more capacity, slower)
            lora_alpha: LoRA scaling factor
            use_qlora: Use 4-bit quantization for lower memory
        """
        super().__init__(f"FineTune-{base_model} (r={lora_r})")
        self.base_model = base_model
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.use_qlora = use_qlora

    def train(
        self,
        train_data: List[Dict[str, str]],
        eval_data: Optional[List[Dict[str, str]]],
        config: AIModelConfig
    ) -> Dict[str, float]:
        """
        TODO: Implement LLM fine-tuning with LoRA/QLoRA (load model, apply LoRA config, train, evaluate with perplexity/BLEU)

        📖 See: notes/03-ai/ch03-llm-training-pipeline/ (LoRA, QLoRA, parameter-efficient fine-tuning)
        🎯 Unlocks: Constraint #1 BUSINESS VALUE (domain adaptation → pizza ordering style)
                    Constraint #6 RELIABILITY (consistent responses → production quality)
        """
        # TODO: Your implementation here
        raise NotImplementedError("Implement LLM fine-tuning")

    def generate(
        self,
        prompt: str,
        max_length: int = 256,
        temperature: float = 0.7
    ) -> str:
        """
        TODO: Generate text using fine-tuned model (tokenize, generate, decode)

        📖 See: notes/03-ai/ch02-llm-inference-mechanics/ (tokenization, sampling, generation)
                notes/03-ai/ch03-llm-training-pipeline/ (using fine-tuned models)
        🎯 Unlocks: Constraint #1 BUSINESS VALUE (domain-adapted generation)
        """
        # TODO: Your implementation here
        raise NotImplementedError("Implement text generation")


class DPOTrainer(AIModel):
    """Direct Preference Optimisation trainer.

    DPO (Rafailov et al., NeurIPS 2023) aligns model preferences without
    a separate reward model.  It operates on preference triples
    (prompt x, preferred response y_w, rejected response y_l) and minimises:

        L_DPO = -E[log σ(β·log π_θ(y_w|x)/π_ref(y_w|x)
                       - β·log π_θ(y_l|x)/π_ref(y_l|x))]

    Typical usage: run LoRA first (style/behaviour), then DPO on top
    (preference alignment).  For PizzaBot this lifts brand voice score
    from 95% → 99%+ and AOV from $41.00 → $42.50.
    """

    def __init__(
        self,
        base_model: str = "meta-llama/Meta-Llama-3-8B-Instruct",
        beta: float = 0.1,
        learning_rate: float = 5e-7,
    ):
        """Initialise DPO trainer.

        Args:
            base_model: LoRA-tuned SFT model that acts as both policy and
                        (frozen copy as) reference.
            beta: KL-regularisation strength — controls how far the trained
                  policy can deviate from the reference (typical: 0.1–0.5).
            learning_rate: Lower than LoRA; typical range 5e-7 to 1e-6.
        """
        super().__init__(f"DPO-{base_model} (β={beta})")
        self.base_model = base_model
        self.beta = beta
        self.learning_rate = learning_rate

    def prepare_preference_dataset(
        self,
        raw_pairs: List[Dict[str, str]],
    ) -> List[Dict[str, str]]:
        """
        TODO: Validate and format raw annotation pairs into DPO triples.

        Each output dict must contain exactly the keys:
            "prompt", "chosen", "rejected"

        Validation gates:
          - Drop examples where chosen == rejected
          - Ensure minimum token length (≥10 tokens each)
          - Shuffle and return

        📖 See: notes/03-ai/ch03-llm-training-pipeline/ §5.5
                (DPO data collection strategy, PizzaBot preference examples)
        🎯 Unlocks: Constraint #1 BUSINESS VALUE (brand voice alignment)
        """
        # TODO: Your implementation here
        raise NotImplementedError("Implement preference dataset preparation")

    def train(
        self,
        train_data: List[Dict[str, str]],
        eval_data: Optional[List[Dict[str, str]]],
        config: AIModelConfig,
    ) -> Dict[str, float]:
        """
        TODO: Run DPO fine-tuning loop using TRL DPOTrainer.

        Steps:
          1. Load the LoRA-tuned policy model and a frozen copy as ref_model
          2. Build DPOConfig (beta, learning_rate, num_train_epochs=1)
          3. Instantiate trl.DPOTrainer(model, ref_model, args, train_dataset)
          4. Call trainer.train()
          5. Return metrics: rewards/chosen, rewards/rejected, rewards/margins

        📖 See: notes/03-ai/ch03-llm-training-pipeline/ §5.5
                (DPO loss derivation, TRL DPOTrainer code skeleton)
        🎯 Unlocks: Constraint #1 BUSINESS VALUE (AOV lift $41 → $42.50)
        """
        # TODO: Your implementation here
        raise NotImplementedError("Implement DPO training")

    def evaluate(
        self,
        test_data: List[Dict[str, str]],
        config: AIModelConfig,
    ) -> Dict[str, float]:
        """
        TODO: Evaluate alignment quality on a held-out preference set.

        Metric — implicit reward accuracy:
          For each (prompt, chosen, rejected) triple, compute:
            reward_chosen  = β · log π_θ(y_w|x) / π_ref(y_w|x)
            reward_rejected = β · log π_θ(y_l|x) / π_ref(y_l|x)
          accuracy = mean(reward_chosen > reward_rejected)

        Target: accuracy > 0.80 before deploying.

        📖 See: notes/03-ai/ch03-llm-training-pipeline/ §5.5.1
                (DPO decision checkpoint, brand voice score measurement)
        🎯 Unlocks: Constraint #2 ACCURACY (verify alignment quality)
        """
        # TODO: Your implementation here
        raise NotImplementedError("Implement DPO evaluation")


class RAGPipeline(AIModel):
    """Retrieval-Augmented Generation (RAG) pipeline.

    RAG enhances LLM responses with relevant retrieved context:
    1. Retrieve: Vector search finds relevant documents
    2. Augment: Inject documents into prompt context
    3. Generate: LLM generates response with grounded information

    Benefits:
    - Reduces hallucinations (facts come from retrieved docs)
    - No retraining needed (just update document collection)
    - Better for factual Q&A vs. pure generation
    """

    def __init__(
        self,
        base_model: str = "gpt2",
        embedding_model: str = "all-MiniLM-L6-v2",
        vector_db: str = "chromadb",
        top_k: int = 3
    ):
        """Initialize RAG pipeline.

        Args:
            base_model: LLM for generation
            embedding_model: Model for document embeddings
            vector_db: Vector database (chromadb or faiss)
            top_k: Number of documents to retrieve
        """
        super().__init__(f"RAG-{base_model} (k={top_k})")
        self.base_model = base_model
        self.embedding_model = embedding_model
        self.vector_db_type = vector_db
        self.top_k = top_k
        self.vector_db = None
        self.embedder = None

    def train(
        self,
        train_data: List[Dict[str, str]],
        eval_data: Optional[List[Dict[str, str]]],
        config: AIModelConfig
    ) -> Dict[str, float]:
        """
        TODO: Build RAG pipeline with vector database (load LLM, create embedder, index documents, evaluate retrieval + generation)

        📖 See: notes/03-ai/ch07-rag-and-embeddings/ (RAG architecture, retrieval + generation)
                notes/03-ai/ch08-vector-dbs/ (vector database integration)
        🎯 Unlocks: Constraint #2 ACCURACY (ground answers in menu facts → <5% error rate)
        """
        # TODO: Your implementation here
        raise NotImplementedError("Implement RAG pipeline")

    def generate(
        self,
        prompt: str,
        max_length: int = 256,
        temperature: float = 0.7
    ) -> str:
        """
        TODO: Generate response with retrieved context (encode query, retrieve docs, build augmented prompt, generate)

        📖 See: notes/03-ai/ch07-rag-and-embeddings/ (RAG pipeline: retrieve → augment → generate)
        🎯 Unlocks: Constraint #2 ACCURACY (context-grounded generation prevents hallucination)
        """
        # TODO: Your implementation here
        raise NotImplementedError("Implement RAG generation")


class PromptEngineer(AIModel):
    """Prompt engineering with few-shot examples.

    Few-shot learning provides examples in the prompt:
    - Zero-shot: No examples, just instruction
    - Few-shot: 2-5 examples showing desired behavior
    - Many-shot: 10+ examples (context window permitting)

    Often matches fine-tuning quality without training!
    """

    def __init__(
        self,
        base_model: str = "gpt2",
        n_shot: int = 3
    ):
        """Initialize prompt engineer.

        Args:
            base_model: LLM for generation
            n_shot: Number of examples to include in prompt
        """
        super().__init__(f"FewShot-{base_model} (n={n_shot})")
        self.base_model = base_model
        self.n_shot = n_shot
        self.few_shot_examples = []

    def train(
        self,
        train_data: List[Dict[str, str]],
        eval_data: Optional[List[Dict[str, str]]],
        config: AIModelConfig
    ) -> Dict[str, float]:
        """
        TODO: Build few-shot prompt with best examples (load LLM, select diverse examples, evaluate with BLEU/ROUGE)

        📖 See: notes/03-ai/ch05-prompt-engineering/ (few-shot learning, example selection)
        🎯 Unlocks: Constraint #1 BUSINESS VALUE (few-shot examples → better ordering patterns)
                    Constraint #4 COST (no training overhead, just prompt engineering)
        """
        # TODO: Your implementation here
        raise NotImplementedError("Implement few-shot prompt engineering")

    def generate(
        self,
        prompt: str,
        max_length: int = 256,
        temperature: float = 0.7
    ) -> str:
        """
        TODO: Generate with few-shot examples (build prompt with examples, generate, extract answer)

        📖 See: notes/03-ai/ch05-prompt-engineering/ (few-shot prompting patterns)
        🎯 Unlocks: Constraint #1 BUSINESS VALUE (in-context learning for ordering)
        """
        # TODO: Your implementation here
        raise NotImplementedError("Implement few-shot generation")


class ExperimentRunner:
    """Run experiments with multiple AI models and compare results.

    Provides plug-and-play framework for trying different approaches:
    1. Register AI models to try (fine-tuning, RAG, few-shot)
    2. Run all experiments with immediate feedback
    3. Print leaderboard sorted by performance

    Example:
        >>> runner = ExperimentRunner()
        >>> runner.register("LoRA r=8", LLMFineTuner("gpt2", lora_r=8))
        >>> runner.register("RAG k=3", RAGPipeline("gpt2", top_k=3))
        >>> runner.register("3-shot", PromptEngineer("gpt2", n_shot=3))
        >>> runner.run_experiment(train_data, eval_data, AIModelConfig())
        >>> runner.print_leaderboard()
    """

    def __init__(self):
        """Initialize empty experiment runner."""
        self.models: Dict[str, AIModel] = {}
        self.results: List[Dict[str, Any]] = []

    def register(self, name: str, model: AIModel):
        """Register an AI model to try in experiments.

        Args:
            name: Display name for results
            model: AIModel instance to train
        """
        self.models[name] = model
        console.print(f"Registered: {name}", style="dim")

    def run_experiment(
        self,
        train_data: List[Dict[str, str]],
        eval_data: List[Dict[str, str]],
        config: AIModelConfig
    ):
        """
        TODO: Run all registered AI models and collect results with error handling

        📖 See: notes/03-ai/ch08_evaluating_ai_systems/ (A/B testing, experiment design, comparing models)
        🎯 Unlocks: Constraint #6 RELIABILITY (systematic evaluation → choose best approach)
        """
        # TODO: Your implementation here
        raise NotImplementedError("Implement experiment runner")

    def print_leaderboard(self):
        """
        TODO: Print sorted leaderboard table comparing all AI models by BLEU score

        📖 See: notes/03-ai/ch08_evaluating_ai_systems/ (metrics reporting, model comparison)
        🎯 Unlocks: Constraint #6 RELIABILITY (visibility → data-driven decisions)
        """
        # TODO: Your implementation here
        raise NotImplementedError("Implement leaderboard")

    def get_best_model(self) -> AIModel:
        """Return model with highest BLEU score."""
        if not self.results:
            raise ValueError("No experiments run yet")
        valid_results = [r for r in self.results if 'bleu' in r]
        if not valid_results:
            raise ValueError("No results with BLEU scores")
        best_result = max(valid_results, key=lambda x: x['bleu'])
        return self.models[best_result['model']]
