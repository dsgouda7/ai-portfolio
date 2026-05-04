"""
Evaluation metrics for RAG chatbot performance
"""

import time
from typing import Dict, List, Any
import numpy as np
from sklearn.metrics import accuracy_score
from .utils import setup_logger

logger = setup_logger(__name__)


class ChatbotEvaluator:
    """Evaluate chatbot performance metrics."""
    
    def __init__(self):
        self.metrics_history = []
    
    def evaluate_intent_accuracy(
        self,
        predicted_intents: List[str],
        true_intents: List[str]
    ) -> float:
        """
        Calculate intent detection accuracy.
        
        Args:
            predicted_intents: Predicted intent labels
            true_intents: Ground truth intent labels
            
        Returns:
            Accuracy score
        """
        if not predicted_intents or not true_intents:
            return 0.0
        
        accuracy = accuracy_score(true_intents, predicted_intents)
        logger.info(f"Intent accuracy: {accuracy:.2%}")
        return accuracy
    
    def evaluate_response_relevance(
        self,
        responses: List[str],
        context_docs: List[List[Dict[str, Any]]],
        similarity_threshold: float = 0.7
    ) -> float:
        """
        Evaluate response relevance based on retrieved context.
        
        Args:
            responses: Generated responses
            context_docs: Retrieved context for each response
            similarity_threshold: Threshold for relevance
            
        Returns:
            Average relevance score
        """
        relevance_scores = []
        
        for docs in context_docs:
            if docs:
                # Average similarity of retrieved docs
                avg_similarity = np.mean([doc['similarity'] for doc in docs])
                relevance_scores.append(avg_similarity)
            else:
                relevance_scores.append(0.0)
        
        if not relevance_scores:
            return 0.0
        
        avg_relevance = np.mean(relevance_scores)
        logger.info(f"Response relevance: {avg_relevance:.2%}")
        return avg_relevance
    
    def evaluate_latency(
        self,
        latencies: List[float],
        target_ms: float = 500
    ) -> Dict[str, float]:
        """
        Evaluate response latency metrics.
        
        Args:
            latencies: List of latency measurements (seconds)
            target_ms: Target latency in milliseconds
            
        Returns:
            Latency statistics
        """
        if not latencies:
            return {
                'mean_ms': 0,
                'median_ms': 0,
                'p95_ms': 0,
                'p99_ms': 0,
                'within_target': 0.0
            }
        
        latencies_ms = [l * 1000 for l in latencies]
        
        stats = {
            'mean_ms': np.mean(latencies_ms),
            'median_ms': np.median(latencies_ms),
            'p95_ms': np.percentile(latencies_ms, 95),
            'p99_ms': np.percentile(latencies_ms, 99),
            'within_target': sum(l <= target_ms for l in latencies_ms) / len(latencies_ms)
        }
        
        logger.info(
            f"Latency - Mean: {stats['mean_ms']:.0f}ms, "
            f"P95: {stats['p95_ms']:.0f}ms, "
            f"Within target: {stats['within_target']:.1%}"
        )
        
        return stats
    
    def evaluate_conversation_quality(
        self,
        conversations: List[List[Dict[str, str]]],
        target_intent_accuracy: float = 0.90
    ) -> Dict[str, Any]:
        """
        Evaluate overall conversation quality.
        
        Args:
            conversations: List of conversation histories
            target_intent_accuracy: Target intent accuracy
            
        Returns:
            Quality metrics
        """
        metrics = {
            'total_conversations': len(conversations),
            'avg_turns': np.mean([len(conv) for conv in conversations]) if conversations else 0,
            'completion_rate': 0.0,  # Would track successful order completions
            'satisfaction_score': 0.0  # Would use feedback data
        }
        
        logger.info(
            f"Conversation quality - Total: {metrics['total_conversations']}, "
            f"Avg turns: {metrics['avg_turns']:.1f}"
        )
        
        return metrics
    
    def calculate_cost(
        self,
        total_tokens: int,
        cost_per_1k_tokens: float = 0.002  # GPT-3.5-turbo pricing
    ) -> float:
        """
        Calculate conversation cost.
        
        Args:
            total_tokens: Total tokens used
            cost_per_1k_tokens: Cost per 1000 tokens
            
        Returns:
            Total cost in dollars
        """
        cost = (total_tokens / 1000) * cost_per_1k_tokens
        logger.info(f"Cost: ${cost:.4f} for {total_tokens} tokens")
        return cost
    
    def run_evaluation(
        self,
        test_conversations: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Run complete evaluation suite.
        
        Args:
            test_conversations: List of test conversation data
            
        Returns:
            Comprehensive evaluation results
        """
        logger.info(f"Running evaluation on {len(test_conversations)} conversations")
        
        # Extract data
        predicted_intents = []
        true_intents = []
        latencies = []
        context_docs = []
        total_tokens = 0
        
        for conv in test_conversations:
            if 'predicted_intent' in conv and 'true_intent' in conv:
                predicted_intents.append(conv['predicted_intent'])
                true_intents.append(conv['true_intent'])
            
            if 'latency' in conv:
                latencies.append(conv['latency'])
            
            if 'context_docs' in conv:
                context_docs.append(conv['context_docs'])
            
            if 'tokens_used' in conv:
                total_tokens += conv['tokens_used']
        
        # Calculate metrics
        results = {
            'intent_accuracy': self.evaluate_intent_accuracy(predicted_intents, true_intents) if predicted_intents else 0,
            'response_relevance': self.evaluate_response_relevance([], context_docs) if context_docs else 0,
            'latency': self.evaluate_latency(latencies) if latencies else {},
            'cost': self.calculate_cost(total_tokens),
            'total_conversations': len(test_conversations)
        }
        
        # Store for history
        self.metrics_history.append({
            'timestamp': time.time(),
            'results': results
        })
        
        logger.info("Evaluation complete")
        return results
