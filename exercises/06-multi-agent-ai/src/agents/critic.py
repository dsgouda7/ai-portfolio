"""
Critic Agent - Evaluate results and provide feedback
"""

from typing import Dict, Any, List
from src.agents.base import BaseAgent


class CriticAgent(BaseAgent):
    """
    CriticAgent evaluates task results and provides constructive feedback
    """
    
    def __init__(self, name: str, config: Dict[str, Any]):
        super().__init__(name, config)
        self.threshold_score = config.get("threshold_score", 0.7)
    
    def process(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate task results
        
        Args:
            task: Task with 'result' to evaluate and optional 'criteria'
            
        Returns:
            Dictionary with evaluation score and feedback
        """
        result = task.get("result", {})
        criteria = task.get("criteria", {})
        
        self.logger.info("Evaluating task result")
        
        # Evaluate result
        evaluation = self._evaluate_result(result, criteria)
        
        # Generate feedback
        feedback = self._generate_feedback(evaluation)
        
        # Determine if result meets threshold
        passes = evaluation["score"] >= self.threshold_score
        
        response = {
            "status": "success",
            "evaluation": evaluation,
            "feedback": feedback,
            "passes_threshold": passes,
            "threshold": self.threshold_score
        }
        
        self.logger.info(f"Evaluation complete: score={evaluation['score']}, passes={passes}")
        return response
    
    def _evaluate_result(self, result: Dict[str, Any], criteria: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate result against criteria
        
        In production, this would use sophisticated evaluation metrics.
        For demonstration, we use rule-based evaluation.
        """
        scores = {}
        
        # Completeness check
        scores["completeness"] = self._check_completeness(result)
        
        # Quality check
        scores["quality"] = self._check_quality(result)
        
        # Accuracy check (if criteria provided)
        if criteria:
            scores["accuracy"] = self._check_accuracy(result, criteria)
        else:
            scores["accuracy"] = 0.8  # Default if no criteria
        
        # Efficiency check
        scores["efficiency"] = self._check_efficiency(result)
        
        # Calculate weighted average
        weights = {
            "completeness": 0.3,
            "quality": 0.3,
            "accuracy": 0.25,
            "efficiency": 0.15
        }
        
        overall_score = sum(scores[k] * weights[k] for k in scores)
        
        return {
            "score": round(overall_score, 2),
            "detailed_scores": scores,
            "weights": weights
        }
    
    def _check_completeness(self, result: Dict[str, Any]) -> float:
        """Check if result is complete"""
        # Check for required fields
        required_fields = ["status", "result"]
        present_fields = sum(1 for f in required_fields if f in result)
        
        # Check for errors
        has_error = "error" in result
        
        score = present_fields / len(required_fields)
        if has_error:
            score *= 0.5
        
        return min(1.0, score)
    
    def _check_quality(self, result: Dict[str, Any]) -> float:
        """Check result quality"""
        quality_score = 0.8  # Base quality
        
        # Deduct for errors
        if "error" in result:
            quality_score -= 0.3
        
        # Bonus for detailed output
        if isinstance(result.get("result"), dict):
            output_keys = len(result["result"])
            if output_keys > 3:
                quality_score += 0.1
        
        return max(0.0, min(1.0, quality_score))
    
    def _check_accuracy(self, result: Dict[str, Any], criteria: Dict[str, Any]) -> float:
        """Check result accuracy against criteria"""
        # Simple criteria matching
        matches = 0
        total = len(criteria)
        
        for key, expected in criteria.items():
            actual = result.get("result", {}).get(key)
            if actual == expected:
                matches += 1
            elif isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
                # Numeric tolerance
                if abs(actual - expected) / (expected + 1e-10) < 0.1:
                    matches += 0.8
        
        return matches / total if total > 0 else 1.0
    
    def _check_efficiency(self, result: Dict[str, Any]) -> float:
        """Check execution efficiency"""
        execution_time = result.get("execution_time", 0)
        
        # Efficiency based on execution time
        if execution_time < 0.1:
            return 1.0
        elif execution_time < 1.0:
            return 0.9
        elif execution_time < 5.0:
            return 0.7
        else:
            return 0.5
    
    def _generate_feedback(self, evaluation: Dict[str, Any]) -> Dict[str, Any]:
        """Generate constructive feedback"""
        score = evaluation["score"]
        detailed_scores = evaluation["detailed_scores"]
        
        # Identify areas for improvement
        improvements = []
        strengths = []
        
        for metric, value in detailed_scores.items():
            if value < 0.7:
                improvements.append({
                    "metric": metric,
                    "score": value,
                    "suggestion": self._get_improvement_suggestion(metric, value)
                })
            elif value > 0.85:
                strengths.append(metric)
        
        # Overall feedback
        if score >= 0.9:
            summary = "Excellent work! Result exceeds expectations."
        elif score >= 0.7:
            summary = "Good result. Minor improvements possible."
        elif score >= 0.5:
            summary = "Acceptable result. Several areas need improvement."
        else:
            summary = "Result needs significant improvement."
        
        return {
            "summary": summary,
            "strengths": strengths,
            "improvements": improvements,
            "next_steps": self._suggest_next_steps(improvements)
        }
    
    def _get_improvement_suggestion(self, metric: str, score: float) -> str:
        """Get specific improvement suggestion for metric"""
        suggestions = {
            "completeness": "Ensure all required fields are populated and no steps are skipped.",
            "quality": "Add more detailed output and error handling. Validate intermediate results.",
            "accuracy": "Review calculation logic and verify against expected criteria.",
            "efficiency": "Optimize execution path and reduce redundant operations."
        }
        return suggestions.get(metric, "Review and improve this aspect.")
    
    def _suggest_next_steps(self, improvements: List[Dict[str, Any]]) -> List[str]:
        """Suggest concrete next steps"""
        if not improvements:
            return ["Maintain current quality standards", "Consider optimizations"]
        
        steps = []
        for imp in improvements[:3]:  # Top 3 improvements
            steps.append(f"Address {imp['metric']}: {imp['suggestion']}")
        
        return steps
