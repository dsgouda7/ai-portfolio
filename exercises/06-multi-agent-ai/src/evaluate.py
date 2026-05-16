"""
Evaluation metrics for multi-agent system
"""

from typing import Dict, Any, List
from datetime import datetime
from src.utils import get_logger, TaskStatus


class MultiAgentEvaluator:
    """
    Evaluate multi-agent system performance
    """
    
    def __init__(self):
        self.logger = get_logger("evaluator")
        self.metrics_history = []
    
    def evaluate_task_execution(
        self, task_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Evaluate task execution performance
        
        Args:
            task_result: Task execution result from coordinator
            
        Returns:
            Dictionary with evaluation metrics
        """
        # Calculate metrics
        completion_rate = self._calculate_completion_rate(task_result)
        coordination_efficiency = self._calculate_coordination_efficiency(task_result)
        avg_time = self._calculate_average_time(task_result)
        agent_utilization = self._calculate_agent_utilization(task_result)
        
        metrics = {
            "task_id": task_result.get("task_id"),
            "timestamp": datetime.now().isoformat(),
            "completion_rate": completion_rate,
            "coordination_efficiency": coordination_efficiency,
            "average_execution_time": avg_time,
            "agent_utilization": agent_utilization,
            "total_execution_time": task_result.get("execution_time", 0)
        }
        
        # Store in history
        self.metrics_history.append(metrics)
        
        # Log summary
        self.logger.info(
            f"Task {metrics['task_id']}: "
            f"completion={completion_rate:.2f}, "
            f"efficiency={coordination_efficiency:.2f}"
        )
        
        return metrics
    
    def _calculate_completion_rate(self, task_result: Dict[str, Any]) -> float:
        """
        Calculate task completion rate
        
        Returns value between 0 and 1
        """
        results = task_result.get("results", [])
        
        if not results:
            return 0.0
        
        completed = sum(
            1 for r in results
            if r.get("status") == "success" or r.get("status") == "completed"
        )
        
        return completed / len(results)
    
    def _calculate_coordination_efficiency(self, task_result: Dict[str, Any]) -> float:
        """
        Calculate coordination efficiency
        
        Measures how efficiently agents coordinated:
        - Parallel execution utilization
        - Message overhead
        - Dependency management
        
        Returns value between 0 and 1
        """
        plan = task_result.get("plan", {})
        results = task_result.get("results", [])
        
        if not results:
            return 0.0
        
        # Factor 1: Parallel execution utilization
        parallel_groups = plan.get("plan", {}).get("parallel_groups", [])
        total_tasks = len(results)
        parallelizable_tasks = sum(len(group) for group in parallel_groups)
        parallel_utilization = parallelizable_tasks / (total_tasks + 1)
        
        # Factor 2: Successful task ratio
        success_ratio = self._calculate_completion_rate(task_result)
        
        # Factor 3: Time efficiency (compared to sequential execution)
        total_time = task_result.get("execution_time", 0)
        estimated_sequential_time = sum(
            r.get("execution_time", 0) for r in results
        )
        
        if estimated_sequential_time > 0:
            time_efficiency = min(1.0, estimated_sequential_time / (total_time + 0.1))
        else:
            time_efficiency = 0.5
        
        # Weighted average
        efficiency = (
            0.3 * parallel_utilization +
            0.4 * success_ratio +
            0.3 * time_efficiency
        )
        
        return round(efficiency, 2)
    
    def _calculate_average_time(self, task_result: Dict[str, Any]) -> float:
        """Calculate average execution time per subtask"""
        results = task_result.get("results", [])
        
        if not results:
            return 0.0
        
        total_time = sum(r.get("execution_time", 0) for r in results)
        return round(total_time / len(results), 3)
    
    def _calculate_agent_utilization(self, task_result: Dict[str, Any]) -> Dict[str, float]:
        """
        Calculate utilization for each agent type
        
        Returns dictionary mapping agent names to utilization scores
        """
        results = task_result.get("results", [])
        agents_used = task_result.get("agents_used", [])
        
        utilization = {}
        
        for agent in agents_used:
            agent_tasks = [r for r in results if r.get("agent") == agent]
            
            if agent_tasks:
                success_rate = sum(
                    1 for t in agent_tasks if t.get("status") == "success"
                ) / len(agent_tasks)
                
                avg_time = sum(
                    t.get("execution_time", 0) for t in agent_tasks
                ) / len(agent_tasks)
                
                # Lower time is better (more efficient)
                time_score = max(0, 1 - (avg_time / 10))
                
                utilization[agent] = round((success_rate + time_score) / 2, 2)
            else:
                utilization[agent] = 0.0
        
        return utilization
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """
        Get aggregate system metrics across all tasks
        
        Returns:
            Dictionary with system-wide metrics
        """
        if not self.metrics_history:
            return {
                "total_tasks": 0,
                "avg_completion_rate": 0.0,
                "avg_coordination_efficiency": 0.0,
                "avg_execution_time": 0.0
            }
        
        total_tasks = len(self.metrics_history)
        
        avg_completion = sum(
            m["completion_rate"] for m in self.metrics_history
        ) / total_tasks
        
        avg_efficiency = sum(
            m["coordination_efficiency"] for m in self.metrics_history
        ) / total_tasks
        
        avg_time = sum(
            m["total_execution_time"] for m in self.metrics_history
        ) / total_tasks
        
        return {
            "total_tasks": total_tasks,
            "avg_completion_rate": round(avg_completion, 2),
            "avg_coordination_efficiency": round(avg_efficiency, 2),
            "avg_execution_time": round(avg_time, 2),
            "recent_tasks": self.metrics_history[-10:]  # Last 10 tasks
        }
    
    def check_success_criteria(self) -> Dict[str, Any]:
        """
        Check if system meets success criteria
        
        Success criteria:
        - Task completion rate > 85%
        - Coordination efficiency > 0.7
        
        Returns:
            Dictionary with pass/fail status and details
        """
        system_metrics = self.get_system_metrics()
        
        completion_rate = system_metrics["avg_completion_rate"]
        efficiency = system_metrics["avg_coordination_efficiency"]
        
        completion_pass = completion_rate > 0.85
        efficiency_pass = efficiency > 0.7
        
        overall_pass = completion_pass and efficiency_pass
        
        return {
            "overall_pass": overall_pass,
            "completion_rate": {
                "value": completion_rate,
                "threshold": 0.85,
                "pass": completion_pass
            },
            "coordination_efficiency": {
                "value": efficiency,
                "threshold": 0.7,
                "pass": efficiency_pass
            },
            "total_tasks_evaluated": system_metrics["total_tasks"]
        }
    
    def reset_metrics(self) -> None:
        """Reset metrics history"""
        self.metrics_history = []
        self.logger.info("Metrics history reset")
