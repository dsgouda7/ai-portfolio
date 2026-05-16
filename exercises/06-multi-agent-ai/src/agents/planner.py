"""
Planner Agent - Task decomposition and planning
"""

from typing import Dict, Any, List
from src.agents.base import BaseAgent


class PlannerAgent(BaseAgent):
    """
    PlannerAgent breaks down complex tasks into executable steps
    """
    
    def __init__(self, name: str, config: Dict[str, Any]):
        super().__init__(name, config)
        self.max_iterations = config.get("max_iterations", 5)
    
    def process(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Decompose task into subtasks
        
        Args:
            task: Task with 'description' and optional 'constraints'
            
        Returns:
            Dictionary with execution plan and subtasks
        """
        description = task.get("description", "")
        constraints = task.get("constraints", {})
        
        self.logger.info(f"Planning task: {description}")
        
        # Decompose task into steps
        subtasks = self._decompose_task(description, constraints)
        
        # Create execution plan
        plan = self._create_execution_plan(subtasks)
        
        # Estimate complexity and duration
        complexity = self._estimate_complexity(subtasks)
        
        result = {
            "status": "success",
            "plan": plan,
            "subtasks": subtasks,
            "complexity": complexity,
            "total_steps": len(subtasks),
            "estimated_duration": complexity * 10  # seconds
        }
        
        self.logger.info(f"Created plan with {len(subtasks)} subtasks")
        return result
    
    def _decompose_task(self, description: str, constraints: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Decompose task into subtasks
        
        In a real system, this would use LLM to intelligently break down tasks.
        For this implementation, we use rule-based decomposition.
        """
        subtasks = []
        
        # Example decomposition logic
        if "research" in description.lower():
            subtasks.append({
                "id": 1,
                "type": "research",
                "description": f"Research: {description}",
                "agent": "researcher_agent",
                "dependencies": []
            })
        
        if "analyze" in description.lower() or "evaluate" in description.lower():
            subtasks.append({
                "id": len(subtasks) + 1,
                "type": "analysis",
                "description": f"Analyze results for: {description}",
                "agent": "critic_agent",
                "dependencies": [t["id"] for t in subtasks]
            })
        
        # Default execution step
        subtasks.append({
            "id": len(subtasks) + 1,
            "type": "execution",
            "description": f"Execute: {description}",
            "agent": "executor_agent",
            "dependencies": [t["id"] for t in subtasks if t["type"] == "research"]
        })
        
        # Add validation step
        subtasks.append({
            "id": len(subtasks) + 1,
            "type": "validation",
            "description": f"Validate results for: {description}",
            "agent": "critic_agent",
            "dependencies": [t["id"] for t in subtasks if t["type"] == "execution"]
        })
        
        return subtasks
    
    def _create_execution_plan(self, subtasks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create execution plan from subtasks"""
        # Topological sort for dependency order
        execution_order = self._topological_sort(subtasks)
        
        return {
            "execution_order": execution_order,
            "parallel_groups": self._identify_parallel_tasks(subtasks),
            "critical_path": self._find_critical_path(subtasks)
        }
    
    def _topological_sort(self, subtasks: List[Dict[str, Any]]) -> List[int]:
        """Sort tasks by dependencies"""
        # Simple topological sort
        sorted_ids = []
        remaining = subtasks.copy()
        
        while remaining:
            # Find tasks with no unmet dependencies
            ready = [
                t for t in remaining
                if all(dep in sorted_ids for dep in t["dependencies"])
            ]
            
            if not ready:
                # Circular dependency detected
                self.logger.warning("Circular dependency detected, breaking cycle")
                ready = [remaining[0]]
            
            for task in ready:
                sorted_ids.append(task["id"])
                remaining.remove(task)
        
        return sorted_ids
    
    def _identify_parallel_tasks(self, subtasks: List[Dict[str, Any]]) -> List[List[int]]:
        """Identify tasks that can run in parallel"""
        parallel_groups = []
        processed = set()
        
        for task in subtasks:
            if task["id"] in processed:
                continue
            
            # Find tasks with same dependencies (can run in parallel)
            group = [
                t["id"] for t in subtasks
                if t["dependencies"] == task["dependencies"] and t["id"] not in processed
            ]
            
            if len(group) > 1:
                parallel_groups.append(group)
                processed.update(group)
        
        return parallel_groups
    
    def _find_critical_path(self, subtasks: List[Dict[str, Any]]) -> List[int]:
        """Find critical path through task graph"""
        # Simple implementation: longest dependency chain
        def get_depth(task_id, depth_cache):
            if task_id in depth_cache:
                return depth_cache[task_id]
            
            task = next(t for t in subtasks if t["id"] == task_id)
            if not task["dependencies"]:
                depth = 1
            else:
                depth = 1 + max(get_depth(dep, depth_cache) for dep in task["dependencies"])
            
            depth_cache[task_id] = depth
            return depth
        
        depth_cache = {}
        max_depth_task = max(subtasks, key=lambda t: get_depth(t["id"], depth_cache))
        
        # Trace back to find critical path
        critical_path = [max_depth_task["id"]]
        current = max_depth_task
        
        while current["dependencies"]:
            # Find dependency with max depth
            next_task = max(
                (t for t in subtasks if t["id"] in current["dependencies"]),
                key=lambda t: depth_cache[t["id"]]
            )
            critical_path.insert(0, next_task["id"])
            current = next_task
        
        return critical_path
    
    def _estimate_complexity(self, subtasks: List[Dict[str, Any]]) -> float:
        """Estimate task complexity (0-1 scale)"""
        base_complexity = len(subtasks) / self.max_iterations
        dependency_factor = sum(len(t["dependencies"]) for t in subtasks) / (len(subtasks) + 1)
        
        complexity = min(1.0, (base_complexity + dependency_factor) / 2)
        return round(complexity, 2)
