"""
Executor Agent - Execute atomic tasks
"""

import time
from typing import Dict, Any
from src.agents.base import BaseAgent


class ExecutorAgent(BaseAgent):
    """
    ExecutorAgent executes atomic tasks and returns results
    """
    
    def __init__(self, name: str, config: Dict[str, Any]):
        super().__init__(name, config)
        self.max_parallel_tasks = config.get("max_parallel_tasks", 3)
        self.active_tasks = []
    
    def process(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute atomic task
        
        Args:
            task: Task with 'description', 'type', and optional 'input_data'
            
        Returns:
            Dictionary with execution results
        """
        task_type = task.get("type", "execution")
        description = task.get("description", "")
        input_data = task.get("input_data", {})
        
        self.logger.info(f"Executing task: {description}")
        
        start_time = time.time()
        
        # Execute based on task type
        if task_type == "execution":
            result = self._execute_generic_task(description, input_data)
        elif task_type == "computation":
            result = self._execute_computation(input_data)
        elif task_type == "data_processing":
            result = self._execute_data_processing(input_data)
        else:
            result = {"error": f"Unknown task type: {task_type}"}
        
        execution_time = time.time() - start_time
        
        response = {
            "status": "success" if "error" not in result else "failed",
            "result": result,
            "execution_time": round(execution_time, 3),
            "task_type": task_type
        }
        
        self.logger.info(f"Task completed in {execution_time:.3f}s")
        return response
    
    def _execute_generic_task(self, description: str, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute generic task
        
        In production, this would interface with actual execution systems.
        For demonstration, we simulate execution with mock results.
        """
        # Simulate task execution
        time.sleep(0.1)  # Simulate work
        
        return {
            "description": description,
            "output": f"Completed: {description}",
            "input_processed": len(str(input_data)),
            "success": True
        }
    
    def _execute_computation(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute computational task"""
        try:
            # Example: simple arithmetic operations
            operation = input_data.get("operation", "sum")
            values = input_data.get("values", [])
            
            if operation == "sum":
                result = sum(values)
            elif operation == "product":
                result = 1
                for v in values:
                    result *= v
            elif operation == "average":
                result = sum(values) / len(values) if values else 0
            else:
                return {"error": f"Unknown operation: {operation}"}
            
            return {
                "operation": operation,
                "result": result,
                "input_count": len(values)
            }
        except Exception as e:
            return {"error": str(e)}
    
    def _execute_data_processing(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute data processing task"""
        try:
            # Example: data transformation
            data = input_data.get("data", [])
            transform = input_data.get("transform", "identity")
            
            if transform == "uppercase":
                processed = [str(item).upper() for item in data]
            elif transform == "lowercase":
                processed = [str(item).lower() for item in data]
            elif transform == "reverse":
                processed = list(reversed(data))
            else:
                processed = data
            
            return {
                "processed_data": processed,
                "count": len(processed),
                "transform_applied": transform
            }
        except Exception as e:
            return {"error": str(e)}
    
    def can_accept_task(self) -> bool:
        """Check if agent can accept more tasks"""
        return len(self.active_tasks) < self.max_parallel_tasks
    
    def get_active_tasks(self) -> list:
        """Get list of active tasks"""
        return self.active_tasks.copy()
