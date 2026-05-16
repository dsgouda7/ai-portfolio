"""
Agent Coordinator - Orchestrate multi-agent workflows
"""

import time
from typing import Dict, Any, List, Optional
from src.agents.base import BaseAgent
from src.utils import get_logger, TaskStatus, StateManager, create_message, MessageType


class AgentCoordinator:
    """
    Coordinates multiple agents to accomplish complex tasks
    """
    
    def __init__(self, agents: Dict[str, BaseAgent], config: Dict[str, Any]):
        """
        Initialize coordinator
        
        Args:
            agents: Dictionary mapping agent names to agent instances
            config: Coordinator configuration
        """
        self.agents = agents
        self.config = config
        self.logger = get_logger("coordinator")
        self.state_manager = StateManager()
        
        self.max_concurrent = config.get("max_concurrent_agents", 4)
        self.task_timeout = config.get("task_timeout", 300)
        
        self.active_workflows = {}
        
        self.logger.info(f"Coordinator initialized with {len(agents)} agents")
    
    def execute_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute task using coordinated agents
        
        Args:
            task: Task with 'description' and optional 'workflow'
            
        Returns:
            Task execution results
        """
        task_id = f"task_{int(time.time() * 1000)}"
        description = task.get("description", "")
        
        self.logger.info(f"Executing task {task_id}: {description}")
        
        # Initialize task state
        self.state_manager.update(f"{task_id}_status", TaskStatus.IN_PROGRESS.value)
        self.state_manager.update(f"{task_id}_start_time", time.time())
        
        try:
            # Step 1: Plan task
            plan = self._plan_task(task)
            
            # Step 2: Research if needed
            if self._needs_research(plan):
                research = self._research_task(task)
                plan["context"] = research
            
            # Step 3: Execute subtasks
            results = self._execute_plan(plan)
            
            # Step 4: Evaluate results
            evaluation = self._evaluate_results(results)
            
            # Step 5: Handle feedback loop if needed
            if not evaluation.get("passes_threshold", True):
                results = self._refine_results(results, evaluation)
            
            execution_time = time.time() - self.state_manager.get(f"{task_id}_start_time")
            
            # Update final state
            self.state_manager.update(f"{task_id}_status", TaskStatus.COMPLETED.value)
            
            response = {
                "task_id": task_id,
                "status": "success",
                "plan": plan,
                "results": results,
                "evaluation": evaluation,
                "execution_time": round(execution_time, 2),
                "agents_used": list(set(r.get("agent", "") for r in results))
            }
            
            self.logger.info(f"Task {task_id} completed successfully")
            return response
            
        except Exception as e:
            self.logger.error(f"Task {task_id} failed: {str(e)}")
            self.state_manager.update(f"{task_id}_status", TaskStatus.FAILED.value)
            
            return {
                "task_id": task_id,
                "status": "failed",
                "error": str(e)
            }
    
    def _plan_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Use planner agent to decompose task"""
        planner = self.agents.get("planner_agent")
        if not planner:
            raise ValueError("Planner agent not available")
        
        message = create_message(
            msg_type=MessageType.TASK_REQUEST,
            sender="coordinator",
            receiver="planner_agent",
            content=task
        )
        
        response = planner.receive_message(message)
        return response.get("content", {})
    
    def _needs_research(self, plan: Dict[str, Any]) -> bool:
        """Determine if task needs research phase"""
        subtasks = plan.get("subtasks", [])
        return any(t.get("type") == "research" for t in subtasks)
    
    def _research_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Use researcher agent to gather information"""
        researcher = self.agents.get("researcher_agent")
        if not researcher:
            return {"summary": "Research agent not available"}
        
        research_task = {
            "query": task.get("description", ""),
            "depth": 2
        }
        
        message = create_message(
            msg_type=MessageType.TASK_REQUEST,
            sender="coordinator",
            receiver="researcher_agent",
            content=research_task
        )
        
        response = researcher.receive_message(message)
        return response.get("content", {})
    
    def _execute_plan(self, plan: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Execute plan subtasks in order"""
        subtasks = plan.get("subtasks", [])
        execution_order = plan.get("plan", {}).get("execution_order", [])
        
        results = []
        completed_tasks = {}
        
        for task_id in execution_order:
            subtask = next((t for t in subtasks if t["id"] == task_id), None)
            if not subtask:
                continue
            
            # Check dependencies
            dependencies_met = all(
                dep in completed_tasks for dep in subtask.get("dependencies", [])
            )
            
            if not dependencies_met:
                self.logger.warning(f"Dependencies not met for task {task_id}")
                continue
            
            # Execute subtask
            result = self._execute_subtask(subtask, completed_tasks)
            results.append(result)
            completed_tasks[task_id] = result
        
        return results
    
    def _execute_subtask(
        self, subtask: Dict[str, Any], completed_tasks: Dict[int, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Execute single subtask"""
        agent_name = subtask.get("agent", "executor_agent")
        agent = self.agents.get(agent_name)
        
        if not agent:
            return {
                "subtask_id": subtask["id"],
                "agent": agent_name,
                "status": "failed",
                "error": f"Agent {agent_name} not available"
            }
        
        # Prepare task with dependency results
        task_data = {
            "description": subtask["description"],
            "type": subtask.get("type", "execution"),
            "dependencies": {
                dep_id: completed_tasks.get(dep_id, {})
                for dep_id in subtask.get("dependencies", [])
            }
        }
        
        message = create_message(
            msg_type=MessageType.TASK_REQUEST,
            sender="coordinator",
            receiver=agent_name,
            content=task_data
        )
        
        response = agent.receive_message(message)
        result = response.get("content", {})
        result["subtask_id"] = subtask["id"]
        result["agent"] = agent_name
        
        return result
    
    def _evaluate_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Use critic agent to evaluate results"""
        critic = self.agents.get("critic_agent")
        if not critic:
            return {"passes_threshold": True, "score": 0.8}
        
        evaluation_task = {
            "result": {
                "results": results,
                "total_subtasks": len(results),
                "failed_subtasks": sum(1 for r in results if r.get("status") == "failed")
            }
        }
        
        message = create_message(
            msg_type=MessageType.TASK_REQUEST,
            sender="coordinator",
            receiver="critic_agent",
            content=evaluation_task
        )
        
        response = critic.receive_message(message)
        return response.get("content", {})
    
    def _refine_results(
        self, results: List[Dict[str, Any]], evaluation: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Refine results based on feedback"""
        self.logger.info("Refining results based on feedback")
        
        # In production, this would re-execute failed/low-quality tasks
        # For now, we just log the feedback
        feedback = evaluation.get("feedback", {})
        improvements = feedback.get("improvements", [])
        
        for imp in improvements:
            self.logger.info(f"Improvement needed: {imp.get('metric')} - {imp.get('suggestion')}")
        
        return results
    
    def get_workflow_status(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        """Get status of active workflow"""
        return self.active_workflows.get(workflow_id)
    
    def list_active_workflows(self) -> List[str]:
        """List all active workflow IDs"""
        return list(self.active_workflows.keys())
    
    def get_agent_status(self) -> Dict[str, Any]:
        """Get status of all agents"""
        return {
            name: agent.get_status()
            for name, agent in self.agents.items()
        }
