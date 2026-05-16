"""Multi-agent system with experiment framework

This module provides:
- Abstract Agent base class for plug-and-play agents
- Concrete implementations: CoordinatorAgent, WorkerAgent, ResearchAgent (with TODOs)
- ExperimentRunner for multi-agent task orchestration
- Immediate feedback with rich console output

Learning objectives:
1. Implement agent communication with message passing
2. Build task decomposition and coordination logic
3. Manage shared state across agents
4. Experiment with different agent configurations
5. See coordination results immediately after each task
"""

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
from enum import Enum

import json
from rich.console import Console
from rich.table import Table

logger = logging.getLogger("multi_agent")
console = Console()


class AgentRole(Enum):
    """Agent role types in the multi-agent system."""
    COORDINATOR = "coordinator"
    WORKER = "worker"
    RESEARCHER = "researcher"


@dataclass
class AgentConfig:
    """Configuration for agent behavior."""
    max_iterations: int = 10
    timeout_seconds: int = 30
    verbose: bool = True
    enable_logging: bool = True


@dataclass
class Message:
    """Message structure for agent communication."""
    sender: str
    recipient: str
    content: Dict[str, Any]
    timestamp: float
    message_type: str  # "request", "response", "broadcast"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary."""
        return {
            "sender": self.sender,
            "recipient": self.recipient,
            "content": self.content,
            "timestamp": self.timestamp,
            "message_type": self.message_type
        }


class Agent(ABC):
    """Abstract base class for all agents.
    
    Provides common interface for plug-and-play experimentation.
    Subclasses implement process_task() and respond_to_message() methods.
    
    All agents have:
    - Unique name for identification
    - Role (coordinator, worker, researcher)
    - Message inbox/outbox
    - Shared state access
    """
    
    def __init__(self, name: str, role: AgentRole):
        """Initialize agent with name and role.
        
        Args:
            name: Unique identifier for this agent
            role: Agent role type
        """
        self.name = name
        self.role = role
        self.inbox: List[Message] = []
        self.outbox: List[Message] = []
        self.state: Dict[str, Any] = {}
        self.metrics: Dict[str, Any] = {
            "messages_sent": 0,
            "messages_received": 0,
            "tasks_completed": 0,
            "errors": 0
        }
    
    @abstractmethod
    def process_task(self, task: Dict[str, Any], config: AgentConfig) -> Dict[str, Any]:
        """Process a task and return results with immediate console feedback.
        
        Args:
            task: Task description with type, parameters, etc.
            config: Agent configuration
        
        Returns:
            Dictionary with results: {"status": str, "result": Any, "metrics": Dict}
        """
        pass
    
    @abstractmethod
    def respond_to_message(self, message: Message) -> Optional[Message]:
        """Respond to incoming message from another agent.
        
        Args:
            message: Incoming message
        
        Returns:
            Response message or None
        """
        pass
    
    def send_message(self, recipient: str, content: Dict[str, Any], message_type: str = "request"):
        """Send message to another agent.
        
        Args:
            recipient: Name of recipient agent
            content: Message content
            message_type: Type of message (request/response/broadcast)
        """
        msg = Message(
            sender=self.name,
            recipient=recipient,
            content=content,
            timestamp=time.time(),
            message_type=message_type
        )
        self.outbox.append(msg)
        self.metrics["messages_sent"] += 1
    
    def receive_message(self, message: Message):
        """Receive message from another agent.
        
        Args:
            message: Incoming message
        """
        self.inbox.append(message)
        self.metrics["messages_received"] += 1
    
    def update_state(self, key: str, value: Any):
        """Update agent's internal state.
        
        Args:
            key: State key
            value: State value
        """
        self.state[key] = value
    
    def get_status(self) -> Dict[str, Any]:
        """Get agent status and metrics."""
        return {
            "name": self.name,
            "role": self.role.value,
            "inbox_size": len(self.inbox),
            "outbox_size": len(self.outbox),
            "state_keys": list(self.state.keys()),
            "metrics": self.metrics
        }


class CoordinatorAgent(Agent):
    """Coordinator agent for task decomposition and orchestration.
    
    The Coordinator:
    - Receives high-level tasks from users
    - Breaks them down into subtasks
    - Assigns subtasks to worker agents
    - Monitors progress and handles failures
    - Aggregates results
    """
    
    def __init__(self, name: str = "coordinator"):
        """Initialize coordinator agent."""
        super().__init__(name, AgentRole.COORDINATOR)
        self.worker_assignments: Dict[str, List[Dict]] = {}
    
    def process_task(self, task: Dict[str, Any], config: AgentConfig) -> Dict[str, Any]:
        """
        TODO: Decompose task into subtasks, assign to workers round-robin, track assignments
        
        📖 See: notes/04-multi_agent_ai/ch03_a2a/ (task delegation & lifecycle)
                notes/04-multi_agent_ai/ch04_event_driven_agents/ (orchestrator pattern)
        ⚡ Advances constraint #4 SCALABILITY (10 agents/PO without coordinator bottleneck)
        """
        # TODO: Your implementation here
        raise NotImplementedError("Implement coordinator task processing")
    
    def respond_to_message(self, message: Message) -> Optional[Message]:
        """
        TODO: Extract result from worker, update state, check completion, send acknowledgment
        
        📖 See: notes/04-multi_agent_ai/ch01_message_formats/ (message passing)
                notes/04-multi_agent_ai/ch03_a2a/ (task status tracking)
        ⚡ Advances constraint #4 SCALABILITY (coordinator aggregates results without blocking)
        """
        # TODO: Your implementation here
        raise NotImplementedError("Implement coordinator message handling")
    
    def _decompose_task(self, task: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Decompose high-level task into subtasks.
        
        This is a simplified implementation. In production, you'd use
        LLM-based planning or graph-based task decomposition.
        """
        description = task.get("description", "")
        complexity = task.get("complexity", "medium")
        
        # Simple decomposition based on complexity
        if complexity == "low":
            return [
                {"type": "execute", "description": f"Complete: {description}", "priority": "high"}
            ]
        elif complexity == "medium":
            return [
                {"type": "research", "description": f"Research requirements for: {description}", "priority": "high"},
                {"type": "execute", "description": f"Implement: {description}", "priority": "medium"},
                {"type": "validate", "description": f"Validate: {description}", "priority": "low"}
            ]
        else:  # high complexity
            return [
                {"type": "research", "description": f"Research requirements for: {description}", "priority": "high"},
                {"type": "design", "description": f"Design solution for: {description}", "priority": "high"},
                {"type": "execute", "description": f"Implement: {description}", "priority": "medium"},
                {"type": "test", "description": f"Test: {description}", "priority": "medium"},
                {"type": "validate", "description": f"Validate: {description}", "priority": "low"}
            ]


class WorkerAgent(Agent):
    """Worker agent for executing atomic tasks.
    
    Workers:
    - Receive subtasks from coordinator
    - Execute tasks independently
    - Report results back to coordinator
    - Handle retries on failure
    """
    
    def __init__(self, name: str):
        """Initialize worker agent."""
        super().__init__(name, AgentRole.WORKER)
        self.current_task: Optional[Dict] = None
    
    def process_task(self, task: Dict[str, Any], config: AgentConfig) -> Dict[str, Any]:
        """
        TODO: Execute task using _execute_task, track timing, update metrics
        
        📖 See: notes/04-multi_agent_ai/ch03_a2a/ (worker role in A2A delegation)
        ⚡ Advances constraint #1 THROUGHPUT (parallel worker execution enables 1,000 POs/day)
        """
        # TODO: Your implementation here
        raise NotImplementedError("Implement worker task execution")
    
    def respond_to_message(self, message: Message) -> Optional[Message]:
        """
        TODO: Extract task from message, process it, send result back to coordinator
        
        📖 See: notes/04-multi_agent_ai/ch01_message_formats/ (message extraction)
                notes/04-multi_agent_ai/ch03_a2a/ (task response protocol)
        ⚡ Advances constraint #1 THROUGHPUT (workers process subtasks asynchronously)
        """
        # TODO: Your implementation here
        raise NotImplementedError("Implement worker message handling")
    
    def _execute_task(self, task_type: str, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute specific task type.
        
        Simplified implementation. In production, this would:
        - Call external APIs
        - Run computations
        - Access databases
        - Generate content
        """
        # Simulate work with small delay
        time.sleep(0.1)
        
        if task_type == "research":
            return {"findings": f"Research complete for: {task.get('description')}", "confidence": 0.85}
        elif task_type == "execute":
            return {"output": f"Executed: {task.get('description')}", "success": True}
        elif task_type == "validate":
            return {"validation": "passed", "issues": []}
        else:
            return {"status": "completed", "details": f"Processed {task_type}"}


class ResearchAgent(Agent):
    """Research agent for information gathering and analysis.
    
    Researchers:
    - Gather information from various sources
    - Analyze data and generate insights
    - Provide context for decision-making
    - Cache research results for reuse
    """
    
    def __init__(self, name: str = "researcher"):
        """Initialize research agent."""
        super().__init__(name, AgentRole.RESEARCHER)
        self.research_cache: Dict[str, Any] = {}
    
    def process_task(self, task: Dict[str, Any], config: AgentConfig) -> Dict[str, Any]:
        """
        TODO: Check cache, conduct research using _conduct_research, cache results, update metrics
        
        📖 See: notes/04-multi_agent_ai/ch05_shared_memory/ (research caching in shared state)
        ⚡ Advances constraint #1 THROUGHPUT (caching eliminates redundant research calls)
        """
        # TODO: Your implementation here
        raise NotImplementedError("Implement research task processing")
    
    def respond_to_message(self, message: Message) -> Optional[Message]:
        """Handle research requests from other agents."""
        if "research_query" in message.content:
            query = message.content["research_query"]
            task = {"query": query, "type": "research"}
            result = self.process_task(task, AgentConfig())
            
            return Message(
                sender=self.name,
                recipient=message.sender,
                content={"research_result": result},
                timestamp=time.time(),
                message_type="response"
            )
        return None
    
    def _conduct_research(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Conduct research for given task.
        
        Simplified implementation. In production, this would:
        - Query search engines
        - Access knowledge bases
        - Call LLM APIs for analysis
        - Aggregate from multiple sources
        """
        # Simulate research with delay
        time.sleep(0.2)
        
        query = task.get("query", "")
        return {
            "query": query,
            "sources": [
                {"title": f"Source 1 for {query}", "relevance": 0.92},
                {"title": f"Source 2 for {query}", "relevance": 0.87},
                {"title": f"Source 3 for {query}", "relevance": 0.75}
            ],
            "summary": f"Research summary for: {query}",
            "confidence": 0.85
        }


class ExperimentRunner:
    """Run multi-agent experiments with different configurations.
    
    Provides framework for experimenting with agent teams:
    1. Register agents with different configurations
    2. Run collaborative tasks with immediate feedback
    3. Print coordination metrics and performance
    
    Example:
        >>> runner = ExperimentRunner()
        >>> coordinator = CoordinatorAgent("coordinator")
        >>> runner.register_agent(coordinator)
        >>> runner.register_agent(WorkerAgent("worker_0"))
        >>> runner.register_agent(WorkerAgent("worker_1"))
        >>> runner.run_experiment(task, AgentConfig())
        >>> runner.print_metrics()
    """
    
    def __init__(self):
        """Initialize empty experiment runner."""
        self.agents: Dict[str, Agent] = {}
        self.results: List[Dict[str, Any]] = []
        self.message_log: List[Message] = []
    
    def register_agent(self, agent: Agent):
        """Register an agent to participate in experiments.
        
        Args:
            agent: Agent instance to register
        """
        self.agents[agent.name] = agent
        console.print(f"Registered: {agent.name} ({agent.role.value})", style="dim")
    
    def run_experiment(self, task: Dict[str, Any], config: AgentConfig):
        """
        TODO: Get coordinator, process task, route messages between agents, aggregate results
        
        📖 See: notes/04-multi_agent_ai/ch04_event_driven_agents/ (orchestration patterns)
                notes/04-multi_agent_ai/ch07_agent_frameworks/ (AutoGen, LangGraph)
        ⚡ Advances constraint #1 THROUGHPUT (multi-agent orchestration without blocking)
        """
        # TODO: Your implementation here
        raise NotImplementedError("Implement experiment runner")
    
    def print_metrics(self):
        """
        TODO: Create performance table with agent metrics, display totals and averages
        
        📖 See: notes/04-multi_agent_ai/ch07_agent_frameworks/ (distributed tracing & observability)
        ⚡ Advances constraint #7 OBSERVABILITY (real-time monitoring of agent performance)
        """
        # TODO: Your implementation here
        raise NotImplementedError("Implement metrics printing")
    
    def get_message_flow(self) -> Dict[str, int]:
        """Analyze message flow between agents."""
        flow = {}
        for msg in self.message_log:
            key = f"{msg.sender} → {msg.recipient}"
            flow[key] = flow.get(key, 0) + 1
        return flow
    
    def print_leaderboard(self):
        """Print leaderboard of experiment results."""
        if not self.results:
            console.print("No experiments run yet", style="yellow")
            return
        
        console.print("\n[bold cyan]🏆 EXPERIMENT LEADERBOARD[/bold cyan]")
        
        table = Table(title="Task Completion Summary")
        table.add_column("Task", style="cyan")
        table.add_column("Status", justify="center")
        table.add_column("Subtasks", justify="right")
        table.add_column("Messages", justify="right")
        table.add_column("Time", justify="right")
        
        for result in self.results:
            status_emoji = "✅" if result["status"] == "distributed" else "⚠️"
            table.add_row(
                result["task"][:40] + "..." if len(result["task"]) > 40 else result["task"],
                status_emoji,
                str(result["subtasks"]),
                str(result["messages_exchanged"]),
                f"{result['elapsed_time']:.2f}s"
            )
        
        console.print(table)
