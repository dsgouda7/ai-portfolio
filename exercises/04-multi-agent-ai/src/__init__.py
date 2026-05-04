"""
Multi-Agent AI System Package
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A production-ready multi-agent coordination system with task planning,
execution, and monitoring capabilities.
"""

from src.agents.planner import PlannerAgent
from src.agents.executor import ExecutorAgent
from src.agents.critic import CriticAgent
from src.agents.researcher import ResearcherAgent
from src.coordinator import AgentCoordinator
from src.messaging import MessageQueue

__version__ = "1.0.0"
__all__ = [
    "PlannerAgent",
    "ExecutorAgent",
    "CriticAgent",
    "ResearcherAgent",
    "AgentCoordinator",
    "MessageQueue",
]
