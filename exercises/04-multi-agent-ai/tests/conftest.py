"""
Test configuration and fixtures
"""

import pytest
import yaml
from src.agents.planner import PlannerAgent
from src.agents.executor import ExecutorAgent
from src.agents.critic import CriticAgent
from src.agents.researcher import ResearcherAgent
from src.coordinator import AgentCoordinator
from src.messaging import MessageQueue
from src.evaluate import MultiAgentEvaluator
from src.monitoring import MultiAgentMonitoring


@pytest.fixture
def config():
    """Load test configuration"""
    with open("config.yaml", 'r') as f:
        return yaml.safe_load(f)


@pytest.fixture
def planner_agent(config):
    """Create planner agent fixture"""
    return PlannerAgent("planner_agent", config["agents"]["planner_agent"])


@pytest.fixture
def executor_agent(config):
    """Create executor agent fixture"""
    return ExecutorAgent("executor_agent", config["agents"]["executor_agent"])


@pytest.fixture
def critic_agent(config):
    """Create critic agent fixture"""
    return CriticAgent("critic_agent", config["agents"]["critic_agent"])


@pytest.fixture
def researcher_agent(config):
    """Create researcher agent fixture"""
    return ResearcherAgent("researcher_agent", config["agents"]["researcher_agent"])


@pytest.fixture
def all_agents(planner_agent, executor_agent, critic_agent, researcher_agent):
    """Create all agents fixture"""
    return {
        "planner_agent": planner_agent,
        "executor_agent": executor_agent,
        "critic_agent": critic_agent,
        "researcher_agent": researcher_agent
    }


@pytest.fixture
def coordinator(all_agents, config):
    """Create coordinator fixture"""
    return AgentCoordinator(all_agents, config["coordination"])


@pytest.fixture
def message_queue(config):
    """Create message queue fixture"""
    queue = MessageQueue(config["communication"])
    queue.connect()
    return queue


@pytest.fixture
def evaluator():
    """Create evaluator fixture"""
    return MultiAgentEvaluator()


@pytest.fixture
def monitoring():
    """Create monitoring fixture"""
    return MultiAgentMonitoring(metrics_prefix="test_multiagent")


@pytest.fixture
def sample_task():
    """Sample task for testing"""
    return {
        "description": "Research and analyze machine learning best practices",
        "constraints": {
            "time_limit": 60,
            "quality_threshold": 0.8
        }
    }


@pytest.fixture
def sample_execution_result():
    """Sample execution result for testing"""
    return {
        "status": "success",
        "result": {
            "output": "Task completed successfully",
            "data": [1, 2, 3, 4, 5]
        },
        "execution_time": 0.5
    }
