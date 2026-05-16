"""
Tests for agent implementations
"""

import pytest
from src.utils import MessageType, create_message


class TestPlannerAgent:
    """Test cases for PlannerAgent"""
    
    def test_initialization(self, planner_agent):
        """Test planner agent initialization"""
        assert planner_agent.name == "planner_agent"
        assert planner_agent.enabled is True
        assert planner_agent.max_iterations == 5
    
    def test_task_decomposition(self, planner_agent, sample_task):
        """Test task decomposition"""
        result = planner_agent.process(sample_task)
        
        assert result["status"] == "success"
        assert "subtasks" in result
        assert "plan" in result
        assert len(result["subtasks"]) > 0
    
    def test_execution_plan_creation(self, planner_agent):
        """Test execution plan creation"""
        task = {"description": "Research and execute analysis"}
        result = planner_agent.process(task)
        
        plan = result["plan"]
        assert "execution_order" in plan
        assert "parallel_groups" in plan
        assert "critical_path" in plan
    
    def test_complexity_estimation(self, planner_agent):
        """Test complexity estimation"""
        task = {"description": "Simple task"}
        result = planner_agent.process(task)
        
        assert "complexity" in result
        assert 0 <= result["complexity"] <= 1
    
    def test_message_handling(self, planner_agent, sample_task):
        """Test message handling"""
        message = create_message(
            msg_type=MessageType.TASK_REQUEST,
            sender="test",
            receiver="planner_agent",
            content=sample_task
        )
        
        response = planner_agent.receive_message(message)
        assert response is not None
        assert response["type"] == MessageType.TASK_RESPONSE.value


class TestExecutorAgent:
    """Test cases for ExecutorAgent"""
    
    def test_initialization(self, executor_agent):
        """Test executor agent initialization"""
        assert executor_agent.name == "executor_agent"
        assert executor_agent.enabled is True
        assert executor_agent.max_parallel_tasks == 3
    
    def test_generic_task_execution(self, executor_agent):
        """Test generic task execution"""
        task = {
            "description": "Execute generic task",
            "type": "execution"
        }
        
        result = executor_agent.process(task)
        
        assert result["status"] == "success"
        assert "result" in result
        assert "execution_time" in result
    
    def test_computation_task(self, executor_agent):
        """Test computation task"""
        task = {
            "type": "computation",
            "input_data": {
                "operation": "sum",
                "values": [1, 2, 3, 4, 5]
            }
        }
        
        result = executor_agent.process(task)
        
        assert result["status"] == "success"
        assert result["result"]["result"] == 15
    
    def test_data_processing_task(self, executor_agent):
        """Test data processing task"""
        task = {
            "type": "data_processing",
            "input_data": {
                "data": ["hello", "world"],
                "transform": "uppercase"
            }
        }
        
        result = executor_agent.process(task)
        
        assert result["status"] == "success"
        assert result["result"]["processed_data"] == ["HELLO", "WORLD"]
    
    def test_task_capacity(self, executor_agent):
        """Test task capacity checking"""
        assert executor_agent.can_accept_task() is True


class TestCriticAgent:
    """Test cases for CriticAgent"""
    
    def test_initialization(self, critic_agent):
        """Test critic agent initialization"""
        assert critic_agent.name == "critic_agent"
        assert critic_agent.enabled is True
        assert critic_agent.threshold_score == 0.7
    
    def test_result_evaluation(self, critic_agent, sample_execution_result):
        """Test result evaluation"""
        task = {"result": sample_execution_result}
        
        result = critic_agent.process(task)
        
        assert result["status"] == "success"
        assert "evaluation" in result
        assert "feedback" in result
        assert "passes_threshold" in result
    
    def test_score_calculation(self, critic_agent):
        """Test evaluation score calculation"""
        task = {
            "result": {
                "status": "success",
                "result": {"output": "test"},
                "execution_time": 0.1
            }
        }
        
        result = critic_agent.process(task)
        evaluation = result["evaluation"]
        
        assert "score" in evaluation
        assert 0 <= evaluation["score"] <= 1
        assert "detailed_scores" in evaluation
    
    def test_feedback_generation(self, critic_agent):
        """Test feedback generation"""
        task = {
            "result": {
                "status": "success",
                "result": {"output": "test"}
            }
        }
        
        result = critic_agent.process(task)
        feedback = result["feedback"]
        
        assert "summary" in feedback
        assert "improvements" in feedback
        assert "next_steps" in feedback


class TestResearcherAgent:
    """Test cases for ResearcherAgent"""
    
    def test_initialization(self, researcher_agent):
        """Test researcher agent initialization"""
        assert researcher_agent.name == "researcher_agent"
        assert researcher_agent.enabled is True
        assert researcher_agent.max_search_depth == 3
    
    def test_information_gathering(self, researcher_agent):
        """Test information gathering"""
        task = {
            "query": "machine learning",
            "depth": 2
        }
        
        result = researcher_agent.process(task)
        
        assert result["status"] == "success"
        assert "findings" in result
        assert "synthesis" in result
        assert len(result["findings"]) > 0
    
    def test_knowledge_base_search(self, researcher_agent):
        """Test knowledge base search"""
        task = {
            "query": "machine learning",
            "sources": ["knowledge_base"]
        }
        
        result = researcher_agent.process(task)
        
        assert len(result["findings"]) > 0
        assert any(f["source"] == "knowledge_base" for f in result["findings"])
    
    def test_synthesis_generation(self, researcher_agent):
        """Test findings synthesis"""
        task = {"query": "testing"}
        
        result = researcher_agent.process(task)
        synthesis = result["synthesis"]
        
        assert "summary" in synthesis
        assert "key_points" in synthesis
        assert "confidence" in synthesis
    
    def test_recommendations(self, researcher_agent):
        """Test recommendation generation"""
        task = {"query": "api design"}
        
        result = researcher_agent.process(task)
        
        assert "recommendations" in result
        assert len(result["recommendations"]) > 0
