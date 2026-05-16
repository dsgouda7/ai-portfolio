"""
Tests for AgentCoordinator
"""

import pytest


class TestAgentCoordinator:
    """Test cases for AgentCoordinator"""
    
    def test_initialization(self, coordinator):
        """Test coordinator initialization"""
        assert coordinator is not None
        assert len(coordinator.agents) == 4
        assert coordinator.max_concurrent == 4
    
    def test_task_execution(self, coordinator, sample_task):
        """Test full task execution workflow"""
        result = coordinator.execute_task(sample_task)
        
        assert result["status"] == "success"
        assert "task_id" in result
        assert "plan" in result
        assert "results" in result
        assert "evaluation" in result
        assert "execution_time" in result
    
    def test_planning_phase(self, coordinator, sample_task):
        """Test task planning phase"""
        plan = coordinator._plan_task(sample_task)
        
        assert "subtasks" in plan
        assert "plan" in plan
        assert len(plan["subtasks"]) > 0
    
    def test_needs_research(self, coordinator):
        """Test research requirement detection"""
        # Plan with research subtask
        plan_with_research = {
            "subtasks": [
                {"type": "research", "description": "Research topic"}
            ]
        }
        assert coordinator._needs_research(plan_with_research) is True
        
        # Plan without research subtask
        plan_without_research = {
            "subtasks": [
                {"type": "execution", "description": "Execute task"}
            ]
        }
        assert coordinator._needs_research(plan_without_research) is False
    
    def test_research_phase(self, coordinator):
        """Test research phase"""
        task = {"description": "Research machine learning"}
        research = coordinator._research_task(task)
        
        assert research is not None
        assert isinstance(research, dict)
    
    def test_agent_status(self, coordinator):
        """Test agent status retrieval"""
        status = coordinator.get_agent_status()
        
        assert isinstance(status, dict)
        assert "planner_agent" in status
        assert "executor_agent" in status
    
    def test_error_handling(self, coordinator):
        """Test error handling in task execution"""
        # Test with invalid task
        invalid_task = {}
        
        result = coordinator.execute_task(invalid_task)
        
        # Should complete even with minimal input
        assert "status" in result


class TestMultiAgentWorkflow:
    """Integration tests for multi-agent workflows"""
    
    def test_simple_workflow(self, coordinator):
        """Test simple end-to-end workflow"""
        task = {
            "description": "Execute simple task"
        }
        
        result = coordinator.execute_task(task)
        
        assert result["status"] == "success"
        assert len(result["results"]) > 0
    
    def test_research_workflow(self, coordinator):
        """Test workflow with research phase"""
        task = {
            "description": "Research and analyze data processing patterns"
        }
        
        result = coordinator.execute_task(task)
        
        assert result["status"] == "success"
        # Should have research subtask
        subtasks = result["plan"]["subtasks"]
        assert any(t["type"] == "research" for t in subtasks)
    
    def test_complex_workflow(self, coordinator):
        """Test complex workflow with multiple phases"""
        task = {
            "description": "Research, analyze, and evaluate machine learning deployment strategies",
            "constraints": {
                "quality_threshold": 0.8
            }
        }
        
        result = coordinator.execute_task(task)
        
        assert result["status"] == "success"
        assert len(result["results"]) >= 2
        assert "evaluation" in result
    
    def test_parallel_execution(self, coordinator):
        """Test parallel task execution"""
        task = {
            "description": "Execute multiple independent analyses"
        }
        
        result = coordinator.execute_task(task)
        plan = result["plan"]["plan"]
        
        # Should identify parallel opportunities
        assert "parallel_groups" in plan
