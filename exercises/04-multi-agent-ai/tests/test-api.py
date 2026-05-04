"""
Tests for Flask API
"""

import pytest
import json
from src.api import create_app


@pytest.fixture
def client():
    """Create test client"""
    app = create_app("config.yaml")
    app.config['TESTING'] = True
    
    with app.test_client() as client:
        yield client


class TestAPI:
    """Test cases for Flask API"""
    
    def test_health_check(self, client):
        """Test health check endpoint"""
        response = client.get('/health')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data["status"] == "healthy"
    
    def test_execute_task(self, client):
        """Test task execution endpoint"""
        task = {
            "description": "Test task execution"
        }
        
        response = client.post(
            '/task',
            data=json.dumps(task),
            content_type='application/json'
        )
        
        assert response.status_code == 200
        data = json.loads(response.data)
        
        assert "task_id" in data
        assert data["status"] == "success"
        assert "plan" in data
        assert "results" in data
    
    def test_execute_task_with_constraints(self, client):
        """Test task execution with constraints"""
        task = {
            "description": "Research machine learning",
            "constraints": {
                "time_limit": 60,
                "quality_threshold": 0.8
            }
        }
        
        response = client.post(
            '/task',
            data=json.dumps(task),
            content_type='application/json'
        )
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data["status"] == "success"
    
    def test_execute_task_missing_description(self, client):
        """Test task execution with missing description"""
        task = {}
        
        response = client.post(
            '/task',
            data=json.dumps(task),
            content_type='application/json'
        )
        
        assert response.status_code == 400
        data = json.loads(response.data)
        assert "error" in data
    
    def test_execute_task_invalid_json(self, client):
        """Test task execution with invalid JSON"""
        response = client.post(
            '/task',
            data="not json",
            content_type='application/json'
        )
        
        assert response.status_code == 400
    
    def test_list_agents(self, client):
        """Test agent listing endpoint"""
        response = client.get('/agents')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        
        assert "agents" in data
        assert "planner_agent" in data["agents"]
        assert "executor_agent" in data["agents"]
    
    def test_get_metrics_json(self, client):
        """Test metrics endpoint (JSON format)"""
        response = client.get('/metrics')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        
        assert "system_metrics" in data
        assert "monitoring" in data
        assert "success_criteria" in data
    
    def test_get_metrics_prometheus(self, client):
        """Test metrics endpoint (Prometheus format)"""
        response = client.get(
            '/metrics',
            headers={'Accept': 'text/plain'}
        )
        
        assert response.status_code == 200
        assert response.content_type.startswith('text/plain')
    
    def test_queue_status(self, client):
        """Test queue status endpoint"""
        response = client.get('/queue/status')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        
        assert "stats" in data
        assert "queues" in data
    
    def test_not_found(self, client):
        """Test 404 error handling"""
        response = client.get('/nonexistent')
        
        assert response.status_code == 404
        data = json.loads(response.data)
        assert "error" in data


class TestAPIIntegration:
    """Integration tests for API workflows"""
    
    def test_complete_workflow(self, client):
        """Test complete workflow through API"""
        # Execute task
        task = {
            "description": "Research and analyze API design patterns"
        }
        
        response = client.post(
            '/task',
            data=json.dumps(task),
            content_type='application/json'
        )
        
        assert response.status_code == 200
        result = json.loads(response.data)
        
        # Check task result
        assert result["status"] == "success"
        task_id = result["task_id"]
        
        # Check metrics updated
        metrics_response = client.get('/metrics')
        metrics = json.loads(metrics_response.data)
        
        assert metrics["system_metrics"]["total_tasks"] > 0
    
    def test_multiple_tasks(self, client):
        """Test executing multiple tasks"""
        tasks = [
            {"description": "Task 1: Research"},
            {"description": "Task 2: Execute"},
            {"description": "Task 3: Analyze"}
        ]
        
        for task in tasks:
            response = client.post(
                '/task',
                data=json.dumps(task),
                content_type='application/json'
            )
            assert response.status_code == 200
        
        # Check metrics
        metrics_response = client.get('/metrics')
        metrics = json.loads(metrics_response.data)
        
        assert metrics["system_metrics"]["total_tasks"] >= 3
