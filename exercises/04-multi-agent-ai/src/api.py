"""
Flask API for multi-agent system
"""

import yaml
from flask import Flask, request, jsonify
from typing import Dict, Any
from src.agents.planner import PlannerAgent
from src.agents.executor import ExecutorAgent
from src.agents.critic import CriticAgent
from src.agents.researcher import ResearcherAgent
from src.coordinator import AgentCoordinator
from src.messaging import MessageQueue
from src.evaluate import MultiAgentEvaluator
from src.monitoring import MultiAgentMonitoring
from src.utils import get_logger


# Initialize Flask app
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB

# Initialize logger
logger = get_logger("api")

# Global variables (initialized in create_app)
coordinator = None
evaluator = None
monitoring = None
message_queue = None


def create_app(config_path: str = "config.yaml") -> Flask:
    """
    Create and configure Flask application
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configured Flask app
    """
    global coordinator, evaluator, monitoring, message_queue
    
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    logger.info("Initializing multi-agent system")
    
    # Initialize message queue
    message_queue = MessageQueue(config["communication"])
    message_queue.connect()
    
    # Initialize agents
    agents = {}
    
    if config["agents"]["planner_agent"]["enabled"]:
        agents["planner_agent"] = PlannerAgent(
            "planner_agent",
            config["agents"]["planner_agent"]
        )
    
    if config["agents"]["executor_agent"]["enabled"]:
        agents["executor_agent"] = ExecutorAgent(
            "executor_agent",
            config["agents"]["executor_agent"]
        )
    
    if config["agents"]["critic_agent"]["enabled"]:
        agents["critic_agent"] = CriticAgent(
            "critic_agent",
            config["agents"]["critic_agent"]
        )
    
    if config["agents"]["researcher_agent"]["enabled"]:
        agents["researcher_agent"] = ResearcherAgent(
            "researcher_agent",
            config["agents"]["researcher_agent"]
        )
    
    # Initialize coordinator
    coordinator = AgentCoordinator(agents, config["coordination"])
    
    # Initialize evaluator
    evaluator = MultiAgentEvaluator()
    
    # Initialize monitoring
    monitoring = MultiAgentMonitoring(
        metrics_prefix=config["monitoring"]["metrics_prefix"]
    )
    
    logger.info(f"Multi-agent system initialized with {len(agents)} agents")
    
    return app


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "service": "multi-agent-system",
        "version": "1.0.0"
    }), 200


@app.route('/task', methods=['POST'])
def execute_task():
    """
    Execute task using multi-agent system
    
    Request body:
    {
        "description": "Task description",
        "constraints": {...},  # Optional
        "workflow": "custom"   # Optional
    }
    
    Returns:
    {
        "task_id": "task_123",
        "status": "success",
        "plan": {...},
        "results": [...],
        "evaluation": {...},
        "execution_time": 1.23
    }
    """
    try:
        # Validate request
        if not request.is_json:
            return jsonify({"error": "Request must be JSON"}), 400
        
        task = request.get_json()
        
        if "description" not in task:
            return jsonify({"error": "Missing 'description' field"}), 400
        
        logger.info(f"Received task: {task.get('description')}")
        
        # Update monitoring
        monitoring.update_active_tasks(
            monitoring.active_tasks._value.get() + 1
        )
        
        # Execute task
        result = coordinator.execute_task(task)
        
        # Evaluate execution
        metrics = evaluator.evaluate_task_execution(result)
        
        # Update monitoring
        monitoring.update_metrics_from_evaluation(metrics)
        monitoring.record_task_status(result.get("status", "unknown"))
        monitoring.update_active_tasks(
            max(0, monitoring.active_tasks._value.get() - 1)
        )
        
        # Add metrics to response
        result["metrics"] = metrics
        
        return jsonify(result), 200
        
    except Exception as e:
        logger.error(f"Error executing task: {str(e)}")
        monitoring.record_task_status("failed")
        return jsonify({
            "error": str(e),
            "status": "failed"
        }), 500


@app.route('/agents', methods=['GET'])
def list_agents():
    """
    List all available agents and their status
    
    Returns:
    {
        "agents": {
            "planner_agent": {...},
            "executor_agent": {...},
            ...
        }
    }
    """
    try:
        agent_status = coordinator.get_agent_status()
        return jsonify({"agents": agent_status}), 200
        
    except Exception as e:
        logger.error(f"Error listing agents: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route('/metrics', methods=['GET'])
def get_metrics():
    """
    Get system metrics
    
    Returns Prometheus-format metrics or JSON summary
    """
    try:
        # Check if Prometheus format requested
        accept_header = request.headers.get('Accept', '')
        
        if 'text/plain' in accept_header or 'prometheus' in accept_header:
            # Return Prometheus format
            metrics = monitoring.get_metrics()
            return metrics, 200, {'Content-Type': 'text/plain; charset=utf-8'}
        else:
            # Return JSON summary
            system_metrics = evaluator.get_system_metrics()
            monitoring_metrics = monitoring.get_metrics_summary()
            
            return jsonify({
                "system_metrics": system_metrics,
                "monitoring": monitoring_metrics,
                "success_criteria": evaluator.check_success_criteria()
            }), 200
            
    except Exception as e:
        logger.error(f"Error getting metrics: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route('/queue/status', methods=['GET'])
def queue_status():
    """
    Get message queue status
    
    Returns:
    {
        "stats": {...},
        "queues": [...]
    }
    """
    try:
        stats = message_queue.get_stats()
        queues = message_queue.list_queues()
        
        return jsonify({
            "stats": stats,
            "queues": queues
        }), 200
        
    except Exception as e:
        logger.error(f"Error getting queue status: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({"error": "Endpoint not found"}), 404


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    logger.error(f"Internal error: {str(error)}")
    return jsonify({"error": "Internal server error"}), 500


if __name__ == "__main__":
    # Create app with config
    app = create_app("config.yaml")
    
    # Run server
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False
    )
