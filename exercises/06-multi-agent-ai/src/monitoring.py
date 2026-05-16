"""
Prometheus monitoring for multi-agent system
"""

from prometheus_client import Counter, Histogram, Gauge, generate_latest, REGISTRY
from typing import Dict, Any
from src.utils import get_logger


class MultiAgentMonitoring:
    """
    Prometheus metrics for multi-agent system
    """
    
    def __init__(self, metrics_prefix: str = "multiagent"):
        """
        Initialize monitoring
        
        Args:
            metrics_prefix: Prefix for all metrics
        """
        self.logger = get_logger("monitoring")
        self.prefix = metrics_prefix
        
        # Agent action counters
        self.agent_actions = Counter(
            f"{self.prefix}_agent_actions_total",
            "Total number of agent actions",
            ["agent_name", "action_type"]
        )
        
        # Message counters
        self.messages_sent = Counter(
            f"{self.prefix}_messages_sent_total",
            "Total messages sent",
            ["sender", "receiver", "message_type"]
        )
        
        self.messages_received = Counter(
            f"{self.prefix}_messages_received_total",
            "Total messages received",
            ["receiver", "message_type"]
        )
        
        # Task status counters
        self.tasks_total = Counter(
            f"{self.prefix}_tasks_total",
            "Total tasks executed",
            ["status"]
        )
        
        # Execution time histograms
        self.task_duration = Histogram(
            f"{self.prefix}_task_duration_seconds",
            "Task execution duration",
            ["task_type"]
        )
        
        self.agent_processing_time = Histogram(
            f"{self.prefix}_agent_processing_seconds",
            "Agent processing time",
            ["agent_name"]
        )
        
        # Active resource gauges
        self.active_tasks = Gauge(
            f"{self.prefix}_active_tasks",
            "Number of active tasks"
        )
        
        self.active_agents = Gauge(
            f"{self.prefix}_active_agents",
            "Number of active agents"
        )
        
        self.queue_length = Gauge(
            f"{self.prefix}_queue_length",
            "Message queue length",
            ["queue_name"]
        )
        
        # Performance metrics
        self.coordination_efficiency = Gauge(
            f"{self.prefix}_coordination_efficiency",
            "Coordination efficiency score"
        )
        
        self.completion_rate = Gauge(
            f"{self.prefix}_completion_rate",
            "Task completion rate"
        )
        
        self.logger.info(f"Monitoring initialized with prefix: {self.prefix}")
    
    def record_agent_action(self, agent_name: str, action_type: str) -> None:
        """Record agent action"""
        self.agent_actions.labels(
            agent_name=agent_name,
            action_type=action_type
        ).inc()
    
    def record_message_sent(
        self, sender: str, receiver: str, message_type: str
    ) -> None:
        """Record message sent"""
        self.messages_sent.labels(
            sender=sender,
            receiver=receiver,
            message_type=message_type
        ).inc()
    
    def record_message_received(self, receiver: str, message_type: str) -> None:
        """Record message received"""
        self.messages_received.labels(
            receiver=receiver,
            message_type=message_type
        ).inc()
    
    def record_task_status(self, status: str) -> None:
        """Record task status"""
        self.tasks_total.labels(status=status).inc()
    
    def record_task_duration(self, task_type: str, duration: float) -> None:
        """Record task execution duration"""
        self.task_duration.labels(task_type=task_type).observe(duration)
    
    def record_agent_processing_time(self, agent_name: str, duration: float) -> None:
        """Record agent processing time"""
        self.agent_processing_time.labels(agent_name=agent_name).observe(duration)
    
    def update_active_tasks(self, count: int) -> None:
        """Update active tasks gauge"""
        self.active_tasks.set(count)
    
    def update_active_agents(self, count: int) -> None:
        """Update active agents gauge"""
        self.active_agents.set(count)
    
    def update_queue_length(self, queue_name: str, length: int) -> None:
        """Update queue length gauge"""
        self.queue_length.labels(queue_name=queue_name).set(length)
    
    def update_coordination_efficiency(self, efficiency: float) -> None:
        """Update coordination efficiency gauge"""
        self.coordination_efficiency.set(efficiency)
    
    def update_completion_rate(self, rate: float) -> None:
        """Update completion rate gauge"""
        self.completion_rate.set(rate)
    
    def update_metrics_from_evaluation(self, metrics: Dict[str, Any]) -> None:
        """
        Update metrics from evaluation results
        
        Args:
            metrics: Metrics dictionary from evaluator
        """
        if "completion_rate" in metrics:
            self.update_completion_rate(metrics["completion_rate"])
        
        if "coordination_efficiency" in metrics:
            self.update_coordination_efficiency(metrics["coordination_efficiency"])
        
        if "total_execution_time" in metrics:
            self.record_task_duration("overall", metrics["total_execution_time"])
    
    def get_metrics(self) -> bytes:
        """
        Get current metrics in Prometheus format
        
        Returns:
            Metrics in Prometheus text format
        """
        return generate_latest(REGISTRY)
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """
        Get human-readable metrics summary
        
        Returns:
            Dictionary with current metric values
        """
        # This is a simplified version - in production, you'd query Prometheus
        return {
            "active_tasks": self.active_tasks._value.get(),
            "active_agents": self.active_agents._value.get(),
            "coordination_efficiency": self.coordination_efficiency._value.get(),
            "completion_rate": self.completion_rate._value.get()
        }
