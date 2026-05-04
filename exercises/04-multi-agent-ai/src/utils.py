"""
Utility functions for multi-agent system
"""

import logging
import time
from typing import Dict, Any, Optional
from datetime import datetime
from enum import Enum


class TaskStatus(Enum):
    """Task execution status"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class MessageType(Enum):
    """Agent message types"""
    TASK_REQUEST = "task_request"
    TASK_RESPONSE = "task_response"
    STATUS_UPDATE = "status_update"
    FEEDBACK = "feedback"
    ERROR = "error"


def get_logger(name: str = "multiagent") -> logging.Logger:
    """
    Get configured logger for multi-agent system
    
    Args:
        name: Logger name (default: "multiagent")
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    
    return logger


def route_message(message: Dict[str, Any], agent_registry: Dict[str, Any]) -> Optional[str]:
    """
    Route message to appropriate agent based on message type and content
    
    Args:
        message: Message dictionary with type, sender, receiver, content
        agent_registry: Dictionary of available agents
        
    Returns:
        Target agent name or None if routing fails
    """
    receiver = message.get("receiver")
    
    if receiver and receiver in agent_registry:
        return receiver
    
    # Default routing based on message type
    msg_type = message.get("type")
    if msg_type == MessageType.TASK_REQUEST.value:
        return "planner_agent"
    elif msg_type == MessageType.FEEDBACK.value:
        return "critic_agent"
    
    return None


class StateManager:
    """Manage shared state across agents"""
    
    def __init__(self):
        self.state: Dict[str, Any] = {}
        self.history: list = []
        self.logger = get_logger("state_manager")
    
    def update(self, key: str, value: Any) -> None:
        """Update state value"""
        old_value = self.state.get(key)
        self.state[key] = value
        
        self.history.append({
            "timestamp": datetime.now().isoformat(),
            "key": key,
            "old_value": old_value,
            "new_value": value
        })
        
        self.logger.debug(f"State updated: {key} = {value}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get state value"""
        return self.state.get(key, default)
    
    def get_history(self, key: Optional[str] = None) -> list:
        """Get state change history"""
        if key:
            return [h for h in self.history if h["key"] == key]
        return self.history
    
    def snapshot(self) -> Dict[str, Any]:
        """Get current state snapshot"""
        return self.state.copy()
    
    def restore(self, snapshot: Dict[str, Any]) -> None:
        """Restore state from snapshot"""
        self.state = snapshot.copy()
        self.logger.info("State restored from snapshot")


def measure_time(func):
    """Decorator to measure function execution time"""
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        duration = time.time() - start
        logger = get_logger()
        logger.debug(f"{func.__name__} took {duration:.3f}s")
        return result
    return wrapper


def create_message(
    msg_type: MessageType,
    sender: str,
    receiver: str,
    content: Dict[str, Any],
    metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Create standardized message structure
    
    Args:
        msg_type: Type of message
        sender: Sender agent name
        receiver: Receiver agent name
        content: Message content
        metadata: Optional metadata
        
    Returns:
        Structured message dictionary
    """
    message = {
        "type": msg_type.value,
        "sender": sender,
        "receiver": receiver,
        "content": content,
        "timestamp": datetime.now().isoformat(),
        "id": f"{sender}_{int(time.time() * 1000)}"
    }
    
    if metadata:
        message["metadata"] = metadata
    
    return message
