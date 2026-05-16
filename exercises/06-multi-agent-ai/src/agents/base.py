"""
Base agent abstract class
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from src.utils import get_logger, MessageType, create_message


class BaseAgent(ABC):
    """
    Abstract base class for all agents in the multi-agent system
    """
    
    def __init__(self, name: str, config: Dict[str, Any]):
        """
        Initialize base agent
        
        Args:
            name: Agent name
            config: Agent configuration dictionary
        """
        self.name = name
        self.config = config
        self.logger = get_logger(f"agent.{name}")
        self.state = {}
        self.message_history = []
        self.enabled = config.get("enabled", True)
        
        self.logger.info(f"Initialized {self.__class__.__name__}: {name}")
    
    @abstractmethod
    def process(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process task and return result
        
        Args:
            task: Task dictionary with input data
            
        Returns:
            Result dictionary with output data
        """
        pass
    
    def receive_message(self, message: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Receive and handle incoming message
        
        Args:
            message: Message dictionary
            
        Returns:
            Response message or None
        """
        self.message_history.append(message)
        self.logger.debug(f"Received message from {message.get('sender')}")
        
        if not self.enabled:
            self.logger.warning(f"Agent {self.name} is disabled, ignoring message")
            return None
        
        msg_type = message.get("type")
        
        if msg_type == MessageType.TASK_REQUEST.value:
            return self._handle_task_request(message)
        elif msg_type == MessageType.FEEDBACK.value:
            return self._handle_feedback(message)
        
        return None
    
    def _handle_task_request(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Handle task request message"""
        task = message.get("content", {})
        result = self.process(task)
        
        return create_message(
            msg_type=MessageType.TASK_RESPONSE,
            sender=self.name,
            receiver=message.get("sender"),
            content=result
        )
    
    def _handle_feedback(self, message: Dict[str, Any]) -> None:
        """Handle feedback message"""
        feedback = message.get("content", {})
        self.logger.info(f"Received feedback: {feedback.get('summary', 'N/A')}")
        # Store feedback for future improvements
        self.state["last_feedback"] = feedback
    
    def get_status(self) -> Dict[str, Any]:
        """Get agent status"""
        return {
            "name": self.name,
            "enabled": self.enabled,
            "messages_received": len(self.message_history),
            "state": self.state
        }
    
    def reset(self) -> None:
        """Reset agent state"""
        self.state = {}
        self.message_history = []
        self.logger.info(f"Agent {self.name} reset")
