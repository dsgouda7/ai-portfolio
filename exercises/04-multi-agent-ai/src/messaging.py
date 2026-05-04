"""
Message Queue implementation with Redis backend
"""

import json
import time
from typing import Dict, Any, Optional, List
from src.utils import get_logger


class MessageQueue:
    """
    Message queue for inter-agent communication using Redis
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize message queue
        
        Args:
            config: Configuration with Redis connection details
        """
        self.config = config
        self.logger = get_logger("message_queue")
        
        self.host = config.get("redis_host", "localhost")
        self.port = config.get("redis_port", 6379)
        self.db = config.get("redis_db", 0)
        self.max_retries = config.get("max_retries", 3)
        self.retry_delay = config.get("retry_delay", 2)
        self.message_ttl = config.get("message_ttl", 3600)
        
        # In production, initialize Redis client
        # For demonstration, use in-memory queue
        self.redis_client = None
        self.in_memory_queue: Dict[str, List[Dict[str, Any]]] = {}
        self.message_history: List[Dict[str, Any]] = []
        
        self.logger.info(f"MessageQueue initialized (mode: in-memory)")
    
    def connect(self) -> bool:
        """
        Connect to Redis server
        
        Returns:
            True if connected successfully
        """
        try:
            # In production, connect to Redis:
            # import redis
            # self.redis_client = redis.Redis(
            #     host=self.host,
            #     port=self.port,
            #     db=self.db,
            #     decode_responses=True
            # )
            # return self.redis_client.ping()
            
            self.logger.info("Using in-memory queue (Redis not connected)")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to connect to Redis: {str(e)}")
            return False
    
    def publish(self, queue_name: str, message: Dict[str, Any]) -> bool:
        """
        Publish message to queue
        
        Args:
            queue_name: Name of the queue/channel
            message: Message dictionary
            
        Returns:
            True if published successfully
        """
        try:
            message["published_at"] = time.time()
            
            # Add to in-memory queue
            if queue_name not in self.in_memory_queue:
                self.in_memory_queue[queue_name] = []
            
            self.in_memory_queue[queue_name].append(message)
            self.message_history.append({
                "queue": queue_name,
                "message": message,
                "action": "publish"
            })
            
            self.logger.debug(f"Published message to {queue_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to publish message: {str(e)}")
            return False
    
    def subscribe(self, queue_name: str, callback: callable) -> None:
        """
        Subscribe to queue and process messages
        
        Args:
            queue_name: Name of the queue/channel
            callback: Function to call for each message
        """
        self.logger.info(f"Subscribed to {queue_name}")
        
        # In production, this would be a blocking operation listening to Redis
        # For demonstration, process existing messages
        messages = self.in_memory_queue.get(queue_name, [])
        
        for message in messages:
            try:
                callback(message)
            except Exception as e:
                self.logger.error(f"Error processing message: {str(e)}")
    
    def get_messages(self, queue_name: str, count: int = 10) -> List[Dict[str, Any]]:
        """
        Get messages from queue
        
        Args:
            queue_name: Name of the queue
            count: Maximum number of messages to retrieve
            
        Returns:
            List of messages
        """
        messages = self.in_memory_queue.get(queue_name, [])
        
        # Get latest messages
        result = messages[-count:] if len(messages) > count else messages
        
        self.logger.debug(f"Retrieved {len(result)} messages from {queue_name}")
        return result
    
    def pop_message(self, queue_name: str) -> Optional[Dict[str, Any]]:
        """
        Pop oldest message from queue
        
        Args:
            queue_name: Name of the queue
            
        Returns:
            Message dictionary or None if queue is empty
        """
        messages = self.in_memory_queue.get(queue_name, [])
        
        if not messages:
            return None
        
        message = messages.pop(0)
        self.logger.debug(f"Popped message from {queue_name}")
        return message
    
    def peek_message(self, queue_name: str) -> Optional[Dict[str, Any]]:
        """
        Peek at oldest message without removing it
        
        Args:
            queue_name: Name of the queue
            
        Returns:
            Message dictionary or None if queue is empty
        """
        messages = self.in_memory_queue.get(queue_name, [])
        return messages[0] if messages else None
    
    def queue_length(self, queue_name: str) -> int:
        """
        Get number of messages in queue
        
        Args:
            queue_name: Name of the queue
            
        Returns:
            Number of messages
        """
        return len(self.in_memory_queue.get(queue_name, []))
    
    def clear_queue(self, queue_name: str) -> int:
        """
        Clear all messages from queue
        
        Args:
            queue_name: Name of the queue
            
        Returns:
            Number of messages cleared
        """
        messages = self.in_memory_queue.get(queue_name, [])
        count = len(messages)
        self.in_memory_queue[queue_name] = []
        
        self.logger.info(f"Cleared {count} messages from {queue_name}")
        return count
    
    def list_queues(self) -> List[str]:
        """
        List all queue names
        
        Returns:
            List of queue names
        """
        return list(self.in_memory_queue.keys())
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get message queue statistics
        
        Returns:
            Dictionary with queue statistics
        """
        return {
            "total_queues": len(self.in_memory_queue),
            "total_messages": sum(len(msgs) for msgs in self.in_memory_queue.values()),
            "messages_by_queue": {
                queue: len(msgs) for queue, msgs in self.in_memory_queue.items()
            },
            "total_history": len(self.message_history)
        }
    
    def close(self) -> None:
        """Close Redis connection"""
        if self.redis_client:
            self.redis_client.close()
            self.logger.info("Redis connection closed")
