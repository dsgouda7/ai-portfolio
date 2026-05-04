"""
Tests for MessageQueue
"""

import pytest
from src.utils import MessageType, create_message


class TestMessageQueue:
    """Test cases for MessageQueue"""
    
    def test_initialization(self, message_queue):
        """Test message queue initialization"""
        assert message_queue is not None
        assert message_queue.host == "localhost"
        assert message_queue.port == 6379
    
    def test_connection(self, message_queue):
        """Test queue connection"""
        result = message_queue.connect()
        assert result is True
    
    def test_publish_message(self, message_queue):
        """Test message publishing"""
        message = create_message(
            msg_type=MessageType.TASK_REQUEST,
            sender="test_agent",
            receiver="target_agent",
            content={"data": "test"}
        )
        
        result = message_queue.publish("test_queue", message)
        assert result is True
    
    def test_get_messages(self, message_queue):
        """Test message retrieval"""
        # Publish test messages
        for i in range(3):
            message = create_message(
                msg_type=MessageType.TASK_REQUEST,
                sender="test",
                receiver="target",
                content={"index": i}
            )
            message_queue.publish("test_queue", message)
        
        # Retrieve messages
        messages = message_queue.get_messages("test_queue", count=5)
        
        assert len(messages) == 3
    
    def test_pop_message(self, message_queue):
        """Test message popping"""
        # Publish message
        message = create_message(
            msg_type=MessageType.TASK_REQUEST,
            sender="test",
            receiver="target",
            content={"data": "pop_test"}
        )
        message_queue.publish("pop_queue", message)
        
        # Pop message
        popped = message_queue.pop_message("pop_queue")
        
        assert popped is not None
        assert popped["content"]["data"] == "pop_test"
        
        # Queue should be empty now
        assert message_queue.queue_length("pop_queue") == 0
    
    def test_peek_message(self, message_queue):
        """Test message peeking"""
        # Publish message
        message = create_message(
            msg_type=MessageType.TASK_REQUEST,
            sender="test",
            receiver="target",
            content={"data": "peek_test"}
        )
        message_queue.publish("peek_queue", message)
        
        # Peek at message
        peeked = message_queue.peek_message("peek_queue")
        
        assert peeked is not None
        assert peeked["content"]["data"] == "peek_test"
        
        # Message should still be in queue
        assert message_queue.queue_length("peek_queue") == 1
    
    def test_queue_length(self, message_queue):
        """Test queue length calculation"""
        queue_name = "length_test_queue"
        
        # Empty queue
        assert message_queue.queue_length(queue_name) == 0
        
        # Add messages
        for i in range(5):
            message = create_message(
                msg_type=MessageType.TASK_REQUEST,
                sender="test",
                receiver="target",
                content={"index": i}
            )
            message_queue.publish(queue_name, message)
        
        assert message_queue.queue_length(queue_name) == 5
    
    def test_clear_queue(self, message_queue):
        """Test queue clearing"""
        queue_name = "clear_test_queue"
        
        # Add messages
        for i in range(3):
            message = create_message(
                msg_type=MessageType.TASK_REQUEST,
                sender="test",
                receiver="target",
                content={"index": i}
            )
            message_queue.publish(queue_name, message)
        
        # Clear queue
        cleared = message_queue.clear_queue(queue_name)
        
        assert cleared == 3
        assert message_queue.queue_length(queue_name) == 0
    
    def test_list_queues(self, message_queue):
        """Test queue listing"""
        # Create multiple queues
        for i in range(3):
            queue_name = f"queue_{i}"
            message = create_message(
                msg_type=MessageType.TASK_REQUEST,
                sender="test",
                receiver="target",
                content={}
            )
            message_queue.publish(queue_name, message)
        
        queues = message_queue.list_queues()
        
        assert len(queues) >= 3
        assert "queue_0" in queues
    
    def test_get_stats(self, message_queue):
        """Test statistics retrieval"""
        # Add messages to different queues
        for i in range(2):
            queue_name = f"stats_queue_{i}"
            for j in range(i + 1):
                message = create_message(
                    msg_type=MessageType.TASK_REQUEST,
                    sender="test",
                    receiver="target",
                    content={}
                )
                message_queue.publish(queue_name, message)
        
        stats = message_queue.get_stats()
        
        assert "total_queues" in stats
        assert "total_messages" in stats
        assert "messages_by_queue" in stats
