"""Message parsing and state management for multi-agent systems

This module provides:
- Message parser for agent communication
- Shared state manager for coordination
- Conversation history tracking
- Message validation and routing

Learning objectives:
1. Parse and validate messages between agents
2. Manage shared state across distributed agents
3. Track conversation history for context
4. Implement message routing logic
5. Handle state conflicts and synchronization
"""

import logging
import time
from typing import Any, Dict, List, Optional, Set
from dataclasses import dataclass, field
from collections import defaultdict
import json

from rich.console import Console

logger = logging.getLogger("multi_agent")
console = Console()


@dataclass
class MessageMetadata:
    """Metadata for message tracking."""
    message_id: str
    conversation_id: str
    priority: int = 1  # 1=low, 2=medium, 3=high
    requires_response: bool = True
    timeout_seconds: int = 30


class MessageParser:
    """Parse and validate messages in multi-agent communication.
    
    Handles:
    - Message format validation
    - Content extraction
    - Type conversion
    - Error handling
    """
    
    def __init__(self):
        """Initialize message parser."""
        self.parsed_count = 0
        self.validation_errors = 0
    
    def parse_message(self, raw_message: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        TODO: Validate required fields, check message_type, extract and clean content
        
        📖 See: notes/04-multi_agent_ai/ch01_message_formats/
        ⚡ Advances constraint #4 SCALABILITY (structured message validation prevents context overflow)
        """
        # TODO: Your implementation here
        raise NotImplementedError("Implement message parsing")
    
    def extract_task_from_message(self, message: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        TODO: Extract task details (type, description, priority, parameters) from message content
        
        📖 See: notes/04-multi_agent_ai/ch01_message_formats/ (message schemas)
                notes/04-multi_agent_ai/ch03_a2a/ (A2A task lifecycle)
        ⚡ Advances constraint #4 SCALABILITY (task extraction enables agent coordination)
        """
        # TODO: Your implementation here
        raise NotImplementedError("Implement task extraction")
    
    def validate_response(self, request: Dict[str, Any], response: Dict[str, Any]) -> bool:
        """
        TODO: Validate sender/recipient match, message type, and timestamp order
        
        📖 See: notes/04-multi_agent_ai/ch01_message_formats/
        ⚡ Advances constraint #3 ACCURACY (validation prevents message corruption)
        """
        # TODO: Your implementation here
        raise NotImplementedError("Implement response validation")
    
    def get_statistics(self) -> Dict[str, int]:
        """Get parser statistics."""
        return {
            "total_parsed": self.parsed_count,
            "validation_errors": self.validation_errors,
            "success_rate": (self.parsed_count - self.validation_errors) / max(self.parsed_count, 1)
        }


class SharedStateManager:
    """Manage shared state across distributed agents.
    
    Provides:
    - Thread-safe state updates
    - Conflict detection and resolution
    - State versioning
    - Rollback capability
    """
    
    def __init__(self, enable_versioning: bool = True):
        """Initialize shared state manager.
        
        Args:
            enable_versioning: Enable state versioning for rollback
        """
        self.state: Dict[str, Any] = {}
        self.versions: List[Dict[str, Any]] = []
        self.enable_versioning = enable_versioning
        self.update_count = 0
        self.conflict_count = 0
        self.locks: Set[str] = set()
    
    def update(self, agent_name: str, key: str, value: Any) -> bool:
        """
        TODO: Check lock status, save version if enabled, update state, handle conflicts
        
        📖 See: notes/04-multi_agent_ai/ch05_shared_memory/
        ⚡ Advances constraint #3 ACCURACY (conflict-free state updates prevent race conditions)
        """
        # TODO: Your implementation here
        raise NotImplementedError("Implement state update")
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        TODO: Return value from self.state with default fallback
        
        📖 See: notes/04-multi_agent_ai/ch05_shared_memory/
        ⚡ Advances constraint #3 ACCURACY (consistent state reads)
        """
        # TODO: Your implementation here
        raise NotImplementedError("Implement state get")
    
    def lock(self, key: str, agent_name: str) -> bool:
        """
        TODO: Add key to self.locks if not already locked, return success status
        
        📖 See: notes/04-multi_agent_ai/ch05_shared_memory/ (optimistic/pessimistic locking)
        ⚡ Advances constraint #3 ACCURACY (prevents concurrent write conflicts)
        """
        # TODO: Your implementation here
        raise NotImplementedError("Implement state lock")
    
    def unlock(self, key: str, agent_name: str) -> bool:
        """
        TODO: Remove key from self.locks if locked, return success status
        
        📖 See: notes/04-multi_agent_ai/ch05_shared_memory/
        ⚡ Advances constraint #3 ACCURACY (releases locks for other agents)
        """
        # TODO: Your implementation here
        raise NotImplementedError("Implement state unlock")
    
    def rollback(self, key: str, steps: int = 1) -> bool:
        """Rollback state to previous version.
        
        Args:
            key: State key to rollback
            steps: Number of versions to rollback
        
        Returns:
            True if rollback successful
        """
        if not self.enable_versioning:
            console.print("❌ Versioning not enabled", style="red")
            return False
        
        # Find previous versions for this key
        key_versions = [v for v in reversed(self.versions) if v["key"] == key]
        
        if len(key_versions) < steps:
            console.print(f"❌ Not enough versions to rollback {steps} steps", style="red")
            return False
        
        # Rollback to previous version
        prev_version = key_versions[steps - 1]
        self.state[key] = prev_version["value"]
        
        console.print(
            f"  ↩️  Rolled back {key} to version from {prev_version['agent']}",
            style="yellow"
        )
        return True
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get state manager statistics."""
        return {
            "total_updates": self.update_count,
            "conflicts": self.conflict_count,
            "state_keys": len(self.state),
            "versions_stored": len(self.versions),
            "active_locks": len(self.locks)
        }
    
    def snapshot(self) -> Dict[str, Any]:
        """Create snapshot of current state."""
        return {
            "state": self.state.copy(),
            "timestamp": time.time(),
            "version": len(self.versions)
        }


class ConversationHistory:
    """Track conversation history between agents.
    
    Maintains context for multi-turn interactions and enables
    conversation analysis, replay, and debugging.
    """
    
    def __init__(self, max_history: int = 100):
        """Initialize conversation history.
        
        Args:
            max_history: Maximum number of messages to retain
        """
        self.messages: List[Dict[str, Any]] = []
        self.max_history = max_history
        self.conversations: Dict[str, List[Dict]] = defaultdict(list)
    
    def add_message(self, message: Dict[str, Any]):
        """
        TODO: Append message to history, track by conversation thread, enforce max_history limit
        
        📖 See: notes/04-multi_agent_ai/ch01_message_formats/ (message history tracking)
        ⚡ Advances constraint #6 AUDITABILITY (full conversation lineage for audit trails)
        """
        # TODO: Your implementation here
        raise NotImplementedError("Implement add message")
    
    def get_conversation(self, agent1: str, agent2: str) -> List[Dict[str, Any]]:
        """
        TODO: Combine and sort bidirectional conversation messages by timestamp
        
        📖 See: notes/04-multi_agent_ai/ch01_message_formats/
        ⚡ Advances constraint #6 AUDITABILITY (reconstruct agent interactions for compliance)
        """
        # TODO: Your implementation here
        raise NotImplementedError("Implement get conversation")
    
    def get_recent(self, n: int = 10) -> List[Dict[str, Any]]:
        """Get N most recent messages."""
        return self.messages[-n:] if len(self.messages) >= n else self.messages
    
    def search(self, keyword: str, agent: Optional[str] = None) -> List[Dict[str, Any]]:
        """Search conversation history by keyword and/or agent.
        
        Args:
            keyword: Search term to find in message content
            agent: Optional agent name to filter by
        
        Returns:
            List of matching messages
        """
        results = []
        for msg in self.messages:
            # Filter by agent if specified
            if agent and msg.get("sender") != agent and msg.get("recipient") != agent:
                continue
            
            # Search in content
            content_str = json.dumps(msg.get("content", {})).lower()
            if keyword.lower() in content_str:
                results.append(msg)
        
        return results
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get conversation statistics."""
        # Count messages by type
        type_counts = defaultdict(int)
        for msg in self.messages:
            type_counts[msg.get("message_type", "unknown")] += 1
        
        # Count messages by agent
        agent_counts = defaultdict(int)
        for msg in self.messages:
            agent_counts[msg.get("sender", "unknown")] += 1
        
        return {
            "total_messages": len(self.messages),
            "active_conversations": len(self.conversations),
            "by_type": dict(type_counts),
            "by_agent": dict(agent_counts),
            "history_utilization": len(self.messages) / self.max_history
        }
    
    def print_summary(self):
        """Print conversation history summary."""
        stats = self.get_statistics()
        
        console.print("\n[bold cyan]💬 CONVERSATION HISTORY[/bold cyan]")
        console.print(f"Total messages: {stats['total_messages']}", style="green")
        console.print(f"Active conversations: {stats['active_conversations']}", style="green")
        
        console.print("\nMessages by type:", style="yellow")
        for msg_type, count in stats["by_type"].items():
            console.print(f"  {msg_type}: {count}", style="dim")
        
        console.print("\nMessages by agent:", style="yellow")
        for agent, count in stats["by_agent"].items():
            console.print(f"  {agent}: {count}", style="dim")
        
        console.print(
            f"\nHistory utilization: {stats['history_utilization']:.1%}",
            style="cyan"
        )


class MessageRouter:
    """Route messages between agents with priority handling.
    
    Implements message queuing, priority-based delivery,
    and broadcast capabilities.
    """
    
    def __init__(self):
        """Initialize message router."""
        self.queues: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.broadcast_subscriptions: Dict[str, Set[str]] = defaultdict(set)
        self.delivered_count = 0
        self.dropped_count = 0
    
    def route(self, message: Dict[str, Any], agents: Dict[str, Any]) -> bool:
        """
        TODO: Validate recipient, handle broadcast, queue by priority, deliver message
        
        📖 See: notes/04-multi_agent_ai/ch04_event_driven_agents/ (async message routing)
        ⚡ Advances constraint #1 THROUGHPUT (priority queuing enables 1,000 POs/day)
        """
        # TODO: Your implementation here
        raise NotImplementedError("Implement message routing")
    
    def _handle_broadcast(self, message: Dict[str, Any], agents: Dict[str, Any]) -> bool:
        """Handle broadcast message to all agents."""
        sender = message.get("sender")
        delivered = 0
        
        for agent_name, agent in agents.items():
            if agent_name != sender:  # Don't send to self
                agent.receive_message(message)
                delivered += 1
        
        console.print(
            f"  📢 Broadcast from {sender}: delivered to {delivered} agents",
            style="cyan dim"
        )
        self.delivered_count += delivered
        return True
    
    def subscribe_to_broadcast(self, agent_name: str, channel: str):
        """Subscribe agent to broadcast channel."""
        self.broadcast_subscriptions[channel].add(agent_name)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get routing statistics."""
        return {
            "delivered": self.delivered_count,
            "dropped": self.dropped_count,
            "delivery_rate": self.delivered_count / max(self.delivered_count + self.dropped_count, 1),
            "active_queues": len([q for q in self.queues.values() if q]),
            "broadcast_channels": len(self.broadcast_subscriptions)
        }
