"""
Utility functions for PizzaBot
"""

import logging
import os
import yaml
from typing import Dict, Any, List
from datetime import datetime
import json


def setup_logger(name: str = "pizzabot", level: str = "INFO") -> logging.Logger:
    """
    Set up logger with consistent formatting.
    
    Args:
        name: Logger name
        level: Logging level
        
    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level))
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, level))
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(formatter)
    
    # File handler
    os.makedirs('logs', exist_ok=True)
    file_handler = logging.FileHandler(f'logs/{name}.log')
    file_handler.setLevel(getattr(logging, level))
    file_handler.setFormatter(formatter)
    
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to config file
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def save_conversation(session_id: str, messages: List[Dict[str, str]]) -> None:
    """
    Save conversation history to file.
    
    Args:
        session_id: Unique session identifier
        messages: List of message dictionaries
    """
    os.makedirs('logs/conversations', exist_ok=True)
    filepath = f'logs/conversations/{session_id}.json'
    
    with open(filepath, 'w') as f:
        json.dump({
            'session_id': session_id,
            'timestamp': datetime.now().isoformat(),
            'messages': messages
        }, f, indent=2)


def load_conversation(session_id: str) -> List[Dict[str, str]]:
    """
    Load conversation history from file.
    
    Args:
        session_id: Unique session identifier
        
    Returns:
        List of message dictionaries
    """
    filepath = f'logs/conversations/{session_id}.json'
    
    if not os.path.exists(filepath):
        return []
    
    with open(filepath, 'r') as f:
        data = json.load(f)
        return data.get('messages', [])


def format_price(amount: float) -> str:
    """
    Format price as currency string.
    
    Args:
        amount: Price amount
        
    Returns:
        Formatted price string
    """
    return f"${amount:.2f}"


def extract_order_items(text: str, menu: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Extract order items from user text.
    
    Args:
        text: User message text
        menu: Menu data
        
    Returns:
        List of order items with quantities
    """
    items = []
    text_lower = text.lower()
    
    # Simple keyword matching for pizza types
    for pizza_name, pizza_data in menu.items():
        pizza_lower = pizza_name.lower()
        if pizza_lower in text_lower:
            # Extract quantity (default to 1)
            quantity = 1
            # Look for numbers before pizza name
            words = text_lower.split()
            for i, word in enumerate(words):
                if word.isdigit() and i < len(words) - 1:
                    if pizza_lower in ' '.join(words[i:]):
                        quantity = int(word)
                        break
            
            items.append({
                'name': pizza_name,
                'quantity': quantity,
                'price': pizza_data['price']
            })
    
    return items


def calculate_total(items: List[Dict[str, Any]]) -> float:
    """
    Calculate total order amount.
    
    Args:
        items: List of order items
        
    Returns:
        Total amount
    """
    return sum(item['quantity'] * item['price'] for item in items)


def validate_delivery_address(address: str) -> bool:
    """
    Validate delivery address.
    
    Args:
        address: Delivery address string
        
    Returns:
        True if valid, False otherwise
    """
    # Simple validation: must have street number and name
    if not address or len(address) < 10:
        return False
    
    # Check for basic components
    has_number = any(c.isdigit() for c in address)
    has_letters = any(c.isalpha() for c in address)
    
    return has_number and has_letters


class ConversationTracker:
    """Track conversation state and context."""
    
    def __init__(self, max_history: int = 10):
        self.sessions = {}
        self.max_history = max_history
    
    def add_message(self, session_id: str, role: str, content: str) -> None:
        """Add message to conversation history."""
        if session_id not in self.sessions:
            self.sessions[session_id] = {
                'messages': [],
                'context': {},
                'created_at': datetime.now()
            }
        
        self.sessions[session_id]['messages'].append({
            'role': role,
            'content': content,
            'timestamp': datetime.now().isoformat()
        })
        
        # Keep only recent messages
        if len(self.sessions[session_id]['messages']) > self.max_history:
            self.sessions[session_id]['messages'] = \
                self.sessions[session_id]['messages'][-self.max_history:]
    
    def get_history(self, session_id: str) -> List[Dict[str, str]]:
        """Get conversation history for session."""
        if session_id not in self.sessions:
            return []
        return self.sessions[session_id]['messages']
    
    def update_context(self, session_id: str, key: str, value: Any) -> None:
        """Update conversation context."""
        if session_id not in self.sessions:
            self.sessions[session_id] = {
                'messages': [],
                'context': {},
                'created_at': datetime.now()
            }
        self.sessions[session_id]['context'][key] = value
    
    def get_context(self, session_id: str) -> Dict[str, Any]:
        """Get conversation context."""
        if session_id not in self.sessions:
            return {}
        return self.sessions[session_id]['context']
