"""
Chatbot models including conversation state and intent detection
"""

from typing import Dict, Any, List, Optional
import re
from datetime import datetime
from .rag import RAGPipeline
from .utils import setup_logger, extract_order_items, calculate_total, validate_delivery_address

logger = setup_logger(__name__)


class IntentDetector:
    """Detect user intent from message."""
    
    def __init__(self, confidence_threshold: float = 0.6):
        self.confidence_threshold = confidence_threshold
        
        # Intent patterns (keyword-based for simplicity)
        self.intent_patterns = {
            'order_pizza': [
                r'\border\b', r'\bbuy\b', r'\bwant\b', r'\bget\b',
                r'\bpizza\b.*\bplease\b', r'\bi\'ll have\b'
            ],
            'check_menu': [
                r'\bmenu\b', r'\bwhat.*\bhave\b', r'\boptions\b',
                r'\bpizzas?\b.*\bavailable\b', r'\bwhat.*\boffer\b'
            ],
            'ask_question': [
                r'\bwhat\b', r'\bhow\b', r'\bwhen\b', r'\bwhere\b',
                r'\bcan i\b', r'\bdo you\b', r'\btell me\b'
            ],
            'track_order': [
                r'\btrack\b', r'\bwhere.*\border\b', r'\bstatus\b',
                r'\bdelivery.*\btime\b', r'\bwhen.*\barrive\b'
            ],
            'cancel_order': [
                r'\bcancel\b', r'\bdon\'t want\b', r'\bnevermind\b',
                r'\bremove\b.*\border\b'
            ],
            'complain': [
                r'\bcomplaint\b', r'\bproblem\b', r'\bwrong\b',
                r'\bcold\b', r'\blate\b', r'\bnot happy\b'
            ],
            'general_chat': [
                r'\bhello\b', r'\bhi\b', r'\bhey\b', r'\bthanks?\b',
                r'\bbye\b', r'\bgoodbye\b'
            ]
        }
    
    def detect(self, message: str) -> Dict[str, Any]:
        """
        Detect intent from user message.
        
        Args:
            message: User message
            
        Returns:
            Intent detection result with confidence
        """
        message_lower = message.lower()
        
        # Check each intent
        intent_scores = {}
        for intent, patterns in self.intent_patterns.items():
            score = 0
            for pattern in patterns:
                if re.search(pattern, message_lower):
                    score += 1
            
            # Normalize score
            if patterns:
                intent_scores[intent] = score / len(patterns)
        
        # Get highest scoring intent
        if intent_scores:
            best_intent = max(intent_scores, key=intent_scores.get)
            confidence = intent_scores[best_intent]
        else:
            best_intent = 'general_chat'
            confidence = 0.5
        
        return {
            'intent': best_intent,
            'confidence': confidence,
            'all_scores': intent_scores
        }


class OrderValidator:
    """Validate pizza orders."""
    
    def __init__(self, config: Dict[str, Any]):
        self.min_quantity = config['orders']['min_quantity']
        self.max_quantity = config['orders']['max_quantity']
        self.max_pizzas_per_order = config['orders']['max_pizzas_per_order']
        self.require_delivery_address = config['orders']['require_delivery_address']
    
    def validate(
        self,
        order_items: List[Dict[str, Any]],
        delivery_address: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Validate order.
        
        Args:
            order_items: List of order items
            delivery_address: Delivery address
            
        Returns:
            Validation result
        """
        errors = []
        
        # Check if order is empty
        if not order_items:
            errors.append("No items in order")
        
        # Check number of pizzas
        total_pizzas = sum(item['quantity'] for item in order_items)
        if total_pizzas > self.max_pizzas_per_order:
            errors.append(f"Maximum {self.max_pizzas_per_order} pizzas per order")
        
        # Check quantity limits
        for item in order_items:
            if item['quantity'] < self.min_quantity:
                errors.append(f"Minimum quantity is {self.min_quantity}")
            if item['quantity'] > self.max_quantity:
                errors.append(f"Maximum quantity per item is {self.max_quantity}")
        
        # Check delivery address
        if self.require_delivery_address:
            if not delivery_address:
                errors.append("Delivery address required")
            elif not validate_delivery_address(delivery_address):
                errors.append("Invalid delivery address format")
        
        return {
            'valid': len(errors) == 0,
            'errors': errors,
            'total_pizzas': total_pizzas if not errors else 0
        }


class ChatbotEngine:
    """
    Main chatbot engine coordinating RAG, intent detection, and conversation state.
    """
    
    def __init__(
        self,
        rag_pipeline: RAGPipeline,
        config: Dict[str, Any]
    ):
        """
        Initialize chatbot engine.
        
        Args:
            rag_pipeline: RAGPipeline instance
            config: Configuration dictionary
        """
        self.rag_pipeline = rag_pipeline
        self.config = config
        
        # Initialize components
        self.intent_detector = IntentDetector(
            confidence_threshold=config['intents']['confidence_threshold']
        )
        self.order_validator = OrderValidator(config)
        
        # Conversation state
        self.sessions = {}
    
    def process_message(
        self,
        session_id: str,
        message: str
    ) -> Dict[str, Any]:
        """
        Process user message and generate response.
        
        Args:
            session_id: Session identifier
            message: User message
            
        Returns:
            Response with metadata
        """
        # Initialize session if new
        if session_id not in self.sessions:
            self.sessions[session_id] = {
                'messages': [],
                'context': {},
                'order': None,
                'created_at': datetime.now()
            }
        
        session = self.sessions[session_id]
        
        # Detect intent
        intent_result = self.intent_detector.detect(message)
        intent = intent_result['intent']
        
        logger.info(
            f"Session {session_id}: Intent={intent} "
            f"(confidence={intent_result['confidence']:.2f})"
        )
        
        # Add message to history
        session['messages'].append({
            'role': 'user',
            'content': message,
            'intent': intent,
            'timestamp': datetime.now().isoformat()
        })
        
        # Handle based on intent
        if intent == 'order_pizza':
            response = self._handle_order(session_id, message)
        elif intent == 'check_menu':
            response = self._handle_menu_query(session_id, message)
        elif intent == 'track_order':
            response = self._handle_order_tracking(session_id, message)
        elif intent == 'cancel_order':
            response = self._handle_cancellation(session_id, message)
        else:
            # Use RAG for general questions
            response = self._handle_general_query(session_id, message)
        
        # Add response to history
        session['messages'].append({
            'role': 'assistant',
            'content': response['response'],
            'timestamp': datetime.now().isoformat()
        })
        
        # Add intent and session info to response
        response['intent'] = intent
        response['session_id'] = session_id
        
        return response
    
    def _handle_order(self, session_id: str, message: str) -> Dict[str, Any]:
        """Handle pizza order intent."""
        session = self.sessions[session_id]
        
        # Use RAG to get menu context
        rag_result = self.rag_pipeline.query(
            message,
            conversation_history=session['messages']
        )
        
        # Try to extract order items (simplified)
        # In production, this would use more sophisticated NER
        response_text = rag_result['response']
        
        # Check if we need delivery address
        if not session['context'].get('delivery_address'):
            response_text += "\n\nCould you please provide your delivery address?"
        
        return {
            'response': response_text,
            'context_docs': rag_result['context_docs'],
            'tokens_used': rag_result['tokens_used']
        }
    
    def _handle_menu_query(self, session_id: str, message: str) -> Dict[str, Any]:
        """Handle menu inquiry."""
        session = self.sessions[session_id]
        
        rag_result = self.rag_pipeline.query(
            message,
            conversation_history=session['messages']
        )
        
        return {
            'response': rag_result['response'],
            'context_docs': rag_result['context_docs'],
            'tokens_used': rag_result['tokens_used']
        }
    
    def _handle_order_tracking(self, session_id: str, message: str) -> Dict[str, Any]:
        """Handle order tracking."""
        # Simplified - would integrate with actual order system
        return {
            'response': "I don't see any active orders for this session. Would you like to place a new order?",
            'context_docs': [],
            'tokens_used': 0
        }
    
    def _handle_cancellation(self, session_id: str, message: str) -> Dict[str, Any]:
        """Handle order cancellation."""
        session = self.sessions[session_id]
        
        if session.get('order'):
            session['order'] = None
            response = "Your order has been cancelled. Is there anything else I can help you with?"
        else:
            response = "I don't see any active order to cancel. Can I help you with anything else?"
        
        return {
            'response': response,
            'context_docs': [],
            'tokens_used': 0
        }
    
    def _handle_general_query(self, session_id: str, message: str) -> Dict[str, Any]:
        """Handle general questions using RAG."""
        session = self.sessions[session_id]
        
        rag_result = self.rag_pipeline.query(
            message,
            conversation_history=session['messages']
        )
        
        return {
            'response': rag_result['response'],
            'context_docs': rag_result['context_docs'],
            'tokens_used': rag_result['tokens_used']
        }
    
    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session data."""
        return self.sessions.get(session_id)
    
    def clear_session(self, session_id: str) -> None:
        """Clear session data."""
        if session_id in self.sessions:
            del self.sessions[session_id]
