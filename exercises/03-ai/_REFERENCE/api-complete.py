"""
Flask API for PizzaBot
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
import uuid
from datetime import datetime

from .models import ChatbotEngine
from .rag import RAGPipeline
from .embeddings import EmbeddingManager
from .data import DataLoader
from .utils import setup_logger, load_config, ConversationTracker
from .monitoring import (
    track_api_request,
    update_conversation_metrics,
    log_system_info
)

logger = setup_logger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Load configuration
config = load_config()

# Initialize components
logger.info("Initializing PizzaBot components...")

# Data loader
data_loader = DataLoader()

# Embedding manager
embedding_manager = EmbeddingManager(
    model_name=config['models']['embedding_model'],
    persist_directory=config['vector_db']['persist_directory'],
    collection_name=config['vector_db']['collection_name']
)

# Check if vector DB needs initialization
if embedding_manager.get_collection_stats()['count'] == 0:
    logger.info("Initializing vector database with knowledge base...")
    documents = data_loader.get_all_documents()
    embedding_manager.add_documents(documents)
    logger.info("Vector database initialized")

# RAG pipeline
rag_pipeline = RAGPipeline(embedding_manager, config)

# Chatbot engine
chatbot = ChatbotEngine(rag_pipeline, config)

# Conversation tracker
conversation_tracker = ConversationTracker(
    max_history=config['conversation']['max_history']
)

# Log system info
log_system_info(config)

logger.info("PizzaBot initialized successfully!")


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'version': '1.0.0'
    })


@app.route('/chat', methods=['POST'])
@track_api_request
def chat():
    """
    Main chat endpoint.
    
    Request:
        {
            "message": "I want to order a pepperoni pizza",
            "session_id": "optional-session-id"
        }
    
    Response:
        {
            "response": "Great! I can help you order a pepperoni pizza...",
            "session_id": "abc-123",
            "intent": "order_pizza",
            "context_docs": [...],
            "timestamp": "2024-01-01T12:00:00"
        }
    """
    try:
        # Parse request
        data = request.get_json()
        
        if not data or 'message' not in data:
            return jsonify({
                'error': 'Missing required field: message'
            }), 400
        
        message = data['message']
        session_id = data.get('session_id') or str(uuid.uuid4())
        
        logger.info(f"Processing message for session {session_id}")
        
        # Track conversation start
        if session_id not in chatbot.sessions:
            update_conversation_metrics('started')
        
        # Process message
        result = chatbot.process_message(session_id, message)
        
        # Track intent
        if 'intent' in result:
            update_conversation_metrics(
                'active',
                intent=result['intent'],
                confidence=0.8  # Would use actual confidence
            )
        
        # Build response
        response = {
            'response': result['response'],
            'session_id': session_id,
            'intent': result.get('intent', 'unknown'),
            'timestamp': datetime.now().isoformat()
        }
        
        # Add context if requested
        if data.get('include_context', False):
            response['context_docs'] = [
                {
                    'text': doc['text'][:200] + '...',
                    'similarity': doc['similarity']
                }
                for doc in result.get('context_docs', [])
            ]
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error processing chat request: {e}", exc_info=True)
        return jsonify({
            'error': 'Internal server error',
            'message': str(e)
        }), 500


@app.route('/menu', methods=['GET'])
def get_menu():
    """Get pizza menu."""
    try:
        menu = data_loader.load_menu()
        return jsonify({
            'menu': menu,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Error retrieving menu: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/session/<session_id>', methods=['GET'])
def get_session(session_id: str):
    """Get session information."""
    try:
        session = chatbot.get_session(session_id)
        
        if not session:
            return jsonify({'error': 'Session not found'}), 404
        
        return jsonify({
            'session_id': session_id,
            'message_count': len(session['messages']),
            'created_at': session['created_at'].isoformat(),
            'context': session['context']
        })
        
    except Exception as e:
        logger.error(f"Error retrieving session: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/session/<session_id>', methods=['DELETE'])
def clear_session(session_id: str):
    """Clear session data."""
    try:
        chatbot.clear_session(session_id)
        update_conversation_metrics('ended')
        
        return jsonify({
            'message': 'Session cleared',
            'session_id': session_id
        })
        
    except Exception as e:
        logger.error(f"Error clearing session: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/metrics', methods=['GET'])
def metrics():
    """Prometheus metrics endpoint."""
    return generate_latest(), 200, {'Content-Type': CONTENT_TYPE_LATEST}


@app.route('/stats', methods=['GET'])
def stats():
    """Get system statistics."""
    try:
        vector_stats = embedding_manager.get_collection_stats()
        
        return jsonify({
            'vector_db': vector_stats,
            'active_sessions': len(chatbot.sessions),
            'config': {
                'embedding_model': config['models']['embedding_model'],
                'llm_model': config['models']['llm_model'],
                'top_k': config['rag']['top_k']
            }
        })
        
    except Exception as e:
        logger.error(f"Error retrieving stats: {e}")
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    # Run Flask app
    app.run(
        host=config['api']['host'],
        port=config['api']['port'],
        debug=config['api']['debug']
    )
