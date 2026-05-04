"""
RAG (Retrieval-Augmented Generation) Pipeline
"""

from typing import List, Dict, Any, Optional
import time
from .embeddings import EmbeddingManager
from .utils import setup_logger, load_config

logger = setup_logger(__name__)

# Import LLM providers
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logger.warning("OpenAI not available, will use fallback")


class RAGPipeline:
    """
    Retrieval-Augmented Generation pipeline for PizzaBot.
    
    Workflow:
    1. Query: User asks a question
    2. Retrieve: Find relevant documents from vector DB
    3. Augment: Add retrieved context to prompt
    4. Generate: LLM generates response with context
    """
    
    def __init__(
        self,
        embedding_manager: EmbeddingManager,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize RAG pipeline.
        
        Args:
            embedding_manager: EmbeddingManager instance
            config: Configuration dictionary
        """
        self.embedding_manager = embedding_manager
        self.config = config or load_config()
        
        # RAG parameters
        self.top_k = self.config['rag']['top_k']
        self.chunk_size = self.config['rag']['chunk_size']
        self.similarity_threshold = self.config['rag']['similarity_threshold']
        self.max_context_length = self.config['rag']['max_context_length']
        
        # LLM settings
        self.llm_model = self.config['models']['llm_model']
        self.llm_provider = self.config['models']['llm_provider']
        self.max_tokens = self.config['generation']['max_tokens']
        self.temperature = self.config['generation']['temperature']
        
        # Initialize LLM client
        if self.llm_provider == "openai" and OPENAI_AVAILABLE:
            self.llm_client = openai.OpenAI()
        else:
            self.llm_client = None
            logger.warning("LLM client not initialized, using fallback responses")
    
    def retrieve(self, query: str) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents for query.
        
        Args:
            query: User query
            
        Returns:
            List of relevant documents
        """
        start_time = time.time()
        
        documents = self.embedding_manager.retrieve_context(
            query=query,
            top_k=self.top_k,
            similarity_threshold=self.similarity_threshold
        )
        
        retrieval_time = time.time() - start_time
        logger.info(f"Retrieved {len(documents)} documents in {retrieval_time:.3f}s")
        
        return documents
    
    def augment_prompt(
        self,
        query: str,
        context_docs: List[Dict[str, Any]],
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> str:
        """
        Augment prompt with retrieved context and conversation history.
        
        Args:
            query: User query
            context_docs: Retrieved context documents
            conversation_history: Previous messages
            
        Returns:
            Augmented prompt
        """
        # Build context from retrieved documents
        context = "\n\n".join([
            f"[Context {i+1}] {doc['text']}"
            for i, doc in enumerate(context_docs)
        ])
        
        # Truncate if too long
        if len(context) > self.max_context_length:
            context = context[:self.max_context_length] + "..."
        
        # Build conversation context
        history_text = ""
        if conversation_history:
            recent_history = conversation_history[-5:]  # Last 5 messages
            history_text = "\n".join([
                f"{msg['role']}: {msg['content']}"
                for msg in recent_history
            ])
        
        # Build full prompt
        system_prompt = """You are PizzaBot, a friendly and helpful pizza ordering assistant.
Your job is to help customers order pizza, answer questions about the menu, and provide information.

Guidelines:
- Be conversational and friendly
- Use the provided context to answer questions accurately
- If ordering, confirm the pizza type, size, and quantity
- Ask for delivery address if needed
- If you don't know something, admit it rather than making up information
- Keep responses concise but helpful"""
        
        user_prompt = f"""Context information:
{context}

Previous conversation:
{history_text}

Customer question: {query}

Please provide a helpful response based on the context and conversation history."""
        
        return system_prompt, user_prompt
    
    def generate(
        self,
        query: str,
        context_docs: List[Dict[str, Any]],
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> Dict[str, Any]:
        """
        Generate response using LLM with retrieved context.
        
        Args:
            query: User query
            context_docs: Retrieved context documents
            conversation_history: Previous messages
            
        Returns:
            Generated response with metadata
        """
        start_time = time.time()
        
        # Augment prompt
        system_prompt, user_prompt = self.augment_prompt(
            query, context_docs, conversation_history
        )
        
        # Generate response
        if self.llm_client and self.llm_provider == "openai":
            try:
                response = self.llm_client.chat.completions.create(
                    model=self.llm_model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    max_tokens=self.max_tokens,
                    temperature=self.temperature
                )
                
                text = response.choices[0].message.content
                tokens_used = response.usage.total_tokens
                
            except Exception as e:
                logger.error(f"LLM generation error: {e}")
                text = self._generate_fallback_response(query, context_docs)
                tokens_used = 0
        else:
            text = self._generate_fallback_response(query, context_docs)
            tokens_used = 0
        
        generation_time = time.time() - start_time
        
        return {
            'response': text,
            'context_docs': context_docs,
            'tokens_used': tokens_used,
            'generation_time': generation_time
        }
    
    def _generate_fallback_response(
        self,
        query: str,
        context_docs: List[Dict[str, Any]]
    ) -> str:
        """
        Generate fallback response without LLM.
        
        Args:
            query: User query
            context_docs: Retrieved context
            
        Returns:
            Fallback response
        """
        if not context_docs:
            return "I'm sorry, I don't have enough information to answer that question. Could you please rephrase or ask something else?"
        
        # Use first context document as response basis
        first_doc = context_docs[0]['text']
        
        return f"Based on our menu and policies: {first_doc}\n\nIs there anything specific you'd like to know more about?"
    
    def query(
        self,
        user_query: str,
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> Dict[str, Any]:
        """
        Complete RAG pipeline: retrieve, augment, generate.
        
        Args:
            user_query: User question
            conversation_history: Previous messages
            
        Returns:
            Response with metadata
        """
        start_time = time.time()
        
        # Step 1: Retrieve relevant documents
        context_docs = self.retrieve(user_query)
        
        # Step 2 & 3: Augment and generate
        result = self.generate(user_query, context_docs, conversation_history)
        
        # Add total pipeline time
        result['pipeline_time'] = time.time() - start_time
        
        logger.info(
            f"RAG pipeline completed in {result['pipeline_time']:.3f}s "
            f"({len(context_docs)} docs retrieved)"
        )
        
        return result
