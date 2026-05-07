"""
Data loading and preprocessing for PizzaBot knowledge base
"""

import json
import os
from typing import Dict, List, Any
from .utils import setup_logger

logger = setup_logger(__name__)


class DataLoader:
    """Load and manage PizzaBot knowledge base."""
    
    def __init__(self, knowledge_base_dir: str = "knowledge_base"):
        self.knowledge_base_dir = knowledge_base_dir
        self.menu = None
        self.faqs = None
        self.policies = None
        
    def load_menu(self) -> Dict[str, Any]:
        """
        Load pizza menu from JSON file.
        
        Returns:
            Menu dictionary
        """
        menu_path = os.path.join(self.knowledge_base_dir, "menu.json")
        
        if not os.path.exists(menu_path):
            logger.warning(f"Menu file not found at {menu_path}")
            return self._get_default_menu()
        
        with open(menu_path, 'r') as f:
            self.menu = json.load(f)
        
        logger.info(f"Loaded menu with {len(self.menu)} items")
        return self.menu
    
    def load_faqs(self) -> List[Dict[str, str]]:
        """
        Load FAQs from text file.
        
        Returns:
            List of FAQ dictionaries
        """
        faq_path = os.path.join(self.knowledge_base_dir, "faqs.txt")
        
        if not os.path.exists(faq_path):
            logger.warning(f"FAQ file not found at {faq_path}")
            return self._get_default_faqs()
        
        faqs = []
        with open(faq_path, 'r') as f:
            content = f.read()
            
        # Parse Q&A pairs (format: Q: question\nA: answer\n)
        sections = content.split('\n\n')
        for section in sections:
            if 'Q:' in section and 'A:' in section:
                lines = section.strip().split('\n')
                question = lines[0].replace('Q:', '').strip()
                answer = lines[1].replace('A:', '').strip()
                faqs.append({'question': question, 'answer': answer})
        
        self.faqs = faqs
        logger.info(f"Loaded {len(faqs)} FAQs")
        return faqs
    
    def load_policies(self) -> str:
        """
        Load policies from text file.
        
        Returns:
            Policies text
        """
        policy_path = os.path.join(self.knowledge_base_dir, "policies.txt")
        
        if not os.path.exists(policy_path):
            logger.warning(f"Policy file not found at {policy_path}")
            return self._get_default_policies()
        
        with open(policy_path, 'r') as f:
            self.policies = f.read()
        
        logger.info("Loaded policies")
        return self.policies
    
    def get_all_documents(self) -> List[str]:
        """
        Get all knowledge base documents as text chunks.
        
        Returns:
            List of document strings
        """
        documents = []
        
        # Add menu items
        menu = self.load_menu()
        for name, data in menu.items():
            doc = f"Pizza: {name}\n"
            doc += f"Price: ${data['price']:.2f}\n"
            doc += f"Description: {data['description']}\n"
            doc += f"Ingredients: {', '.join(data['ingredients'])}\n"
            doc += f"Sizes: {', '.join(data['sizes'])}"
            documents.append(doc)
        
        # Add FAQs
        faqs = self.load_faqs()
        for faq in faqs:
            doc = f"Q: {faq['question']}\nA: {faq['answer']}"
            documents.append(doc)
        
        # Add policies
        policies = self.load_policies()
        # Split policies into paragraphs
        policy_sections = [p.strip() for p in policies.split('\n\n') if p.strip()]
        documents.extend(policy_sections)
        
        logger.info(f"Prepared {len(documents)} documents for embedding")
        return documents
    
    @staticmethod
    def _get_default_menu() -> Dict[str, Any]:
        """Get default menu if file not found."""
        return {
            "Margherita": {
                "price": 12.99,
                "description": "Classic tomato sauce, mozzarella, and basil",
                "ingredients": ["tomato sauce", "mozzarella", "basil", "olive oil"],
                "sizes": ["small", "medium", "large"]
            },
            "Pepperoni": {
                "price": 14.99,
                "description": "Tomato sauce, mozzarella, and pepperoni",
                "ingredients": ["tomato sauce", "mozzarella", "pepperoni"],
                "sizes": ["small", "medium", "large"]
            },
            "Vegetarian": {
                "price": 13.99,
                "description": "Fresh vegetables with mozzarella",
                "ingredients": ["tomato sauce", "mozzarella", "bell peppers", "mushrooms", "onions", "olives"],
                "sizes": ["small", "medium", "large"]
            },
            "Hawaiian": {
                "price": 14.99,
                "description": "Ham and pineapple with mozzarella",
                "ingredients": ["tomato sauce", "mozzarella", "ham", "pineapple"],
                "sizes": ["small", "medium", "large"]
            },
            "Meat Lovers": {
                "price": 16.99,
                "description": "Loaded with pepperoni, sausage, bacon, and ham",
                "ingredients": ["tomato sauce", "mozzarella", "pepperoni", "sausage", "bacon", "ham"],
                "sizes": ["small", "medium", "large"]
            }
        }
    
    @staticmethod
    def _get_default_faqs() -> List[Dict[str, str]]:
        """Get default FAQs if file not found."""
        return [
            {
                "question": "What are your delivery hours?",
                "answer": "We deliver from 11 AM to 11 PM, 7 days a week."
            },
            {
                "question": "How long does delivery take?",
                "answer": "Average delivery time is 30-45 minutes depending on your location."
            },
            {
                "question": "Do you offer gluten-free options?",
                "answer": "Yes! We offer gluten-free crusts for all our pizzas at no extra charge."
            }
        ]
    
    @staticmethod
    def _get_default_policies() -> str:
        """Get default policies if file not found."""
        return """
Delivery Policy:
We deliver within a 5-mile radius of our location. Delivery fee is $2.99.
Free delivery on orders over $25.

Refund Policy:
If you're not satisfied with your order, please call us within 30 minutes
of delivery for a full refund or replacement.

Cancellation Policy:
Orders can be cancelled within 5 minutes of placement for a full refund.
"""
