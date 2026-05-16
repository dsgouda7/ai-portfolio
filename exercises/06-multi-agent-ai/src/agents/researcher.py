"""
Researcher Agent - Gather information and context
"""

from typing import Dict, Any, List
from src.agents.base import BaseAgent


class ResearcherAgent(BaseAgent):
    """
    ResearcherAgent gathers information and provides context for tasks
    """
    
    def __init__(self, name: str, config: Dict[str, Any]):
        super().__init__(name, config)
        self.max_search_depth = config.get("max_search_depth", 3)
        self.knowledge_base = self._initialize_knowledge_base()
    
    def process(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Research and gather information
        
        Args:
            task: Task with 'query' and optional 'depth', 'sources'
            
        Returns:
            Dictionary with research findings
        """
        query = task.get("query", "")
        depth = min(task.get("depth", 2), self.max_search_depth)
        sources = task.get("sources", ["knowledge_base", "context"])
        
        self.logger.info(f"Researching: {query}")
        
        # Gather information
        findings = self._gather_information(query, depth, sources)
        
        # Synthesize findings
        synthesis = self._synthesize_findings(findings)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(synthesis)
        
        response = {
            "status": "success",
            "query": query,
            "findings": findings,
            "synthesis": synthesis,
            "recommendations": recommendations,
            "sources_consulted": sources,
            "depth": depth
        }
        
        self.logger.info(f"Research complete: {len(findings)} findings")
        return response
    
    def _initialize_knowledge_base(self) -> Dict[str, Any]:
        """Initialize knowledge base with common information"""
        return {
            "machine_learning": {
                "description": "Field of AI focused on learning from data",
                "subtopics": ["supervised", "unsupervised", "reinforcement"],
                "applications": ["prediction", "classification", "clustering"]
            },
            "api_design": {
                "description": "Designing RESTful and GraphQL APIs",
                "best_practices": ["versioning", "authentication", "rate_limiting"],
                "patterns": ["REST", "GraphQL", "gRPC"]
            },
            "data_processing": {
                "description": "ETL and data transformation",
                "tools": ["pandas", "spark", "airflow"],
                "patterns": ["batch", "streaming", "real-time"]
            },
            "testing": {
                "description": "Software testing strategies",
                "types": ["unit", "integration", "e2e"],
                "frameworks": ["pytest", "unittest", "selenium"]
            }
        }
    
    def _gather_information(
        self, query: str, depth: int, sources: List[str]
    ) -> List[Dict[str, Any]]:
        """Gather information from various sources"""
        findings = []
        
        # Search knowledge base
        if "knowledge_base" in sources:
            kb_findings = self._search_knowledge_base(query, depth)
            findings.extend(kb_findings)
        
        # Search context (simulated)
        if "context" in sources:
            context_findings = self._search_context(query)
            findings.extend(context_findings)
        
        # Web search (simulated - in production would use actual search APIs)
        if "web" in sources:
            web_findings = self._simulated_web_search(query)
            findings.extend(web_findings)
        
        return findings
    
    def _search_knowledge_base(self, query: str, depth: int) -> List[Dict[str, Any]]:
        """Search internal knowledge base"""
        results = []
        query_lower = query.lower()
        
        for topic, info in self.knowledge_base.items():
            # Simple keyword matching
            if query_lower in topic.lower() or any(
                query_lower in str(v).lower() for v in info.values()
            ):
                result = {
                    "source": "knowledge_base",
                    "topic": topic,
                    "content": info,
                    "relevance": self._calculate_relevance(query, topic, info),
                    "depth": 1
                }
                results.append(result)
                
                # Deep dive if depth > 1
                if depth > 1 and "subtopics" in info:
                    for subtopic in info["subtopics"]:
                        results.append({
                            "source": "knowledge_base",
                            "topic": f"{topic}.{subtopic}",
                            "content": f"Subtopic of {topic}",
                            "relevance": 0.7,
                            "depth": 2
                        })
        
        # Sort by relevance
        results.sort(key=lambda x: x["relevance"], reverse=True)
        return results[:5]  # Top 5 results
    
    def _search_context(self, query: str) -> List[Dict[str, Any]]:
        """Search current context and state"""
        # In production, this would search conversation history, user context, etc.
        return [{
            "source": "context",
            "content": f"Context information related to: {query}",
            "relevance": 0.6,
            "depth": 1
        }]
    
    def _simulated_web_search(self, query: str) -> List[Dict[str, Any]]:
        """Simulate web search results"""
        # In production, this would use actual search APIs
        return [{
            "source": "web",
            "title": f"Search result for: {query}",
            "content": f"Relevant information about {query}",
            "url": f"https://example.com/search?q={query}",
            "relevance": 0.5,
            "depth": 1
        }]
    
    def _calculate_relevance(self, query: str, topic: str, info: Dict[str, Any]) -> float:
        """Calculate relevance score"""
        query_lower = query.lower()
        
        # Exact topic match
        if query_lower == topic.lower():
            return 1.0
        
        # Partial topic match
        if query_lower in topic.lower():
            return 0.8
        
        # Content match
        content_str = str(info).lower()
        if query_lower in content_str:
            return 0.6
        
        # Word overlap
        query_words = set(query_lower.split())
        topic_words = set(topic.lower().split())
        overlap = len(query_words & topic_words) / len(query_words) if query_words else 0
        
        return overlap * 0.5
    
    def _synthesize_findings(self, findings: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Synthesize research findings into coherent summary"""
        if not findings:
            return {
                "summary": "No relevant information found",
                "key_points": [],
                "confidence": 0.0
            }
        
        # Extract key points
        key_points = []
        sources_used = set()
        
        for finding in findings[:3]:  # Top 3 findings
            topic = finding.get("topic", "Unknown")
            content = finding.get("content", {})
            
            if isinstance(content, dict):
                key_point = content.get("description", str(content))
            else:
                key_point = str(content)
            
            key_points.append({
                "topic": topic,
                "point": key_point,
                "relevance": finding.get("relevance", 0.5)
            })
            
            sources_used.add(finding.get("source", "unknown"))
        
        # Calculate overall confidence
        avg_relevance = sum(f.get("relevance", 0) for f in findings) / len(findings)
        confidence = min(1.0, avg_relevance * 1.2)
        
        return {
            "summary": f"Found {len(findings)} relevant results from {len(sources_used)} sources",
            "key_points": key_points,
            "confidence": round(confidence, 2),
            "sources": list(sources_used)
        }
    
    def _generate_recommendations(self, synthesis: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations"""
        confidence = synthesis.get("confidence", 0)
        key_points = synthesis.get("key_points", [])
        
        recommendations = []
        
        if confidence >= 0.8:
            recommendations.append("High confidence in findings - proceed with implementation")
        elif confidence >= 0.5:
            recommendations.append("Moderate confidence - consider additional research")
        else:
            recommendations.append("Low confidence - expand search or consult experts")
        
        # Recommendations based on key points
        for kp in key_points[:2]:
            topic = kp.get("topic", "")
            recommendations.append(f"Consider exploring: {topic}")
        
        return recommendations
    
    def add_to_knowledge_base(self, topic: str, info: Dict[str, Any]) -> None:
        """Add new information to knowledge base"""
        self.knowledge_base[topic] = info
        self.logger.info(f"Added {topic} to knowledge base")
