"""Retrieval tool for ReAct agent."""

import logging
from typing import Any, Dict, List, Optional
from tools.base import Tool
from services.retriever import RetrieverService
from models import QueryIntent
from state import Document

logger = logging.getLogger(__name__)


class RetrievalTool(Tool):
    """Tool for retrieving documents from vector database."""
    
    def __init__(self, retriever: RetrieverService):
        self._retriever = retriever
    
    @property
    def name(self) -> str:
        return "retrieve_documents"
    
    @property
    def description(self) -> str:
        return "Search the vector database for relevant educational content. Returns documents with content and metadata."
    
    @property
    def parameters_schema(self) -> Dict[str, Any]:
        return {
            "query": {
                "type": "string",
                "description": "Search query to find relevant documents",
                "required": True,
            },
            "class_name": {
                "type": "string",
                "description": "Optional class name to filter by (e.g., 'Class 10', 'Civil')",
                "required": False,
            },
            "subject": {
                "type": "string",
                "description": "Optional subject to filter by",
                "required": False,
            },
            "chapter": {
                "type": "string",
                "description": "Optional chapter name to filter by",
                "required": False,
            },
            "topics": {
                "type": "string",
                "description": "Optional topics to filter by",
                "required": False,
            },
            "lecture_id": {
                "type": "integer",
                "description": "Optional lecture ID to filter by",
                "required": False,
            },
            "trainer_name": {
                "type": "string",
                "description": "Optional trainer name to filter by",
                "required": False,
            },
            "class_id": {
                "type": "integer",
                "description": "Optional class ID to filter by",
                "required": False,
            },
            "subject_id": {
                "type": "integer",
                "description": "Optional subject ID to filter by",
                "required": False,
            },
        }
    
    async def execute(
        self, 
        query: str, 
        filters: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> str:
        """
        Execute retrieval. 
        Strictly uses the injected 'filters' (from user request) for Pinecone search.
        LLM provided arguments (kwargs) are ignored for filtering to enforce user context.
        """
        logger.info("[TRACE] RetrievalTool.execute started for query: %s", query)
        try:
            # filters argument takes precedence and is used exclusively for Pinecone search.
            if kwargs:
                logger.info("Ignoring LLM-extracted filters for search: %s", kwargs)

            docs: List[Document] = await self._retriever.retrieve(
                query_en=query,
                filters=filters,
                intent=QueryIntent.CONCEPT_EXPLANATION,
            )
            
            if not docs:
                return f"No documents found for query: '{query}'"
            
            return self.format_documents(docs, min_score=0.450)
        except Exception as exc:
            logger.error("[TRACE] RetrievalTool.execute FAILED: %s", exc)
            return f"Error during retrieval: {str(exc)}"

    @staticmethod
    def format_documents(docs: List[Document], min_score: float = 0.0) -> str:
        """Format documents for agent observation with minimal metadata."""
        if not docs:
            return "No documents found."
            
        # Filter docs by score
        docs = [d for d in docs if d.get("score", 0.0) >= min_score]
        
        if not docs:
            return f"No documents found with score >= {min_score}."
            
        result = f"Found {len(docs)} relevant documents (Top 5 shown):\n\n"
        for i, doc in enumerate(docs[:5], 1):
            text_content = doc.get("text", "")
            # Only include Text and Score to avoid LLM distraction
            result += f"Source {i} [Score: {doc.get('score', 0):.2f}]: {text_content}\n\n"
        
        if len(docs) > 5:
            result += f"... and {len(docs) - 5} more documents\n"
            
        return result


__all__ = ["RetrievalTool"]
