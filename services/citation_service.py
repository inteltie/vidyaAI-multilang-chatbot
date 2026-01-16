"""Service for extracting and formatting citations from retrieval observations."""

import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


class CitationService:
    """Service to parse and standardize citations from agent observations."""

    @staticmethod
    def extract_citations(
        reasoning_chain: List[Dict[str, Any]], 
        source_documents: List[Any], 
        min_score: float = 0.0
    ) -> List[Dict[str, Any]]:
        """
        Extract citations by matching 'Source {i}' labels in observations back to source_documents.
        
        Args:
            reasoning_chain: List of agent reasoning steps (action/observation)
            source_documents: The list of Document objects from the state (containing full metadata).
            min_score: Minimum relevance score to include a citation.
            
        Returns:
            List of unique citation dictionaries with full metadata.
        """
        if not source_documents:
            return []

        cited_doc_ids = set()
        unique_citations = []
        
        for step in reasoning_chain:
            action = step.get("action")
            
            # We strictly extract from 'retrieve_documents' action
            if action == "retrieve_documents":
                observation = step.get("observation", "")
                
                # Split by lines to find labels like "Source 1"
                lines = observation.split("\n")
                
                for line in lines:
                    line = line.strip()
                    
                    # Pattern: "Source {i} [Score: {score}]"
                    if "Source" in line and "[Score:" in line:
                        try:
                            # Extract Index (e.g., "Source 1" -> 1)
                            label_part = line.split("[")[0].strip() # "Source 1"
                            idx_str = label_part.split("Source")[-1].strip()
                            idx = int(idx_str)
                            
                            # Validate index (1-based from tool output)
                            if 1 <= idx <= len(source_documents):
                                doc = source_documents[idx-1]
                                
                                # Check score threshold
                                score = doc.get("score", 0.0)
                                if score >= min_score:
                                    doc_id = doc.get("id")
                                    if doc_id and doc_id not in cited_doc_ids:
                                        cited_doc_ids.add(doc_id)
                                        
                                        meta = doc.get("metadata", {}) or {}
                                        unique_citations.append({
                                            "id": doc_id,
                                            "score": score,
                                            "lecture_id": str(meta.get("lecture_id")) if meta.get("lecture_id") is not None else None,
                                            "transcript_id": str(meta.get("transcript_id")) if meta.get("transcript_id") is not None else None,
                                            "chunk_id": str(meta.get("chunk_id")) if meta.get("chunk_id") is not None else None,
                                            "subject": meta.get("subject"),
                                            "subject_id": meta.get("subject_id"),
                                            "topics": str(meta.get("topics")) if meta.get("topics") is not None else None,
                                            "chapter": meta.get("chapter"),
                                            "class_name": meta.get("class_name"),
                                            "class_id": meta.get("class_id"),
                                            "teacher_name": meta.get("teacher_name"),
                                            "teacher_id": meta.get("teacher_id"),
                                        })
                        except (ValueError, IndexError, AttributeError):
                            continue
        
        # Sort by score descending
        unique_citations.sort(key=lambda x: x["score"], reverse=True)
        return unique_citations


__all__ = ["CitationService"]
