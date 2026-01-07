"""Service for extracting and formatting citations from retrieval observations."""

import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


class CitationService:
    """Service to parse and standardize citations from agent observations."""

    @staticmethod
    def extract_citations(reasoning_chain: List[Dict[str, Any]], min_score: float = 0.0) -> List[Dict[str, Any]]:
        """
        Extract citations from retrieved documents in reasoning chain.
        
        Args:
            reasoning_chain: List of agent reasoning steps (action/observation)
            min_score: Minimum relevance score to include a citation.
            
        Returns:
            List of unique citation dictionaries with standardized fields.
        """
        citations = []
        
        for step in reasoning_chain:
            action = step.get("action")
            
            # We strictly extract from 'retrieve_documents' action
            if action == "retrieve_documents":
                observation = step.get("observation", "")
                
                # Split by lines and parse
                lines = observation.split("\n")
                current_doc = {}
                
                for line in lines:
                    line = line.strip()
                    
                    # Detect start of a new document (e.g., "1. [Score: 0.85]")
                    if "Score:" in line and "[" in line and "]" in line:
                        # Save previous document if valid
                        if current_doc and "lecture_id" in current_doc:
                            if current_doc.get("score", 0.0) >= min_score:
                                citations.append({
                                    "id": f"doc_{current_doc.get('lecture_id', 'unknown')}",
                                    "score": current_doc.get("score", 0.0),
                                    **current_doc
                                })
                            current_doc = {} # Reset
                        
                        # Start new document
                        try:
                            score_str = line.split("Score:")[1].split("]")[0].strip()
                            current_doc["score"] = float(score_str)
                        except (ValueError, IndexError):
                            current_doc["score"] = 0.0
                            
                    # Field Extraction
                    elif line.startswith("Lecture ID:"):
                        val = line.split(":", 1)[1].strip()
                        if val != "N/A": current_doc["lecture_id"] = val
                    
                    elif line.startswith("Transcript ID:"):
                        val = line.split(":", 1)[1].strip()
                        if val != "N/A": current_doc["transcript_id"] = val
                    
                    elif line.startswith("Chunk ID:"):
                        val = line.split(":", 1)[1].strip()
                        if val != "N/A": current_doc["chunk_id"] = val
                    
                    elif line.startswith("Subject ID:"):
                        val = line.split(":", 1)[1].strip()
                        if val != "N/A": current_doc["subject_id"] = val
                    
                    elif line.startswith("Topics:"):
                        val = line.split(":", 1)[1].strip()
                        if val != "N/A": current_doc["topics"] = val

                    elif line.startswith("Chapter:"):
                        val = line.split(":", 1)[1].strip()
                        if val != "N/A": current_doc["chapter"] = val
                    
                    elif line.startswith("Class Name:"):
                        val = line.split(":", 1)[1].strip()
                        if val != "N/A": current_doc["class_name"] = val

                    elif line.startswith("Class ID:"):
                        val = line.split(":", 1)[1].strip()
                        if val != "N/A": current_doc["class_id"] = val

                    elif line.startswith("Teacher Name:"):
                        val = line.split(":", 1)[1].strip()
                        if val != "N/A": current_doc["teacher_name"] = val

                    elif line.startswith("Teacher ID:"):
                        val = line.split(":", 1)[1].strip()
                        if val != "N/A": current_doc["teacher_id"] = val
                    
                    elif line.startswith("Subject:"):
                        val = line.split(":", 1)[1].strip()
                        if val != "N/A": current_doc["subject"] = val
                
                # Append last document after loop
                if current_doc and "lecture_id" in current_doc:
                    if current_doc.get("score", 0.0) >= min_score:
                        citations.append({
                            "id": f"doc_{current_doc.get('lecture_id', 'unknown')}",
                            "score": current_doc.get("score", 0.0),
                            **current_doc
                        })
        
        # Deduplicate citations by lecture_id
        seen = set()
        unique_citations = []
        for citation in citations:
            key = citation.get("lecture_id")
            if key and key not in seen:
                seen.add(key)
                unique_citations.append(citation)
        
        return unique_citations


__all__ = ["CitationService"]
