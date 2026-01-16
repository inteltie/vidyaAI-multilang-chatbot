
import asyncio
import logging
from typing import List, Dict, Any
from services.citation_service import CitationService
from tools.retrieval_tool import RetrievalTool

# Mock Document object
class MockDocument(dict):
    def get(self, key, default=None):
        return super().get(key, default)

async def test_citation_linking():
    print("Testing Citation Linking Logic...")
    
    # 1. Create mock documents with full metadata
    source_docs = [
        MockDocument({
            "id": "doc_1",
            "score": 0.95,
            "text": "Photosynthesis is the process by which plants make food.",
            "metadata": {
                "lecture_id": 101,
                "subject": "Biology",
                "chapter": "Plant Life",
                "transcript_id": "T1"
            }
        }),
        MockDocument({
            "id": "doc_2",
            "score": 0.35,  # Should be filtered by tool
            "text": "Low score document.",
            "metadata": {"lecture_id": 102}
        }),
        MockDocument({
            "id": "doc_3",
            "score": 0.85,
            "text": "Chlorophyll is the green pigment in plants.",
            "metadata": {
                "lecture_id": 103,
                "subject": "Biology",
                "chapter": "Plant Life",
                "transcript_id": "T3"
            }
        })
    ]
    
    # 2. Verify RetrievalTool format (LLM's view)
    observation = RetrievalTool.format_documents(source_docs, min_score=0.4)
    print("\n--- LLM Observation (Metadata Hidden) ---")
    print(observation)
    
    assert "Lecture ID" not in observation
    assert "Source 1 [Score: 0.95]" in observation
    assert "Source 2 [Score: 0.85]" in observation # Note: doc_2 filtered, so doc_3 becomes Source 2
    assert "doc_1" not in observation
    
    # 3. Filter docs as RetrieveDocumentsNode would
    filtered_docs = [d for d in source_docs if d.get("score", 0.0) >= 0.4]
    
    # 4. Mock Reasoning Chain (Agent uses Source 1 and Source 2 from the filtered view)
    reasoning_chain = [
        {
            "action": "retrieve_documents",
            "observation": observation
        }
    ]
    
    # 5. Verify CitationService mapping back to metadata
    citations = CitationService.extract_citations(reasoning_chain, filtered_docs, min_score=0.4)
    
    print("\n--- Extracted Citations (Rich Metadata Recovered) ---")
    for c in citations:
        print(f"ID: {c['id']}, Score: {c['score']}, Subject: {c['subject']}, Lecture ID: {c['lecture_id']}")
    
    assert len(citations) == 2
    assert citations[0]["id"] == "doc_1"
    assert citations[0]["lecture_id"] == "101"
    assert citations[1]["id"] == "doc_3"
    assert citations[1]["lecture_id"] == "103"
    
    print("\n✅ Citation Linking Verified Successfully!")

if __name__ == "__main__":
    asyncio.run(test_citation_linking())
