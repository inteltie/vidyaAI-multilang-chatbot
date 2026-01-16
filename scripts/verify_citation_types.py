
import asyncio
from services.citation_service import CitationService

def test_citation_type_casting():
    print("Testing CitationService Type Casting...")
    
    # Mock documents with integer metadata
    source_documents = [
        {
            "id": "doc1",
            "score": 0.9,
            "metadata": {
                "transcript_id": 54,
                "lecture_id": 123,
                "chunk_id": 789,
                "subject": "Physics",
                "topics": "Kinetic Energy"
            }
        }
    ]
    
    # Mock reasoning chain citing Source 1
    reasoning_chain = [
        {
            "action": "retrieve_documents",
            "observation": "Source 1 [Score: 0.9]: Context text"
        }
    ]
    
    citations = CitationService.extract_citations(reasoning_chain, source_documents)
    
    assert len(citations) == 1
    cit = citations[0]
    
    print(f"Citation: {cit}")
    
    # Verify types
    assert isinstance(cit["transcript_id"], str)
    assert cit["transcript_id"] == "54"
    assert isinstance(cit["lecture_id"], str)
    assert cit["lecture_id"] == "123"
    assert isinstance(cit["chunk_id"], str)
    assert cit["chunk_id"] == "789"
    
    print("\n✅ Citation Type Casting Verified!")

if __name__ == "__main__":
    test_citation_type_casting()
