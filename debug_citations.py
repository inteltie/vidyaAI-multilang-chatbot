
import asyncio
from services.citation_service import CitationService

def test_citation_parsing():
    print("=== TEST 1: Multiple Different Lectures (Should extract 3 citations) ===")
    obs_diff = """
    1. [Score: 0.95]
       Lecture ID: 101
       Subject: Math
       Content: Doc 1 content...

    2. [Score: 0.90]
       Lecture ID: 102
       Subject: Math
       Content: Doc 2 content...

    3. [Score: 0.85]
       Lecture ID: 103
       Subject: Math
       Content: Doc 3 content...
    """
    chain_diff = [{"action": "retrieve_documents", "observation": obs_diff}]
    citations_diff = CitationService.extract_citations(chain_diff)
    print(f"Extracted: {len(citations_diff)}")
    for c in citations_diff:
        print(f" - {c.get('id')} (Lecture: {c.get('lecture_id')})")


    print("\n=== TEST 2: Multiple Chunks from SAME Lecture (Should extract 1 citation) ===")
    obs_same = """
    1. [Score: 0.95]
       Lecture ID: 200
       Chunk ID: 200_1
       Content: Chunk 1...

    2. [Score: 0.94]
       Lecture ID: 200
       Chunk ID: 200_2
       Content: Chunk 2...

    3. [Score: 0.93]
       Lecture ID: 200
       Chunk ID: 200_3
       Content: Chunk 3...
    """
    chain_same = [{"action": "retrieve_documents", "observation": obs_same}]
    citations_same = CitationService.extract_citations(chain_same)
    print(f"Extracted: {len(citations_same)}")
    for c in citations_same:
        print(f" - {c.get('id')} (Lecture: {c.get('lecture_id')})")

if __name__ == "__main__":
    test_citation_parsing()
