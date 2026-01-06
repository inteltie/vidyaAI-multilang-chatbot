import sys
import os
import logging

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.retriever import _hybrid_scale

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_hybrid_scale():
    print("Testing _hybrid_scale...")
    
    dense = [0.1] * 1536
    alpha = 0.5
    
    # Case 1: Normal sparse vector
    sparse = {"indices": [1, 2, 3], "values": [0.5, 0.6, 0.7]}
    scaled_dense, scaled_sparse = _hybrid_scale(dense, sparse, alpha)
    print(f"Case 1 (Normal): Indices count={len(scaled_sparse['indices'])}, Values count={len(scaled_sparse['values'])}")
    
    # Case 2: Empty sparse vector (simulating empty query or stop words)
    sparse_empty = {"indices": [], "values": []}
    scaled_dense, scaled_sparse = _hybrid_scale(dense, sparse_empty, alpha)
    print(f"Case 2 (Empty): Indices count={len(scaled_sparse['indices'])}, Values count={len(scaled_sparse['values'])}")
    
    if not scaled_sparse['values']:
        print("⚠️  Warning: Scaled sparse vector has no values!")

if __name__ == "__main__":
    test_hybrid_scale()
