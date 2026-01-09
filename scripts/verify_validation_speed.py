import asyncio
import logging
import os
import sys
from time import perf_counter
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.response_validator import ResponseValidator, ValidationResult

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from config import settings

async def verify_validation_speed():
    load_dotenv()
    
    # Using configured validator model
    llm_fast = ChatOpenAI(model=settings.validator_model_name, temperature=0)
    validator = ResponseValidator(llm_fast)
    
    print("\n--- Testing Validation Performance (gpt-4o-mini) ---")
    
    query = "What is the capital of France?"
    response = "The capital of France is Paris."
    documents = [{"text": "France is a country in Europe. Its capital is Paris.", "metadata": {"subject": "Geography"}}]
    
    start = perf_counter()
    result = await validator.validate(query, response, documents, ["Geography"])
    duration = perf_counter() - start
    
    print(f"\nValidation Duration: {duration:.3f}s")
    print(f"Is Valid: {result.is_valid}")
    print(f"Reasoning: {result.reasoning}")
    
    if duration < 3.0:
        print("✅ Validation is fast (< 3s).")
    else:
        print("⚠️ Validation took longer than expected (> 3s).")

if __name__ == "__main__":
    asyncio.run(verify_validation_speed())
