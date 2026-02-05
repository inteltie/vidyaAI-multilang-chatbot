
import asyncio
import logging
import sys
from tools.web_search_tool import WebSearchTool
from dotenv import load_dotenv

# Setup logging
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger("verify_web_search")

async def verify_tool():
    load_dotenv()
    
    logger.info("Initializing WebSearchTool...")
    tool = WebSearchTool()
    
    query = "What is the latest news about LangChain?"
    logger.info(f"Executing search for query: '{query}'")
    
    try:
        result = await tool.execute(query)
        logger.info("\n=== SEARCH RESULT START ===")
        print(result)
        logger.info("=== SEARCH RESULT END ===\n")
        
        if "WEB_SEARCH_OBSERVATION" in result and "Sources:" in result:
             logger.info("VERIFICATION PASSED: Result contains observation and sources.")
        else:
             logger.warning("VERIFICATION WARNING: Result format might be unexpected check output.")
             
    except Exception as e:
        logger.error(f"VERIFICATION FAILED: {e}")

if __name__ == "__main__":
    asyncio.run(verify_tool())
