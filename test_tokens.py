import time
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))

print("Testing get_num_tokens...")
start = time.perf_counter()
# langchain-openai internal token counting
try:
    tokens = llm.get_num_tokens("Hello world. How are you?")
    duration = time.perf_counter() - start
    print(f"Tokens: {tokens}, Time: {duration:.3f}s")
except Exception as e:
    print(f"Error: {e}")

start = time.perf_counter()
from langchain_core.messages import HumanMessage, AIMessage, trim_messages
messages = [HumanMessage(content="Hello"), AIMessage(content="Hi there!")]
print("Testing trim_messages...")
trimmed = trim_messages(messages, max_tokens=10, strategy="last", token_counter=llm)
duration = time.perf_counter() - start
print(f"Trimmed: {trimmed}, Time: {duration:.3f}s")
