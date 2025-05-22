from langchain_ollama import ChatOllama
from langchain.schema import HumanMessage

llm = ChatOllama(model="llama3.1:8b", temperature=0.8)
