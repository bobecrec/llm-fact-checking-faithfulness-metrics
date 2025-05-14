from langchain_ollama import ChatOllama
from langchain.schema import HumanMessage

llm = ChatOllama(model="llama3.2", temperature=1)
