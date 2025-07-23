import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama.embeddings import OllamaEmbeddings
from qdrant_client import QdrantClient

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

COLLECTION_NAME = os.getenv("COLLECTION_NAME", "documents")

LLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
embedder = OllamaEmbeddings(model="nomic-embed-text", base_url=LLAMA_BASE_URL)

qdrant_client = QdrantClient(url="http://vector-db:6333")


supported_types = ["text/plain", "text/markdown", "application/octet-stream"]

max_file_size = 10 * 1024 * 1024  # 10MB