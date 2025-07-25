import os
from qdrant_client import QdrantClient


OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL")

VOYAGE_AI_SECRET_KEY = os.getenv("VOYAGE_AI_SECRET_KEY")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

COLLECTION_NAME = os.getenv("COLLECTION_NAME", "documents")

QDRANT_URL = os.getenv("QDRANT_URL")


qdrant_client = QdrantClient(url=QDRANT_URL)

supported_types = ["text/plain", "text/markdown", "application/octet-stream"]

max_file_size = 10 * 1024 * 1024  # 10MB
