# Gera e envia os embeddings para o vector db
from typing import List
from langchain_ollama.embeddings import OllamaEmbeddings
from config import LLAMA_BASE_URL

embedder = OllamaEmbeddings(model="nomic-embed-text", base_url=LLAMA_BASE_URL)


def generate_embeddings(chunks: List[str]) -> None:
    return embedder.embed_documents(chunks)

def generate_embeddings_query(query: str) -> None:
    return embedder.embed_query(query)