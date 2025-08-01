from typing import List

from langchain_ollama import OllamaEmbeddings

embedder = OllamaEmbeddings(model="nomic-embed-text")


def generate_embeddings(chunks: List[str]) -> List[List[float]]:
    # return embedder.embed_documents(chunks)
    return embedder.embed_documents(chunks)


def generate_embedding_query(query: str) -> List[float]:
    return embedder.embed_query(query)
