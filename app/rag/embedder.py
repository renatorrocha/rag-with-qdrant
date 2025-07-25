# Gera e envia os embeddings para o vector db
from typing import List
from config import VOYAGE_AI_SECRET_KEY
import voyageai

# embedder = OllamaEmbeddings(model="nomic-embed-text", base_url=OLLAMA_BASE_URL)
vo = voyageai.Client(api_key=VOYAGE_AI_SECRET_KEY)


def generate_embeddings(chunks: List[str]) -> None:
    # return embedder.embed_documents(chunks)
    return vo.embed(chunks, model="voyage-3.5")


def generate_embeddings_query(query: str) -> None:
    return vo.embed.embed_query(query)
