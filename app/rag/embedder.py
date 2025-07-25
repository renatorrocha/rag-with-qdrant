from typing import List
from config import VOYAGE_AI_SECRET_KEY
import voyageai

vo = voyageai.Client(api_key=VOYAGE_AI_SECRET_KEY)
# embedder = OllamaEmbeddings(model="nomic-embed-text",


def generate_embeddings(chunks: List[str]) -> List[List[float]]:
    # return embedder.embed_documents(chunks)
    response = vo.embed(
        texts=chunks,
        model="voyage-3.5",
        input_type="document",  # importante para distinguir entre doc e query
    )
    return response.embeddings


def generate_embedding_query(query: str) -> List[float]:
    response = vo.embed(texts=[query], model="voyage-3.5", input_type="query")
    return response.embeddings[0]
