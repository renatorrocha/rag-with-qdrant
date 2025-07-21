import os
import uuid
from fastapi import FastAPI, File, HTTPException, UploadFile
from langchain_ollama.embeddings import OllamaEmbeddings
from qdrant_client import QdrantClient, models
from langchain_text_splitters import RecursiveCharacterTextSplitter

app = FastAPI()

COLLECTION_NAME = os.getenv("COLLECTION_NAME", "documents")

LLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
embedder = OllamaEmbeddings(model="nomic-embed-text", base_url=LLAMA_BASE_URL)

qdrant_client = QdrantClient(url="http://vector-db:6333")


supported_types = ["text/plain", "text/markdown", "application/octet-stream"]

max_file_size = 10 * 1024 * 1024  # 10MB


@app.get("/")
def hello_world():
    return {"message": "OK"}


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    if file.content_type not in supported_types:
        raise HTTPException(
            status_code=400, detail=f"File type not supported: {file.content_type}"
        )

    content = await file.read()

    if not content:
        raise HTTPException(status_code=400, detail="File is empty")

    if len(content) > max_file_size:
        raise HTTPException(
            status_code=400, detail="File size exceeds the maximum allowed size (10MB)"
        )

    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="File is not a valid text file")

    chunks = text_splitter.split_text(text)

    # Embedding chunks
    vectors = embedder.embed_documents(chunks)

    try:
        qdrant_client.get_collection(COLLECTION_NAME)
    except Exception:
        qdrant_client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=models.VectorParams(
                size=len(vectors[0]),
                distance=models.Distance.COSINE,
            ),
        )

    points = [
        models.PointStruct(id=str(uuid.uuid4()), vector=vector, payload={"text": chunk})
        for vector, chunk in zip(vectors, chunks)
    ]

    # Save chunks to vector database
    qdrant_client.upsert(collection_name=COLLECTION_NAME, points=points)

    return {"text": text, "chunks": len(chunks)}
