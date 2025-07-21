from fastapi import FastAPI, File, HTTPException, UploadFile
from langchain_ollama.embeddings import OllamaEmbeddings
from qdrant_client import QdrantClient
from langchain_text_splitters import RecursiveCharacterTextSplitter

app = FastAPI()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
embedder = OllamaEmbeddings(model="nomic-embed-text")

qdrant_client = QdrantClient(url="http://vector-db:6333")

supported_types = ["text/plain", "text/markdown", "application/octet-stream"]

max_file_size = 10 * 1024 * 1024  # 10MB

@app.get("/")
def hello_world():
    return {"message": "OK"}

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):

    if file.content_type not in supported_types:
        raise HTTPException(status_code=400, detail=f"File type not supported: {file.content_type}")

    content = await file.read()

    if not content:
        raise HTTPException(status_code=400, detail="File is empty")

    if len(content) > max_file_size:
        raise HTTPException(status_code=400, detail="File size exceeds the maximum allowed size (10MB)")

    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="File is not a valid text file")
    
    chunks = text_splitter.split_text(text)
    
    # Embedding chunks
    vectors = embedder.embed_documents(chunks)

    # Save chunks to vector database
    qdrant_client.add(
        collection_name="test",
        documents=chunks,
        embeddings=vectors,
        ids=list(range(len(chunks)))
    )

    return {"text": text, "chunks": len(chunks)}