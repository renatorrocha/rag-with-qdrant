from pathlib import Path
from fastapi import FastAPI, File, HTTPException, UploadFile
import tempfile
from app.clean_nans import clean_nans
from langchain_ollama.embeddings import OllamaEmbeddings
from qdrant_client import QdrantClient
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyMuPDFLoader,
    TextLoader,
    Docx2txtLoader,
)

app = FastAPI()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
embedder = OllamaEmbeddings(model="nomic-embed-text")

qdrant_client = QdrantClient(url="http://vector-db:6333")

supported_types = {
        "application/pdf": PyMuPDFLoader,
        "text/plain": TextLoader,
        "text/markdown": TextLoader,
        "text/md": TextLoader,
        "application/octet-stream": TextLoader,
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document": Docx2txtLoader,
    }

max_file_size = 10 * 1024 * 1024  # 10MB

@app.get("/")
def hello_world():
    return {"message": "OK"}

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):

    if file.content_type not in supported_types:
        raise HTTPException(status_code=400, detail=f"File type not supported: {file.content_type}")

    if not file.size:
        raise HTTPException(status_code=400, detail="Tamanho do arquivo não identificado")
    elif file.size > max_file_size:
        raise HTTPException(status_code=400, detail="Arquivo excede o tamanho máximo permitido (10MB)")

    try:
        suffix = Path(file.filename or "file").suffix or ".tmp"

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        LoaderClass = supported_types[file.content_type]
        loader = LoaderClass(tmp_path)
        docs = loader.load()

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        Path(tmp_path).unlink(missing_ok=True)  # remove o arquivo temporário


    full_text = "\n".join([doc.page_content for doc in docs])
    clean_text = clean_nans(full_text)

    chunks = text_splitter.split_text(clean_text)
    
    # Embedding chunks
    vectors = embedder.embed_documents(chunks)

    teste = qdrant_client.add(
        collection_name="test",
        documents=chunks,
        embeddings=vectors,
        ids=list(range(len(chunks)))
    )
    print(teste)

    # TODO: save chunks to vector database

    return {"text": clean_text, "chunks": len(chunks)}