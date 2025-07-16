from fastapi import FastAPI, File, HTTPException, UploadFile
from docling.document_converter import DocumentConverter
import tempfile
from app.clean_nans import clean_nans
from langchain_ollama.embeddings import OllamaEmbeddings
from qdrant_client import QdrantClient

app = FastAPI()

text_splitter = OllamaEmbeddings(model="nomic-embed-text")

qdrant_client = QdrantClient(url="http://vector-db:6333")

@app.get("/")
def hello_world():
    return {"message": "OK"}

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    supported_types = [
        "application/pdf",
        "text/plain",
        "text/md",
        "text/docx",
        "application/octet-stream" # for .md files
    ]

    max_file_size = 10 * 1024 * 1024  # 10MB

    if file.content_type not in supported_types:
        raise HTTPException(status_code=400, detail="File type not supported")

    if not file.size:
        raise HTTPException(status_code=400, detail="Could not determine file size")
    elif file.size > max_file_size:
        raise HTTPException(status_code=400, detail="File size exceeds the maximum allowed")
    
    try:
        file_content = await file.read()
        converter = DocumentConverter()

        # TXT
        if file.content_type == "text/plain":
            text = file_content.decode("utf-8")
            result = converter.convert(text)
        
        # PDF or DOCX
        else:
            with tempfile.NamedTemporaryFile(delete=True, suffix=file.filename) as tmp:
                tmp.write(file_content)
                tmp.flush()
                result = converter.convert(tmp.name)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 

    text = result.document.export_to_text()
    clean_text = clean_nans(text)

    # TODO: split text into chunks
    chunks = text_splitter.embed_documents(clean_text)
    print(chunks)

    # TODO: save chunks to vector database

    return {"text": clean_text}

@app.get("/test-qdrant")
def test_qdrant_connection():
    try:
        # Tenta fazer uma operação simples
        collections = qdrant_client.get_collections()
        return {
            "status": "success", 
            "message": "Conexão com Qdrant estabelecida",
            "collections": [col.name for col in collections.collections]
        }
    except Exception as e:
        return {
            "status": "error", 
            "message": f"Erro na conexão: {str(e)}"
        }
