import uuid
from fastapi import FastAPI, File, HTTPException, UploadFile
from qdrant_client import models
from call_groq import call_groq
from config import supported_types, max_file_size, text_splitter, embedder, qdrant_client, COLLECTION_NAME

app = FastAPI()

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

    qdrant_client.upsert(collection_name=COLLECTION_NAME, points=points)

    return {"data": "Uploaded successfully", "chunks": len(chunks)}

@app.post("/ask")
async def ask(question: str):
    question_embedding = embedder.embed_query(question)

    search_result = qdrant_client.search(
        collection_name=COLLECTION_NAME,
        query_vector=question_embedding,
        limit=5,
        with_payload=True,
    )

    context_chunks = [hit.payload["text"] for hit in search_result]
    context = "\n\n".join(context_chunks)

    answer = call_groq(context, question)

    return {"answer": answer}
