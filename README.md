# rag-with-qdrant

A simple Proof of Concept (PoC) for a Retrieval-Augmented Generation (RAG) service using:

- **FastAPI** → backend API
- **Qdrant** → vector database

This service allows:
- Uploading documents (PDF, DOCX, TXT)
- Extracting and splitting text into chunks
- Generating embeddings for each chunk
- Storing embeddings in Qdrant
- Answering questions by searching relevant chunks via vector similarity

---

## 🚀 Running the Project

`docker-compose up`

### Qdrant

- **REST API** → [http://localhost:6333](http://localhost:6333)
- **Web UI** → [http://localhost:6333/dashboard](http://localhost:6333/dashboard)

### FastAPI

- **API** → [http://localhost:8000](http://localhost:8000)

Run locally in development mode:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

or via Docker-compose:
```bash
docker-compose up --build
```

## 🗂️ API Endpoints (Planned)

### File Upload

- `POST /upload`
  - Receive uploaded files (PDF, DOCX, TXT)
  - Extract and split text into chunks
  - Generate embeddings
  - Store vectors in Qdrant

### Ask a Question

- `POST /ask`
  - Receive a question
  - Generate question embedding
  - Search Qdrant for similar vectors
  - Return relevant context to answer the question

---

## ✅ TO-DO

- [ ] Implement `POST /upload` endpoint
- [ ] Add support for extracting text from PDF, DOCX, and TXT
- [ ] Implement text splitting into smaller chunks
- [ ] Integrate embedding generation
- [ ] Store embeddings in Qdrant
- [ ] Implement question-answering logic

