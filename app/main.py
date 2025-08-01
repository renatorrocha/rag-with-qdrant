from fastapi import FastAPI
from app.routes import rag

app = FastAPI()


@app.get("/")
def health_check():
    return {"message": "OK"}


app.include_router(rag.router)
