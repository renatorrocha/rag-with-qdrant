FROM tiangolo/uvicorn-gunicorn-fastapi:python3.11-slim AS builder

COPY ./requirements.txt /app/

RUN pip install --no-cache-dir -r /app/requirements.txt

COPY ./app /app