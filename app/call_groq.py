import requests
from config import GROQ_API_KEY


def call_groq(context: str, question: str) -> str:
    prompt = f"""Responda à pergunta com base nos documentos abaixo.

DOCUMENTOS: 
{context}

PERGUNTA:
{question}

RESPOSTA:"""

    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": "gemma2-9b-it",
        "messages": [
            {
                "role": "system",
                "content": "Você é um assistente útil que responde com base nos documentos fornecidos.",
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.3,
    }

    response = requests.post(url, headers=headers, json=payload)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]
