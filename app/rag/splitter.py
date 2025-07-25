from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter


text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)


def split_text(text: str) -> List[str]:
    return text_splitter.split_text(text)
