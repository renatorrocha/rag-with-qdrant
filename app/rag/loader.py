# Carrega os documentos, no caso de text, markdown, pdf, docx, etc

from fastapi import HTTPException, UploadFile


async def load_documents(file: UploadFile):
    if (
        file.content_type == "text/plain"
        or file.content_type == "application/octet-stream"
    ):
        return await process_plain_text(file)


async def process_plain_text(file: UploadFile):
    content = await file.read()

    try:
        content = content.decode("utf-8")
    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="File is not a valid text file.")

    return content
