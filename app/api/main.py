from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

from app.services.rag_service import RAGService

app = FastAPI(
    title="PDF RAG Chatbot",
    version="1.0.0",
)

# Directories
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

# Initialize the RAG service once
rag_service = RAGService()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    # Validate file type
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported.",
        )

    # Save uploaded PDF
    file_path = DATA_DIR / file.filename

    with open(file_path, "wb") as f:
        f.write(await file.read())

    # Delegate indexing to the RAG service
    stats = rag_service.index_pdf(file_path)

    return {
        "message": "Document indexed successfully",
        "filename": file.filename,
        **stats,
    }


class ChatRequest(BaseModel):
    question: str


@app.post("/chat")
def chat(request: ChatRequest):
    # Delegate retrieval and generation to the RAG service
    result = rag_service.ask(request.question)
    return result