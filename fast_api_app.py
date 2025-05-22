from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import tempfile
import shutil
import torch
from rag_pipeline import load_pdf, generate_embeddings, query_pipeline  # Ensure these exist

app = FastAPI()

# Global cache for embeddings to avoid reprocessing on every query
pdf_cache = {
    "chunks": None,
    "tensor": None
}

class QueryInput(BaseModel):
    query: str

@app.post("/upload_pdf/")
async def upload_pdf(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    print(f"Saved uploaded file to {tmp_path}")
    
    # Load and embed
    pages_and_chunks = load_pdf(tmp_path)
    tensor = generate_embeddings(pages_and_chunks)

    # Save to cache
    pdf_cache["chunks"] = pages_and_chunks
    pdf_cache["tensor"] = tensor

    return {"message": "PDF processed successfully", "chunks": len(pages_and_chunks)}

@app.post("/query/")
async def ask_question(data: QueryInput):
    if pdf_cache["chunks"] is None or pdf_cache["tensor"] is None:
        return {"error": "No PDF uploaded. Please upload a PDF first."}

    answer, context = query_pipeline(data.query, pdf_cache["chunks"], pdf_cache["tensor"])
    return {
        "answer": answer,
        "context": context
    }
