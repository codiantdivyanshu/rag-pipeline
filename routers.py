from fastapi import APIRouter, UploadFile, File, Form
from fastapi.responses import JSONResponse
import os
import tempfile
import torch

from rag_pipeline import extract_chunks, embed_chunks, save_embeddings, load_embeddings, query_pipeline

router = APIRouter()


session_store = {
    "chunks": None,
    "embeddings_tensor": None
}


@router.post("/upload/")
async def upload_pdf(file: UploadFile = File(...)):
    """
    Upload a PDF and generate embeddings.
    """
    try:
     
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(await file.read())
            temp_path = tmp_file.name

        chunks = extract_chunks(temp_path)
        embedded_chunks, embeddings_tensor = embed_chunks(chunks)

        save_path = f"embeddings_{os.path.basename(temp_path)}.csv"
        save_embeddings(embedded_chunks, save_path)


        session_store["chunks"] = chunks
        session_store["embeddings_tensor"] = embeddings_tensor

        return {"message": "PDF processed and embeddings saved.", "embedding_file": save_path}

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@router.post("/ask/")
async def ask_question(query: str = Form(...)):
    """
    Ask a question based on the uploaded PDF.
    """
    try:
        if session_store["chunks"] is None or session_store["embeddings_tensor"] is None:
            return JSONResponse(status_code=400, content={"error": "Please upload a PDF first."})

        answer, context = query_pipeline(query, session_store["chunks"], session_store["embeddings_tensor"])
        return {"question": query, "answer": answer, "context": context}

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@router.post("/load/")
async def load_existing_embeddings(path: str = Form(...)):
    """
    Load existing embeddings from a CSV file.
    """
    try:
        chunks, embeddings_tensor = load_embeddings(path)
        session_store["chunks"] = chunks
        session_store["embeddings_tensor"] = embeddings_tensor
        return {"message": f"Loaded embeddings from {path}"}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
