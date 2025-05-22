from fastapi import FastAPI, UploadFile, File, Form
from pydantic import BaseModel
import shutil
import os
from rag_pipeline import extract_chunks, embed_chunks, save_embeddings, load_embeddings, query_pipeline

app = FastAPI()

class QueryRequest(BaseModel):
    query: str
    pdf_path: str  # path to already processed file

@app.post("/upload-pdf/")
async def upload_pdf(file: UploadFile = File(...)):
    file_path = f"./uploads/{file.filename}"
    os.makedirs("./uploads", exist_ok=True)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    embeddings_csv = os.path.splitext(file_path)[0] + "_embeddings.csv"
    pages_and_chunks = extract_chunks(file_path)
    pages_and_chunks = embed_chunks(pages_and_chunks)
    save_embeddings(pages_and_chunks, embeddings_csv)

    return {"message": "PDF processed", "pdf_path": file_path}

@app.post("/query/")
def run_query(request: QueryRequest):
    if not os.path.exists(request.pdf_path):
        return {"error": "PDF not found"}

    embeddings_csv = os.path.splitext(request.pdf_path)[0] + "_embeddings.csv"
    if not os.path.exists(embeddings_csv):
        return {"error": "Embeddings not found. Please upload and process the PDF first."}

    pages_and_chunks, embeddings_tensor = load_embeddings(embeddings_csv)
    answer, context = query_pipeline(request.query, pages_and_chunks, embeddings_tensor)
    return {
        "query": request.query,
        "answer": answer,
        "context": context
    }
