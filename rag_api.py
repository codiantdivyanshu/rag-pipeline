from fastapi import FastAPI, UploadFile, File, HTTPException, Query, Depends, Request
from pydantic import BaseModel
from typing import List
import os
import fitz  # PyMuPDF
import shutil
from rag_pipeline import extract_chunks, embed_chunks, save_embeddings, load_embeddings, query_pipeline
from starlette.status import HTTP_403_FORBIDDEN
from fastapi.openapi.utils import get_openapi

# === API KEY AUTH ===
API_KEY = "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"
API_KEY_NAME = "access_token"

app = FastAPI()

# === API Key Validator ===
async def get_api_key(request: Request):
    api_key = request.headers.get(API_KEY_NAME)
    if api_key != API_KEY:
        raise HTTPException(
            status_code=HTTP_403_FORBIDDEN,
            detail="Could not validate credentials"
        )
    return api_key

# === MODELS ===
class QueryRequest(BaseModel):
    query: str

# === UPLOAD MULTIPLE PDFs AND MERGE ===
@app.post("/upload/", dependencies=[Depends(get_api_key)])
async def upload(file: List[UploadFile] = File(...)):
    try:
        upload_folder = "./uploads"
        os.makedirs(upload_folder, exist_ok=True)
        merged_pdf_path = os.path.join(upload_folder, "merged.pdf")

        merged_doc = fitz.open()

        for fi in file:
            content = await fi.read()
            temp_path = os.path.join(upload_folder, fi.filename)
            with open(temp_path, "wb") as f:
                f.write(content)

            src_pdf = fitz.open(temp_path)
            merged_doc.insert_pdf(src_pdf)
            src_pdf.close()
            os.remove(temp_path)

        merged_doc.save(merged_pdf_path)
        merged_doc.close()

        chunks = extract_chunks(merged_pdf_path)
        embedded_chunks, _ = embed_chunks(chunks)
        save_embeddings(embedded_chunks, os.path.splitext(merged_pdf_path)[0] + "_embeddings.csv")

        return {"message": "PDFs merged and processed", "path": merged_pdf_path}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

# === QUERY ===
@app.post("/query/", dependencies=[Depends(get_api_key)])
def run_query(request: QueryRequest, source: str = Query("merged", description="Specify 'merged' or filename")):
    try:
        pdf_path = os.path.join("./uploads", "merged.pdf" if source == "merged" else os.path.basename(source))
        if not os.path.exists(pdf_path):
            raise HTTPException(status_code=404, detail=f"PDF '{pdf_path}' not found")

        embeddings_csv = os.path.splitext(pdf_path)[0] + "_embeddings.csv"
        if not os.path.exists(embeddings_csv):
            raise HTTPException(status_code=400, detail="Embeddings not found. Please upload the PDF first.")

        chunks, embeddings_tensor = load_embeddings(embeddings_csv)
        answer, context = query_pipeline(request.query, chunks, embeddings_tensor)

        return {
            "query": request.query,
            "source": source,
            "answer": answer,
            "context": context
        }

    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")

# === ROOT ===
@app.get("/", dependencies=[Depends(get_api_key)])
def root():
    return {"message": "RAG API is running"}

# === CUSTOM OPENAPI ===
def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    openapi_schema = get_openapi(
        title="RAG API with API Key Auth",
        version="1.0",
        description="Upload PDFs and ask questions using secure RAG pipeline",
        routes=app.routes,
    )
    openapi_schema["components"]["securitySchemes"] = {
        "APIKeyHeader": {
            "type": "apiKey",
            "in": "header",
            "name": API_KEY_NAME
        }
    }
    openapi_schema["security"] = [{"APIKeyHeader": []}]
    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi
