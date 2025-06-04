from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from pydantic import BaseModel
from typing import List
import shutil
import os
import fitz
from rag_pipeline import extract_chunks, embed_chunks, save_embeddings, load_embeddings, query_pipeline
from .routes import router

app = FastAPI(title="RAG PDF QA API")
app.include_router(router)

class QueryRequest(BaseModel):
    query: str

def merge_pdfs(pdf_paths, output_path):
    merged_doc = fitz.open()
    for path in pdf_paths:
        doc = fitz.open(path)
        merged_doc.insert_pdf(doc)
        doc.close()
    merged_doc.save(output_path)
    merged_doc.close()

@app.post("/upload/")
async def upload(files: List[UploadFile] = File(...)):
    try:
        os.makedirs("./uploads", exist_ok=True)
        uploaded_paths = []

        for file in files:
            file_path = f"./uploads/{file.filename}"
            with open(file_path, "wb") as f:
                shutil.copyfileobj(file.file, f)
            uploaded_paths.append(file_path)

        if not uploaded_paths:
            raise HTTPException(status_code=400, detail="No files uploaded.")

        merged_pdf_path = "./uploads/merged.pdf"
        merge_pdfs(uploaded_paths, merged_pdf_path)

        individual_results = []
        for pdf_path in uploaded_paths:
            try:
                chunks = extract_chunks(pdf_path)
                embedded_chunks, _ = embed_chunks(chunks)
                save_embeddings(embedded_chunks, os.path.splitext(pdf_path)[0] + "_embeddings.csv")
                individual_results.append({"file": os.path.basename(pdf_path), "result": "success"})
            except Exception as e:
                individual_results.append({"file": os.path.basename(pdf_path), "result": f"error: {str(e)}"})

        try:
            chunks = extract_chunks(merged_pdf_path)
            embedded_chunks, _ = embed_chunks(chunks)
            save_embeddings(embedded_chunks, os.path.splitext(merged_pdf_path)[0] + "_embeddings.csv")
            merged_result = {"file": os.path.basename(merged_pdf_path), "result": "success"}
        except Exception as e:
            merged_result = {"file": os.path.basename(merged_pdf_path), "result": f"error: {str(e)}"}

        return {
            "message": "Upload and processing complete",
            "individual_files": individual_results,
            "merged_file": merged_result,
            "merged_pdf_path": merged_pdf_path
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@app.post("/query/")
def run_query(request: QueryRequest, source: str = Query("merged", description="Specify 'merged' or filename ")):
    try:
        if source == "merged":
            pdf_path = "./uploads/merged.pdf"
        else:
            pdf_path = os.path.join("./uploads", os.path.basename(source))

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

@app.get("/")
def root():
    return {"message": "RAG API is running"}
