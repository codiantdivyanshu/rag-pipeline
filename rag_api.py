from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from pydantic import BaseModel
import shutil
import os
import fitz
from rag_pipeline import extract_chunks, embed_chunks, save_embeddings, load_embeddings, query_pipeline

app = FastAPI()

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
async def upload(file1: UploadFile = File(...), file2: UploadFile = File(...)):
    try:
        os.makedirs("./uploads", exist_ok=True)

        file1_path = f"./uploads/{file1.filename}"
        file2_path = f"./uploads/{file2.filename}"

        with open(file1_path, "wb") as f1, open(file2_path, "wb") as f2:
            shutil.copyfileobj(file1.file, f1)
            shutil.copyfileobj(file2.file, f2)

        merged_pdf_path = "./uploads/merged.pdf"
        merge_pdfs([file1_path, file2_path], merged_pdf_path)

        individual_results = []
        for pdf_path in [file1_path, file2_path]:
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
