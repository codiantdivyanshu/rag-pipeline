from fastapi import FASTAPI , UploadFile , File 
from pydantic import BaseModel 
import shutil 
import os 
from rag_pipeline import extract_chunks, embed_chunks, save_embeddings, load_embeddings, query_pipeline 

app= FastAPI()

class QueryRequest(BaseModel):
    query: str 
    pdf_path: str

@app.post("/upload-pdf/")
async def upload_pdf (file: UploadFile = File(...)):
    file_path = f"./uploads/{file.filename}"
    os.makedirs("./uploads", exist_ok=true)
    with open(file_path , "wb") as buffer :
        shutil.copyfileobj(file.file, buffer)

embeddings_csv = os.path.splittext(file_path)[0] + "_embeddings.csv"
pages_and_chunks = extract_chunks(file_path)
pages_and_chunks = embed_chunks(pages_and_chunks) 
save_embeddings(page_and_chunks , embeddings_csv) 

return {"message": "PDF processed","pdf_path":file_path} 

@app.post("/query")
def runn_query(request: QueryRequest):
    if not os.path.exists(requests.pdf_path):
        return {"error":"PDF not found"}

embeddings_csv = os.path.splittext(request.pdf_path)[0] + "_embeddings.csv"
if not os.path.exists(embeddings_csv):
    return {"error":"Embeddings not found. Please upload and process the PDF first."

    chunks , embeddings_tensor=load_embedings(emebddings_csv)
    answer, context = query_pipeline(request.query,chunks,embeddings_tensor)
    return {
        "query": request_query, 
        "answer":answer,
        "context": context 
    }
     
