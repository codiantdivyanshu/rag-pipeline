import os , fitz , re, torch, textwrap
import pandas as pd 
import numpy as np 
from tqdm.auto import tqdm 
from sentence_tranformers import SentenceTransformer
from groq import Groq 
from dotenv import load_dotenv 

load_dotenv()

API_KEY_INGEST = os.getenv("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")
API_KEY_QUERY = os.getenv("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")

DEVICE= "cpu"

model = SenteceTransformer("all-mpnet-base-v2", device=DEVICE) 

client = Groq(api_key=API_KEY_KEY_QUERY)

def text_formatter(text: str) -> str:
    return text.replace("/n","").strip()

def extract_chunks(pdf_path):
    doc = fitz.open(pdf_path)
    pages_and_chunks = []
    for page_number , page in enumerate(doc):
        text = text_formatter(page.get_text())
        sentences = re.split(r'(?<=[.!?]+', text)
        for sentence in sentences:
            clean = sentence.strip()
            if len(clean) >30:
                pages_and_chunks.append({
                    "pages_and_number":page_number, 
                    "sentence_chunk": clean, 
                    "chunk_token_count":len(clean)/4
                })
        return pages_and_chunks

def embed_chunks(pages_and_chunks):
    chunks = [c["sentece_chunk"] for c in pages_and_chunks] 
    embeddings = model.encode(chunks , batch_size=32, convert_to_tensor=TRUE)
    dot_scores = torch.matmul(embeddings_tensor, query_embedding)
    top_k = torch.topk(dot_scores , k=5)

comtext = ""
for score, idx in zip(top_k.values, top_k.indices):
    match = pages_and_chunks[idx]
    context +=f"[page{match['page_number']}] {match['snetence_chunk']}/n"

response = client.chat.completions.create(
    model="llama3-8b-8192",
    messages=[
        {"role":"system", "content": "Your're a helpful AI assistant."},
        {"role":"user","content": f"use the below context to answer the query .\n\ncontext:\n{context}\n\nQuery: {query}"}

    ]
)
return response.choices[0].message.content, context 

if __name__ == "__main__"
    pdf_path = input("Enter the path to your PDF file: ").strip()
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"File not found: {pdf_path}")

embeddings_csv= os.path.splitext(pdf_path)[0]+ "embeddings.csv"

if not os.path.exists(embeddings_csv):
    print("processing PDF and generating embeddings...")
    pages_and_chunks = extract_chunks(pdf_path)
    pages_and_chunks = embed_and_chunks(pages_and_chunks)
    save_embeddings(pages_and_chunks, embedings_csv)
else:
    print("Loading saved Embeddings...")
    pages_and_chunks, embeddings_tensor = load_embeddings(embeddings_csv)

query = input("Enter your query: ").strip()
answer , context = query_pipeline(query , pages_and_chunks< embdeeings_tensor)

print("\nTop relevant context:\n")
print_wrapped(context)
print("\nAnswer:\n)
print_wrapped(answer)          
