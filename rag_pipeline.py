import os
import re
import fitz  # PyMuPDF
import torch
import textwrap
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from sentence_transformers import SentenceTransformer, util
from groq import Groq
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

API_KEY_QUERY = os.getenv("API_KEY_QUERY")

# Config
DEVICE = "cpu"
model = SentenceTransformer("all-mpnet-base-v2", device=DEVICE)
client = Groq(api_key=API_KEY_QUERY)

# Utility
def text_formatter(text: str) -> str:
    return text.replace("\n", " ").strip()

def print_wrapped(text, width=100):
    print(textwrap.fill(text, width=width))

# Step 1: Extract Chunks
def extract_chunks(pdf_path):
    doc = fitz.open(pdf_path)
    pages_and_chunks = []
    for page_number, page in enumerate(doc):
        text = text_formatter(page.get_text())
        sentences = re.split(r'(?<=[.!?]) +', text)
        for sentence in sentences:
            clean = sentence.strip()
            if len(clean) > 30:
                pages_and_chunks.append({
                    "page_number": page_number,
                    "sentence_chunk": clean,
                    "chunk_token_count": len(clean) / 4
                })
    return pages_and_chunks

# Step 2: Embed Chunks
def embed_chunks(pages_and_chunks):
    chunks = [c["sentence_chunk"] for c in pages_and_chunks]
    embeddings = model.encode(chunks, batch_size=32, convert_to_tensor=True)
    for i, embedding in enumerate(embeddings):
        pages_and_chunks[i]["embedding"] = embedding.cpu().numpy()
    return pages_and_chunks

# Step 3: Save/Load Embeddings
def save_embeddings(pages_and_chunks, path):
    df = pd.DataFrame(pages_and_chunks)
    df["embedding"] = df["embedding"].apply(lambda x: np.array2string(x, separator=' '))
    df.to_csv(path, index=False)

def load_embeddings(path):
    df = pd.read_csv(path)
    df["embedding"] = df["embedding"].apply(lambda x: np.fromstring(x.strip("[]"), sep=" "))
    return df.to_dict(orient="records"), torch.tensor(np.array(df["embedding"].tolist()), dtype=torch.float32)

# Step 4: Query Pipeline
def query_pipeline(query, chunks, embeddings_tensor):
    query_embedding = model.encode(query, convert_to_tensor=True)
    scores = util.cos_sim(query_embedding, embeddings_tensor)[0]
    top_k = torch.topk(scores, k=5)
    top_chunks = [chunks[idx] for idx in top_k.indices.tolist()]

    context = ""
    for chunk in top_chunks:
        context += f"[Page {chunk['page_number']}] {chunk['sentence_chunk']}\n"

    response = client.chat.completions.create(
        model="llama3-8b-8192",  # or gpt-3.5-turbo if you prefer
        messages=[
            {"role": "system", "content": "You are a helpful assistant who answers based on the provided document."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
        ]
    )

    return response.choices[0].message.content.strip(), context

# Optional: CLI Usage for local testing
if __name__ == "__main__":
    pdf_path = input("Enter the path to your PDF file: ").strip()
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"File not found: {pdf_path}")

    embeddings_csv = os.path.splitext(pdf_path)[0] + "_embeddings.csv"

    if not os.path.exists(embeddings_csv):
        print("Processing PDF and generating embeddings...")
        chunks = extract_chunks(pdf_path)
        chunks = embed_chunks(chunks)
        save_embeddings(chunks, embeddings_csv)
    else:
        print("Loading saved embeddings...")
        chunks, embeddings_tensor = load_embeddings(embeddings_csv)

    query = input("Enter your query: ").strip()
    answer, context = query_pipeline(query, chunks, embeddings_tensor)

    print("\nTop relevant context:\n")
    print_wrapped(context)
    print("\nAnswer:\n")
    print_wrapped(answer)
