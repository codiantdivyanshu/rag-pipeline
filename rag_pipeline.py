import os
import fitz
import re
import torch
import textwrap
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

API_KEY_QUERY = os.getenv("GROQ_API_KEY")
if not API_KEY_QUERY:
     raise ValueError("Missing GROQ_API_KEY in environment. Check .env file.")
DEVICE = "cpu"

model = SentenceTransformer("all-mpnet-base-v2", device=DEVICE)
client = Groq(api_key=API_KEY_QUERY)

def text_formatter(text: str) -> str:
    return text.replace("\n", " ").strip()

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

def embed_chunks(pages_and_chunks):
    chunks = [c["sentence_chunk"] for c in pages_and_chunks]
    embeddings = model.encode(chunks, batch_size=32, convert_to_tensor=True)

    embedded_chunks = []
    for i, emb in enumerate(embeddings):
        embedded_chunks.append({
            "sentence_chunk": pages_and_chunks[i]["sentence_chunk"],
            "meta": {"page": pages_and_chunks[i]["page_number"]},
            "embedding": emb.detach().cpu().numpy().tolist()
        })

    return embedded_chunks, embeddings

def save_embeddings(embedded_chunks, path):
    for i, chunk in enumerate(embedded_chunks):
        if "embedding" not in chunk:
            raise ValueError(f"Chunk at index {i} missing 'embedding' field: {chunk}")

    df = pd.DataFrame(embedded_chunks)
    df["embedding"] = df["embedding"].apply(lambda x: np.array2string(np.array(x), separator=' '))
    df["meta"] = df["meta"].apply(str)  # Convert dict to string for CSV
    df.to_csv(path, index=False)

def load_embeddings(path):
    df = pd.read_csv(path)
    df["embedding"] = df["embedding"].apply(lambda x: np.fromstring(x.strip("[]"), sep=" "))
    df["meta"] = df["meta"].apply(eval)  # Convert string back to dict
    chunks = df[["sentence_chunk", "meta"]].to_dict(orient="records")
    embeddings = torch.tensor(np.stack(df["embedding"].to_numpy()), dtype=torch.float32)
    return chunks, embeddings

def query_pipeline(query, chunks, embeddings_tensor):
    query_embedding = model.encode(query, convert_to_tensor=True)
    scores = util.cos_sim(query_embedding, embeddings_tensor)[0]
    top_k = torch.topk(scores, k=5)
    top_chunks = [chunks[i] for i in top_k.indices.tolist()]

    context = "\n".join(f"[Page {c['meta']['page']}] {c['sentence_chunk']}" for c in top_chunks)

    response = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
        ]
    )

    return response.choices[0].message.content, context

def print_wrapped(text, width=100):
    print("\n".join(textwrap.wrap(text, width)))
