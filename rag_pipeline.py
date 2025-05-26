import os, fitz, re, random, torch, textwrap
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from sentence_transformers import SentenceTransformer
from groq import Groq  
from dotenv import load_dotenv

load_dotenv()

DEVICE = "cpu"
GROQ_API_KEY = os.getenv("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")

model = SentenceTransformer("all-mpnet-base-v2", device=DEVICE)

client = Groq(api_key=GROQ_API_KEY)

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
    for i, chunk in enumerate(pages_and_chunks):
        chunk["embedding"] = embeddings[i].cpu().numpy()
    return pages_and_chunks

def save_embeddings(pages_and_chunks, path):
    df = pd.DataFrame(pages_and_chunks)
    df["embedding"] = df["embedding"].apply(lambda x: np.array2string(x, separator=' '))
    df.to_csv(path, index=False)

def load_embeddings(path):
    df = pd.read_csv(path)
    df["embedding"] = df["embedding"].apply(lambda x: np.fromstring(x.strip("[]"), sep=" "))
    return df.to_dict(orient="records"), torch.tensor(np.array(df["embedding"].tolist()), dtype=torch.float32)

def print_wrapped(text, width=80):
    print(textwrap.fill(text, width))

def query_pipeline(query, pages_and_chunks, embeddings_tensor):
    query_embedding = model.encode(query, convert_to_tensor=True)
    dot_scores = torch.matmul(embeddings_tensor, query_embedding)
    top_k = torch.topk(dot_scores, k=5)

    context = ""
    for score, idx in zip(top_k.values, top_k.indices):
        match = pages_and_chunks[idx]
        context += f"[Page {match['page_number']}] {match['sentence_chunk']}\n"

    response = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[
            {"role": "system", "content": "You're a helpful AI assistant."},
            {"role": "user", "content": f"Use the below context to answer the query.\n\nContext:\n{context}\n\nQuery: {query}"}
        ]
    )
    return response.choices[0].message.content, context

if __name__ == "__main__":
    pdf_path = input("Enter the path to your PDF file: ").strip()
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"File not found: {pdf_path}")

    embeddings_csv = os.path.splitext(pdf_path)[0] + "_embeddings.csv"

    if not os.path.exists(embeddings_csv):
        print("Processing PDF and generating embeddings...")
        pages_and_chunks = extract_chunks(pdf_path)
        pages_and_chunks = embed_chunks(pages_and_chunks)
        save_embeddings(pages_and_chunks, embeddings_csv)
    else:
        print("Loading saved embeddings...")
        pages_and_chunks, embeddings_tensor = load_embeddings(embeddings_csv)

    query = input("Enter your query: ").strip()
    answer, context = query_pipeline(query, pages_and_chunks, embeddings_tensor)

    print("\nTop relevant context:\n")
    print_wrapped(context)
    print("\nAnswer:\n")
    print_wrapped(answer)
