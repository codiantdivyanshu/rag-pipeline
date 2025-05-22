# rag_pipeline.py

import os, fitz, re, requests, random, torch, textwrap
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from sentence_transformers import SentenceTransformer
from groq import Groq  # pip install groq

# --- CONFIG ---
DEVICE = "cpu"
PDF_PATH = "human-nutrition-text.pdf"
EMBEDDINGS_CSV = "text_chunks_and_embeddings_df.csv"
GROQ_API_KEY = os.getenv("GROQ_API_KEY") or "your-groq-api-key"

# --- DOWNLOAD PDF IF NEEDED ---
if not os.path.exists(PDF_PATH):
    url = "https://pressbooks.oer.hawaii.edu/humannutrition2/open/download?type=pdf"
    response = requests.get(url)
    with open(PDF_PATH, "wb") as f:
        f.write(response.content)

# --- FORMATTERS ---
def text_formatter(text: str) -> str:
    return text.replace("\n", " ").strip()

# --- LOAD & CHUNK PDF ---
def extract_chunks(pdf_path):
    doc = fitz.open(pdf_path)
    pages_and_chunks = []
    for page_number, page in enumerate(doc):
        text = text_formatter(page.get_text())
        sentences = re.split(r'(?<=[.!?]) +', text)
        for sentence in sentences:
            clean = sentence.strip()
            if len(clean) > 30:  # token approx
                pages_and_chunks.append({
                    "page_number": page_number - 41,
                    "sentence_chunk": clean,
                    "chunk_token_count": len(clean) / 4
                })
    return pages_and_chunks

# --- EMBEDDING ---
def embed_chunks(pages_and_chunks):
    model = SentenceTransformer("all-mpnet-base-v2", device=DEVICE)
    chunks = [c["sentence_chunk"] for c in pages_and_chunks]
    embeddings = model.encode(chunks, batch_size=32, convert_to_tensor=True)
    for i, chunk in enumerate(pages_and_chunks):
        chunk["embedding"] = embeddings[i].cpu().numpy()
    return pages_and_chunks

# --- SAVE EMBEDDINGS ---
def save_embeddings(pages_and_chunks, path):
    df = pd.DataFrame(pages_and_chunks)
    df["embedding"] = df["embedding"].apply(lambda x: np.array2string(x, separator=' '))
    df.to_csv(path, index=False)

# --- LOAD EMBEDDINGS ---
def load_embeddings(path):
    df = pd.read_csv(path)
    df["embedding"] = df["embedding"].apply(lambda x: np.fromstring(x.strip("[]"), sep=" "))
    return df.to_dict(orient="records"), torch.tensor(np.array(df["embedding"].tolist()), dtype=torch.float32)

# --- PRINT UTILITY ---
def print_wrapped(text, width=80):
    print(textwrap.fill(text, width))

# --- RUN QUERY ---
def query_pipeline(query, pages_and_chunks, embeddings_tensor):
    model = SentenceTransformer("all-mpnet-base-v2", device=DEVICE)
    query_embedding = model.encode(query, convert_to_tensor=True)
    dot_scores = torch.matmul(embeddings_tensor, query_embedding)
    top_k = torch.topk(dot_scores, k=5)

    context = ""
    for score, idx in zip(top_k.values, top_k.indices):
        match = pages_and_chunks[idx]
        context += f"[Page {match['page_number']}] {match['sentence_chunk']}\n"

    # Call Groq (Mixtral) for answer generation
    client = Groq(api_key=# rag_pipeline.py

import os, fitz, re, requests, random, torch, textwrap
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from sentence_transformers import SentenceTransformer
from groq import Groq  # pip install groq

# --- CONFIG ---
DEVICE = "cpu"
PDF_PATH = "human-nutrition-text.pdf"
EMBEDDINGS_CSV = "text_chunks_and_embeddings_df.csv"
GROQ_API_KEY = os.getenv("GROQ_API_KEY") or "your-groq-api-key"

# --- DOWNLOAD PDF IF NEEDED ---
if not os.path.exists(PDF_PATH):
    url = "https://pressbooks.oer.hawaii.edu/humannutrition2/open/download?type=pdf"
    response = requests.get(url)
    with open(PDF_PATH, "wb") as f:
        f.write(response.content)

# --- FORMATTERS ---
def text_formatter(text: str) -> str:
    return text.replace("\n", " ").strip()

# --- LOAD & CHUNK PDF ---
def extract_chunks(pdf_path):
    doc = fitz.open(pdf_path)
    pages_and_chunks = []
    for page_number, page in enumerate(doc):
        text = text_formatter(page.get_text())
        sentences = re.split(r'(?<=[.!?]) +', text)
        for sentence in sentences:
            clean = sentence.strip()
            if len(clean) > 30:  # token approx
                pages_and_chunks.append({
                    "page_number": page_number - 41,
                    "sentence_chunk": clean,
                    "chunk_token_count": len(clean) / 4
                })
    return pages_and_chunks

# --- EMBEDDING ---
def embed_chunks(pages_and_chunks):
    model = SentenceTransformer("all-mpnet-base-v2", device=DEVICE)
    chunks = [c["sentence_chunk"] for c in pages_and_chunks]
    embeddings = model.encode(chunks, batch_size=32, convert_to_tensor=True)
    for i, chunk in enumerate(pages_and_chunks):
        chunk["embedding"] = embeddings[i].cpu().numpy()
    return pages_and_chunks

# --- SAVE EMBEDDINGS ---
def save_embeddings(pages_and_chunks, path):
    df = pd.DataFrame(pages_and_chunks)
    df["embedding"] = df["embedding"].apply(lambda x: np.array2string(x, separator=' '))
    df.to_csv(path, index=False)

# --- LOAD EMBEDDINGS ---
def load_embeddings(path):
    df = pd.read_csv(path)
    df["embedding"] = df["embedding"].apply(lambda x: np.fromstring(x.strip("[]"), sep=" "))
    return df.to_dict(orient="records"), torch.tensor(np.array(df["embedding"].tolist()), dtype=torch.float32)

# --- PRINT UTILITY ---
def print_wrapped(text, width=80):
    print(textwrap.fill(text, width))

# --- RUN QUERY ---
def query_pipeline(query, pages_and_chunks, embeddings_tensor):
    model = SentenceTransformer("all-mpnet-base-v2", device=DEVICE)
    query_embedding = model.encode(query, convert_to_tensor=True)
    dot_scores = torch.matmul(embeddings_tensor, query_embedding)
    top_k = torch.topk(dot_scores, k=5)

    context = ""
    for score, idx in zip(top_k.values, top_k.indices):
        match = pages_and_chunks[idx]
        context += f"[Page {match['page_number']}] {match['sentence_chunk']}\n"

    # Call Groq (Mixtral) for answer generation
    client = Groq(api_key=# rag_pipeline.py

import os, fitz, re, requests, random, torch, textwrap
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from sentence_transformers import SentenceTransformer
from groq import Groq  # pip install groq

# --- CONFIG ---
DEVICE = "cpu"
PDF_PATH = "human-nutrition-text.pdf"
EMBEDDINGS_CSV = "text_chunks_and_embeddings_df.csv"
GROQ_API_KEY = os.getenv("GROQ_API_KEY") or "your-groq-api-key"

# --- DOWNLOAD PDF IF NEEDED ---
if not os.path.exists(PDF_PATH):
    url = "https://pressbooks.oer.hawaii.edu/humannutrition2/open/download?type=pdf"
    response = requests.get(url)
    with open(PDF_PATH, "wb") as f:
        f.write(response.content)

# --- FORMATTERS ---
def text_formatter(text: str) -> str:
    return text.replace("\n", " ").strip()

# --- LOAD & CHUNK PDF ---
def extract_chunks(pdf_path):
    doc = fitz.open(pdf_path)
    pages_and_chunks = []
    for page_number, page in enumerate(doc):
        text = text_formatter(page.get_text())
        sentences = re.split(r'(?<=[.!?]) +', text)
        for sentence in sentences:
            clean = sentence.strip()
            if len(clean) > 30:  # token approx
                pages_and_chunks.append({
                    "page_number": page_number - 41,
                    "sentence_chunk": clean,
                    "chunk_token_count": len(clean) / 4
                })
    return pages_and_chunks

# --- EMBEDDING ---
def embed_chunks(pages_and_chunks):
    model = SentenceTransformer("all-mpnet-base-v2", device=DEVICE)
    chunks = [c["sentence_chunk"] for c in pages_and_chunks]
    embeddings = model.encode(chunks, batch_size=32, convert_to_tensor=True)
    for i, chunk in enumerate(pages_and_chunks):
        chunk["embedding"] = embeddings[i].cpu().numpy()
    return pages_and_chunks

# --- SAVE EMBEDDINGS ---
def save_embeddings(pages_and_chunks, path):
    df = pd.DataFrame(pages_and_chunks)
    df["embedding"] = df["embedding"].apply(lambda x: np.array2string(x, separator=' '))
    df.to_csv(path, index=False)

# --- LOAD EMBEDDINGS ---
def load_embeddings(path):
    df = pd.read_csv(path)
    df["embedding"] = df["embedding"].apply(lambda x: np.fromstring(x.strip("[]"), sep=" "))
    return df.to_dict(orient="records"), torch.tensor(np.array(df["embedding"].tolist()), dtype=torch.float32)

# --- PRINT UTILITY ---
def print_wrapped(text, width=80):
    print(textwrap.fill(text, width))

# --- RUN QUERY ---
def query_pipeline(query, pages_and_chunks, embeddings_tensor):
    model = SentenceTransformer("all-mpnet-base-v2", device=DEVICE)
    query_embedding = model.encode(query, convert_to_tensor=True)
    dot_scores = torch.matmul(embeddings_tensor, query_embedding)
    top_k = torch.topk(dot_scores, k=5)

    context = ""
    for score, idx in zip(top_k.values, top_k.indices):
        match = pages_and_chunks[idx]
        context += f"[Page {match['page_number']}] {match['sentence_chunk']}\n"

    # Call Groq (Mixtral) for answer generation
    client = Groq(api_key=)
    response = client.chat.completions.create(
        model="mixtral-8x7b-32768",
        messages=[
            {"role": "system", "content": "You're a helpful AI assistant and expert in nutrition."},
            {"role": "user", "content": f"Use the below context to answer the query.\n\nContext:\n{context}\n\nQuery: {query}"}
        ]
    )
    return response.choices[0].message.content, context

# === MAIN PIPELINE ===
if __name__ == "__main__":
    if not os.path.exists(EMBEDDINGS_CSV):
        print("Processing PDF and generating embeddings...")
        pages_and_chunks = extract_chunks(PDF_PATH)
        pages_and_chunks = embed_chunks(pages_and_chunks)
        save_embeddings(pages_and_chunks, EMBEDDINGS_CSV)
    else:
        print("Loading saved embeddings...")
        pages_and_chunks, embeddings_tensor = load_embeddings(EMBEDDINGS_CSV)

    query = input("Enter your query: ").strip()
    answer, context = query_pipeline(query, pages_and_chunks, embeddings_tensor)

    print("\nTop relevant context:\n")
    print_wrapped(context)
    print("\nAnswer:\n")
    print_wrapped(answer)
)
    response = client.chat.completions.create(
        model="mixtral-8x7b-32768",
        messages=[
            {"role": "system", "content": "You're a helpful AI assistant and expert in nutrition."},
            {"role": "user", "content": f"Use the below context to answer the query.\n\nContext:\n{context}\n\nQuery: {query}"}
        ]
    )
    return response.choices[0].message.content, context

# === MAIN PIPELINE ===
if __name__ == "__main__":
    if not os.path.exists(EMBEDDINGS_CSV):
        print("Processing PDF and generating embeddings...")
        pages_and_chunks = extract_chunks(PDF_PATH)
        pages_and_chunks = embed_chunks(pages_and_chunks)
        save_embeddings(pages_and_chunks, EMBEDDINGS_CSV)
    else:
        print("Loading saved embeddings...")
        pages_and_chunks, embeddings_tensor = load_embeddings(EMBEDDINGS_CSV)

    query = input("Enter your query: ").strip()
    answer, context = query_pipeline(query, pages_and_chunks, embeddings_tensor)

    print("\nTop relevant context:\n")
    print_wrapped(context)
    print("\nAnswer:\n")
    print_wrapped(answer)
)
    response = client.chat.completions.create(
        model="mixtral-8x7b-32768",
        messages=[
            {"role": "system", "content": "You're a helpful AI assistant and expert in nutrition."},
            {"role": "user", "content": f"Use the below context to answer the query.\n\nContext:\n{context}\n\nQuery: {query}"}
        ]
    )
    return response.choices[0].message.content, context

# === MAIN PIPELINE ===
if __name__ == "__main__":
    if not os.path.exists(EMBEDDINGS_CSV):
        print("Processing PDF and generating embeddings...")
        pages_and_chunks = extract_chunks(PDF_PATH)
        pages_and_chunks = embed_chunks(pages_and_chunks)
        save_embeddings(pages_and_chunks, EMBEDDINGS_CSV)
    else:
        print("Loading saved embeddings...")
        pages_and_chunks, embeddings_tensor = load_embeddings(EMBEDDINGS_CSV)

    query = input("Enter your query: ").strip()
    answer, context = query_pipeline(query, pages_and_chunks, embeddings_tensor)

    print("\nTop relevant context:\n")
    print_wrapped(context)
    print("\nAnswer:\n")
    print_wrapped(answer)
