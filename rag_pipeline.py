import fitz
from tqdm.auto import tqdm
import os
import requests
import random
import pandas as pd
import re
from sentence_transformers import SentenceTransformer
import torch
import numpy as np
import textwrap
import torch 
import groq import Groq


device = "cpu"
print(f"Using device: {device}")

pdf_path = "human-nutrition-text.pdf"


if not os.path.exists(pdf_path):
    print("File doesn't exist, downloading...")

    url = "https://pressbooks.oer.hawaii.edu/humannutrition2/open/download?type=pdf"
    response = requests.get(url)

    if response.status_code == 200:
        with open(pdf_path, "wb") as file:
            file.write(response.content)
        print(f"The file has been downloaded and saved as {pdf_path}")
    else:
        print(f"Failed to download the file. Status code: {response.status_code}")
else:
    print(f"File {pdf_path} exists.")


def text_formatter(text: str) -> str:
    """Performs minor formatting on text."""
    cleaned_text = text.replace("\n", " ").strip()
    return cleaned_text


def open_and_read_pdf(pdf_path: str) -> list[dict]:
    doc = fitz.open(pdf_path)
    pages_and_texts = []
    for page_number, page in tqdm(enumerate(doc)):
        text = page.get_text()
        text = text_formatter(text)

        
        sentences = re.split(r'(?<=[.!?]) +', text)

        pages_and_texts.append({
            "page_number": page_number - 41,  
            "page_char_count": len(text),
            "page_word_count": len(text.split()),
            "page_sentence_count_raw": len(sentences),
            "page_token_count": len(text) / 4,
            "text": text,
            "sentence_chunks": sentences  
        })
    return pages_and_texts



pages_and_texts = open_and_read_pdf(pdf_path=pdf_path)
print(f"Loaded {len(pages_and_texts)} pages")


sampled = random.sample(pages_and_texts, k=3)
for page in sampled:
    print(f"\n--- Page {page['page_number']} ---\n")
    print(page["text"][:500], "...\n")

df = pd.DataFrame(pages_and_texts)
print(df.head())
print(df.describe().round(2))


pages_and_chunks = []
for item in tqdm(pages_and_texts):
    for sentence_chunk in item["sentence_chunks"]:
        chunk_dict = {}
        chunk_dict["page_number"] = item["page_number"]

        
        joined_sentence_chunk = sentence_chunk.replace("  ", " ").strip()
        joined_sentence_chunk = re.sub(r'\.([A-Z])', r'. \1', joined_sentence_chunk)

        chunk_dict["sentence_chunk"] = joined_sentence_chunk
        chunk_dict["chunk_char_count"] = len(joined_sentence_chunk)
        chunk_dict["chunk_word_count"] = len(joined_sentence_chunk.split())
        chunk_dict["chunk_token_count"] = len(joined_sentence_chunk) / 4

        pages_and_chunks.append(chunk_dict)

print(f"Total sentence chunks: {len(pages_and_chunks)}")


print(random.choice(pages_and_chunks))


df_chunks = pd.DataFrame(pages_and_chunks)
print(df_chunks.describe().round(2))


min_token_length = 30
pages_and_chunks_over_min_token_len = [
    chunk for chunk in pages_and_chunks if chunk["chunk_token_count"] > min_token_length
]


embedding_model = SentenceTransformer(model_name_or_path="all-mpnet-base-v2", device=device)


text_chunks = [item["sentence_chunk"] for item in pages_and_chunks_over_min_token_len]

text_chunk_embeddings = embedding_model.encode(
    text_chunks,
    batch_size=32,
    convert_to_tensor=True,
    device=device  
)


for i, item in enumerate(pages_and_chunks_over_min_token_len):
    
    item["embedding"] = text_chunk_embeddings[i].cpu().numpy()


embeddings_df_save_path = "text_chunks_and_embeddings_df.csv"
text_chunks_and_embeddings_df = pd.DataFrame(pages_and_chunks_over_min_token_len)


text_chunks_and_embeddings_df["embedding"] = text_chunks_and_embeddings_df["embedding"].apply(lambda x: np.array2string(x, separator=' '))

text_chunks_and_embeddings_df.to_csv(embeddings_df_save_path, index=False)
print(f"Saved embeddings DataFrame to {embeddings_df_save_path}")


text_chunks_and_embedding_df = pd.read_csv(embeddings_df_save_path)


text_chunks_and_embedding_df["embedding"] = text_chunks_and_embedding_df["embedding"].apply(
    lambda x: np.fromstring(x.strip("[]"), sep=" ")
)


pages_and_chunks_loaded = text_chunks_and_embedding_df.to_dict(orient="records")


embeddings_tensor = torch.tensor(
    np.array(text_chunks_and_embedding_df["embedding"].tolist()), dtype=torch.float32
).to(device)

print(f"Embeddings tensor shape: {embeddings_tensor.shape}")
print(f"Device embeddings are on: {embeddings_tensor.device}")


for i in range(3):
    print("Sentence chunk:", pages_and_chunks_loaded[i]["sentence_chunk"][:80], "...")
    print("Embedding vector (first 5 values):", embeddings_tensor[i][:5].tolist())
    print("")




def print_wrapped(text, wrap_length=80):
    wrapped_text = textwrap.fill(text, wrap_length)
    print(wrapped_text)




query = "What are the benefits of good nutrition?"


query_embedding = embedding_model.encode(query, convert_to_tensor=True, device=device)

dot_products = torch.matmul(embeddings_tensor, query_embedding)


top_k = 5
top_results_dot_product = torch.topk(dot_products, k=top_k)

print(f"\nQuery: '{query}'\n")
print("Results:")

retrived_chunks = []

for score, idx in zip(top_results_dot_product.values, top_results_dot_product.indices):
    print(f"Score: {score:.4f}")
    print("Text:")
    chunk_text= pages_and_chunks_loaded[idx]["sentence_chunk"]
    print_wrapped(chunk_text)
 
    print(f"Page number: {pages_and_chunks_loaded[idx]['page_number']}")
    print("\n")
    retrived_chunks.append(chunk_text)

from groq import Groq
import os

# You must set this env var or replace it with the key directly (not recommended for production)
groq_api_key = os.getenv("GROQ_API_KEY", "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")
client = Groq(api_key=groq_api_key)

# Construct the prompt for Groq's LLM
prompt = f"""You are an expert on human nutrition. Based on the following information extracted from a textbook, answer the question comprehensively.

Context:
{chr(10).join(retrieved_chunks)}

Question: {query}
"""

# Generate a response using Groq's LLM
chat_completion = client.chat.completions.create(
    model="llama3-70b-8192",  # Update if needed per Groq's supported models
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ],
    temperature=0.7,
    max_tokens=512,
    top_p=1,
)

# Display the generated answer
print("\n--- Groq LLM Response ---\n")
print_wrapped(chat_completion.choices[0].message.content)


def dot_product(vector1, vector2):
    return torch.dot(vector1, vector2)

def cosine_similarity(vector1, vector2):
    dot_product = torch.dot(vector1, vector2)

   
    norm_vector1 = torch.sqrt(torch.sum(vector1**2))
    norm_vector2 = torch.sqrt(torch.sum(vector2**2))

    return dot_product / (norm_vector1 * norm_vector2)


vector1 = torch.tensor([1, 2, 3], dtype=torch.float32)
vector2 = torch.tensor([1, 2, 3], dtype=torch.float32)
vector3 = torch.tensor([4, 5, 6], dtype=torch.float32)
vector4 = torch.tensor([-1, -2, -3], dtype=torch.float32)

print("Dot product between vector1 and vector2:", dot_product(vector1, vector2))
print("Dot product between vector1 and vector3:", dot_product(vector1, vector3))
print("Dot product between vector1 and vector4:", dot_product(vector1, vector4))

print("Cosine similarity between vector1 and vector2:", cosine_similarity(vector1, vector2))
print("Cosine similarity between vector1 and vector3:", cosine_similarity(vector1, vector3))
print("Cosine similarity between vector1 and vector4:", cosine_similarity(vector1, vector4))

groq_api_key = os.getenv("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")
client = Groq(api_key=groq_api_key)

