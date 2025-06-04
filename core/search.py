import torch
from sentence_transformers import util
import numpy as np
import pandas as pd

def save_embeddings(embedded_chunks, path):
    """
    Save embedded chunks (with metadata and embeddings) to CSV.
    Embeddings saved as strings for easy load.
    """
    try:
        for i, chunk in enumerate(embedded_chunks):
            if "embedding" not in chunk:
                raise ValueError(f"Chunk at index {i} missing 'embedding' field: {chunk}")

        df = pd.DataFrame(embedded_chunks)
        df["embedding"] = df["embedding"].apply(lambda x: np.array2string(np.array(x), separator=' '))
        df["meta"] = df["meta"].apply(str)
        df.to_csv(path, index=False)
    except Exception as e:
        raise RuntimeError(f"Saving embeddings to '{path}' failed: {str(e)}")

def load_embeddings(path):
    """
    Load saved embeddings CSV, reconstruct embeddings tensor and chunk metadata.
    """
    try:
        df = pd.read_csv(path)
        df["embedding"] = df["embedding"].apply(lambda x: np.fromstring(x.strip("[]"), sep=" "))
        df["meta"] = df["meta"].apply(eval) 
        chunks = df[["sentence_chunk", "meta"]].to_dict(orient="records")
        embeddings = torch.tensor(np.stack(df["embedding"].to_numpy()), dtype=torch.float32)
        return chunks, embeddings
    except Exception as e:
        raise RuntimeError(f"Loading embeddings from '{path}' failed: {str(e)}")

def search_top_k(query_embedding, embeddings_tensor, chunks, k=5):
    """
    Compute cosine similarity scores and return top-k matching chunks.
    """
    scores = util.cos_sim(query_embedding, embeddings_tensor)[0]
    top_k = torch.topk(scores, k=k)
    top_chunks = [chunks[i] for i in top_k.indices.tolist()]
    return top_chunks
