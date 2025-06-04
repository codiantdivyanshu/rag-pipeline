from sentence_transformers import SentenceTransformer
import torch

DEVICE = "cpu"  
model = SentenceTransformer("all-mpnet-base-v2", device=DEVICE)

def embed_chunks(pages_and_chunks):
    """
    Takes extracted text chunks, encodes into embeddings tensor.
    Returns list of dicts with embeddings and metadata, and raw embeddings tensor.
    """
    try:
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
    except Exception as e:
        raise RuntimeError(f"Embedding failed: {str(e)}")
