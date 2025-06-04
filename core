from sentence_transformers import SentenceTransformer, util
from config.config_loader import load_config
from logger.logger import get_logger
from settings.base import AppSettings
import torch

config = load_config()
settings = AppSettings()
logger = get_logger()

model = SentenceTransformer(config["embedding_model"], device=config["device"])

from groq import Groq
client = Groq(api_key=settings.groq_api_key)

def query_pipeline(query, chunks, embeddings_tensor, top_k=None):
    try:
        top_k = top_k or config.get("top_k", 5)
        query_embedding = model.encode(query, convert_to_tensor=True)
        scores = util.cos_sim(query_embedding, embeddings_tensor)[0]
        top_k = min(top_k, len(scores))  # prevent overflow
        top_chunks = [chunks[i] for i in torch.topk(scores, k=top_k).indices.tolist()]

        context = "\n".join(f"[Page {c['meta']['page']}] {c['sentence_chunk']}" for c in top_chunks)

        response = client.chat.completions.create(
            model=config["llm_model"],
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
            ]
        )

        return response.choices[0].message.content, context
    except Exception as e:
        logger.error(f"Query pipeline failed: {str(e)}")
        raise
