from pydantic import BaseModel, Field
from typing import List, Dict, Optional


class UploadPDFResponse(BaseModel):
    message: str
    total_chunks: int
    file_name: str


class QueryRequest(BaseModel):
    question: str = Field(..., example="What is the refund policy?")
    embedding_path: str = Field(..., example="embeddings/output.csv")


class QueryResponse(BaseModel):
    answer: str
    context: str


class ChunkMetadata(BaseModel):
    page: int


class EmbeddedChunk(BaseModel):
    sentence_chunk: str
    meta: ChunkMetadata
    embedding: List[float]
