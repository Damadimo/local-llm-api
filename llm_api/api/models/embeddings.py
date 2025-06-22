"""
Embeddings API models.
"""
from typing import List
from pydantic import BaseModel, Field


class EmbeddingsRequest(BaseModel):
    """Request for text embeddings."""
    input: List[str] = Field(description="List of texts to embed")
    model: str = Field(default="jina-embeddings-v3", description="Embedding model to use")


class EmbeddingData(BaseModel):
    """Individual embedding data."""
    object: str = "embedding"
    embedding: List[float]
    index: int


class EmbeddingsUsage(BaseModel):
    """Usage information for embeddings."""
    prompt_tokens: int
    total_tokens: int


class EmbeddingsResponse(BaseModel):
    """Response from embeddings endpoint."""
    object: str = "list"
    data: List[EmbeddingData]
    model: str
    usage: EmbeddingsUsage 