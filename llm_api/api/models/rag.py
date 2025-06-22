"""
RAG API models.
"""
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class RAGQueryRequest(BaseModel):
    """Request for RAG query."""
    query: str = Field(description="The query to search for in the knowledge base")
    num_context_docs: int = Field(default=3, ge=1, le=10, description="Number of context documents to retrieve")
    score_threshold: float = Field(default=0.7, ge=0.0, le=1.0, description="Minimum similarity score threshold")


class RAGDocumentsRequest(BaseModel):
    """Request for adding documents to RAG knowledge base."""
    texts: List[str] = Field(description="List of texts to add to the knowledge base")
    metadata: Optional[List[Dict[str, Any]]] = Field(default=None, description="Optional metadata for each document")


class RAGChatRequest(BaseModel):
    """Request for RAG-enhanced chat completion."""
    query: str = Field(description="The user's question")
    num_context_docs: int = Field(default=3, ge=1, le=10, description="Number of context documents to retrieve")
    score_threshold: float = Field(default=0.5, ge=0.0, le=1.0, description="Minimum similarity score threshold")
    max_tokens: Optional[int] = Field(default=200, ge=1, le=512, description="Maximum tokens in response")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Sampling temperature")
    system_message: Optional[str] = Field(default=None, description="Optional system message override")


class ContextDocument(BaseModel):
    """A context document retrieved for RAG."""
    text: str = Field(description="The document text")
    score: float = Field(description="Similarity score")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Document metadata")


class RAGQueryResponse(BaseModel):
    """Response for RAG query."""
    query: str = Field(description="The original query")
    context_documents: List[ContextDocument] = Field(description="Retrieved context documents")
    num_docs_retrieved: int = Field(description="Number of documents retrieved")
    augmented_prompt: str = Field(description="The augmented prompt with context")


class RAGChatResponse(BaseModel):
    """Response for RAG-enhanced chat completion."""
    query: str = Field(description="The original query")
    answer: str = Field(description="The generated answer")
    context_documents: List[ContextDocument] = Field(description="Retrieved context documents used")
    num_docs_retrieved: int = Field(description="Number of documents retrieved")
    tokens_used: int = Field(description="Total tokens used in generation")
    retrieval_time_ms: float = Field(description="Time taken for retrieval in milliseconds")
    generation_time_ms: float = Field(description="Time taken for generation in milliseconds")


class RAGDocumentsResponse(BaseModel):
    """Response for adding documents."""
    document_ids: List[str] = Field(description="IDs of added documents")
    count: int = Field(description="Number of documents added")
    status: str = Field(description="Operation status")


class RAGStatsResponse(BaseModel):
    """Response for RAG statistics."""
    total_documents: int = Field(description="Total number of documents in knowledge base")
    vector_dimensions: int = Field(description="Vector embedding dimensions")
    status: str = Field(description="Knowledge base status") 