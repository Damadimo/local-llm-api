"""API models package."""

from llm_api.api.models.chat import (
    Message,
    FunctionCall, 
    ChatCompletionRequest,
    ChatCompletionResponse,
    Choice,
    Delta,
    Usage
)
from llm_api.api.models.embeddings import (
    EmbeddingsRequest,
    EmbeddingsResponse,
    EmbeddingData,
    EmbeddingsUsage
)
from llm_api.api.models.rag import (
    RAGQueryRequest,
    RAGDocumentsRequest,
    RAGChatRequest,
    ContextDocument,
    RAGQueryResponse,
    RAGChatResponse,
    RAGDocumentsResponse,
    RAGStatsResponse
)

__all__ = [
    "Message",
    "FunctionCall",
    "ChatCompletionRequest", 
    "ChatCompletionResponse",
    "Choice",
    "Delta",
    "Usage",
    "EmbeddingsRequest",
    "EmbeddingsResponse",
    "EmbeddingData",
    "EmbeddingsUsage",
    "RAGQueryRequest",
    "RAGDocumentsRequest",
    "RAGChatRequest",
    "ContextDocument",
    "RAGQueryResponse",
    "RAGChatResponse",
    "RAGDocumentsResponse",
    "RAGStatsResponse"
]
