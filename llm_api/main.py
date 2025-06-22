"""
Main FastAPI application entry point.
"""
import logging
import time
import uuid
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse

from llm_api.config.settings import settings
from llm_api.api.models.chat import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    Message,
    Choice,
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
    RAGQueryResponse,
    RAGChatResponse,
    RAGDocumentsResponse,
    RAGStatsResponse
)
from llm_api.core.llm.embeddings_handler import embeddings_handler
from llm_api.core.llm.chat_handler import chat_handler
from llm_api.core.functions.registry import function_registry
from llm_api.core.rag.pipeline import rag_pipeline
from llm_api.core.rag.vector_store import vector_store

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper()),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="LLM API Server",
    description="Local LLM API with Function Calling and RAG",
    version="0.1.0",
    debug=settings.debug
)


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "version": "0.1.0"
    }


@app.post("/v1/embeddings", response_model=EmbeddingsResponse)
async def create_embeddings(request: EmbeddingsRequest):
    """
    Create embeddings for the input texts.
    """
    try:
        logger.info(f"Processing embeddings request for {len(request.input)} texts")
        
        # Generate embeddings
        embeddings_list = embeddings_handler.embed_to_list(request.input)
        
        # Format response
        embedding_data = [
            EmbeddingData(
                embedding=embedding,
                index=i
            )
            for i, embedding in enumerate(embeddings_list)
        ]
        
        # Calculate usage (rough estimate)
        total_tokens = sum(len(text.split()) for text in request.input)
        usage = EmbeddingsUsage(
            prompt_tokens=total_tokens,
            total_tokens=total_tokens
        )
        
        return EmbeddingsResponse(
            data=embedding_data,
            model=request.model,
            usage=usage
        )
        
    except Exception as e:
        logger.error(f"Error generating embeddings: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def create_chat_completion(request: ChatCompletionRequest):
    """
    Chat completion endpoint (non-streaming).
    """
    try:
        # Check if streaming is requested
        if request.stream:
            return StreamingResponse(
                chat_handler.create_stream_completion(request),
                media_type="text/event-stream"
            )
        
        # Regular (non-streaming) completion
        response = await chat_handler.create_completion(request)
        return response
        
    except Exception as e:
        logger.error(f"Error in chat completion: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/chat/completions/stream")
async def stream_chat_completion(request: ChatCompletionRequest):
    """
    Streaming chat completion endpoint.
    """
    try:
        return StreamingResponse(
            chat_handler.create_stream_completion(request),
            media_type="text/event-stream"
        )
        
    except Exception as e:
        logger.error(f"Error in streaming chat completion: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/v1/functions")
async def list_functions():
    """List available functions."""
    return {
        "functions": function_registry.get_function_specs(),
        "count": len(function_registry.list_functions())
    }


@app.post("/v1/rag/documents", response_model=RAGDocumentsResponse)
async def add_documents(request: RAGDocumentsRequest):
    """Add documents to the RAG knowledge base."""
    try:
        doc_ids = rag_pipeline.add_documents(texts=request.texts, metadata=request.metadata)
        return RAGDocumentsResponse(
            document_ids=doc_ids,
            count=len(doc_ids),
            status="success"
        )
    except Exception as e:
        logger.error(f"Error adding documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/rag/query", response_model=RAGQueryResponse)
async def rag_query(request: RAGQueryRequest):
    """Perform a RAG query to retrieve relevant context."""
    try:
        result = rag_pipeline.query(
            user_query=request.query,
            num_context_docs=request.num_context_docs,
            score_threshold=request.score_threshold
        )
        return RAGQueryResponse(**result)
    except Exception as e:
        logger.error(f"Error in RAG query: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/rag/chat", response_model=RAGChatResponse)
async def rag_chat_completion(request: RAGChatRequest):
    """Perform RAG-enhanced chat completion with retrieved context."""
    try:
        result = await rag_pipeline.chat_completion(
            user_query=request.query,
            num_context_docs=request.num_context_docs,
            score_threshold=request.score_threshold,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            system_message=request.system_message
        )
        return RAGChatResponse(**result)
    except Exception as e:
        logger.error(f"Error in RAG chat completion: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/v1/rag/stats", response_model=RAGStatsResponse)
async def get_rag_stats():
    """Get RAG knowledge base statistics."""
    try:
        stats = rag_pipeline.get_knowledge_base_stats()
        return RAGStatsResponse(**stats)
    except Exception as e:
        logger.error(f"Error getting RAG stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    
    logger.info("Starting LLM API server...")
    uvicorn.run(
        "llm_api.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.reload,
        log_level=settings.log_level.lower()
    ) 