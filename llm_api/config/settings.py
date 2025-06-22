"""
Configuration settings for the LLM API.
"""
import os
from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )
    
    # Server Settings
    host: str = Field(default="127.0.0.1", description="Server host")
    port: int = Field(default=8000, description="Server port")
    reload: bool = Field(default=False, description="Enable auto-reload in development")
    
    # Model Settings
    model_repo_id: str = Field(
        default="TheBloke/Llama-2-7B-Chat-GGUF",
        description="HuggingFace model repository ID"
    )
    model_filename: str = Field(
        default="llama-2-7b-chat.Q4_K_M.gguf",
        description="Model filename"
    )
    model_path: Optional[Path] = Field(
        default=None,
        description="Direct path to model file (overrides repo_id/filename)"
    )
    
    # LLM Generation Settings
    max_tokens: int = Field(default=256, description="Maximum tokens to generate")
    temperature: float = Field(default=0.2, description="Sampling temperature")
    context_length: int = Field(default=2048, description="Context window size")
    
    # Embedding Settings
    embedding_model_name: str = Field(
        default="jinaai/jina-embeddings-v3",
        description="Embedding model name"
    )
    embedding_dimension: int = Field(default=1024, description="Embedding vector dimension")
    
    # Vector Store Settings
    qdrant_host: str = Field(default="localhost", description="Qdrant host")
    qdrant_port: int = Field(default=6333, description="Qdrant port")
    qdrant_collection_name: str = Field(default="knowledge_base", description="Qdrant collection name")
    qdrant_use_memory: bool = Field(default=True, description="Use in-memory Qdrant for development")
    
    # RAG Settings
    rag_top_k: int = Field(default=3, description="Number of documents to retrieve for RAG")
    rag_score_threshold: float = Field(default=0.7, description="Minimum similarity score for RAG")
    
    # Paths
    data_dir: Path = Field(default=Path("data"), description="Data directory")
    models_dir: Path = Field(default=Path("data/models"), description="Models directory")
    knowledge_dir: Path = Field(default=Path("data/knowledge"), description="Knowledge base directory")
    
    # Development Settings
    debug: bool = Field(default=False, description="Enable debug mode")
    log_level: str = Field(default="INFO", description="Logging level")
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Ensure directories exist
        self.data_dir.mkdir(exist_ok=True)
        self.models_dir.mkdir(exist_ok=True)
        self.knowledge_dir.mkdir(exist_ok=True)


# Global settings instance
settings = Settings() 