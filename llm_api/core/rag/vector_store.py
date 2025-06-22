"""
Vector store implementation using Qdrant.
"""
import logging
import os
from typing import List, Optional, Dict, Any, Union
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
import uuid

from llm_api.config.settings import settings

logger = logging.getLogger(__name__)


class VectorStore:
    """Vector store interface using Qdrant."""
    
    def __init__(self, collection_name: str = "documents"):
        """Initialize vector store."""
        self.collection_name = collection_name
        self.client = self._get_qdrant_client()
        self._ensure_collection_exists()
    
    def _get_qdrant_client(self) -> QdrantClient:
        """Get Qdrant client based on configuration."""
        qdrant_mode = os.getenv("QDRANT_MODE", "memory")
        
        if qdrant_mode == "memory":
            logger.info("Using in-memory Qdrant client")
            return QdrantClient(":memory:")
        elif qdrant_mode == "local":
            qdrant_path = os.getenv("QDRANT_PATH", "./qdrant_storage")
            logger.info(f"Using local Qdrant client at {qdrant_path}")
            return QdrantClient(path=qdrant_path)
        elif qdrant_mode == "remote":
            qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
            qdrant_api_key = os.getenv("QDRANT_API_KEY")
            logger.info(f"Using remote Qdrant client at {qdrant_url}")
            return QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
        else:
            logger.warning(f"Unknown QDRANT_MODE: {qdrant_mode}, defaulting to memory")
            return QdrantClient(":memory:")
    
    def _ensure_collection_exists(self):
        """Ensure the collection exists with proper configuration."""
        try:
            # Check if collection exists
            collections = self.client.get_collections().collections
            collection_exists = any(col.name == self.collection_name for col in collections)
            
            if not collection_exists:
                logger.info(f"Creating collection: {self.collection_name}")
                
                # Create collection with vector configuration
                # Using 1024 dimensions for Jina embeddings v3
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=1024,  # Jina embeddings v3 dimension
                        distance=Distance.COSINE
                    )
                )
                logger.info(f"Collection '{self.collection_name}' created successfully")
            else:
                logger.info(f"Collection '{self.collection_name}' already exists")
                
        except Exception as e:
            logger.error(f"Error ensuring collection exists: {e}")
            raise
    
    def add_documents(
        self, 
        texts: List[str], 
        embeddings: List[List[float]], 
        metadata: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None
    ) -> List[str]:
        """
        Add documents to the vector store.
        
        Args:
            texts: List of document texts
            embeddings: List of embedding vectors
            metadata: Optional metadata for each document
            ids: Optional IDs for each document
            
        Returns:
            List of document IDs
        """
        if len(texts) != len(embeddings):
            raise ValueError("Number of texts must match number of embeddings")
        
        if metadata and len(metadata) != len(texts):
            raise ValueError("Number of metadata entries must match number of texts")
        
        # Generate IDs if not provided
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in texts]
        
        # Prepare points for upload
        points = []
        for i, (text, embedding) in enumerate(zip(texts, embeddings)):
            payload = {
                "text": text,
                "source": metadata[i].get("source", "unknown") if metadata else "unknown"
            }
            
            # Add additional metadata if provided
            if metadata and i < len(metadata):
                payload.update(metadata[i])
            
            points.append(PointStruct(
                id=ids[i],
                vector=embedding,
                payload=payload
            ))
        
        try:
            # Upload points to collection
            self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )
            
            logger.info(f"Added {len(points)} documents to vector store")
            return ids
            
        except Exception as e:
            logger.error(f"Error adding documents: {e}")
            raise
    
    def search(
        self, 
        query_embedding: List[float], 
        limit: int = 5,
        score_threshold: Optional[float] = None,
        filter_conditions: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar documents.
        
        Args:
            query_embedding: Query vector
            limit: Maximum number of results
            score_threshold: Minimum similarity score
            filter_conditions: Optional filter conditions
            
        Returns:
            List of search results with text, metadata, and scores
        """
        try:
            # Prepare filter if provided
            search_filter = None
            if filter_conditions:
                conditions = []
                for key, value in filter_conditions.items():
                    conditions.append(
                        FieldCondition(key=key, match=MatchValue(value=value))
                    )
                if conditions:
                    search_filter = Filter(must=conditions)
            
            # Perform search
            search_results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=limit,
                score_threshold=score_threshold,
                query_filter=search_filter
            )
            
            # Format results
            results = []
            for result in search_results:
                results.append({
                    "id": result.id,
                    "text": result.payload.get("text", ""),
                    "score": result.score,
                    "metadata": {k: v for k, v in result.payload.items() if k != "text"}
                })
            
            logger.info(f"Found {len(results)} similar documents")
            return results
            
        except Exception as e:
            logger.error(f"Error searching documents: {e}")
            raise
    
    def delete_documents(self, ids: List[str]) -> bool:
        """Delete documents by IDs."""
        try:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=ids
            )
            logger.info(f"Deleted {len(ids)} documents")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting documents: {e}")
            return False
    
    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the collection."""
        try:
            info = self.client.get_collection(self.collection_name)
            return {
                "total_documents": info.points_count,
                "vector_dimensions": info.config.params.vectors.size,
                "vectors_count": info.vectors_count,
                "status": "green" if str(info.status).endswith("GREEN") else "yellow"
            }
        except Exception as e:
            logger.error(f"Error getting collection info: {e}")
            return {
                "total_documents": 0,
                "vector_dimensions": 1024,
                "vectors_count": 0,
                "status": "error"
            }


# Global vector store instance
vector_store = VectorStore() 