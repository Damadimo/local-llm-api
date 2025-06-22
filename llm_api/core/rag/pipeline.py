"""
RAG (Retrieval-Augmented Generation) pipeline.
"""
import logging
import time
from typing import List, Dict, Any, Optional

from llm_api.core.llm.embeddings_handler import embeddings_handler
from llm_api.core.rag.vector_store import vector_store
from llm_api.core.llm.chat_handler import chat_handler
from llm_api.api.models.chat import ChatCompletionRequest, Message
from llm_api.api.models.rag import ContextDocument

logger = logging.getLogger(__name__)


class RAGPipeline:
    """Retrieval-Augmented Generation pipeline."""
    
    def __init__(self):
        """Initialize RAG pipeline."""
        self.embeddings = embeddings_handler
        self.vector_store = vector_store
        self.chat_handler = chat_handler
        # RAG-specific settings
        self.max_context_length = 1500  # Increased from 800 to use more of the available context
        self.max_tokens = 300  # Increased from 200 for better responses
    
    def add_documents(
        self, 
        texts: List[str], 
        metadata: Optional[List[Dict[str, Any]]] = None
    ) -> List[str]:
        """
        Add documents to the RAG knowledge base.
        
        Args:
            texts: List of document texts
            metadata: Optional metadata for each document
            
        Returns:
            List of document IDs
        """
        logger.info(f"Adding {len(texts)} documents to RAG knowledge base")
        
        try:
            # Generate embeddings for the texts
            embeddings = self.embeddings.embed_to_list(texts)
            
            # Store in vector database
            doc_ids = self.vector_store.add_documents(
                texts=texts,
                embeddings=embeddings,
                metadata=metadata
            )
            
            logger.info(f"Successfully added {len(doc_ids)} documents")
            return doc_ids
            
        except Exception as e:
            logger.error(f"Error adding documents to RAG: {e}")
            raise
    
    def retrieve_context(
        self, 
        query: str, 
        num_results: int = 3,
        score_threshold: float = 0.7
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant context for a query.
        
        Args:
            query: Query text
            num_results: Number of results to retrieve
            score_threshold: Minimum similarity score
            
        Returns:
            List of relevant documents with metadata
        """
        logger.info(f"Retrieving context for query: '{query[:50]}...'")
        
        try:
            # Generate embedding for the query
            query_embedding = self.embeddings.embed_to_list([query])[0]
            
            # Search for relevant documents
            results = self.vector_store.search(
                query_embedding=query_embedding,
                limit=num_results,
                score_threshold=score_threshold
            )
            
            logger.info(f"Retrieved {len(results)} relevant documents")
            return results
            
        except Exception as e:
            logger.error(f"Error retrieving context: {e}")
            return []
    
    def format_context_documents(self, context_docs: List[Dict[str, Any]]) -> List[ContextDocument]:
        """Convert raw context documents to ContextDocument models."""
        formatted_docs = []
        for doc in context_docs:
            formatted_docs.append(ContextDocument(
                text=doc.get("text", ""),
                score=doc.get("score", 0.0),
                metadata=doc.get("metadata")
            ))
        return formatted_docs
    
    def generate_augmented_prompt(
        self, 
        user_query: str, 
        context_docs: List[Dict[str, Any]],
        system_message: Optional[str] = None,
        max_context_length: int = 800
    ) -> str:
        """
        Generate an augmented prompt with retrieved context.
        
        Args:
            user_query: Original user query
            context_docs: Retrieved context documents
            system_message: Optional custom system message
            max_context_length: Maximum length of context to include
            
        Returns:
            Augmented prompt string
        """
        if not context_docs:
            return user_query
        
        # Build context from retrieved documents
        context_parts = []
        current_length = 0
        
        for i, doc in enumerate(context_docs, 1):
            text = doc.get("text", "")
            score = doc.get("score", 0.0)
            source = doc.get("metadata", {}).get("source", f"Doc{i}")
            
            # Truncate text if needed to fit within context limit
            max_text_length = max_context_length // len(context_docs) - 50  # Reserve space for formatting
            if len(text) > max_text_length:
                text = text[:max_text_length] + "..."
            
            # Add document with minimal formatting
            doc_text = f"[{source}]: {text}\n"
            
            if current_length + len(doc_text) > max_context_length:
                break
                
            context_parts.append(doc_text)
            current_length += len(doc_text)
        
        if not context_parts:
            return user_query
        
        # Build compact augmented prompt
        context = "\n".join(context_parts)
        
        # Use very concise system instruction
        if system_message:
            instruction = system_message
        else:
            instruction = "Answer based on the provided context. If context lacks info, say so."
        
        # Compact prompt format
        augmented_prompt = f"""{instruction}

Context:
{context}

Q: {user_query}
A:"""
        
        return augmented_prompt
    
    def query(
        self, 
        user_query: str, 
        num_context_docs: int = 3,
        score_threshold: float = 0.7
    ) -> Dict[str, Any]:
        """
        Perform a complete RAG query.
        
        Args:
            user_query: User's question
            num_context_docs: Number of context documents to retrieve
            score_threshold: Minimum similarity score for retrieval
            
        Returns:
            Dictionary with augmented prompt and context information
        """
        try:
            # Retrieve relevant context
            context_docs = self.retrieve_context(
                query=user_query,
                num_results=num_context_docs,
                score_threshold=score_threshold
            )
            
            # Generate augmented prompt
            augmented_prompt = self.generate_augmented_prompt(
                user_query=user_query,
                context_docs=context_docs
            )
            
            # Format context documents
            formatted_docs = self.format_context_documents(context_docs)
            
            return {
                "query": user_query,
                "augmented_prompt": augmented_prompt,
                "context_documents": formatted_docs,
                "num_docs_retrieved": len(context_docs)
            }
            
        except Exception as e:
            logger.error(f"Error in RAG query: {e}")
            return {
                "query": user_query,
                "augmented_prompt": user_query,  # Fallback to original query
                "context_documents": [],
                "num_docs_retrieved": 0,
                "error": str(e)
            }
    
    async def chat_completion(
        self,
        user_query: str,
        num_context_docs: int = 3,
        score_threshold: float = 0.5,
        max_tokens: int = 200,
        temperature: float = 0.7,
        system_message: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Perform RAG-enhanced chat completion.
        
        Args:
            user_query: User's question
            num_context_docs: Number of context documents to retrieve
            score_threshold: Minimum similarity score for retrieval
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature
            system_message: Optional system message override
            
        Returns:
            Dictionary with answer and context information
        """
        retrieval_start = time.time()
        
        try:
            # Retrieve relevant context
            context_docs = self.retrieve_context(
                query=user_query,
                num_results=num_context_docs,
                score_threshold=score_threshold
            )
            
            retrieval_time_ms = (time.time() - retrieval_start) * 1000
            
            # Generate augmented prompt
            augmented_prompt = self.generate_augmented_prompt(
                user_query=user_query,
                context_docs=context_docs,
                system_message=system_message
            )
            
            # Create chat completion request
            chat_request = ChatCompletionRequest(
                model="local-llm",
                messages=[Message(role="user", content=augmented_prompt)],
                max_tokens=max_tokens,
                temperature=temperature,
                stream=False
            )
            
            # Generate response
            generation_start = time.time()
            response = await self.chat_handler.create_completion(chat_request)
            generation_time_ms = (time.time() - generation_start) * 1000
            
            # Extract answer from response
            answer = response.choices[0].message.content if response.choices else "No response generated"
            tokens_used = response.usage.total_tokens if response.usage else 0
            
            # Format context documents
            formatted_docs = self.format_context_documents(context_docs)
            
            return {
                "query": user_query,
                "answer": answer,
                "context_documents": formatted_docs,
                "num_docs_retrieved": len(context_docs),
                "tokens_used": tokens_used,
                "retrieval_time_ms": retrieval_time_ms,
                "generation_time_ms": generation_time_ms
            }
            
        except Exception as e:
            logger.error(f"Error in RAG chat completion: {e}")
            return {
                "query": user_query,
                "answer": f"Error generating response: {str(e)}",
                "context_documents": [],
                "num_docs_retrieved": 0,
                "tokens_used": 0,
                "retrieval_time_ms": (time.time() - retrieval_start) * 1000,
                "generation_time_ms": 0.0,
                "error": str(e)
            }
    
    def get_knowledge_base_stats(self) -> Dict[str, Any]:
        """Get statistics about the knowledge base."""
        try:
            return self.vector_store.get_collection_info()
        except Exception as e:
            logger.error(f"Error getting knowledge base stats: {e}")
            return {"error": str(e)}


# Global RAG pipeline instance
rag_pipeline = RAGPipeline() 