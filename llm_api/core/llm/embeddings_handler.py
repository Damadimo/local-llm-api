"""
Embeddings handler for text embedding generation.
"""
import logging
from typing import List

import torch
from transformers import AutoTokenizer, AutoModel

from llm_api.config.settings import settings

logger = logging.getLogger(__name__)


class EmbeddingsHandler:
    """Handles text embeddings using the configured model."""
    
    def __init__(self):
        """Initialize the embeddings handler."""
        self.model_name = settings.embedding_model_name
        self.model = None
        self.tokenizer = None
        self._load_model()
    
    def _load_model(self) -> None:
        """Load the embedding model and tokenizer."""
        logger.info(f"Loading embedding model: {self.model_name}")
        
        try:
            self.model = AutoModel.from_pretrained(
                self.model_name, 
                trust_remote_code=True
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, 
                trust_remote_code=True
            )
            
            # Move to GPU if available
            if torch.cuda.is_available():
                self.model.cuda()
                logger.info("Model moved to GPU")
            else:
                logger.info("Using CPU for embedding model")
                
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise
    
    def embed(self, texts: List[str]) -> torch.Tensor:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            Tensor of embeddings with shape (len(texts), embedding_dim)
        """
        if not texts:
            raise ValueError("No texts provided for embedding")
        
        logger.debug(f"Generating embeddings for {len(texts)} texts")
        
        # Tokenize inputs
        encoded = self.tokenizer(
            texts, 
            return_tensors="pt", 
            padding=True, 
            truncation=True
        )
        
        # Move to GPU if available
        if torch.cuda.is_available():
            encoded = {key: tensor.cuda() for key, tensor in encoded.items()}
        
        # Generate embeddings
        with torch.no_grad():
            output = self.model(**encoded)
        
        # Extract embeddings (with pooling if needed)
        if hasattr(output, "pooler_output") and output.pooler_output is not None:
            embeddings = output.pooler_output
        else:
            # Mean pooling with attention mask
            last_hidden_state = output.last_hidden_state
            mask = encoded["attention_mask"].unsqueeze(-1).float()
            mask_hidden = last_hidden_state * mask
            sum_hidden = mask_hidden.sum(dim=1)
            counts = mask.sum(dim=1).clamp(min=1e-9)
            embeddings = sum_hidden / counts
        
        # Move back to CPU for return
        embeddings = embeddings.cpu()
        return embeddings
    
    def embed_to_list(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings and return as list of lists.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors as lists of floats
        """
        embeddings = self.embed(texts)
        return embeddings.tolist()


# Global instance
embeddings_handler = EmbeddingsHandler() 