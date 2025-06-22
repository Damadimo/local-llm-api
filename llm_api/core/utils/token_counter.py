"""
Token counting utilities.
"""
import logging
from typing import List, Optional

logger = logging.getLogger(__name__)


class TokenCounter:
    """Handles token counting for chat completions."""
    
    def __init__(self, tokenizer=None):
        """Initialize with optional tokenizer."""
        self.tokenizer = tokenizer
    
    def count_tokens_in_text(self, text: str) -> int:
        """Count tokens in a text string."""
        if self.tokenizer is None:
            # Fallback: rough estimation (1 token ≈ 4 characters for English)
            return max(1, len(text) // 4)
        
        try:
            # Use actual tokenizer if available
            tokens = self.tokenizer.encode(text)
            return len(tokens)
        except Exception as e:
            logger.warning(f"Token counting failed, using fallback: {e}")
            return max(1, len(text) // 4)
    
    def count_tokens_in_messages(self, messages: List) -> int:
        """Count tokens in a list of messages."""
        total = 0
        for message in messages:
            # Count role tokens (approximate)
            total += 3  # For role formatting
            # Count content tokens
            if hasattr(message, 'content'):
                total += self.count_tokens_in_text(message.content)
            elif isinstance(message, dict) and 'content' in message:
                total += self.count_tokens_in_text(message['content'])
        
        # Add overhead for conversation formatting
        total += len(messages) * 2  # Approximate formatting overhead
        return total
    
    def estimate_response_tokens(self, response_text: str) -> int:
        """Estimate tokens in a response."""
        return self.count_tokens_in_text(response_text)


# Global token counter instance (will be updated when models are loaded)
token_counter = TokenCounter() 