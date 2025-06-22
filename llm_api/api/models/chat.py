"""
Chat completion API models.
"""
from typing import List, Optional, Dict, Any, Literal
from pydantic import BaseModel, Field


class Message(BaseModel):
    """A message in a conversation."""
    role: Literal["system", "user", "assistant", "function"]
    content: str
    name: Optional[str] = None  # For function messages
    function_call: Optional["FunctionCall"] = None  # For assistant messages with function calls


class FunctionCall(BaseModel):
    """Function call information."""
    name: str
    arguments: Dict[str, Any]


class ChatCompletionRequest(BaseModel):
    """Request for chat completion."""
    model: str = Field(default="llama-2-7b-chat", description="Model to use")
    messages: List[Message] = Field(description="List of messages in the conversation")
    max_tokens: Optional[int] = Field(default=256, description="Maximum tokens to generate")
    temperature: Optional[float] = Field(default=0.2, ge=0.0, le=2.0, description="Sampling temperature")
    stream: Optional[bool] = Field(default=False, description="Whether to stream the response")
    functions: Optional[List[Dict[str, Any]]] = Field(default=None, description="Available functions")
    function_call: Optional[str] = Field(default=None, description="Function call behavior")


class Delta(BaseModel):
    """Delta for streaming responses."""
    role: Optional[str] = None
    content: Optional[str] = None
    function_call: Optional[FunctionCall] = None


class Choice(BaseModel):
    """A choice in the completion response."""
    message: Optional[Message] = None
    delta: Optional[Delta] = None  # For streaming
    index: int = 0
    finish_reason: Optional[Literal["stop", "length", "function_call"]] = None


class Usage(BaseModel):
    """Token usage information."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatCompletionResponse(BaseModel):
    """Response from chat completion."""
    id: str
    object: Literal["chat.completion", "chat.completion.chunk"] = "chat.completion"
    created: int = Field(description="Unix timestamp")
    model: str
    choices: List[Choice]
    usage: Optional[Usage] = None


class ChatCompletionStreamResponse(BaseModel):
    """Streaming response from chat completion."""
    id: str
    object: Literal["chat.completion.chunk"] = "chat.completion.chunk"
    created: int = Field(description="Unix timestamp")
    model: str
    choices: List[Choice] 