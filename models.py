from pydantic import BaseModel
from typing import List, Literal, Optional, Dict, Any

class Message(BaseModel):
    role: Literal["system", "user", "assistant", "function"]
    content: str

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Message
    max_tokens: Optional[int] = 256
    temperature: Optional[float] = 0.2
    stream: Optional[bool] = False

class FunctionCall(BaseModel):
    name: str
    arguments: Dict[str, Any]

class Delta(BaseModel):
    role: Optional[str] = None
    content: Optional[str] = None
    function_call: Optional[FunctionCall] = None

class Choice(BaseModel):
    message: Optional[Message] = None
    delta: Optional[Delta] = None
    index: int = 0
    finish_reason: Optional[Literal["stop", "length", "function_call"]] = None

class ChatCompletionResponse(BaseModel):
    id: str
    object: Literal["chat.completion", "chat.completion.chunk"] = "chat.completion"
    choices: List[Choice]
    usage: Dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

class EmbeddingsRequest(BaseModel):
    texts: List[str]

class EmbeddingsResponse(BaseModel):
    embeddings: List[List[float]]
    model: str
