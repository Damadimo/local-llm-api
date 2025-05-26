from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import List, Literal, Optional
from llama_cpp import Llama

app = FastAPI(title = "Llama 2 Chat API", description = "FastAPI servive which mimics OpenAI's API", version = "1.0.0")


# Load the model
llm = Llama.from_pretrained(
	repo_id="TheBloke/Llama-2-7B-Chat-GGUF",
	filename="llama-2-7b-chat.Q4_K_M.gguf",
)

class Message(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str = Field("...", description = "The message content")

class ChatCompletionRequest(BaseModel):
    model: str = Field("...", description = "The model name")
    messages: List[Message] = Field("...", description = "The list of messages with role and content")
    max_tokens: Optional[int] = Field(100, description = "The maximum number of tokens to generate")
    temperature: Optional[float] = Field(0.7, description = "The temperature")

class Choice(BaseModel):
    message: Message
    index: int
    finish_reason: Literal["stop", "length", "tool_calls"]

class Usage(BaseModel):
    prompt_tokens: int = Field(..., description = "The number of tokens in the prompt")
    completion_tokens: int = Field(..., description = "The number of tokens in the completion")
    total_tokens: int = Field(..., description = "The total number of tokens")

class ChatCompletionResponse(BaseModel):
    id: str = Field("chatcmpl-1", description="Unique ID for the completion")
    object: Literal["chat.completion"] = "chat.completion"
    choices: List[Choice]
    usage: Usage


@app.get("/")
async def rootPage():
    return {"message": "Make a POST request to /v1/chat/completions"}

@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_completion(request: ChatCompletionRequest):
# Extract system instruction (if present)
    system_instr = next((m.content for m in request.messages if m.role == "system"), "")

    # Collect non-system messages (history)
    history = [m for m in request.messages if m.role != "system"]

    # Truncate history to last N turns to keep the prompt a reasonable size

    MAX_HISTORY = 10 #Can change this depending on your use case

    if len(history) > MAX_HISTORY:
        history = history[-MAX_HISTORY:]

    # Build the prompt for the model
    prompt_lines = []
    if system_instr:
        prompt_lines.append(f"### System: {system_instr}")
    for msg in history:
        tag = "Human" if msg.role == "user" else "Assistant"
        prompt_lines.append(f"### {tag}: {msg.content}")

    # Start the assistant turn
    prompt_lines.append("### Assistant:")
    prompt = "\n".join(prompt_lines)

    # Generate the response
    output = llm(prompt, max_tokens=request.max_tokens, temperature=request.temperature, stop = ["###"])

    reply = output["choices"][0]["text"].strip()
    # Format the response
    assistant_message = Message(role="assistant", content=reply)
    

    # Create the choice object
    choice = Choice(
        message=assistant_message,
        index=0,
        finish_reason="stop" if output["choices"][0]["finish_reason"] == "stop" else "length"
    )
    
    # Create usage statistics
    usage = Usage(
        prompt_tokens=output["usage"]["prompt_tokens"],
        completion_tokens=output["usage"]["completion_tokens"],
        total_tokens=output["usage"]["total_tokens"]
    )
    
    # Return the formatted response
    return ChatCompletionResponse(
        id="chatcmpl-" + str(hash(prompt))[:8],  # Generate a unique ID
        choices=[choice],
        usage=usage
    )
    
