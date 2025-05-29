# main.py ────────────────────────────────────────────────────────────────────
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Literal, Optional, Dict, Any
from datetime import datetime
from zoneinfo import ZoneInfo
from llama_cpp import Llama
import json, uuid, re

# ── 0. FastAPI & model ──────────────────────────────────────────────────────
app = FastAPI(title="Llama-2 Function-Calling API", version="2.0.0")

llm = Llama.from_pretrained(
    repo_id="TheBloke/Llama-2-7B-Chat-GGUF",
    filename="llama-2-7b-chat.Q4_K_M.gguf",
)

# ── 1. Server-side Python tools ─────────────────────────────────────────────
def get_current_time() -> str:
    """Get current time in US Eastern Time."""
    return datetime.now(ZoneInfo("America/New_York")).isoformat()


def get_weather(location: str) -> str:
    return f" Weather in {location}: -25 °C, partly cloudy and extremely cold."

FUNCTIONS = {
    "get_current_time": {"fn": get_current_time, "params": {}},
    "get_weather": {"fn": get_weather, "params": {"location": str}},
}

# ── 2. Pydantic I/O models (unchanged except roles) ─────────────────────────
class Message(BaseModel):
    role: Literal["system", "user", "assistant", "function"]
    content: str

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Message]
    max_tokens: Optional[int] = 256
    temperature: Optional[float] = 0.2

class FunctionCall(BaseModel):
    function: str
    arguments: Dict[str, Any]

class Choice(BaseModel):
    message: Message
    index: int
    finish_reason: Literal["stop"]
    function_call: Optional[FunctionCall] = None

class ChatCompletionResponse(BaseModel):
    id: str
    object: Literal["chat.completion"] = "chat.completion"
    choices: List[Choice]
    usage: Dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

# ── 3. Utilities ────────────────────────────────────────────────────────────


def build_system_message() -> str:
    """Create clear instructions for function calling behavior."""
    # Build function list with parameters
    function_list = []
    # Build examples for each function
    examples = []
    
    for name, info in FUNCTIONS.items():
        # Function list entry
        params = ", ".join(f"{param}" for param in info["params"].keys())
        params = f"({params})" if params else "(no parameters)"
        function_list.append(f"- {name}{params}")
        
        # Example entry
        example_args = {}
        for param in info["params"].keys():
            example_args[param] = "London" if "location" in param else "default"
        
        examples.append(f"Q: What is the {name.replace('get_', '')} in {next(iter(example_args.values())) if example_args else 'now'}?\nA: " + 
                      json.dumps({"function": name, "arguments": example_args}))

    return f"""You are an AI assistant with access to real-time information through functions.

STRICT RULES:

1. ALWAYS call a function when asked about:
   - Current time → use get_current_time
   - Weather → use get_weather
   NO EXCEPTIONS. If the question matches, make the function call.

2. Function Call Format:
   {{"function": "name", "arguments": {{...}}}}
   Output ONLY this JSON, nothing else.

3. Never try to answer time/weather questions yourself.
   Let the functions provide the real data.

Available Functions:
{chr(10).join(function_list)}

Example Interactions:
{chr(10).join(examples)}

WRONG RESPONSES:
Q: What's the weather in Tokyo?
A: Let me check the weather...
A: The weather is sunny...
A: What's the weather like in Tokyo?

CORRECT RESPONSES:
Q: What's the weather in Tokyo?
A: {{"function": "get_weather", "arguments": {{"location": "Tokyo"}}}}

Q: How are you today?
A: I'm doing well, thank you for asking! How can I help you?"""

def build_conversation(messages: List[Message]) -> str:
    """Convert message list to conversation format."""
    parts = []
    for msg in messages:
        tag = "Human" if msg.role == "user" else "Assistant" if msg.role == "assistant" else "System"
        parts.append(f"### {tag}: {msg.content}")
    return "\n".join(parts)

def is_valid_function_call(text: str) -> bool:
    """Check if the text is a valid function call JSON."""
    if not text.startswith("{"):
        return False
    try:
        data = json.loads(text)
        # Must have exactly function and arguments keys
        if set(data.keys()) != {"function", "arguments"}:
            return False
        # Function must exist
        if data["function"] not in FUNCTIONS:
            return False
        # Arguments must be a dict
        if not isinstance(data["arguments"], dict):
            return False
        return True
    except json.JSONDecodeError:
        return False

def execute_function_call(text: str) -> str:
    """Execute a validated function call and return its result."""
    data = json.loads(text)
    fn_name = data["function"]
    fn_info = FUNCTIONS[fn_name]
    
    # Execute the function with provided arguments
    return fn_info["fn"](**data["arguments"])

# ── 4. Chat endpoint ────────────────────────────────────────────────────────
@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat(req: ChatCompletionRequest):
    # Initialize conversation with system message
    messages = [Message(role="system", content=build_system_message())]
    messages.extend(req.messages)
    
    # Get initial response from model
    response = llm(
        build_conversation(messages) + "\n### Assistant:",
        max_tokens=req.max_tokens,
        temperature=req.temperature,
        stop=["###"]
    )["choices"][0]["text"].strip()
    function_call = None
    # Check if it's a valid function call
    
    if is_valid_function_call(response):
        try:
            function_call = json.loads(response)
            print("")
            print("----------------MODEL CALLED FUNCTION----------------")
            print(response)
            print("-----------------------------------------------------")
            print("")
            # Execute function and get result
            result = execute_function_call(response)
            
            # Add function call and result to conversation
            messages.extend([
                Message(role="assistant", content=response),
                Message(role="function", content=result)
            ])
            
            # Get final response incorporating the function result
            response = llm(
                build_conversation(messages) + "\n### Assistant:",
                max_tokens=req.max_tokens,
                temperature=req.temperature,
                stop=["###"]
            )["choices"][0]["text"].strip()
        except Exception as e:
            response = f"Error executing function: {str(e)}"

    return ChatCompletionResponse(
        id=f"chatcmpl-{uuid.uuid4().hex[:8]}",
        choices=[Choice(
            message=Message(role="assistant", content=response),
            index=0,
            finish_reason="stop",
            function_call=function_call
        )]
    )
