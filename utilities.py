from typing import List
import json
from models import *
from functions import *

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


