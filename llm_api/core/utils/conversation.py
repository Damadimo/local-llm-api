"""
Conversation handling utilities.
"""
import json
import logging
from typing import List

from llm_api.api.models.chat import Message
from llm_api.core.functions.registry import function_registry

logger = logging.getLogger(__name__)


def build_system_message() -> str:
    """Create clear instructions for function calling behavior."""
    # Build function list with parameters
    function_list = []
    # Build examples for each function
    examples = []
    
    function_specs = function_registry.get_function_specs()
    
    for spec in function_specs:
        name = spec["name"]
        params = spec["parameters"]
        
        # Function list entry
        param_names = ", ".join(params.get("properties", {}).keys())
        param_str = f"({param_names})" if param_names else "(no parameters)"
        function_list.append(f"- {name}{param_str}")
        
        # Example entry
        example_args = {}
        for param_name in params.get("properties", {}).keys():
            example_args[param_name] = "London" if "location" in param_name else "default"
        
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
        if data["function"] not in function_registry.list_functions():
            return False
        # Arguments must be a dict
        if not isinstance(data["arguments"], dict):
            return False
        return True
    except json.JSONDecodeError:
        return False


def execute_function_call(text: str) -> str:
    """Execute a validated function call and return its result."""
    try:
        data = json.loads(text)
        fn_name = data["function"]
        arguments = data["arguments"]
        
        # Execute the function using the registry
        result = function_registry.call_function(fn_name, arguments)
        logger.info(f"Function call executed: {fn_name}({arguments}) -> {result}")
        return result
        
    except Exception as e:
        error_msg = f"Error executing function call: {str(e)}"
        logger.error(error_msg)
        return error_msg 