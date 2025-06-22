"""
Chat completion handler for LLM inference.
"""
import json
import logging
import time
import uuid
from typing import List, AsyncGenerator, Dict, Any

from llama_cpp import Llama

from llm_api.config.settings import settings
from llm_api.api.models.chat import (
    Message,
    ChatCompletionRequest,
    ChatCompletionResponse,
    Choice,
    Usage,
    FunctionCall,
    Delta,
    ChatCompletionStreamResponse
)
from llm_api.core.utils.conversation import (
    build_system_message,
    build_conversation,
    is_valid_function_call,
    execute_function_call
)
from llm_api.core.utils.token_counter import token_counter

logger = logging.getLogger(__name__)


class ChatHandler:
    """Handles chat completions using the LLM."""
    
    def __init__(self):
        """Initialize the chat handler."""
        self.llm = None
        self._load_model()
    
    def _load_model(self) -> None:
        """Load the LLM model."""
        logger.info("Loading LLM model...")
        
        try:
            if settings.model_path:
                # Use direct path if provided
                model_path = settings.model_path
            else:
                # Use repo_id and filename
                model_path = None
                
            if model_path:
                self.llm = Llama(
                    model_path=str(model_path),
                    n_ctx=settings.context_length,
                    verbose=False
                )
            else:
                self.llm = Llama.from_pretrained(
                    repo_id=settings.model_repo_id,
                    filename=settings.model_filename,
                    n_ctx=settings.context_length,
                    verbose=False
                )
            
            # Update token counter with the model's tokenizer
            token_counter.tokenizer = self.llm
            
            logger.info(f"LLM model loaded successfully with context length: {settings.context_length}")
            
        except Exception as e:
            logger.error(f"Failed to load LLM model: {e}")
            raise
    
    async def create_completion(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        """
        Create a chat completion response.
        
        Args:
            request: Chat completion request
            
        Returns:
            Chat completion response
        """
        logger.info(f"Creating chat completion for {len(request.messages)} messages")
        
        # Build messages with system prompt
        messages = [Message(role="system", content=build_system_message())]
        messages.extend(request.messages)
        
        # Generate initial response
        conversation_text = build_conversation(messages) + "\n### Assistant:"
        
        response = self.llm(
            conversation_text,
            max_tokens=request.max_tokens or settings.max_tokens,
            temperature=request.temperature or settings.temperature,
            stop=["###"]
        )["choices"][0]["text"].strip()

        function_call = None
        
        # Check if response is a function call
        if is_valid_function_call(response):
            try:
                data = json.loads(response)
                function_call = FunctionCall(
                    name=data["function"],
                    arguments=data["arguments"]
                )
                
                # Execute the function
                result = execute_function_call(response)
                
                # Add function call and result to conversation
                messages.extend([
                    Message(role="assistant", content=response),
                    Message(role="function", content=result)
                ])
                
                # Generate final response
                final_conversation = build_conversation(messages) + "\n### Assistant:"
                response = self.llm(
                    final_conversation,
                    max_tokens=request.max_tokens or settings.max_tokens,
                    temperature=request.temperature or settings.temperature,
                    stop=["###"]
                )["choices"][0]["text"].strip()
                
            except Exception as e:
                logger.error(f"Error executing function: {e}")
                response = f"Error executing function: {str(e)}"

        # Calculate token usage
        prompt_tokens = token_counter.count_tokens_in_messages(messages)
        completion_tokens = token_counter.estimate_response_tokens(response)
        total_tokens = prompt_tokens + completion_tokens

        # Create response
        completion_response = ChatCompletionResponse(
            id=f"chatcmpl-{uuid.uuid4().hex[:8]}",
            created=int(time.time()),
            model=request.model,
            choices=[Choice(
                message=Message(role="assistant", content=response),
                index=0,
                finish_reason="function_call" if function_call else "stop"
            )],
            usage=Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens
            )
        )
        
        # Add function call to response if it occurred
        if function_call:
            # Update the message to include the function call
            completion_response.choices[0].message.function_call = function_call
            
        return completion_response
    
    async def create_stream_completion(self, request: ChatCompletionRequest) -> AsyncGenerator[str, None]:
        """
        Create a streaming chat completion response.
        
        Args:
            request: Chat completion request
            
        Yields:
            SSE-formatted response chunks
        """
        logger.info(f"Creating streaming chat completion for {len(request.messages)} messages")
        
        # Build messages with system prompt
        messages = [Message(role="system", content=build_system_message())]
        messages.extend(request.messages)
        
        chat_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
        
        buffer = ""
        collecting_fn = False
        fn_text = ""

        # First response generation
        try:
            conversation_text = build_conversation(messages) + "\n### Assistant:"
            
            generator = self.llm(
                conversation_text,
                max_tokens=request.max_tokens or settings.max_tokens,
                temperature=request.temperature or settings.temperature,
                stop=["###"],
                stream=True,
                echo=False
            )
            
            for chunk in generator:
                if not chunk or "choices" not in chunk or not chunk["choices"]:
                    continue
                    
                tok = chunk["choices"][0].get("text", "")
                if not tok:
                    continue

                if not collecting_fn:
                    if "{" in tok:
                        collecting_fn = True
                        fn_text = tok[tok.index("{"):]
                        head = tok[:tok.index("{")]
                        if head.strip():
                            yield f"data: {json.dumps({'id': chat_id, 'object': 'chat.completion.chunk', 'choices': [{'delta': {'content': head}, 'index': 0}]})}\n\n"
                    else:
                        if tok.strip():
                            yield f"data: {json.dumps({'id': chat_id, 'object': 'chat.completion.chunk', 'choices': [{'delta': {'content': tok}, 'index': 0}]})}\n\n"
                else:
                    fn_text += tok
                    if fn_text.strip().endswith("}"):
                        if is_valid_function_call(fn_text):
                            try:
                                data = json.loads(fn_text)
                                fc = FunctionCall(name=data["function"], arguments=data["arguments"])
                                
                                # Emit function call
                                yield f"data: {json.dumps({'id': chat_id, 'object': 'chat.completion.chunk', 'choices': [{'delta': {'function_call': fc.model_dump()}, 'index': 0}]})}\n\n"
                                
                                # Execute function
                                generator.close()
                                result = execute_function_call(fn_text)
                                
                                # Generate response with function result
                                followup_msgs = messages + [
                                    Message(role="assistant", content=fn_text),
                                    Message(role="function", content=result)
                                ]
                                
                                # Stream the final response
                                followup_conversation = build_conversation(followup_msgs) + "\n### Assistant:"
                                followup_gen = self.llm(
                                    followup_conversation,
                                    max_tokens=request.max_tokens or settings.max_tokens,
                                    temperature=request.temperature or settings.temperature,
                                    stream=True,
                                    stop=["###"],
                                    echo=False
                                )
                                
                                for sub in followup_gen:
                                    if not sub or "choices" not in sub or not sub["choices"]:
                                        continue
                                    text = sub["choices"][0].get("text", "")
                                    if text and text.strip():
                                        yield f"data: {json.dumps({'id': chat_id, 'object': 'chat.completion.chunk', 'choices': [{'delta': {'content': text}, 'index': 0}]})}\n\n"
                                        
                            except Exception as e:
                                yield f"data: {json.dumps({'id': chat_id, 'object': 'chat.completion.chunk', 'choices': [{'delta': {'content': f'Error: {str(e)}'}, 'index': 0}]})}\n\n"
                        collecting_fn = False
                        fn_text = ""
                    
        except Exception as e:
            logger.error(f"Error in streaming completion: {e}")
            yield f"data: {json.dumps({'id': chat_id, 'object': 'chat.completion.chunk', 'choices': [{'delta': {'content': f'Error: {str(e)}'}, 'index': 0}]})}\n\n"
        
        finally:
            yield f"data: {json.dumps({'id': chat_id, 'object': 'chat.completion.chunk', 'choices': [{'delta': {}, 'finish_reason': 'stop', 'index': 0}]})}\n\n"
            yield "data: [DONE]\n\n"


# Global chat handler instance
chat_handler = ChatHandler() 