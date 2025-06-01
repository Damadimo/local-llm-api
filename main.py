# main.py ────────────────────────────────────────────────────────────────────
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from llama_cpp import Llama
import json, uuid


from models import * #Import all models from models.py
from functions import * #Import all functions available to model from functions.py
from utilities import * #Import all utilities from utilities.py
from embedder import * #Import all utilities from embedder.py

# ── 0. FastAPI & model ──────────────────────────────────────────────────────
app = FastAPI(title="Llama-2 Function-Calling API", version="2.0.0")

llm = Llama.from_pretrained(
    repo_id="TheBloke/Llama-2-7B-Chat-GGUF",
    filename="llama-2-7b-chat.Q4_K_M.gguf",
)

# ── API Endpoints ──────────────────────────────────────────────────────


# Chat Completion Endpoint
@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_completion(req: ChatCompletionRequest):
    """Regular chat completion endpoint - returns complete response."""
    messages = [Message(role="system", content=build_system_message())]
    messages.extend(req.messages)
    
    response = llm(
        build_conversation(messages) + "\n### Assistant:",
        max_tokens=req.max_tokens,
        temperature=req.temperature,
        stop=["###"]
    )["choices"][0]["text"].strip()

    function_call = None
    if is_valid_function_call(response):
        try:
            data = json.loads(response)
            function_call = FunctionCall(
                name=data["function"],
                arguments=data["arguments"]
            )
            result = execute_function_call(response)
            
            messages.extend([
                Message(role="assistant", content=response),
                Message(role="function", content=result)
            ])
            
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
            finish_reason="function_call" if function_call else "stop",
            function_call=function_call
        )]
    )


# Streaming Chat Completion Endpoint
@app.post("/v1/chat/completions/stream")
async def stream_chat_completion(req: ChatCompletionRequest):
    """Streaming chat completion endpoint - returns chunks of the response."""
    messages = [Message(role="system", content=build_system_message())]
    messages.extend(req.messages)
    
    chat_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
    
    async def event_generator():
        buffer = ""
        collecting_fn = False
        fn_text = ""

        # First response generation
        try:
            generator = llm(
                build_conversation(messages) + "\n### Assistant:",
                max_tokens=req.max_tokens or 256,  # Ensure we have a max_tokens value
                temperature=req.temperature,
                stop=["###"],
                stream=True,
                echo=False  # Explicitly disable echo to prevent logit issues
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
                                yield f"data: {json.dumps({'id': chat_id, 'object': 'chat.completion.chunk', 'choices': [{'delta': {'function_call': fc.dict()}, 'index': 0}]})}\n\n"
                                
                                # Execute function
                                generator.close()
                                result = execute_function_call(fn_text)
                                
                                # Generate response with function result
                                followup_msgs = messages + [
                                    Message(role="assistant", content=fn_text),
                                    Message(role="function", content=result)
                                ]
                                
                                # Stream the final response with safe parameters
                                followup_gen = llm(
                                    build_conversation(followup_msgs) + "\n### Assistant:",
                                    max_tokens=req.max_tokens or 256,
                                    temperature=req.temperature,
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
            yield f"data: {json.dumps({'id': chat_id, 'object': 'chat.completion.chunk', 'choices': [{'delta': {'content': f'Error: {str(e)}'}, 'index': 0}]})}\n\n"
        
        finally:
            yield f"data: {json.dumps({'id': chat_id, 'object': 'chat.completion.chunk', 'choices': [{'delta': {}, 'finish_reason': 'stop', 'index': 0}]})}\n\n"
            yield "data: [DONE]\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


# Embedding Endpoint
@app.post("/v1/embeddings")
async def embeddings(req: EmbeddingsRequest):
    embeddings = embed(req.texts)

    return EmbeddingsResponse(embeddings=embeddings, model = EMBED_MODEL_NAME)

