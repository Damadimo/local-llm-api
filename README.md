# LLM API Server

A production-ready local language model API server that replicates OpenAI's API interface while running entirely on consumer hardware. Built with FastAPI, llama.cpp, and Qdrant for high-performance inference, function calling, and retrieval-augmented generation (RAG).

## Overview

This project provides a complete local alternative to cloud-based LLM APIs, featuring:

- **OpenAI-compatible API**: Drop-in replacement for OpenAI's chat completions and embeddings endpoints
- **Function Calling**: Extensible system for tool integration with automatic schema generation
- **RAG Pipeline**: Semantic search and context augmentation using vector embeddings
- **Production Ready**: Docker deployment, health monitoring, and comprehensive testing
- **Hardware Optimized**: Quantized models for efficient CPU inference on consumer hardware

Perfect for developers who need AI capabilities without cloud dependencies, API costs, or data privacy concerns.

## Features

### Core Capabilities
- **Chat Completions** - Conversational AI with system prompts and context management
- **Function Calling** - Structured tool integration with automatic JSON schema validation
- **Embeddings Generation** - High-quality vector representations using Jina AI models
- **RAG System** - Document indexing, semantic search, and context-aware responses
- **Streaming Responses** - Real-time token streaming for better user experience

### Technical Implementation
- **Quantized Models** - GGUF format for 4-bit quantization (3.8GB vs 13GB)
- **Vector Database** - Qdrant for high-performance similarity search
- **FastAPI Framework** - Async endpoints with automatic OpenAPI documentation
- **Docker Deployment** - Full-stack containerization with service orchestration
- **Comprehensive Testing** - Automated test suite covering all functionality

### Built-in Functions
- **Time & Date** - Current time, timezone conversion, date calculations
- **Weather Data** - Real-time weather information and forecasts
- **Extensible Registry** - Easy addition of custom functions with type safety

## Quick Start

### Docker Deployment (Recommended)

```bash
# Clone the repository
git clone https://github.com/yourusername/llm-api.git
cd llm-api

# Start the full stack
docker-compose up --build

# API will be available at http://localhost:8000
```

The Docker setup automatically:
- Downloads the Llama 2 7B Chat model (3.8GB)
- Starts Qdrant vector database
- Configures service networking
- Sets up health monitoring

### Local Development

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -e .

# Start the server
python -m llm_api.main
```

## API Usage Examples

### Chat Completion

```python
import requests

response = requests.post("http://localhost:8000/v1/chat/completions", json={
    "model": "llama-2-7b-chat",
    "messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain quantum computing in simple terms."}
    ],
    "max_tokens": 256,
    "temperature": 0.7
})

print(response.json()["choices"][0]["message"]["content"])
```

### Function Calling

```python
response = requests.post("http://localhost:8000/v1/chat/completions", json={
    "model": "llama-2-7b-chat",
    "messages": [
        {"role": "user", "content": "What's the weather like in San Francisco?"}
    ],
    "functions": [
        {
            "name": "get_weather",
            "description": "Get current weather for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "City name"}
                },
                "required": ["location"]
            }
        }
    ],
    "function_call": "auto"
})
```

### Embeddings Generation

```python
response = requests.post("http://localhost:8000/v1/embeddings", json={
    "model": "jina-embeddings-v3",
    "input": ["Hello world", "Machine learning is fascinating"]
})

embeddings = [item["embedding"] for item in response.json()["data"]]
print(f"Generated {len(embeddings)} embeddings of dimension {len(embeddings[0])}")
```

### RAG Document Management

```python
# Add documents to knowledge base
requests.post("http://localhost:8000/v1/rag/documents", json={
    "texts": [
        "Docker is a containerization platform that packages applications.",
        "Kubernetes orchestrates containerized applications at scale.",
        "FastAPI is a modern Python web framework for building APIs."
    ],
    "metadata": [
        {"source": "docker-docs", "type": "definition"},
        {"source": "k8s-docs", "type": "definition"},
        {"source": "fastapi-docs", "type": "definition"}
    ]
})

# Query with context
response = requests.post("http://localhost:8000/v1/rag/chat", json={
    "query": "How do Docker and Kubernetes work together?",
    "max_tokens": 256,
    "num_context_docs": 3
})

print(response.json()["answer"])
```

### cURL Examples

```bash
# Health check
curl http://localhost:8000/health

# Simple chat
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama-2-7b-chat",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 50
  }'

# List available functions
curl http://localhost:8000/v1/functions
```

## Architecture

### System Design

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Client App    │───▶│   FastAPI       │───▶│   llama.cpp     │
│                 │    │   Server        │    │   Engine        │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │                       │
                                ▼                       ▼
                       ┌─────────────────┐    ┌─────────────────┐
                       │   Qdrant        │    │   Function      │
                       │   Vector DB     │    │   Registry      │
                       └─────────────────┘    └─────────────────┘
```

### Component Breakdown

**FastAPI Server** (`llm_api/main.py`)
- OpenAI-compatible REST endpoints
- Request validation and response formatting
- Async request handling
- Automatic OpenAPI documentation

**LLM Handler** (`llm_api/core/llm/`)
- Model loading and inference management
- Context window optimization (2048 tokens)
- Token counting and usage tracking
- Streaming response generation

**Function Registry** (`llm_api/core/functions/`)
- Dynamic function discovery
- JSON schema generation
- Type-safe parameter validation
- Extensible plugin architecture

**RAG Pipeline** (`llm_api/core/rag/`)
- Document chunking and embedding
- Vector similarity search
- Context ranking and selection
- Response augmentation

**Vector Store** (`llm_api/core/rag/vector_store.py`)
- Qdrant client management
- Embedding storage and retrieval
- Similarity search optimization
- Collection management

### Data Flow

1. **Chat Request** → FastAPI validates → Function detection → LLM inference → Response
2. **RAG Query** → Embed query → Vector search → Rank results → Context injection → LLM response
3. **Function Call** → Parse parameters → Execute function → Format result → Continue conversation

## Performance & Limitations

### Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| CPU | 4 cores | 8+ cores |
| RAM | 8GB | 16GB+ |
| Storage | 10GB | 50GB+ |
| Network | - | Stable internet (model download) |

### Performance Metrics

**Inference Speed** (CPU-only, Llama 2 7B)
- Token generation: 5-15 tokens/second
- Context processing: 15-20 tokens/second  
- Cold start: ~30 seconds (model loading)
- Warm requests: <1 second overhead

**RAG Performance**
- Document embedding: ~200ms per document
- Vector search: <100ms for 10k documents
- Context retrieval: 3 documents in ~200ms
- End-to-end RAG query: 2-5 seconds

**Memory Usage**
- Base model: ~4GB RAM
- Vector database: ~100MB per 10k documents
- API server: ~200MB
- Total system: 6-8GB RAM

### Current Limitations

- **Model Size**: Limited to models that fit in available RAM
- **Context Window**: 2048 tokens (expandable to 4096 with more RAM)
- **Inference Speed**: CPU-bound; GPU acceleration possible with CUDA builds
- **Function Calling**: Currently supports synchronous functions only
- **RAG Scale**: Optimized for <100k documents; larger collections need optimization

### Scaling Considerations

- **Horizontal**: Multiple API instances behind load balancer
- **Vertical**: GPU acceleration for 10x speed improvement
- **Storage**: Qdrant clustering for large document collections
- **Caching**: Redis for response caching and session management

## Installation & Configuration

### Environment Variables

```bash
# Model Configuration
MODEL_REPO_ID=TheBloke/Llama-2-7B-Chat-GGUF
MODEL_FILENAME=llama-2-7b-chat.Q4_K_M.gguf
CONTEXT_LENGTH=2048
MAX_TOKENS=256
TEMPERATURE=0.2

# Vector Database
QDRANT_HOST=localhost
QDRANT_PORT=6333
QDRANT_USE_MEMORY=true  # false for Docker deployment

# Embeddings
EMBEDDING_MODEL_NAME=jinaai/jina-embeddings-v3
EMBEDDING_DIMENSION=1024

# RAG Settings
RAG_TOP_K=3
RAG_SCORE_THRESHOLD=0.7

# Server
HOST=127.0.0.1
PORT=8000
LOG_LEVEL=INFO
```

### Custom Model Usage

```bash
# Use a different model
export MODEL_REPO_ID="microsoft/DialoGPT-medium"
export MODEL_FILENAME="pytorch_model.bin"

# Or specify direct path
export MODEL_PATH="/path/to/your/model.gguf"
```

### Adding Custom Functions

```python
# llm_api/core/functions/builtin/my_function.py
from typing import Dict, Any
from llm_api.core.functions.registry import function_registry

@function_registry.register
def calculate_tip(bill_amount: float, tip_percentage: float = 15.0) -> Dict[str, Any]:
    """Calculate tip amount and total bill."""
    tip = bill_amount * (tip_percentage / 100)
    total = bill_amount + tip
    
    return {
        "tip_amount": round(tip, 2),
        "total_amount": round(total, 2),
        "tip_percentage": tip_percentage
    }
```

## Testing

### Automated Test Suite

```bash
# Run all tests
python test_docker_deployment.py

# Test local deployment
python test_docker_deployment.py --local

# Test specific components
python -m pytest tests/ -v
```

### Manual Testing

```bash
# Test API health
curl http://localhost:8000/health

# Test chat completion
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "llama-2-7b-chat", "messages": [{"role": "user", "content": "Hello"}]}'

# Test RAG system
curl -X POST http://localhost:8000/v1/rag/documents \
  -H "Content-Type: application/json" \
  -d '{"texts": ["Test document content"]}'
```

## Development

### Project Structure

```
llm_api/
├── api/
│   └── models/          # Pydantic request/response models
├── core/
│   ├── llm/            # Language model handlers
│   ├── functions/      # Function calling system
│   ├── rag/           # RAG pipeline components
│   └── utils/         # Shared utilities
├── config/
│   └── settings.py    # Configuration management
└── main.py           # FastAPI application
```

### Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make changes with tests
4. Run the test suite: `python test_docker_deployment.py --local`
5. Submit a pull request

### Code Quality

```bash
# Format code
black llm_api/
isort llm_api/

# Type checking
mypy llm_api/

# Run tests
pytest tests/
```

## Credits & Acknowledgments

### Core Technologies
- **[llama.cpp](https://github.com/ggerganov/llama.cpp)** - High-performance LLM inference engine
- **[FastAPI](https://fastapi.tiangolo.com/)** - Modern Python web framework
- **[Qdrant](https://qdrant.tech/)** - Vector similarity search engine
- **[Pydantic](https://docs.pydantic.dev/)** - Data validation and settings management

### Models & Embeddings
- **[Llama 2](https://ai.meta.com/llama/)** by Meta - Base language model
- **[TheBloke](https://huggingface.co/TheBloke)** - GGUF model quantization and distribution
- **[Jina AI](https://jina.ai/)** - High-quality embedding models

### Ecosystem
- **[Hugging Face](https://huggingface.co/)** - Model hosting and transformers library
- **[Docker](https://www.docker.com/)** - Containerization platform
- **[uvicorn](https://www.uvicorn.org/)** - ASGI server implementation

This project demonstrates the integration of multiple AI technologies into a cohesive, production-ready system that can serve as a foundation for AI-powered applications without cloud dependencies.

## License

MIT License - see [LICENSE](LICENSE) file for details. 