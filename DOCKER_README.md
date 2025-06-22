# Docker Deployment Guide

This guide explains how to run the LLM API server using Docker and Docker Compose.

## 🐳 Quick Start

### Prerequisites

- Docker Engine 20.10+
- Docker Compose V2
- At least 8GB RAM (for model loading)
- 10GB+ free disk space

### 1. Launch the Full Stack

```bash
# Clone the repository (if not already done)
git clone <your-repo-url>
cd LLM-API

# Create necessary directories
mkdir -p data/models data/knowledge

# Start both API server and Qdrant database
docker-compose up --build
```

This single command will:
- Build the API server image
- Pull the Qdrant vector database image
- Start both services with proper networking
- Mount volumes for persistent data

### 2. Verify the Deployment

```bash
# Check if services are running
docker-compose ps

# Test API health
curl http://localhost:8000/health

# Test Qdrant health
curl http://localhost:6333/health
```

### 3. Test RAG Functionality

```bash
# Run the Docker deployment test
python test_docker_deployment.py
```

## 📋 Service Configuration

### API Server (llm-api)
- **Port**: 8000
- **Context Window**: 2048 tokens (configurable via `CONTEXT_LENGTH`)
- **Model**: Downloads Llama 2 7B Chat automatically
- **Health Check**: `GET /health`

### Qdrant Vector Database (qdrant)
- **REST API Port**: 6333
- **gRPC Port**: 6334
- **Data Persistence**: `qdrant_data` volume
- **Health Check**: `GET /health` on port 6333

## 🔧 Environment Variables

Create a `.env` file to customize the deployment:

```bash
# Model Configuration
MODEL_REPO_ID=TheBloke/Llama-2-7B-Chat-GGUF
MODEL_FILENAME=llama-2-7b-chat.Q4_K_M.gguf
CONTEXT_LENGTH=2048
MAX_TOKENS=256
TEMPERATURE=0.2

# Qdrant Configuration
QDRANT_HOST=qdrant
QDRANT_PORT=6333
QDRANT_USE_MEMORY=false

# RAG Configuration
RAG_TOP_K=3
RAG_SCORE_THRESHOLD=0.7

# Development
DEBUG=false
LOG_LEVEL=INFO
```

## 🧪 Testing the Deployment

### Automated Testing

#### With Docker
Run the comprehensive test suite against Docker containers:

```bash
# Start Docker services first
docker-compose up --build

# Then run tests in another terminal
python test_docker_deployment.py
```

#### Without Docker (Local Testing)
Test the local development setup:

```bash
# Start the local server
python -m llm_api.main

# Run tests against local services
python test_docker_deployment.py --local
```

This will test:
- ✅ API server health and connectivity
- ✅ Model loading and chat completions
- ✅ Embedding generation
- ✅ Document indexing and retrieval
- ✅ End-to-end RAG functionality
- ✅ Vector database integration (in-memory mode for local, persistent for Docker)

### Manual Testing

1. **Basic Chat Completion**:
```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama-2-7b-chat",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 100
  }'
```

2. **Add Documents to Knowledge Base**:
```bash
curl -X POST http://localhost:8000/v1/rag/documents \
  -H "Content-Type: application/json" \
  -d '{
    "texts": ["Your company knowledge goes here..."]
  }'
```

3. **RAG Query**:
```bash
curl -X POST http://localhost:8000/v1/rag/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is our company policy?",
    "top_k": 3
  }'
```

4. **RAG Chat Completion**:
```bash
curl -X POST http://localhost:8000/v1/rag/chat \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is our company policy?",
    "max_tokens": 200
  }'
```

## 🐛 Troubleshooting

### Common Issues

1. **Port Already in Use**:
```bash
# Check what's using the ports
netstat -tulpn | grep :8000
netstat -tulpn | grep :6333

# Stop conflicting services
docker-compose down
```

2. **Out of Memory**:
```bash
# Check Docker memory limits
docker stats

# Increase Docker Desktop memory allocation to 8GB+
# Or use a smaller model by setting MODEL_FILENAME
```

3. **Model Download Fails**:
```bash
# Check logs
docker-compose logs llm-api

# Manually download model to data/models/
# Then set MODEL_PATH environment variable
```

4. **Qdrant Connection Issues**:
```bash
# Check Qdrant logs
docker-compose logs qdrant

# Verify Qdrant is accessible
curl http://localhost:6333/health
```

### Debug Mode

Enable debug logging:

```bash
# Add to .env file
DEBUG=true
LOG_LEVEL=DEBUG

# Restart services
docker-compose down
docker-compose up --build
```

## 🔄 Development Workflow

### Local Development with Docker

1. **Mount source code for live reloading**:
```yaml
# Add to docker-compose.yml under llm-api service
volumes:
  - ./llm_api:/app/llm_api
  - ./data:/app/data
environment:
  - RELOAD=true
```

2. **Use development configuration**:
```bash
# Copy development environment
cp env.example .env.development

# Use development compose file
docker-compose -f docker-compose.yml -f docker-compose.dev.yml up
```

### Building Custom Images

```bash
# Build API server image
docker build -t llm-api:latest .

# Build with specific model
docker build --build-arg MODEL_PATH=/models/custom-model.gguf -t llm-api:custom .

# Push to registry
docker tag llm-api:latest your-registry/llm-api:latest
docker push your-registry/llm-api:latest
```

## 📊 Performance Tuning

### Model Optimization

- **CPU-only**: Use Q4_K_M quantization (default)
- **GPU**: Use Q8_0 or F16 models with CUDA support
- **Memory-constrained**: Use Q2_K or Q3_K_S models

### Context Window

- **Default**: 2048 tokens (good balance)
- **Large documents**: 4096 tokens (requires more RAM)
- **Memory-constrained**: 1024 tokens

### Qdrant Optimization

- **Production**: Use persistent storage with SSD
- **Development**: Use in-memory mode for speed
- **Large datasets**: Increase `m` and `ef_construct` parameters

## 🚀 Production Deployment

### Security Considerations

1. **API Authentication**: Add API key validation
2. **Network Security**: Use reverse proxy (nginx/traefik)
3. **Data Privacy**: Encrypt volumes and use secrets management
4. **Resource Limits**: Set container memory/CPU limits

### Scaling

1. **Horizontal Scaling**: Multiple API replicas behind load balancer
2. **Vertical Scaling**: Increase container resources
3. **Model Sharding**: Distribute large models across containers
4. **Qdrant Clustering**: Multi-node Qdrant deployment

### Monitoring

```yaml
# Add to docker-compose.yml
services:
  prometheus:
    image: prom/prometheus
    ports:
      - "9090:9090"
  
  grafana:
    image: grafana/grafana
    ports:
      - "3000:3000"
```

## 📚 Additional Resources

- [Docker Compose Documentation](https://docs.docker.com/compose/)
- [Qdrant Documentation](https://qdrant.tech/documentation/)
- [llama-cpp-python Documentation](https://github.com/abetlen/llama-cpp-python)
- [Project README](README.md) for API usage examples 