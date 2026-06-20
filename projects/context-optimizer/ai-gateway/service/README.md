# Context Optimizer AI Gateway - Docker Service

Production-ready AI gateway with semantic compression and LiteLLM integration.

## 🚀 Quick Start

### 1. Configure API Keys

Create `.env` file:
```bash
# OpenAI
OPENAI_API_KEY=sk-...

# Anthropic
ANTHROPIC_API_KEY=sk-ant-...

# Groq
GROQ_API_KEY=gsk_...

# Azure OpenAI (optional)
AZURE_API_KEY=...
AZURE_API_BASE=https://your-resource.openai.azure.com/
```

### 2. Build & Deploy

```bash
cd ai-gateway/service

# Build image
docker compose build

# Start gateway + Redis cache
docker compose up -d

# With local Ollama (optional)
docker compose --profile local-llm up -d
```

### 3. Test Endpoint

```bash
# Health check
curl http://localhost:8080/health

# Chat completion
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -d '{
    "model": "gpt-4",
    "messages": [
      {"role": "system", "content": "Large documentation context (50KB)..."},
      {"role": "user", "content": "Summarize the main points"}
    ]
  }'
```

## 📖 API Documentation

Visit http://localhost:8080/docs for interactive API documentation.

## 🎯 Endpoints

### Chat Completions (OpenAI-Compatible)
```
POST /v1/chat/completions
```

**Request:**
```json
{
  "model": "gpt-4",
  "messages": [
    {"role": "system", "content": "Context..."},
    {"role": "user", "content": "Query..."}
  ],
  "temperature": 0.7,
  "max_tokens": 1000,
  "enable_compression": true,
  "compression_threshold": 2000
}
```

**Response:**
```json
{
  "id": "chatcmpl-123",
  "object": "chat.completion",
  "created": 1234567890,
  "model": "gpt-4",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "Response..."
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 250,
    "completion_tokens": 100,
    "total_tokens": 350
  },
  "compression_stats": {
    "enabled": true,
    "latency_ms": 245,
    "cache_hit": false
  }
}
```

### Health & Stats
```
GET /health          # Service health
GET /stats           # Compression & cost stats
GET /v1/models       # List available models
```

### Admin
```
POST /admin/cache/clear   # Clear semantic cache
GET  /admin/config        # Get configuration
```

## 🏗️ Architecture

```
Client (OpenAI SDK compatible)
   ↓
AI Gateway (Port 8080)
   ├─ Compression Layer
   ├─ Semantic Cache (Redis)
   └─ LiteLLM Router
       ├─ OpenAI
       ├─ Anthropic
       ├─ Groq
       ├─ Azure
       ├─ Ollama (local)
       └─ 95+ more providers
```

## 🔧 Configuration

### Environment Variables

```bash
# API Keys
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GROQ_API_KEY=gsk_...

# Compression
COMPRESSION_THRESHOLD=2000  # Tokens before compression

# Cache
REDIS_URL=redis://redis:6379

# Ollama (local)
OLLAMA_BASE_URL=http://ollama:11434

# Logging
LOG_LEVEL=INFO
```

### Docker Compose Profiles

```bash
# Default: Gateway + Redis
docker compose up

# With local Ollama
docker compose --profile local-llm up

# Production (no local LLM)
docker compose up gateway redis
```

## 📊 Performance

**Compression:**
- Token reduction: 50-98%
- Latency overhead: 200-500ms
- Cache hit latency: <10ms

**Scaling:**
- Horizontal: Add more gateway replicas
- Vertical: Increase Redis memory
- Load balancing: Use nginx/traefik

## 🔒 Security

### API Key Validation

```python
# Add to gateway_service.py
@app.post("/v1/chat/completions")
async def create_chat_completion(
    authorization: str = Header(...),
):
    # Validate API key
    if not is_valid_key(authorization):
        raise HTTPException(401, "Invalid API key")
```

### Rate Limiting

```bash
# Add Redis-based rate limiting
docker compose exec gateway pip install slowapi
```

### HTTPS

```yaml
# Use nginx proxy
services:
  nginx:
    image: nginx:alpine
    ports:
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
```

## 📈 Monitoring

### Prometheus Metrics

```python
# Add prometheus-client
from prometheus_client import Counter, Histogram

compression_requests = Counter("compression_requests_total", "Total compressions")
compression_latency = Histogram("compression_latency_seconds", "Compression latency")
```

### Logs

```bash
# View logs
docker compose logs -f gateway

# Export to file
docker compose logs gateway > gateway.log
```

## 🧪 Testing

### Smoke Test

```bash
# Test health
curl http://localhost:8080/health

# Test compression
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-3.5-turbo",
    "messages": [
      {"role": "user", "content": "Hello!"}
    ]
  }'
```

### Load Test

```bash
# Install hey
go install github.com/rakyll/hey@latest

# Run load test
hey -n 100 -c 10 \
  -m POST \
  -H "Content-Type: application/json" \
  -d '{"model":"gpt-3.5-turbo","messages":[{"role":"user","content":"test"}]}' \
  http://localhost:8080/v1/chat/completions
```

## 🐛 Troubleshooting

### Service Won't Start

```bash
# Check logs
docker compose logs gateway

# Check port availability
netstat -an | grep 8080

# Rebuild image
docker compose build --no-cache
```

### Compression Not Working

```bash
# Check logs for compression events
docker compose logs gateway | grep Compression

# Verify threshold
curl http://localhost:8080/admin/config
```

### Redis Connection Failed

```bash
# Check Redis
docker compose exec redis redis-cli ping

# Check network
docker network inspect gateway-network
```

## 📦 Production Deployment

### AWS ECS

```bash
# Push to ECR
docker tag context-optimizer-gateway:latest $ECR_REPO/gateway:latest
docker push $ECR_REPO/gateway:latest

# Deploy task definition
aws ecs create-service --cluster prod --service gateway ...
```

### Kubernetes

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ai-gateway
spec:
  replicas: 3
  template:
    spec:
      containers:
      - name: gateway
        image: context-optimizer-gateway:latest
        ports:
        - containerPort: 8080
        env:
        - name: OPENAI_API_KEY
          valueFrom:
            secretKeyRef:
              name: api-keys
              key: openai
```

## 📚 Related Documentation

- [Python Package](../wrapper/README.md) - Pip-installable wrapper
- [Core Compression](../../src/context_optimizer/) - Low-level APIs
- [Main README](../../README.md) - Project overview

## 📄 License

MIT License
