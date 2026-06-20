# Context Optimizer AI Gateway

LiteLLM-powered AI gateway with built-in semantic compression.

## 📁 Structure

```
ai-gateway/
├── wrapper/              # Pip-installable package
│   └── context_optimizer_gateway/
├── service/             # Deployable Docker container
│   ├── Dockerfile
│   └── docker-compose.yml
└── README.md (this file)
```

## 🎯 Two Deployment Options

### Option 1: Python Package (Wrapper)
```bash
cd wrapper
pip install -e .

# Use in your code
from context_optimizer_gateway import CompressedLiteLLM
```

### Option 2: Docker Service (Gateway)
```bash
cd service
docker compose up
# Access at http://localhost:8080
```

## 🔑 Key Features

- ✅ **Transparent Compression**: 50-98% token reduction
- ✅ **100+ LLM Providers**: OpenAI, Anthropic, Groq, Azure, Bedrock, etc.
- ✅ **Semantic Caching**: Redis-backed, cross-user savings
- ✅ **Cost Analytics**: Real-time cost + savings tracking
- ✅ **Smart Routing**: Cost-based, quality-aware provider selection
- ✅ **Drop-in Replacement**: Compatible with OpenAI SDK

## 📊 Value Proposition

**Without Compression:**
```python
response = openai.ChatCompletion.create(
    model="gpt-4",
    messages=[{"role": "system", "content": "50KB context"}]
)
# Cost: $1.50 per request
```

**With Gateway:**
```python
response = openai.ChatCompletion.create(
    model="gpt-4-compressed",
    messages=[{"role": "system", "content": "50KB → 1KB"}]
)
# Cost: $0.03 per request (50x savings)
```

## 🚀 Quick Start

See subdirectories for specific deployment guides:
- [wrapper/README.md](wrapper/README.md) - Python package
- [service/README.md](service/README.md) - Docker service
