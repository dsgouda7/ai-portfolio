# Playground

Exploration notebooks for learning and experimentation.

## What's Here

All notebooks:
- Use config files (no hardcoded credentials)
- Have proper .gitignore for API keys and vector DBs
- Download data inline (self-contained)
- Link back to theory in [notes/](../notes/)
- Include setup scripts

---

## Projects

### [AI Agents](ai-agents/)
7-notebook progression: LLM basics → RAG → Agentic AI.

Topics:
- LLM basics (OpenAI, Gemini, Groq)
- Structured output (Pydantic)
- RAG (fundamentals + advanced)
- Agentic workflows
- Agent memory
- LangGraph

See also: [notes/03-ai/](../notes/03-ai/), [notes/05-agentic-ai/](../notes/05-agentic-ai/)

---

### [ML Feature Engineering](ml-features/)
6 notebooks on classical ML feature engineering.

Topics:
- Feature engineering basics
- Categorical encoding
- Temporal features
- Text features (TF-IDF)
- Text embeddings
- Feature selection

See also: [notes/01-ml/](../notes/01-ml/)

---

### [Azure AI Exploration](azure-exploration/)
Experimenting with Azure OpenAI, AI Search, and AI Foundry.

Topics:
- Azure OpenAI auth + chat
- Embeddings
- Vector search (Azure AI Search)
- RAG workflows
- Azure AI Foundry agents

See also: [notes/07-ai-infrastructure/](../notes/07-ai-infrastructure/), [notes/03-ai/ch07-rag-and-embeddings/](../notes/03-ai/ch07-rag-and-embeddings/)

---

### [Chatbots](chatbots/)
Conversational AI experiments.

Projects:
- `customer-service-bot/` - Customer service chatbot (Agile Fever use case)

See also: [notes/03-ai/](../notes/03-ai/), [notes/05-agentic-ai/](../notes/05-agentic-ai/)

## Setup

Each subdirectory has setup scripts:
- `setup.ps1` (Windows)
- `setup.sh` (Linux/macOS)

Run the script to create venv and install dependencies.
