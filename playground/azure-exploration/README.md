# Azure AI Exploration

A Jupyter notebook exploring Azure OpenAI, Azure AI Search, and Azure AI Foundry integration using LangChain for personal learning and experimentation.

## Overview

Self-contained notebook with:

- No hardcoded credentials (centralized config management via `config.txt`)
- Gitignored sensitive data (API keys, config files, vector databases)
- Self-contained execution (inline Wikipedia data downloads)
- Documented code (markdown explanations before every code block)
- Linked to theory in notes/
- Automated setup (venv creation via scripts)

## Notebooks

### 01. Azure AI Integration ([01-azure-ai-integration.ipynb](01-azure-ai-integration.ipynb))
**Topics:** Azure OpenAI, LangChain, embeddings, vector search, Azure AI Foundry agents
**Related Theory:** [AI Fundamentals](../../notes/03-ai/README.md), [RAG](../../notes/03-ai/ch04_rag_and_embeddings), [Vector Databases](../../notes/03-ai/ch05_vector_dbs)

Learn to:
- Authenticate with Azure OpenAI using LangChain
- Create chat completions with streaming and batch processing
- Generate embeddings using Azure OpenAI
- Fetch and chunk Wikipedia articles for famous soccer players
- Build and query a vector store with Azure AI Search
- Connect to Azure AI Foundry agents using device code authentication
- Implement semantic search with vector similarity scoring
- Combine retrieval with LLM generation for RAG workflows

## Prerequisites

- Azure subscription with:
  - Azure OpenAI resource (chat and embeddings deployments)
  - Azure AI Search service
  - Azure AI Foundry project (optional, for agent section)
- Python 3.8+
- Azure CLI (for authentication)

## Setup

### 1. Environment Setup

Run the setup script for your platform:

**Windows (PowerShell):**
```powershell
.\setup.ps1
```

**Linux/macOS:**
```bash
chmod +x setup.sh
./setup.sh
```

This will:
- Create a Python virtual environment (`.venv/`)
- Install all dependencies from `requirements.txt`
- Create `config.txt` from template (if it doesn't exist)

### 2. Azure Configuration

1. Copy the template: `config.txt.template` → `config.txt`
2. Fill in your Azure credentials in `config.txt`:
   - Azure OpenAI endpoint and API key
   - Azure AI Search service name and admin key
   - Azure AI Foundry endpoint (if using agents)
   - Azure tenant ID (if using device code authentication)

**Security Note:** `config.txt` is gitignored. Never commit credentials to version control.

### 3. Activate Virtual Environment

**Windows:**
```powershell
.\.venv\Scripts\Activate.ps1
```

**Linux/macOS:**
```bash
source .venv/bin/activate
```

### 4. Launch Jupyter

```bash
jupyter notebook
```

Open `01-azure-ai-integration.ipynb` and run cells sequentially.

## Project Structure

```
azure-ai-exploration/
├── 01-azure-ai-integration.ipynb   # Main exploration notebook
├── config.txt.template              # Azure credentials template
├── config.txt                       # Your actual config (gitignored)
├── requirements.txt                 # Python dependencies
├── setup.ps1                        # Windows setup script
├── setup.sh                         # Linux/macOS setup script
├── .gitignore                       # Git ignore rules
└── README.md                        # This file
```

## What's Gitignored

- `config.txt` (your Azure credentials)
- `.venv/` (virtual environment)
- Vector databases and cached data
- `.ipynb_checkpoints/`

## Azure Services Used

### Azure OpenAI
- Chat completions (GPT models)
- Text embeddings (text-embedding-3-small)
- Streaming and batch processing

### Azure AI Search
- Vector store for semantic search
- Chunk-based document indexing
- Similarity search with scoring

### Azure AI Foundry (optional)
- Pre-built agents with device code authentication
- Conversational AI workflows

## Learning Path

This notebook builds on concepts from:
- [AI Fundamentals](../../notes/03-ai/README.md) - LLMs, prompt engineering
- [RAG and Embeddings](../../notes/03-ai/ch04_rag_and_embeddings) - Retrieval-augmented generation
- [Vector Databases](../../notes/03-ai/ch05_vector_dbs) - Semantic search patterns

## Security Best Practices

1. ✅ Use `config.txt` for credentials (gitignored)
2. ✅ Never hardcode API keys in notebooks
3. ✅ Use Azure AD authentication where possible
4. ✅ Keep `config.txt.template` generic (no real credentials)
5. ✅ Review `.gitignore` before committing

## Troubleshooting

### "Module not found" errors
```bash
# Ensure virtual environment is activated
.\.venv\Scripts\Activate.ps1  # Windows
source .venv/bin/activate      # Linux/macOS

# Reinstall dependencies
pip install -r requirements.txt
```

### Azure authentication errors
- Verify `config.txt` has correct endpoints and keys
- Check Azure resource permissions (Cognitive Services User role)
- For AI Foundry: ensure device code authentication completes

### Vector store upload failures
- Check Azure AI Search service tier (Basic+ required)
- Verify admin key has write permissions
- Try smaller batch sizes if timeouts occur

## Next Steps

After completing this notebook, explore:
- [Advanced RAG patterns](../rag-agents/) - Multi-hop retrieval, agents
- [ML Feature Engineering](../ml-features/) - Embeddings in ML pipelines
- [Agentic AI](../../notes/05-agentic-ai/) - Multi-agent systems

## Resources

- [Azure OpenAI Documentation](https://learn.microsoft.com/azure/cognitive-services/openai/)
- [Azure AI Search Documentation](https://learn.microsoft.com/azure/search/)
- [LangChain Azure Integration](https://python.langchain.com/docs/integrations/platforms/microsoft)
- [Azure AI Foundry](https://learn.microsoft.com/azure/ai-studio/)
