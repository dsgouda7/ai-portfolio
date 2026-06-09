"""
API Key Management System

This module provides a centralized way to manage API keys for various LLM providers.
Store your actual API keys in this file and it will be gitignored for security.

INSTRUCTIONS:
1. Copy this file to 'api_keys.py' in the same directory
2. Replace the placeholder values with your actual API keys
3. Never commit api_keys.py to version control

SUPPORTED PROVIDERS:
- OpenAI: https://platform.openai.com/api-keys
- Google Gemini: https://ai.google.dev/
- Groq: https://console.groq.com/keys
- Tavily: https://tavily.com/
"""

# OpenAI API Key
# Get from: https://platform.openai.com/api-keys
OPENAI_API_KEY = "your-openai-api-key-here"

# Google Gemini API Key
# Get from: https://ai.google.dev/
GOOGLE_API_KEY = "your-google-api-key-here"

# Groq API Key
# Get from: https://console.groq.com/keys
GROQ_API_KEY = "your-groq-api-key-here"

# Tavily API Key (for web search)
# Get from: https://tavily.com/
TAVILY_API_KEY = "your-tavily-api-key-here"


def get_api_key(provider: str) -> str:
    """
    Get API key for a specific provider.

    Args:
        provider: One of 'openai', 'google', 'groq', 'tavily'

    Returns:
        The API key string

    Raises:
        ValueError: If provider is not supported or key not configured
    """
    keys = {
        'openai': OPENAI_API_KEY,
        'google': GOOGLE_API_KEY,
        'groq': GROQ_API_KEY,
        'tavily': TAVILY_API_KEY
    }

    if provider not in keys:
        raise ValueError(f"Unsupported provider: {provider}. Use one of: {list(keys.keys())}")

    key = keys[provider]
    if key.startswith("your-") or key == "":
        raise ValueError(f"API key for {provider} not configured. Please update api_keys.py")

    return key


def validate_keys():
    """Validate that all API keys have been configured."""
    providers = ['openai', 'google', 'groq', 'tavily']
    unconfigured = []

    for provider in providers:
        try:
            get_api_key(provider)
        except ValueError:
            unconfigured.append(provider)

    if unconfigured:
        print(f"⚠️  Warning: The following API keys are not configured: {', '.join(unconfigured)}")
        print("   Update api_keys.py with your actual keys to use these providers.")
        return False
    else:
        print("✅ All API keys configured!")
        return True
