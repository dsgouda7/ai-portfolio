"""
LLM provider builders for context-optimizer.

Each builder returns a protocol-compatible LLM object (anything with an
``.invoke(prompt: str) -> object`` method whose ``.content`` attribute
holds the response text).

Usage::

    from context_optimizer.providers import ollama, litellm

    llm = ollama.build("llama3.2:3b")
    llm = litellm.build("groq/llama-3.3-70b")
"""
