"""
HuggingFace summarization provider for context-optimizer.

Uses encoder-decoder models (BART / T5 family) via the ``transformers``
``summarization`` pipeline.  These models are 5–15× faster than
decoder-only LLMs (e.g. llama3.2:3b) on CPU because:

* The encoder runs **once** over the full input.
* The decoder generates tokens from a compact encoder hidden state —
  no weight-reload per token the way decoder-only models require.

Default model
-------------
``facebook/bart-large-cnn`` (~400 MB)

* Fine-tuned on CNN/DailyMail news summarization — dense, factual output
* Mature, well-optimised CPU kernels in PyTorch
* ~2–5 s per block on a modern 8-core CPU (vs 30–45 s for llama3.2:3b)
* ``max_new_tokens=220`` reliably respected without special sampling tricks

Other good options
------------------
``google/flan-t5-large``          (~770 MB, follows instruction-style prompts)
``google/flan-t5-base``           (~250 MB, fastest, lower quality)
``philschmid/bart-large-cnn-samsum``  (dialogue-oriented)

Usage
-----
::

    # Via _build_local_llm
    llm = _build_local_llm(provider="hf")
    llm = _build_local_llm(provider="hf", model="google/flan-t5-large")

    # Direct
    from context_optimizer.providers import hf_summarizer
    llm = hf_summarizer.build()
    response = llm.invoke("Summarize: The quick brown fox...")
    print(response.content)

    # Via env var (no code changes needed)
    # CONTEXT_OPTIMIZER_COMPRESSOR_PROVIDER=hf
    # CONTEXT_OPTIMIZER_COMPRESSOR_MODEL=facebook/bart-large-cnn
"""

from __future__ import annotations

import re
import threading
from typing import Any

_DEFAULT_MODEL = "google/flan-t5-small"

# BART's tokenizer limit is 1024 tokens ≈ 3500–4000 chars of English prose.
# We stay well under it to leave room for the summary tokens.
_MAX_INPUT_CHARS = 3200

# Output budget: match the ~200-token semantic-core target used by the
# Ollama compressor prompt.
_MAX_NEW_TOKENS = 220
_MIN_NEW_TOKENS = 60


class _Response:
    """Minimal response wrapper — mirrors the LangChain AIMessage interface."""

    __slots__ = ("content",)

    def __init__(self, content: str) -> None:
        self.content = content

    def __repr__(self) -> str:  # pragma: no cover
        return f"_Response(content={self.content[:60]!r}…)"


class HFSummarizerLLM:
    """
    Drop-in compressor-LLM backed by a HuggingFace summarization pipeline.

    Thread-safe: the pipeline is loaded once and protected by a lock so
    the single-thread Ollama gate in the compressor still works correctly.

    Parameters
    ----------
    model:
        Any HuggingFace Hub model ID that works with
        ``pipeline("summarization")``.  BART and T5 models are well-tested.
    device:
        Passed to the pipeline.  ``-1`` = CPU (default), ``0`` = first GPU.
    max_new_tokens:
        Hard cap on summary length.  Defaults to 220 tokens.
    min_new_tokens:
        Forces the model to produce at least this many tokens.
        Prevents degenerate one-sentence outputs.
    """

    def __init__(
        self,
        model: str = _DEFAULT_MODEL,
        device: int = -1,
        max_new_tokens: int = _MAX_NEW_TOKENS,
        min_new_tokens: int = _MIN_NEW_TOKENS,
    ) -> None:
        self._model_name = model
        self._device = device
        self._max_new_tokens = max_new_tokens
        self._min_new_tokens = min_new_tokens
        self._lock = threading.Lock()
        self._pipe: Any = None  # lazy-loaded on first invoke

    # ── Lazy initialisation ───────────────────────────────────────────────────

    def _get_pipe(self) -> Any:
        """Load model+tokenizer on first call; subsequent calls return cached tuple."""
        if self._pipe is not None:
            return self._pipe
        try:
            from transformers import (  # type: ignore[import]
                AutoModelForSeq2SeqLM,
                AutoTokenizer,
            )
        except ImportError as exc:
            raise ImportError(
                "transformers is required for the HF provider.\n"
                "Install with:  pip install transformers torch"
            ) from exc

        # Silence the "Setting `pad_token_id`" warning — expected for BART/T5
        import logging as _logging

        _logging.getLogger("transformers").setLevel(_logging.ERROR)

        tokenizer = AutoTokenizer.from_pretrained(self._model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(self._model_name)
        if self._device >= 0:
            import torch  # type: ignore[import]

            model = model.to(f"cuda:{self._device}")
        model.eval()  # disable dropout for deterministic inference
        self._pipe = (model, tokenizer)
        return self._pipe

    # ── Public interface ──────────────────────────────────────────────────────

    def invoke(self, prompt: str) -> _Response:
        """
        Summarize *prompt* and return a response with a ``.content`` attribute.

        BART/T5 are seq2seq summarizers, not instruction-following models.
        When given a prompt like "You are a semantic index builder … **Input
        Text:** <actual content> …" they summarize the *instructions* rather
        than the content.  We therefore strip the instruction preamble and
        feed BART only the bare document text.

        Two prompt shapes are handled:
        - Block-level (compressor.py COMPRESSION_PROMPT_TEMPLATE):
            contains ``**Input Text:**`` … ``**Output (JSON only):**``
        - Cluster-level (tree_index.py _CLUSTER_PROMPT):
            contains ``Block summaries:\\n`` … ``\\n\\nCompressed cluster core:``

        The prompt is then truncated to :data:`_MAX_INPUT_CHARS` characters.

        Post-processing
        ---------------
        BART outputs fluent prose rather than the semicolon-separated semantic
        core that the Ollama prompt produces.  We do a lightweight pass to:

        * Remove leading "This article…" / "In this…" boilerplate
        * Collapse multiple spaces / newlines
        * Strip trailing incomplete sentences (ends on ``[.!?]``)
        """
        with self._lock:
            model, tokenizer = self._get_pipe()
            text = _extract_content_from_prompt(prompt)[:_MAX_INPUT_CHARS]
            import torch  # type: ignore[import]

            inputs = tokenizer(
                text,
                return_tensors="pt",
                max_length=1024,
                truncation=True,
            )
            with torch.no_grad():
                ids = model.generate(
                    inputs["input_ids"],
                    max_new_tokens=self._max_new_tokens,
                    min_new_tokens=self._min_new_tokens,
                    no_repeat_ngram_size=3,
                    early_stopping=True,
                )
        raw: str = tokenizer.decode(ids[0], skip_special_tokens=True)
        return _Response(_postprocess(raw))

    def __repr__(self) -> str:  # pragma: no cover
        return f"HFSummarizerLLM(model={self._model_name!r}, device={self._device})"


# ── Prompt content extraction ─────────────────────────────────────────────────


def _extract_content_from_prompt(prompt: str) -> str:
    """
    Strip instruction preamble from an LLM prompt and return the bare content.

    context-optimizer uses two prompt shapes:

    1. *Block-level* (compressor.py ``COMPRESSION_PROMPT_TEMPLATE``):
       ``… **Input Text:**\\n<content>\\n\\n**Output (JSON only):** …``

    2. *Cluster-level* (tree_index.py ``_CLUSTER_PROMPT``):
       ``… Block summaries:\\n<summaries>\\n\\nCompressed cluster core:``

    BART/T5 are pure seq2seq summarizers; they cannot follow instructions,
    so handing them the full prompt causes them to "summarize" the instruction
    text instead of the document.  This function extracts just the content
    so that BART receives what it was designed to process.
    """
    # ── Cluster-level prompt ──────────────────────────────────────────────────
    BLOCK_SUMMARIES_MARKER = "Block summaries:\n"
    CLUSTER_CORE_MARKER = "\n\nCompressed cluster core:"
    if BLOCK_SUMMARIES_MARKER in prompt and CLUSTER_CORE_MARKER in prompt:
        start = prompt.index(BLOCK_SUMMARIES_MARKER) + len(BLOCK_SUMMARIES_MARKER)
        end = prompt.rfind(CLUSTER_CORE_MARKER)
        return prompt[start:end].strip()

    # ── Block-level ingest prompt (compressor.py BLOCK_SUMMARY_PROMPT) ────────
    # Format: "… Text block (extract facts…):\n<content>\n\nCompressed semantic core:"
    TEXT_BLOCK_MARKER = "\nText block ("
    SEMANTIC_CORE_MARKER = "\n\nCompressed semantic core:"
    if TEXT_BLOCK_MARKER in prompt and SEMANTIC_CORE_MARKER in prompt:
        # Skip past the "Text block (…):\n" header line
        start = prompt.index(TEXT_BLOCK_MARKER)
        # Advance to the newline after the closing ":\n"
        newline_after = prompt.index(":\n", start) + 2
        end = prompt.rfind(SEMANTIC_CORE_MARKER)
        return prompt[newline_after:end].strip()

    # ── Block-level prompt (compressor.py COMPRESSION_PROMPT_TEMPLATE) ────────
    INPUT_TEXT_MARKER = "**Input Text:**"
    if INPUT_TEXT_MARKER in prompt:
        start = prompt.index(INPUT_TEXT_MARKER) + len(INPUT_TEXT_MARKER)
        content = prompt[start:]
        # Trim at the start of the output/instructions section
        for end_marker in ("\n\n**Output", "\n**Output", "\n\nRespond with"):
            if end_marker in content:
                content = content[: content.index(end_marker)]
                break
        return content.strip()

    # ── Unrecognised format — pass through unchanged ──────────────────────────
    return prompt


# ── Post-processing ───────────────────────────────────────────────────────────

_BOILERPLATE = re.compile(
    r"^(summary:|this (?:article|passage|text|document|excerpt)|in this (?:article|passage|text))[,:]?\s*",
    re.IGNORECASE,
)


def _postprocess(text: str) -> str:
    """Clean up BART/T5 output for use as a ChromaDB document."""
    text = text.strip()
    # Remove common encoder-decoder boilerplate prefixes
    text = _BOILERPLATE.sub("", text).strip()
    # Capitalise first letter after stripping
    if text:
        text = text[0].upper() + text[1:]
    # Collapse internal whitespace
    text = re.sub(r"\s{2,}", " ", text)
    # Trim to last complete sentence so ChromaDB doesn't index a fragment
    last_stop = max(text.rfind("."), text.rfind("!"), text.rfind("?"))
    if last_stop > len(text) // 2:
        text = text[: last_stop + 1]
    return text


# ── Builder (called by _build_local_llm) ─────────────────────────────────────


def build(
    model: str = _DEFAULT_MODEL,
    device: int = -1,
    max_new_tokens: int = _MAX_NEW_TOKENS,
    min_new_tokens: int = _MIN_NEW_TOKENS,
) -> HFSummarizerLLM:
    """
    Build a :class:`HFSummarizerLLM` instance.

    Parameters
    ----------
    model:
        HuggingFace model ID.  Defaults to ``facebook/bart-large-cnn``.
    device:
        ``-1`` for CPU (default), ``0`` for first CUDA GPU.
    max_new_tokens / min_new_tokens:
        Output length bounds passed directly to the pipeline.
    """
    return HFSummarizerLLM(
        model=model,
        device=device,
        max_new_tokens=max_new_tokens,
        min_new_tokens=min_new_tokens,
    )
