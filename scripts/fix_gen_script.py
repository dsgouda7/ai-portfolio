"""Fix triple-double-quote docstrings inside code cells of the gen script."""
import pathlib

p = pathlib.Path(r"c:\repos\ai-portfolio\scripts\gen_rag_eval_notebook.py")
content = p.read_text(encoding="utf-8")

# All triple-double-quote docstrings inside cells.append(code(r"""...")) blocks
# need to become triple-single-quote so they don't close the outer r-string.
replacements = [
    ('    """Hybrid retrieval: RRF fusion of semantic and BM25 rankings."""',
     "    '''Hybrid retrieval: RRF fusion of semantic and BM25 rankings.'''"),
    ('    """Mock generation: extract the single most query-relevant sentence from context.\n\n    A real LLM would synthesise across sentences; this deterministic proxy\n    is enough to produce grounded or hallucinated answers on demand.\n    """',
     "    '''Mock generation: extract the single most query-relevant sentence.\n\n    A real LLM synthesises across sentences; this proxy is deterministic.\n    '''"),
    ('    """Standard RAG pipeline: retrieve then generate."""',
     "    '''Standard RAG pipeline: retrieve then generate.'''"),
    ('    """Average cosine similarity between the query and each retrieved document."""',
     "    '''Average cosine similarity between query and retrieved docs.'''"),
    ('    """Lowercase, strip punctuation, return set of non-stop tokens."""',
     "    '''Lowercase, strip punctuation, return set of non-stop tokens.'''"),
    ('    """Token recall: fraction of answer tokens present in the context."""',
     "    '''Token recall: fraction of answer tokens present in the context.'''"),
    ('    """Cosine sim between question and answer embeddings (no reference needed)."""',
     "    '''Cosine sim between question and answer embeddings (no reference needed).'''"),
    ('    """Standard dynamic-programming LCS length."""',
     "    '''Standard dynamic-programming LCS length.'''"),
    ('    """ROUGE-L recall: LCS length / reference length (tokenized on words)."""',
     "    '''ROUGE-L recall: LCS length / reference length (tokenized on words).'''"),
]

for old, new in replacements:
    if old in content:
        content = content.replace(old, new)
        print(f"Replaced: {old[:60]!r}")
    else:
        print(f"NOT FOUND: {old[:60]!r}")

p.write_text(content, encoding="utf-8")
print("\nDone. File updated.")
