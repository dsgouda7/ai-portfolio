# GenAI Learning Arc

This track builds from sequence modeling fundamentals through applied generative AI projects.
Directories `01` through `05` are concept-building notebooks that establish theory and
implementation fluency. Directories `06` through `10` are applied mini-projects that use
those foundations to solve realistic tasks.

See [authoring-guide.md](authoring-guide.md) for notebook conventions, cell-tagging rules,
and how to add new content to this track.

---

## Contents

| # | Directory | Topic | What you build | What you can do when done | Prerequisites |
|---|-----------|-------|----------------|--------------------------|---------------|
| 1 | `01-rnns/` | Recurrent Neural Networks | Character-level RNN in PyTorch and Keras/TF | Implement sequence models from scratch; explain vanishing gradients | Basic Python, NumPy |
| 2 | `02-transformers/` | Transformer architecture | Scaled dot-product attention and full encoder stack from first principles | Read and modify transformer code; explain every component mathematically | `01-rnns/` |
| 3 | `03-encoder-decoder/` | Encoder-Decoder architecture | Seq2seq model with cross-attention for translation | Build and train encoder-decoder models; tune beam search | `02-transformers/` |
| 4 | `04-llm/` | Applied LLM patterns | Hybrid search pipeline, LLM gateway, RAG evaluation harness | Wire together retrieval + generation; evaluate answer quality quantitatively | `03-encoder-decoder/` |
| 5 | `05-llm-tuning/` | Fine-tuning and alignment | LoRA adapter, DPO preference alignment, PEFT training loop | Fine-tune a causal LM on domain data; run preference optimization | `04-llm/` |
| 6 | `06-conversation-analysis/` | Conversation summarization and intent | FLAN-T5 summarizer and intent classifier on real transcripts | Extract structured insights from raw conversation logs | `04-llm/` |
| 7 | `07-conversational-ai/` | Multi-turn conversational AI | Stateful chat agent backed by Qwen 2.5 | Build a context-aware chat system with history management | `04-llm/` |
| 8 | `08-image-captioning/` | Vision-language models | BLIP-2 image captioning pipeline with prompt steering | Caption and query images using a multimodal LLM | `04-llm/` |
| 9 | `09-text-translation/` | Speech-to-text translation | Whisper transcription + Helsinki-NLP translation pipeline | Transcribe audio and translate across languages end-to-end | `04-llm/` |
| 10 | `10-voice-assistant/` | Voice assistant project | Multi-file Flask app with STT, LLM response, and TTS | Ship a runnable voice assistant with a web interface | `06` through `09` |

---

## Gold-standard notebook

`02-transformers/transformers.ipynb` (or `transformers-keras.ipynb`) is the reference
implementation for the entire track. It is the most heavily annotated notebook and
demonstrates the authoring conventions all other notebooks should follow.

---

## Learning path summary

```
01-rnns -> 02-transformers -> 03-encoder-decoder -> 04-llm -> 05-llm-tuning
                                                        |
                              06-conversation-analysis  07-conversational-ai
                              08-image-captioning        09-text-translation
                                                        |
                                                   10-voice-assistant
```
