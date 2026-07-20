"""
Create the fine-tuning-in-action.ipynb notebook programmatically.
Run: python create_notebook.py
"""
import json
import os

NOTEBOOK_PATH = os.path.join(os.path.dirname(__file__), "fine-tuning-in-action.ipynb")
EXERCISE_PATH = os.path.join(os.path.dirname(__file__), "fine-tuning-in-action-exercise.ipynb")

def md(source):
    return {"cell_type": "markdown", "id": next_id(), "metadata": {}, "source": source}

def code(source):
    return {"cell_type": "code", "id": next_id(), "metadata": {}, "source": source,
            "outputs": [], "execution_count": None}

_counter = [0]
def next_id():
    _counter[0] += 1
    return f"cell{_counter[0]:04d}"

def make_notebook(cells):
    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {
                "display_name": "llm-tuning",
                "language": "python",
                "name": "llm-tuning"
            },
            "language_info": {
                "name": "python",
                "version": "3.12.10"
            }
        },
        "cells": cells
    }

# ---------------------------------------------------------------------------
# REFERENCE NOTEBOOK cells
# ---------------------------------------------------------------------------

cells = []

cells.append(md("""# Fine-Tuning in Action: The Everglades Cipher Corpus

This notebook demonstrates every major fine-tuning technique using a real literary corpus:
**The Everglades Cipher** — an original noir-historical detective novel (~650KB of text).

We train GPT-2 on this corpus using six different approaches and evaluate the results with
questions of increasing difficulty about the novel's characters, plot, and historical content.

## Techniques Covered
1. **Continued Pretraining** — further train GPT-2 on novel text chunks
2. **Supervised Fine-Tuning (SFT)** — instruction-following on Q&A pairs
3. **Direct Preference Optimization (DPO)** — train on chosen/rejected RLHF pairs
4. **Full Fine-Tuning** — update all parameters (small model)
5. **LoRA** — Low-Rank Adaptation (parameter-efficient)
6. **QLoRA** — Quantized LoRA (memory-efficient)

## Dataset Summary
| File | Type | Records |
|------|------|---------|
| `data/pretraining_chunks.jsonl` | Plain text chunks (~500 words each) | ~260 |
| `data/instruction_prompts.jsonl` | Instruction → output pairs | 20 |
| `data/rlhf_pairs.jsonl` | Prompt + chosen + rejected | 66 |
| `data/combined_training_data.jsonl` | Unified format | 86 |

> **Note**: This notebook is for learning. GPT-2 is small (117M params) and results will be
> qualitative rather than production-quality. The goal is to see each technique working, not
> to train a state-of-the-art model.
"""))

cells.append(md("## Section 0: Environment Setup"))

cells.append(code("""\
# Install required packages (run once)
# %pip install transformers peft trl datasets torch accelerate bitsandbytes -q
"""))

cells.append(code("""\
import os, json, glob, textwrap
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from transformers import DataCollatorForLanguageModeling
from datasets import Dataset
import warnings
warnings.filterwarnings("ignore")

# Paths
ROOT = Path(".")
DATA_DIR = ROOT / "data"
CONTENT_DIR = ROOT / "content" / "the-everglades-cipher"

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")
print(f"PyTorch: {torch.__version__}")
"""))

cells.append(md("## Section 1: Explore the Corpus"))

cells.append(code("""\
# Load all novel text files
txt_files = sorted(CONTENT_DIR.glob("*.txt"))
print(f"Novel files: {len(txt_files)}")

total_chars = 0
for f in txt_files:
    size = f.stat().st_size
    total_chars += size
    print(f"  {f.name:60s}  {size:>7,} bytes")

print(f"\\nTotal novel corpus: {total_chars:,} bytes ({total_chars / 1024:.1f} KB)")
"""))

cells.append(code("""\
# Load training data files
def load_jsonl(path):
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
    return records

pretraining = load_jsonl(DATA_DIR / "pretraining_chunks.jsonl")
instructions = load_jsonl(DATA_DIR / "instruction_prompts.jsonl")
rlhf_pairs = load_jsonl(DATA_DIR / "rlhf_pairs.jsonl")

print(f"Pretraining chunks: {len(pretraining)}")
print(f"Instruction prompts: {len(instructions)}")
print(f"RLHF pairs: {len(rlhf_pairs)}")
print()

# Show a sample from each
print("=== Pretraining chunk sample (first 200 chars) ===")
print(pretraining[0]["text"][:200])
print()
print("=== Instruction sample ===")
sample = instructions[0]
print(f"Instruction: {sample['instruction'][:100]}...")
print(f"Output: {sample['output'][:100]}...")
print()
print("=== RLHF pair sample ===")
pair = rlhf_pairs[0]
print(f"Prompt: {pair['prompt'][:80]}...")
print(f"Chosen (first 80 chars): {pair['chosen'][:80]}...")
print(f"Rejected (first 80 chars): {pair['rejected'][:80]}...")
"""))

cells.append(code("""\
# Token count statistics using GPT-2 tokenizer
from transformers import GPT2Tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

# Count tokens in pretraining data
all_token_counts = [len(tokenizer.encode(r["text"])) for r in pretraining]
print(f"Pretraining chunks — token stats:")
print(f"  Total tokens: {sum(all_token_counts):,}")
print(f"  Mean per chunk: {sum(all_token_counts)/len(all_token_counts):.0f}")
print(f"  Max: {max(all_token_counts)}, Min: {min(all_token_counts)}")
"""))

cells.append(md("## Section 2: Continued Pretraining"))

cells.append(md("""\
**Continued pretraining** extends the language model's base knowledge by training on
unlabeled text. We feed the novel's text in ~500-word windows and train the model to
predict the next token.

The model learns the novel's vocabulary, style, and factual content at a statistical level.
After training, the model is more likely to generate text in the novel's style, use its
character names and locations, and produce content consistent with its plot.

**When to use**: Building a domain-specific language model. The model learns *what* is in
the corpus, not *how to follow instructions*.
"""))

cells.append(code("""\
from transformers import GPT2LMHeadModel

# Load base model
model_name = "gpt2"
model_pt = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer_pt = GPT2Tokenizer.from_pretrained(model_name)
tokenizer_pt.pad_token = tokenizer_pt.eos_token
model_pt = model_pt.to(device)

print(f"Model parameters: {sum(p.numel() for p in model_pt.parameters()):,}")
"""))

cells.append(code("""\
def tokenize_for_lm(examples, tokenizer, max_length=512):
    encodings = tokenizer(
        examples["text"],
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_tensors=None,
    )
    encodings["labels"] = encodings["input_ids"].copy()
    return encodings

# Build HuggingFace Dataset from pretraining chunks
pt_dataset = Dataset.from_list(pretraining)
pt_dataset = pt_dataset.map(
    lambda x: tokenize_for_lm(x, tokenizer_pt),
    batched=True,
    remove_columns=["text"]
)
pt_dataset.set_format("torch")

# Use 80% train, 20% eval
split = pt_dataset.train_test_split(test_size=0.2, seed=42)
train_pt = split["train"]
eval_pt = split["test"]
print(f"Train: {len(train_pt)} samples | Eval: {len(eval_pt)} samples")
"""))

cells.append(code("""\
from transformers import DataCollatorForLanguageModeling

training_args_pt = TrainingArguments(
    output_dir="./checkpoints/continued-pretraining",
    num_train_epochs=2,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    warmup_steps=50,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=20,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    fp16=torch.cuda.is_available(),
    report_to="none",
)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer_pt, mlm=False)

trainer_pt = Trainer(
    model=model_pt,
    args=training_args_pt,
    train_dataset=train_pt,
    eval_dataset=eval_pt,
    data_collator=data_collator,
)

print("Starting continued pretraining...")
trainer_pt.train()
print("Continued pretraining complete.")
"""))

cells.append(code("""\
# Quick generation test after pretraining
def generate_text(model, tokenizer, prompt, max_new_tokens=100):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.8,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(output[0], skip_special_tokens=True)

prompt_pt = "Jake Malone sat on the deck of the Reluctant Witness and looked at"
print(f"Prompt: {prompt_pt}")
print()
print(generate_text(model_pt, tokenizer_pt, prompt_pt))
"""))

cells.append(md("## Section 3: Supervised Fine-Tuning (SFT)"))

cells.append(md("""\
**Supervised Fine-Tuning (SFT)** trains the model to follow instructions by formatting
data as `Instruction → Output` pairs and training the model to generate the output given
the instruction.

The key difference from pretraining: we only compute the loss on the *output* tokens, not
the instruction tokens. The model learns to *respond* rather than to *continue*.

**When to use**: Adapting a base model to follow instructions or answer questions in a
specific domain.
"""))

cells.append(code("""\
from transformers import GPT2LMHeadModel

model_sft = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
tokenizer_sft = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer_sft.pad_token = tokenizer_sft.eos_token

def format_instruction(item):
    \"\"\"Format as: ### Instruction:\\n...\\n\\n### Response:\\n...\"\"\"
    instruction = item["instruction"]
    input_text = item.get("input", "").strip()
    output = item["output"]
    
    if input_text:
        text = f"### Instruction:\\n{instruction}\\n\\n### Context:\\n{input_text}\\n\\n### Response:\\n{output}"
    else:
        text = f"### Instruction:\\n{instruction}\\n\\n### Response:\\n{output}"
    return {"text": text}

# Format all instruction examples
formatted = [format_instruction(item) for item in instructions]
sft_dataset = Dataset.from_list(formatted)
sft_tokenized = sft_dataset.map(
    lambda x: tokenize_for_lm(x, tokenizer_sft, max_length=512),
    batched=True,
    remove_columns=["text"]
)
sft_tokenized.set_format("torch")

split_sft = sft_tokenized.train_test_split(test_size=0.2, seed=42)
print(f"SFT train: {len(split_sft['train'])} | eval: {len(split_sft['test'])}")
"""))

cells.append(code("""\
training_args_sft = TrainingArguments(
    output_dir="./checkpoints/sft",
    num_train_epochs=5,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    warmup_steps=10,
    weight_decay=0.01,
    logging_steps=5,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    fp16=torch.cuda.is_available(),
    report_to="none",
)

trainer_sft = Trainer(
    model=model_sft,
    args=training_args_sft,
    train_dataset=split_sft["train"],
    eval_dataset=split_sft["test"],
    data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer_sft, mlm=False),
)

print("Starting SFT training...")
trainer_sft.train()
print("SFT complete.")
"""))

cells.append(code("""\
# Test the SFT model
sft_prompt = "### Instruction:\\nWho is Jake Malone?\\n\\n### Response:\\n"
print(f"Prompt: {sft_prompt}")
print()
print(generate_text(model_sft, tokenizer_sft, sft_prompt, max_new_tokens=150))
"""))

cells.append(md("## Section 4: Direct Preference Optimization (DPO)"))

cells.append(md("""\
**Direct Preference Optimization (DPO)** trains the model to prefer "good" responses over
"bad" ones using paired preference data (chosen vs. rejected responses to the same prompt).

Unlike RLHF with a separate reward model, DPO directly optimizes the policy using the
binary cross-entropy loss on the preference pairs. It is simpler and more stable than full
RLHF while achieving similar alignment results.

**When to use**: Improving response quality and alignment when you have preference data
(human feedback on which of two responses is better).

> **Note**: DPO requires the `trl` library's `DPOTrainer`. We'll also need a reference
> model (the SFT model we just trained) to compute KL divergence.
"""))

cells.append(code("""\
try:
    from trl import DPOTrainer, DPOConfig
    TRL_AVAILABLE = True
    print("trl library available — DPO training enabled")
except ImportError:
    TRL_AVAILABLE = False
    print("trl library not available — install with: pip install trl")
    print("Skipping DPO section.")
"""))

cells.append(code("""\
if TRL_AVAILABLE:
    from transformers import GPT2LMHeadModel
    
    # DPO needs: base (policy) model + reference model
    model_dpo = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
    model_dpo_ref = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
    tokenizer_dpo = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer_dpo.pad_token = tokenizer_dpo.eos_token
    
    # DPOTrainer expects: prompt, chosen, rejected columns
    dpo_dataset = Dataset.from_list(rlhf_pairs)
    
    split_dpo = dpo_dataset.train_test_split(test_size=0.15, seed=42)
    print(f"DPO train: {len(split_dpo['train'])} | eval: {len(split_dpo['test'])}")
"""))

cells.append(code("""\
if TRL_AVAILABLE:
    dpo_config = DPOConfig(
        output_dir="./checkpoints/dpo",
        num_train_epochs=3,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        learning_rate=1e-5,
        beta=0.1,          # KL penalty coefficient
        max_length=512,
        max_prompt_length=128,
        logging_steps=5,
        eval_strategy="epoch",
        save_strategy="epoch",
        fp16=torch.cuda.is_available(),
        report_to="none",
        remove_unused_columns=False,
    )
    
    trainer_dpo = DPOTrainer(
        model=model_dpo,
        ref_model=model_dpo_ref,
        args=dpo_config,
        train_dataset=split_dpo["train"],
        eval_dataset=split_dpo["test"],
        processing_class=tokenizer_dpo,
    )
    
    print("Starting DPO training...")
    trainer_dpo.train()
    print("DPO complete.")
"""))

cells.append(md("## Section 5: Full Fine-Tuning"))

cells.append(md("""\
**Full Fine-Tuning** updates all model parameters during training. This is the most
computationally expensive approach but gives the model the most flexibility to adapt.

For GPT-2 (117M params), this is feasible on a consumer GPU. For larger models (>1B),
parameter-efficient methods like LoRA are preferred.

**When to use**: When you have sufficient compute and a large, high-quality dataset.
Full fine-tuning can achieve higher peak performance than LoRA but requires more
memory and time.
"""))

cells.append(code("""\
from transformers import GPT2LMHeadModel

# For full fine-tuning we use the combined dataset (pretraining + instruction)
all_training_texts = (
    [r["text"] for r in pretraining] +
    [f\"### Instruction:\\n{i['instruction']}\\n\\n### Response:\\n{i['output']}\"
     for i in instructions]
)
print(f"Total training samples for full fine-tuning: {len(all_training_texts)}")

model_full = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
tokenizer_full = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer_full.pad_token = tokenizer_full.eos_token

full_dataset = Dataset.from_dict({"text": all_training_texts})
full_tokenized = full_dataset.map(
    lambda x: tokenize_for_lm(x, tokenizer_full, max_length=512),
    batched=True,
    remove_columns=["text"]
)
full_tokenized.set_format("torch")
split_full = full_tokenized.train_test_split(test_size=0.1, seed=42)
print(f"Full FT train: {len(split_full['train'])} | eval: {len(split_full['test'])}")
"""))

cells.append(code("""\
training_args_full = TrainingArguments(
    output_dir="./checkpoints/full-finetune",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    warmup_steps=50,
    weight_decay=0.01,
    learning_rate=2e-5,
    logging_steps=20,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    fp16=torch.cuda.is_available(),
    report_to="none",
)

trainer_full = Trainer(
    model=model_full,
    args=training_args_full,
    train_dataset=split_full["train"],
    eval_dataset=split_full["test"],
    data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer_full, mlm=False),
)

print("Starting full fine-tuning...")
trainer_full.train()
print("Full fine-tuning complete.")
"""))

cells.append(md("## Section 6: LoRA Fine-Tuning"))

cells.append(md("""\
**Low-Rank Adaptation (LoRA)** is a parameter-efficient fine-tuning method that injects
small trainable matrices into the frozen model's weight matrices. Instead of updating all
117M parameters, we only update ~0.5% (the LoRA adapters).

**How it works**: For a weight matrix W (d×k), LoRA adds ΔW = A×B where A is d×r and
B is r×k, with r << min(d,k). The rank r controls the capacity of the adaptation.

**Key hyperparameters**:
- `r`: rank of the adaptation matrices (typical: 4-64)
- `lora_alpha`: scaling factor (typical: r or 2r)
- `target_modules`: which layers to adapt (for GPT-2: `c_attn`)
- `lora_dropout`: regularization

**When to use**: When compute or memory is limited, or when you want to serve many
fine-tuned variants of the same base model efficiently.
"""))

cells.append(code("""\
try:
    from peft import get_peft_model, LoraConfig, TaskType
    PEFT_AVAILABLE = True
    print("peft library available — LoRA enabled")
except ImportError:
    PEFT_AVAILABLE = False
    print("peft not available — install with: pip install peft")
"""))

cells.append(code("""\
if PEFT_AVAILABLE:
    from transformers import GPT2LMHeadModel
    from peft import get_peft_model, LoraConfig, TaskType
    
    base_model_lora = GPT2LMHeadModel.from_pretrained("gpt2")
    tokenizer_lora = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer_lora.pad_token = tokenizer_lora.eos_token
    
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        target_modules=["c_attn"],   # GPT-2's combined QKV projection
        lora_dropout=0.05,
        bias="none",
    )
    
    model_lora = get_peft_model(base_model_lora, lora_config)
    model_lora = model_lora.to(device)
    model_lora.print_trainable_parameters()
"""))

cells.append(code("""\
if PEFT_AVAILABLE:
    training_args_lora = TrainingArguments(
        output_dir="./checkpoints/lora",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        warmup_steps=50,
        weight_decay=0.01,
        learning_rate=3e-4,  # LoRA typically uses higher LR
        logging_steps=20,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        fp16=torch.cuda.is_available(),
        report_to="none",
    )
    
    trainer_lora = Trainer(
        model=model_lora,
        args=training_args_lora,
        train_dataset=split_full["train"],
        eval_dataset=split_full["test"],
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer_lora, mlm=False),
    )
    
    print("Starting LoRA training...")
    trainer_lora.train()
    print("LoRA training complete.")
"""))

cells.append(code("""\
if PEFT_AVAILABLE:
    # Test LoRA model generation
    lora_prompt = "Arturo Vasquez-Cortez opened the manuscript box and"
    print(f"LoRA model generation:")
    print(f"Prompt: {lora_prompt}")
    print()
    result = generate_text(model_lora, tokenizer_lora, lora_prompt)
    print(result)
"""))

cells.append(md("## Section 7: QLoRA (Quantized LoRA)"))

cells.append(md("""\
**QLoRA** combines 4-bit quantization with LoRA to dramatically reduce memory requirements.
The base model is loaded in 4-bit precision (reducing memory by ~4x), and the LoRA adapters
are trained in bfloat16. This allows fine-tuning models that would otherwise not fit in GPU
memory.

**Key components**:
- `BitsAndBytesConfig`: 4-bit quantization configuration
- `prepare_model_for_kbit_training`: prepares quantized model for gradient computation
- LoRA adapters trained on top of the frozen quantized model

**When to use**: Fine-tuning large models (7B+) on consumer hardware. For GPT-2 the
memory savings are less critical, but the technique is identical for larger models.

> **Hardware note**: QLoRA requires `bitsandbytes` and a CUDA GPU. On CPU-only machines,
> this section will fall back to standard LoRA.
"""))

cells.append(code("""\
try:
    import bitsandbytes as bnb
    from transformers import BitsAndBytesConfig
    BNB_AVAILABLE = True and torch.cuda.is_available()
    print(f"bitsandbytes available: {BNB_AVAILABLE}")
except ImportError:
    BNB_AVAILABLE = False
    print("bitsandbytes not available — install with: pip install bitsandbytes")
    print("QLoRA will fall back to standard LoRA on CPU.")
"""))

cells.append(code("""\
from peft import prepare_model_for_kbit_training, get_peft_model, LoraConfig, TaskType

if BNB_AVAILABLE and PEFT_AVAILABLE:
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    
    base_model_qlora = AutoModelForCausalLM.from_pretrained(
        "gpt2",
        quantization_config=bnb_config,
        device_map="auto",
    )
    tokenizer_qlora = AutoTokenizer.from_pretrained("gpt2")
    tokenizer_qlora.pad_token = tokenizer_qlora.eos_token
    
    base_model_qlora = prepare_model_for_kbit_training(base_model_qlora)
    
    qlora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        target_modules=["c_attn"],
        lora_dropout=0.05,
        bias="none",
    )
    
    model_qlora = get_peft_model(base_model_qlora, qlora_config)
    model_qlora.print_trainable_parameters()
    print("QLoRA model ready (4-bit quantized + LoRA adapters)")
    
elif PEFT_AVAILABLE:
    # Fall back to standard LoRA on CPU
    print("Falling back to standard LoRA (no GPU/bitsandbytes)")
    model_qlora = model_lora  # reuse the LoRA model from section 6
    tokenizer_qlora = tokenizer_lora
"""))

cells.append(code("""\
if PEFT_AVAILABLE:
    training_args_qlora = TrainingArguments(
        output_dir="./checkpoints/qlora",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        warmup_steps=50,
        learning_rate=2e-4,
        logging_steps=20,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        fp16=False,   # use bf16 with 4-bit quantization
        bf16=BNB_AVAILABLE,
        report_to="none",
    )
    
    trainer_qlora = Trainer(
        model=model_qlora,
        args=training_args_qlora,
        train_dataset=split_full["train"],
        eval_dataset=split_full["test"],
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer_qlora, mlm=False),
    )
    
    print("Starting QLoRA training...")
    trainer_qlora.train()
    print("QLoRA training complete.")
"""))

cells.append(md("## Section 8: Q&A Evaluation — Increasing Difficulty"))

cells.append(md("""\
Now we evaluate all trained models on questions about **The Everglades Cipher**,
organized by difficulty:

| Level | Type | Example |
|-------|------|---------|
| Easy | Direct fact recall | "What is Jake Malone's boat called?" |
| Medium | Multi-sentence comprehension | "What was in the Cayo Espiritu archive?" |
| Hard | Historical reasoning | "Explain the Philip IV–Templar intelligence relationship" |
| Expert | Synthesis & analysis | "Analyze the preservation theme across centuries" |

> **Honest expectation**: GPT-2 is a small base model. Its answers will often be
> plausible-sounding but not always factually correct about the novel. The point is to
> observe the *difference* between base and fine-tuned models, not perfect accuracy.
"""))

cells.append(code("""\
QUESTIONS = [
    # Easy
    {"level": "Easy", "q": "What is the name of Jake Malone's houseboat?"},
    {"level": "Easy", "q": "Where is Arturo Vasquez-Cortez's bookshop located?"},
    {"level": "Easy", "q": "What does Elena Vasquez-Cortez do for a living?"},
    # Medium
    {"level": "Medium", "q": "What is the Codex Almeida and why is it significant?"},
    {"level": "Medium", "q": "How does Diego de Almeida's cipher system work?"},
    {"level": "Medium", "q": "Who is Nacho Reyes and how does he help the investigation?"},
    # Hard
    {"level": "Hard", "q": "What do the Philip IV letters reveal about the Templar dissolution?"},
    {"level": "Hard", "q": "How did the Knights Templar's maritime network make the archive transfer possible?"},
    {"level": "Hard", "q": "Explain the connection between the Templar dissolution and the Portuguese Age of Discovery."},
    # Expert
    {"level": "Expert", "q": "Analyze the parallel between Diego de Almeida and Arturo Vasquez-Cortez as figures of preservation across centuries."},
    {"level": "Expert", "q": "How does the novel use the ecological subplot to reinforce its central thematic argument about preservation?"},
]

def evaluate_model(model, tokenizer, questions, model_name, max_new_tokens=120):
    results = []
    for item in questions:
        prompt = f"### Instruction:\\n{item['q']}\\n\\n### Response:\\n"
        response = generate_text(model, tokenizer, prompt, max_new_tokens=max_new_tokens)
        # Extract just the response part
        if "### Response:" in response:
            answer = response.split("### Response:")[-1].strip()
        else:
            answer = response[len(prompt):].strip()
        results.append({
            "level": item["level"],
            "question": item["q"],
            "answer": answer[:300]  # truncate for display
        })
    return results

print("Running evaluations across all trained models...")
print("(This may take a few minutes)")
"""))

cells.append(code("""\
# Evaluate SFT model
print("\\n" + "="*60)
print("SFT MODEL RESPONSES")
print("="*60)
sft_results = evaluate_model(model_sft, tokenizer_sft, QUESTIONS, "SFT")
for r in sft_results:
    print(f"\\n[{r['level']}] {r['question']}")
    print(f"→ {r['answer']}")
"""))

cells.append(code("""\
# Evaluate LoRA model
if PEFT_AVAILABLE:
    print("\\n" + "="*60)
    print("LoRA MODEL RESPONSES")
    print("="*60)
    lora_results = evaluate_model(model_lora, tokenizer_lora, QUESTIONS, "LoRA")
    for r in lora_results:
        print(f"\\n[{r['level']}] {r['question']}")
        print(f"→ {r['answer']}")
"""))

cells.append(code("""\
# Compare base model vs fine-tuned on 3 representative questions
print("\\n" + "="*60)
print("BASE MODEL vs FINE-TUNED: Side-by-side comparison")
print("="*60)

base_model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
base_tok = GPT2Tokenizer.from_pretrained("gpt2")
base_tok.pad_token = base_tok.eos_token

sample_qs = [
    QUESTIONS[0],   # Easy: houseboat name
    QUESTIONS[3],   # Medium: Codex Almeida
    QUESTIONS[6],   # Hard: Philip IV letters
]

for item in sample_qs:
    prompt = f"### Instruction:\\n{item['q']}\\n\\n### Response:\\n"
    base_ans = generate_text(base_model, base_tok, prompt, 80).split("### Response:")[-1].strip()
    sft_ans = generate_text(model_sft, tokenizer_sft, prompt, 80).split("### Response:")[-1].strip()
    
    print(f"\\n[{item['level']}] {item['q']}")
    print(f"BASE:     {base_ans[:200]}")
    print(f"SFT:      {sft_ans[:200]}")
"""))

cells.append(md("## Section 9: Comparison Summary"))

cells.append(code("""\
import time

summary = [
    {
        "Method": "Continued Pretraining",
        "Params Updated": "All (117M)",
        "Data Type": "Raw text chunks",
        "Epochs": 2,
        "Key Hyperparams": "lr=5e-5, batch=4",
        "Strengths": "Learns domain vocabulary and style",
        "Weaknesses": "Cannot follow instructions",
        "Use When": "Building domain-adapted base model",
    },
    {
        "Method": "SFT",
        "Params Updated": "All (117M)",
        "Data Type": "Instruction → Output pairs",
        "Epochs": 5,
        "Key Hyperparams": "lr=5e-5, batch=2",
        "Strengths": "Learns to follow instructions",
        "Weaknesses": "May overfit on small datasets",
        "Use When": "Adapting to Q&A / chat format",
    },
    {
        "Method": "DPO",
        "Params Updated": "All (policy model)",
        "Data Type": "Prompt + chosen + rejected",
        "Epochs": 3,
        "Key Hyperparams": "lr=1e-5, beta=0.1",
        "Strengths": "Improves response quality/alignment",
        "Weaknesses": "Requires preference pairs",
        "Use When": "Aligning model to human preferences",
    },
    {
        "Method": "Full Fine-Tuning",
        "Params Updated": "All (117M)",
        "Data Type": "Mixed (text + instruction)",
        "Epochs": 3,
        "Key Hyperparams": "lr=2e-5, batch=4",
        "Strengths": "Maximum flexibility",
        "Weaknesses": "High compute, catastrophic forgetting risk",
        "Use When": "Sufficient compute + large high-quality dataset",
    },
    {
        "Method": "LoRA",
        "Params Updated": "~0.5% (LoRA adapters)",
        "Data Type": "Mixed (text + instruction)",
        "Epochs": 3,
        "Key Hyperparams": "r=16, alpha=32, lr=3e-4",
        "Strengths": "Efficient, composable adapters",
        "Weaknesses": "Lower peak capacity than full FT",
        "Use When": "Limited compute; multiple task adapters",
    },
    {
        "Method": "QLoRA",
        "Params Updated": "~0.5% on 4-bit base",
        "Data Type": "Mixed (text + instruction)",
        "Epochs": 3,
        "Key Hyperparams": "r=16, nf4 quantization",
        "Strengths": "Fits large models on consumer GPU",
        "Weaknesses": "Requires CUDA + bitsandbytes",
        "Use When": "Fine-tuning 7B+ models on single GPU",
    },
]

print(f"{'Method':<25} {'Params Updated':<25} {'Use When'}")
print("-" * 90)
for row in summary:
    print(f"{row['Method']:<25} {row['Params Updated']:<25} {row['Use When']}")
"""))

cells.append(code("""\
# Decision flowchart summary
print(\"\"\"
Fine-Tuning Method Selection Guide:
====================================

Do you have preference (chosen/rejected) data?
  YES → DPO (or RLHF if you need a reward model)
  NO  → Continue below

Are you adapting a base model to follow instructions?
  YES → SFT first, then optionally DPO

Do you have compute constraints?
  YES (limited GPU memory) → LoRA or QLoRA
  NO  (full GPU available) → Full Fine-Tuning

Do you want a domain-specific language model (not instruction-following)?
  YES → Continued Pretraining
  NO  → SFT/LoRA/DPO

Typical production pipeline:
  1. Continued Pretraining on domain corpus
  2. SFT on instruction pairs
  3. DPO with RLHF preference data
\"\"\")
"""))

cells.append(md("""\
## Summary

This notebook demonstrated all six fine-tuning approaches on **The Everglades Cipher**
corpus using GPT-2 as the base model. Key takeaways:

1. **Continued pretraining** teaches the model *what* is in the corpus — it learns the
   novel's vocabulary, character names, and writing style, but cannot answer questions.

2. **SFT** teaches the model *how to respond* — it learns the instruction-following format
   and can produce relevant answers about the novel.

3. **DPO** improves *response quality* by training the model to prefer better answers over
   worse ones. It requires paired preference data.

4. **Full fine-tuning** is powerful but expensive; LoRA and QLoRA are practical alternatives
   that retain most of the performance at a fraction of the compute cost.

5. For real production use with larger models (7B+), the typical pipeline is:
   continued pretraining → SFT → DPO, using LoRA/QLoRA throughout.

### The Everglades Cipher: Novel Summary
*The Everglades Cipher* is a noir-historical detective novel following Miami PI Jake Malone
as he searches for a missing elderly bookseller (Arturo Vasquez-Cortez) who spent 34 years
decoding a medieval Templar cipher. The trail leads from Little Havana to the Everglades
to a Caribbean sea cave holding 212 Templar documents — including letters from Philip IV
of France that reframe the history of the Templar dissolution.
"""))

# Write the notebook
notebook = make_notebook(cells)
with open(NOTEBOOK_PATH, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)
    f.write("\n")
print(f"Written: {NOTEBOOK_PATH}")
print(f"  Size: {os.path.getsize(NOTEBOOK_PATH) / 1024:.1f} KB")
print(f"  Cells: {len(cells)}")

# ---------------------------------------------------------------------------
# EXERCISE NOTEBOOK — same structure, TODO stubs for key implementations
# ---------------------------------------------------------------------------

_counter[0] = 0  # reset ID counter

exercise_cells = []

exercise_cells.append(md("""\
# Fine-Tuning in Action: Exercise Notebook

This is the exercise version of the fine-tuning notebook.
Complete the `# TODO` stubs to implement each fine-tuning technique.

Refer to the reference notebook (`fine-tuning-in-action.ipynb`) if you get stuck.
"""))

exercise_cells.append(md("## Section 0: Setup"))
exercise_cells.append(code("""\
import os, json, glob
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from transformers import DataCollatorForLanguageModeling
from datasets import Dataset

ROOT = Path(".")
DATA_DIR = ROOT / "data"
CONTENT_DIR = ROOT / "content" / "the-everglades-cipher"
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")
"""))

exercise_cells.append(md("## Section 1: Load Data"))
exercise_cells.append(code("""\
def load_jsonl(path):
    # TODO: implement this function to load a JSONL file
    # Return a list of dicts, one per line
    pass

pretraining = load_jsonl(DATA_DIR / "pretraining_chunks.jsonl")
instructions = load_jsonl(DATA_DIR / "instruction_prompts.jsonl")
rlhf_pairs = load_jsonl(DATA_DIR / "rlhf_pairs.jsonl")

print(f"Pretraining chunks: {len(pretraining)}")
print(f"Instructions: {len(instructions)}")
print(f"RLHF pairs: {len(rlhf_pairs)}")
"""))

exercise_cells.append(md("## Section 2: Tokenization Helper"))
exercise_cells.append(code("""\
def tokenize_for_lm(examples, tokenizer, max_length=512):
    # TODO: tokenize examples["text"] with truncation and padding
    # Set labels = input_ids (for causal LM training)
    pass
"""))

exercise_cells.append(md("## Section 3: Continued Pretraining"))
exercise_cells.append(code("""\
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# TODO: Load gpt2 model and tokenizer
# Set pad_token = eos_token
model_pt = None
tokenizer_pt = None

# TODO: Build HuggingFace Dataset from pretraining chunks
# Map tokenize_for_lm over it
# Split 80/20 train/eval

# TODO: Define TrainingArguments (2 epochs, batch size 4)

# TODO: Create Trainer and call .train()
"""))

exercise_cells.append(md("## Section 4: SFT"))
exercise_cells.append(code("""\
def format_instruction(item):
    # TODO: format as ### Instruction: ... ### Response: ...
    pass

# TODO: Load gpt2, format instruction data, tokenize, train
"""))

exercise_cells.append(md("## Section 5: DPO"))
exercise_cells.append(code("""\
# TODO: Import DPOTrainer from trl
# TODO: Load rlhf_pairs as Dataset
# TODO: Configure DPOConfig with beta=0.1
# TODO: Train with DPOTrainer
"""))

exercise_cells.append(md("## Section 6: LoRA"))
exercise_cells.append(code("""\
from peft import get_peft_model, LoraConfig, TaskType

# TODO: Create LoraConfig with r=16, lora_alpha=32, target_modules=["c_attn"]
# TODO: Apply to GPT-2 with get_peft_model
# TODO: Print trainable parameters
# TODO: Train with Trainer

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,
    lora_alpha=32,
    # TODO: fill in remaining parameters
)
"""))

exercise_cells.append(md("## Section 7: Evaluation"))
exercise_cells.append(code("""\
QUESTIONS = [
    {"level": "Easy", "q": "What is the name of Jake Malone's houseboat?"},
    {"level": "Medium", "q": "What is the Codex Almeida and why is it significant?"},
    {"level": "Hard", "q": "What do the Philip IV letters reveal about the Templar dissolution?"},
    {"level": "Expert", "q": "Analyze the parallel between Diego de Almeida and Arturo Vasquez-Cortez."},
]

def evaluate_model(model, tokenizer, questions, max_new_tokens=100):
    # TODO: for each question, format as ### Instruction: ### Response:
    # generate a response and return the results
    pass

# TODO: evaluate your trained SFT model
# TODO: compare base gpt2 vs fine-tuned
"""))

exercise_notebook = make_notebook(exercise_cells)
with open(EXERCISE_PATH, "w", encoding="utf-8") as f:
    json.dump(exercise_notebook, f, indent=1, ensure_ascii=False)
    f.write("\n")
print(f"\nWritten: {EXERCISE_PATH}")
print(f"  Size: {os.path.getsize(EXERCISE_PATH) / 1024:.1f} KB")
print(f"  Cells: {len(exercise_cells)}")
