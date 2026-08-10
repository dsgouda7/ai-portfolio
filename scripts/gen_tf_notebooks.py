"""
Transform PyTorch notebooks to TensorFlow/Keras equivalents.
Run: python scripts/gen_tf_notebooks.py
"""
import json, re, uuid, os, shutil
from pathlib import Path

ROOT = Path(r"c:\r\ai-portfolio\learning\genai")

# ---------------------------------------------------------------------------
# Source → destination mapping
# ---------------------------------------------------------------------------
NOTEBOOKS = [
    (ROOT / "11-llm-evaluation" / "01-llm-evaluation-metrics-and-benchmarks-pytorch.ipynb",
     ROOT / "11-llm-evaluation" / "01-llm-evaluation-metrics-and-benchmarks.ipynb"),
    (ROOT / "11-llm-evaluation" / "02-llm-as-judge-safety-and-pipeline-pytorch.ipynb",
     ROOT / "11-llm-evaluation" / "02-llm-as-judge-safety-and-pipeline.ipynb"),
    (ROOT / "11-llm-evaluation" / "03-hallucination-detection-pytorch.ipynb",
     ROOT / "11-llm-evaluation" / "03-hallucination-detection.ipynb"),
    (ROOT / "11-llm-evaluation" / "04-calibration-and-confidence-pytorch.ipynb",
     ROOT / "11-llm-evaluation" / "04-calibration-and-confidence.ipynb"),
]

# ---------------------------------------------------------------------------
# Code substitution rules (applied to code-cell source only)
# ---------------------------------------------------------------------------

def tf_embed_function():
    """Return the TF embedding helper as a list of source lines."""
    return (
        "import tensorflow as tf\n"
        "from transformers import TFAutoModel, AutoTokenizer\n"
        "\n"
        "# Load TF-based sentence embedding model\n"
        "print('Loading TF embedding model (all-MiniLM-L6-v2)...')\n"
        "_tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')\n"
        "_tf_embed_model = TFAutoModel.from_pretrained(\n"
        "    'sentence-transformers/all-MiniLM-L6-v2', from_pt=True)\n"
        "print('[OK] TF embedding model loaded')\n"
        "\n"
        "def _tf_encode(texts, batch_size=32):\n"
        "    \"\"\"Encode texts to L2-normalized CLS embeddings using TF.\"\"\"\n"
        "    import numpy as np\n"
        "    if isinstance(texts, str):\n"
        "        texts = [texts]\n"
        "    all_embs = []\n"
        "    for i in range(0, len(texts), batch_size):\n"
        "        batch = texts[i:i + batch_size]\n"
        "        enc = _tokenizer(\n"
        "            batch, return_tensors='tf', padding=True,\n"
        "            truncation=True, max_length=128)\n"
        "        out = _tf_embed_model(**enc, training=False)\n"
        "        emb = out.last_hidden_state[:, 0, :]  # CLS pooling\n"
        "        emb = tf.math.l2_normalize(emb, axis=-1)\n"
        "        all_embs.append(emb.numpy())\n"
        "    return np.vstack(all_embs)\n"
    )


def apply_code_substitutions(src: str, notebook_name: str) -> str:
    """Apply TF substitutions to a code-cell source string."""

    # ----------------------------------------------------------------
    # 1. Install-cell tweaks: replace sentence-transformers with tensorflow
    # ----------------------------------------------------------------
    # Replace sentence_transformers install with tensorflow
    src = re.sub(
        r'["\']sentence[_-]transformers["\'],\s*["\']sentence-transformers["\']',
        '"tensorflow", "tensorflow"),\n    ("transformers", "transformers"',
        src)
    # Also handle plain install mentions in subprocess calls
    src = re.sub(
        r'pip install sentence[-_]transformers',
        'pip install tensorflow transformers',
        src)

    # ----------------------------------------------------------------
    # 2. Import-level substitutions
    # ----------------------------------------------------------------
    # Remove sentence_transformers SentenceTransformer import
    src = re.sub(
        r'from sentence_transformers import SentenceTransformer\n?',
        '', src)
    # Remove CrossEncoder import from sentence_transformers (keep the actual CrossEncoder usage)
    # We keep CrossEncoder as-is (uses PyTorch internally but is a utility)

    # Replace torch imports (for eval notebooks)
    src = src.replace('import torch\n', 'import tensorflow as tf\n')
    src = src.replace("import torch", "import tensorflow as tf")

    # Replace GPT2LMHeadModel with TFGPT2LMHeadModel (keep tokenizer)
    src = src.replace(
        'from transformers import GPT2LMHeadModel, GPT2TokenizerFast',
        'from transformers import TFGPT2LMHeadModel, GPT2TokenizerFast')
    # Replace bare GPT2LMHeadModel only where not already prefixed with TF
    src = re.sub(r'(?<!TF)GPT2LMHeadModel', 'TFGPT2LMHeadModel', src)

    # ----------------------------------------------------------------
    # 3. Model loading and seeding
    # ----------------------------------------------------------------
    # SentenceTransformer("all-MiniLM-L6-v2") → use _tf_encode
    src = re.sub(
        r'semantic_model\s*=\s*SentenceTransformer\(["\']all-MiniLM-L6-v2["\']\)',
        '# TF embedding model loaded above; use _tf_encode() for all encoding\n'
        'semantic_model_encode = _tf_encode  # alias for compatibility',
        src)
    src = re.sub(
        r'(?<!["\'])SentenceTransformer\(["\']all-MiniLM-L6-v2["\']\)',
        '_TFEmbedWrapper()',  # placeholder; handled by model-load cell rewrite
        src)

    # semantic_model.encode(...) → _tf_encode(...)
    src = src.replace('semantic_model.encode(', '_tf_encode(')

    # _embed_model = SentenceTransformer(...) for judge/attribution notebooks
    src = re.sub(
        r"_embed_model\s*=\s*SentenceTransformer\('[^']+'\)",
        "# _embed_model replaced by TF-based encoder\n"
        "_embed_model = None  # use _tf_encode() instead",
        src)
    # embs = _embed_model.encode([a, b], ...) → use _tf_encode
    src = re.sub(
        r'embs\s*=\s*_embed_model\.encode\(\[a,\s*b\],\s*convert_to_tensor=True\)',
        'embs_np = _tf_encode([a, b])\n'
        'embs = tf.constant(embs_np)',
        src)
    # st_util.pytorch_cos_sim(embs[0], embs[1])
    src = re.sub(
        r'float\(st_util\.pytorch_cos_sim\(embs\[0\],\s*embs\[1\]\)\)',
        'float(tf.reduce_sum(embs[0] * embs[1]).numpy())',
        src)
    # from sentence_transformers import SentenceTransformer, util as st_util
    src = re.sub(
        r'from sentence_transformers import SentenceTransformer,\s*util\s*as\s*st_util\n?',
        'from sentence_transformers import CrossEncoder  # CrossEncoder kept for reranking\n',
        src)
    # from sentence_transformers import SentenceTransformer, util as st_util (variant)
    src = re.sub(
        r'from sentence_transformers import SentenceTransformer\n?',
        '',
        src)

    # ----------------------------------------------------------------
    # 4. Torch-specific replacements in eval notebooks
    # ----------------------------------------------------------------
    # return_tensors='pt' → 'tf'
    src = src.replace("return_tensors='pt'", "return_tensors='tf'")
    src = src.replace('return_tensors="pt"', 'return_tensors="tf"')

    # torch.no_grad() → tf equivalent (we just remove the context manager)
    src = src.replace('with torch.no_grad():', 'with tf.device("/CPU:0"):  # TF inference')
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    src = re.sub(
        r"device\s*=\s*'cuda'\s+if\s+torch\.cuda\.is_available\(\)\s+else\s+'cpu'",
        "device = '/CPU:0'  # TF uses device strings",
        src)
    # .to(device) → (no-op for TF)
    src = re.sub(r'\.to\(device\)', '', src)
    # model.eval() → (no-op for TF)
    src = re.sub(r'\bbase_model\.eval\(\)\n', '', src)
    # out = model(**enc, labels=...) → TF-style
    # The perplexity function gets rewritten at cell level

    # torch.cuda.is_available → tf.config.list_physical_devices
    src = src.replace(
        'torch.cuda.is_available()',
        'len(tf.config.list_physical_devices("GPU")) > 0')

    # np.random.seed first, then torch.manual_seed → tf.random.set_seed
    src = src.replace(
        'torch.manual_seed(42)',
        'tf.random.set_seed(42)')
    src = re.sub(r'torch\.manual_seed\((\d+)\)', r'tf.random.set_seed(\1)', src)

    # ----------------------------------------------------------------
    # 5. Rewrite PyTorch perplexity / MCQ logic to TF
    # ----------------------------------------------------------------
    # Replace the torch-based perplexity function body with TF equivalent
    # Pattern: out = model(**enc, labels=enc['input_ids']) / out.loss.item()
    src = re.sub(
        r"out\s*=\s*model\(\*\*enc,\s*labels=enc\['input_ids'\]\)\s*\n\s*return\s*math\.exp\(out\.loss\.item\(\)\)",
        (
            "input_ids = enc['input_ids']\n"
            "    outputs = model(input_ids, training=False)\n"
            "    logits = outputs.logits\n"
            "    shift_logits = logits[:, :-1, :]\n"
            "    shift_labels = input_ids[:, 1:]\n"
            "    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(\n"
            "        labels=tf.cast(tf.reshape(shift_labels, [-1]), tf.int32),\n"
            "        logits=tf.reshape(shift_logits, [-1, tf.shape(shift_logits)[-1]])\n"
            "    ))\n"
            "    return float(tf.exp(loss).numpy())"
        ),
        src)
    # MCQ: loss = model(**enc, labels=enc['input_ids']).loss.item()
    src = re.sub(
        r"loss\s*=\s*model\(\*\*enc,\s*labels=enc\['input_ids'\]\)\.loss\.item\(\)",
        (
            "input_ids = enc['input_ids']\n"
            "            outputs = model(input_ids, training=False)\n"
            "            logits = outputs.logits\n"
            "            shift_logits = logits[:, :-1, :]\n"
            "            shift_labels = input_ids[:, 1:]\n"
            "            loss = float(tf.reduce_mean(\n"
            "                tf.nn.sparse_softmax_cross_entropy_with_logits(\n"
            "                    labels=tf.cast(tf.reshape(shift_labels, [-1]), tf.int32),\n"
            "                    logits=tf.reshape(shift_logits, [-1, tf.shape(shift_logits)[-1]])\n"
            "                )\n"
            "            ).numpy())"
        ),
        src)
    # Remove .loss.item() fallthrough
    src = re.sub(r'\.loss\.item\(\)', '.numpy()', src)

    # ----------------------------------------------------------------
    # 7. sentence_transformers in HuggingFaceEmbeddings — keep as-is
    # ----------------------------------------------------------------
    # HuggingFaceEmbeddings is fine to keep (LangChain wraps ST)

    return src


def apply_markdown_substitutions(src: str) -> str:
    """Apply TF substitutions to markdown cell source."""
    # Replace framework references
    replacements = [
        ("PyTorch", "TensorFlow"),
        ("pytorch", "tensorflow"),
        ("torch", "TensorFlow"),
        ("`torch`", "`tensorflow`"),
        ("AutoModel", "TFAutoModel"),
        ("`AutoModel`", "`TFAutoModel`"),
        # Keep HuggingFace references as-is
        # Fix perplexity references
        ("GPT2LMHeadModel", "TFGPT2LMHeadModel"),
    ]
    for old, new in replacements:
        src = src.replace(old, new)
    return src


def make_tf_install_cell_source(orig_src: str, notebook_name: str) -> str:
    """Rewrite install cell to use tensorflow instead of sentence-transformers."""
    # Common install pattern in these notebooks
    if 'sentence_transformers' in orig_src or 'sentence-transformers' in orig_src:
        # Replace sentence-transformers with tensorflow + transformers
        orig_src = re.sub(
            r'\(\s*["\']sentence_transformers["\']\s*,\s*["\']sentence-transformers["\']\s*\)',
            '("tensorflow", "tensorflow"),\n    ("transformers", "transformers")',
            orig_src)
        orig_src = re.sub(
            r'\(["\']sentence[-_]transformers["\'],\s*["\']sentence[-_]transformers["\']\)',
            '("tensorflow", "tensorflow"),\n    ("transformers", "transformers")',
            orig_src)
    return orig_src


def new_cell_id():
    return uuid.uuid4().hex[:8]


def transform_notebook(src_path: Path, dst_path: Path):
    """Load source notebook, apply TF substitutions, save to dst_path."""
    with open(src_path, encoding='utf-8') as f:
        nb = json.load(f)

    notebook_name = src_path.stem

    new_cells = []

    # Track whether we've inserted the TF embedding loader
    inserted_tf_loader = False

    for cell in nb['cells']:
        new_cell = dict(cell)  # shallow copy
        new_cell['id'] = new_cell_id()  # fresh IDs
        new_cell['outputs'] = [] if cell['cell_type'] == 'code' else []
        if cell['cell_type'] == 'code':
            new_cell['execution_count'] = None

        src_lines = cell.get('source', [])
        src_str = ''.join(src_lines)

        if cell['cell_type'] == 'code':
            # Check if this is the main imports/setup cell that loads SentenceTransformer
            is_embed_load_cell = bool(re.search(
                r'SentenceTransformer\s*\(\s*["\']all-MiniLM', src_str))

            if is_embed_load_cell and not inserted_tf_loader:
                # Inject TF embedding loader
                new_src = tf_embed_function()
                inserted_tf_loader = True
                new_cell['source'] = new_src.splitlines(keepends=True)
                # Fix last line (no trailing newline)
                if new_cell['source'] and new_cell['source'][-1].endswith('\n'):
                    new_cell['source'][-1] = new_cell['source'][-1].rstrip('\n')
            else:
                new_src = apply_code_substitutions(src_str, notebook_name)
                new_cell['source'] = new_src.splitlines(keepends=True)
                if new_cell['source'] and new_cell['source'][-1].endswith('\n'):
                    new_cell['source'][-1] = new_cell['source'][-1].rstrip('\n')

        elif cell['cell_type'] == 'markdown':
            new_src = apply_markdown_substitutions(src_str)
            new_cell['source'] = new_src.splitlines(keepends=True)
            if new_cell['source'] and new_cell['source'][-1].endswith('\n'):
                new_cell['source'][-1] = new_cell['source'][-1].rstrip('\n')

        new_cells.append(new_cell)

    nb['cells'] = new_cells
    nb['nbformat'] = 4
    nb['nbformat_minor'] = 5
    nb['metadata'] = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.11.0"
        }
    }

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    with open(dst_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
        f.write('\n')

    print(f"[OK] {dst_path.name}  ({len(new_cells)} cells)")


def main():
    for src, dst in NOTEBOOKS:
        if not src.exists():
            print(f"[SKIP] source not found: {src}")
            continue
        transform_notebook(src, dst)
        print(f"      written → {dst}")

    print("\nAll notebooks generated.")


if __name__ == '__main__':
    main()
