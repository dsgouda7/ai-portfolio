"""Build the PyTorch Transformers notebook (transformers.ipynb)."""

import json
import pathlib
import uuid

NB_PATH = pathlib.Path(
    r"c:\repos\ai-portfolio\learning\genai\02-transformers\transformers.ipynb"
)


def nid():
    return uuid.uuid4().hex[:8]


def md(src):
    return {"cell_type": "markdown", "id": nid(), "metadata": {}, "source": src}


def code(src):
    return {
        "cell_type": "code",
        "id": nid(),
        "metadata": {},
        "outputs": [],
        "execution_count": None,
        "source": src,
    }


cells = []

# ═══════════════════════════════════════════════════════════════════════════════
# PART 1-3 : Title, Imports, Vocab, Viz, Tokeniser, PE, RoPE
# ═══════════════════════════════════════════════════════════════════════════════

cells.append(
    md(
        "# Transformers from the Ground Up\n\n"
        "## Building Intuition One Step at a Time\n\n"
        "This notebook builds the **complete mental model for the Transformer architecture** from "
        "first principles - starting with a tiny 3-dimensional semantic space that you can visualise, "
        "rotate, and reason about concretely.\n\n"
        "Every concept is demonstrated on the same running example:\n\n"
        '> **"the cat sat on the mat"**\n\n'
        "| Step | Concept | Key Idea |\n"
        "| ---- | ------- | -------- |\n"
        "| 1  | Vocabulary + 3D Embeddings     | Words as points in semantic space |\n"
        "| 2  | The Ordering Problem           | Why bags of words lose meaning |\n"
        "| 3  | Sinusoidal PE                  | Adding position with sine waves |\n"
        "| 4  | RoPE                           | Rotating Q/K vectors for relative position |\n"
        "| 5  | Q, K, V + Attention            | Soft dictionary lookup |\n"
        "| 6  | Multi-Head Attention           | Parallel attention heads |\n"
        "| 7  | Feed-Forward + Layer Norm      | Per-token transformation + stabilisation |\n"
        "| 8  | Full Transformer Block         | All components assembled |\n"
        "| 9  | Mini Language Model            | End-to-end training from scratch |\n"
        "| 10 | W_V as Relevance Filter        | What each token contributes to the blend |\n"
        "| 11 | Causal Triangle                | Layer stacking and last-position richness |\n"
        "| 12 | Encoder Architecture           | Bidirectional attention - mask=None |\n"
        "| 13 | Cross-Attention                | Q from decoder, K/V from encoder |\n"
        "| 14 | Encoder-Decoder                | Reversal task with cross-attention |\n"
        "| 15 | Architecture Comparison        | Reader, Writer, Translator side by side |\n"
        "| 16 | GPT-2 Internals                | A real model, cracked open |\n"
    )
)

cells.append(
    code(
        "# \u2500\u2500 Install dependencies (run once) \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n"
        "import subprocess, sys\n\n"
        "required = [\n"
        '    ("numpy",        "numpy"),\n'
        '    ("matplotlib",   "matplotlib"),\n'
        '    ("torch",        "torch"),\n'
        '    ("seaborn",      "seaborn"),\n'
        '    ("plotly",       "plotly"),\n'
        '    ("transformers", "transformers"),\n'
        "]\n"
        "for imp, pkg in required:\n"
        "    try:\n"
        "        __import__(imp)\n"
        '        print(f"  ok  {pkg}")\n'
        "    except ImportError:\n"
        '        print(f"  installing {pkg}...")\n'
        '        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "-q"])\n'
        '        print(f"  done {pkg}")\n'
    )
)

cells.append(
    code(
        "# \u2500\u2500 Imports \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n"
        "import numpy as np\n"
        "import matplotlib.pyplot as plt\n"
        "from mpl_toolkits.mplot3d import Axes3D\n"
        "import math, warnings\n"
        "import torch\n"
        "import torch.nn as nn\n"
        "import torch.nn.functional as F\n"
        "import seaborn as sns\n"
        "from IPython.display import HTML, display\n"
        "from matplotlib.animation import FuncAnimation\n\n"
        "warnings.filterwarnings('ignore')\n\n"
        "try:\n"
        "    import plotly.graph_objects as go\n"
        "    HAS_PLOTLY = True\n"
        "except ImportError:\n"
        "    HAS_PLOTLY = False\n\n"
        'plt.rcParams.update({"figure.dpi": 100, "figure.facecolor": "white"})\n'
        'print(f"torch   {torch.__version__}")\n'
        'print(f"numpy   {np.__version__}")\n'
    )
)

cells.append(
    md(
        "---\n\n"
        "## Part 1 - Our Mini Universe: Vocabulary & Embeddings\n\n"
        "Before anything else, we need a way to represent words as numbers. This is the job of an "
        "**embedding**.\n\n"
        "In a real model (e.g. GPT-2), each word lives in a 768-dimensional space. Instead we build "
        "a **3-dimensional** semantic space where each axis captures a meaningful property:\n\n"
        "| Axis | Meaning | Low (0) | High (1) |\n"
        "| ---- | ------- | ------- | -------- |\n"
        "| **dim 0** | Concreteness | abstract (articles) | physical objects (cat, mat) |\n"
        "| **dim 1** | Animacy | inanimate (mat, fence) | living beings (cat, dog) |\n"
        "| **dim 2** | Dynamism | static (mat, fence) | action words (ran, jumped) |\n"
    )
)

cells.append(
    code(
        "# \u2500\u2500 Vocabulary \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n"
        "VOCAB = {\n"
        '    "<PAD>": 0, "<BOS>": 1, "<EOS>": 2,\n'
        '    "the": 3, "a": 4, "cat": 5, "dog": 6,\n'
        '    "mat": 7, "fence": 8, "sat": 9, "ran": 10,\n'
        '    "jumped": 11, "on": 12, "over": 13, "big": 14,\n'
        "}\n"
        "IDX2WORD = {v: k for k, v in VOCAB.items()}\n"
        "VOCAB_SIZE = len(VOCAB)\n\n"
        "# \u2500\u2500 3D Semantic Embeddings (Concreteness, Animacy, Dynamism) \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n"
        "E = {\n"
        '    "<PAD>": [0.00, 0.00, 0.00], "<BOS>": [0.08, 0.08, 0.15], "<EOS>": [0.08, 0.08, 0.15],\n'
        '    "the":   [0.05, 0.04, 0.08], "a":     [0.05, 0.04, 0.08],\n'
        '    "cat":   [0.91, 0.94, 0.38], "dog":   [0.88, 0.92, 0.55],\n'
        '    "mat":   [0.96, 0.04, 0.04], "fence": [0.93, 0.03, 0.03],\n'
        '    "sat":   [0.34, 0.18, 0.78], "ran":   [0.28, 0.12, 0.96],\n'
        '    "jumped":[0.30, 0.14, 0.98], "on":    [0.14, 0.04, 0.18],\n'
        '    "over":  [0.17, 0.04, 0.24], "big":   [0.44, 0.04, 0.09],\n'
        "}\n\n"
        "embedding_matrix = torch.tensor(\n"
        "    [E[IDX2WORD[i]] for i in range(VOCAB_SIZE)], dtype=torch.float32\n"
        ")\n\n"
        'SENTENCE = "the cat sat on the mat"\n'
        "TOKENS = SENTENCE.split()\n"
        "TOKEN_IDS = [VOCAB[w] for w in TOKENS]\n"
        "SEQ_LEN = len(TOKENS)\n\n"
        'print(f"Vocab size       : {VOCAB_SIZE}")\n'
        'print(f"Embedding shape  : {tuple(embedding_matrix.shape)}  (vocab x 3D)")\n'
        'print(f"Running sentence : {SENTENCE!r}")\n'
        'print(f"Token IDs        : {TOKEN_IDS}")\n'
        'print()\nprint("Embedding matrix -> Concreteness, Animacy, Dynamism:")\n'
        "for word, vec in list(E.items())[3:]:\n"
        '    print(f"  {word:<10} [{vec[0]:.2f}, {vec[1]:.2f}, {vec[2]:.2f}]")\n'
    )
)

cells.append(
    code(
        "# \u2500\u2500 Interactive 3D Vocabulary Visualisation \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n"
        "# Plotly = interactive (drag to rotate). Matplotlib = static fallback.\n\n"
        "CATEGORIES = {\n"
        '    "Article":        (["the", "a"],            "#636EFA"),\n'
        '    "Animate Noun":   (["cat", "dog"],           "#00CC96"),\n'
        '    "Inanimate Noun": (["mat", "fence"],         "#AB63FA"),\n'
        '    "Verb":           (["sat", "ran", "jumped"], "#EF553B"),\n'
        '    "Preposition":    (["on", "over"],           "#FFA15A"),\n'
        '    "Adjective":      (["big"],                  "#19D3F3"),\n'
        "}\n\n"
        "if HAS_PLOTLY:\n"
        "    fig = go.Figure()\n"
        "    for cat, (words, color) in CATEGORIES.items():\n"
        "        xs, ys, zs = zip(*[E[w] for w in words])\n"
        "        fig.add_trace(go.Scatter3d(x=xs, y=ys, z=zs, mode='markers+text', text=words,\n"
        "            textposition='top center', name=cat,\n"
        "            marker=dict(size=10, color=color, opacity=0.85, line=dict(color='white', width=1))))\n"
        "    sx, sy, sz = zip(*[E[w] for w in TOKENS])\n"
        "    fig.add_trace(go.Scatter3d(x=sx, y=sy, z=sz, mode='lines', name='Sentence path',\n"
        "        line=dict(color='gold', width=4, dash='dot')))\n"
        "    fig.update_layout(\n"
        "        title=dict(text='<b>3D Semantic Embedding Space</b> - drag to rotate', x=0.5),\n"
        "        scene=dict(xaxis_title='Concreteness', yaxis_title='Animacy', zaxis_title='Dynamism'),\n"
        "        width=820, height=560)\n"
        "    fig.show()\n"
        "else:\n"
        "    fig = plt.figure(figsize=(9, 7))\n"
        "    ax = fig.add_subplot(111, projection='3d')\n"
        "    cmap = {'Article': 'royalblue', 'Animate Noun': 'mediumseagreen',\n"
        "            'Inanimate Noun': 'mediumpurple', 'Verb': 'tomato',\n"
        "            'Preposition': 'darkorange', 'Adjective': 'deepskyblue'}\n"
        "    for cat, (words, _) in CATEGORIES.items():\n"
        "        xs, ys, zs = zip(*[E[w] for w in words])\n"
        "        ax.scatter(xs, ys, zs, s=90, label=cat, color=cmap[cat], alpha=0.9)\n"
        "        for w in words:\n"
        "            ax.text(E[w][0], E[w][1], E[w][2], f' {w}', fontsize=9)\n"
        "    sx, sy, sz = zip(*[E[w] for w in TOKENS])\n"
        "    ax.plot(sx, sy, sz, 'o--', color='gold', lw=2, label='Sentence path')\n"
        "    ax.set_xlabel('Concreteness'); ax.set_ylabel('Animacy'); ax.set_zlabel('Dynamism')\n"
        "    ax.set_title('3D Semantic Embedding Space'); ax.legend(fontsize=8)\n"
        "    plt.tight_layout(); plt.show()\n"
        "    print('Tip: pip install plotly for an interactive, rotatable version')\n"
    )
)

cells.append(
    md(
        "### Tokenisation\n\nA **tokeniser** converts a raw string into integer IDs the model can process. In our toy system, one word = one token. Production models use sub-word tokenisation (BPE / SentencePiece).\n"
    )
)

cells.append(
    code(
        "# \u2500\u2500 Tokeniser \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n"
        "def encode(text: str, add_bos: bool = False, add_eos: bool = False):\n"
        '    ids = [VOCAB.get(w, VOCAB["<PAD>"]) for w in text.lower().split()]\n'
        "    if add_bos:\n"
        '        ids = [VOCAB["<BOS>"]] + ids\n'
        "    if add_eos:\n"
        '        ids = ids + [VOCAB["<EOS>"]]\n'
        "    return ids\n\n\n"
        "def decode(ids):\n"
        '    return " ".join(IDX2WORD.get(i, "<?>") for i in ids)\n\n\n'
        'phrase = "the big cat jumped over the fence"\n'
        "enc = encode(phrase)\n"
        "dec = decode(enc)\n"
        'print(f"Input  : {phrase!r}")\n'
        'print(f"Encoded: {enc}")\n'
        'print(f"Decoded: {dec!r}")\n'
        "print()\n"
        "print('With BOS/EOS markers:')\n"
        "enc2 = encode(phrase, add_bos=True, add_eos=True)\n"
        'print(f"  {enc2}")\n'
        'print()\nprint("  -> One word = one token; BPE splits rare words in real models.")\n'
    )
)

cells.append(
    md(
        "---\n\n## Attention: First Contact\n\n"
        "Before positions, before $Q/K/V$ projections, before multi-head - the beating heart of the "
        "transformer is one simple idea:\n\n"
        "> **Every token looks at every other token and builds a weighted average of them, where "
        'the weights answer "how much do I care about you?"**\n\n'
        '**Running it on our sentence:** `"the cat sat on the mat"`. Query = `"cat"`. No positions, '
        "no learned projections - just raw dot products of the 3D semantic vectors.\n"
    )
)

cells.append(
    code(
        "# \u2500\u2500 Attention, step by step - freezing at each completed step \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n"
        "# Minimal attention: query = key = value = the raw embedding.\n\n"
        "emb_min = embedding_matrix[TOKEN_IDS].numpy()   # (S, 3)\n"
        "S_min = len(TOKENS)\n"
        'QUERY = "cat"\n'
        "qi = TOKENS.index(QUERY)\n\n"
        "scores_min = emb_min @ emb_min[qi]\n"
        "w_min = np.exp(scores_min - scores_min.max())\n"
        "w_min /= w_min.sum()\n"
        "output_min = (w_min[:, None] * emb_min).sum(0)\n\n"
        "key_x = np.arange(S_min)\n"
        "q_x = (S_min - 1) / 2.0\n"
        'dim_names = ["Concrete", "Animate", "Dynamic"]\n\n'
        "PH, REVEAL, HOLD = 4, 18, 12\n"
        "plen = REVEAL + HOLD\n"
        "TOTAL = PH * plen + 18\n\n\n"
        "def _phase(f):\n"
        "    if f >= PH * plen:\n"
        "        return PH - 1, 1.0, True\n"
        "    p = f // plen\n"
        "    loc = f % plen\n"
        "    return p, min(loc / REVEAL, 1.0), loc >= REVEAL\n\n\n"
        "fig = plt.figure(figsize=(12, 7))\n"
        "gs = plt.GridSpec(2, 2, height_ratios=[1.5, 1], hspace=0.5, wspace=0.25)\n"
        "ax_g = fig.add_subplot(gs[0, :])\n"
        "ax_w = fig.add_subplot(gs[1, 0])\n"
        "ax_o = fig.add_subplot(gs[1, 1])\n\n"
        "captions = [\n"
        '    \'Step 1 - pick a query token: "cat" asks "who matters to me?"\',\n'
        "    'Step 2 - score \"cat\" against every token by dot product',\n"
        "    'Step 3 - softmax turns scores into weights that sum to 1',\n"
        "    'Step 4 - output = weighted sum of the value vectors',\n"
        "]\n"
        "done_caps = [\n"
        "    'Step 1 - query selected', 'Step 2 - every token scored',\n"
        "    'Step 3 - weights sum to 1', 'Step 4 - context-aware vector for \"cat\" is ready',\n"
        "]\n\n\n"
        "def update(f):\n"
        "    p, t, done = _phase(f)\n"
        "    ax_g.clear(); ax_w.clear(); ax_o.clear()\n"
        "    ax_g.set_xlim(-1, S_min); ax_g.set_ylim(-0.6, 1.7); ax_g.axis('off')\n"
        "    if p >= 1:\n"
        "        wshow = (scores_min - scores_min.min()) / max((scores_min.max() - scores_min.min()), 1e-9)\n"
        "        reveal = t if p == 1 else 1.0\n"
        "        for j in range(S_min):\n"
        "            lw = 0.5 + 6 * wshow[j] * reveal\n"
        "            a = min(0.15 + 0.85 * wshow[j] * reveal, 1.0)\n"
        "            ax_g.plot([q_x, key_x[j]], [0, 1], color='#4c72b0', lw=lw, alpha=a, zorder=1)\n"
        "    for j, tok in enumerate(TOKENS):\n"
        "        ax_g.scatter(key_x[j], 1, s=520, color='#dddddd', edgecolor='#888', zorder=3)\n"
        "        ax_g.text(key_x[j], 1, tok, ha='center', va='center', fontsize=9, zorder=4)\n"
        "        if p >= 2:\n"
        "            ax_g.text(key_x[j], 1.32, f'{w_min[j]:.2f}', ha='center', fontsize=9, color='#c44e52', fontweight='bold')\n"
        "    qsize = 300 + 420 * (t if p == 0 else 1.0)\n"
        "    ax_g.scatter(q_x, 0, s=qsize, color='gold', edgecolor='#b8860b', zorder=5)\n"
        "    ax_g.text(q_x, 0, QUERY, ha='center', va='center', fontsize=10, fontweight='bold', zorder=6)\n"
        "    ax_g.text(q_x, -0.42, 'query', ha='center', fontsize=9, color='#b8860b')\n"
        "    ax_g.text((S_min-1)/2, 1.6, 'keys / values (every token)', ha='center', fontsize=9, color='#555')\n"
        "    ax_w.set_xlim(-0.6, S_min-0.4); ax_w.set_ylim(0, 1.05)\n"
        "    ax_w.set_xticks(key_x); ax_w.set_xticklabels(TOKENS, fontsize=8, rotation=20)\n"
        "    if p == 0:\n"
        "        ax_w.set_title('scores appear in step 2', fontsize=9, color='#999')\n"
        "    elif p == 1:\n"
        "        sc = (scores_min - scores_min.min()) / max(scores_min.max()-scores_min.min(), 1e-9)\n"
        "        ax_w.bar(key_x, sc * t, color='#4c72b0', alpha=0.85)\n"
        "        ax_w.set_title('Step 2 - raw dot-product scores', fontsize=10)\n"
        "        ax_w.set_ylabel('score (scaled)')\n"
        "    else:\n"
        "        sc = (scores_min-scores_min.min()) / max(scores_min.max()-scores_min.min(), 1e-9)\n"
        "        blend = (1-t)*sc + t*w_min if p == 2 else w_min\n"
        "        colors = ['gold' if j==w_min.argmax() else '#4c72b0' for j in range(S_min)]\n"
        "        ax_w.bar(key_x, blend, color=colors, alpha=0.85)\n"
        "        ax_w.set_title('Step 3 - softmax -> weights (sum=1)' if p==2 else 'Step 3 - attention weights', fontsize=10)\n"
        "        ax_w.set_ylabel('weight')\n"
        "    ax_o.set_ylim(0, 1.05); ax_o.set_xticks(range(3)); ax_o.set_xticklabels(dim_names, fontsize=8)\n"
        "    if p < 3:\n"
        "        ax_o.bar(range(3), [0,0,0], color='#55a868'); ax_o.set_title('output builds in step 4', fontsize=9, color='#999')\n"
        "    else:\n"
        "        frac=t*S_min; k=int(frac); part=frac-k\n"
        "        out_v=np.zeros(3)\n"
        "        for j in range(min(k, S_min)): out_v += w_min[j]*emb_min[j]\n"
        "        if k < S_min: out_v += part*w_min[k]*emb_min[k]\n"
        "        ax_o.bar(range(3), out_v, color='#55a868')\n"
        "        cur=TOKENS[min(k, S_min-1)]\n"
        "        ax_o.set_title(f'Step 4 - sum w*value (adding \"{cur}\")' if not done else 'Step 4 - context vector', fontsize=10)\n"
        "    fig.suptitle(done_caps[p] if done else captions[p], fontsize=12, fontweight='bold', color='#333')\n\n\n"
        "ani = FuncAnimation(fig, update, frames=TOTAL, interval=45, blit=False, repeat=True, repeat_delay=1200)\n"
        "plt.close(fig)\n"
        "print('Minimal attention: query = key = value = embedding, no positions, no projections.')\n"
        "top3 = w_min.argsort()[::-1][:3]\n"
        'print(""cat" attends most to: " + ", ".join(f"{TOKENS[j]} ({w_min[j]:.0%})" for j in top3))\n'
        "HTML(ani.to_jshtml(default_mode='loop'))\n"
    )
)

cells.append(
    md(
        "#### What just happened - and what's missing\n\n"
        '`"cat"` pulled most strongly toward **itself** and **`"dog"`** - its semantic neighbours. '
        "Attention found *meaning* without anyone hand-coding grammar.\n\n"
        "But look closely at what we **never used**: *position*. Query, key and value were the raw "
        'embeddings. Shuffle the sentence and `"cat"` keeps the exact same neighbours. '
        "**Attention, on its own, is position-blind.**\n"
    )
)

cells.append(
    md(
        "---\n\n## Part 2 - The Ordering Problem\n\n"
        "What happens if we just **sum or average** the token vectors?\n\n"
        '> "the cat sat on the mat"\n> "mat the on sat the cat" - shuffled nonsense\n\n'
        "Both sentences contain exactly the same words. Their mean-pooled embedding is **identical** - "
        "the model cannot tell them apart. Position information is load-bearing.\n"
    )
)

cells.append(
    code(
        "# \u2500\u2500 Bag of Words - loses all positional information \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n"
        "def bag_of_words(sentence: str) -> torch.Tensor:\n"
        '    """Mean-pool embeddings - loses all positional information."""\n'
        "    ids = torch.tensor(encode(sentence), dtype=torch.long)\n"
        "    return embedding_matrix[ids].mean(dim=0)\n\n\n"
        "sentences = [\n"
        '    "the cat sat on the mat",\n'
        '    "mat the on sat the cat",\n'
        '    "sat cat mat on the the",\n'
        "]\n"
        "print('Mean-pooled vectors (all contain the same words):')\n"
        "for s in sentences:\n"
        "    v = bag_of_words(s).numpy()\n"
        '    print(f"  {s!r:<42}  [{v[0]:.3f}, {v[1]:.3f}, {v[2]:.3f}]")\n\n'
        "all_same = all(\n"
        "    torch.allclose(bag_of_words(sentences[0]), bag_of_words(s))\n"
        "    for s in sentences[1:]\n"
        ")\n"
        'print(f"\\nAll three vectors identical: {all_same}")\n'
        "print()\n"
        "print('  -> A model with no positional encoding treats meaningful sentences and')\n"
        "print('     complete nonsense as the SAME input.  We need positional encoding.')\n"
    )
)

cells.append(
    md(
        "---\n\n## Part 3 - Positional Encoding\n\n"
        '### 3a. Sinusoidal PE (original Transformer, "Attention Is All You Need")\n\n'
        "$$PE_{(m,\\, 2i)} = \\sin\\!\\left(\\frac{m}{10000^{2i/d}}\\right)$$\n\n"
        "$$PE_{(m,\\, 2i+1)} = \\cos\\!\\left(\\frac{m}{10000^{2i/d}}\\right)$$\n\n"
        "Each dimension pair oscillates at a different frequency - a unique fingerprint for every position.\n"
    )
)

cells.append(
    code(
        "# \u2500\u2500 Sinusoidal Positional Encoding \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n"
        "def sinusoidal_pe(seq_len: int, d_model: int) -> torch.Tensor:\n"
        '    """Classic additive positional encoding (Vaswani et al. 2017).\n'
        "    Handles both even and odd d_model gracefully.\n"
        '    """\n'
        "    pe = np.zeros((seq_len, d_model), dtype=np.float32)\n"
        "    positions = np.arange(seq_len)[:, None].astype(np.float32)\n"
        "    dims = np.arange(0, d_model, 2).astype(np.float32)\n"
        "    freqs = 1.0 / (10000 ** (dims / d_model))\n"
        "    pe[:, 0::2] = np.sin(positions * freqs)\n"
        "    n_cos = pe[:, 1::2].shape[1]\n"
        "    pe[:, 1::2] = np.cos(positions * freqs[:n_cos])\n"
        "    return torch.tensor(pe)\n\n\n"
        "D_VIS = 16\n"
        "pe_matrix = sinusoidal_pe(SEQ_LEN, D_VIS)\n\n"
        "fig, axes = plt.subplots(1, 2, figsize=(13, 4))\n\n"
        "ax = axes[0]\n"
        "im = ax.imshow(pe_matrix.numpy(), aspect='auto', cmap='RdBu', vmin=-1, vmax=1)\n"
        "ax.set_xticks(range(D_VIS))\n"
        "ax.set_xticklabels([f'd{i}' for i in range(D_VIS)], fontsize=8, rotation=45)\n"
        "ax.set_yticks(range(SEQ_LEN)); ax.set_yticklabels(TOKENS, fontsize=10)\n"
        "ax.set_title('Sinusoidal PE - our sentence')\n"
        "ax.set_xlabel('Embedding dimension'); ax.set_ylabel('Token position')\n"
        "plt.colorbar(im, ax=ax)\n\n"
        "ax2 = axes[1]\n"
        "pe_long = sinusoidal_pe(50, D_VIS).numpy()\n"
        "for i in [0, 2, 6, 14]:\n"
        '    label = f\'dim {i} - {"fast" if i < 4 else "slow"}\'\n'
        "    ax2.plot(pe_long[:, i], label=label, lw=1.8)\n"
        "ax2.set_title('PE signal per dimension over 50 positions')\n"
        "ax2.set_xlabel('Token position'); ax2.set_ylabel('PE value')\n"
        "ax2.legend(fontsize=8); ax2.set_ylim(-1.1, 1.1)\n\n"
        "plt.suptitle('Sinusoidal Positional Encoding', fontsize=13, fontweight='bold')\n"
        "plt.tight_layout(); plt.show()\n\n"
        "emb_vectors = embedding_matrix[TOKEN_IDS]\n"
        "pe_3d = sinusoidal_pe(SEQ_LEN, 3)\n"
        "enriched = emb_vectors + pe_3d\n"
        "print('After adding 3D sinusoidal PE:')\n"
        "for i, w in enumerate(TOKENS):\n"
        "    def fmt(v_list):\n"
        "        return f'[{v_list[0]:+.3f}, {v_list[1]:+.3f}, {v_list[2]:+.3f}]'\n"
        "    print(f'[{i}] {w:<8}  orig={fmt(emb_vectors[i].tolist())}  "
        "pe={fmt(pe_3d[i].tolist())}  sum={fmt(enriched[i].tolist())}')\n"
    )
)

cells.append(
    md(
        "#### A nagging question before we move on\n\n"
        "Sinusoidal PE looks great in the heatmap - so why did the field move to RoPE? "
        "**Position is injected at the input, but attention projects through $W_Q$ first. "
        "If that projection smears the position signal, it was partially wasted.** "
        "RoPE fixes this by injecting position *after* the $W_Q$ projection, directly into "
        "the dot-product.\n"
    )
)

cells.append(
    code(
        "# \u2500\u2500 WHY not just keep sinusoidal PE? Watch the clean signal dilute \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n"
        "torch.manual_seed(0)\n"
        "d_demo = 16; seq_demo = 12\n\n"
        "pe_demo = sinusoidal_pe(seq_demo, d_demo)\n"
        "content = torch.randn(seq_demo, d_demo)\n"
        "W_Q_demo = torch.randn(d_demo, d_demo) * (1 / math.sqrt(d_demo))\n\n"
        "x_in = content + pe_demo\n"
        "q_proj = x_in @ W_Q_demo.T\n\n\n"
        "def pos_sim(mat):\n"
        "    mat = mat.detach().numpy() if isinstance(mat, torch.Tensor) else np.asarray(mat)\n"
        "    m = mat / (np.linalg.norm(mat, axis=-1, keepdims=True) + 1e-9)\n"
        "    return m @ m.T\n\n\n"
        "sim_pe = pos_sim(pe_demo)\n"
        "sim_q  = pos_sim(q_proj)\n\n"
        "fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))\n"
        "for ax_s, sim, title in [(axes[0], sim_pe, 'PURE sinusoidal PE\\nclean diagonal band = distance-aware'),\n"
        "                          (axes[1], sim_q,  'AFTER (content+PE) @ W_Q  (RANDOM W_Q)\\nstructure can wash out')]:\n"
        "    sns.heatmap(sim, ax=ax_s, cmap='RdBu_r', vmin=-1, vmax=1, square=True, cbar_kws={'label': 'cosine sim'})\n"
        "    ax_s.set_title(title); ax_s.set_xlabel('position'); ax_s.set_ylabel('position')\n"
        "plt.suptitle('Sinusoidal PE can dilute once it is summed with content and projected', fontsize=12, fontweight='bold')\n"
        "plt.tight_layout(); plt.show()\n\n\n"
        "def monotonicity(sim):\n"
        "    n = sim.shape[0]\n"
        "    closeness = np.array([[-abs(i-j) for j in range(n)] for i in range(n)])\n"
        "    return np.corrcoef(closeness.flatten(), sim.flatten())[0, 1]\n\n\n"
        "print('\"Closer positions = more similar\" correlation:')\n"
        "print(f'  Pure PE (at the input)         : {monotonicity(sim_pe):+.3f}   <- strong, clean')\n"
        "print(f'  After content + W_Q projection : {monotonicity(sim_q):+.3f}   <- weaker')\n"
        "print()\n"
        "print('The principled case for RoPE: position is injected AFTER projection,')\n"
        "print('so W_Q cannot dilute it, and the Q.K score depends only on (m-n) BY CONSTRUCTION.')\n"
    )
)

cells.append(
    md(
        "### 3b. RoPE - Rotary Positional Embeddings\n\n"
        "RoPE **rotates** the Query and Key vectors just before the dot-product, by an angle that "
        "depends on absolute position. The rotation cancels in a relative way - only the gap $m - n$ "
        "survives.\n\n"
        "$$\\theta_i = \\frac{1}{10000^{2i/d}}$$\n\n"
        "Token at position $m$ gets its $i$-th pair rotated by angle $m \\cdot \\theta_i$:\n\n"
        "$$\\text{RoPE}(x, m)_{2i:2i+2} = \\begin{pmatrix} \\cos(m\\theta_i) & -\\sin(m\\theta_i) \\\\ "
        "\\sin(m\\theta_i) & \\cos(m\\theta_i) \\end{pmatrix} \\begin{pmatrix} x_{2i} \\\\ x_{2i+1} \\end{pmatrix}$$\n"
    )
)

cells.append(
    code(
        "# \u2500\u2500 RoPE theta values - frequency decay visualisation \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n"
        "D_ROPE = 6\n"
        "half = D_ROPE // 2\n\n"
        "thetas = np.array([1.0 / (10000 ** (2 * i / D_ROPE)) for i in range(half)])\n"
        "print(f'theta values for d={D_ROPE}: {thetas}')\n\n"
        "steps = 50\n"
        "angles = np.outer(np.arange(steps), thetas)\n\n"
        "fig, axes = plt.subplots(1, 2, figsize=(13, 4))\n\n"
        "ax = axes[0]\n"
        "for i in range(half):\n"
        "    ax.plot(np.cos(angles[:, i]), label=f'Pair {i} (theta={thetas[i]:.4f})', lw=1.8)\n"
        "ax.set_xlabel('Token position'); ax.set_ylabel('cos(m * theta_i)')\n"
        "ax.set_title('RoPE: cosine component per dimension pair over 50 positions')\n"
        "ax.legend(fontsize=8); ax.set_ylim(-1.1, 1.1)\n\n"
        "ax2 = axes[1]\n"
        "for i in range(half):\n"
        "    ax2.plot(range(steps), angles[:, i] % (2 * np.pi), label=f'Pair {i}', lw=1.8)\n"
        "ax2.set_xlabel('Token position'); ax2.set_ylabel('angle mod 2pi')\n"
        "ax2.set_title('Accumulated rotation angle\\nPair 0 spins fastest, pair 2 slowest')\n"
        "ax2.legend(fontsize=8)\n\n"
        "plt.suptitle('RoPE theta values: high-frequency pairs capture local position', fontsize=10, fontweight='bold')\n"
        "plt.tight_layout(); plt.show()\n"
    )
)

cells.append(
    md(
        "### 3d. Building the RoPE animation the way you'd actually discover it\n\n"
        "Nobody arrives at a good visualisation in one shot. We build it in front of you, "
        "refinement by refinement, each step motivated by a genuine complaint about the one before.\n\n"
        "**Step 1 - the crudest possible picture:** just plot one pair (pair 0) as a clock dial.\n"
    )
)

cells.append(
    md(
        "#### Steps 2, 3 and 4 - all at once, because the complaints compound\n\n"
        "- **Step 2 (stacking)** - each of the 3 pairs gets its own disc at its own height.\n"
        "- **Step 3 (one tower per token)** - a column per position rather than a single position.\n"
        "- **Step 4 (labels)** - degree labels on each disc.\n"
    )
)

cells.append(
    code(
        "# \u2500\u2500 Attempt 1 - the crudest possible RoPE picture: one dial \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n"
        "th0 = 1.0 / (10000 ** (0 / 6))\n"
        "circle = np.linspace(0, 2 * np.pi, 100)\n\n"
        "fig, axes = plt.subplots(1, 4, figsize=(14, 3.4), subplot_kw={'aspect': 'equal'})\n"
        "for ax, m in zip(axes, range(4)):\n"
        "    ang = m * th0\n"
        "    ax.plot(np.cos(circle), np.sin(circle), 'lightgray', lw=1)\n"
        "    ax.arrow(0, 0, math.cos(ang)*0.85, math.sin(ang)*0.85,\n"
        "             head_width=0.1, head_length=0.08, fc='#4c72b0', ec='#4c72b0', lw=2)\n"
        "    ax.set_xlim(-1.3, 1.3); ax.set_ylim(-1.3, 1.3)\n"
        "    ax.set_title(f'position {m}\\nangle = {math.degrees(ang):.1f}°', fontsize=9)\n"
        "    ax.axis('off')\n\n"
        "plt.suptitle(f'Pair 0: angle grows by theta_0 = {th0:.4f} rad per step', fontsize=10, fontweight='bold')\n"
        "plt.tight_layout(); plt.show()\n"
        "print('Complaint: we only see one pair out of three, and only one position at a time.')\n"
    )
)

cells.append(
    code(
        "# \u2500\u2500 Animated RoPE - watch each pair rotate as position increases \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n"
        "RADII = [1.0, 0.7, 0.4]\n"
        "COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c']\n"
        "HEIGHTS = [1.2, 0.6, 0.0]\n"
        "N_FRAMES = 60\n\n"
        "fig3d = plt.figure(figsize=(7, 9))\n"
        "ax3d = fig3d.add_subplot(111, projection='3d')\n\n\n"
        "def rope_frame(step_m):\n"
        "    ax3d.clear()\n"
        "    circle_t = np.linspace(0, 2 * np.pi, 200)\n"
        "    for pair_i, (r, col, h) in enumerate(zip(RADII, COLORS, HEIGHTS)):\n"
        "        theta_i = 1.0 / (10000 ** (2 * pair_i / D_ROPE))\n"
        "        ang = step_m * theta_i\n"
        "        ax3d.plot(r*np.cos(circle_t), r*np.sin(circle_t), h, color=col, lw=0.5, alpha=0.3)\n"
        "        ax3d.quiver(0, 0, h, r*math.cos(ang), r*math.sin(ang), 0, color=col, lw=2, arrow_length_ratio=0.15)\n"
        "        ax3d.text(r*1.05, 0, h, f'Pair {pair_i}', fontsize=8, color=col)\n"
        "    ax3d.set_xlim(-1.3, 1.3); ax3d.set_ylim(-1.3, 1.3); ax3d.set_zlim(-0.3, 1.7)\n"
        "    ax3d.set_xlabel('cos', fontsize=7); ax3d.set_ylabel('sin', fontsize=7)\n"
        "    ax3d.set_title(f'RoPE rotation at position m={step_m}', fontsize=10)\n"
        "    ax3d.tick_params(labelsize=7)\n\n\n"
        "ani3d = FuncAnimation(fig3d, rope_frame, frames=N_FRAMES, interval=80, repeat=True)\n"
        "plt.close(fig3d)\n"
        "print('Each disc = one dimension pair. Arrow angle = m * theta_i.')\n"
        "print('Pair 0 (blue) spins fastest; pair 2 (green) barely moves.')\n"
        "HTML(ani3d.to_jshtml(default_mode='loop'))\n"
    )
)

cells.append(
    code(
        "# \u2500\u2500 RoPE implementation + relative-distance proof \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n\n"
        "def rope_rotate(x, m: int, thetas_arr: np.ndarray) -> np.ndarray:\n"
        '    """Rotate vector x (shape: d) at token position m using RoPE."""\n'
        "    x_rot = np.array(x, dtype=np.float32).copy()\n"
        "    for i, th in enumerate(thetas_arr):\n"
        "        angle = m * th\n"
        "        c, s = math.cos(angle), math.sin(angle)\n"
        "        a, b = x_rot[2*i], x_rot[2*i+1]\n"
        "        x_rot[2*i]   = a*c - b*s\n"
        "        x_rot[2*i+1] = a*s + b*c\n"
        "    return x_rot\n\n\n"
        "thetas_demo = np.array([1.0 / (10000 ** (2*i/D_ROPE)) for i in range(D_ROPE//2)])\n"
        "cat_emb = embedding_matrix[VOCAB['cat']].numpy()[:D_ROPE]\n"
        "mat_emb = embedding_matrix[VOCAB['mat']].numpy()[:D_ROPE]\n\n"
        "print('Relative-distance property of RoPE:')\n"
        "print('  cat at pos m, mat at pos n: Q_cat * K_mat depends only on (m-n)')\n"
        "print()\n\n"
        "for gap in [1, 2, 3]:\n"
        "    results = []\n"
        "    for base in [0, 1, 2, 3]:\n"
        "        q = rope_rotate(cat_emb, base, thetas_demo)\n"
        "        k = rope_rotate(mat_emb, base + gap, thetas_demo)\n"
        "        results.append(float(q @ k))\n"
        "    print(f'  gap={gap}: dot products = {[f\"{r:.4f}\" for r in results]}  '\n"
        "          f'(all equal -> {np.allclose(results, results[0], atol=1e-5)})')\n\n"
        "print()\n"
        "print('  -> RoPE guarantees relative position: the dot product depends ONLY on (m-n).')\n"
    )
)

# ═══════════════════════════════════════════════════════════════════════════════
# PART 4-7 : Q/K/V, Attention, MHA, FFN, TransformerBlock
# ═══════════════════════════════════════════════════════════════════════════════

cells.append(
    md(
        "---\n\n## Part 4 - Queries, Keys & Values\n\n"
        "The transformer's attention mechanism is a **soft dictionary lookup**.\n\n"
        "| Component | Intuition | Created by |\n"
        "| --------- | --------- | ---------- |\n"
        '| **Q** (Query) | "What am I looking for?" | $W_Q \\cdot x$ |\n'
        '| **K** (Key)   | "What do I advertise?"   | $W_K \\cdot x$ |\n'
        '| **V** (Value) | "What do I contribute?"  | $W_V \\cdot x$ |\n\n'
        "$W_Q$, $W_K$, $W_V$ are **learned** projection matrices. Each head has its own set.\n"
    )
)

cells.append(
    code(
        "# \u2500\u2500 Q / K / V projections in 3D space \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n"
        "torch.manual_seed(7)\n"
        "D_MODEL = 3\n\n"
        "W_Q = torch.randn(D_MODEL, D_MODEL) * 0.5\n"
        "W_K = torch.randn(D_MODEL, D_MODEL) * 0.5\n"
        "W_V = torch.randn(D_MODEL, D_MODEL) * 0.5\n\n"
        "embs = embedding_matrix[TOKEN_IDS]   # (6, 3)\n\n"
        "Q = embs @ W_Q.T   # (6, 3)\n"
        "K = embs @ W_K.T\n"
        "V = embs @ W_V.T\n\n"
        "fig = plt.figure(figsize=(15, 4))\n"
        "titles = ['Input Embeddings', 'Queries  (W_Q @ x)', 'Keys  (W_K @ x)', 'Values  (W_V @ x)']\n"
        "arrays = [embs, Q, K, V]\n"
        "colors = plt.cm.tab10(np.linspace(0, 0.6, SEQ_LEN))\n\n"
        "for idx, (title, arr) in enumerate(zip(titles, arrays)):\n"
        "    ax = fig.add_subplot(1, 4, idx + 1, projection='3d')\n"
        "    arr_np = arr.detach().numpy()\n"
        "    for j, (word, vec) in enumerate(zip(TOKENS, arr_np)):\n"
        "        ax.scatter(*vec, color=colors[j], s=70, zorder=5)\n"
        "        ax.text(*vec, f' {word}', fontsize=7, color=colors[j])\n"
        "    ax.set_title(title, fontsize=9, pad=6)\n"
        "    ax.set_xlabel('d0', fontsize=7); ax.set_ylabel('d1', fontsize=7); ax.set_zlabel('d2', fontsize=7)\n"
        "    ax.tick_params(labelsize=6)\n\n"
        "plt.suptitle('How W_Q, W_K, W_V rotate/stretch the embedding space', fontsize=11, y=1.01)\n"
        "plt.tight_layout(); plt.show()\n"
    )
)

cells.append(
    code(
        "# \u2500\u2500 Scaled Dot-Product Attention \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n"
        "def scaled_attention(Q_in: torch.Tensor, K_in: torch.Tensor, V_in: torch.Tensor, mask=None):\n"
        '    """Scaled dot-product attention. Returns (output, attn_weights).\n'
        "    Q_in, K_in, V_in: (seq_len, d_k)\n"
        '    """\n'
        "    d_k = Q_in.shape[-1]\n\n"
        "    scores = Q_in @ K_in.T / math.sqrt(d_k)\n"
        "    print(f'  Raw score matrix (QK^T / sqrt(d_k)):\\n  {scores.detach().numpy().round(3)}')\n\n"
        "    if mask is not None:\n"
        "        scores = scores.masked_fill(mask.bool(), float('-inf'))\n\n"
        "    attn_w = torch.softmax(scores, dim=-1)\n"
        "    out = attn_w @ V_in\n"
        "    return out, attn_w\n\n\n"
        "print('=== Step-by-step attention on our sentence ===')\n"
        "print(f'Q shape: {tuple(Q.shape)}  K shape: {tuple(K.shape)}  V shape: {tuple(V.shape)}\\n')\n\n"
        "out, attn_w = scaled_attention(Q, K, V)\n\n"
        "print(f'\\n  Attention weight matrix (row = query token, col = key token):')\n"
        "print(f'  Tokens: {TOKENS}')\n"
        "print(f'  {attn_w.detach().numpy().round(3)}')\n"
        "print(f'\\n  Output shape: {tuple(out.shape)}')\n"
    )
)

cells.append(
    code(
        "# \u2500\u2500 Attention heatmap visualisation \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n"
        "fig, axes = plt.subplots(1, 2, figsize=(13, 4))\n\n"
        "ax = axes[0]\n"
        "w = attn_w.detach().numpy()\n"
        "sns.heatmap(w, ax=ax, annot=True, fmt='.2f', cmap='Blues',\n"
        "            xticklabels=TOKENS, yticklabels=TOKENS, linewidths=0.5,\n"
        "            cbar_kws={'label': 'attention weight'})\n"
        "ax.set_title('Bidirectional Attention (encoder-style)')\n"
        "ax.set_xlabel('Key token'); ax.set_ylabel('Query token'); ax.tick_params(axis='x', rotation=30)\n\n"
        "ax2 = axes[1]\n"
        "S6 = SEQ_LEN\n"
        "causal_mask_vis = torch.triu(torch.ones(S6, S6, dtype=torch.bool), diagonal=1)\n"
        "_, attn_w_causal = scaled_attention(Q, K, V, mask=causal_mask_vis)\n"
        "wc = attn_w_causal.detach().numpy()\n"
        "sns.heatmap(wc, ax=ax2, annot=True, fmt='.2f', cmap='Oranges',\n"
        "            xticklabels=TOKENS, yticklabels=TOKENS, linewidths=0.5,\n"
        "            cbar_kws={'label': 'attention weight'})\n"
        "ax2.set_title('Causal Attention (decoder-style)\\nupper triangle masked to -inf')\n"
        "ax2.set_xlabel('Key token'); ax2.set_ylabel('Query token'); ax2.tick_params(axis='x', rotation=30)\n\n"
        "plt.suptitle('Bidirectional vs Causal Attention', fontsize=11, fontweight='bold')\n"
        "plt.tight_layout(); plt.show()\n"
    )
)

cells.append(
    md(
        "### Your turn - attention\n\nYou've watched attention; now drive it. Change `my_query` and **predict the top attention target before you run it**.\n"
    )
)

cells.append(
    code(
        "# \u2500\u2500 EXERCISE 1 - attention by hand \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n"
        "# Change `my_query` to any token in TOKENS and PREDICT its top attention target BEFORE running.\n"
        "# TOKENS = ['the', 'cat', 'sat', 'on', 'the', 'mat']\n"
        "my_query = 'cat'   # try 'sat', 'mat', 'on', ...\n\n"
        "qi = TOKENS.index(my_query)\n"
        "scores_ex = Q @ K.T / math.sqrt(Q.shape[-1])\n"
        "w_ex = torch.softmax(scores_ex, dim=-1).detach().numpy()[qi]\n\n"
        "print(f'\"{my_query}\" (position {qi}) attends most to:')\n"
        "for r in w_ex.argsort()[::-1][:3]:\n"
        "    print(f'   {TOKENS[r]:<8} (pos {r})  weight={w_ex[r]:.3f}')\n"
    )
)

cells.append(
    code(
        "# \u2500\u2500 RoPE applied to Q and K inside attention - step 1: rotate Q and K \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n"
        "def apply_rope_to_qk(Q_in, K_in, thetas_arr: np.ndarray) -> tuple:\n"
        '    """Apply RoPE to Q and K (seq_len x d). Returns (Q_rot, K_rot) as torch.Tensor."""\n'
        "    Q_np = Q_in.detach().numpy() if isinstance(Q_in, torch.Tensor) else np.asarray(Q_in)\n"
        "    K_np = K_in.detach().numpy() if isinstance(K_in, torch.Tensor) else np.asarray(K_in)\n"
        "    seq_len, d = Q_np.shape\n"
        "    n_pairs = d // 2\n"
        "    positions = np.arange(seq_len, dtype=np.float32)\n\n"
        "    def rope(x):\n"
        "        out_r = x.copy()\n"
        "        for i in range(n_pairs):\n"
        "            ang = positions * float(thetas_arr[i])\n"
        "            cos_a, sin_a = np.cos(ang), np.sin(ang)\n"
        "            a, b = x[:, 2*i], x[:, 2*i+1]\n"
        "            out_r[:, 2*i]   = a*cos_a - b*sin_a\n"
        "            out_r[:, 2*i+1] = a*sin_a + b*cos_a\n"
        "        return out_r\n\n"
        "    return torch.tensor(rope(Q_np)), torch.tensor(rope(K_np))\n\n\n"
        "th_vis = np.array([1.0 / (10000 ** (2*i/D_MODEL)) for i in range(D_MODEL // 2)])\n"
        "Q_raw, K_raw = Q, K\n"
        "Q_rot, K_rot = apply_rope_to_qk(Q_raw, K_raw, th_vis)\n\n"
        'print(f\'{"Token":<8}  {"Q_raw":>26}  {"Q_rotated (RoPE)":>26}  {"delta norm":>10}\')\n'
        "print('  ' + '-' * 74)\n"
        "for i, tok in enumerate(TOKENS):\n"
        "    qr, qn = Q_raw[i].numpy(), Q_rot[i].numpy()\n"
        "    delta = np.linalg.norm(qn - qr)\n"
        "    print(f'  {tok:<8}  [{qr[0]:+.3f}, {qr[1]:+.3f}, {qr[2]:+.3f}]  '\n"
        "          f'[{qn[0]:+.3f}, {qn[1]:+.3f}, {qn[2]:+.3f}]  {delta:>10.4f}')\n"
        "print()\n"
        "print(\"Notice: 'the' at position 0 has m=0, so angle=0 -> Q_rotated = Q_raw.\")\n"
    )
)

cells.append(
    md(
        "#### Does that rotation actually change what attends to what?\n\nSame projections, one difference - RoPE applied or not - side by side.\n"
    )
)

cells.append(
    code(
        "# \u2500\u2500 RoPE in attention - step 2: the payoff on attention weights \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n"
        "scores_raw = Q_raw @ K_raw.T / math.sqrt(D_MODEL)\n"
        "scores_rot = Q_rot @ K_rot.T / math.sqrt(D_MODEL)\n"
        "attn_raw = torch.softmax(scores_raw, dim=-1)\n"
        "attn_rot = torch.softmax(scores_rot, dim=-1)\n\n"
        "fig, axes = plt.subplots(1, 2, figsize=(12, 4))\n"
        "for ax, w, title in zip(axes, [attn_raw, attn_rot], ['Attention WITHOUT RoPE', 'Attention WITH RoPE']):\n"
        "    sns.heatmap(w.detach().numpy(), ax=ax, annot=True, fmt='.2f', cmap='Blues',\n"
        "                xticklabels=TOKENS, yticklabels=TOKENS, linewidths=0.5)\n"
        "    ax.set_title(title, fontsize=10, fontweight='bold')\n"
        "    ax.set_xlabel('Key'); ax.set_ylabel('Query'); ax.tick_params(axis='x', rotation=30)\n"
        "plt.suptitle('RoPE shifts attention weights by baking position into Q*K scores', fontsize=11, fontweight='bold')\n"
        "plt.tight_layout(); plt.show()\n"
    )
)

cells.append(
    md(
        "### 4c. Discovering the attention formula - softmax and sqrt(d)\n\n"
        "#### Predict first\n\n"
        "1. Raw $QK^T$ scores can be negative and don't sum to 1. **What single operation turns "
        "an arbitrary score vector into a probability distribution?**\n"
        "2. In a real model $d_k$ is 64-128. The dot product of two random vectors grows with "
        "dimension. **What happens to softmax when scores get very large?**\n"
    )
)

cells.append(
    code(
        "# \u2500\u2500 DISCOVERING softmax and the sqrt(d) scale - decision 1: why softmax? \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n"
        "raw = (Q_rot @ K_rot.T)[0].detach().numpy()\n"
        "print('Raw QK^T scores for one query row:')\n"
        "print('  ', raw.round(3))\n"
        "print(f'  sum = {raw.sum():+.3f}  (not 1)  and some are negative -> NOT a probability.')\n"
        "print('  softmax fixes both: exp() makes them positive, then normalise to sum = 1.')\n"
        "print()\n"
        "print('  -> Softmax is the only differentiable function that produces a probability')\n"
        "print('     distribution from arbitrary real-valued scores. -> conclusion')\n"
    )
)

cells.append(
    md(
        "#### Decision 2 - why divide by sqrt(d)?\n\nSoftmax alone isn't enough. The dot product of two random vectors has variance that **grows with dimension**.\n"
    )
)

cells.append(
    code(
        "# \u2500\u2500 Decision 2, step 1: dot-product variance grows with dimension \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n"
        "dims = [4, 16, 64, 256, 1024]\n"
        'print(f\'{"d_k":>6} | {"std(QK^T) unscaled":>18} | {"std after /sqrt(d_k)":>15}\')\n'
        "print('  ' + '-' * 46)\n"
        "peak_unscaled, peak_scaled = [], []\n"
        "for d in dims:\n"
        "    q_v = torch.randn(4000, d)\n"
        "    k_v = torch.randn(4000, d)\n"
        "    dot = (q_v * k_v).sum(dim=-1).numpy()\n"
        "    print(f'{d:>6} | {dot.std():>18.2f} | {(dot / math.sqrt(d)).std():>15.2f}')\n"
        "    qr2 = torch.randn(500, 8, d)\n"
        "    kr2 = torch.randn(500, 8, d)\n"
        "    s_un = (qr2 * kr2).sum(dim=-1)\n"
        "    s_sc = s_un / math.sqrt(d)\n"
        "    peak_unscaled.append(float(torch.softmax(s_un, dim=-1).max(dim=-1).values.mean()))\n"
        "    peak_scaled.append(float(torch.softmax(s_sc, dim=-1).max(dim=-1).values.mean()))\n"
        "print('  Unscaled variance grows like sqrt(d); dividing by sqrt(d) pins it ~1 at every width.')\n"
    )
)

cells.append(
    md(
        "#### The consequence - saturation kills the gradient\n\nBig scores push softmax toward a one-hot spike. A one-hot softmax has almost no slope, so the gradient vanishes.\n"
    )
)

cells.append(
    code(
        "# \u2500\u2500 Decision 2, step 2: the CONSEQUENCE - saturation kills the gradient \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n"
        "d_big = 256\n"
        "k_sat = torch.randn(8, d_big)\n"
        "for label, scale in [('WITHOUT /sqrt(d)', 1.0), ('WITH /sqrt(d)', math.sqrt(d_big))]:\n"
        "    q_sat = torch.randn(8, d_big, requires_grad=True)\n"
        "    p = torch.softmax((q_sat @ k_sat.T) / scale, dim=-1)\n"
        "    loss_sat = torch.sum(torch.max(p, dim=-1).values)\n"
        "    loss_sat.backward()\n"
        "    g = q_sat.grad\n"
        "    peak_p = torch.max(p, dim=-1).values.mean().item()\n"
        "    print(f'  {label:<16}: mean peak prob = {peak_p:.3f}   |grad(q)| = {torch.norm(g).item():.2e}')\n\n"
        "print()\n"
        "print('WITHOUT scaling: softmax -> one-hot (peak~1) -> gradient ~0 -> layer cannot learn.')\n"
        "print('WITH   scaling: distribution stays soft -> gradient flows -> training works.')\n\n"
        "fig, ax = plt.subplots(figsize=(8, 4))\n"
        "ax.plot(dims, peak_unscaled, 'o-', color='tomato',   lw=2, label='unscaled QK^T')\n"
        "ax.plot(dims, peak_scaled,   'o-', color='seagreen', lw=2, label='scaled QK^T/sqrt(d)')\n"
        "ax.axhline(1/8, color='gray', ls='--', lw=1, label='uniform (1/8)')\n"
        "ax.set_xscale('log', base=2); ax.set_xlabel('d_k  (attention head dimension)')\n"
        "ax.set_ylabel('mean peak softmax probability')\n"
        "ax.set_title('Without sqrt(d) scaling, softmax saturates to one-hot as d_k grows')\n"
        "ax.set_ylim(0, 1.05); ax.legend()\n"
        "plt.tight_layout(); plt.show()\n"
    )
)

cells.append(
    md(
        "---\n\n## Part 5 - Multi-Head Attention\n\n"
        "**Multi-Head Attention** (MHA) runs $H$ parallel attention heads:\n\n"
        "$$\\text{MHA}(Q,K,V) = \\text{Concat}(\\text{head}_1, \\ldots, \\text{head}_H) \\cdot W_O$$\n\n"
        "**Why multiple heads?** Each head can specialise on a different relationship type.\n"
    )
)

cells.append(
    code(
        "# \u2500\u2500 Working model constants \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n"
        "D_WORK = 16    # functional model dimension\n"
        "NUM_HEADS = 2  # attention heads\n"
        "D_HEAD = D_WORK // NUM_HEADS   # 8 per head\n"
        "D_FF = 32      # feed-forward hidden size\n\n\n"
        "class MultiHeadAttention(nn.Module):\n"
        "    def __init__(self, d_model, n_heads):\n"
        "        super().__init__()\n"
        "        assert d_model % n_heads == 0\n"
        "        self.n_heads = n_heads\n"
        "        self.d_head = d_model // n_heads\n"
        "        self.d_model = d_model\n"
        "        self.W_Q = nn.Linear(d_model, d_model, bias=False)\n"
        "        self.W_K = nn.Linear(d_model, d_model, bias=False)\n"
        "        self.W_V = nn.Linear(d_model, d_model, bias=False)\n"
        "        self.W_O = nn.Linear(d_model, d_model, bias=False)\n\n"
        "    def forward(self, x, mask=None):\n"
        "        B, S, _ = x.shape\n"
        "        Q_mh = self.W_Q(x).reshape(B, S, self.n_heads, self.d_head).transpose(1, 2)  # (B, H, S, d_head)\n"
        "        K_mh = self.W_K(x).reshape(B, S, self.n_heads, self.d_head).transpose(1, 2)\n"
        "        V_mh = self.W_V(x).reshape(B, S, self.n_heads, self.d_head).transpose(1, 2)\n"
        "        scores = (Q_mh @ K_mh.transpose(-2, -1)) / math.sqrt(self.d_head)  # (B, H, S, S)\n"
        "        if mask is not None:\n"
        "            scores = scores.masked_fill(mask.bool(), float('-inf'))\n"
        "        attn_w_mh = torch.softmax(scores, dim=-1)   # (B, H, S, S)\n"
        "        out = attn_w_mh @ V_mh                       # (B, H, S, d_head)\n"
        "        out = out.transpose(1, 2).reshape(B, S, self.d_model)\n"
        "        return self.W_O(out), attn_w_mh\n\n\n"
        "# \u2500\u2500 Demo \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n"
        "torch.manual_seed(42)\n"
        "mha = MultiHeadAttention(D_WORK, NUM_HEADS)\n\n"
        "proj = nn.Linear(D_MODEL, D_WORK, bias=False)\n"
        "with torch.no_grad():\n"
        "    x_work = proj(embs).unsqueeze(0)   # (1, 6, 16)\n\n"
        "mha_out, head_weights = mha(x_work)\n"
        "print(f'MHA output shape: {tuple(mha_out.shape)}   head_weights shape: {tuple(head_weights.shape)}')\n\n"
        "fig, axes = plt.subplots(1, 2, figsize=(11, 4))\n"
        "for h in range(NUM_HEADS):\n"
        "    ax = axes[h]\n"
        "    w_h = head_weights[0, h].detach().numpy()\n"
        "    sns.heatmap(w_h, ax=ax, annot=True, fmt='.2f', cmap='Purples',\n"
        "                xticklabels=TOKENS, yticklabels=TOKENS, linewidths=0.5, cbar=False)\n"
        "    ax.set_title(f'Head {h} attention weights')\n"
        "    ax.set_xlabel('Key'); ax.set_ylabel('Query'); ax.tick_params(axis='x', rotation=30)\n"
        "plt.suptitle('Multi-Head Attention - each head learns a different relationship', fontsize=11, fontweight='bold')\n"
        "plt.tight_layout(); plt.show()\n"
    )
)

cells.append(
    md(
        "#### Predict first - can one head do two jobs?\n\nA single attention head produces **one** score matrix. Suppose you want every token to attend to **both** its previous token and its most semantically similar token.\n\n**Predict:** can one head satisfy both relations at once, or must it pick one?\n"
    )
)

cells.append(
    code(
        "# \u2500\u2500 PROVING the multi-head claim - step 1: two heads, two patterns \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n"
        "torch.manual_seed(0)\n"
        "S = SEQ_LEN\n"
        "X = embs   # (S, 3)\n"
        "V_shared = X @ torch.randn(3, 3)\n\n"
        "# Relation P (positional): attend to the PREVIOUS token\n"
        "prev_idx = np.array([max(i-1, 0) for i in range(S)])\n"
        "scores_pos = np.full((S, S), -9.0, dtype=np.float32)\n"
        "for i in range(S):\n"
        "    scores_pos[i, prev_idx[i]] = 9.0\n"
        "A_pos = torch.softmax(torch.tensor(scores_pos), dim=-1)\n\n"
        "# Relation C (content): attend to the most SEMANTICALLY SIMILAR token\n"
        "sim = (X @ X.T).detach().numpy()\n"
        "np.fill_diagonal(sim, -1e9)\n"
        "near_idx = sim.argmax(-1)\n"
        "scores_con = np.full((S, S), -9.0, dtype=np.float32)\n"
        "for i in range(S):\n"
        "    scores_con[i, near_idx[i]] = 9.0\n"
        "A_con = torch.softmax(torch.tensor(scores_con), dim=-1)\n\n"
        "corr = np.corrcoef(A_pos.numpy().flatten(), A_con.numpy().flatten())[0, 1]\n"
        "print(f'Correlation between the two head patterns: {corr:+.3f}   (~0 -> different information)')\n\n"
        "fig, axes = plt.subplots(1, 2, figsize=(11, 4))\n"
        "for ax, A, title, cmap in [\n"
        "    (axes[0], A_pos.numpy(), 'Head P - positional (attend to previous token)', 'Greens'),\n"
        "    (axes[1], A_con.numpy(), 'Head C - content (attend to most similar token)', 'Purples'),\n"
        "]:\n"
        "    sns.heatmap(A, ax=ax, annot=True, fmt='.2f', cmap=cmap,\n"
        "                xticklabels=TOKENS, yticklabels=TOKENS, linewidths=0.5, cbar=False)\n"
        "    ax.set_title(title, fontsize=10); ax.set_xlabel('Key'); ax.set_ylabel('Query')\n"
        "    ax.tick_params(axis='x', rotation=30)\n"
        "plt.suptitle('Two heads, two DIFFERENT relations', fontsize=11, fontweight='bold')\n"
        "plt.tight_layout(); plt.show()\n"
    )
)

cells.append(
    md(
        "#### So they differ - but could a *single* head carry both?\n\nEach head yields exactly **one** output per token. Let's measure how well each head recovers each relation.\n"
    )
)

cells.append(
    code(
        "# \u2500\u2500 PROVING the multi-head claim - step 2: one head can't do both \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n"
        "target_prev = V_shared[prev_idx]\n"
        "target_near = V_shared[near_idx]\n"
        "out_pos = A_pos @ V_shared\n"
        "out_con = A_con @ V_shared\n\n\n"
        "def mse(a, b):\n"
        "    return float(((a - b) ** 2).mean())\n\n\n"
        "print('Reconstruction error (lower = that relation is captured):')\n"
        "print(f'  Head P alone -> previous-token target : {mse(out_pos, target_prev):.4f}   <- nails it')\n"
        "print(f'  Head P alone -> similar-token  target : {mse(out_pos, target_near):.4f}   <- misses it')\n"
        "print(f'  Head C alone -> previous-token target : {mse(out_con, target_prev):.4f}   <- misses it')\n"
        "print(f'  Head C alone -> similar-token  target : {mse(out_con, target_near):.4f}   <- nails it')\n"
        "print()\n"
        "print('  -> A single head serves ONE relation well, never both.')\n"
        "print('  -> Concatenating [Head P ; Head C] delivers BOTH targets in parallel.')\n"
        "print('  -> That is why H heads exist: H independent relations, computed at once.')\n"
    )
)

cells.append(
    md(
        "### Your turn - heads\n\nDial the number of heads up and down and watch how independent their patterns become.\n"
    )
)

cells.append(
    code(
        "# \u2500\u2500 EXERCISE 2 - how many heads? \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n"
        "# Change `n_heads` (must divide D_WORK = 16: try 1, 2, 4, 8).\n"
        "# Predict: more heads = more independent relations captured at once.\n"
        "n_heads = 1   # try 1, then 4, then 8\n\n"
        "torch.manual_seed(42)\n"
        "mha_ex = MultiHeadAttention(D_WORK, n_heads)\n"
        "_, w_ex = mha_ex(x_work)\n"
        "print(f'{n_heads} head(s) -> {n_heads} attention pattern(s), each {D_WORK // n_heads}-dim wide.')\n\n"
        "pats = w_ex[0].detach().reshape(n_heads, -1).numpy()\n"
        "if n_heads > 1:\n"
        "    corrs = np.corrcoef(pats)\n"
        "    print('Pairwise correlations between head patterns:')\n"
        "    for hi in range(n_heads):\n"
        "        for hj in range(hi + 1, n_heads):\n"
        "            print(f'  head {hi} vs head {hj}: r = {corrs[hi, hj]:+.3f}')\n"
        "else:\n"
        "    print('  (only one head - no pairwise comparison)')\n"
    )
)

cells.append(
    md(
        "---\n\n## Part 6 - Feed-Forward Network & Layer Normalisation\n\n"
        "### Feed-Forward Network (FFN)\n\n"
        "$$\\text{FFN}(x) = \\text{GELU}(x W_1 + b_1)\\, W_2 + b_2$$\n\n"
        "Expands by 4x then projects back. Adds non-linear transformation capacity.\n\n"
        "### Layer Normalisation\n\n"
        "Applied **before** each sub-layer (Pre-LN style). Normalises each token's vector "
        "to zero mean and unit variance.\n"
    )
)

cells.append(
    code(
        "# \u2500\u2500 FeedForward + LayerNorm \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n"
        "class FeedForward(nn.Module):\n"
        "    def __init__(self, d_model, d_ff):\n"
        "        super().__init__()\n"
        "        self.net = nn.Sequential(\n"
        "            nn.Linear(d_model, d_ff),\n"
        "            nn.GELU(),\n"
        "            nn.Linear(d_ff, d_model),\n"
        "        )\n\n"
        "    def forward(self, x):\n"
        "        return self.net(x)\n\n\n"
        "# \u2500\u2500 Visualise LayerNorm effect \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n"
        "torch.manual_seed(42)\n"
        "ffn = FeedForward(D_WORK, D_FF)\n"
        "norm = nn.LayerNorm(D_WORK, eps=1e-5)\n\n"
        "x_raw = x_work[0]   # (6, 16)\n"
        "with torch.no_grad():\n"
        "    x_after = ffn(x_raw)       # (6, 16) raw FFN output\n"
        "    x_normed = norm(x_after)   # (6, 16) after LayerNorm\n\n"
        "fig, axes = plt.subplots(1, 3, figsize=(14, 4))\n"
        "for ax, data, title in zip(axes, [x_raw, x_after, x_normed],\n"
        "                           ['Input to FFN', 'FFN output (raw)', 'After LayerNorm']):\n"
        "    data_np = data.detach().numpy()\n"
        "    for j, token in enumerate(TOKENS):\n"
        "        vals = data_np[j]\n"
        "        ax.plot(vals, alpha=0.7, label=f'{token}  mu={vals.mean():.2f}, s={vals.std():.2f}')\n"
        "    ax.set_title(title); ax.set_xlabel('Hidden dimension'); ax.set_ylabel('Activation value')\n"
        "    ax.legend(fontsize=7); ax.axhline(0, color='black', lw=0.5, ls='--')\n"
        "plt.suptitle('FFN activations before and after LayerNorm', fontsize=11, fontweight='bold')\n"
        "plt.tight_layout(); plt.show()\n\n"
        "print('LayerNorm centres and normalises each token slice.')\n"
        "print('Mean and std across dimensions after LN:')\n"
        "x_normed_np = x_normed.detach().numpy()\n"
        "for j, token in enumerate(TOKENS):\n"
        "    v = x_normed_np[j]\n"
        "    print(f'  {token:<8}  mean={v.mean():+.4f}  std={v.std():.4f}')\n"
    )
)

cells.append(
    md(
        "---\n\n## Part 7 - Full Transformer Block & RNN Comparison\n\n"
        "A single **Transformer Block** wires MHA + FFN together with layer norm and residuals:\n\n"
        "```\n"
        "x --> LayerNorm --> MHA --> (+x) --> LayerNorm --> FFN --> (+x) --> output\n"
        "```\n\n"
        "Stack $L$ of these blocks = the full encoder/decoder stack.\n"
    )
)

cells.append(
    code(
        "# \u2500\u2500 TransformerBlock \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n"
        "class TransformerBlock(nn.Module):\n"
        "    def __init__(self, d_model, n_heads, d_ff):\n"
        "        super().__init__()\n"
        "        self.norm1 = nn.LayerNorm(d_model, eps=1e-5)\n"
        "        self.mha   = MultiHeadAttention(d_model, n_heads)\n"
        "        self.norm2 = nn.LayerNorm(d_model, eps=1e-5)\n"
        "        self.ffn   = FeedForward(d_model, d_ff)\n\n"
        "    def forward(self, x, mask=None):\n"
        "        mha_out, attn_w_b = self.mha(self.norm1(x), mask=mask)\n"
        "        x = x + mha_out\n"
        "        x = x + self.ffn(self.norm2(x))\n"
        "        return x, attn_w_b\n\n\n"
        "torch.manual_seed(42)\n"
        "block1 = TransformerBlock(D_WORK, NUM_HEADS, D_FF)\n"
        "block2 = TransformerBlock(D_WORK, NUM_HEADS, D_FF)\n\n"
        "x0 = x_work.clone()   # (1, 6, 16)\n"
        "with torch.no_grad():\n"
        "    x1, aw1 = block1(x0)\n"
        "    x2, aw2 = block2(x1)\n\n"
        "print('Input -> Block 1 -> Block 2:')\n"
        "print(f'  x0: {tuple(x0.shape)}  norm={float(torch.norm(x0)):.3f}')\n"
        "print(f'  x1: {tuple(x1.shape)}  norm={float(torch.norm(x1)):.3f}')\n"
        "print(f'  x2: {tuple(x2.shape)}  norm={float(torch.norm(x2)):.3f}')\n\n"
        "fig, ax = plt.subplots(figsize=(8, 4))\n"
        "norms = {\n"
        "    'Layer 0 (input)': torch.norm(x0[0], dim=-1).detach().numpy(),\n"
        "    'Layer 1 output':  torch.norm(x1[0], dim=-1).detach().numpy(),\n"
        "    'Layer 2 output':  torch.norm(x2[0], dim=-1).detach().numpy(),\n"
        "}\n"
        "x_pos = np.arange(SEQ_LEN); width = 0.25\n"
        "for k, (label, vals) in enumerate(norms.items()):\n"
        "    ax.bar(x_pos + k*width, vals, width, label=label, alpha=0.85)\n"
        "ax.set_xticks(x_pos + width); ax.set_xticklabels(TOKENS)\n"
        "ax.set_ylabel('Representation L2 norm')\n"
        "ax.set_title('Token representations grow through transformer blocks')\n"
        "ax.legend(); plt.tight_layout(); plt.show()\n"
    )
)

cells.append(
    md(
        "#### Predict first - does the skip connection really matter?\n\nWe'll stack **24** simple layers and read the gradient that reaches **layer 1** (furthest from the loss), with and without the skip $x + \\text{SubLayer}(x)$.\n\n**Predict:** without the skip, will the gradient at layer 1 be vanishingly small?\n"
    )
)

cells.append(
    code(
        "# \u2500\u2500 DEMONSTRATING why residual connections make depth trainable \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n\n"
        "DEPTH = 24\n"
        "d = D_WORK\n\n\n"
        "class ProbeBlock(nn.Module):\n"
        "    def __init__(self, d_in, residual):\n"
        "        super().__init__()\n"
        "        self.lin = nn.Linear(d_in, d_in)\n"
        "        self.residual = residual\n\n"
        "    def forward(self, x):\n"
        "        y = torch.tanh(self.lin(x))\n"
        "        return x + y if self.residual else y\n\n\n"
        "def gradient_reaching_each_layer(residual):\n"
        "    torch.manual_seed(0)\n"
        "    blocks = nn.ModuleList([ProbeBlock(d, residual) for _ in range(DEPTH)])\n"
        "    x_p = torch.randn(1, d)\n"
        "    h = x_p\n"
        "    for b in blocks:\n"
        "        h = b(h)\n"
        "    loss_p = torch.mean(h ** 2)\n"
        "    loss_p.backward()\n"
        "    return [b.lin.weight.grad.norm().item() for b in blocks]\n\n\n"
        "g_res   = gradient_reaching_each_layer(residual=True)\n"
        "g_plain = gradient_reaching_each_layer(residual=False)\n\n"
        "fig, ax = plt.subplots(figsize=(9, 4.2))\n"
        "ax.plot(range(1, DEPTH+1), g_plain, 'o-', color='tomato',   lw=2, label='WITHOUT residual')\n"
        "ax.plot(range(1, DEPTH+1), g_res,   'o-', color='seagreen', lw=2, label='WITH residual')\n"
        "ax.set_yscale('log')\n"
        "ax.set_xlabel('Layer (1 = furthest from loss, closest to input)')\n"
        "ax.set_ylabel('|gradient| reaching this layer  (log scale)')\n"
        "ax.set_title('Residual connections keep gradients alive all the way to layer 1')\n"
        "ax.legend(); plt.tight_layout(); plt.show()\n\n"
        "print('Gradient norm reaching layer 1 (the earliest, hardest-to-train layer):')\n"
        "print(f'  WITHOUT residual: {g_plain[0]:.2e}   <- vanished')\n"
        "print(f'  WITH    residual: {g_res[0]:.2e}   <- healthy')\n"
        "ratio = g_res[0] / max(g_plain[0], 1e-30)\n"
        "print(f'  The skip path delivers ~{ratio:.1e}x more gradient to the earliest layer.')\n"
    )
)

cells.append(
    md(
        "### Your turn - depth\n\nPush the stack deeper and watch the no-residual gradient collapse while the residual one stays alive.\n"
    )
)

cells.append(
    code(
        "# \u2500\u2500 EXERCISE 3 - how deep can you go WITHOUT residuals? \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n"
        "# Change `DEPTH` (try 8, 24, 60) and PREDICT how far the gradient survives.\n"
        "DEPTH = 40\n\n"
        "g_res_ex   = gradient_reaching_each_layer(residual=True)\n"
        "g_plain_ex = gradient_reaching_each_layer(residual=False)\n"
        "print(f'At depth {DEPTH}, gradient reaching layer 1 (the hardest to train):')\n"
        "print(f'  WITHOUT residual: {g_plain_ex[0]:.2e}')\n"
        "print(f'  WITH    residual: {g_res_ex[0]:.2e}')\n"
        "ratio_ex = g_res_ex[0] / max(g_plain_ex[0], 1e-30)\n"
        "print(f'  -> Ratio: {ratio_ex:.1e}x more gradient with residuals.')\n"
    )
)

print(f"Built {len(cells)} cells for Part 1-7 scaffolding")
