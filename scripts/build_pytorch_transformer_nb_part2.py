"""Second half: Part 8-16 + Summary cells for the PyTorch Transformer notebook."""

# This file is appended to build_pytorch_transformer_nb.py via the runner script.
# Assumes `cells` list already exists from Part 1 of the builder.

# ═══════════════════════════════════════════════════════════════════════════════
# PART 8-9 : MiniLM Training + Inference
# ═══════════════════════════════════════════════════════════════════════════════

cells.append(
    md(
        "---\n\n## Part 8 - Mini Language Model: Training & Inference\n\n"
        "We now wire everything together into a **Mini Language Model** - a decoder-only transformer "
        "that learns to predict the next token. This is the architecture of GPT, LLaMA, Mistral etc.\n\n"
        "**Training task**: given a context window, predict the next token.\n"
    )
)

cells.append(
    code(
        "# \u2500\u2500 MiniLM - decoder-only transformer language model \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n"
        "class MiniLM(nn.Module):\n"
        '    """\n'
        "    Decoder-only transformer language model.\n"
        "    Architecture: token_emb -> sinusoidal_PE -> n_layers x TransformerBlock -> lm_head\n"
        '    """\n\n'
        "    def __init__(self, vocab_size, d_model, n_heads, d_ff, n_layers, max_seq=64):\n"
        "        super().__init__()\n"
        "        self.d_model = d_model\n"
        "        self.vocab_size = vocab_size\n"
        "        self.token_emb = nn.Embedding(vocab_size, d_model)\n"
        "        self.blocks = nn.ModuleList([\n"
        "            TransformerBlock(d_model, n_heads, d_ff) for _ in range(n_layers)\n"
        "        ])\n"
        "        self.norm_out = nn.LayerNorm(d_model, eps=1e-5)\n"
        "        pe = sinusoidal_pe(max_seq, d_model)\n"
        "        self.register_buffer('pe', pe)\n\n"
        "    def forward(self, token_ids, return_attn=False):\n"
        '        """\n'
        "        token_ids: (batch, seq_len)  int tensor\n"
        "        Returns logits: (batch, seq_len, vocab_size)\n"
        '        """\n'
        "        S = token_ids.shape[1]\n"
        "        x = self.token_emb(token_ids) + self.pe[:S]   # (B, S, d_model)\n"
        "        causal_mask = torch.triu(torch.ones(S, S, dtype=torch.bool), diagonal=1).to(x.device)\n"
        "        all_attn = []\n"
        "        for block in self.blocks:\n"
        "            x, aw = block(x, mask=causal_mask)\n"
        "            all_attn.append(aw)\n"
        "        x = self.norm_out(x)\n"
        "        # Weight tying: reuse token embedding matrix as output projection\n"
        "        logits = x @ self.token_emb.weight.T\n"
        "        if return_attn:\n"
        "            return logits, all_attn\n"
        "        return logits\n\n\n"
        "torch.manual_seed(42)\n"
        "model_demo = MiniLM(vocab_size=VOCAB_SIZE, d_model=D_WORK, n_heads=NUM_HEADS, d_ff=D_FF, n_layers=2)\n"
        "with torch.no_grad():\n"
        "    _ = model_demo(torch.tensor([TOKEN_IDS]))\n"
        "n_params = sum(p.numel() for p in model_demo.parameters())\n"
        "print(f'MiniLM -> {n_params:,} trainable parameters')\n"
        "for name, p in model_demo.named_parameters():\n"
        "    print(f'  {name:<40} {tuple(p.shape)}')\n"
    )
)

cells.append(
    code(
        "# \u2500\u2500 Training data - (context, next_token) pairs \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n"
        "full_corpus = [\n"
        '    "the cat sat on the mat",\n'
        '    "the dog ran over the fence",\n'
        '    "a big cat jumped over the fence",\n'
        '    "a dog sat on the mat",\n'
        '    "the cat jumped over the fence",\n'
        '    "the big dog ran on the mat",\n'
        "]\n\n"
        "TRAIN_PAIRS = []\n"
        "for sentence in full_corpus:\n"
        "    ids = encode(sentence)\n"
        "    for end in range(1, len(ids)):\n"
        "        TRAIN_PAIRS.append((ids[:end], ids[end]))\n\n"
        "print(f'Training pairs: {len(TRAIN_PAIRS)}')\n"
        "print('\\nFirst 6 examples:')\n"
        "for ctx, tgt in TRAIN_PAIRS[:6]:\n"
        "    print(f'  {[IDX2WORD[i] for i in ctx]}  ->  \"{IDX2WORD[tgt]}\"')\n"
    )
)

cells.append(
    md(
        "Now the training loop: full-batch gradient descent, predicting each next token from the position before it.\n"
    )
)

cells.append(
    code(
        "# \u2500\u2500 Training loop \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n"
        "def pad_collate(pairs, pad_id=0):\n"
        "    max_len = max(len(ctx) for ctx, _ in pairs)\n"
        "    xs, ys = [], []\n"
        "    for ctx, tgt in pairs:\n"
        "        pad = [pad_id] * (max_len - len(ctx))\n"
        "        xs.append(pad + ctx)\n"
        "        ys.append(tgt)\n"
        "    return torch.tensor(xs, dtype=torch.long), torch.tensor(ys, dtype=torch.long)\n\n\n"
        "torch.manual_seed(42)\n"
        "model = MiniLM(VOCAB_SIZE, D_WORK, NUM_HEADS, D_FF, n_layers=2)\n"
        "optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)\n"
        "loss_fn = nn.CrossEntropyLoss()\n\n"
        "EPOCHS = 300\n"
        "x_train, y_train = pad_collate(TRAIN_PAIRS)\n"
        "loss_history, acc_history = [], []\n\n"
        "for epoch in range(EPOCHS):\n"
        "    model.train()\n"
        "    optimizer.zero_grad()\n"
        "    logits = model(x_train)\n"
        "    last_logits = logits[:, -1, :]\n"
        "    loss = loss_fn(last_logits, y_train)\n"
        "    loss.backward()\n"
        "    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)\n"
        "    optimizer.step()\n\n"
        "    if (epoch + 1) % 10 == 0:\n"
        "        with torch.no_grad():\n"
        "            preds = torch.argmax(last_logits, dim=-1)\n"
        "            acc = (preds == y_train).float().mean().item()\n"
        "        loss_history.append(loss.item())\n"
        "        acc_history.append(acc)\n"
        "        if (epoch + 1) % 50 == 0:\n"
        "            print(f'Epoch {epoch+1:4d} | loss={loss.item():.4f} | acc={acc:.2%}')\n\n"
        "print('\\nTraining complete.')\n"
    )
)

cells.append(
    code(
        "# \u2500\u2500 Training loss & accuracy plot \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n"
        "epochs_logged = list(range(10, EPOCHS + 1, 10))\n\n"
        "fig, axes = plt.subplots(1, 2, figsize=(12, 4))\n\n"
        "ax = axes[0]\n"
        "ax.plot(epochs_logged, loss_history, color='royalblue', lw=2)\n"
        "ax.fill_between(epochs_logged, loss_history, alpha=0.15, color='royalblue')\n"
        "ax.set_xlabel('Epoch'); ax.set_ylabel('Cross-Entropy Loss'); ax.set_title('Training Loss')\n\n"
        "ax2 = axes[1]\n"
        "ax2.plot(epochs_logged, [a * 100 for a in acc_history], color='mediumseagreen', lw=2)\n"
        "ax2.fill_between(epochs_logged, [a * 100 for a in acc_history], alpha=0.15, color='mediumseagreen')\n"
        "ax2.set_xlabel('Epoch'); ax2.set_ylabel('Accuracy (%)'); ax2.set_title('Next-Token Prediction Accuracy')\n"
        "ax2.set_ylim(0, 105); ax2.axhline(100, color='grey', ls='--', lw=0.8)\n\n"
        "plt.suptitle('MiniLM Training Progress', fontsize=12, fontweight='bold')\n"
        "plt.tight_layout(); plt.show()\n"
    )
)

cells.append(
    md(
        "---\n\n## Part 9 - Inference: Autoregressive Token Generation\n\n"
        "At inference time a language model generates text **one token at a time**:\n\n"
        "1. Feed the current context into the model\n"
        "2. Take the logits at the **last position**\n"
        "3. Apply temperature scaling + softmax\n"
        "4. Sample (or argmax = greedy)\n"
        "5. Append the new token -> go to step 1\n"
    )
)

cells.append(
    code(
        "# \u2500\u2500 Inference - autoregressive token generation \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n"
        "def generate_next(context_words, temperature=1.0):\n"
        '    """Single next-token generation step with probability bar chart."""\n'
        "    model.eval()\n"
        "    with torch.no_grad():\n"
        "        ids = torch.tensor([encode(' '.join(context_words))], dtype=torch.long)\n"
        "        logits_inf = model(ids)\n"
        "        last = logits_inf[0, -1, :]\n"
        "        probs = torch.softmax(last / max(temperature, 1e-6), dim=-1).numpy()\n\n"
        "    topk_idx = probs.argsort()[::-1][:8]\n"
        "    top_words = [IDX2WORD[int(i)] for i in topk_idx]\n"
        "    top_probs = probs[topk_idx]\n\n"
        "    fig, ax = plt.subplots(figsize=(8, 3))\n"
        "    ax.barh(top_words[::-1], top_probs[::-1],\n"
        "            color=['gold' if w == top_words[0] else 'steelblue' for w in top_words[::-1]])\n"
        "    ax.set_xlabel('Probability')\n"
        "    ax.set_title(f'Next token probabilities | context: {context_words}  T={temperature}')\n"
        "    ax.set_xlim(0, 1.0)\n"
        "    for bar, prob in zip(ax.patches, top_probs[::-1]):\n"
        "        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,\n"
        "                f'{prob:.3f}', va='center', fontsize=9)\n"
        "    plt.tight_layout(); plt.show()\n\n"
        "    best = IDX2WORD[int(probs.argmax())]\n"
        "    print(f'  Greedy prediction: \"{best}\"')\n"
        "    return best\n\n\n"
        "# \u2500\u2500 Demo: step-by-step generation \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n"
        "print('=== Autoregressive generation ===')\n"
        "print()\n"
        "context = ['the']\n"
        "for step in range(5):\n"
        "    print(f'Step {step+1}: context = {context}')\n"
        "    next_tok = generate_next(context, temperature=0.8)\n"
        "    context.append(next_tok)\n"
        "    print()\n\n"
        "print(f'Generated sequence: {\" \".join(context)}')\n"
    )
)

cells.append(
    md(
        "### From toy to real - same mechanism, bigger numbers\n\nEverything you've built used tiny dimensions so the vectors stayed readable. A production model is the **identical machinery** scaled up.\n"
    )
)

cells.append(
    code(
        "# \u2500\u2500 Toy (this notebook) vs. a real model (GPT-2 / DistilGPT-2) \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n"
        "rows = [\n"
        "    ('embedding dim  d_model', D_MODEL, D_WORK, 768),\n"
        "    ('attention heads',        '?',     NUM_HEADS, 12),\n"
        "    ('dim per head  d_head',   '?',     D_HEAD, 64),\n"
        "    ('feed-forward hidden',    '?',     D_FF, 3072),\n"
        "    ('transformer layers',     '?',     2, 12),\n"
        "    ('vocabulary size',        VOCAB_SIZE, VOCAB_SIZE, 50257),\n"
        "]\n"
        'print(f\'{"component":<24}{"viz":>8}{"toy model":>12}{"GPT-2":>10}\')\n'
        "print('  ' + '-' * 52)\n"
        "for name, viz, toy, real in rows:\n"
        "    print(f'  {name:<22} {str(viz):>8} {str(toy):>12} {str(real):>10}')\n"
        "print()\n"
        "print('Every component in GPT-2 is identical in kind to what you built.')\n"
        "print('  -> Scale, not novelty, is what makes GPT-2 impressive.')\n"
    )
)

# ═══════════════════════════════════════════════════════════════════════════════
# PART 10-12 : W_V Filter, Causal Triangle, Accumulation Tower
# ═══════════════════════════════════════════════════════════════════════════════

cells.append(
    md(
        "---\n\n## Part 11 - Why W_V Is a Relevance Filter, Not a Passthrough\n\n"
        "$W_V$ is a **task-specific extraction lens**. Each attention layer has a different job. "
        "$W_V$ lets each layer extract exactly the slice it needs from each token's information.\n"
    )
)

cells.append(
    md(
        "#### Predict first - do we even need W_V?\n\n"
        "Suppose we deleted $W_V$ entirely and blended the **raw token embeddings** directly.\n\n"
        '**Predict:** if a *grammar* attention layer wants to know "is this token part of an active '
        'event?", it cares mainly about Dynamism and Animacy. Without $W_V$, can it ignore Concreteness?\n'
    )
)

cells.append(
    code(
        "# \u2500\u2500 Part 11: W_V as a relevance filter \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n"
        "embs_sentence = embedding_matrix[TOKEN_IDS].numpy()   # (6, 3)\n\n"
        "# W_V_action: extract the ACTION channel (Animacy + Dynamism)\n"
        "W_V_action = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float32)\n\n"
        "# W_V_object: extract the OBJECT channel (Concreteness + Animacy)\n"
        "W_V_object = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]], dtype=np.float32)\n\n"
        "V_action = embs_sentence @ W_V_action   # (6, 2)\n"
        "V_object = embs_sentence @ W_V_object   # (6, 2)\n\n"
        "attn_row_cat = attn_w.detach().numpy()[TOKENS.index('cat')]\n"
        "blend_action = (attn_row_cat[:, None] * V_action).sum(0)\n"
        "blend_object = (attn_row_cat[:, None] * V_object).sum(0)\n\n"
        "fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))\n"
        "specs = [\n"
        "    (axes[0], V_action, blend_action, 'W_V_action - ACTION lens\\n(Animacy x Dynamism)', 'Animacy', 'Dynamism'),\n"
        "    (axes[1], V_object, blend_object, 'W_V_object - OBJECT lens\\n(Concreteness x Animacy)', 'Concreteness', 'Animacy'),\n"
        "]\n"
        "for ax, V, blend, title, xl, yl in specs:\n"
        "    ax.scatter(V[:, 0], V[:, 1], s=80, c='lightgray', edgecolor='#888', zorder=3)\n"
        "    for i, tok in enumerate(TOKENS):\n"
        "        ax.annotate(f'[{i}]{tok}', (V[i, 0], V[i, 1] + 0.03), fontsize=8, ha='center', color='dimgray')\n"
        "    ax.scatter(*blend, s=280, color='gold', edgecolor='#b8860b', zorder=5,\n"
        "               label='\"cat\" blended context', linewidths=2)\n"
        "    ax.set_xlabel(xl); ax.set_ylabel(yl); ax.set_title(title, fontsize=10); ax.legend(fontsize=9)\n"
        "plt.suptitle('W_V shapes WHAT part of each token enters the weighted blend.\\n'\n"
        "             'Same sentence, same attention weights - different W_V - different context vector.',\n"
        "             fontsize=11, fontweight='bold')\n"
        "plt.tight_layout(); plt.show()\n\n"
        "print('Key insight:')\n"
        "print('  Attention weights (W_Q/W_K path) answer WHO gets blended.')\n"
        "print('  W_V answers WHAT each token contributes to that blend.')\n"
        "print('  -> W_V is not a passthrough; it is a task-specific extraction lens.')\n"
    )
)

cells.append(
    md(
        "---\n\n## Part 12 - The Causal Triangle and the Accumulation Tower\n\n"
        "The causal mask determines *how much of the sentence each position gets to know about*. "
        "Position 0 sees only itself. Position 5 sees all six tokens.\n\n"
        "> **By the time the last position exits the final transformer block, it has absorbed a chain "
        "of increasingly enriched representations from every earlier position.**\n"
    )
)

cells.append(
    code(
        "# \u2500\u2500 The Causal Triangle - explicit lower-triangular structure \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n"
        "S = SEQ_LEN\n"
        "mask_vis = np.tril(np.ones((S, S), dtype=int))\n\n"
        "fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))\n\n"
        "ax = axes[0]\n"
        "sns.heatmap(mask_vis.astype(float), ax=ax, cmap='Blues', vmin=0, vmax=1,\n"
        "            xticklabels=TOKENS, yticklabels=TOKENS, linewidths=1.0, cbar=False,\n"
        "            annot=mask_vis, fmt='d', annot_kws={'size': 14, 'weight': 'bold'})\n"
        "ax.set_title('Causal Mask\\n1 = allowed to attend  |  0 = blocked')\n"
        "ax.set_xlabel('Key token'); ax.set_ylabel('Query token')\n\n"
        "ax2 = axes[1]\n"
        "history_counts = np.arange(1, S + 1)\n"
        "bar_colors = plt.cm.Blues(np.linspace(0.35, 0.9, S))\n"
        "ax2.bar(range(S), history_counts, color=bar_colors, edgecolor='white', lw=1)\n"
        "ax2.set_xticks(range(S)); ax2.set_xticklabels(TOKENS)\n"
        "ax2.set_ylabel('Tokens visible to this position')\n"
        "ax2.set_title('How many tokens each position knows about')\n"
        "for i, c in enumerate(history_counts):\n"
        "    ax2.text(i, c + 0.05, str(c), ha='center', fontsize=11, fontweight='bold')\n\n"
        "plt.suptitle('Causal Triangle: position 0 is isolated; position 5 absorbs all 6 tokens',\n"
        "             fontsize=11, fontweight='bold')\n"
        "plt.tight_layout(); plt.show()\n"
    )
)

cells.append(
    md(
        "#### Predict first - does the accumulated history actually matter?\n\n"
        '- **Full** context: `"the cat sat on the mat"` -> last token `"mat"`, 6 tokens of history\n'
        '- **Truncated** context: `"the cat sat"` -> last token `"sat"`, 3 tokens of history\n\n'
        "**Predict:** at block 3, will these two last-position vectors be very similar or clearly different?\n"
    )
)

cells.append(
    code(
        "# \u2500\u2500 Accumulation Tower: last-position richness grows with depth \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n"
        "torch.manual_seed(1)\n"
        "tower_model = MiniLM(VOCAB_SIZE, D_WORK, NUM_HEADS, D_FF, n_layers=3)\n"
        "with torch.no_grad():\n"
        "    _ = tower_model(torch.tensor([TOKEN_IDS]))\n\n\n"
        "def trace_last_position(ids_list):\n"
        '    """Return the last-position hidden state after each transformer block."""\n'
        "    tower_model.eval()\n"
        "    with torch.no_grad():\n"
        "        token_ids_t = torch.tensor([ids_list], dtype=torch.long)\n"
        "        S_t = len(ids_list)\n"
        "        x = tower_model.token_emb(token_ids_t) + tower_model.pe[:S_t]\n"
        "        causal_m = torch.triu(torch.ones(S_t, S_t, dtype=torch.bool), diagonal=1)\n"
        "        reps = []\n"
        "        for block in tower_model.blocks:\n"
        "            x, _ = block(x, mask=causal_m)\n"
        "            reps.append(x[0, -1, :].detach().numpy())\n"
        "    return reps\n\n\n"
        "reps_full  = trace_last_position(TOKEN_IDS)\n"
        "reps_trunc = trace_last_position(TOKEN_IDS[:3])\n\n\n"
        "def cos_sim(a, b):\n"
        "    a = a / (np.linalg.norm(a) + 1e-9)\n"
        "    b = b / (np.linalg.norm(b) + 1e-9)\n"
        "    return float(a @ b)\n\n\n"
        "similarities = [cos_sim(reps_full[d], reps_trunc[d]) for d in range(3)]\n\n"
        "fig, axes = plt.subplots(1, 2, figsize=(13, 4))\n"
        "ax = axes[0]\n"
        "for label, reps, col in [('full (6 tokens)', reps_full, 'royalblue'), ('truncated (3 tokens)', reps_trunc, 'tomato')]:\n"
        "    norms = [np.linalg.norm(r) for r in reps]\n"
        "    ax.plot(range(1, 4), norms, 'o-', lw=2, label=label, color=col)\n"
        "ax.set_xticks([1, 2, 3]); ax.set_xticklabels(['Block 1', 'Block 2', 'Block 3'])\n"
        "ax.set_ylabel('L2 norm of last-position vector'); ax.set_title('Representation grows as context accumulates'); ax.legend(fontsize=9)\n\n"
        "ax2 = axes[1]\n"
        "bar_c = ['#5ab4ac' if s > 0.9 else ('#d8b365' if s > 0.7 else 'tomato') for s in similarities]\n"
        "ax2.bar(range(3), similarities, color=bar_c, alpha=0.9, edgecolor='white', lw=1.2)\n"
        "ax2.plot(range(3), similarities, 'ko--', lw=1.5, ms=7)\n"
        "for i, s in enumerate(similarities):\n"
        "    ax2.text(i, s + 0.01, f'{s:.3f}', ha='center', fontsize=11, fontweight='bold')\n"
        "ax2.set_xticks([0, 1, 2]); ax2.set_xticklabels(['Block 1', 'Block 2', 'Block 3'])\n"
        "ax2.set_ylabel('Cosine similarity')\n"
        "ax2.set_title('Last-position vectors (full vs truncated context)\\ndiverge as depth grows', fontsize=10)\n"
        "ax2.set_ylim(0, 1.1); ax2.axhline(1.0, color='lightgray', ls='--', lw=1)\n\n"
        "plt.suptitle('Accumulation Tower: same token ID, different history -> representations diverge with depth',\n"
        "             fontsize=11, fontweight='bold')\n"
        "plt.tight_layout(); plt.show()\n\n"
        "for d, s in enumerate(similarities):\n"
        "    tag = 'still similar' if s > 0.9 else ('diverging' if s > 0.7 else 'very different')\n"
        "    print(f'  Block {d+1}: cosine similarity = {s:.3f}  ({tag})')\n"
        "print('  -> Deeper stacks make the last position a richer accumulation point.')\n"
    )
)

# ═══════════════════════════════════════════════════════════════════════════════
# PART 13 : Three Architectures
# ═══════════════════════════════════════════════════════════════════════════════

cells.append(
    md(
        "---\n\n## Part 13 - Three Architectures: Reader, Writer, Translator\n\n"
        "| Architecture | Mask on self-attn | Primary output | Real examples |\n"
        "| ------------ | ----------------- | -------------- | ------------- |\n"
        "| **Encoder-only** | None (bidirectional) | Enriched vector per input token | BERT, RoBERTa |\n"
        "| **Decoder-only** | Causal (lower-tri) | Next-token logits | GPT, LLaMA |\n"
        "| **Encoder-Decoder** | Enc: none / Dec: causal | Seq2seq translation | T5, BART |\n"
    )
)

cells.append(
    md(
        "### 13a - Encoder: The Reader (Bidirectional Attention)\n\n"
        "An encoder removes the mask entirely. Every token can attend to every other token. "
        '`"cat"` at position 1 can immediately see `"mat"` at position 5.\n\n'
        "The encoder does **not** predict next tokens. It produces a sequence of enriched context "
        "vectors - one per input token.\n"
    )
)

cells.append(
    code(
        "# \u2500\u2500 Part 13a: MiniEncoder - decoder-only twin, minus the mask \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n\n\n"
        "class MiniEncoder(nn.Module):\n"
        '    """\n'
        "    Encoder-only transformer (BERT-style).\n"
        "    Passes mask=None to every TransformerBlock - all tokens see all tokens.\n"
        "    Output: one enriched d_model-dimensional vector per input token.\n"
        '    """\n\n'
        "    def __init__(self, vocab_size, d_model, n_heads, d_ff, n_layers, max_seq=64):\n"
        "        super().__init__()\n"
        "        self.token_emb = nn.Embedding(vocab_size, d_model)\n"
        "        pe = sinusoidal_pe(max_seq, d_model)\n"
        "        self.register_buffer('pe', pe)\n"
        "        self.blocks = nn.ModuleList([\n"
        "            TransformerBlock(d_model, n_heads, d_ff) for _ in range(n_layers)\n"
        "        ])\n"
        "        self.norm_out = nn.LayerNorm(d_model, eps=1e-5)\n\n"
        "    def forward(self, token_ids, return_attn=False):\n"
        "        S = token_ids.shape[1]\n"
        "        x = self.token_emb(token_ids) + self.pe[:S]\n"
        "        all_attn = []\n"
        "        for block in self.blocks:\n"
        "            x, aw = block(x, mask=None)   # None = bidirectional; the only difference\n"
        "            all_attn.append(aw)\n"
        "        out = self.norm_out(x)\n"
        "        if return_attn:\n"
        "            return out, all_attn\n"
        "        return out\n\n\n"
        "torch.manual_seed(42)\n"
        "encoder = MiniEncoder(VOCAB_SIZE, D_WORK, NUM_HEADS, D_FF, n_layers=2)\n"
        "ids_batch = torch.tensor([TOKEN_IDS])\n"
        "with torch.no_grad():\n"
        "    enc_out, enc_attns = encoder(ids_batch, return_attn=True)\n\n"
        "print(f'Encoder output shape: {tuple(enc_out.shape)}')\n"
        "print(f'  -> One enriched {D_WORK}-dim vector per token (same shape as decoder output)')\n"
        "print(f'  -> These are NOT next-token predictions - they are context carriers')\n"
        "print()\n"
        "print('The entire MiniEncoder class differs from MiniLM in exactly one place:')\n"
        "print('  MiniLM      block(x, mask=causal_mask)   # upper triangle -> -inf')\n"
        "print('  MiniEncoder block(x, mask=None)           # nothing blocked')\n"
    )
)

cells.append(
    md(
        "#### Predict first - which cells in the heatmap open up?\n\n"
        'The decoder\'s causal heatmap has a hard lower-triangle: `"the"` at [0] attends only to itself.\n\n'
        "Now we set `mask=None`. **Predict:**\n"
        '- How many tokens will `"the"` at [0] attend to now - 1, 3, or 6?\n'
        '- Will `"the"` at [0] have a non-zero score for `"mat"` at position [5]?\n'
    )
)

cells.append(
    code(
        "# \u2500\u2500 Encoder vs Decoder: SAME MHA weights, SAME input - only the mask differs \u2500\u2500\u2500\u2500\u2500\u2500\n"
        "torch.manual_seed(42)\n"
        "shared_mha = MultiHeadAttention(D_WORK, NUM_HEADS)\n"
        "causal_mask_cmp = torch.triu(torch.ones(SEQ_LEN, SEQ_LEN, dtype=torch.bool), diagonal=1)\n\n"
        "with torch.no_grad():\n"
        "    _, w_bidir      = shared_mha(x_work, mask=None)\n"
        "    _, w_causal_cmp = shared_mha(x_work, mask=causal_mask_cmp)\n\n"
        "bidir_h0  = w_bidir[0, 0].detach().numpy()\n"
        "causal_h0 = w_causal_cmp[0, 0].detach().numpy()\n\n"
        "fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))\n"
        "for ax, data, title, cmap in [\n"
        "    (axes[0], bidir_h0,  'Encoder  (mask=None, bidirectional)\\n'\n"
        '              \'"the" at [0] already attends to "cat", "sat", "mat" in layer 1\', \'Greens\'),\n'
        "    (axes[1], causal_h0, 'Decoder  (mask=causal_mask, lower triangle only)\\n'\n"
        "              '\"the\" at [0] sees only itself', 'Oranges'),\n"
        "]:\n"
        "    sns.heatmap(data, ax=ax, annot=True, fmt='.2f', cmap=cmap,\n"
        "                xticklabels=TOKENS, yticklabels=TOKENS, linewidths=0.5,\n"
        "                cbar_kws={'label': 'attention weight'})\n"
        "    ax.set_title(title, fontsize=10)\n"
        "    ax.set_xlabel('Key'); ax.set_ylabel('Query'); ax.tick_params(axis='x', rotation=30)\n\n"
        "plt.suptitle('Same MHA layer, same weights, same input - the mask is the ONLY difference\\n'\n"
        "             'This is the complete implementation difference between Encoder and Decoder',\n"
        "             fontsize=11, fontweight='bold')\n"
        "plt.tight_layout(); plt.show()\n\n"
        "print('Encoder: \"the\" at [0] attends to all 6 tokens from layer 1.')\n"
        "print('Decoder: \"the\" at [0] sees only itself.')\n"
        "print()\n"
        "print('Consequence:')\n"
        "print('  Encoder -> each output holds full bidirectional context')\n"
        "print('  Decoder -> each output depends ONLY on what came before')\n"
    )
)

cells.append(
    code(
        "# \u2500\u2500 EXERCISE 4 - toggle the mask and watch the heatmap change \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n"
        "# Flip USE_ENCODER_MASK to False.\n"
        "# PREDICT first: which cells in row 0 will go to zero?\n"
        "USE_ENCODER_MASK = True   # set to False to turn the encoder into a decoder\n\n"
        "mask_ex4 = None if USE_ENCODER_MASK else causal_mask_cmp\n"
        "with torch.no_grad():\n"
        "    _, w_ex4 = shared_mha(x_work, mask=mask_ex4)\n"
        "w_ex4_h0 = w_ex4[0, 0].detach().numpy()\n\n"
        "fig, ax = plt.subplots(figsize=(5.5, 4.5))\n"
        "mode_label = 'Encoder (mask=None)' if USE_ENCODER_MASK else 'Decoder (causal mask)'\n"
        "sns.heatmap(w_ex4_h0, ax=ax, annot=True, fmt='.2f',\n"
        "            cmap='Greens' if USE_ENCODER_MASK else 'Oranges',\n"
        "            xticklabels=TOKENS, yticklabels=TOKENS, linewidths=0.5, cbar=False)\n"
        "ax.set_title(f'Head 0 attention - {mode_label}')\n"
        "ax.set_xlabel('Key'); ax.set_ylabel('Query'); ax.tick_params(axis='x', rotation=30)\n"
        "plt.tight_layout(); plt.show()\n"
        "print(f'Mode: {mode_label}')\n"
    )
)

cells.append(
    md(
        "#### But wait - why not just pass the encoder output directly as the decoder's starting state?\n\n"
        "There are two problems:\n\n"
        "**Problem 1 - The causal mask cuts off the source.** The decoder still applies its causal mask.\n\n"
        "**Problem 2 - Source and target have different lengths.** You can't stack them as positions.\n\n"
        'The solution is **Cross-Attention**: the decoder queries the encoder output as a separate "table" '
        "at every decoding step.\n"
    )
)

cells.append(
    md(
        "### 13b - Cross-Attention: The Bridge\n\n"
        "In cross-attention:\n\n"
        "$$Q = \\text{decoder state} \\cdot W_Q \\qquad K, V = \\text{encoder output} \\cdot W_K, W_V$$\n\n"
        'The decoder asks: *"Given what I have generated so far ($Q$), which part of the source text ($K$) '
        'is most relevant, and what should I extract from it ($V$)?"*\n'
    )
)

cells.append(
    code(
        "# \u2500\u2500 CrossAttention: Q from decoder, K/V from encoder \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n\n\n"
        "class CrossAttention(nn.Module):\n"
        '    """\n'
        "    Cross-attention layer.\n"
        "    Q  <- decoder's current state\n"
        "    K  <- encoder's output\n"
        "    V  <- encoder's output\n"
        "    No mask on the encoder dimension - the decoder can attend to ANY source position.\n"
        '    """\n\n'
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
        "    def forward(self, decoder_x, encoder_kv):\n"
        '        """\n'
        "        decoder_x : (B, tgt_len, d_model)\n"
        "        encoder_kv: (B, src_len, d_model)\n"
        "        Score matrix: (B, heads, tgt_len, src_len)  - no mask applied.\n"
        '        """\n'
        "        B, tgt_len, _ = decoder_x.shape\n"
        "        src_len = encoder_kv.shape[1]\n"
        "        Q_ca = self.W_Q(decoder_x).reshape(B, tgt_len, self.n_heads, self.d_head).transpose(1, 2)\n"
        "        K_ca = self.W_K(encoder_kv).reshape(B, src_len, self.n_heads, self.d_head).transpose(1, 2)\n"
        "        V_ca = self.W_V(encoder_kv).reshape(B, src_len, self.n_heads, self.d_head).transpose(1, 2)\n"
        "        scores = (Q_ca @ K_ca.transpose(-2, -1)) / math.sqrt(self.d_head)\n"
        "        cross_w = torch.softmax(scores, dim=-1)   # (B, H, tgt, src)\n"
        "        out = cross_w @ V_ca\n"
        "        out = out.transpose(1, 2).reshape(B, tgt_len, self.d_model)\n"
        "        return self.W_O(out), cross_w\n\n\n"
        "# \u2500\u2500 Quick demo: one decoder query token attending to six encoder positions \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n"
        "torch.manual_seed(5)\n"
        "cross_demo = CrossAttention(D_WORK, NUM_HEADS)\n"
        "dec_state_demo = enc_out[:, :1, :]\n"
        "with torch.no_grad():\n"
        "    ca_out_demo, ca_w_demo = cross_demo(dec_state_demo, enc_out)\n\n"
        "print('Cross-attention shapes:')\n"
        "print(f'  Decoder query  : {tuple(dec_state_demo.shape)}  (1 decoder token)')\n"
        "print(f'  Encoder K/V    : {tuple(enc_out.shape)}  (6 source tokens)')\n"
        "print(f'  Score matrix   : {tuple(ca_w_demo.shape)}')\n"
        "print()\n"
        "print('The single decoder query scored all 6 encoder positions.')\n"
        "print('Which source token it attends to is entirely learned by the loss.')\n"
    )
)

cells.append(
    md(
        "### 13c - Encoder-Decoder: The Translator\n\n"
        "A full encoder-decoder wires both halves together with cross-attention in every decoder block. "
        "We train it on a toy task - **reversing a three-word phrase** - to force cross-attention to do "
        "real work.\n\n"
        "```\n"
        'Encoder: ["the", "cat", "sat"]  ->  [ enriched vectors: V_the, V_cat, V_sat ]\n'
        'Decoder: [<BOS>]  -> "sat" ;  [<BOS>, sat]  -> "cat" ;  [<BOS>, sat, cat]  -> "the"\n'
        "```\n"
    )
)

cells.append(
    code(
        "# \u2500\u2500 DecoderBlockWithCrossAttn + MiniEncoderDecoder \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n\n\n"
        "class DecoderBlockWithCrossAttn(nn.Module):\n"
        '    """\n'
        "    Full decoder block - three sub-layers:\n"
        "      1. Causal self-attention  - the decoder looks at its own generated tokens\n"
        "      2. Cross-attention        - the decoder queries the encoder's source map\n"
        "      3. Feed-forward           - per-token nonlinear transformation\n"
        '    """\n\n'
        "    def __init__(self, d_model, n_heads, d_ff):\n"
        "        super().__init__()\n"
        "        self.norm1 = nn.LayerNorm(d_model, eps=1e-5)\n"
        "        self.self_attn = MultiHeadAttention(d_model, n_heads)\n"
        "        self.norm2 = nn.LayerNorm(d_model, eps=1e-5)\n"
        "        self.cross_attn = CrossAttention(d_model, n_heads)\n"
        "        self.norm3 = nn.LayerNorm(d_model, eps=1e-5)\n"
        "        self.ffn = FeedForward(d_model, d_ff)\n\n"
        "    def forward(self, x, encoder_output, causal_mask=None):\n"
        "        sa_out, sa_w = self.self_attn(self.norm1(x), mask=causal_mask)\n"
        "        x = x + sa_out\n"
        "        ca_out, ca_w = self.cross_attn(self.norm2(x), encoder_output)\n"
        "        x = x + ca_out\n"
        "        x = x + self.ffn(self.norm3(x))\n"
        "        return x, sa_w, ca_w\n\n\n"
        "class MiniEncoderDecoder(nn.Module):\n"
        '    """\n'
        "    Encoder-Decoder transformer (T5 / original-Transformer style).\n"
        "    Encoder : bidirectional - produces a frozen source map (K, V for cross-attn)\n"
        "    Decoder : causal self-attn + cross-attn at every block - LM head\n"
        '    """\n\n'
        "    def __init__(self, vocab_size, d_model, n_heads, d_ff, n_layers, max_seq=64):\n"
        "        super().__init__()\n"
        "        self.d_model = d_model\n"
        "        self.emb = nn.Embedding(vocab_size, d_model)\n"
        "        pe = sinusoidal_pe(max_seq, d_model)\n"
        "        self.register_buffer('pe', pe)\n"
        "        self.enc_blocks = nn.ModuleList([\n"
        "            TransformerBlock(d_model, n_heads, d_ff) for _ in range(n_layers)\n"
        "        ])\n"
        "        self.enc_norm = nn.LayerNorm(d_model, eps=1e-5)\n"
        "        self.dec_blocks = nn.ModuleList([\n"
        "            DecoderBlockWithCrossAttn(d_model, n_heads, d_ff) for _ in range(n_layers)\n"
        "        ])\n"
        "        self.dec_norm = nn.LayerNorm(d_model, eps=1e-5)\n"
        "        self.lm_head = nn.Linear(d_model, vocab_size)\n\n"
        "    def encode(self, src_ids):\n"
        "        S_enc = src_ids.shape[1]\n"
        "        x = self.emb(src_ids) + self.pe[:S_enc]\n"
        "        for block in self.enc_blocks:\n"
        "            x, _ = block(x, mask=None)\n"
        "        return self.enc_norm(x)\n\n"
        "    def forward(self, src_ids, tgt_ids):\n"
        '        """\n'
        "        src_ids: (B, src_len) - what the encoder reads\n"
        "        tgt_ids: (B, tgt_len) - decoder input shifted right (teacher-forced)\n"
        "        Returns logits (B, tgt_len, vocab_size) and all cross-attn weight tensors.\n"
        '        """\n'
        "        enc_out_s2s = self.encode(src_ids)\n"
        "        T_dec = tgt_ids.shape[1]\n"
        "        x = self.emb(tgt_ids) + self.pe[:T_dec]\n"
        "        causal_mask_s2s = torch.triu(torch.ones(T_dec, T_dec, dtype=torch.bool), diagonal=1)\n"
        "        causal_mask_s2s = causal_mask_s2s.to(x.device)\n"
        "        all_ca_w = []\n"
        "        for block in self.dec_blocks:\n"
        "            x, _, ca_w = block(x, enc_out_s2s, causal_mask=causal_mask_s2s)\n"
        "            all_ca_w.append(ca_w)\n"
        "        x = self.dec_norm(x)\n"
        "        logits = self.lm_head(x)\n"
        "        return logits, all_ca_w\n\n\n"
        "torch.manual_seed(42)\n"
        "seq2seq_demo = MiniEncoderDecoder(VOCAB_SIZE, D_WORK, NUM_HEADS, D_FF, n_layers=2)\n"
        "print('MiniEncoderDecoder architecture:')\n"
        "print(f'  Encoder: {len(seq2seq_demo.enc_blocks)} x TransformerBlock  (mask=None, bidirectional)')\n"
        "print(f'  Decoder: {len(seq2seq_demo.dec_blocks)} x DecoderBlockWithCrossAttn')\n"
        "print(f'           -> self-attn (causal) + cross-attn + FFN')\n"
        "print(f'  Shared vocab: {VOCAB_SIZE} tokens')\n"
    )
)

cells.append(
    code(
        "# \u2500\u2500 Toy training data: reverse a 3-word phrase \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n"
        '# Source:  ["the", "cat", "sat"]\n'
        '# Target:  ["sat", "cat", "the", "<EOS>"]  (decoder input = ["<BOS>"] + target[:-1])\n\n'
        "BOS_ID, EOS_ID = VOCAB['<BOS>'], VOCAB['<EOS>']\n\n"
        "source_phrases_rev = [\n"
        "    ['the', 'cat', 'sat'], ['the', 'dog', 'ran'],\n"
        "    ['a',   'big', 'cat'], ['the', 'cat', 'ran'],\n"
        "    ['a',   'dog', 'sat'], ['the', 'big', 'dog'],\n"
        "]\n\n"
        "REV_DATA = []\n"
        "for phrase in source_phrases_rev:\n"
        "    src_ids_r = [VOCAB[w] for w in phrase]\n"
        "    rev_ids   = [VOCAB[w] for w in reversed(phrase)]\n"
        "    tgt_in_r  = [BOS_ID] + rev_ids\n"
        "    tgt_out_r = rev_ids + [EOS_ID]\n"
        "    REV_DATA.append((src_ids_r, tgt_in_r, tgt_out_r))\n\n"
        "print(f'Reversal task - {len(REV_DATA)} training pairs:')\n"
        "for src_r, _, tout_r in REV_DATA[:3]:\n"
        "    print(f'  {[IDX2WORD[i] for i in src_r]}  ->  {[IDX2WORD[i] for i in tout_r]}')\n"
        "print('  ...')\n\n\n"
        "def build_rev_batch(data, pad_id=0):\n"
        "    src_max = max(len(s) for s, _, _ in data)\n"
        "    tgt_max = max(len(t) for _, t, _ in data)\n"
        "    srcs, tins, touts = [], [], []\n"
        "    for s, ti, to in data:\n"
        "        srcs.append(s  + [pad_id] * (src_max - len(s)))\n"
        "        tins.append(ti + [pad_id] * (tgt_max - len(ti)))\n"
        "        touts.append(to + [pad_id] * (tgt_max - len(to)))\n"
        "    return (torch.tensor(srcs,  dtype=torch.long),\n"
        "            torch.tensor(tins,  dtype=torch.long),\n"
        "            torch.tensor(touts, dtype=torch.long))\n\n\n"
        "src_rev, tgt_in_rev, tgt_out_rev = build_rev_batch(REV_DATA)\n"
        "print(f'Batch shapes - src: {tuple(src_rev.shape)}  '\n"
        "      f'tgt_in: {tuple(tgt_in_rev.shape)}  tgt_out: {tuple(tgt_out_rev.shape)}')\n"
    )
)

cells.append(
    code(
        "# \u2500\u2500 Train the encoder-decoder on the reversal task \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n"
        "torch.manual_seed(42)\n"
        "seq2seq = MiniEncoderDecoder(VOCAB_SIZE, D_WORK, NUM_HEADS, D_FF, n_layers=2)\n"
        "optimizer_s2s = torch.optim.Adam(seq2seq.parameters(), lr=5e-3)\n"
        "loss_fn_s2s = nn.CrossEntropyLoss(ignore_index=0)\n\n"
        "EPOCHS_S2S = 500\n"
        "s2s_loss_h, s2s_acc_h = [], []\n\n"
        "for epoch in range(EPOCHS_S2S):\n"
        "    seq2seq.train()\n"
        "    optimizer_s2s.zero_grad()\n"
        "    logits_s2s, _ = seq2seq(src_rev, tgt_in_rev)\n"
        "    loss_s2s = loss_fn_s2s(logits_s2s.reshape(-1, VOCAB_SIZE), tgt_out_rev.reshape(-1))\n"
        "    loss_s2s.backward()\n"
        "    torch.nn.utils.clip_grad_norm_(seq2seq.parameters(), 1.0)\n"
        "    optimizer_s2s.step()\n\n"
        "    if (epoch + 1) % 25 == 0:\n"
        "        with torch.no_grad():\n"
        "            preds_s2s = torch.argmax(logits_s2s, dim=-1)\n"
        "            pad_mask_s2s = (tgt_out_rev != 0)\n"
        "            correct = (preds_s2s == tgt_out_rev) & pad_mask_s2s\n"
        "            acc_s2s = correct.float().sum().item() / pad_mask_s2s.float().sum().item()\n"
        "        s2s_loss_h.append(loss_s2s.item())\n"
        "        s2s_acc_h.append(acc_s2s)\n"
        "        if (epoch + 1) % 100 == 0:\n"
        "            print(f'Epoch {epoch+1:4d}  loss={loss_s2s.item():.4f}  acc={acc_s2s:.2%}')\n\n"
        "print('\\nTraining complete.  Greedy decoding results:')\n"
        "seq2seq.eval()\n"
        "for src_r, tin_r, tout_r in REV_DATA:\n"
        "    with torch.no_grad():\n"
        "        lgts, _ = seq2seq(\n"
        "            torch.tensor([src_r],  dtype=torch.long),\n"
        "            torch.tensor([tin_r],  dtype=torch.long),\n"
        "        )\n"
        "    pred_ids = torch.argmax(lgts[0], dim=-1).tolist()\n"
        "    src_w  = [IDX2WORD[i] for i in src_r]\n"
        "    pred_w = [IDX2WORD[i] for i in pred_ids]\n"
        "    gold_w = [IDX2WORD[i] for i in tout_r]\n"
        "    ok = '[ok]' if pred_ids == tout_r else '[xx]'\n"
        "    print(f'  {ok}  source={src_w}  pred={pred_w}  gold={gold_w}')\n"
    )
)

cells.append(
    md(
        "#### Predict first - draw the cross-attention map\n\n"
        'Source: `["the", "cat", "sat"]` -> reversed target: `["sat", "cat", "the"]`.\n\n'
        "| Decoder step | Generating | Highest source attention should be at? |\n"
        "| ------------ | ---------- | --------------------------------------- |\n"
        '| Step 0 (`<BOS>` -> "sat") | "sat" | source[ ? ] |\n'
        '| Step 1 (`"sat"` -> "cat") | "cat" | source[ ? ] |\n'
        '| Step 2 (`"sat cat"` -> "the") | "the" | source[ ? ] |\n'
    )
)

cells.append(
    code(
        "# \u2500\u2500 Cross-attention heatmap: decoder reading the encoder blueprint \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n"
        "test_src_s2s = [VOCAB['the'], VOCAB['cat'], VOCAB['sat']]\n"
        "test_tgt_s2s = [BOS_ID, VOCAB['sat'], VOCAB['cat']]\n\n"
        "seq2seq.eval()\n"
        "with torch.no_grad():\n"
        "    logits_vis, ca_vis = seq2seq(\n"
        "        torch.tensor([test_src_s2s], dtype=torch.long),\n"
        "        torch.tensor([test_tgt_s2s], dtype=torch.long),\n"
        "    )\n\n"
        "src_lbls = [IDX2WORD[i] for i in test_src_s2s]\n"
        "dec_lbls = ['<BOS>->sat', 'sat->cat', 'cat->the']\n"
        "n_dec_layers = len(seq2seq.dec_blocks)\n"
        "fig, axes = plt.subplots(1, n_dec_layers, figsize=(6 * n_dec_layers, 4.2))\n"
        "if n_dec_layers == 1:\n"
        "    axes = [axes]\n\n"
        "for layer_i, (ax, ca_w) in enumerate(zip(axes, ca_vis)):\n"
        "    mean_ca = ca_w[0].mean(dim=0).detach().numpy()   # avg heads -> (tgt, src)\n"
        "    sns.heatmap(mean_ca, ax=ax, annot=True, fmt='.2f',\n"
        "                cmap='YlOrRd', xticklabels=src_lbls, yticklabels=dec_lbls,\n"
        "                linewidths=0.6, vmin=0, vmax=1, cbar_kws={'label': 'cross-attn weight'})\n"
        "    ax.set_title(f'Decoder layer {layer_i}  (mean over {NUM_HEADS} heads)')\n"
        "    ax.set_xlabel('Source position (encoder output)')\n"
        "    ax.set_ylabel('Decoder step -> predicted token')\n"
        "    ax.tick_params(axis='x', rotation=0)\n\n"
        "plt.suptitle(f'Cross-attention: decoder querying the encoder  |  source: {src_lbls}',\n"
        "             fontsize=11, fontweight='bold')\n"
        "plt.tight_layout(); plt.show()\n\n"
        'print(\'If reversal is learned, row 0 (->"sat") should attend most to source[2]="sat".\')\n'
        'print(\'Row 1 (->"cat") should attend most to source[1]="cat".  Etc.\')\n'
        "print('The cross-attention map IS the learned \"look at the right source position\" rule.')\n"
    )
)

cells.append(
    md(
        "### 13d - Why the Industry Moved to Decoder-Only\n\n"
        "| Issue | Encoder-Decoder | Decoder-Only |\n"
        "| ----- | --------------- | ------------ |\n"
        "| Training data | Needs paired input/output sequences | Eats *any* raw text |\n"
        "| KV cache at inference | Two separate caches | One growing cache |\n"
        "| Information routing | Must compress source through cross-attention | Implicit in early layers |\n"
        "| Scaling | Complex orchestration | Simple stack |\n"
    )
)

cells.append(
    code(
        "# \u2500\u2500 Architecture comparison: parameter cost and capability table \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n"
        "torch.manual_seed(0)\n\n"
        "_enc_cmp    = MiniEncoder(VOCAB_SIZE, D_WORK, NUM_HEADS, D_FF, n_layers=2)\n"
        "_dec_cmp    = MiniLM(VOCAB_SIZE, D_WORK, NUM_HEADS, D_FF, n_layers=2)\n"
        "_encdec_cmp = MiniEncoderDecoder(VOCAB_SIZE, D_WORK, NUM_HEADS, D_FF, n_layers=2)\n\n"
        "_dummy6 = torch.tensor([TOKEN_IDS])\n"
        "_dummy3 = torch.tensor([TOKEN_IDS[:3]])\n"
        "with torch.no_grad():\n"
        "    _enc_cmp(_dummy6)\n"
        "    _dec_cmp(_dummy6)\n"
        "    _encdec_cmp(_dummy6, _dummy3)\n\n"
        "p_enc_cmp    = sum(p.numel() for p in _enc_cmp.parameters())\n"
        "p_dec_cmp    = sum(p.numel() for p in _dec_cmp.parameters())\n"
        "p_encdec_cmp = sum(p.numel() for p in _encdec_cmp.parameters())\n\n"
        "rows_cmp = [\n"
        "    ('',                'Encoder-Only',  'Decoder-Only',     'Encoder-Decoder'),\n"
        "    ('Self-attn mask',  'None (bidir)',  'Causal',           'Enc: none / Dec: causal'),\n"
        "    ('Cross-attn',      'No',            'No',               f'{len(_encdec_cmp.dec_blocks)} layers'),\n"
        "    ('Parameters',      f'{p_enc_cmp:,}', f'{p_dec_cmp:,}', f'{p_encdec_cmp:,}'),\n"
        "    ('Training data',   'Labelled corpus', 'Any raw text',  'Paired sequences'),\n"
        "    ('Real examples',   'BERT, RoBERTa',   'GPT, LLaMA',    'T5, BART'),\n"
        "]\n\n"
        "col_w = [20, 18, 20, 26]\n"
        "sep = '  ' + '-' * (sum(col_w) + 6)\n"
        "print(sep)\n"
        "for i, row in enumerate(rows_cmp):\n"
        "    line = '  ' + '  '.join(f'{c:<{w}}' for c, w in zip(row, col_w))\n"
        "    print(line)\n"
        "    if i == 0:\n"
        "        print(sep)\n"
        "print(sep)\n"
        "print()\n"
        "overhead = p_encdec_cmp - p_dec_cmp\n"
        "print(f'Cross-attention overhead : {overhead:+,} params  '\n"
        "      f'(+{overhead/p_dec_cmp*100:.1f}% vs decoder-only at same depth/width)')\n"
        "print()\n"
        "print('At GPT-3 scale (175B parameters), that overhead is non-trivial -')\n"
        "print('one reason the industry consolidated around decoder-only for general LLMs.')\n"
    )
)

# ═══════════════════════════════════════════════════════════════════════════════
# PART 10 : GPT-2 (placed last per the Keras notebook ordering)
# ═══════════════════════════════════════════════════════════════════════════════

cells.append(
    md(
        "---\n\n## Part 10 - Cracking Open distilgpt2\n\n"
        "Now we scale from our toy model to a real one: **DistilGPT-2** with:\n\n"
        "- 6 transformer blocks\n"
        "- 12 attention heads per block\n"
        "- d_model = 768\n"
        "- ~82M parameters\n\n"
        "We will:\n"
        "1. Load the model and inspect its architecture\n"
        "2. Run a forward pass with `output_attentions=True` to capture all 6 layers x 12 heads\n"
        "3. Visualise the attention patterns\n"
        "4. Plot the next-token probability distribution\n"
    )
)

cells.append(
    code(
        "# \u2500\u2500 Load DistilGPT-2 (PyTorch backend) \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n"
        "from transformers import GPT2LMHeadModel, GPT2Tokenizer\n\n"
        "gpt_tokenizer = GPT2Tokenizer.from_pretrained('distilgpt2')\n"
        "gpt2 = GPT2LMHeadModel.from_pretrained('distilgpt2')\n"
        "gpt2.eval()\n\n"
        "cfg = gpt2.config\n"
        "print('=== DistilGPT-2 Architecture ===')\n"
        "print(f'  n_layer         : {cfg.n_layer}')\n"
        "print(f'  n_head          : {cfg.n_head}')\n"
        "print(f'  n_embd (d_model): {cfg.n_embd}')\n"
        "print(f'  vocab_size      : {cfg.vocab_size}')\n"
        "print(f'  n_positions     : {cfg.n_positions}  (max context)')\n"
        "print()\n"
        "n_params_gpt = sum(p.numel() for p in gpt2.parameters())\n"
        "print(f'  Total parameters: {n_params_gpt:,}  (~{n_params_gpt/1e6:.1f}M)')\n"
        "print()\n"
        "print('Top-level modules:')\n"
        "for name, module in gpt2.named_children():\n"
        "    print(f'  {name}: {module.__class__.__name__}')\n"
        "print()\n"
        "print('First transformer block sub-modules:')\n"
        "block0 = gpt2.transformer.h[0]\n"
        "for name, mod in block0.named_children():\n"
        "    print(f'  h[0].{name}: {mod.__class__.__name__}')\n"
    )
)

cells.append(
    code(
        "# \u2500\u2500 Run inference with all attentions captured \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n"
        "PROMPT = 'The cat sat on the'\n"
        "inputs = gpt_tokenizer(PROMPT, return_tensors='pt')\n"
        "input_ids = inputs['input_ids']\n\n"
        "gpt_tokens = [gpt_tokenizer.decode([int(t)]) for t in input_ids[0]]\n"
        "print(f'Prompt       : {PROMPT!r}')\n"
        "print(f'GPT-2 tokens : {gpt_tokens}')\n"
        "print(f'Token IDs    : {input_ids[0].tolist()}')\n"
        "print()\n\n"
        "with torch.no_grad():\n"
        "    out = gpt2(**inputs, output_attentions=True)\n\n"
        "print(f'Number of attention layers returned: {len(out.attentions)}')\n"
        "print(f'Each layer shape: {tuple(out.attentions[0].shape)}')\n"
        "print(f'  (batch=1, n_heads=12, seq={len(gpt_tokens)}, seq={len(gpt_tokens)})')\n"
        "print()\n\n"
        "next_logits = out.logits[0, -1, :]\n"
        "next_probs = torch.softmax(next_logits, dim=-1)\n"
        "top10 = torch.topk(next_probs, k=10)\n\n"
        "print(f'Top-10 next token predictions after \"{PROMPT}\":')\n"
        "for rank, (tid, prob) in enumerate(zip(top10.indices.tolist(), top10.values.tolist()), 1):\n"
        "    tok = gpt_tokenizer.decode([int(tid)])\n"
        "    print(f'  {rank:2d}. {tok!r:<20} {float(prob):.4f}')\n"
    )
)

cells.append(
    code(
        "# \u2500\u2500 Attention pattern visualisation - all 6 layers, mean over heads \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n"
        "n_layers_gpt = len(out.attentions)\n"
        "seq_len_gpt = len(gpt_tokens)\n\n"
        "fig, axes = plt.subplots(2, 3, figsize=(14, 8))\n"
        "axes = axes.flatten()\n\n"
        "for layer_idx, (layer_attn, ax) in enumerate(zip(out.attentions, axes)):\n"
        "    mean_attn = layer_attn[0].mean(dim=0).detach().numpy()\n"
        "    sns.heatmap(mean_attn, ax=ax, cmap='viridis',\n"
        "                xticklabels=gpt_tokens, yticklabels=gpt_tokens,\n"
        "                linewidths=0.3, cbar_kws={'label': 'attn weight'})\n"
        "    ax.set_title(f'Layer {layer_idx}  (mean over 12 heads)', fontsize=9)\n"
        "    ax.set_xlabel('Key'); ax.set_ylabel('Query')\n"
        "    ax.tick_params(axis='x', rotation=45, labelsize=8)\n"
        "    ax.tick_params(axis='y', labelsize=8)\n\n"
        "plt.suptitle('DistilGPT-2: mean attention patterns across all 6 layers',\n"
        "             fontsize=11, fontweight='bold')\n"
        "plt.tight_layout(); plt.show()\n"
    )
)

cells.append(
    code(
        "# \u2500\u2500 Head diversity - compare individual heads in layer 0 \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n"
        "layer0_attn = out.attentions[0][0].detach().numpy()   # (12, seq, seq)\n\n"
        "fig, axes = plt.subplots(1, 4, figsize=(16, 4))\n"
        "for h, ax in enumerate(axes):\n"
        "    sns.heatmap(layer0_attn[h], ax=ax, cmap='Blues',\n"
        "                xticklabels=gpt_tokens, yticklabels=gpt_tokens,\n"
        "                linewidths=0.3, annot=True, fmt='.2f', annot_kws={'size': 8})\n"
        "    ax.set_title(f'Layer 0  Head {h}', fontsize=9)\n"
        "    ax.set_xlabel('Key'); ax.set_ylabel('Query')\n"
        "    ax.tick_params(axis='x', rotation=45, labelsize=7)\n"
        "    ax.tick_params(axis='y', labelsize=7)\n\n"
        "plt.suptitle('DistilGPT-2 Layer 0: first 4 heads show diverse specialisation',\n"
        "             fontsize=11, fontweight='bold')\n"
        "plt.tight_layout(); plt.show()\n"
    )
)

cells.append(
    code(
        "# \u2500\u2500 Next-token probability bar chart \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n"
        "top_n = 15\n"
        "topk_gpt = torch.topk(next_probs, k=top_n)\n"
        "top_tokens_gpt = [gpt_tokenizer.decode([int(t)]) for t in topk_gpt.indices.tolist()]\n"
        "top_probs_gpt = topk_gpt.values.detach().numpy()\n\n"
        "fig, ax = plt.subplots(figsize=(9, 5))\n"
        "colors = ['gold'] + ['steelblue'] * (top_n - 1)\n"
        "ax.barh(top_tokens_gpt[::-1], top_probs_gpt[::-1], color=colors[::-1], alpha=0.85)\n"
        "ax.set_xlabel('Probability')\n"
        "ax.set_title(f'DistilGPT-2 next-token probabilities\\nPrompt: \"{PROMPT}\"',\n"
        "             fontsize=11)\n"
        "for bar, prob in zip(ax.patches, top_probs_gpt[::-1]):\n"
        "    ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height() / 2,\n"
        "            f'{prob:.4f}', va='center', fontsize=8)\n"
        "plt.tight_layout(); plt.show()\n\n"
        "print(f'Top prediction: {top_tokens_gpt[0]!r}  (prob={top_probs_gpt[0]:.4f})')\n"
    )
)

cells.append(
    code(
        "# \u2500\u2500 Full autoregressive generation with distilgpt2 \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n\n\n"
        "def gpt2_generate(prompt: str, max_new_tokens: int = 15, temperature: float = 0.8, top_k: int = 50):\n"
        '    """Generate tokens one at a time, printing each step."""\n'
        "    gpt2.eval()\n"
        "    ids = gpt_tokenizer.encode(prompt, return_tensors='pt')\n"
        "    print(f'Prompt: {prompt!r}')\n"
        "    print(f'Starting IDs: {ids[0].tolist()}')\n"
        "    print()\n\n"
        "    generated = ids\n"
        "    for step in range(max_new_tokens):\n"
        "        with torch.no_grad():\n"
        "            out_step = gpt2(generated)\n"
        "        logits_step = out_step.logits[0, -1, :].clone()\n\n"
        "        # Temperature + top-k filtering\n"
        "        logits_step = logits_step / max(temperature, 1e-8)\n"
        "        if top_k > 0:\n"
        "            top_vals, top_idx = torch.topk(logits_step, k=top_k)\n"
        "            filtered = torch.full_like(logits_step, float('-inf'))\n"
        "            filtered.scatter_(0, top_idx, top_vals)\n"
        "            logits_step = filtered\n\n"
        "        probs_step = torch.softmax(logits_step, dim=-1)\n"
        "        next_id = int(torch.multinomial(probs_step, num_samples=1).item())\n"
        "        next_tok = gpt_tokenizer.decode([next_id])\n\n"
        "        print(f'  Step {step+1:2d} - token ID {next_id:5d}  {next_tok!r:<20}  '\n"
        "              f'p={probs_step[next_id].item():.4f}')\n\n"
        "        generated = torch.cat([generated, torch.tensor([[next_id]])], dim=1)\n\n"
        "        if next_id == gpt_tokenizer.eos_token_id:\n"
        "            print('  [EOS - stopping]')\n"
        "            break\n\n"
        "    final_text = gpt_tokenizer.decode(generated[0].tolist(), skip_special_tokens=True)\n"
        "    print()\n"
        "    print(f'Final: {final_text!r}')\n"
        "    return final_text\n\n\n"
        "gpt2_generate('The cat sat on the', max_new_tokens=12, temperature=0.7)\n"
    )
)

# ═══════════════════════════════════════════════════════════════════════════════
# Summary cell
# ═══════════════════════════════════════════════════════════════════════════════

cells.append(
    md(
        "---\n\n## Summary - The Complete Transformer Journey\n\n"
        "We've traced every component from raw words to generated tokens:\n\n"
        "| Step | Component | What happens |\n"
        "| ---- | --------- | ------------ |\n"
        "| 1  | **Tokeniser** | Text -> integer IDs |\n"
        "| 2  | **Token Embedding** | IDs -> dense vectors in semantic space |\n"
        "| 3  | **Positional Encoding** | Add position signal (sin/cos or RoPE rotation) |\n"
        "| 4  | **Q/K/V Projection** | Three learned views of each token vector |\n"
        "| 5  | **Scaled Dot-Product Attention** | Soft dictionary lookup - compute relevance scores |\n"
        "| 6  | **Multi-Head Attention** | H parallel attention views concatenated |\n"
        "| 7  | **Feed-Forward Network** | Per-token nonlinear transformation |\n"
        "| 8  | **LayerNorm + Residuals** | Stabilise activations, guarantee gradient flow |\n"
        "| 9  | **Repeat x L** | Stack L transformer blocks |\n"
        "| 10 | **LM Head** | Project to vocabulary -> logits -> softmax -> distribution |\n"
        "| 11 | **W_V Relevance Filter** | W_V extracts the task-specific payload |\n"
        "| 12 | **Causal Triangle** | Position n accumulates n+1 tokens; depth chains richness |\n"
        "| 13 | **Encoder Architecture** | mask=None gives bidirectional context |\n"
        "| 14 | **Cross-Attention** | Q from decoder, K/V from frozen encoder |\n"
        "| 15 | **Encoder-Decoder** | Source encoded once; decoder cross-attends at every step |\n"
        "| 16 | **Architecture Comparison** | Decoder-only won at scale; encoder stays essential |\n"
        "| 17 | **GPT-2 Internals** | A real model, cracked open |\n"
        "| 18 | **Autoregressive loop** | Sample next token -> append -> repeat |\n\n"
        "### Key insights to keep\n\n"
        "- **RoPE** encodes position by rotating Q and K; only the relative gap survives in dot-products\n"
        "- **Attention is O(n^2)** in sequence length - this is why long-context models are expensive\n"
        "- **Multi-head attention** lets each head specialise on a different relationship type\n"
        "- **Residual connections** make depth practical - gradients always have a direct path home\n"
        "- **Temperature** is the single most intuitive control knob at inference time\n"
        "- **W_V is a relevance filter** - it extracts the task-specific slice of each token's information\n"
        "- **The causal triangle means depth compounds** - position n at layer L has processed a chain "
        "of enriched representations from all n earlier tokens\n"
        "- **Encoder = mask removed** - `mask=None` gives every token a full-sentence view from layer 1\n"
        "- **Cross-attention decouples source and target** - Q from the decoder re-queries a frozen "
        "encoder map at every step; no compression bottleneck\n"
        "- **Decoder-only scales cleanly** - no paired data needed\n"
    )
)

print(f"Total cells built: {len(cells)}")
