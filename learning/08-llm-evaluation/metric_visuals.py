"""Measure and compose the two numerical panels inside the MAI metric illustrations."""

from __future__ import annotations

from collections import Counter
from io import BytesIO
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
from nltk.stem import PorterStemmer
from nltk.translate.meteor_score import align_words, meteor_score
from PIL import Image, ImageDraw, ImageFont, PngImagePlugin
from rouge_score import rouge_scorer
import torch
from torch.nn import functional as F
from transformers import AutoModel, AutoTokenizer

from evaluation_metrics import (
    ENCODER_ID, ENCODER_REVISION, HERE, RESULT_PATH, LessonTokenizer,
    download_snapshot, load_eval_model, prepare_wordnet, prompt_for,
)
from evaluation_shared import RIVERSIDE_CASES, tokenize


IMAGES = HERE / "images"
EVIDENCE = IMAGES / "01-metric-visual-evidence.json"
REFERENCE = "the pendant hides a letter"
PARAPHRASE = "the pendant conceals a note"
NAVY, TEAL, PURPLE, GOLD = "#183047", "#087f83", "#6240a2", "#e5b42c"


def font(size: int):
    return ImageFont.truetype(font_manager.findfont("DejaVu Sans"), size)


def tile(draw, bounds, text, color, size=22, outline=None):
    draw.rounded_rectangle(bounds, radius=9, fill=color, outline=outline, width=3)
    draw.text(((bounds[0] + bounds[2]) / 2, (bounds[1] + bounds[3]) / 2),
              text, font=font(size), fill="white", anchor="mm")


def save_checked_page(page, output: Path, *, numerical=False) -> None:
    metadata = PngImagePlugin.PngInfo()
    metadata.add_text("LayoutModel", "MAI-Image-2.6")
    metadata.add_text("CodePanel", "Exact code-drawn matches or recorded model values; not invented by MAI.")
    if numerical:
        metadata.add_text("NumericalPanel", "Deterministic plot of recorded model values, not image-model output.")
    page.save(output, pnginfo=metadata)


def chunks(matches: list[tuple[int, int]]) -> int:
    ordered = sorted(matches)
    return 0 if not ordered else 1 + sum(
        later[0] != earlier[0] + 1 or later[1] != earlier[1] + 1
        for earlier, later in zip(ordered, ordered[1:])
    )


def measure() -> dict:
    prepare_wordnet()
    inserted = "the pendant hides a small letter"
    reordered = "a letter hides the pendant"
    rouge = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], tokenizer=LessonTokenizer())
    insertion_scores = rouge.score(REFERENCE, inserted)
    reference_pairs = Counter(zip(tokenize(REFERENCE), tokenize(REFERENCE)[1:]))
    answer_pairs = Counter(zip(tokenize(inserted), tokenize(inserted)[1:]))
    meteor_cases = {}
    for name, answer in (
        ("synonym", "the pendant conceals a letter"),
        ("ordered", REFERENCE),
        ("reversed", "letter a hides pendant the"),
    ):
        matches, _, _ = align_words(tokenize(answer), tokenize(REFERENCE))
        meteor_cases[name] = {
            "answer": answer, "matches": matches, "chunks": chunks(matches),
            "score": meteor_score([tokenize(REFERENCE)], tokenize(answer)),
        }
    encoder_path = download_snapshot(ENCODER_ID, ENCODER_REVISION)
    tokenizer = AutoTokenizer.from_pretrained(encoder_path, local_files_only=True)
    encoder = AutoModel.from_pretrained(encoder_path, local_files_only=True).eval()
    vectors = []
    tokens = []
    for text in (REFERENCE, PARAPHRASE):
        inputs = tokenizer(text, return_tensors="pt")
        with torch.inference_mode():
            hidden = encoder(**inputs).last_hidden_state[0]
        vectors.append(F.normalize(hidden, dim=-1)[1:-1])
        tokens.append(tokenizer.convert_ids_to_tokens(inputs.input_ids[0][1:-1].tolist()))
    similarities = (vectors[1] @ vectors[0].T).tolist()
    del encoder
    model, tokenizer, identity = load_eval_model()
    case = RIVERSIDE_CASES[0]
    prompt_ids = tokenizer(prompt_for(case, tokenizer), add_special_tokens=False, return_tensors="pt").input_ids
    targets = tokenizer(case["reference"], add_special_tokens=False, return_tensors="pt").input_ids
    joined = torch.cat((prompt_ids, targets), dim=1)
    with torch.inference_mode():
        logits = model(joined, use_cache=False).logits[0, prompt_ids.shape[1] - 1:-1]
        losses = F.cross_entropy(logits, targets[0], reduction="none")
    saved = json.loads(RESULT_PATH.read_bytes())["items"][0]
    assert math.isclose(losses.sum().item(), saved["reference_nll_sum"], abs_tol=1e-4)
    stemmer = PorterStemmer()
    return {
        "reference": REFERENCE,
        "rouge": {
            "inserted": inserted,
            "precision": insertion_scores["rougeL"].precision,
            "recall": insertion_scores["rougeL"].recall,
            "f1": insertion_scores["rougeL"].fmeasure,
            "shared_adjacent_pairs": sum((reference_pairs & answer_pairs).values()),
            "reference_adjacent_pairs": sum(reference_pairs.values()),
            "reordered": reordered,
            "reordered_rouge1": rouge.score(REFERENCE, reordered)["rouge1"].fmeasure,
            "reordered_lcs_length": round(rouge.score(REFERENCE, reordered)["rougeL"].recall * 5),
        },
        "bleu": {"repeated": "letter letter letter", "clipped_matches": 1, "produced": 3,
                 "short": "letter", "reference_length": 5, "short_length": 1,
                 "short_brevity_penalty": math.exp(1 - 5)},
        "meteor": {"cases": meteor_cases,
                   "stem_example": {word: stemmer.stem(word) for word in ("hides", "hiding")}},
        "bertscore": {
            "encoder_id": ENCODER_ID, "encoder_revision": ENCODER_REVISION,
            "layers": 6, "reference": REFERENCE, "candidate": PARAPHRASE,
            "reference_tokens": tokens[0], "candidate_tokens": tokens[1],
            "cosine_similarities": similarities,
            "note": "Same layer-six contextual cosine construction as BERTScore's example plot; special tokens hidden.",
        },
        "perplexity": {
            "model": identity, "case": case["id"], "prompt": prompt_for(case, tokenizer),
            "reference": case["reference"],
            "token_pieces": [tokenizer.decode([token]) for token in targets[0].tolist()],
            "token_ids": targets[0].tolist(), "token_losses": losses.tolist(),
            "token_probabilities": losses.neg().exp().tolist(),
            "mean_loss": losses.mean().item(), "perplexity": losses.mean().exp().item(),
            "note": "Actual fixed-reference tokens conditioned on the exact prompt; prompt tokens are not scored.",
        },
    }


def save_panel(page: Image.Image, figure, bounds: tuple[int, int, int, int], output: Path) -> None:
    buffer = BytesIO()
    try:
        figure.savefig(buffer, format="png", dpi=160, facecolor="white", bbox_inches="tight")
    finally:
        plt.close(figure)
    with Image.open(BytesIO(buffer.getvalue())) as plot:
        plot = plot.convert("RGB")
        width, height = bounds[2] - bounds[0], bounds[3] - bounds[1]
        plot.thumbnail((width, height), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (width, height), "white")
        canvas.paste(plot, ((width - plot.width) // 2, (height - plot.height) // 2))
        page.paste(canvas, bounds[:2])
    save_checked_page(page, output, numerical=True)


def compose_word_matches(evidence: dict) -> None:
    for name in ("rouge", "bleu", "meteor"):
        raw = IMAGES / f"01-metrics-{name}-mai.png"
        if not raw.exists():
            (IMAGES / f"01-metrics-{name}.png").rename(raw)
            print("Preserved original MAI layout:", raw.name)
    with Image.open(IMAGES / "01-metrics-rouge-mai.png") as original:
        page = original.convert("RGB")
    draw = ImageDraw.Draw(page)
    reference = evidence["reference"].split()
    inserted = evidence["rouge"]["inserted"].split()
    for top, color in ((260, "#f1f8f6"), (632, "#f6f5f8")):
        draw.rectangle((530, top, 1230, top + 157), fill=color)
        for i in range(5):
            target = i if i < 4 else 5
            draw.line((592 + i * 105, top + 62, 592 + target * 105, top + 83), fill=GOLD, width=5)
        for i, word in enumerate(reference):
            tile(draw, (545 + i * 105, top + 15, 640 + i * 105, top + 62), word, TEAL, 20)
        for i, word in enumerate(inserted):
            tile(draw, (545 + i * 105, top + 83, 640 + i * 105, top + 130),
                 word, PURPLE if word != "small" else "#a391c6", 20)
    draw.rectangle((530, 450, 1268, 589), fill="#f2f8f9")
    pairs = [" ".join(pair) for pair in zip(reference, reference[1:])]
    for i, pair in enumerate(pairs):
        x = 538 + i * 155
        tile(draw, (x, 462, x + 145, 509), pair, TEAL, 16)
        tile(draw, (x, 529, x + 145, 576), pair if i < 3 else "no match",
             PURPLE if i < 3 else "#b54f50", 16, GOLD if i < 3 else None)
    draw.rectangle((530, 831, 1115, 948), fill="#fbf4f3")
    for i, word in enumerate(reference):
        tile(draw, (540 + i * 105, 839, 635 + i * 105, 886), word, TEAL, 20, GOLD if i >= 3 else None)
    for i, word in enumerate(evidence["rouge"]["reordered"].split()):
        tile(draw, (540 + i * 105, 897, 635 + i * 105, 944), word, PURPLE, 20, GOLD if i < 2 else None)
    save_checked_page(page, IMAGES / "01-metrics-rouge.png")

    with Image.open(IMAGES / "01-metrics-bleu-mai.png") as original:
        page = original.convert("RGB")
    draw = ImageDraw.Draw(page)
    tile(draw, (1008, 407, 1214, 485), "letter", PURPLE, 28)
    groups = [
        ["the", "pendant", "hides", "a", "letter"],
        ["the pendant", "hides a"], ["the pendant hides"], ["the pendant hides a"],
    ]
    for (mask_left, right, left), labels in zip(
        ((43, 339, 50), (367, 705, 382), (732, 1055, 745), (1087, 1491, 1100)),
        groups, strict=True,
    ):
        draw.rectangle((mask_left, 839, right, 912), fill="white")
        if len(labels) == 5:
            for i, label in enumerate(labels):
                tile(draw, (left + i * 55, 852, left + i * 55 + 51, 897), label, TEAL, 10)
        elif len(labels) == 2:
            for i, label in enumerate(labels):
                tile(draw, (left + i * 135, 852, left + i * 135 + 130, 897), label, TEAL, 13)
        else:
            tile(draw, (left, 852, left + 270, 897), labels[0], TEAL, 16)
    save_checked_page(page, IMAGES / "01-metrics-bleu.png")

    with Image.open(IMAGES / "01-metrics-meteor-mai.png") as original:
        page = original.convert("RGB")
    draw = ImageDraw.Draw(page)
    draw.rectangle((228, 675, 1413, 840), fill="white")
    for i in range(5):
        draw.line((342 + i * 240, 728, 342 + (4 - i) * 240, 785), fill="#e57f76", width=5)
    for i, word in enumerate(reference):
        tile(draw, (237 + i * 240, 680, 447 + i * 240, 729), word, TEAL)
    for i, word in enumerate(evidence["meteor"]["cases"]["reversed"]["answer"].split()):
        tile(draw, (237 + i * 240, 785, 447 + i * 240, 834), word, PURPLE)
    save_checked_page(page, IMAGES / "01-metrics-meteor.png")


def compose(evidence: dict) -> None:
    compose_word_matches(evidence)
    data = evidence["bertscore"]
    values = np.asarray(data["cosine_similarities"])
    fig, axis = plt.subplots(figsize=(8, 6))
    colors = axis.imshow(values, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    axis.set_xticks(range(len(data["reference_tokens"])), data["reference_tokens"])
    axis.set_yticks(range(len(data["candidate_tokens"])), data["candidate_tokens"])
    axis.set(xlabel="Reference tokens", ylabel="Answer tokens", title="Measured contextual-token similarity")
    for i, row in enumerate(values):
        for j, value in enumerate(row):
            axis.text(j, i, f"{value:.2f}", ha="center", va="center",
                      color="white" if value > .75 else "#142b42", fontsize=12)
        j = int(np.argmax(row))
        axis.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, fill=False, edgecolor="#e8aa24", lw=2))
    fig.colorbar(colors, ax=axis, label="Cosine similarity (not correctness)")
    fig.tight_layout()
    with Image.open(IMAGES / "01-metrics-bertscore-mai.png") as raw:
        save_panel(raw.convert("RGB"), fig, (55, 240, 850, 965), IMAGES / "01-metrics-bertscore.png")

    data = evidence["perplexity"]
    probabilities = np.asarray(data["token_probabilities"])
    losses = np.asarray(data["token_losses"])
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(13, 5), gridspec_kw={"height_ratios": [1.1, 1]})
    for i, (piece, probability, loss) in enumerate(zip(
        data["token_pieces"][:6], probabilities[:6], losses[:6], strict=True,
    )):
        top.text((i + .5) / 6, .57, repr(piece), ha="center", fontsize=13)
        top.text((i + .5) / 6, .28, f"probability {probability:.3g}\nloss {loss:.2f}",
                 ha="center", va="center", fontsize=11, color="#263e53")
    top.set(title="First six actual token pieces: less expected means more loss (values rounded)")
    top.set_axis_off()
    bottom.bar(range(1, len(losses) + 1), losses, color="#4b8d9f")
    bottom.set(xlabel=f"All {len(losses)} reference-token positions (not whole words)",
               ylabel="Token loss", title=(
                   f"Average loss {data['mean_loss']:.3f}; "
                   f"exp(average loss) gives perplexity {data['perplexity']:.3f} before rounding"
               ))
    bottom.set_xticks([1, 6, 11, 16, 21])
    fig.tight_layout()
    with Image.open(IMAGES / "01-metrics-perplexity-mai.png") as raw:
        page = raw.convert("RGB")
    draw = ImageDraw.Draw(page)
    draw.rectangle((34, 24, 136, 193), fill="white")
    draw.rectangle((815, 129, 972, 157), fill="#f7f5fc")
    for x, word in ((822, "the"), (866, "pendant"), (923, "...")):
        tile(draw, (x, 132, x + (50 if word == "pendant" else 36), 155), word, PURPLE, 8)
    save_panel(page, fig, (65, 245, 1470, 935), IMAGES / "01-metrics-perplexity.png")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--measure", action="store_true", help="Record actual model values before generating artwork.")
    args = parser.parse_args()
    if args.measure:
        previous_threads = torch.get_num_threads()
        try:
            torch.set_num_threads(4)
            data = measure()
        finally:
            torch.set_num_threads(previous_threads)
        EVIDENCE.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print("Saved checked visual evidence:", EVIDENCE)
    else:
        compose(json.loads(EVIDENCE.read_bytes()))
        print("Inserted measured plots into the MAI layouts.")
