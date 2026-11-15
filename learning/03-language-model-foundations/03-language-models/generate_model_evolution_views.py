"""Rebuild the six conceptual views from the shared cat example, without a model API."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch

from generate_sensor_analogy_diagrams import NAVY, ORANGE, PURPLE, TEAL, arrow, canvas, card
from language_data import CAT_SENTENCE, CAT_WORDS


HERE = Path(__file__).resolve().parent
IMAGES = HERE / "images"
FONT = HERE.parent / "01-sequence-memory" / "images" / "kalam" / "Kalam-Regular.ttf"


def words(axis, values, y, *, x=0.05, width=0.90, color=TEAL, positions=False):
    step = width / len(values)
    for index, word in enumerate(values):
        label = f"{word}\nposition {index}" if positions else word
        card(axis, x + index * step, y, step * 0.76, 0.12, label, color)
    return [x + index * step + step * 0.38 for index in range(len(values))]


def evolution():
    figure, axis = canvas("Same words. Different questions and routes.")
    words(axis, CAT_WORDS, 0.73)
    stages = (
        ("Counts", "What followed\nmat?"),
        ("Ordered window", "Give recent\nwords seats"),
        ("Running state", "Carry the\nearlier cat clue"),
        ("LSTM", "Keep content;\nchoose a view"),
        ("Reader / writer", "Read a source;\nwrite an answer"),
        ("Attention", "Consult source\nstates again"),
    )
    for index, (title, detail) in enumerate(stages):
        x = 0.035 + index * 0.16
        card(axis, x, 0.32, 0.12, 0.23, f"{title}\n{detail}", TEAL if index < 3 else PURPLE)
        if index < len(stages) - 1:
            arrow(axis, (x + 0.134, 0.435), (x + 0.146, 0.435))
    axis.text(0.5, 0.15, "Keep the sentence familiar. Change only the question or access route.",
              ha="center", color=NAVY, fontsize=16)
    axis.text(0.5, 0.06, "Gap filling, next-word prediction and word reversal are different tasks.",
              ha="center", color=ORANGE, fontsize=13)
    return figure


def blind_window():
    figure, axis = canvas("Same window, different required next words")
    for animal, answer, y, color in (
        ("cat", "purring", 0.61, TEAL), ("dog", "panting", 0.32, ORANGE),
    ):
        phrase = list(CAT_WORDS)
        phrase[1] = animal
        words(axis, phrase, y, x=0.03, width=0.72, color=color)
        axis.add_patch(FancyBboxPatch(
            (0.38, y - 0.025), 0.355, 0.17, boxstyle="round,pad=0.012",
            fill=False, edgecolor=PURPLE, linewidth=3,
        ))
        card(axis, 0.80, y, 0.16, 0.12, f"next: {answer}", color)
        arrow(axis, (0.755, y + 0.06), (0.78, y + 0.06))
    axis.text(0.555, 0.80, "Only the last three words enter this window.", ha="center",
              color=PURPLE, fontsize=14)
    axis.text(0.5, 0.16, "on the mat is identical. The animal clue is outside the window.",
              ha="center", color=NAVY, fontsize=16)
    axis.text(0.5, 0.06, "Made-up rule: cats finish purring; dogs finish panting. Not a language benchmark.",
              ha="center", color=ORANGE, fontsize=12)
    return figure


def running_note():
    figure, axis = canvas("A running note is a route, not a promise of recall.")
    centers = words(axis, CAT_WORDS, 0.68)
    shades = ("#91a1a7", TEAL, "#4d9a9e", "#7da7aa", "#9eb4b6", "#bcc6c7")
    for index, (center, color) in enumerate(zip(centers, shades, strict=True)):
        detail = "cat clue" if index == 1 else "context"
        card(axis, center - 0.052, 0.36, 0.104, 0.14, f"h{index + 1}\n{detail}", color)
        arrow(axis, (center, 0.66), (center, 0.52), color)
        if index < len(centers) - 1:
            arrow(axis, (center + 0.068, 0.43), (centers[index + 1] - 0.068, 0.43), color)
    axis.text(0.5, 0.20, "Later words can preserve or overwrite the clue from cat.",
              ha="center", color=NAVY, fontsize=16)
    axis.text(0.5, 0.09, "Conceptual fading, not measured distances. The notebook traces the actual run.",
              ha="center", color=ORANGE, fontsize=12)
    return figure


def one_card():
    figure, axis = canvas("One final card crosses the reader / writer handoff.")
    centers = words(axis, CAT_WORDS, 0.70)
    for previous, following in zip(centers, centers[1:]):
        arrow(axis, (previous + 0.071, 0.76), (following - 0.071, 0.76))
    card(axis, 0.10, 0.36, 0.28, 0.20,
         "Learned final state\nOne fixed-size summary\nNot mean pooling", TEAL)
    card(axis, 0.58, 0.36, 0.29, 0.20,
         "Writer\nCannot reopen the earlier\nsource states", PURPLE)
    axis.plot((centers[-1], centers[-1], 0.24), (0.68, 0.625, 0.625), color=NAVY, linewidth=2)
    arrow(axis, (0.24, 0.625), (0.24, 0.59))
    arrow(axis, (0.40, 0.46), (0.56, 0.46))
    answer = " ".join(reversed(CAT_WORDS)) + " <eos>"
    axis.text(0.5, 0.22, f"Required word-order answer: {answer}",
              ha="center", color=NAVY, fontsize=15)
    axis.text(0.5, 0.10, "A summary can work. Measure when its restricted access becomes limiting.",
              ha="center", color=ORANGE, fontsize=13)
    return figure


def looks_back():
    figure, axis = canvas("A writing step can consult every contextual reader state.")
    centers = words(axis, CAT_WORDS, 0.69, positions=True)
    for index, center in enumerate(centers):
        card(axis, center - 0.052, 0.46, 0.104, 0.12, f"reader h{index + 1}", TEAL)
        arrow(axis, (center, 0.67), (center, 0.60), TEAL)
        arrow(axis, (0.44, 0.35), (center, 0.435), "#869ca8")
    card(axis, 0.28, 0.16, 0.31, 0.17,
         "Current writer state\nQuery the source notes", PURPLE)
    card(axis, 0.73, 0.16, 0.22, 0.17,
         "Reader states + query weights\nmake fresh context", TEAL)
    arrow(axis, (0.61, 0.245), (0.71, 0.245))
    axis.text(0.5, 0.075,
              "Illustrative access paths, not measured weights. The two occurrences of the have different addresses.",
              ha="center", color=ORANGE, fontsize=11)
    return figure


def practice_and_generation():
    figure, axis = canvas("Same source; different previous inputs while writing.")
    words(axis, CAT_WORDS, 0.73)
    axis.text(0.26, 0.61, "Practice: correct previous word", ha="center", color=TEAL, fontsize=16)
    axis.text(0.75, 0.61, "Generation: its own previous guess", ha="center", color=PURPLE, fontsize=16)
    card(axis, 0.07, 0.38, 0.38, 0.17, "<bos> -> ask for mat\nthen input mat -> ask for the", TEAL)
    card(axis, 0.55, 0.38, 0.38, 0.17,
         "<bos> -> suppose it guesses cat\nthen input cat -> ask again", PURPLE)
    axis.text(0.5, 0.23, "The source stays fixed. A wrong guess changes the next writer input.",
              ha="center", color=NAVY, fontsize=15)
    axis.text(0.5, 0.11, "Word-reversal illustration: cat is a hypothetical wrong guess, not a measured output.",
              ha="center", color=ORANGE, fontsize=12)
    return figure


VIEWS = (
    ("03-evo-01-model-evolution-map.png", "shared-word-route", evolution,
     "The cat sentence stays fixed while counts, windows, recurrence and source access answer different questions."),
    ("03-evo-02-fixed-window-blind-spot.png", "animal-outside-window", blind_window,
     "Cat and dog prefixes share on the mat but require purring and panting; the three-word window loses the animal clue."),
    ("03-evo-03-fading-notecard.png", "running-context-boundary", running_note,
     "The cat clue passes through successive running states; fading is conceptual rather than measured."),
    ("03-evo-04-one-card-bottleneck.png", "fixed-summary-access", one_card,
     "A reader processes the full cat sentence but passes only its final summary to a writer asked to reverse the words."),
    ("03-evo-05-writer-looks-back.png", "source-state-access", looks_back,
     "A writer can query all six contextual reader states; repeated the tokens occupy distinct source positions."),
    ("03-evo-06-practice-vs-generation.png", "previous-word-alignment", practice_and_generation,
     "Correct previous words and hypothetical generated words give different inputs while reversing the same cat sentence."),
)


def main():
    font_manager.fontManager.addfont(str(FONT))
    family = font_manager.FontProperties(fname=FONT).get_name()
    prompts = []
    with plt.rc_context({"font.family": family, "path.sketch": (0.7, 100, 1.5)}):
        for filename, anchor, render, description in VIEWS:
            figure = render()
            try:
                figure.savefig(
                    IMAGES / filename, dpi=140, facecolor=figure.get_facecolor(),
                    metadata={
                        "ExampleSentence": CAT_SENTENCE, "SemanticAnchor": anchor,
                        "Measured": "false", "Generator": Path(__file__).name,
                    },
                )
            finally:
                plt.close(figure)
            prompts.append({
                "filename": filename, "semantic_anchor": anchor, "aspect_ratio": "16:9",
                "prompt": description,
                "acceptance_criteria": "Use the canonical cat words; keep causal routes and source positions explicit.",
                "prohibited_content": "No fabricated probabilities, gradients or performance; no digit or color-clue story.",
                "alt_text": description,
            })
            print(f"Saved {filename}")
    (IMAGES / "03-cat-example-prompts.json").write_text(
        json.dumps(prompts, indent=2) + "\n", encoding="utf-8",
    )


if __name__ == "__main__":
    main()
