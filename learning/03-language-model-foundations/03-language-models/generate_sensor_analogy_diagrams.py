"""Rebuild the registry, sensor-profile and observatory diagrams locally."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.patches import Circle, Ellipse, FancyBboxPatch
import numpy as np


HERE = Path(__file__).resolve().parent
TOKEN_IMAGES = HERE.parent / "02-tokenization-and-embeddings" / "images"
NAVY = "#172b4d"
TEAL = "#007f86"
PURPLE = "#7950a3"
ORANGE = "#bf5700"


def canvas(title: str):
    figure, axis = plt.subplots(figsize=(14, 7.875))
    figure.patch.set_facecolor("#fffaf1")
    axis.set(xlim=(0, 1), ylim=(0, 1))
    axis.axis("off")
    axis.text(0.5, 0.95, title, ha="center", color=NAVY, fontsize=22, weight="bold")
    return figure, axis


def card(axis: Axes, x: float, y: float, width: float, height: float,
         text: str, color: str = TEAL):
    axis.add_patch(FancyBboxPatch(
        (x, y), width, height, boxstyle="round,pad=0.012",
        edgecolor=color, facecolor="white", linewidth=2,
    ))
    axis.text(x + width / 2, y + height / 2, text, ha="center", va="center",
              color=NAVY, fontsize=13, linespacing=1.7)


def arrow(axis: Axes, start: tuple[float, float], end: tuple[float, float],
          color: str = NAVY):
    axis.annotate("", xy=end, xytext=start,
                  arrowprops=dict(arrowstyle="->", color=color, lw=2))


def save(figure, path: Path):
    figure.savefig(path, dpi=140, facecolor=figure.get_facecolor())
    plt.close(figure)
    print(f"Saved {path.name}")


def registry():
    figure, axis = canvas("An ID identifies. A sensor profile supplies features.")
    card(axis, 0.03, 0.57, 0.20, 0.20, "Token: ' cat'\nRegistry ID: 3797")
    card(axis, 0.33, 0.57, 0.28, 0.20,
         "Stored profile\n[e1, e2, ..., eD]\nOne reading per channel")
    card(axis, 0.72, 0.57, 0.24, 0.20,
         "Later processing\nuses the profile\nwith sentence context", PURPLE)
    arrow(axis, (0.25, 0.67), (0.31, 0.67))
    arrow(axis, (0.63, 0.67), (0.70, 0.67))
    axis.text(0.5, 0.47, "GPT-2's tokenizer: 50,257 registry entries; ID size is not meaning",
              ha="center", color=NAVY, fontsize=13)
    card(axis, 0.03, 0.18, 0.28, 0.17, "the loud bark\nwoke the cat")
    card(axis, 0.38, 0.18, 0.24, 0.17, "Same bark ID\nSame starting profile")
    card(axis, 0.69, 0.18, 0.28, 0.17, "the rough bark\ncovered the tree")
    arrow(axis, (0.33, 0.265), (0.36, 0.265))
    arrow(axis, (0.67, 0.265), (0.64, 0.265))
    axis.text(0.5, 0.07, "Abstract learned channels, not literal temperature or mass readings.",
              ha="center", color=ORANGE, fontsize=13)
    save(figure, TOKEN_IMAGES / "02-03-id-selects-a-sensor-profile.png")


def numeric_trace():
    table = np.array([[2., 0.], [1., 3.], [0., 4.]])
    weights = np.array([[0., 1.], [1., 0.], [1., 1.]])
    profile = table[1]
    scores = weights @ profile
    assert profile.tolist() == [1., 3.] and scores.tolist() == [3., 1., 4.]
    figure, axis = canvas("One token. Two layers. Different numerical objects.")
    card(axis, 0.02, 0.45, 0.16, 0.29, "cat\nRegistry ID 1")
    card(axis, 0.27, 0.45, 0.24, 0.29,
         "Layer 1: lookup\nSensor profile [1, 3]\nTHIS is the embedding")
    card(axis, 0.60, 0.45, 0.17, 0.29, "Layer 2\nSeparate linear\npredictor", PURPLE)
    card(axis, 0.85, 0.45, 0.13, 0.29, "Scores\n"
         + "\n".join(f"{word}: {score:g}" for word, score in zip(
             ("the", "cat", "sat"), scores)), PURPLE)
    for start, end in ((0.20, 0.25), (0.53, 0.58), (0.79, 0.83)):
        arrow(axis, (start, 0.595), (end, 0.595))
    axis.text(0.5, 0.29, "Local registry: the=0 -> [2, 0]    cat=1 -> [1, 3]    sat=2 -> [0, 4]",
              ha="center", color=NAVY, fontsize=13)
    axis.text(0.5, 0.18, "Two sensor channels, not two named meanings. Scores are not embeddings.",
              ha="center", color=TEAL, fontsize=13)
    axis.text(0.5, 0.08, "Hand-picked teaching numbers, not a trained result or a physical measurement.",
              ha="center", color=ORANGE, fontsize=13)
    save(figure, TOKEN_IMAGES / "02-07-word-id-to-embedding-to-scores.png")


def learning():
    figure, axis = canvas("Practice adjusts the profile, not the registry number.")
    card(axis, 0.03, 0.59, 0.25, 0.19, "Fixed cat ID\nCat's four-channel profile")
    card(axis, 0.39, 0.59, 0.23, 0.19, "Separate predictor\nWord scores", PURPLE)
    card(axis, 0.73, 0.59, 0.23, 0.19, "Check against sat\nLoss", ORANGE)
    arrow(axis, (0.30, 0.685), (0.37, 0.685))
    arrow(axis, (0.64, 0.685), (0.71, 0.685))
    card(axis, 0.12, 0.23, 0.29, 0.18, "Update cat's profile\nKeep its ID", TEAL)
    card(axis, 0.59, 0.23, 0.29, 0.18, "Update predictor weights", PURPLE)
    arrow(axis, (0.84, 0.56), (0.265, 0.43), ORANGE)
    arrow(axis, (0.84, 0.56), (0.735, 0.43), ORANGE)
    axis.text(0.5, 0.10, "Isolated plain-SGD probe: unused dog and bird input profiles stay unchanged.",
              ha="center", color=NAVY, fontsize=13)
    save(figure, TOKEN_IMAGES / "02-05-practice-nudges-rows.png")


def object_signatures():
    figure, axis = canvas("Names identify. Sensor profiles let you compare.")
    axis.text(0.5, 0.86, "Which discovered objects resemble Earth?",
              ha="center", color=NAVY, fontsize=16)
    card(axis, 0.03, 0.43, 0.25, 0.35,
         "Catalogue labels\nEarth\nObject-417\nObject-892")
    arrow(axis, (0.30, 0.605), (0.35, 0.605))
    table = axis.table(
        cellText=[
            ["Earth", "r_E", "T_E", "a_E"],
            ["Object-417", "r_A", "T_A", "a_A"],
            ["Object-892", "r_B", "T_B", "a_B"],
        ],
        colLabels=["Object", "Size", "Temperature", "Atmosphere"],
        colWidths=[0.28, 0.20, 0.26, 0.26],
        cellLoc="center", bbox=(0.38, 0.43, 0.59, 0.35),
    )
    table.auto_set_font_size(False)
    table.set_fontsize(13)
    for (row, _), cell in table.get_celld().items():
        cell.set_edgecolor(TEAL)
        cell.set_text_props(color="white" if row == 0 else NAVY)
        cell.set_facecolor(TEAL if row == 0 else "#e0f2f1" if row == 1 else "white")
    axis.text(0.155, 0.35, "Which record?\nNot what it is like.",
              ha="center", va="center", color=NAVY, fontsize=13)
    axis.text(0.675, 0.35, "Symbols stand for readings, not invented measurement values.",
              ha="center", va="center", color=NAVY, fontsize=11)
    card(axis, 0.38, 0.13, 0.59, 0.15,
         "Compare each candidate's readings with Earth's.\n"
         "Match channels and account for their units and scales.", PURPLE)
    axis.text(0.5, 0.04,
              "A representation analogy, not a prediction task or proof of habitability.",
              ha="center", color=ORANGE, fontsize=13)
    save(figure, HERE / "images" / "03a-object-sensor-signatures.png")


def cbow_profile_training():
    table = np.array([[2., 0.], [1., 3.], [0., 4.]])
    context = table[[0, 2]].mean(axis=0)
    head = np.array([[1., 0., 0.], [0., 2., 1.]])
    scores = context @ head
    assert context.tolist() == [1., 2.] and scores.tolist() == [1., 4., 2.]
    figure, axis = canvas("Prediction trains profiles. It is not the profile itself.")
    axis.text(0.5, 0.85, "Word-only training question: the ___ sat",
              ha="center", color=NAVY, fontsize=16)
    card(axis, 0.03, 0.59, 0.25, 0.19,
         "Fetch the and sat profiles\nAverage: [1, 2]")
    card(axis, 0.39, 0.59, 0.23, 0.19,
         "Separate predictor\n2 features -> 3 scores", PURPLE)
    card(axis, 0.73, 0.59, 0.23, 0.19,
         "Scores [the, cat, sat]\n[1, 4, 2]", PURPLE)
    arrow(axis, (0.30, 0.685), (0.37, 0.685))
    arrow(axis, (0.64, 0.685), (0.71, 0.685))
    card(axis, 0.35, 0.32, 0.30, 0.14,
         "Check scores against cat\nCalculate loss", ORANGE)
    arrow(axis, (0.845, 0.57), (0.67, 0.39), ORANGE)
    card(axis, 0.03, 0.09, 0.28, 0.14,
         "Update fetched profiles\nKeep token IDs fixed")
    card(axis, 0.69, 0.09, 0.28, 0.14,
         "Update predictor weights", PURPLE)
    arrow(axis, (0.42, 0.30), (0.17, 0.25), ORANGE)
    arrow(axis, (0.58, 0.30), (0.83, 0.25), ORANGE)
    axis.text(0.5, 0.025,
              "Hand-picked forward-pass numbers; training feedback is schematic, not a measured update.",
              ha="center", color=ORANGE, fontsize=12)
    save(figure, HERE / "images" / "03a-cbow-profile-training.png")


def observatory():
    figure, axis = canvas("Encoder-decoder: read a system, then write a likely report.")
    card(axis, 0.02, 0.47, 0.28, 0.34, "", TEAL)
    axis.add_patch(Ellipse((0.16, 0.68), 0.20, 0.10, fill=False, edgecolor=TEAL))
    for x, y in ((0.07, 0.69), (0.16, 0.72), (0.25, 0.66)):
        axis.add_patch(Circle((x, y), 0.012, color=TEAL))
    axis.text(0.16, 0.54, "Ordered observing log\nProfiles read one at a time",
              ha="center", color=NAVY, fontsize=12)
    card(axis, 0.39, 0.47, 0.22, 0.34,
         "Encoder briefing h\nA compact learned state\nNot a raw-profile average", PURPLE)
    card(axis, 0.70, 0.47, 0.28, 0.34,
         "Decoder's likely report\nAbout a newly observed object\nOne word, then the next", ORANGE)
    arrow(axis, (0.32, 0.64), (0.37, 0.64))
    arrow(axis, (0.63, 0.64), (0.68, 0.64))
    axis.text(0.16, 0.34, "Actual source:\nthe -> sleepy -> cat",
              ha="center", color=TEAL, fontsize=13)
    axis.text(0.50, 0.34, "Actual handoff:\nfinal encoder state",
              ha="center", color=PURPLE, fontsize=13)
    axis.text(0.84, 0.34, "Training answer:\nsat -> on -> <eos>",
              ha="center", color=ORANGE, fontsize=13)
    axis.text(0.5, 0.16, "The galaxy is the analogy. This model predicts vocabulary tokens, not physical properties.",
              ha="center", color=NAVY, fontsize=12)
    axis.text(0.5, 0.06, "Only the final briefing reaches the writer here. Attention will later let it revisit the log.",
              ha="center", color=NAVY, fontsize=12)
    save(figure, HERE / "images" / "03a-encoder-decoder-observatory.png")


def toy_embedding_lookup():
    figure, axis = canvas("Lookup selects existing rows. It does not create another table.")
    axis.text(0.5, 0.85, "Symbolic values stand for the random numbers printed in the toy.",
              ha="center", color=NAVY, fontsize=14)
    card(axis, 0.025, 0.48, 0.22, 0.24,
         "Input IDs [B, C] = [2, 2]\nQuestion A: [0, 2]\nQuestion B: [2, 0]")
    axis.text(0.455, 0.76, "Stored table [V, D] = [3, 2]",
              ha="center", color=TEAL, fontsize=13, weight="bold")
    table = axis.table(
        cellText=[["0: the", "[t0, t1]"], ["1: cat", "[c0, c1]"], ["2: sat", "[s0, s1]"]],
        colLabels=["ID / word", "D = 2 values"], cellLoc="center",
        bbox=(0.32, 0.46, 0.27, 0.25),
    )
    table.auto_set_font_size(False)
    table.set_fontsize(13)
    for (row, _), cell in table.get_celld().items():
        cell.set_edgecolor(TEAL)
        cell.set_text_props(color="white" if row == 0 else NAVY)
        cell.set_facecolor(TEAL if row == 0 else "#f3f4f6" if row == 2 else "#e0f2f1")
    axis.text(0.815, 0.76, "Fetched [B, C, D] = [2, 2, 2]",
              ha="center", color=PURPLE, fontsize=13, weight="bold")
    card(axis, 0.66, 0.53, 0.31, 0.17,
         "Question A\n[[t0, t1], [s0, s1]]", PURPLE)
    card(axis, 0.66, 0.27, 0.31, 0.17,
         "Question B\n[[s0, s1], [t0, t1]]", PURPLE)
    arrow(axis, (0.26, 0.60), (0.30, 0.60))
    arrow(axis, (0.61, 0.60), (0.64, 0.615))
    arrow(axis, (0.61, 0.49), (0.64, 0.355))
    axis.text(0.455, 0.36, "cat is stored but not fetched.\n"
              "The table has not changed.", ha="center", color=NAVY, fontsize=13)
    axis.text(0.5, 0.14, "embedding(IDs) selects the same values as embedding.weight[IDs].",
              ha="center", color=TEAL, fontsize=15, weight="bold")
    axis.text(0.5, 0.06, "Each question keeps its own word order here. CBOW averaging happens later.",
              ha="center", color=NAVY, fontsize=13)
    save(figure, HERE / "images" / "03a-toy-embedding-lookup.png")


def toy_model_ownership():
    figure, axis = canvas("Fixed input IDs. Trainable model parameters.")
    axis.add_patch(FancyBboxPatch(
        (0.22, 0.29), 0.75, 0.53, boxstyle="round,pad=0.012",
        edgecolor=NAVY, facecolor="#eef4fa", linewidth=2,
    ))
    axis.text(0.60, 0.775, "ToyCBOW owns both child layers",
              ha="center", color=NAVY, fontsize=16, weight="bold")
    card(axis, 0.025, 0.56, 0.15, 0.14, "Integer IDs\n[[0, 2], [2, 0]]")
    card(axis, 0.25, 0.54, 0.20, 0.18,
         "self.embedding\nweight [3, 2]\nTrainable Parameter")
    card(axis, 0.51, 0.54, 0.18, 0.18,
         "Lookup [2, 2, 2]\nMean [2, 2]\nRecorded operations", PURPLE)
    card(axis, 0.75, 0.54, 0.19, 0.18,
         "self.head\nTrainable weights\nScores [2, 3]", PURPLE)
    arrow(axis, (0.19, 0.63), (0.23, 0.63))
    arrow(axis, (0.47, 0.63), (0.49, 0.63))
    arrow(axis, (0.71, 0.63), (0.73, 0.63))
    card(axis, 0.25, 0.33, 0.29, 0.13,
         "embedding.weight.grad\nAccumulate into saved ID rows\n0 and 2; row 1 gets zero", ORANGE)
    card(axis, 0.75, 0.33, 0.19, 0.13, "Loss checks scores\nagainst cat IDs [1, 1]", ORANGE)
    arrow(axis, (0.845, 0.52), (0.845, 0.48), ORANGE)
    arrow(axis, (0.73, 0.395), (0.56, 0.395), ORANGE)
    axis.text(0.645, 0.44, "backward()", ha="center", color=ORANGE, fontsize=12)
    axis.text(0.645, 0.345, "through head,\nmean, lookup",
              ha="center", va="center", color=ORANGE, fontsize=11)
    axis.text(0.10, 0.44, "No gradients\nor updates\nto integer IDs",
              ha="center", va="center", color=NAVY, fontsize=12)
    card(axis, 0.22, 0.06, 0.75, 0.14,
         "SGD(model.parameters()) discovers embedding.weight and head.weight.\n"
         "step() changes stored parameters, not IDs or previously returned vectors.", PURPLE)
    arrow(axis, (0.60, 0.27), (0.60, 0.22), PURPLE)
    axis.text(0.5, 0.015, "The next forward call fetches updated rows from the SAME embedding object.",
              ha="center", color=TEAL, fontsize=12)
    save(figure, HERE / "images" / "03a-toy-model-ownership.png")


def toy_parameter_registration():
    figure, axis = canvas("Where does embedding.weight become a model parameter?")
    axis.text(0.5, 0.85, "Attribute assignment calls nn.Module.__setattr__; parameter discovery follows the registries.",
              ha="center", color=NAVY, fontsize=13)
    card(axis, 0.025, 0.56, 0.28, 0.21,
         "Inside nn.Embedding\nself.weight = nn.Parameter(...)\nregister_parameter('weight', ...)")
    card(axis, 0.38, 0.56, 0.24, 0.21,
         "embedding._parameters\n'weight' -> table [3, 2]\nNo table copy")
    card(axis, 0.70, 0.56, 0.275, 0.21,
         "Inside ToyCBOW\nself.embedding = embedding\nRegister child in _modules", PURPLE)
    arrow(axis, (0.32, 0.665), (0.36, 0.665))
    arrow(axis, (0.64, 0.665), (0.68, 0.665))
    card(axis, 0.70, 0.23, 0.275, 0.21,
         "model.named_parameters()\nWalk child parameter registries\nYield 'embedding.weight'", PURPLE)
    arrow(axis, (0.835, 0.54), (0.835, 0.46), PURPLE)
    card(axis, 0.025, 0.23, 0.595, 0.21,
         "SGD(model.parameters())\nKeeps that SAME Parameter in optimizer.param_groups\n"
         "step() changes its values in place", ORANGE)
    arrow(axis, (0.68, 0.335), (0.64, 0.335), ORANGE)
    axis.text(0.5, 0.13, "The toy checks object identity at every hop. There is only one trainable table.",
              ha="center", color=NAVY, fontsize=14)
    axis.text(0.5, 0.055, "Registration discovers weights. Backward fills gradients. Step changes values.",
              ha="center", color=TEAL, fontsize=15, weight="bold")
    save(figure, HERE / "images" / "03a-toy-parameter-registration.png")


if __name__ == "__main__":
    registry()
    numeric_trace()
    learning()
    object_signatures()
    cbow_profile_training()
    toy_embedding_lookup()
    toy_model_ownership()
    toy_parameter_registration()
    observatory()
