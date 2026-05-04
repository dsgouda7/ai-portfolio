"""
gen_ch09_dvc_pipeline.py
Generates: ../img/ch09_dvc_pipeline.png

Shows a DVC (Data Version Control) pipeline with versioned stages:
Raw data → preprocess → train → evaluate

Each stage tracks inputs, outputs, parameters, and git-like versioning.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
import matplotlib.patches as mpatches

OUT_PATH = os.path.join(os.path.dirname(__file__), "..", "img", "ch09_dvc_pipeline.png")

# Ensure output directory exists
os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

# Create figure
fig, ax = plt.subplots(figsize=(16, 10), facecolor="#fafafa")
ax.set_facecolor("#fafafa")
ax.set_xlim(0, 16)
ax.set_ylim(0, 10)
ax.axis("off")

# Title
ax.text(8, 9.3, "DVC Pipeline: Data Version Control", 
        ha="center", va="center", fontsize=26, fontweight="bold", color="#1a1a1a")
ax.text(8, 8.7, "Reproducible ML: Track data, code, and models like Git tracks code",
        ha="center", va="center", fontsize=12, color="#666666", style="italic")

# Define stage colors
DATA_COLOR = "#3b82f6"      # Blue
PROCESS_COLOR = "#8b5cf6"   # Purple  
TRAIN_COLOR = "#f59e0b"     # Amber
EVAL_COLOR = "#10b981"      # Green
ARROW_COLOR = "#64748b"     # Gray

# Pipeline stages
stages = [
    {
        "name": "Raw Data",
        "x": 2.5,
        "y": 5.5,
        "color": DATA_COLOR,
        "script": "fetch_data.py",
        "inputs": ["raw/train.csv\n(500MB)"],
        "outputs": ["data/raw/\ntrain.csv"],
        "params": [],
        "hash": "a7f9e3d"
    },
    {
        "name": "Preprocess",
        "x": 6.0,
        "y": 5.5,
        "color": PROCESS_COLOR,
        "script": "preprocess.py",
        "inputs": ["data/raw/\ntrain.csv"],
        "outputs": ["data/processed/\ntrain_clean.csv"],
        "params": ["min_length: 10", "max_length: 512"],
        "hash": "b2e8f4c"
    },
    {
        "name": "Train",
        "x": 9.5,
        "y": 5.5,
        "color": TRAIN_COLOR,
        "script": "train.py",
        "inputs": ["data/processed/\ntrain_clean.csv"],
        "outputs": ["models/\nmodel.pt"],
        "params": ["lr: 5e-5", "batch: 32", "epochs: 3"],
        "hash": "c3f9a5d"
    },
    {
        "name": "Evaluate",
        "x": 13.0,
        "y": 5.5,
        "color": EVAL_COLOR,
        "script": "evaluate.py",
        "inputs": ["models/\nmodel.pt", "data/test/\ntest.csv"],
        "outputs": ["metrics.json", "confusion.png"],
        "params": ["threshold: 0.5"],
        "hash": "d4a8b6e"
    }
]

# Draw pipeline stages
box_width = 2.2
box_height = 4.5

for stage in stages:
    # Main stage box
    ax.add_patch(FancyBboxPatch((stage["x"] - box_width/2, stage["y"] - box_height/2), 
                                box_width, box_height,
                                boxstyle="round,pad=0.1",
                                facecolor=stage["color"], edgecolor="white", linewidth=2.5, alpha=0.95))
    
    # Stage name
    ax.text(stage["x"], stage["y"] + 1.9, stage["name"], 
            ha="center", va="center", fontsize=14, fontweight="bold", color="white")
    
    # Script name
    ax.add_patch(Rectangle((stage["x"] - box_width/2 + 0.1, stage["y"] + 1.3), 
                           box_width - 0.2, 0.35,
                           facecolor="#ffffff", alpha=0.3, edgecolor="white", linewidth=1))
    ax.text(stage["x"], stage["y"] + 1.47, stage["script"],
            ha="center", va="center", fontsize=8, color="white", 
            family="monospace", fontweight="bold")
    
    # DVC hash
    ax.text(stage["x"], stage["y"] + 0.9, f'DVC: {stage["hash"]}',
            ha="center", va="center", fontsize=7, color="white", 
            family="monospace", style="italic")
    
    # Inputs section
    ax.text(stage["x"], stage["y"] + 0.5, "Inputs:", 
            ha="center", va="center", fontsize=9, color="white", fontweight="bold")
    for i, inp in enumerate(stage["inputs"]):
        ax.text(stage["x"], stage["y"] + 0.1 - i*0.3, inp,
                ha="center", va="center", fontsize=7, color="white", family="monospace")
    
    # Outputs section
    y_offset = -0.2 - len(stage["inputs"]) * 0.3
    ax.text(stage["x"], stage["y"] + y_offset, "Outputs:", 
            ha="center", va="center", fontsize=9, color="white", fontweight="bold")
    for i, out in enumerate(stage["outputs"]):
        ax.text(stage["x"], stage["y"] + y_offset - 0.3 - i*0.3, out,
                ha="center", va="center", fontsize=7, color="white", family="monospace")
    
    # Parameters section (if any)
    if stage["params"]:
        y_offset = y_offset - 0.3 - len(stage["outputs"]) * 0.3
        ax.text(stage["x"], stage["y"] + y_offset, "Params:", 
                ha="center", va="center", fontsize=9, color="white", fontweight="bold")
        for i, param in enumerate(stage["params"]):
            ax.text(stage["x"], stage["y"] + y_offset - 0.3 - i*0.25, param,
                    ha="center", va="center", fontsize=7, color="white", family="monospace")

# Draw arrows between stages
arrow_y = 5.5
arrow_style = "->,head_width=0.3,head_length=0.3"

arrows = [
    {"from": stages[0]["x"] + box_width/2, "to": stages[1]["x"] - box_width/2, "label": ""},
    {"from": stages[1]["x"] + box_width/2, "to": stages[2]["x"] - box_width/2, "label": ""},
    {"from": stages[2]["x"] + box_width/2, "to": stages[3]["x"] - box_width/2, "label": ""},
]

for arrow in arrows:
    ax.add_patch(FancyArrowPatch((arrow["from"] + 0.1, arrow_y), 
                                (arrow["to"] - 0.1, arrow_y),
                                arrowstyle=arrow_style,
                                color=ARROW_COLOR, linewidth=3))

# DVC commands panel
cmd_y = 1.8
ax.add_patch(FancyBboxPatch((0.8, cmd_y - 0.6), 14.4, 1.3,
                            boxstyle="round,pad=0.1",
                            facecolor="#1e293b", edgecolor="#475569", linewidth=2))

ax.text(8, cmd_y + 0.5, "🔧 DVC Commands to Reproduce This Pipeline",
        ha="center", va="center", fontsize=13, fontweight="bold", color="white")

commands = [
    "# Initialize DVC and track data",
    "$ dvc init",
    "$ dvc add data/raw/train.csv",
    "$ git add data/raw/train.csv.dvc .dvc/",
    "",
    "# Define and run pipeline",
    "$ dvc stage add -n preprocess -d data/raw/train.csv -o data/processed/train_clean.csv python preprocess.py",
    "$ dvc repro  # Run entire pipeline",
    "",
    "# Version everything with Git",
    "$ git add dvc.yaml dvc.lock",
    "$ git commit -m 'Pipeline with DVC hash c3f9a5d'",
]

# Split commands into two columns
left_commands = commands[:6]
right_commands = commands[6:]

col1_x = 1.2
col2_x = 8.5
cmd_y_start = cmd_y + 0.2

for i, cmd in enumerate(left_commands):
    color = "#94a3b8" if cmd.startswith("#") else "#e2e8f0"
    weight = "bold" if cmd.startswith("#") else "normal"
    ax.text(col1_x, cmd_y_start - i*0.17, cmd,
            ha="left", va="center", fontsize=7.5, color=color, 
            family="monospace", fontweight=weight)

for i, cmd in enumerate(right_commands):
    color = "#94a3b8" if cmd.startswith("#") else "#e2e8f0"
    weight = "bold" if cmd.startswith("#") else "normal"
    ax.text(col2_x, cmd_y_start - i*0.17, cmd,
            ha="left", va="center", fontsize=7.5, color=color, 
            family="monospace", fontweight=weight)

# Benefits callout
benefits_y = 0.5
ax.add_patch(FancyBboxPatch((0.8, benefits_y - 0.3), 14.4, 0.7,
                            boxstyle="round,pad=0.08",
                            facecolor="#fef3c7", edgecolor="#f59e0b", linewidth=2))

ax.text(8, benefits_y + 0.2, "✨ Benefits: Reproduce any experiment 6 months later with ONE command: dvc repro",
        ha="center", va="center", fontsize=11, color="#92400e", fontweight="bold")

ax.text(8, benefits_y - 0.1, "DVC tracks: data versions (like Git LFS), pipeline DAG, metrics, and remote storage (S3, Azure, GCS)",
        ha="center", va="center", fontsize=9, color="#78350f")

plt.tight_layout()
plt.savefig(OUT_PATH, dpi=150, facecolor="#fafafa", edgecolor="none")
print(f"✓ Saved: {OUT_PATH}")
