"""
create_ai_infra_img_dirs.py
Create missing img/ directories in AI Infrastructure track chapters.
gpu_architecture already has img/; creates it for the other 4.
"""
import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AI_INFRA_ROOT = os.path.join(REPO_ROOT, "notes", "ai_infrastructure")

CHAPTERS = [
    "memory_and_compute_budgets",
    "quantization_and_precision",
    "parallelism_and_distributed_training",
    "inference_optimization",
]

for chapter in CHAPTERS:
    img_dir = os.path.join(AI_INFRA_ROOT, chapter, "img")
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)
        print(f"Created: {img_dir}")
    else:
        print(f"Already exists: {img_dir}")
