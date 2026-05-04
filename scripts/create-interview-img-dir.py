"""
create_interview_img_dir.py
Create the img/ directory under notes/interview_guides/ (referenced by guides but missing).
"""
import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMG_DIR = os.path.join(REPO_ROOT, "notes", "interview_guides", "img")

if not os.path.exists(IMG_DIR):
    os.makedirs(IMG_DIR)
    print(f"Created: {IMG_DIR}")
else:
    print(f"Already exists: {IMG_DIR}")
