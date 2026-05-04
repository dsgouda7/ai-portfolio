#!/usr/bin/env python3
"""
Audit and clean unreferenced images and generator scripts in notes/01-ml.

Tasks:
1. Find all image files (*.png, *.gif, *.jpg, *.jpeg, *.webp) in img/ folders
2. Find all Python generator scripts (gen_*.py or in gen_scripts/ folders)
3. Search ALL .md and .ipynb files in notes/ to see if each image is referenced
4. Delete unreferenced image files
5. For generator scripts: identify which images they generate
6. Delete generator scripts ONLY if ALL images they generate are unreferenced
7. Report what was deleted
"""

import os
import re
import json
from pathlib import Path
from typing import Set, Dict, List, Tuple

# Base directories
REPO_ROOT = Path(__file__).parent.parent
ML_DIR = REPO_ROOT / "notes" / "01-ml"
NOTES_DIR = REPO_ROOT / "notes"

# Image extensions to look for
IMAGE_EXTS = {".png", ".gif", ".jpg", ".jpeg", ".webp"}


def find_all_images() -> List[Path]:
    """Find all image files in img/ folders under notes/01-ml."""
    images = []
    for ext in IMAGE_EXTS:
        images.extend(ML_DIR.rglob(f"img/*{ext}"))
    return sorted(images)


def find_all_generator_scripts() -> List[Path]:
    """Find all Python generator scripts in notes/01-ml."""
    scripts = []
    # Scripts in gen_scripts/ folders
    scripts.extend(ML_DIR.rglob("gen_scripts/*.py"))
    # Scripts matching gen_*.py pattern
    for script in ML_DIR.rglob("gen_*.py"):
        if script not in scripts:
            scripts.append(script)
    return sorted(scripts)


def find_all_markdown_and_notebooks() -> List[Path]:
    """Find all .md and .ipynb files in notes/."""
    files = []
    files.extend(NOTES_DIR.rglob("*.md"))
    files.extend(NOTES_DIR.rglob("*.ipynb"))
    return sorted(files)


def is_image_referenced(image_path: Path, search_files: List[Path]) -> bool:
    """Check if image is referenced in any of the search files."""
    image_name = image_path.name
    
    for file_path in search_files:
        try:
            content = file_path.read_text(encoding="utf-8", errors="ignore")
            
            # For notebooks, extract cell content
            if file_path.suffix == ".ipynb":
                try:
                    nb_data = json.loads(content)
                    content = ""
                    for cell in nb_data.get("cells", []):
                        if "source" in cell:
                            if isinstance(cell["source"], list):
                                content += "".join(cell["source"])
                            else:
                                content += cell["source"]
                except json.JSONDecodeError:
                    continue
            
            # Check for image reference (name only, as relative paths vary)
            if image_name in content:
                return True
                
        except Exception as e:
            print(f"  Warning: Could not read {file_path}: {e}")
            continue
    
    return False


def analyze_script_outputs(script_path: Path) -> Set[str]:
    """Analyze a generator script to identify which image files it generates."""
    try:
        content = script_path.read_text(encoding="utf-8", errors="ignore")
    except Exception as e:
        print(f"  Warning: Could not read script {script_path}: {e}")
        return set()
    
    output_files = set()
    
    # Look for common patterns:
    # 1. plt.savefig("path/to/file.png")
    # 2. fig.savefig("path/to/file.gif")
    # 3. save_path = "something.png"
    # 4. output_file = "file.gif"
    # 5. write_gif("file.gif", ...)
    
    patterns = [
        r'savefig\(["\']([^"\']+\.(?:png|gif|jpg|jpeg|webp))["\']',
        r'write_gif\(["\']([^"\']+\.gif)["\']',
        r'to_file\(["\']([^"\']+\.(?:png|gif|jpg|jpeg|webp))["\']',
        r'(?:save_path|output_file|output_path|filename)\s*=\s*["\']([^"\']+\.(?:png|gif|jpg|jpeg|webp))["\']',
        r'open\(["\']([^"\']+\.(?:png|gif|jpg|jpeg|webp))["\'],\s*["\']wb["\']',
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, content, re.IGNORECASE)
        for match in matches:
            # Extract just the filename (not the full path)
            filename = Path(match).name
            output_files.add(filename)
    
    return output_files


def main():
    print("=" * 80)
    print("ML Image and Generator Script Audit")
    print("=" * 80)
    print()
    
    # Step 1: Find all images
    print("Step 1: Finding all image files...")
    all_images = find_all_images()
    print(f"  Found {len(all_images)} image files")
    print()
    
    # Step 2: Find all generator scripts
    print("Step 2: Finding all generator scripts...")
    all_scripts = find_all_generator_scripts()
    print(f"  Found {len(all_scripts)} generator scripts")
    print()
    
    # Step 3: Find all markdown and notebook files to search
    print("Step 3: Finding all markdown and notebook files...")
    search_files = find_all_markdown_and_notebooks()
    print(f"  Found {len(search_files)} files to search")
    print()
    
    # Step 4: Check which images are referenced
    print("Step 4: Checking image references...")
    unreferenced_images = []
    referenced_images = []
    
    for i, image in enumerate(all_images, 1):
        if i % 20 == 0:
            print(f"  Checked {i}/{len(all_images)} images...")
        
        if is_image_referenced(image, search_files):
            referenced_images.append(image)
        else:
            unreferenced_images.append(image)
    
    print(f"  ✓ Referenced: {len(referenced_images)} images")
    print(f"  ✗ Unreferenced: {len(unreferenced_images)} images")
    print()
    
    # Step 5: Analyze generator scripts
    print("Step 5: Analyzing generator scripts...")
    script_outputs: Dict[Path, Set[str]] = {}
    
    for script in all_scripts:
        outputs = analyze_script_outputs(script)
        if outputs:
            script_outputs[script] = outputs
    
    print(f"  Analyzed {len(script_outputs)} scripts with identifiable outputs")
    print()
    
    # Step 6: Determine which scripts to delete
    print("Step 6: Determining which scripts to delete...")
    unreferenced_image_names = {img.name for img in unreferenced_images}
    scripts_to_delete = []
    
    for script, outputs in script_outputs.items():
        if outputs:  # Only consider if we found outputs
            all_outputs_unreferenced = all(
                output_name in unreferenced_image_names 
                for output_name in outputs
            )
            if all_outputs_unreferenced:
                scripts_to_delete.append(script)
    
    print(f"  Found {len(scripts_to_delete)} scripts to delete (all outputs unreferenced)")
    print()
    
    # Step 7: Report findings
    print("=" * 80)
    print("AUDIT RESULTS")
    print("=" * 80)
    print()
    
    if unreferenced_images:
        print(f"Unreferenced images ({len(unreferenced_images)}):")
        for img in unreferenced_images:
            rel_path = img.relative_to(REPO_ROOT)
            print(f"  - {rel_path}")
        print()
    
    if scripts_to_delete:
        print(f"Generator scripts to delete ({len(scripts_to_delete)}):")
        for script in scripts_to_delete:
            rel_path = script.relative_to(REPO_ROOT)
            outputs = script_outputs.get(script, set())
            print(f"  - {rel_path}")
            if outputs:
                print(f"    Generates: {', '.join(sorted(outputs))}")
        print()
    
    # Step 8: Confirm deletion
    if unreferenced_images or scripts_to_delete:
        print("=" * 80)
        response = input(f"Delete {len(unreferenced_images)} images and {len(scripts_to_delete)} scripts? (yes/no): ")
        print()
        
        if response.lower() == "yes":
            # Delete unreferenced images
            deleted_images = 0
            for img in unreferenced_images:
                try:
                    img.unlink()
                    deleted_images += 1
                except Exception as e:
                    print(f"  Error deleting {img}: {e}")
            
            # Delete scripts
            deleted_scripts = 0
            for script in scripts_to_delete:
                try:
                    script.unlink()
                    deleted_scripts += 1
                except Exception as e:
                    print(f"  Error deleting {script}: {e}")
            
            print(f"✓ Deleted {deleted_images} images")
            print(f"✓ Deleted {deleted_scripts} scripts")
            print()
        else:
            print("Deletion cancelled.")
            print()
    else:
        print("No unreferenced images or scripts found. Nothing to delete.")
        print()
    
    print("=" * 80)
    print("Audit complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
