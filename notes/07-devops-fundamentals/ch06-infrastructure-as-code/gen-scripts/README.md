# Chapter 6 Diagram Generation Scripts

This directory contains Python scripts that generate visual diagrams for the Infrastructure as Code chapter.

## Scripts

### 1. gen_ch06_terraform_workflow.py
**Purpose:** Generate the Terraform workflow diagram  
**Output:** `../img/ch06-terraform-workflow.png`  
**Content:** Shows the complete workflow: Write → Init → Plan → Apply → State

**Key elements:**
- Step-by-step workflow visualization
- State file management
- Infrastructure provisioning
- Destroy operation
- Key command reference box

**Run:**
```bash
python gen_ch06_terraform_workflow.py
```

---

### 2. gen_ch06_resource_graph.py
**Purpose:** Generate the resource dependency graph diagram  
**Output:** `../img/ch06-resource-graph.png`  
**Content:** Shows how Terraform builds a directed acyclic graph (DAG) of resource dependencies

**Key elements:**
- Network foundation layer
- Storage and database dependencies
- Multiple web containers (parallel execution)
- Load balancer at the top
- Explicit vs implicit dependencies
- Execution levels and parallelism

**Run:**
```bash
python gen_ch06_resource_graph.py
```

---

### 3. gen_ch06_state_management.py
**Purpose:** Generate the state management comparison diagram  
**Output:** `../img/ch06-state-management.png`  
**Content:** Compares local vs remote state backends

**Key elements:**
- Local state (single developer)
- Remote state (team collaboration)
- State locking mechanism
- Pros/cons comparison
- State operation commands
- Security warnings

**Run:**
```bash
python gen_ch06_state_management.py
```

---

## Generate All Diagrams

To generate all diagrams at once:

**Windows (PowerShell):**
```powershell
Get-ChildItem -Filter "gen_ch06_*.py" | ForEach-Object { python $_.FullName }
```

**Linux/macOS (Bash):**
```bash
for script in gen_ch06_*.py; do python "$script"; done
```

---

## Dependencies

All scripts use matplotlib with the dark background theme:

```python
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
```

Ensure matplotlib is installed:
```bash
pip install matplotlib
```

---

## Color Palette

All diagrams use the consistent DevOps color palette:

| Color | Hex | Usage |
|-------|-----|-------|
| Primary Blue | `#1e3a8a` | Main workflow elements |
| Success Green | `#15803d` | Positive states, apply actions |
| Caution Orange | `#b45309` | Planning, warnings |
| Info Blue | `#1d4ed8` | Information, init steps |
| Purple | `#9333ea` | State file, special elements |
| Red | `#dc2626` | Destroy, errors, dangers |
| Background | `#1a1a2e` | Canvas background |
| Box BG | `#2d3748` | Element backgrounds |
| Text | `#ffffff` | Primary text |
| Secondary Text | `#a0a0a0` | Descriptions, notes |

---

## Output

All diagrams are saved to `../img/` with:
- **Format:** PNG
- **DPI:** 150 (high quality)
- **Background:** Dark theme (`#1a1a2e`)
- **Size:** 14x10 inches (landscape)

---

## Troubleshooting

### Error: "No module named 'matplotlib'"
**Solution:** Install matplotlib
```bash
pip install matplotlib
```

### Error: "Tkinter not found"
**Cause:** GUI backend not available  
**Solution:** Scripts use `Agg` backend (headless) — should work without GUI

### Diagrams look pixelated
**Solution:** Increase DPI in script (currently 150)
```python
plt.savefig(output_path, dpi=200, ...)  # Higher quality
```

### Font warnings
**Cause:** Custom fonts not available  
**Solution:** Warnings are safe to ignore — falls back to default fonts

---

## Customization

To modify diagram appearance:
1. Edit color palette at top of script
2. Adjust box sizes and positions
3. Change font sizes in `ax.text()` calls
4. Modify arrow styles in `FancyArrowPatch()`

Example:
```python
# Make title larger
ax.text(5, 9.5, 'Title', fontsize=28, ...)  # Changed from 24
```

---

## Animation

The chapter animation is stored separately at:
`../img/ch06-terraform-workflow.gif`

This is a multi-frame GIF showing the workflow in action. To regenerate:
1. Generate individual frames with modified scripts
2. Combine using `imageio` or similar tool

---

## Related Files

- `../README.md` — Chapter content (references these diagrams)
- `../notebook.ipynb` — Interactive tutorial (uses workflow concepts)
- `../notebook_supplement.ipynb` — Azure examples (uses state management)
