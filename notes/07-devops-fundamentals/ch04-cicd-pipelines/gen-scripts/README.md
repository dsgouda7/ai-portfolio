# Ch.4 Diagram Generation Scripts

This directory contains Python scripts to generate diagrams for Ch.4: CI/CD Pipelines.

## Generated Diagrams

| Script | Output Files | Description |
|--------|-------------|-------------|
| `gen_ch04_cicd_pipeline.py` | `ch04_cicd_pipeline.png`<br>`ch04_cicd_pipeline.gif` | Full CI/CD workflow: commit → test → build → deploy |
| `gen_ch04_workflow_triggers.py` | `ch04_workflow_triggers.png`<br>`ch04_workflow_triggers.gif` | Different trigger types: push, PR, schedule, manual |
| `gen_ch04_secrets_management.py` | `ch04_secrets_management.png`<br>`ch04_secrets_management.gif` | Secure secrets flow: GitHub Secrets → Actions → Docker Hub |

## Usage

### Generate All Diagrams

```bash
python generate_all.py
```

### Generate Individual Diagrams

```bash
python gen_ch04_cicd_pipeline.py
python gen_ch04_workflow_triggers.py
python gen_ch04_secrets_management.py
```

## Requirements

```bash
pip install matplotlib pillow
```

## Output Location

All diagrams are saved to `../img/` directory.

## Diagram Conventions

- **Static PNG**: High-resolution diagram suitable for README
- **Animated GIF**: Shows progression/interaction (e.g., highlighting stages)
- **Colors**: Consistent across all DevOps track chapters
  - Red: Triggers/critical actions
  - Teal: Testing/validation
  - Blue: Build/containerization
  - Green: Deployment/success

## Modifying Diagrams

Each script uses matplotlib with a consistent style:
- Non-interactive backend (`matplotlib.use('Agg')`)
- 150 DPI for readability
- Pillow for GIF generation
- Configurable colors at top of each file

To adjust colors, modify the `COLOR_*` constants at the top of each script.
