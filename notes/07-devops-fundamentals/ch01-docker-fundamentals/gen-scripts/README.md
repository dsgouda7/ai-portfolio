# Diagram Generation Scripts

This directory contains Python scripts to generate static diagrams for Ch.1 Docker Fundamentals.

## Scripts

1. **gen_ch01_docker_architecture.py** — Image layers diagram
   - Shows how Dockerfile instructions create cached layers
   - Demonstrates layer caching optimization
   - Output: `../img/ch01-docker-architecture.png`

2. **gen_ch01_container_lifecycle.py** — Container lifecycle flow
   - Build → Run → Stop → Remove state transitions
   - Shows relationship between images and containers
   - Output: `../img/ch01-container-lifecycle.png`

3. **gen_ch01_volume_mounts.py** — Volume persistence diagram
   - Host filesystem vs container filesystem
   - Named volumes vs bind mounts comparison
   - Output: `../img/ch01-volume-mounts.png`

## Requirements

```bash
pip install matplotlib numpy
```

## Usage

Run all scripts from this directory:

```bash
python gen_ch01_docker_architecture.py
python gen_ch01_container_lifecycle.py
python gen_ch01_volume_mounts.py
```

Or run all at once (PowerShell):

```powershell
Get-ChildItem -Filter "gen_ch01_*.py" | ForEach-Object { python $_.Name }
```

## Output

All diagrams are saved to `../img/` with dark theme styling (#1a1a2e background) to match other tracks in this repository.

## Animation Note

The chapter README references `ch01-container-lifecycle.gif` as an animation. To create a GIF from the static PNG:

1. Duplicate the PNG 10 times with slight variations (or use same frame)
2. Use ImageMagick or similar tool to create GIF:
   ```bash
   convert -delay 100 -loop 0 frame*.png ch01-container-lifecycle.gif
   ```

For this chapter, the static PNG serves as the animation reference.
