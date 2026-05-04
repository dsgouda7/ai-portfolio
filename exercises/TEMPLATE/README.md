# Exercise Template

This folder contains reusable project scaffolding files for all ML exercises.

## Files

- **`setup.sh`** — Unix/macOS/WSL setup script  
  Creates venv, installs dependencies, validates installation
  
- **`setup.ps1`** — Windows PowerShell setup script  
  Same functionality as setup.sh for Windows users

- **`.gitignore`** — Standard Python .gitignore  
  Excludes venv, models, data, IDE files, etc.

## Usage

### For Exercise Creators

Copy these files to a new exercise directory:

```bash
# From exercises/ root
cp TEMPLATE/setup.sh 01-ml/NEW_EXERCISE/
cp TEMPLATE/setup.ps1 01-ml/NEW_EXERCISE/
cp TEMPLATE/.gitignore 01-ml/NEW_EXERCISE/
```

The scripts are generic — they work with any `requirements.txt` file.

### For Exercise Completers

You don't need to interact with this folder directly. Each exercise already has its own setup scripts copied from here.

---

**Last Updated:** April 28, 2026
