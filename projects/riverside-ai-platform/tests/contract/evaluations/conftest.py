from __future__ import annotations

import sys
from pathlib import Path

RIVERSIDE_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = RIVERSIDE_ROOT / "src"
EVALUATIONS_ROOT = RIVERSIDE_ROOT / "evaluations"
sys.path.insert(0, str(SRC_ROOT))
