from __future__ import annotations

import sys
from pathlib import Path


PHASE3_ROOT = Path(__file__).parent.parent
PROJECT_ROOT = PHASE3_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PHASE3_ROOT))
