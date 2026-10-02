"""
Shared infrastructure for the three LZS pipeline scripts.

The physics lives untouched in the repository's `src/` package; this package
only adds configuration loading, run-directory management, reporting and
figure capture around it. Importing it puts `src/` on `sys.path`, so the
scripts can `from lsz_experiment import ...` exactly as the notebooks do.
"""
from __future__ import annotations

import sys
from pathlib import Path

# refactored_code/lzs_pipeline/__init__.py -> repo root is two levels up.
REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

__all__ = ["REPO_ROOT", "SRC_DIR"]
