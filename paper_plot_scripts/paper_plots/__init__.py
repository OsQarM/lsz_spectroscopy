"""
Shared helpers for the paper-figure scripts.

These scripts read the run directories produced by `refactored_code/` and
re-draw selected figures with everything the paper needs to control exposed
through a YAML file: axis labels, legend, peak markers, colours, limits,
figure size.

Importing this package puts the repository's `src/` on `sys.path`, so the
figure style (`plotting.style`) and the analysis helpers are the same ones the
rest of the pipeline uses.
"""
from __future__ import annotations

import sys
from pathlib import Path

# paper_plot_scripts/paper_plots/__init__.py -> repo root is two levels up.
REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

__all__ = ["REPO_ROOT", "SRC_DIR"]
