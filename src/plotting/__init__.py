"""Paper-quality plotting helpers for the LZS spectroscopy figures."""

from . import style
from . import paper_typography
from . import palettes

__all__ = ["style", "paper_typography", "palettes", "apply"]


def apply(palette="sober"):
    """Shorthand for `plotting.style.apply()`. See `plotting.palettes`."""
    return style.apply(palette)
