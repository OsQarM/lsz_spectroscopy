"""
Single source of truth for the paper figures' look.

Every plotting helper in `src/plotting/` reads its fonts, sizes, colours and
line weights from here, so the whole figure set can be retuned by editing this
file alone. Typography (Computer Modern, usetex when available) is delegated to
`paper_typography.apply()`.

Usage from a notebook:

    from plotting import style
    style.apply()            # once, at the top of the notebook
"""
import matplotlib as mpl
import matplotlib.pyplot as plt

from . import paper_typography
from .palettes import PALETTES, DEFAULT

# --------------------------------------------------------------------------
# Palette -- deliberately sober: black data, red reference lines, greys for
# secondary/supporting content. Nothing else, so the figures stay printable in
# greyscale and read consistently across the paper.
# --------------------------------------------------------------------------
# These are rebound by `apply(palette=...)`. Plotting helpers read them as
# `style.DATA` etc. at call time, so switching palette needs no other edits.
DATA = PALETTES[DEFAULT]["DATA"]            # primary data: markers, main curve
REFERENCE = PALETTES[DEFAULT]["REFERENCE"]  # identity diagonals, predicted rates
SECONDARY = PALETTES[DEFAULT]["SECONDARY"]  # second series when one is needed
SUPPORT = PALETTES[DEFAULT]["SUPPORT"]      # background/context curves
GUIDE = PALETTES[DEFAULT]["GUIDE"]          # segment boundaries, annotations
CYCLE = PALETTES[DEFAULT]["CYCLE"]          # multi-series cycle
PALETTE = DEFAULT                           # name of the active palette

# Marker cycle for the rare panel that must distinguish series without colour.
MARKERS = ("o", "x", "s", "^", "D", "v")

# --------------------------------------------------------------------------
# Font sizes -- tuned to stay legible when a two-column figure is reduced to
# ~3.4 in wide in the final PDF.
# --------------------------------------------------------------------------
FONT_LABEL = 15          # axis labels
FONT_TICK = 13           # tick labels
FONT_LEGEND = 13         # legend entries
FONT_TITLE = 15          # panel titles (used sparingly; prefer captions)

# --------------------------------------------------------------------------
# Figure geometry (inches). Widths follow the usual single/double column feel.
# --------------------------------------------------------------------------
FIG_SINGLE = (6.0, 4.5)      # one square-ish panel
FIG_WIDE = (8.0, 3.5)        # one wide panel (time traces)
FIG_DOUBLE = (11.0, 4.5)     # two panels side by side
FIG_TRIPLE = (16.0, 4.0)     # three panels side by side

LINEWIDTH = 1.3
MARKERSIZE = 6.0
REF_LINEWIDTH = 1.2
DPI_SAVE = 500


def set_palette(name):
    """Rebind the module-level colour names to the named palette."""
    global DATA, REFERENCE, SECONDARY, SUPPORT, GUIDE, CYCLE, PALETTE
    try:
        pal = PALETTES[name]
    except KeyError:
        raise ValueError(
            f"unknown palette {name!r}; available: {sorted(PALETTES)}") from None
    DATA = pal["DATA"]
    REFERENCE = pal["REFERENCE"]
    SECONDARY = pal["SECONDARY"]
    SUPPORT = pal["SUPPORT"]
    GUIDE = pal["GUIDE"]
    CYCLE = pal["CYCLE"]
    PALETTE = name
    mpl.rcParams["axes.prop_cycle"] = mpl.cycler(color=list(CYCLE))
    return name


def apply(palette=DEFAULT) -> str:
    """Install the paper style globally. Returns the typography path used.

    `palette` selects the colour scheme -- "sober" (default) or "modern";
    see `plotting.palettes`.
    """
    path = paper_typography.apply()
    set_palette(palette)

    mpl.rcParams.update({
        # Sizes
        "axes.labelsize": FONT_LABEL,
        "axes.titlesize": FONT_TITLE,
        "xtick.labelsize": FONT_TICK,
        "ytick.labelsize": FONT_TICK,
        "legend.fontsize": FONT_LEGEND,
        "font.size": FONT_TICK,

        # Sober frame: no grid, no coloured spines, ticks pointing in.
        "axes.grid": False,
        "axes.spines.top": True,
        "axes.spines.right": True,
        "axes.edgecolor": "black",
        "axes.linewidth": 0.8,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "xtick.major.size": 4.0,
        "ytick.major.size": 4.0,
        "xtick.minor.size": 2.0,
        "ytick.minor.size": 2.0,

        # Lines and markers
        "lines.linewidth": LINEWIDTH,
        "lines.markersize": MARKERSIZE,

        # Legend: frameless by default, no shadow, tight.
        "legend.frameon": False,
        "legend.handlelength": 1.6,
        "legend.borderpad": 0.3,
        "legend.labelspacing": 0.3,

        # Output
        "figure.dpi": 110,
        "savefig.dpi": DPI_SAVE,
        "savefig.bbox": "tight",
        "figure.autolayout": False,
    })
    return path


def style_axis(ax, xlabel=None, ylabel=None, title=None, legend=False,
               legend_loc="best", legend_frameon=False):
    """Apply the common per-axis treatment (labels, tick sizes, legend)."""
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=FONT_LABEL)
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=FONT_LABEL)
    if title is not None:
        ax.set_title(title, fontsize=FONT_TITLE)
    ax.tick_params(axis="both", which="major", labelsize=FONT_TICK)
    if legend:
        ax.legend(frameon=legend_frameon, fontsize=FONT_LEGEND, loc=legend_loc)
    return ax


def identity_line(ax, values, label="estimate = true", **kwargs):
    """Draw the red dashed y = x reference over the span of `values`."""
    import numpy as np
    v = np.asarray(values, dtype=float)
    diag = np.linspace(np.nanmin(v), np.nanmax(v), 100)
    opts = dict(linestyle="--", color=REFERENCE, lw=REF_LINEWIDTH, label=label)
    opts.update(kwargs)
    line, = ax.plot(diag, diag, **opts)
    return line


def save(fig, path, **kwargs):
    """Save a figure with the paper defaults (high dpi, tight bbox)."""
    opts = dict(dpi=DPI_SAVE, bbox_inches="tight")
    opts.update(kwargs)
    fig.savefig(path, **opts)
    return path
