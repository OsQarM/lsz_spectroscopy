"""
Selectable colour palettes for the paper figures.

`style.apply()` takes a palette name; everything else in the package reads the
active palette through `style`, so switching is a one-line change in the
notebook and no plotting function needs editing.

Available:

  "sober"   -- black data, red dashed references, greys. Maximum restraint,
               survives greyscale printing perfectly. The default.

  "modern"  -- a muted, colour-blind-safe accent set in the register current
               APS/Nature papers use: desaturated blue/vermillion/teal on a
               near-black base. Still restrained -- colour marks the data
               series, never decorates -- but multi-series panels are readable
               at a glance instead of relying on dash patterns alone.

The "modern" accents are the Okabe-Ito colour-blind-safe set, slightly
darkened. They stay distinguishable under deuteranopia and protanopia, and
their luminances differ enough to remain separable in greyscale.
"""

PALETTES = {
    "sober": {
        "DATA": "black",
        "REFERENCE": "red",
        "SECONDARY": "0.45",
        "SUPPORT": "0.75",
        "GUIDE": "0.6",
        # Multi-series cycle: shades of grey, separated by dash pattern.
        "CYCLE": ("black", "0.35", "0.55", "0.7"),
    },
    "modern": {
        "DATA": "#1a1a1a",        # near-black, softer than pure black in print
        "REFERENCE": "#c1442a",   # vermillion, for identity lines / predictions
        "SECONDARY": "#2f6f9f",   # desaturated blue
        "SUPPORT": "#bfc4c8",     # cool light grey for background traces
        "GUIDE": "#9aa0a6",
        # Okabe-Ito (darkened): blue, vermillion, teal, orange, purple.
        "CYCLE": ("#1a1a1a", "#2f6f9f", "#c1442a", "#3a8a7d", "#d98b28"),
    },
}

DEFAULT = "sober"
