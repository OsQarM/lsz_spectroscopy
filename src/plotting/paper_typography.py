#!/usr/bin/env python3
"""
APS-style typography for the paper figures.

Call `apply()` to switch matplotlib to true LaTeX Computer Modern (text.usetex).
If a working LaTeX + dvipng toolchain is NOT present, it transparently falls
back to matplotlib's built-in Computer-Modern mathtext (no full TeX needed) and
reports which path was taken. Font-only: nothing about data/colour/layout here.
"""
import matplotlib as mpl

_PREAMBLE = (r"\usepackage[T1]{fontenc}"
             r"\usepackage{lmodern}"
             r"\usepackage{amsmath}")


def _usetex_works() -> bool:
    """Probe: try to compile a tiny label through the real LaTeX pipeline."""
    import matplotlib.pyplot as plt
    from io import BytesIO
    try:
        mpl.rcParams["text.usetex"] = True
        mpl.rcParams["text.latex.preamble"] = _PREAMBLE
        fig = plt.figure(figsize=(1, 1))
        fig.text(0.5, 0.5, r"$\tilde r\,\langle h_z\rangle\,\tau_\mathrm{Th}$")
        fig.savefig(BytesIO(), format="png", dpi=50)
        plt.close(fig)
        return True
    except Exception as exc:                       # LaTeX/dvipng missing or broken
        import matplotlib.pyplot as plt
        plt.close("all")
        print(f"[paper_style] usetex probe failed ({exc.__class__.__name__}): {exc}")
        return False


def apply() -> str:
    """Set Computer Modern serif typography. Returns the path used."""
    if _usetex_works():
        mpl.rcParams.update({
            "text.usetex": True,
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman"],
            "text.latex.preamble": _PREAMBLE,
            "axes.formatter.use_mathtext": True,
        })
        print("[paper_style] typography = LaTeX Computer Modern (text.usetex=True)")
        return "usetex"
    # Fallback: matplotlib's own Computer-Modern mathtext (no TeX toolchain).
    mpl.rcParams.update({
        "text.usetex": False,
        "mathtext.fontset": "cm",
        "font.family": "serif",
        "font.serif": ["cmr10", "DejaVu Serif"],
        "axes.formatter.use_mathtext": True,
    })
    print("[paper_style] typography = mathtext 'cm' fallback (usetex unavailable)")
    return "mathtext-cm"
