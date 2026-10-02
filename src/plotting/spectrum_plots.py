"""
Spectrum (FFT peak) figures and the mode-amplitude figure.

The spectrum is produced in two independent variants -- linear y and
logarithmic y -- so the one that shows the small peaks best can be chosen for
the paper. The mode amplitudes live in their own figure rather than sharing a
row with the spectrum.

All appearance is inherited from `plotting.style`; nothing is configured here.
"""
import numpy as np
import matplotlib.pyplot as plt

from . import style


def plot_spectrum(frq, mag, peak_freqs=None, yscale="linear", x_max=None,
                  xlabel=r"frequency", ylabel=r"$|\mathrm{FFT}|$",
                  mark_peaks=True, ax=None, figsize=None, log_decades=4.0):
    """
    One spectrum panel.

    Parameters
    ----------
    frq, mag : arrays
        Frequency axis and FFT magnitude.
    peak_freqs : array or None
        Detected peak positions, marked with thin dashed reference lines.
    yscale : {"linear", "log"}
        Chooses the variant. Produce both and pick one for the paper.
    x_max : float or None
        Upper frequency limit; defaults to 1.2 * max(peak_freqs).
    """
    created = ax is None
    if created:
        fig, ax = plt.subplots(figsize=figsize or style.FIG_WIDE)
    else:
        fig = ax.figure

    mag = np.asarray(mag, dtype=float)
    ax.plot(frq, mag, color=style.DATA, lw=style.LINEWIDTH)

    if mark_peaks and peak_freqs is not None and len(peak_freqs):
        for f in np.asarray(peak_freqs):
            ax.axvline(f, color=style.REFERENCE, lw=0.8, linestyle="--",
                       alpha=0.75, zorder=0)

    if x_max is None and peak_freqs is not None and len(peak_freqs):
        x_max = float(np.max(peak_freqs)) * 1.2
    if x_max is not None:
        ax.set_xlim(0, x_max)

    if yscale == "log":
        positive = mag[mag > 0]
        if positive.size:
            ax.set_yscale("log")
            # Show a fixed number of decades below the tallest peak. Without a
            # floor the axis stretches down to the leakage sidelobes and the
            # peaks themselves get squeezed into the top sliver of the panel.
            top = mag.max() * 2.0
            floor = max(positive.min() * 0.5, top / 10 ** log_decades)
            ax.set_ylim(floor, top)
    else:
        ax.set_ylim(0, mag.max() * 1.08)

    style.style_axis(ax, xlabel=xlabel, ylabel=ylabel)
    if created:
        fig.tight_layout()
    return fig, ax


def plot_spectrum_both_scales(frq, mag, peak_freqs=None, x_max=None,
                              xlabel=r"frequency", ylabel=r"$|\mathrm{FFT}|$",
                              show=True, log_decades=4.0):
    """
    Produce the spectrum twice -- once linear-y, once log-y -- as two separate
    figures. Returns (fig_linear, fig_log).
    """
    fig_lin, _ = plot_spectrum(frq, mag, peak_freqs, yscale="linear",
                               x_max=x_max, xlabel=xlabel, ylabel=ylabel)
    fig_log, _ = plot_spectrum(frq, mag, peak_freqs, yscale="log",
                               x_max=x_max, xlabel=xlabel, ylabel=ylabel,
                               log_decades=log_decades)
    if show:
        plt.show()
    return fig_lin, fig_log


def plot_spectrum_with_predictions(frq, mag, detected_freqs, true_freqs,
                                   yscale="linear", x_max=None,
                                   xlabel=r"frequency",
                                   ylabel=r"$|\mathrm{FFT}|$",
                                   ax=None, figsize=None, show=False,
                                   log_decades=4.0):
    """
    Spectrum with detected peaks (black, solid) and predicted/true peaks (red,
    dashed) marked separately, so mismatches between the two are visible at a
    glance. `true_freqs` must already be in the same units as `frq` (i.e. the
    true energy differences divided by 2*pi, not the raw eigenvalue gaps).
    """
    created = ax is None
    if created:
        fig, ax = plt.subplots(figsize=figsize or style.FIG_WIDE)
    else:
        fig = ax.figure

    mag = np.asarray(mag, dtype=float)
    detected_freqs = np.asarray(detected_freqs, dtype=float)
    true_freqs = np.asarray(true_freqs, dtype=float)

    ax.plot(frq, mag, color=style.DATA, lw=style.LINEWIDTH, zorder=1)

    for i, f in enumerate(true_freqs):
        ax.axvline(f, color=style.REFERENCE, lw=1.2, linestyle="--",
                   alpha=0.85, zorder=2, label="predicted" if i == 0 else None)
    for i, f in enumerate(detected_freqs):
        ax.axvline(f, color=style.GUIDE, lw=0.8, linestyle=":",
                   alpha=0.9, zorder=3, label="detected" if i == 0 else None)

    if x_max is None:
        span = np.concatenate([detected_freqs, true_freqs]) if len(true_freqs) else detected_freqs
        x_max = float(np.max(span)) * 1.2 if len(span) else None
    if x_max is not None:
        ax.set_xlim(0, x_max)

    if yscale == "log":
        positive = mag[mag > 0]
        if positive.size:
            ax.set_yscale("log")
            top = mag.max() * 2.0
            floor = max(positive.min() * 0.5, top / 10 ** log_decades)
            ax.set_ylim(floor, top)
    else:
        ax.set_ylim(0, mag.max() * 1.08)

    style.style_axis(ax, xlabel=xlabel, ylabel=ylabel,
                     legend=True, legend_loc="upper right")
    if created:
        fig.tight_layout()
    if show:
        plt.show()
    return fig, ax


def plot_mode_amplitudes(amplitudes, yscale="linear", xlabel="mode index",
                         ylabel="amplitude", ax=None, figsize=None, show=False):
    """
    Mode amplitudes as a standalone figure (bar chart, uniform dark grey).
    """
    created = ax is None
    if created:
        fig, ax = plt.subplots(figsize=figsize or style.FIG_SINGLE)
    else:
        fig = ax.figure

    amplitudes = np.asarray(amplitudes, dtype=float)
    idx = np.arange(len(amplitudes))
    ax.bar(idx, amplitudes, color=style.DATA, width=0.65,
           edgecolor=style.DATA)

    if yscale == "log":
        positive = amplitudes[amplitudes > 0]
        if positive.size:
            ax.set_yscale("log")
            ax.set_ylim(positive.min() * 0.5, amplitudes.max() * 2.0)

    ax.set_xticks(idx)
    style.style_axis(ax, xlabel=xlabel, ylabel=ylabel)
    if created:
        fig.tight_layout()
    if show:
        plt.show()
    return fig, ax


def plot_global_fit(t, y, y_fit, rms=None, dc=None, ax=None, figsize=None,
                    show=False):
    """Time-domain signal with the fitted model overlaid."""
    created = ax is None
    if created:
        fig, ax = plt.subplots(figsize=figsize or style.FIG_WIDE)
    else:
        fig = ax.figure

    ax.plot(t, y, "-", color=style.SUPPORT, lw=1.0, label="data")
    ax.plot(t, y_fit, "--", color=style.DATA, lw=style.LINEWIDTH, label="fit")
    style.style_axis(ax, xlabel="wait time", ylabel=r"$P(\psi_0)$",
                     legend=True, legend_loc="upper right")
    if created:
        fig.tight_layout()
    if show:
        plt.show()
    return fig, ax
