"""
The FFT of a swept signal, computed exactly as the analysis pipeline does.

`src/fourier.py:fourier_analysis` removes the mean, applies a window,
zero-pads, takes the real FFT and drops the DC bin. This module repeats that
recipe so a paper figure shows the same spectrum the peak detection ran on --
it does not re-implement the physics, only the transform, and the peak
positions are reused from the stored diagnostics whenever they exist.
"""
from __future__ import annotations

import numpy as np


def compute_spectrum(signal, tw, window="hann", zero_pad_factor=32,
                     normalise=None):
    """Return (freqs, magnitude) for one swept signal.

    Mirrors `fourier_analysis`: subtract the mean, apply `window`, zero-pad by
    `zero_pad_factor`, rfft, then drop the DC bin (index 0) so the near-zero
    component does not dominate the vertical scale.

    `normalise` rescales the magnitude so several runs of different lengths
    can share one panel:
        None   -> raw |FFT|
        "max"  -> peak magnitude becomes 1
        "area" -> unit area under the curve
        "coherent" -> divide by the window's coherent gain times N, which
                      puts the magnitude in units of the signal's amplitude
    """
    from scipy.signal import get_window

    y = np.asarray(signal, dtype=float)
    t = np.asarray(tw, dtype=float)
    n = y.size
    dt = float(t[1] - t[0])
    if not dt > 0:
        raise ValueError("the wait-time grid must be increasing")

    win = get_window(window, n)
    y_win = (y - y.mean()) * win

    n_pad = max(int(zero_pad_factor), 1) * n
    spec = np.fft.rfft(y_win, n=n_pad)
    freqs = np.fft.rfftfreq(n_pad, dt)

    # Drop DC, exactly as fourier_analysis does before peak-finding.
    freqs = freqs[1:]
    mag = np.abs(spec)[1:]

    if normalise in (None, "none", False):
        pass
    elif normalise == "max":
        peak = mag.max()
        if peak > 0:
            mag = mag / peak
    elif normalise == "area":
        area = np.trapezoid(mag, freqs)
        if area > 0:
            mag = mag / area
    elif normalise == "coherent":
        cg = win.sum() / n
        if cg > 0:
            mag = mag * (2.0 / (n * cg))
    else:
        raise ValueError(
            f"unknown normalise option: {normalise!r} "
            f"(expected none, max, area or coherent)")

    return freqs, mag


def detect_peaks(freqs, mag, n_peaks=None, detect_prominence=0.001,
                 f_min=None, zero_pad_factor=32, n_raw=None, dt=None):
    """Find peak positions in a computed spectrum.

    This is the fallback for sources with no stored diagnostics. It follows
    the same gates as `fourier_analysis` -- a prominence floor relative to the
    tallest peak, an optional low-frequency cutoff, and a minimum separation
    of ~3 raw FFT bins -- so the marked peaks match what the analysis would
    have detected.
    """
    from scipy.signal import find_peaks

    mag = np.asarray(mag, dtype=float)
    if mag.size == 0:
        return np.array([])

    min_dist = max(1, int(zero_pad_factor) // 2)
    locs, _ = find_peaks(mag, distance=min_dist,
                         prominence=float(detect_prominence) * mag.max())
    if locs.size == 0:
        return np.array([])

    if f_min is not None:
        locs = locs[freqs[locs] >= float(f_min)]
        if locs.size == 0:
            return np.array([])

    # Minimum physical separation: 3 raw (un-padded) bins, as in fourier.py.
    if n_raw and dt:
        min_sep = 3.0 / (n_raw * dt)
    else:
        min_sep = 0.0

    order = np.argsort(mag[locs])[::-1]
    accepted: list[float] = []
    for k in locs[order]:
        if n_peaks is not None and len(accepted) >= int(n_peaks):
            break
        f = freqs[k]
        if accepted and min_sep > 0 and np.min(np.abs(np.array(accepted) - f)) < min_sep:
            continue
        accepted.append(float(f))

    return np.sort(np.array(accepted))


def peaks_for_source(source, cfg_peaks: dict, freqs, mag):
    """The peak positions to mark for one dataset.

    Prefers what the pipeline actually detected (`diagnostics_results.npz`
    from script 2). `source: true_differences` instead marks the exact
    transition frequencies; `source: detect` always re-runs the detection.
    """
    which = str(cfg_peaks.get("source", "auto")).lower()
    diag = source.get("diagnostics", {})

    if which in ("auto", "detected") and "freqs" in diag:
        return np.sort(np.asarray(diag["freqs"], dtype=float)), "diagnostics"

    if which == "true_differences":
        if "true_differences" not in diag:
            raise ValueError(
                f"{source['label']}: peaks.source is 'true_differences' but "
                f"this run has no diagnostics_results.npz -- run script 2 on "
                f"it first, or use peaks.source: detect")
        # Stored as angular energy differences; the spectrum axis is ordinary
        # frequency, so convert.
        return np.sort(np.asarray(diag["true_differences"], dtype=float)
                       / (2 * np.pi)), "true differences"

    if which == "detected" and "freqs" not in diag:
        raise ValueError(
            f"{source['label']}: peaks.source is 'detected' but this run has "
            f"no diagnostics_results.npz -- run script 2, or use "
            f"peaks.source: detect")

    tw = source["tw"]
    found = detect_peaks(
        freqs, mag,
        n_peaks=cfg_peaks.get("n_peaks"),
        detect_prominence=cfg_peaks.get("detect_prominence", 0.001),
        f_min=cfg_peaks.get("f_min"),
        zero_pad_factor=cfg_peaks.get("zero_pad_factor", 32),
        n_raw=source["signal"].size,
        dt=float(tw[1] - tw[0]),
    )
    return found, "re-detected"
