"""
Spectral peak detection for the single-fluxonium LZS experiment.

This is deliberately separate from the multi-qubit ``src/fourier.py``. The
fluxonium interference signal ``P(tw)`` is *near single-tone* (the plateau qubit
gap) with a small number of extra tones from leakage into |2>, |3>, ... . The
peak finder here differs from the multi-qubit one in two ways that matter for
this experiment:

  1. It does NOT take a fixed ``n_peaks``. The number of transitions is unknown a
     priori, so every candidate that survives the gates is returned.

  2. It suppresses the *sidelobes of the tallest peak(s)* without penalising
     genuine weak modes elsewhere. Instead of one global exclusion zone (which
     either kills real nearby peaks or lets skirts through), each candidate near
     a strong neighbour must clear a floor that scales with that neighbour's
     height and falls off with distance according to the window's known sidelobe
     envelope. Far from any tall peak the floor collapses back to the ordinary
     ``detect_prominence``, so resolution and faint-peak sensitivity elsewhere
     are untouched.

The main entry point is :func:`find_all_peaks`. :func:`analyze_peaks` is a thin
wrapper that returns ``(freqs, phases)`` in the same shape as the project's
``fourier_analysis`` so it drops straight into ``fit_decay_rates``.
"""

import numpy as np
import matplotlib.pyplot as plt

# np.trapz was renamed np.trapezoid in NumPy 2.0 (trapz removed there).
_trapz = getattr(np, 'trapezoid', None) or np.trapz


# Approximate first-sidelobe level (linear, not dB) and mainlobe half-width in
# raw FFT bins for the common windows. The skirt of a peak of height H is
# modelled as decaying from ~sidelobe_level * H at the mainlobe edge with a
# ~1/distance falloff (a conservative envelope for these windows).
_WINDOW_SIDELOBE = {
    # window        (first-sidelobe fraction, mainlobe half-width in raw bins)
    'boxcar':        (0.217, 1.0),
    'rect':          (0.217, 1.0),
    'hann':          (0.027, 2.0),
    'hanning':       (0.027, 2.0),
    'hamming':       (0.0063, 2.0),
    'blackman':      (0.0012, 3.0),
    'blackmanharris':(2.0e-5, 4.0),
    'flattop':       (1.0e-4, 5.0),
}


def _sidelobe_params(window):
    key = window.lower() if isinstance(window, str) else 'hann'
    return _WINDOW_SIDELOBE.get(key, _WINDOW_SIDELOBE['hann'])


def find_all_peaks(pc_list, tw_l, *, prominence_threshold=0.01,
                   detect_prominence=0.01, zero_pad_factor=32, window='hann',
                   f_min=None, f_max=None,
                   sidelobe_suppression=True, sidelobe_safety=3.0,
                   drop_dc_peak=False, return_spectrum=False):
    """
    Detect *all* spectral peaks of a real signal, with peak-relative sidelobe
    rejection. No fixed peak count.

    Parameters
    ----------
    pc_list, tw_l : array
        Signal and its (uniform) time base.
    detect_prominence : float
        Baseline prominence floor for ``scipy.signal.find_peaks``, as a fraction
        of the spectrum max. This is the FIRST gate and sets the true faint-peak
        sensitivity everywhere that is *not* in a tall peak's shadow.
    prominence_threshold : float
        Final magnitude floor, as a fraction of the strongest accepted peak.
        Applied after detection; cannot recover peaks the ``detect_prominence``
        gate already rejected.
    sidelobe_suppression : bool
        If True, apply the peak-relative dynamic floor described in the module
        docstring: a candidate within a strong peak's skirt must exceed
        ``sidelobe_safety`` times the modelled sidelobe envelope of that peak.
        This is what "ignore only the vicinity of the highest peak" means, and
        it is local — weak peaks far from any tall one are unaffected.
    sidelobe_safety : float
        Multiplier on the modelled sidelobe envelope. Larger = more aggressive
        suppression. ~2-4 is usually right; raise it if skirts still leak.
    f_min, f_max : float or None
        Restrict ANALYSIS to [f_min, f_max]. The returned spectrum is left full
        for plotting; only detection is windowed.
    drop_dc_peak : {False, True, 'auto'}
        Drop the lowest-frequency accepted peak as a DC/offset residual. Defaults
        to False here: the DC bin is already excluded from the spectrum and, for
        this experiment, the lowest tone is usually the genuine qubit gap -- so
        dropping it would throw away the dominant mode. Set True (or 'auto', which
        drops only when ``f_min`` is None) only if you know a spurious low-freq
        residual is present.

    Returns
    -------
    dict with keys:
        'freqs', 'phases', 'amps'      : refined peak properties (sorted by amp)
        'mag', 'frq'                   : full one-sided spectrum for plotting
        'gap_dom'                      : dominant (largest-amp) frequency
        'accepted_idx'                 : spectrum indices of accepted peaks
    (a dict rather than a tuple so callers can pull only what they need.)
    """
    from scipy.signal import find_peaks, get_window

    y = np.asarray(pc_list, dtype=float)
    t = np.asarray(tw_l, dtype=float)
    N = len(y)
    dt = t[1] - t[0]
    dc = y.mean()

    win = get_window(window, N)
    cg = win.sum() / N
    y_win = (y - dc) * win

    N_pad = int(zero_pad_factor) * N
    spectrum = np.fft.rfft(y_win, n=N_pad)
    freqs = np.fft.rfftfreq(N_pad, dt)
    magnitude = np.abs(spectrum)

    # Drop the DC bin; keep it aside only for reference.
    mag = magnitude[1:]
    frq = freqs[1:]
    spc = spectrum[1:]

    raw_df = 1.0 / (N * dt)          # true resolution (independent of padding)
    df = frq[1] - frq[0]             # padded bin width
    bins_per_raw = raw_df / df       # ~= zero_pad_factor

    # 1. Baseline detection: peaks above the flat prominence floor.
    min_dist = max(1, int(zero_pad_factor) // 2)
    peak_locs, _ = find_peaks(mag, distance=min_dist,
                              prominence=detect_prominence * mag.max())
    if len(peak_locs) == 0:
        peak_locs = np.array([int(np.argmax(mag))])

    # 2. Analysis-band restriction (spectrum itself left intact for plotting).
    if f_min is not None:
        peak_locs = peak_locs[frq[peak_locs] >= f_min]
    if f_max is not None:
        peak_locs = peak_locs[frq[peak_locs] <= f_max]
    if len(peak_locs) == 0:
        # nothing in band; fall back to the strongest in-band bin
        band = np.ones_like(frq, dtype=bool)
        if f_min is not None:
            band &= frq >= f_min
        if f_max is not None:
            band &= frq <= f_max
        peak_locs = np.array([int(np.argmax(np.where(band, mag, -np.inf)))])

    # 3. Peak-relative sidelobe rejection. Walk candidates strongest-first;
    #    accept a peak only if it clears the sidelobe envelope cast by every
    #    already-accepted (stronger) peak. The envelope is anchored to the
    #    window's first-sidelobe level and decays like 1/distance beyond the
    #    mainlobe, so the shadow is *local* to each tall peak.
    sl_level, mainlobe_bins = _sidelobe_params(window)
    mainlobe_df = mainlobe_bins * raw_df

    order = np.argsort(mag[peak_locs])[::-1]
    accepted = []
    acc_f = []
    acc_h = []
    for k in peak_locs[order]:
        f_k = frq[k]
        h_k = mag[k]
        if sidelobe_suppression and acc_f:
            d = np.abs(np.array(acc_f) - f_k)
            # modelled skirt height of each stronger peak at this candidate
            in_mainlobe = d < mainlobe_df
            # inside the mainlobe: envelope ~ neighbour height (fully shadowed)
            # outside: first-sidelobe level, decaying as mainlobe_df / d
            env = np.where(
                in_mainlobe,
                np.array(acc_h),
                np.array(acc_h) * sl_level * (mainlobe_df / np.maximum(d, 1e-30)),
            )
            floor = sidelobe_safety * env.max()
            if h_k < floor:
                continue
        accepted.append(int(k))
        acc_f.append(f_k)
        acc_h.append(h_k)
    peak_locs = np.array(accepted, dtype=int)

    # 4. Final magnitude floor relative to the strongest accepted peak.
    if len(peak_locs):
        thr = prominence_threshold * mag[peak_locs].max()
        peak_locs = peak_locs[mag[peak_locs] >= thr]

    # 5. Parabolic (log-magnitude) refinement of freq / phase / amplitude.
    log_mag = np.log(mag + 1e-30)
    n = len(peak_locs)
    freqs_ref = np.empty(n)
    phases = np.empty(n)
    amps = np.empty(n)
    for i, k in enumerate(peak_locs):
        k_l = max(k - 1, 0)
        k_r = min(k + 1, len(mag) - 1)
        a_, b_, c_ = log_mag[k_l], log_mag[k], log_mag[k_r]
        if k_l != k_r:
            delta = 0.5 * (a_ - c_) / (a_ - 2*b_ + c_ + 1e-30)
            delta = float(np.clip(delta, -1.0, 1.0))
        else:
            delta = 0.0
        freqs_ref[i] = frq[k] + delta * df
        amp_interp = mag[k] if delta == 0 else (
            mag[k_l] * (1 - abs(delta)) + mag[min(k_r, len(mag)-1)] * abs(delta))
        amps[i] = 2.0 * amp_interp / (N * cg)
        phase_raw = np.angle(spc[k])
        phases[i] = (phase_raw - 2 * np.pi * freqs_ref[i] * t[0]) % (2 * np.pi)

    # 6. Optional DC-residual peak removal.
    drop = (drop_dc_peak is True) or (drop_dc_peak == 'auto' and f_min is None)
    if drop and n > 1:
        keep = np.argsort(freqs_ref)[1:]         # discard lowest-freq peak
        freqs_ref, phases, amps = freqs_ref[keep], phases[keep], amps[keep]

    # Sort by amplitude, strongest first.
    o = np.argsort(amps)[::-1]
    freqs_ref, phases, amps = freqs_ref[o], phases[o], amps[o]

    out = {
        'freqs': freqs_ref,
        'phases': phases,
        'amps': amps,
        'mag': mag,
        'frq': frq,
        'gap_dom': float(freqs_ref[0]) if len(freqs_ref) else float('nan'),
        'accepted_idx': peak_locs,
    }
    if return_spectrum:
        out['spectrum'] = spc
    return out


def integrate_peak_areas(mag, frq, accepted_idx, *, power=1, freqs_ref=None):
    """
    Integrate the spectrum under each detected peak to get a per-peak *area*.

    This is the accurate replacement for using ``amp`` or ``amp**2`` as the
    per-tone "power": rather than the height of a single bin, it is the integral
    of the (zero-padded) FFT magnitude over the whole lobe that belongs to each
    peak. Because it is a genuine integral it does not over-weight the tallest
    peak the way ``amp**2`` does, and (with ``power=2``) it is Parseval-conserved,
    i.e. the areas sum to the signal's AC energy.

    Peak boundaries. Each peak owns the frequency interval out to the *local
    minimum* (valley) between it and each of its neighbours; the outermost peaks
    extend to the ends of the spectrum. This partitions the band into disjoint
    lobes so the areas add up without double counting.

    Parameters
    ----------
    mag, frq : array
        The one-sided spectrum and its frequency axis, exactly as returned by
        :func:`find_all_peaks` (``res['mag']``, ``res['frq']``). ``frq`` must be
        the DC-dropped axis matching ``mag`` (both start at the first non-DC bin).
    accepted_idx : array of int
        Spectrum indices of the accepted peaks (``res['accepted_idx']``). These
        index into ``mag``/``frq``.
    power : {1, 2}
        Integrand. ``1`` integrates ``|FFT|`` (magnitude); ``2`` integrates
        ``|FFT|**2`` (power / energy, Parseval-conserved). Default 1. Switch to 2
        for a physically-conserved power fraction.
    freqs_ref : array, optional
        Refined peak frequencies (``res['freqs']``), only used to label the
        returned areas so they line up with the amplitude-sorted arrays. If
        given, the returned ``areas`` are ordered to match ``freqs_ref`` (nearest
        accepted peak to each refined freq); otherwise they are in the order of
        ``accepted_idx``.

    Returns
    -------
    dict with keys:
        'areas'      : area under each peak (aligned to ``freqs_ref`` if given,
                       else to ``accepted_idx``)
        'bounds'     : list of (lo_idx, hi_idx) spectrum-index bounds per peak
        'peak_freqs' : the frequency at each peak's bin (same order as 'areas')
        'power'      : the integrand exponent used
    """
    mag = np.asarray(mag, dtype=float)
    frq = np.asarray(frq, dtype=float)
    idx = np.asarray(accepted_idx, dtype=int)
    if idx.size == 0:
        return {'areas': np.array([]), 'bounds': [], 'peak_freqs': np.array([]),
                'power': power}

    integrand = mag if power == 1 else mag**power

    # Order peaks by position along the frequency axis so neighbours are adjacent.
    order_f = np.argsort(idx)
    idx_sorted = idx[order_f]

    # Boundary between two adjacent peaks = the bin of minimum magnitude between
    # them (the valley). Outermost edges go to the ends of the spectrum.
    edges = [0]
    for a, b in zip(idx_sorted[:-1], idx_sorted[1:]):
        lo, hi = int(a), int(b)
        valley = lo + int(np.argmin(mag[lo:hi + 1]))
        edges.append(valley)
    edges.append(len(mag) - 1)

    areas_sorted = np.empty(idx_sorted.size)
    bounds_sorted = []
    for i in range(idx_sorted.size):
        lo = edges[i]
        hi = edges[i + 1]
        if hi <= lo:
            hi = min(lo + 1, len(mag) - 1)
        areas_sorted[i] = _trapz(integrand[lo:hi + 1], frq[lo:hi + 1])
        bounds_sorted.append((int(lo), int(hi)))

    peak_freqs_sorted = frq[idx_sorted]

    # Re-order to match freqs_ref (amplitude-sorted) if the caller wants that
    # alignment; otherwise return in frequency order.
    if freqs_ref is not None:
        freqs_ref = np.asarray(freqs_ref, dtype=float)
        areas = np.empty(freqs_ref.size)
        peak_freqs = np.empty(freqs_ref.size)
        bounds = []
        for j, f in enumerate(freqs_ref):
            m = int(np.argmin(np.abs(peak_freqs_sorted - f)))
            areas[j] = areas_sorted[m]
            peak_freqs[j] = peak_freqs_sorted[m]
            bounds.append(bounds_sorted[m])
    else:
        areas = areas_sorted
        peak_freqs = peak_freqs_sorted
        bounds = bounds_sorted

    return {
        'areas': areas,
        'bounds': bounds,
        'peak_freqs': peak_freqs,
        'power': power,
    }


def analyze_peaks(pc_list, tw_l, *, plot=True, verbose=True, ax=None,
                  return_full=False, **kwargs):
    """
    Thin wrapper around :func:`find_all_peaks` returning ``(freqs, phases)`` in
    the same shape the project's ``fit_decay_rates`` expects. Detects ALL peaks
    (no fixed count) with peak-relative sidelobe suppression.

    Extra kwargs are forwarded to :func:`find_all_peaks`. Set ``plot=False`` to
    skip the figure (e.g. inside a sweep). Set ``return_full=True`` to also get
    the full :func:`find_all_peaks` result dict (with ``mag``, ``frq``,
    ``accepted_idx``, ``amps``) so you can integrate the peak areas afterwards
    via :func:`integrate_peak_areas`; the return is then
    ``(freqs, phases, res)``.
    """
    res = find_all_peaks(pc_list, tw_l, **kwargs)
    freqs, phases, amps = res['freqs'], res['phases'], res['amps']
    mag, frq = res['mag'], res['frq']

    if plot:
        own = ax is None
        if own:
            fig, ax = plt.subplots(figsize=(7, 4))
        colors = plt.cm.tab10.colors
        ax.plot(frq, mag, color='#cc4444', lw=1.2)
        for i, f in enumerate(freqs):
            ax.axvline(f, color=colors[i % len(colors)], lw=1.0, ls='--',
                       alpha=0.8)
        ax.set_xlabel('Frequency')
        ax.set_ylabel('|FFT|')
        x_max = (freqs.max() * 1.2) if len(freqs) else frq[-1]
        ax.set_xlim(0, x_max)
        ax.set_title(f'Spectrum: {len(freqs)} peaks detected')
        ax.grid(True, alpha=0.3)
        if own:
            plt.tight_layout()
            plt.show()

    if verbose:
        print(f"\n{'#':>3}  {'Frequency':>14}  {'Phase (rad)':>12}  {'Amp':>10}")
        print("-" * 46)
        for i, (f, p, A) in enumerate(zip(freqs, phases, amps)):
            print(f"{i+1:>3}  {f:>14.6f}  {p:>12.4f}  {A:>10.5f}")

    if return_full:
        return freqs, phases, res
    return freqs, phases
