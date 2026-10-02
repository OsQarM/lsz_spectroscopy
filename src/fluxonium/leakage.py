"""
Model-free leakage characterization for the single-fluxonium LZS experiment.

The question these estimators answer: *given only the measured interference
signal* ``P(tw)`` (the ground-state return probability vs. wait time), how much
population is leaking out of the {|0>, |1>} qubit subspace during the pulse,
WITHOUT knowing the Hamiltonian?

Physical picture
----------------
For an ideal, leak-free 2-level Landau-Zener-Stuckelberg process, ``P(tw)`` is a
single damped tone at the plateau qubit gap ``f_gap`` sitting on a DC offset:

    P(tw) = DC + 2 A0 cos(2 pi f_gap tw + phi0)      (+ slow decay)

Leakage into |2>, |3>, ... does two measurable things:

  (a) it opens *new* oscillation frequencies (the other level-gap tones), so the
      spectrum grows secondary peaks that are absent in the clean case; and
  (b) it drains contrast from the dominant tone and shifts the mean, because
      population that left the qubit subspace no longer returns coherently to
      |0> at the end of the pulse.

Both are visible in quantities you already fit (tone amplitudes + DC), so we can
form model-free leakage proxies from a SINGLE experiment. They are proxies /
bounds, not the exact instantaneous L(t) = 1 - sum_{k<2} |<e_k|psi>|^2 (that one
genuinely needs the instantaneous eigenstates, i.e. the Hamiltonian).

What you can and cannot get from one experiment
-----------------------------------------------
* Single experiment, no Hamiltonian:
    - :func:`spectral_leakage`      -> secondary-tone power fraction (proxy (a))
    - :func:`reconstruct_two_level_state` + :func:`leakage_from_state_deviation`
                                    -> deviation of the reconstructed qubit
                                       state from the ideal leak-free
                                       expectation (proxy (b))
  Both increase monotonically with true leakage and bound it from below/around;
  neither needs a separate reference run.

* When is a no-leakage reference needed?
    Only to turn the proxies into an *absolute, calibrated* number. If you can
    take a reference with slow-enough ramps that L ~ 0 (or you know the ideal
    DC/contrast for your prep, e.g. |+>), :func:`calibrate_against_reference`
    rescales the proxy onto it. If that reference is NOT accessible, the
    single-experiment proxies still rank and bound leakage; you just lose the
    absolute normalisation.

Note on state reconstruction: reconstructing the qubit state *from the two
lowest levels only* assumes no leakage, so it cannot by itself MEASURE leakage.
Its value is as an independent consistency check -- disagreement with the ideal
leak-free expectation flags (and, via the missing norm, roughly sizes) leakage.
"""

import numpy as np


def spectral_leakage(amps, freqs=None, *, dc=None, dominant='max',
                     dom_tol=None, power=None):
    """
    Model-free leakage proxy from the per-tone spectral weights (proxy (a)).

    Leakage opens oscillation tones away from the single clean qubit-gap tone.
    The fraction of AC (oscillatory) power that does NOT sit in the dominant
    tone is a lower-bound proxy for leakage:

        L_spec = (sum of secondary tone weights) / (total AC tone weight)

    What weight to use is the key modelling choice, and it is now the caller's:

    * ``power=None`` (default) -- fall back to the tone *amplitudes* themselves
      (``|A|``). This is the old height proxy; it over-weights nothing but it is
      a height, not an area.
    * ``power=<array>`` -- pass precomputed per-tone weights, one per entry of
      ``amps``. The intended use is the **integrated peak area** from
      :func:`fluxonium.spectral_peaks.integrate_peak_areas` (the integral of the
      FFT magnitude, or ``|FFT|**2``, under each peak lobe). Areas are the
      accurate, unbiased weight -- they do not over-weight the tallest peak the
      way ``amp**2`` does, and (for the ``|FFT|**2`` integrand) they are
      Parseval-conserved.

    ``amps`` is still required because ``dominant='max'`` and the returned
    ``dominant_idx`` are defined on it; when ``power`` is given, only its length
    (alignment with ``amps``) matters.

    Parameters
    ----------
    amps : array
        Per-tone amplitudes (e.g. from ``fit_decay_rates``). Used to pick the
        dominant tone and as the fallback weight when ``power`` is None.
    freqs : array, optional
        Matching frequencies; only used if ``dominant`` is a frequency value.
    dc : float, optional
        Unused for the ratio; accepted so callers can pass the full fit result.
    dominant : {'max', float}
        Which tone is the "qubit gap". 'max' picks the largest-weight tone.
        Pass a frequency to force a specific tone as dominant (nearest match).
    dom_tol : float, optional
        If given with a numeric ``dominant``, tones within ``dom_tol`` of it are
        ALSO counted as dominant (merges a split main peak before taking the
        ratio).
    power : array, optional
        Precomputed per-tone weights aligned to ``amps`` (typically integrated
        peak areas). If None, ``|amps|`` is used.

    Returns
    -------
    dict: {'L_spec', 'dominant_idx', 'P_total', 'P_secondary'}
    """
    amps = np.asarray(amps, dtype=float)
    if amps.size == 0:
        return {'L_spec': 0.0, 'dominant_idx': None,
                'P_total': 0.0, 'P_secondary': 0.0}

    if power is None:
        power = np.abs(amps)                      # height fallback (old behaviour)
    else:
        power = np.asarray(power, dtype=float)
        if power.shape != amps.shape:
            raise ValueError(
                f"power must align with amps: got {power.shape} vs {amps.shape}")
    if dominant == 'max':
        dom_mask = np.zeros(amps.size, dtype=bool)
        dom_mask[int(np.argmax(power))] = True
    else:
        if freqs is None:
            raise ValueError("freqs required when dominant is a frequency")
        freqs = np.asarray(freqs, dtype=float)
        i0 = int(np.argmin(np.abs(freqs - float(dominant))))
        dom_mask = np.zeros(amps.size, dtype=bool)
        dom_mask[i0] = True
        if dom_tol is not None:
            dom_mask |= np.abs(freqs - freqs[i0]) <= dom_tol

    P_total = power.sum()
    P_dom = power[dom_mask].sum()
    P_sec = P_total - P_dom
    L_spec = P_sec / P_total if P_total > 0 else 0.0
    return {
        'L_spec': float(L_spec),
        'dominant_idx': int(np.argmax(power)) if dominant == 'max' else i0,
        'P_total': float(P_total),
        'P_secondary': float(P_sec),
    }


def reconstruct_two_level_state(dc, dominant_amp, *, contrast_ideal=None):
    """
    Reconstruct effective qubit-subspace quantities from the DC offset and the
    dominant-tone amplitude of P(tw) (proxy (b), part 1).

    For a leak-free return-probability signal prepared in |+> and read out in
    the same state, P(tw) oscillates between ~0 and ~1 about DC ~ 0.5 with
    contrast ~ 1 (peak-to-peak 2 * 2 A0). Reconstructs:

        contrast  = 4 * dominant_amp        (peak-to-peak of the dominant tone)
        p_return_mean = dc                  (mean return probability)

    These assume the observed signal lives in the 2-level subspace. Any missing
    contrast / DC shift relative to the ideal is the leakage fingerprint.

    Returns
    -------
    dict: {'contrast', 'p_return_mean', 'contrast_ideal', 'contrast_deficit'}
    """
    contrast = 4.0 * float(dominant_amp)     # 2*(2 A0)
    out = {
        'contrast': contrast,
        'p_return_mean': float(dc),
        'contrast_ideal': contrast_ideal,
    }
    if contrast_ideal is not None and contrast_ideal > 0:
        out['contrast_deficit'] = float(max(0.0, 1.0 - contrast / contrast_ideal))
    return out


def leakage_from_state_deviation(dc, dominant_amp, *,
                                 dc_ideal=0.5, contrast_ideal=1.0):
    """
    Leakage proxy from the deviation of the reconstructed qubit state from the
    ideal leak-free expectation (proxy (b), part 2).

    Population that leaks out of {|0>,|1>} does not return coherently, so it both
    reduces the oscillation contrast and pulls the mean return probability off
    its ideal value. This combines both deviations into a single [0, 1]-ish
    proxy:

        L_state = 1 - (contrast / contrast_ideal)       # lost coherent contrast

    with the DC deviation reported alongside as an independent cross-check. Pass
    the ideal values for YOUR preparation/readout (defaults are the |+> -> |+>
    return case: DC ~ 0.5, contrast ~ 1).

    Requires only one experiment. If you do not know the ideal contrast, take it
    from a slow-ramp reference (see :func:`calibrate_against_reference`).

    Returns
    -------
    dict: {'L_state', 'dc_deviation', 'reconstructed'}
    """
    rec = reconstruct_two_level_state(dc, dominant_amp,
                                      contrast_ideal=contrast_ideal)
    L_state = rec.get('contrast_deficit', 0.0)
    return {
        'L_state': float(L_state),
        'dc_deviation': float(abs(dc - dc_ideal)),
        'reconstructed': rec,
    }


def leakage_from_norm_loss(dc=None, dominant_amp=None, *, p_signal=None,
                           p_max_ideal=1.0):
    """
    Leakage proxy from the return-probability ceiling (proxy (c)).

    Unlike the contrast proxy (:func:`leakage_from_state_deviation`), this does
    NOT reference the oscillation minima, so it is insensitive to the pulse
    area / precession-cone geometry that makes the ideal contrast depend on the
    X/Z ratio and the sweep length. It uses only the *peak* of the interference
    fringes:

        L_norm = p_max_ideal - P_max

    Rationale. At a constructive-interference maximum an ideal, leak-free state
    returns fully to the readout state, so ``P_max -> 1`` regardless of how
    deep the minima go. Population that leaked out of {|0>,|1>} cannot return,
    so it caps ``P_max`` below 1. The shortfall is a model-free lower bound on
    leakage whose ideal (``p_max_ideal = 1``) is a true constant -- no
    reference run, no contrast calibration.

    Two ways to supply the peak:

    * ``p_signal`` : the raw measured ``P(tw)`` array -> uses ``P.max()``.
      Simple, but a positive noise spike inflates the peak and makes leakage
      look smaller (optimistic bound).
    * ``dc`` + ``dominant_amp`` : reconstructs the fringe ceiling from the fit
      as ``DC + 2 * A0`` (noise-averaged, preferred). Note the factor is 2*A0
      (half the peak-to-peak), NOT the 4*A0 contrast -- the ceiling is the mean
      plus one amplitude.

    If both are given, ``p_signal`` takes precedence.

    Returns
    -------
    dict: {'L_norm', 'p_max', 'p_max_ideal', 'source'}
    """
    if p_signal is not None:
        p = np.asarray(p_signal, dtype=float)
        p_max = float(p.max()) if p.size else 0.0
        source = 'signal'
    elif dc is not None and dominant_amp is not None:
        p_max = float(dc) + 2.0 * float(dominant_amp)
        source = 'fit'
    else:
        raise ValueError("provide either p_signal, or both dc and dominant_amp")

    L_norm = max(0.0, float(p_max_ideal) - p_max)
    return {
        'L_norm': float(L_norm),
        'p_max': float(p_max),
        'p_max_ideal': float(p_max_ideal),
        'source': source,
    }


def calibrate_against_reference(amps_ref, amps_test, *, dominant='max'):
    """
    Optional absolute calibration using a (near) leak-free reference experiment.

    Given amplitudes from a reference run known to have negligible leakage
    (e.g. slow-enough ramps) and from the test run, returns the *excess*
    secondary-power fraction of the test over the reference. This removes any
    residual secondary power that is instrumental rather than leakage.

        L_cal = max(0, L_spec(test) - L_spec(reference))

    Use only when a low-leakage reference is experimentally accessible. When it
    is not, use :func:`spectral_leakage` / :func:`leakage_from_state_deviation`
    on the single test run directly.
    """
    L_ref = spectral_leakage(amps_ref, dominant=dominant)['L_spec']
    L_test = spectral_leakage(amps_test, dominant=dominant)['L_spec']
    return {
        'L_cal': float(max(0.0, L_test - L_ref)),
        'L_spec_reference': float(L_ref),
        'L_spec_test': float(L_test),
    }


def leakage_report(freqs, amps, dc=None, *, f_gap=None, dom_tol=None,
                   power=None, verbose=True, **_ignored):
    """
    Convenience: compute the spectral-leakage proxy ``L_spec`` two ways and
    return them in one dict. Optionally prints a short table.

    The two variants differ only in the per-tone weight used for the
    secondary-power ratio ``L_spec = P_secondary / P_total``:

    * ``L_spec``        -- uses the weights passed in ``power`` (typically the
      integrated peak AREAS from
      :func:`fluxonium.spectral_peaks.integrate_peak_areas`). The accurate,
      unbiased version. Falls back to the height version if ``power`` is None.
    * ``L_spec_height`` -- uses the tone amplitudes (peak heights) directly.
      This is the earlier, simpler proxy, kept alongside for comparison.

    Parameters
    ----------
    freqs, amps : array
        Fitted tone frequencies and amplitudes.
    dc : float, optional
        Fitted DC offset of P(tw). Reported for reference only; not used in the
        ratio.
    f_gap : float, optional
        Known/expected qubit-gap frequency. If given it is used as the dominant
        tone (with ``dom_tol`` to absorb a split main peak); otherwise the
        largest-weight tone is taken as dominant.
    power : array, optional
        Precomputed per-tone weights aligned to ``amps`` (typically the
        integrated peak areas). Drives ``L_spec``. If None, ``L_spec`` equals
        ``L_spec_height``.

    Extra keyword arguments are accepted and ignored (for backward
    compatibility with callers that still pass the removed L_state / L_norm
    options such as ``dc_ideal``, ``contrast_ideal``, ``p_signal``).
    """
    kw = dict(dominant='max') if f_gap is None else dict(dominant=f_gap,
                                                         dom_tol=dom_tol)
    spec = spectral_leakage(amps, freqs, power=power, **kw)      # area-based
    spec_h = spectral_leakage(amps, freqs, power=None, **kw)     # height-based

    dom_idx = spec['dominant_idx']
    dom_amp = np.asarray(amps, dtype=float)[dom_idx] if len(amps) else 0.0

    report = {
        'L_spec': spec['L_spec'],
        'L_spec_height': spec_h['L_spec'],
        'dominant_freq': float(np.asarray(freqs)[dom_idx]) if len(freqs) else None,
        'dominant_amp': float(dom_amp),
        'n_secondary_tones': int(len(amps) - 1) if len(amps) else 0,
        'spectral_detail': spec,
        'spectral_detail_height': spec_h,
    }
    if verbose:
        print("Model-free spectral leakage (single experiment, no Hamiltonian)")
        print("-" * 62)
        print(f"  dominant tone            : f = {report['dominant_freq']}"
              f"   A = {dom_amp:.5f}")
        print(f"  secondary tones          : {report['n_secondary_tones']}")
        print(f"  L_spec (peak area)       : {report['L_spec']:.4f}")
        print(f"  L_spec (peak height)     : {report['L_spec_height']:.4f}")
    return report
