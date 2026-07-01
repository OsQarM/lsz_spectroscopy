import numpy as np
import matplotlib.pyplot as plt


def fourier_analysis(pc_list, tw_l, n_peaks=28, prominence_threshold=0.01,
                     zero_pad_factor=32, window='hann',
                     min_freq_separation=None, detect_prominence=0.02,
                     f_min=None):
    """
    High-resolution spectral analysis returning frequencies and phases of all
    detected modes. Amplitudes and decay rates are left for fit_decay_rates.

    `min_freq_separation` sets an exclusion zone (in frequency units) around
    every accepted peak: weaker candidates that fall within this window of an
    already-accepted, stronger peak are rejected. This suppresses the small
    spurious peaks that live in the spectral-leakage skirts ("envelope") of a
    strong mode. If None, it defaults to ~1.5 raw FFT bins, which is the true
    resolution limit of the (un-padded) transform — peaks closer than that
    cannot be physically distinct.

    `detect_prominence` is the prominence floor handed to `find_peaks`, as a
    fraction of the spectrum's max magnitude. This is the FIRST gate a peak must
    survive, so it sets the true sensitivity to faint peaks. Lower it (e.g.
    0.002 or 0.0) to detect smaller peaks; `prominence_threshold` cannot recover
    peaks that this gate has already rejected.

    `f_min` ignores the low-frequency part of the spectrum for ANALYSIS only:
    any peak below this frequency is excluded from detection, so the near-zero
    peaks and their envelope are not analysed. The spectrum itself is left fully
    intact and the plot still shows the complete, unmodified spectrum. If None,
    the whole frequency range is analysed.
    """
    from scipy.signal import find_peaks, get_window

    y = np.asarray(pc_list, dtype=float)
    t = np.asarray(tw_l, dtype=float)
    N = len(y)
    dt = t[1] - t[0]
    dc = y.mean()

    if min_freq_separation is None:
        # The genuine frequency resolution is 1/(N*dt) regardless of how much
        # we zero-pad; use a small multiple as the minimum spacing between
        # physically distinct modes.
        min_freq_separation = 3 / (N * dt)

    win = get_window(window, N)
    cg = win.sum() / N
    y_win = (y - dc) * win

    N_pad = zero_pad_factor * N
    spectrum = np.fft.rfft(y_win, n=N_pad)
    freqs = np.fft.rfftfreq(N_pad, dt)
    magnitude = np.abs(spectrum)

    mag = magnitude[1:]
    frq = freqs[1:]
    spc = spectrum[1:]

    min_dist = max(1, zero_pad_factor // 2)
    peak_locs, _ = find_peaks(mag, distance=min_dist,
                              prominence=detect_prominence * mag.max())
    if len(peak_locs) == 0:
        peak_locs = np.array([np.argmax(mag)])

    # Frequency cutoff for ANALYSIS only: drop any detected peak below f_min so
    # the near-zero peaks / their envelope are ignored. The spectrum (mag/frq)
    # is untouched, so the plot still shows the full, unmodified spectrum.
    if f_min is not None:
        peak_locs = peak_locs[frq[peak_locs] >= f_min]
        if len(peak_locs) == 0:
            peak_locs = np.array([np.argmax(mag[frq >= f_min]) + np.searchsorted(frq, f_min)])

    threshold = prominence_threshold * mag[peak_locs].max()
    peak_locs = peak_locs[mag[peak_locs] >= threshold]

    # Greedy selection with an exclusion zone: walk candidates strongest-first
    # and accept a peak only if no already-accepted (stronger) peak sits within
    # `min_freq_separation` of it. This discards the small spurious peaks in the
    # leakage skirts around each strong mode.
    order = np.argsort(mag[peak_locs])[::-1]
    accepted = []
    accepted_freqs = []
    for k in peak_locs[order]:
        if len(accepted) >= n_peaks:
            break
        f_k = frq[k]
        if accepted_freqs and np.min(np.abs(np.array(accepted_freqs) - f_k)) < min_freq_separation:
            continue
        accepted.append(k)
        accepted_freqs.append(f_k)
    peak_locs = np.array(accepted)

    log_mag = np.log(mag + 1e-30)
    freqs_refined = np.empty(len(peak_locs))
    phases = np.empty(len(peak_locs))
    amplitudes_fft = np.empty(len(peak_locs))

    for i, k in enumerate(peak_locs):
        k_l = max(k - 1, 0)
        k_r = min(k + 1, len(mag) - 1)
        a_, b_, c_ = log_mag[k_l], log_mag[k], log_mag[k_r]
        if k_l != k_r:
            delta = 0.5 * (a_ - c_) / (a_ - 2*b_ + c_ + 1e-30)
            delta = np.clip(delta, -1.0, 1.0)
        else:
            delta = 0.0

        df = frq[1] - frq[0]
        freqs_refined[i] = frq[k] + delta * df

        amp_interp = mag[k] if delta == 0 else (
            mag[k_l] * (1 - abs(delta)) + mag[min(k_r, len(mag)-1)] * abs(delta))
        amplitudes_fft[i] = 2.0 * amp_interp / (N * cg)

        phase_raw = np.angle(spc[k])
        phases[i] = (phase_raw - 2 * np.pi * freqs_refined[i] * t[0]) % (2 * np.pi)

    # If we detected the full requested number of peaks, one of them may be a
    # spurious zero-frequency (DC residual) peak. Discard the lowest-frequency
    # peak so we keep the n_peaks-1 genuine modes. If fewer than n_peaks were
    # found, assume no spurious peak and keep them all. When f_min is set the
    # low-frequency junk is already excluded, so this discard would only drop a
    # genuine mode — skip it.
    if f_min is None and len(freqs_refined) >= n_peaks:
        keep = np.argsort(freqs_refined)[1:]
        freqs_refined = freqs_refined[keep]
        phases = phases[keep]
        amplitudes_fft = amplitudes_fft[keep]

    order2 = np.argsort(amplitudes_fft)[::-1]
    freqs_refined = freqs_refined[order2]
    phases = phases[order2]
    amplitudes_fft = amplitudes_fft[order2]

    x_max = freqs_refined.max() * 1.2
    colors = plt.cm.tab10.colors
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    ax = axes[0]
    ax.plot(frq, mag, color='#cc4444', lw=1.2)
    for i, f in enumerate(freqs_refined):
        ax.axvline(f, color=colors[i % len(colors)], lw=1.0, linestyle='--', alpha=0.8)
    ax.set_xlabel('Frequency')
    ax.set_ylabel('|FFT|')
    ax.set_xlim(0, x_max)
    ax.set_title('Spectrum')
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.bar(range(len(amplitudes_fft)), amplitudes_fft,
           color=[colors[i % len(colors)] for i in range(len(amplitudes_fft))])
    ax.set_xlabel('Mode index')
    ax.set_ylabel('FFT Amplitude')
    ax.set_title('Mode Amplitudes (FFT)')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.show()

    print(f"\n{'#':>3}  {'Frequency':>14}  {'Phase (rad)':>12}")
    print("-" * 35)
    for i, (f, p) in enumerate(zip(freqs_refined, phases)):
        print(f"{i+1:>3}  {f:>14.6f}  {p:>12.4f}")

    return freqs_refined, phases


def fourier_analysis_iterative(pc_list, tw_l, n_peaks=28,
                               prominence_threshold=0.01, zero_pad_factor=32,
                               window='hann', max_rounds=None,
                               min_freq_separation=None, verbose=False):
    """
    Iterative (CLEAN-style) spectral analysis.

    Each round:
      1. Run an FFT on the current residual signal.
      2. Pick the single strongest peak (parabolic interpolation for freq/phase).
      3. Fit that peak's amplitude, decay rate, and phase in the time domain
         to the residual.
      4. Subtract the fitted damped cosine from the residual.
      5. Repeat until `n_peaks` are collected, no peak is found, or `max_rounds`
         is reached.

    Returns (freqs, phases) like `fourier_analysis`, plus plots the ORIGINAL
    spectrum with all collected peaks marked.
    """
    from scipy.signal import find_peaks, get_window
    from scipy.optimize import curve_fit

    y_orig = np.asarray(pc_list, dtype=float)
    t = np.asarray(tw_l, dtype=float)
    N = len(y_orig)
    dt = t[1] - t[0]
    dc = y_orig.mean()

    win = get_window(window, N)
    cg = win.sum() / N
    N_pad = zero_pad_factor * N

    # Original spectrum, kept only for final plotting.
    spec_orig = np.fft.rfft((y_orig - dc) * win, n=N_pad)
    freqs_axis = np.fft.rfftfreq(N_pad, dt)
    mag_orig = np.abs(spec_orig)[1:]
    frq_axis = freqs_axis[1:]

    residual = y_orig.copy()
    collected_freqs, collected_phases = [], []
    collected_amps, collected_lambdas = [], []

    if max_rounds is None:
        max_rounds = n_peaks

    df_axis = frq_axis[1] - frq_axis[0]
    # Default minimum separation: a few raw FFT bins (independent of zero-padding).
    # Two peaks closer than this are almost certainly the same mode re-detected.
    if min_freq_separation is None:
        raw_df = 1.0 / (N * dt)
        min_freq_separation = 1.5 * raw_df

    def _ranked_candidates(sig):
        sig_dc = sig.mean()
        sig_win = (sig - sig_dc) * win
        spc = np.fft.rfft(sig_win, n=N_pad)
        mag = np.abs(spc)[1:]
        spc_pos = spc[1:]
        min_dist = max(1, zero_pad_factor // 2)
        peak_locs, _ = find_peaks(mag, distance=min_dist,
                                  prominence=0.02 * mag.max())
        if len(peak_locs) == 0:
            return []
        # Order strongest-first.
        peak_locs = peak_locs[np.argsort(mag[peak_locs])[::-1]]

        log_mag = np.log(mag + 1e-30)
        candidates = []
        for k in peak_locs:
            if mag[k] < prominence_threshold * mag_orig.max():
                break  # remaining are weaker, no point continuing
            k_l = max(k - 1, 0)
            k_r = min(k + 1, len(mag) - 1)
            a_, b_, c_ = log_mag[k_l], log_mag[k], log_mag[k_r]
            if k_l != k_r:
                delta = 0.5 * (a_ - c_) / (a_ - 2*b_ + c_ + 1e-30)
                delta = float(np.clip(delta, -1.0, 1.0))
            else:
                delta = 0.0
            f_refined = frq_axis[k] + delta * df_axis
            phase_raw = np.angle(spc_pos[k])
            phase = (phase_raw - 2 * np.pi * f_refined * t[0]) % (2 * np.pi)
            amp_seed = 2.0 * mag[k] / (N * cg)
            candidates.append((f_refined, phase, amp_seed))
        return candidates

    for rnd in range(max_rounds):
        if len(collected_freqs) >= n_peaks:
            break
        candidates = _ranked_candidates(residual)
        if not candidates:
            if verbose:
                print(f"[round {rnd}] no peak above threshold, stopping.")
            break

        # Reject candidates too close to an already-collected peak — those are
        # leftover lobes from imperfect subtraction, not new modes.
        pick = None
        for cand in candidates:
            f_cand = cand[0]
            if collected_freqs and np.min(np.abs(np.array(collected_freqs) - f_cand)) < min_freq_separation:
                if verbose:
                    print(f"[round {rnd}] skipping f={f_cand:.6f} "
                          f"(within {min_freq_separation:.4g} of an existing peak)")
                continue
            pick = cand
            break

        if pick is None:
            if verbose:
                print(f"[round {rnd}] all candidates are duplicates, stopping.")
            break
        f0, phi0, A0 = pick

        # Time-domain fit of a single damped cosine on the residual.
        dc_seed = residual.mean()

        def model(t_, dc_, A_, lam_, f_, phi_):
            return dc_ + 2*A_ * np.exp(-lam_ * t_) * np.cos(2*np.pi*f_*t_ + phi_)

        p0 = [dc_seed, max(A0, 1e-4), 1e-3, f0, phi0]
        lb = [-np.inf, 0.0, 0.0, max(f0 - 5*df_axis, 0.0), -np.inf]
        ub = [ np.inf, np.inf, np.inf, f0 + 5*df_axis,        np.inf]
        try:
            popt, _ = curve_fit(model, t, residual, p0=p0, bounds=(lb, ub),
                                maxfev=20000, xtol=1e-10, ftol=1e-10)
            dc_f, A_f, lam_f, f_f, phi_f = popt
        except Exception as e:
            if verbose:
                print(f"[round {rnd}] fit failed ({e}); using seed values.")
            dc_f, A_f, lam_f, f_f, phi_f = dc_seed, A0, 1e-3, f0, phi0

        residual = residual - 2*A_f * np.exp(-lam_f * t) * np.cos(2*np.pi*f_f*t + phi_f)

        collected_freqs.append(f_f)
        collected_phases.append(phi_f % (2*np.pi))
        collected_amps.append(A_f)
        collected_lambdas.append(lam_f)

        if verbose:
            print(f"[round {rnd}] f={f_f:.6f}  A={A_f:.4f}  "
                  f"lam={lam_f:.4f}  phi={phi_f:.3f}")

    freqs_out = np.array(collected_freqs)
    phases_out = np.array(collected_phases)
    amps_out = np.array(collected_amps)
    lambdas_out = np.array(collected_lambdas)

    # If we collected the full requested number of peaks, one of them may be a
    # spurious zero-frequency (DC residual) peak. Discard the lowest-frequency
    # peak so we keep the n_peaks-1 genuine modes. If fewer than n_peaks were
    # found, assume no spurious peak and keep them all. (Mirrors fourier_analysis.)
    if len(freqs_out) >= n_peaks:
        keep = np.argsort(freqs_out)[1:]
        freqs_out = freqs_out[keep]
        phases_out = phases_out[keep]
        amps_out = amps_out[keep]
        lambdas_out = lambdas_out[keep]

    order = np.argsort(amps_out)[::-1]
    freqs_out = freqs_out[order]
    phases_out = phases_out[order]
    amps_out = amps_out[order]
    lambdas_out = lambdas_out[order]

    # Plot: original spectrum with all collected peaks marked.
    x_max = (freqs_out.max() * 1.2) if len(freqs_out) else frq_axis[-1]
    colors = plt.cm.tab10.colors
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    ax = axes[0]
    ax.plot(frq_axis, mag_orig, color='#cc4444', lw=1.2, label='original')
    for i, f in enumerate(freqs_out):
        ax.axvline(f, color=colors[i % len(colors)], lw=1.0,
                   linestyle='--', alpha=0.8)
    ax.set_xlabel('Frequency')
    ax.set_ylabel('|FFT|')
    ax.set_xlim(0, x_max)
    ax.set_title(f'Spectrum with {len(freqs_out)} iteratively recovered peaks')
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.bar(range(len(amps_out)), amps_out,
           color=[colors[i % len(colors)] for i in range(len(amps_out))])
    ax.set_xlabel('Mode index (sorted by amplitude)')
    ax.set_ylabel('Fitted amplitude')
    ax.set_title('Mode Amplitudes (time-domain fit)')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.show()

    print(f"\n{'#':>3}  {'Frequency':>14}  {'Phase (rad)':>12}  {'Amp':>10}  {'Lambda':>10}")
    print("-" * 60)
    for i, (f, p, A, lam) in enumerate(zip(freqs_out, phases_out, amps_out,
                                           lambdas_out)):
        print(f"{i+1:>3}  {f:>14.6f}  {p:>12.4f}  {A:>10.5f}  {lam:>10.5f}")

    return freqs_out, phases_out


def fit_decay_rates(pc_list, tw_l, freqs_detected, phases_detected, noise=True):
    """
    Fit amplitudes (and optionally decay rates) and DC offset globally to:

        y(t) = DC + sum_k  A_k * exp(-lambda_k * t) * cos(2*pi*f_k*t + phi_k)
    """
    from scipy.optimize import curve_fit

    y = np.asarray(pc_list, dtype=float)
    t = np.asarray(tw_l, dtype=float)
    omegas = 2 * np.pi * freqs_detected
    K = len(freqs_detected)

    if noise:
        def model(t, dc, *params):
            A_k = params[0::2]
            lam_k = params[1::2]
            out = np.full_like(t, dc)
            for A, lam, omega, phi in zip(A_k, lam_k, omegas, phases_detected):
                out += 2*A * np.exp(-lam * t) * np.cos(omega * t + phi)
            return out

        p0 = np.empty(1 + 2 * K)
        p0[0] = y.mean()
        p0[1::2] = 0.05
        p0[2::2] = 1e-3

        lb = np.zeros(1 + 2 * K)
        ub = np.full(1 + 2 * K, np.inf)

        popt, _ = curve_fit(model, t, y, p0=p0, bounds=(lb, ub),
                            maxfev=100000, xtol=1e-10, ftol=1e-10)

        dc_fit = popt[0]
        amplitudes = popt[1::2]
        lambdas = popt[2::2]
        y_fit = model(t, *popt)
    else:
        def model(t, dc, *A_k):
            out = np.full_like(t, dc)
            for A, omega, phi in zip(A_k, omegas, phases_detected):
                out += 2*A * np.cos(omega * t + phi)
            return out

        p0 = np.empty(1 + K)
        p0[0] = y.mean()
        p0[1:] = 0.05

        lb = np.zeros(1 + K)
        ub = np.full(1 + K, np.inf)

        popt, _ = curve_fit(model, t, y, p0=p0, bounds=(lb, ub),
                            maxfev=100000, xtol=1e-10, ftol=1e-10)

        dc_fit = popt[0]
        amplitudes = popt[1:]
        lambdas = np.full(K, 1e-12)
        y_fit = model(t, *popt)

    plt.figure(figsize=(8, 3))
    plt.plot(t, y, 'b-', lw=1, alpha=0.6, label='data')
    plt.plot(t, y_fit, 'r--', lw=1.5, label='fit')
    rms = np.sqrt(np.mean((y - y_fit)**2))
    plt.title(f'Global fit  (RMS residual = {rms:.5f},  DC = {dc_fit:.4f})')
    plt.xlabel('Time')
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.show()

    return amplitudes, lambdas, dc_fit
