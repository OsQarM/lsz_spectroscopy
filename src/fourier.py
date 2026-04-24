import numpy as np
import matplotlib.pyplot as plt


def fourier_analysis(pc_list, tw_l, n_peaks=28, prominence_threshold=0.01,
                     zero_pad_factor=32, window='hann'):
    """
    High-resolution spectral analysis returning frequencies and phases of all
    detected modes. Amplitudes and decay rates are left for fit_decay_rates.
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

    N_pad = zero_pad_factor * N
    spectrum = np.fft.rfft(y_win, n=N_pad)
    freqs = np.fft.rfftfreq(N_pad, dt)
    magnitude = np.abs(spectrum)

    mag = magnitude[1:]
    frq = freqs[1:]
    spc = spectrum[1:]

    min_dist = max(1, zero_pad_factor // 2)
    peak_locs, _ = find_peaks(mag, distance=min_dist, prominence=0.02 * mag.max())
    if len(peak_locs) == 0:
        peak_locs = np.array([np.argmax(mag)])

    threshold = prominence_threshold * mag[peak_locs].max()
    peak_locs = peak_locs[mag[peak_locs] >= threshold]

    order = np.argsort(mag[peak_locs])[::-1]
    peak_locs = peak_locs[order[:n_peaks]]

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
