import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment

from plotting import style
from plotting.diagnostic_plots import plot_energy_difference_match


def compute_true_differences(eigenvalues):
    """Upper-triangular pairwise differences of a set of eigenvalues."""
    diffs = []
    for i in range(len(eigenvalues)):
        for j in range(i + 1, len(eigenvalues)):
            diffs.append(eigenvalues[j] - eigenvalues[i])
    return np.array(diffs)


def check_sampling_adequacy(eigenvalues, max_wait_time, n_sims, verbose=True):
    """
    Check, before running any sweep, whether the planned wait-time grid can
    resolve and correctly sample the oscillations set by `eigenvalues`.

    The population signal during the wait time is a sum of cos(Delta_eps * t)
    terms, one per pair of eigenvalues (Delta_eps in angular-frequency units,
    hbar=1). Two conditions must hold for the FFT peaks to come out right:

      1. Resolution: the FFT's frequency resolution 1/max_wait_time must be
         smaller than the closest spacing between any two true frequencies
         (Delta_eps / 2*pi), or the two peaks blend into one / pull each
         other's fitted position.
      2. Nyquist: the wait-time sampling step dt_wait = max_wait_time/n_sims
         must be fine enough that the fastest true frequency sits below the
         Nyquist limit 1/(2*dt_wait), or it aliases into a lower, spurious
         frequency.

    Returns a dict with the computed quantities and two booleans
    (`resolved`, `nyquist_ok`); prints a human-readable report if `verbose`.
    """
    true_diffs = compute_true_differences(np.asarray(eigenvalues, dtype=float))
    true_freqs = np.sort(np.abs(true_diffs) / (2 * np.pi))
    true_freqs = true_freqs[true_freqs > 0]

    dt_wait = max_wait_time / n_sims
    freq_resolution = 1.0 / max_wait_time
    nyquist_freq = 1.0 / (2 * dt_wait)

    min_spacing = np.min(np.diff(true_freqs)) if len(true_freqs) > 1 else np.inf
    max_freq = true_freqs.max() if len(true_freqs) else 0.0

    resolved = min_spacing >= freq_resolution
    nyquist_ok = max_freq <= nyquist_freq

    if verbose:
        print("--- Sampling adequacy check ---")
        print(f"  {'Closest true freq. spacing:':32} {min_spacing:.6f}")
        print(f"  {'FFT frequency resolution:':32} {freq_resolution:.6f}  "
              f"(1/max_wait_time)")
        status = "OK" if resolved else "FAIL -- peaks may blend/shift"
        print(f"  -> resolution {'sufficient' if resolved else 'INSUFFICIENT'} [{status}]")
        if not resolved:
            needed_wait_time = 1.0 / min_spacing
            print(f"     increase max_wait_time to >= {needed_wait_time:.2f} "
                  f"to resolve the closest pair")
        print()
        print(f"  {'Highest true frequency:':32} {max_freq:.6f}")
        print(f"  {'Nyquist frequency:':32} {nyquist_freq:.6f}  "
              f"(1/(2*dt_wait), dt_wait={dt_wait:.6f})")
        status = "OK" if nyquist_ok else "FAIL -- highest freq. will alias"
        print(f"  -> sampling rate {'sufficient' if nyquist_ok else 'INSUFFICIENT'} [{status}]")
        if not nyquist_ok:
            needed_n_sims = int(np.ceil(max_wait_time * 2 * max_freq)) + 1
            print(f"     increase n_sims to >= {needed_n_sims} "
                  f"(dt_wait <= {1.0 / (2 * max_freq):.6f}) to avoid aliasing")
        print("-" * 32)

    return {
        'true_freqs': true_freqs,
        'min_spacing': min_spacing,
        'freq_resolution': freq_resolution,
        'max_freq': max_freq,
        'nyquist_freq': nyquist_freq,
        'dt_wait': dt_wait,
        'resolved': bool(resolved),
        'nyquist_ok': bool(nyquist_ok),
    }


def table_printout(matched_detected, matched_true, residuals, true_differences, col_idx):
    col_w = 14
    header = f"{'#':>4}  {'Detected Δε':>{col_w}}  {'True Δε':>{col_w}}  {'Error':>{col_w}}  {'Error (%)':>{col_w}}"
    sep = "-" * len(header)
    print(f"\n{'Matched energy differences':^{len(header)}}")
    print(sep)
    print(header)
    print(sep)
    for i, (d, tr, res) in enumerate(zip(matched_detected, matched_true, residuals)):
        pct = res / tr * 100 if tr != 0 else 0.0
        print(f"{i+1:>4}  {d:{col_w}.6f}  {tr:{col_w}.6f}  {res:+{col_w}.6f}  {pct:+{col_w}.3f}%")
    print(sep)
    rms_err = np.sqrt(np.mean(residuals**2))
    print(f"{'RMS error:':>{col_w + 4 + col_w + 2}}  {rms_err:{col_w}.6f}")

    unmatched_true = np.delete(true_differences, col_idx)
    if len(unmatched_true) > 0:
        print(f"\nUnmatched true differences: {np.round(unmatched_true, 6)}")


def scatter_plot(matched_detected, matched_true, residuals, plots=True):
    """Detected vs true energy differences.

    `residuals` is accepted for call-site compatibility but no longer drives a
    colour map -- the sober style shows the deviation as the offset from the
    red dashed identity line, and the numbers are in the table above.
    """
    if not plots:
        return
    plot_energy_difference_match(matched_detected, matched_true, show=True)


def match_energy_differences(experimental_energy_diffs, true_differences):
    """Nearest-neighbour matching via Hungarian algorithm. Returns matched pairs + residuals."""
    cost = np.abs(experimental_energy_diffs[:, None] - true_differences[None, :])
    row_idx, col_idx = linear_sum_assignment(cost)

    matched_detected = experimental_energy_diffs[row_idx]
    matched_true = true_differences[col_idx]
    residuals = matched_detected - matched_true
    return matched_detected, matched_true, residuals, col_idx


def compare_spectrum_to_turnpike(solutions, true_energies):
    """Compare turnpike-reconstructed energy levels to the true spectrum."""
    shifted_true_energies = np.array([e - true_energies[0] for e in true_energies])

    levels = np.array(solutions['levels'])
    levels_rev = np.max(levels) - levels[::-1]

    candidates = {'Forward': levels, 'Reversed': levels_rev}

    def rms_error(est, true):
        return np.sqrt(np.mean((np.sort(est) - np.sort(true))**2))

    best_label = min(candidates, key=lambda k: rms_error(candidates[k], shifted_true_energies))
    experimental_energies = candidates[best_label]

    col_w = 8
    header = "".join(f"{'E'+str(i):>{col_w}}" for i in range(len(shifted_true_energies)))
    print(f"{'':12}{header}")
    print(f"{'True:':12}{''.join(f'{v:{col_w}.4f}' for v in shifted_true_energies)}")
    for label, lvls in candidates.items():
        marker = '  <-- best' if label == best_label else ''
        print(f"{label+':':12}{''.join(f'{v:{col_w}.4f}' for v in lvls)}{marker}")

    fig, axes = plt.subplots(1, 2, figsize=style.FIG_DOUBLE)
    for ax, (label, lvls) in zip(axes, candidates.items()):
        ax.scatter(np.sort(lvls), np.sort(shifted_true_energies),
                   color=style.DATA if label == best_label else style.SECONDARY,
                   marker=style.MARKERS[0] if label == best_label else style.MARKERS[1])
        style.identity_line(ax, lvls)
        style.style_axis(ax, xlabel="estimated energies",
                         ylabel="true energies",
                         title=f"{label}{' (best)' if label == best_label else ''}",
                         legend=True, legend_loc="upper left")
    fig.tight_layout()
    plt.show()

    return experimental_energies, shifted_true_energies
