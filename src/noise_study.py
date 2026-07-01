"""
Helpers for the noise-study notebook (notebooks/noise_study.ipynb).

The notebook runs the LSZ protocol under four noise conditions
(noiseless, ramp-only, wait-only, ramp+wait) and compares the recovered
spectral quantities. To compare quantities across runs they must be aligned
to the SAME physical transition. We do that by matching every run's detected
energy differences to the true transition frequencies (the same Hungarian
matching the diagnostics already use), so column k always refers to the k-th
true transition regardless of detection order or missing peaks.

This module is pure numpy/matplotlib (no qutip), so the alignment and plotting
logic can be unit-tested independently of the physics stack.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment


# ---------------------------------------------------------------------------
# Alignment
# ---------------------------------------------------------------------------

def align_to_true(detected_energy_diffs, amplitudes, phases, true_differences):
    """
    Assign each true transition the detected mode (freq/amp/phase) closest to
    it, via Hungarian matching on |detected - true|.

    Parameters
    ----------
    detected_energy_diffs : (K,) array
        Detected angular energy differences (2*pi*freq) for this run.
    amplitudes, phases : (K,) arrays
        Per-mode amplitude and phase, in the SAME order as detected_energy_diffs.
    true_differences : (T,) array
        True transition energy differences (the reference ordering).

    Returns
    -------
    dict with keys 'energy', 'freq', 'amp', 'phase', each a (T,) array aligned
    to true_differences. Entries with no assigned detected mode are NaN.
    """
    detected_energy_diffs = np.asarray(detected_energy_diffs, dtype=float)
    amplitudes = np.asarray(amplitudes, dtype=float)
    phases = np.asarray(phases, dtype=float)
    true_differences = np.asarray(true_differences, dtype=float)

    T = len(true_differences)
    out = {key: np.full(T, np.nan) for key in ('energy', 'freq', 'amp', 'phase')}

    if len(detected_energy_diffs) == 0:
        return out

    # rows = detected modes, cols = true transitions.
    cost = np.abs(detected_energy_diffs[:, None] - true_differences[None, :])
    row_idx, col_idx = linear_sum_assignment(cost)

    for r, c in zip(row_idx, col_idx):
        out['energy'][c] = detected_energy_diffs[r]
        out['freq'][c] = detected_energy_diffs[r] / (2 * np.pi)
        out['amp'][c] = amplitudes[r]
        out['phase'][c] = phases[r]
    return out


def aligned_from_results(results, true_differences):
    """Pull (energy_diffs, amplitudes, phases) out of a run's diagnostics
    `results` dict and align them to the true transitions."""
    return align_to_true(
        results['energy_diffs'],
        results['amplitudes'],
        results['phases'],
        true_differences,
    )


def _wrap_phase_diff(d):
    """Wrap a phase difference into (-pi, pi]."""
    return (np.asarray(d) + np.pi) % (2 * np.pi) - np.pi


# ---------------------------------------------------------------------------
# Baseline-vs-theory check (noiseless run)
# ---------------------------------------------------------------------------

def plot_baseline_vs_theory(aligned_noiseless, true_differences,
                            experimental_energies, shifted_true_energies,
                            u_m, u_exact_amplitudes, phi_m, u_exact_phases_rel):
    """
    Four scatter panels for the noiseless run: detected vs theoretical
    frequencies, energies, amplitudes (u_m), and phases (phi_m). A y=x line
    marks a perfect match. Used to confirm the noiseless baseline is correct
    before computing noisy differences against it.
    """
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))

    def _scatter(ax, est, true, title, xlabel, ylabel):
        est = np.asarray(est, dtype=float)
        true = np.asarray(true, dtype=float)
        mask = np.isfinite(est) & np.isfinite(true)
        ax.scatter(true[mask], est[mask], color='steelblue', zorder=3, s=55)
        if mask.any():
            lo = min(true[mask].min(), est[mask].min())
            hi = max(true[mask].max(), est[mask].max())
            diag = np.linspace(lo, hi, 100)
            ax.plot(diag, diag, '--', c='red', label='estimate = theory')
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.legend(frameon=False)
        ax.grid(True, alpha=0.3)

    # Frequencies: detected (aligned) vs true transition frequencies.
    _scatter(axes[0, 0],
             aligned_noiseless['freq'],
             np.asarray(true_differences) / (2 * np.pi),
             'Frequencies', 'theory  f', 'detected  f')

    # Energy levels from turnpike vs true (sorted; both are reconstructions up
    # to ordering, compare_spectrum_to_turnpike already picked the best label).
    _scatter(axes[0, 1],
             np.sort(experimental_energies),
             np.sort(shifted_true_energies),
             'Energy levels', 'true  E', 'reconstructed  E')

    # Amplitudes |u_m| vs exact-state amplitudes (sorted, like the diagnostics).
    _scatter(axes[1, 0],
             np.sort(u_m),
             np.sort(u_exact_amplitudes),
             'Amplitudes |u_m|', 'exact  |u_m|', 'estimated  |u_m|')

    # Phases phi_m vs exact-state relative phases (sorted, like the diagnostics).
    _scatter(axes[1, 1],
             np.sort(phi_m),
             np.sort(u_exact_phases_rel),
             'Phases phi_m', 'exact  phi_m', 'estimated  phi_m')

    fig.suptitle('Noiseless baseline vs theory', fontsize=14)
    fig.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Noisy-vs-noiseless differences
# ---------------------------------------------------------------------------

def plot_noisy_differences(aligned_by_case, true_differences, case_labels):
    """
    For each spectral quantity (frequency, energy, amplitude, phase), plot the
    per-transition difference of each noisy case relative to the noiseless run.

    Parameters
    ----------
    aligned_by_case : dict {case_label: aligned_dict}
        Must include the 'Noiseless' baseline plus the noisy cases.
    true_differences : (T,) array
    case_labels : list of str
        Noisy case labels to plot (e.g. ['Ramp only', 'Wait only', 'Ramp+Wait']).
        The baseline key must be 'Noiseless'.
    """
    base = aligned_by_case['Noiseless']
    T = len(true_differences)
    x = np.arange(T)

    quantities = [
        ('freq', 'Δ frequency  (noisy − noiseless)', False),
        ('energy', 'Δ energy diff  (noisy − noiseless)', False),
        ('amp', 'Δ amplitude  (noisy − noiseless)', False),
        ('phase', 'Δ phase  (noisy − noiseless, wrapped)', True),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    colors = plt.cm.tab10.colors

    for ax, (key, title, is_phase) in zip(axes.ravel(), quantities):
        for j, label in enumerate(case_labels):
            est = aligned_by_case[label][key]
            if is_phase:
                diff = _wrap_phase_diff(est - base[key])
            else:
                diff = est - base[key]
            ax.plot(x, diff, 'o-', color=colors[j % len(colors)],
                    label=label, alpha=0.85)
        ax.axhline(0, color='k', lw=0.8, alpha=0.5)
        ax.set_title(title)
        ax.set_xlabel('true transition index')
        ax.set_ylabel('difference')
        ax.set_xticks(x)
        ax.legend(frameon=False)
        ax.grid(True, alpha=0.3)

    fig.suptitle('Spectral differences of noisy runs vs noiseless baseline',
                 fontsize=14)
    fig.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Noise estimation vs configured real rates
# ---------------------------------------------------------------------------

def plot_noise_estimation(estimation_by_case, real_rates, case_labels):
    """
    Compare estimated coherence decay rates (sorted experimental `lambdas`)
    against the real configured rates (sorted `predict_gamma_rates` output) for
    each noisy case that involves wait noise.

    Parameters
    ----------
    estimation_by_case : dict {case_label: {'estimated': (..,), 'real': (..,)}}
    real_rates : (R,) array
        The real per-pair Gamma_mn rates from predict_gamma_rates(t1, t2).
    case_labels : list of str
        Cases to show (e.g. ['Wait only', 'Ramp+Wait']).
    """
    real_sorted = np.sort(np.asarray(real_rates, dtype=float))
    n = len(case_labels)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5), squeeze=False)
    colors = plt.cm.tab10.colors

    for ax, label in zip(axes[0], case_labels):
        est = np.sort(np.asarray(estimation_by_case[label]['estimated'], dtype=float))
        idx = np.arange(len(est))
        ax.scatter(idx, est, color='steelblue', zorder=3, s=55,
                   label='estimated (experiment)')
        for i, rate in enumerate(real_sorted):
            ax.axhline(rate, color='red', linestyle='--', lw=1.1,
                       label='real (configured)' if i == 0 else None)
        ax.set_title(f'Noise estimation — {label}')
        ax.set_xlabel('rate index (sorted)')
        ax.set_ylabel('coherence decay rate Γ')
        ax.legend(frameon=False)
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Per-run vs theory: two combined (4-in-1) views of a single run
# ---------------------------------------------------------------------------

def _run_theory_quantities(aligned_run, results, true_differences):
    """
    Assemble the four (estimated, theory, labels) quantity sets shared by both
    per-run views below. Returns a list of dicts, one per panel.

    Freqs/energies use the per-transition true reference; amplitudes/phases use
    the exact-state-vector values, sorted (consistent with the diagnostics and
    with plot_baseline_vs_theory).
    """
    return [
        {
            'key': 'freq',
            'title': 'Frequencies',
            'est': np.asarray(aligned_run['freq'], dtype=float),
            'true': np.asarray(true_differences, dtype=float) / (2 * np.pi),
            'ylabel': 'f',
            'index_label': 'true transition index',
        },
        {
            'key': 'energy',
            'title': 'Energy levels',
            'est': np.sort(np.asarray(results['experimental_energies'], dtype=float)),
            'true': np.sort(np.asarray(results['shifted_true_energies'], dtype=float)),
            'ylabel': 'E',
            'index_label': 'energy level index (sorted)',
        },
        {
            'key': 'amp',
            'title': 'Amplitudes |u_m|',
            'est': np.sort(np.asarray(results['u_m'], dtype=float)),
            'true': np.sort(np.asarray(results['u_exact_amplitudes'], dtype=float)),
            'ylabel': '|u_m|',
            'index_label': 'mode index (sorted)',
        },
        {
            'key': 'phase',
            'title': 'Phases phi_m',
            'est': np.sort(np.asarray(results['phi_m'], dtype=float)),
            'true': np.sort(np.asarray(results['u_exact_phases_rel'], dtype=float)),
            'ylabel': 'phi_m',
            'index_label': 'mode index (sorted)',
        },
    ]


def plot_run_vs_theory_scatter(aligned_run, results, true_differences, run_label):
    """
    Scatter view (estimate vs theory with a y=x line) for one run, all four
    quantities in a single 4-in-1 image. Generalizes plot_baseline_vs_theory to
    any run; that function is left untouched.
    """
    panels = _run_theory_quantities(aligned_run, results, true_differences)
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))

    for ax, p in zip(axes.ravel(), panels):
        est, true = p['est'], p['true']
        mask = np.isfinite(est) & np.isfinite(true)
        ax.scatter(true[mask], est[mask], color='steelblue', zorder=3, s=55)
        if mask.any():
            lo = min(true[mask].min(), est[mask].min())
            hi = max(true[mask].max(), est[mask].max())
            diag = np.linspace(lo, hi, 100)
            ax.plot(diag, diag, '--', c='red', label='estimate = theory')
        ax.set_title(p['title'])
        ax.set_xlabel(f"theory  {p['ylabel']}")
        ax.set_ylabel(f"estimated  {p['ylabel']}")
        ax.legend(frameon=False)
        ax.grid(True, alpha=0.3)

    fig.suptitle(f'{run_label} vs theory  (scatter)', fontsize=14)
    fig.tight_layout()
    plt.show()


def plot_run_vs_theory_index(aligned_run, results, true_differences, run_label):
    """
    Index view for one run: x-axis is the (transition / level / mode) index and
    each panel overlays two lines, theory and experimental, for all four
    quantities in a single 4-in-1 image. Deviations read off directly as the gap
    between the lines, with a constant slope across panels.
    """
    panels = _run_theory_quantities(aligned_run, results, true_differences)
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    for ax, p in zip(axes.ravel(), panels):
        est, true = p['est'], p['true']
        x = np.arange(max(len(est), len(true)))
        ax.plot(x[:len(true)], true, 's--', color='red',
                label='theory', alpha=0.85)
        ax.plot(x[:len(est)], est, 'o-', color='steelblue',
                label='experimental', alpha=0.85)
        ax.set_title(p['title'])
        ax.set_xlabel(p['index_label'])
        ax.set_ylabel(p['ylabel'])
        ax.set_xticks(x)
        ax.legend(frameon=False)
        ax.grid(True, alpha=0.3)

    fig.suptitle(f'{run_label} vs theory  (index)', fontsize=14)
    fig.tight_layout()
    plt.show()
