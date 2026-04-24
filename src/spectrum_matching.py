import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment


def compute_true_differences(eigenvalues):
    """Upper-triangular pairwise differences of a set of eigenvalues."""
    diffs = []
    for i in range(len(eigenvalues)):
        for j in range(i + 1, len(eigenvalues)):
            diffs.append(eigenvalues[j] - eigenvalues[i])
    return np.array(diffs)


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


def scatter_plot(matched_detected, matched_true, residuals):
    diag_vals = np.linspace(min(matched_true.min(), matched_detected.min()),
                            max(matched_true.max(), matched_detected.max()), 100)

    plt.figure(figsize=(6, 5))
    sc = plt.scatter(matched_detected, matched_true, c=np.abs(residuals),
                     cmap='RdYlGn_r', s=60, zorder=3)
    plt.colorbar(sc, label='|error|')
    plt.plot(diag_vals, diag_vals, '--', c='steelblue', lw=1.5, label='perfect match')
    plt.xlabel('Detected  Δε')
    plt.ylabel('True  Δε')
    plt.title('Detected vs True Energy Differences')
    plt.legend(frameon=False)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


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

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, (label, lvls) in zip(axes, candidates.items()):
        diag = np.linspace(np.min(lvls), np.max(lvls), 100)
        ax.scatter(np.sort(lvls), np.sort(shifted_true_energies),
                   color='steelblue' if label == best_label else 'gray')
        ax.plot(diag, diag, "--", c="red", label="true = estimate")
        ax.set_xlabel("estimated energies")
        ax.set_ylabel("true energies")
        ax.set_title(f"{label} solution{'  (best)' if label == best_label else ''}")
        ax.legend(frameon=False)
    plt.tight_layout()
    plt.show()

    return experimental_energies, shifted_true_energies
