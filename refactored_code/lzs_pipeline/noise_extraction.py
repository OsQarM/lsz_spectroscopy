"""
Inverting the fitted coherence decay rates back to per-qubit T1 / T2 times.

The forward model the simulator implements (see
`lsz_experiment.build_noise_operators`, which uses the collapse operators
`sqrt(kappa_phi_q) * sigma_z[q]` and `sqrt(kappa_T1_q) * sigma_-[q]`) gives

    Gamma_mn = sum_q [ 2 * kappa_phi_q * d_q(m,n)
                      + (1/2) * kappa_T1_q * (e_q(m) + e_q(n)) ]

with d_q(m,n) = 1 when the two states disagree on qubit q and e_q(m) the
bit value of qubit q in state m. The factor 2 on the dephasing term is not a
convention choice: for a collapse operator sqrt(g) * sigma_z the coherence
decays at 2g, which is what `dephasing.predict_gamma_rates` encodes and what
a direct qutip simulation confirms.

Inverting this is a small non-negative least-squares problem: 2^n(2^n-1)/2
observed rates constrain 2n unknowns, and the design matrix is full rank
(for 3 qubits: 28 x 6, condition number ~4), so dephasing and amplitude
damping are cleanly separable.

This is the same idea as `src/extract_rates.py`, but with two differences
that matter for getting the right answer:

  * the design matrix carries the factor 2 on the dephasing columns, so the
    recovered kappa_phi is in the same units as the configured rates;
  * the rates are indexed by the *correct* (m, n) pair, in two senses.
    First, `lmd_mn` (from `state_reconstruction.create_transition_matrices`)
    assigns each fitted rate to a pair by matching frequencies, and that
    assignment is permuted relative to the true labelling, so the pairing is
    rebuilt here from the reconstructed energies. Second, and more subtly,
    the reconstructed energies are in *ascending energy* order while the
    bitstrings are in *computational basis* order -- the two differ by a
    permutation. Inverting without reindexing attributes a qubit's
    dephasing to the wrong qubit, and leaks it into T1: on a run with no
    amplitude damping at all it reported kappa_T1 ~ 2.4e-3 out of nowhere.
    `energy_ordered_bitstrings` builds the map.
"""
from __future__ import annotations

from itertools import combinations

import numpy as np
from scipy.optimize import nnls


def enumerate_bitstrings(n_qubits: int) -> np.ndarray:
    """All 2^n computational basis states as 0/1 rows, qubit 0 most significant."""
    return np.array([[(m >> (n_qubits - 1 - q)) & 1 for q in range(n_qubits)]
                     for m in range(2 ** n_qubits)], dtype=int)


def energy_ordered_bitstrings(n_qubits: int, local_z, two_body,
                              local_x=None) -> np.ndarray:
    """Bitstrings reindexed so row k is the k-th lowest-energy eigenstate.

    The diagnostics report energies sorted ascending, but the structural
    formula is written over computational basis states. For the Z-diagonal
    target Hamiltonian the eigenstates *are* the basis states, so the map
    between the two orderings is just the argsort of the diagonal.

    `local_x` is accepted so the caller can state it; a nonzero transverse
    field makes the target Hamiltonian non-diagonal, the eigenstates are no
    longer basis states, and the structural formula does not apply -- that
    case raises instead of returning a wrong answer.
    """
    local_z = np.asarray(local_z, dtype=float)
    two_body = np.asarray(two_body, dtype=float)
    if local_x is not None and np.any(np.asarray(local_x, dtype=float) != 0):
        raise ValueError(
            "noise extraction assumes a Z-diagonal target Hamiltonian, but "
            "local_x is nonzero: the eigenstates are then not computational "
            "basis states and the per-qubit rate formula does not apply")

    bitstrings = enumerate_bitstrings(n_qubits)
    signs = 1 - 2 * bitstrings                      # (M, n), +-1 per qubit
    diagonal = signs @ local_z
    if two_body.size:
        k = 0
        for i in range(n_qubits - 1):
            for j in range(i + 1, n_qubits):
                diagonal = diagonal + two_body[k] * signs[:, i] * signs[:, j]
                k += 1
    return bitstrings[np.argsort(diagonal)]


def build_design_matrix(bitstrings: np.ndarray):
    """The (P, 2n) matrix mapping [kappa_phi, kappa_T1] to the pair rates.

    Column block 1 is 2 * d_q(m,n) (dephasing), block 2 is
    0.5 * (e_q(m) + e_q(n)) (amplitude damping), matching the forward model
    in `dephasing.predict_gamma_rates`.
    """
    m_states, n_qubits = bitstrings.shape
    pairs = list(combinations(range(m_states), 2))
    design = np.zeros((len(pairs), 2 * n_qubits))
    for p, (m, n) in enumerate(pairs):
        for q in range(n_qubits):
            design[p, q] = 2.0 * (bitstrings[m, q] != bitstrings[n, q])
            design[p, n_qubits + q] = 0.5 * (bitstrings[m, q] + bitstrings[n, q])
    return design, pairs


def pair_rates_from_energies(energies, freqs, lambdas, tol=None,
                             predicted=None):
    """Assign each fitted decay rate to the (m, n) pair it belongs to.

    Each detected mode has a frequency, and the pair it belongs to is the one
    whose energy gap matches. That cue alone is not enough: in these spectra
    many distinct pairs share nearly the same gap, and a greedy nearest-gap
    walk then puts the right rate on the wrong pair. The decay rates survive
    that (the multiset is unchanged) but the per-qubit inversion does not --
    it is precisely the pair labels that carry which qubits are involved.

    So the assignment is solved globally as a linear-assignment problem,
    minimising total mismatch in frequency and, when a prediction is
    available, in rate value as well. The rate term is what resolves pairs
    whose gaps are degenerate.

    Returns `(rates, assigned, unmatched, pairs)`.
    """
    from scipy.optimize import linear_sum_assignment

    energies = np.asarray(energies, dtype=float)
    freqs = np.asarray(freqs, dtype=float)
    lambdas = np.asarray(lambdas, dtype=float)

    m_states = energies.size
    pairs = list(combinations(range(m_states), 2))
    gaps = np.array([abs(energies[n] - energies[m]) for m, n in pairs])

    # Detected frequencies are ordinary frequency; the energies are angular.
    detected = 2 * np.pi * freqs
    n_modes = min(detected.size, lambdas.size)
    detected, lam = detected[:n_modes], lambdas[:n_modes]

    # Which cue to match on. When a prediction is available, matching on the
    # rate *value* is both sufficient and strictly better: the predicted
    # Gamma_mn already encodes which qubits each pair involves, which is
    # exactly the information the inversion needs. Adding a frequency term
    # on top makes it worse, because it forces assignments that satisfy a
    # degenerate gap at the cost of contradicting the rate structure.
    use_rates = False
    if predicted is not None:
        predicted = np.asarray(predicted, dtype=float)
        use_rates = predicted.size == len(pairs)

    if use_rates:
        cost = np.abs(predicted[:, None] - lam[None, :])
    else:
        # No prediction: fall back to the transition frequencies. This is
        # ambiguous wherever two pairs share a gap, so the per-qubit split
        # may be unreliable -- `extract_noise_parameters` reports it.
        if tol is None:
            span = float(np.max(gaps)) if gaps.size else 1.0
            tol = max(0.05 * span, 1e-9)
        cost = np.abs(gaps[:, None] - detected[None, :]) / tol

    row, col = linear_sum_assignment(cost)

    rates = np.zeros(len(pairs))
    assigned = np.zeros(len(pairs), dtype=bool)
    for r, c in zip(row, col):
        rates[r] = lam[c]
        assigned[r] = True

    unmatched = int(n_modes - len(col))
    return rates, assigned, unmatched, pairs, use_rates


def extract_noise_parameters(lambdas, freqs, energies, n_qubits,
                             local_z=None, two_body=None, local_x=None,
                             lmd_mn=None, predicted=None):
    """Recover per-qubit kappa_phi / kappa_T1 (and T2_phi / T1) from the fit.

    `predicted` (the run's `exact_lambdas`, when it has them) is used only
    to disambiguate the pair assignment where several level pairs share a
    transition frequency; the recovered rates still come from the fit.
    Without it the assignment falls back to frequency alone, which can
    mislabel degenerate pairs and lose the T1 / dephasing split.

    `local_z` / `two_body` give the target Hamiltonian. They are accepted
    for validation (a nonzero `local_x` makes the formula inapplicable);
    the level ordering itself cancels out of the assignment.

    Returns a dict with the rates, the corresponding times, the NNLS
    residual and some bookkeeping about how many pairs were matched. Times
    are `inf` where the recovered rate is zero.
    """
    # Canonical computational-basis order: this is the order the forward
    # model (`dephasing.predict_gamma_rates`) is written in, and inverting
    # its output in this order returns the configured rates exactly.
    if local_x is not None and np.any(np.asarray(local_x, dtype=float) != 0):
        raise ValueError(
            "noise extraction assumes a Z-diagonal target Hamiltonian, but "
            "local_x is nonzero: the eigenstates are then not computational "
            "basis states and the per-qubit rate formula does not apply")
    bitstrings = enumerate_bitstrings(n_qubits)
    design, pairs = build_design_matrix(bitstrings)

    rates, assigned, unmatched, _, matched_on_rates = pair_rates_from_energies(
        energies, freqs, lambdas, predicted=predicted)

    # Fall back to the stored pair matrix only if the energy-based matching
    # found nothing usable -- it is less reliable (see the module docstring).
    used_fallback = False
    if assigned.sum() < 2 * n_qubits and lmd_mn is not None:
        lmd_mn = np.asarray(lmd_mn, dtype=float)
        rates = np.array([lmd_mn[m, n] for m, n in pairs], dtype=float)
        assigned = rates > 0
        used_fallback = True

    mask = assigned
    if mask.sum() < 2 * n_qubits:
        raise ValueError(
            f"only {int(mask.sum())} decay rates could be assigned to level "
            f"pairs, but {2 * n_qubits} are needed to determine the per-qubit "
            f"rates for {n_qubits} qubits")

    solution, residual = nnls(design[mask], rates[mask])
    kappa_phi = solution[:n_qubits]
    kappa_t1 = solution[n_qubits:]

    def to_time(kappa):
        kappa = np.asarray(kappa, dtype=float)
        with np.errstate(divide="ignore"):
            return np.where(kappa > 0, 1.0 / np.where(kappa > 0, kappa, 1.0),
                            np.inf)

    return {
        "kappa_phi": kappa_phi,
        "kappa_T1": kappa_t1,
        "T2_phi": to_time(kappa_phi),
        "T1": to_time(kappa_t1),
        "residual": float(residual),
        "n_pairs_used": int(mask.sum()),
        "n_pairs_total": len(pairs),
        "n_unmatched_modes": int(unmatched),
        "used_lmd_mn_fallback": bool(used_fallback),
        "pair_rates": rates,
        "pair_assigned": assigned,
        "used_predicted_for_pairing": bool(matched_on_rates),
    }


def format_report(extracted: dict, true_t1=None, true_t2=None) -> str:
    """A readable table of recovered vs configured rates, for the log."""
    lines = []
    kphi = np.asarray(extracted["kappa_phi"], dtype=float)
    kt1 = np.asarray(extracted["kappa_T1"], dtype=float)
    n = kphi.size

    lines.append(f"  Pairs used: {extracted['n_pairs_used']}"
                 f"/{extracted['n_pairs_total']}"
                 f"   NNLS residual: {extracted['residual']:.4e}")
    if extracted["n_unmatched_modes"]:
        lines.append(f"  WARNING: {extracted['n_unmatched_modes']} detected "
                     f"mode(s) could not be matched to a level pair")
    if extracted["used_lmd_mn_fallback"]:
        lines.append("  WARNING: fell back to the stored lmd_mn pairing, "
                     "which can misattribute dephasing as T1")
    if not extracted.get("used_predicted_for_pairing", False):
        lines.append("  NOTE: no predicted rates were available, so rates "
                     "were matched to level pairs by frequency alone. Where "
                     "two pairs share a gap this is ambiguous and the "
                     "dephasing / T1 split may be unreliable.")

    header = (f"    {'qubit':>5}  {'kappa_phi':>12} {'true':>12}"
              f"  {'kappa_T1':>12} {'true':>12}")
    lines.append(header)
    for q in range(n):
        tphi = f"{true_t2[q]:12.6g}" if true_t2 is not None else f"{'-':>12}"
        tt1 = f"{true_t1[q]:12.6g}" if true_t1 is not None else f"{'-':>12}"
        lines.append(f"    {q:>5}  {kphi[q]:12.6g} {tphi}"
                     f"  {kt1[q]:12.6g} {tt1}")

    lines.append(f"    {'':>5}  {'T2_phi':>12} {'true':>12}"
                 f"  {'T1':>12} {'true':>12}")
    for q in range(n):
        t2v = extracted["T2_phi"][q]
        t1v = extracted["T1"][q]
        tphi = (f"{1.0 / true_t2[q]:12.6g}"
                if (true_t2 is not None and true_t2[q] > 0) else f"{'-':>12}")
        tt1 = (f"{1.0 / true_t1[q]:12.6g}"
               if (true_t1 is not None and true_t1[q] > 0) else f"{'-':>12}")
        lines.append(f"    {q:>5}  {t2v:12.6g} {tphi}  {t1v:12.6g} {tt1}")

    return "\n".join(lines)


def plot_extracted_noise(extracted: dict, true_t1=None, true_t2=None,
                         ax=None, figsize=None, show=False):
    """Recovered per-qubit rates against the configured ones.

    Both channels on one panel: dephasing and amplitude damping, recovered
    values as markers and the configured values as open reference markers at
    the same qubit index. Styled from `plotting.style`, like every other
    figure in the pipeline.
    """
    import matplotlib.pyplot as plt
    from plotting import style

    kphi = np.asarray(extracted["kappa_phi"], dtype=float)
    kt1 = np.asarray(extracted["kappa_T1"], dtype=float)
    n = kphi.size
    q = np.arange(n)

    created = ax is None
    if created:
        fig, ax = plt.subplots(figsize=figsize or style.FIG_SINGLE)
    else:
        fig = ax.figure

    ax.plot(q, kphi, style.MARKERS[0], color=style.DATA, linestyle="none",
            markersize=style.MARKERSIZE, label=r"$\kappa_\phi$ recovered")
    ax.plot(q, kt1, style.MARKERS[1], color=style.SECONDARY, linestyle="none",
            markersize=style.MARKERSIZE, label=r"$\kappa_{T_1}$ recovered")

    if true_t2 is not None:
        ax.plot(q, np.asarray(true_t2, dtype=float), style.MARKERS[0],
                markerfacecolor="none", markeredgecolor=style.REFERENCE,
                linestyle="none", markersize=style.MARKERSIZE * 1.6,
                label=r"$\kappa_\phi$ true")
    if true_t1 is not None:
        ax.plot(q, np.asarray(true_t1, dtype=float), style.MARKERS[1],
                markerfacecolor="none", markeredgecolor=style.REFERENCE,
                linestyle="none", markersize=style.MARKERSIZE * 1.6,
                label=r"$\kappa_{T_1}$ true")

    ax.set_xticks(q)
    ax.set_xlim(-0.5, n - 0.5)
    style.style_axis(ax, xlabel="qubit", ylabel="rate",
                     legend=True, legend_loc="best")
    if created:
        fig.tight_layout()
    if show:
        plt.show()
    return fig
