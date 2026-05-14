"""
Stage A of the noisy-signal refinement: estimate populations {v_S} of the
diagonal (zero-frequency) sector

    y_pop(tau) = sum_S v_S exp(-mu_S tau),    mu_S = sum_{q in S} kappa_q^T1

given previously-extracted per-qubit kappa_q^T1 values. The amplitudes
{v_S} are obtained by NNLS on the residual signal y_meas - y_osc.
"""

import numpy as np
from scipy.optimize import nnls, lsq_linear


def enumerate_subsets(Nq):
    """All 2^Nq subsets, each represented as a frozenset of qubit indices."""
    subsets = []
    for mask in range(1 << Nq):
        S = frozenset(q for q in range(Nq) if (mask >> q) & 1)
        subsets.append(S)
    return subsets


def mu_S_from_kT1(kT1):
    """Return mu_S for every subset S of {0..Nq-1} (length 2^Nq)."""
    Nq = len(kT1)
    subsets = enumerate_subsets(Nq)
    return np.array([sum(kT1[q] for q in S) for S in subsets]), subsets


def oscillatory_signal(tau, a_mn, lmd_mn, w_mn, d_mn):
    """
    Reconstruct the coherence (oscillatory) part of the signal
        y_osc(tau) = 2 sum_{m<n} A_mn exp(-Gamma_mn tau) cos(omega_mn tau + delta_mn).
    """
    M = a_mn.shape[0]
    y = np.zeros_like(tau, dtype=float)
    for m in range(M):
        for n in range(m + 1, M):
            A = a_mn[m, n]
            G = lmd_mn[m, n]
            w = w_mn[m, n]
            d = d_mn[m, n]
            if A == 0 and G == 0 and w == 0:
                continue
            y += 2.0 * A * np.exp(-G * tau) * np.cos(w * tau + d)
    return y


def merge_degenerate_columns(Phi, mu_S, tol=1e-9):
    """
    If multiple subsets share (numerically) the same mu_S, sum their columns
    in Phi and de-duplicate. Returns reduced (Phi', mu_S', groups) where
    groups[i] is a list of original-subset indices contributing to column i.
    """
    order = np.argsort(mu_S)
    Phi_red = []
    mu_red = []
    groups = []
    i = 0
    while i < len(order):
        j = i
        idx = order[i]
        col = Phi[:, idx].copy()
        group = [idx]
        while j + 1 < len(order) and abs(mu_S[order[j + 1]] - mu_S[idx]) < tol:
            j += 1
            col = col + Phi[:, order[j]]
            group.append(order[j])
        Phi_red.append(col)
        mu_red.append(mu_S[idx])
        groups.append(group)
        i = j + 1
    return np.stack(Phi_red, axis=1), np.array(mu_red), groups


def estimate_populations(y_meas, tau, kT1, a_mn, lmd_mn, w_mn, d_mn,
                          allow_negative=False, merge_tol=1e-9):
    """
    Stage A: fit v_S to the residual (y_meas - y_osc) using exp(-mu_S tau)
    columns.

    Returns
    -------
    v        : (2^Nq,) array of populations in the original subset ordering.
    mu_S     : (2^Nq,) corresponding decay rates.
    residual_norm : ||y_meas - y_model||
    """
    mu_S, subsets = mu_S_from_kT1(kT1)
    Phi = np.exp(-mu_S[None, :] * tau[:, None])         # (T, 2^Nq)

    y_osc = oscillatory_signal(tau, a_mn, lmd_mn, w_mn, d_mn)
    r = y_meas - y_osc

    Phi_red, mu_red, groups = merge_degenerate_columns(Phi, mu_S, tol=merge_tol)

    if allow_negative:
        sol = lsq_linear(Phi_red, r)
        v_red = sol.x
    else:
        v_red, _ = nnls(Phi_red, r)

    # Distribute reduced-column amplitude equally over its degenerate originals.
    v_full = np.zeros_like(mu_S)
    for col_amp, group in zip(v_red, groups):
        share = col_amp / len(group)
        for orig_idx in group:
            v_full[orig_idx] = share

    residual_norm = float(np.linalg.norm(Phi @ v_full + y_osc - y_meas))
    return v_full, mu_S, residual_norm
