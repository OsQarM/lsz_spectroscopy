"""
Per-qubit dephasing/T1 rate extraction from observed coherence decay rates
{Gamma_mn} via NNLS on the structural formula

    Gamma_mn = sum_q [ kappa_q^phi d_q(m,n) + (1/2) kappa_q^T1 (e_q(m)+e_q(n)) ]

where
    d_q(m,n) = 1 if bitstrings disagree on qubit q else 0
    e_q(m)   = bit_q(m) in {0,1}.

Inputs:
    lambdas_pair : list of (m, n, lambda) tuples for m < n.
    bitstrings   : (M, N_q) array of {0,1}, row m = bitstring of eigenstate m.

Returns (kphi, kT1) arrays of length N_q (non-negative).
"""

import numpy as np
from itertools import combinations
from scipy.optimize import nnls


def build_rate_design_matrix(bitstrings):
    """
    Build the (P, 2*N_q) design matrix mapping
        x = [kphi_1, ..., kphi_Nq, kT1_1, ..., kT1_Nq]
    to predicted Gamma values stacked over all (m<n) pairs.

    Column ordering: first N_q columns -> d_q(m,n) ; next N_q -> 0.5*(b_q(m)+b_q(n)).
    """
    M, Nq = bitstrings.shape
    pairs = list(combinations(range(M), 2))
    P = len(pairs)
    A = np.zeros((P, 2 * Nq))
    for p, (m, n) in enumerate(pairs):
        for q in range(Nq):
            d_q = 1.0 if bitstrings[m, q] != bitstrings[n, q] else 0.0
            e_q = float(bitstrings[m, q] + bitstrings[n, q])
            A[p, q] = d_q
            A[p, Nq + q] = 0.5 * e_q
    return A, pairs


def nnls_extract_kappas(lambdas_mn, bitstrings):
    """
    Solve
        min ||A x - b||_2   s.t. x >= 0
    where b is the vector of observed Gamma_{m<n} and A is the design matrix.

    Parameters
    ----------
    lambdas_mn : (M, M) upper-triangular matrix of decay rates (only m<n used).
    bitstrings : (M, N_q) array.

    Returns
    -------
    kphi : (N_q,) non-negative.
    kT1  : (N_q,) non-negative.
    pairs : list of (m, n) pairs in same order as b.
    residual : float, ||A x - b||.
    """
    M, Nq = bitstrings.shape
    A, pairs = build_rate_design_matrix(bitstrings)
    b = np.array([lambdas_mn[m, n] for m, n in pairs], dtype=float)
    x, residual = nnls(A, b)
    kphi = x[:Nq]
    kT1 = x[Nq:]
    return kphi, kT1, pairs, float(residual)


def bitstrings_from_egvecs(egvecs):
    """
    Derive per-eigenstate bitstrings from the columns of egvecs of a
    Z-diagonal target Hamiltonian. Each eigenvector should align with a
    single computational basis state; the bitstring is read off from the
    dominant amplitude.

    Convention: leftmost bit = qubit 1 (most significant), matching
    enumerate_bitstrings in dephasing.py.
    """
    M = egvecs.shape[0]
    Nq = int(np.log2(M))
    bitstrings = np.zeros((M, Nq), dtype=int)
    for m in range(M):
        idx = int(np.argmax(np.abs(egvecs[:, m])))
        for q in range(Nq):
            bitstrings[m, q] = (idx >> (Nq - 1 - q)) & 1
    return bitstrings
