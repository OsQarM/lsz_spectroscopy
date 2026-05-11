import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations


def enumerate_bitstrings(N_q):
    return np.array([[(m >> (N_q - 1 - q)) & 1 for q in range(N_q)]
                     for m in range(2 ** N_q)], dtype=int)


def predict_gamma_rates(gamma_dec_list, gamma_dep_list):
    """
    Predict coherence decay rates Gamma_{mn} for all pairs of computational
    basis states from per-qubit amplitude-damping (gamma_dec = 1/T1) and
    pure-dephasing (gamma_dep = 1/T2_phi) rates. Dephasing contribution is
    multiplied by 2 to match the Lindblad direct evaluation.
    """
    kappa_T1 = np.asarray(gamma_dec_list, dtype=float)
    kappa_phi = np.asarray(gamma_dep_list, dtype=float)
    N_q = kappa_T1.size
    bitstrings = enumerate_bitstrings(N_q)
    M = bitstrings.shape[0]

    rates = []
    for m, n in combinations(range(M), 2):
        bm = bitstrings[m]
        bn = bitstrings[n]
        d = (bm != bn).astype(float)
        gamma_phi = float(np.sum(kappa_phi * d)) * 2
        e_sum = bm.astype(float) + bn.astype(float)
        gamma_T1 = 0.5 * float(np.sum(kappa_T1 * e_sum))
        rates.append(gamma_phi + gamma_T1)
    return np.array(rates)


def calculate_dephasing_rates_upper_bound(gamma_list):
    """All subset-sums of single-qubit dephasing rates (upper bound for lambda_mn)."""
    nqubits = len(gamma_list)
    lambdas = []
    for size in range(1, nqubits + 1):
        for subset in combinations(range(nqubits), size):
            lambdas.append(sum(gamma_list[i] for i in subset))
    return np.unique(lambdas)


def predict_dephasing_rates_exact(egvecs, gamma_dep_list, sz_list,
                                   gamma_dec_list=None):
    """
    Exact coherence decay rate Gamma_{mn} for independent per-qubit Lindblad
    channels:
        L_phi,i = sqrt(gamma_dep_i) * sigma_z[i]      (pure dephasing)
        L_T1,i  = sqrt(gamma_dec_i) * sigma_-[i]      (amplitude damping)

    Assuming the eigenstates are computational-basis states (diagonal in Z),
    let b_q(m) in {0,1} be the bit value of qubit q in state m
    (b = 0 -> z = +1, b = 1 -> z = -1). Then:

        Gamma_{mn} = sum_q gamma_dep_q * d_q(m,n)
                   + (1/2) sum_q gamma_dec_q * (b_q(m) + b_q(n))

    where d_q(m,n) = 1 if b_q(m) != b_q(n) else 0. This matches the directly
    validated formula in notebooks/predict_gamma_rates.
    """
    N = egvecs.shape[0]
    sz_exp = np.array([
        np.real(np.diag(egvecs.conj().T @ sz @ egvecs))
        for sz in sz_list
    ])
    bits = ((1.0 - sz_exp) / 2.0)

    lambda_pred = np.zeros((N, N))
    for m in range(N):
        for n in range(m + 1, N):
            gamma_phi = sum(
                gamma * (bits[i, m] != bits[i, n])
                for i, gamma in enumerate(gamma_dep_list)
            )
            gamma_T1 = 0.0
            if gamma_dec_list is not None:
                gamma_T1 = 0.5 * sum(
                    gamma * (bits[i, m] + bits[i, n])
                    for i, gamma in enumerate(gamma_dec_list)
                )
            lambda_pred[m, n] = gamma_phi + gamma_T1
    return lambda_pred


def plot_dephasing_comparison(lambdas_experimental, possible_lambdas):
    """Scatter experimental lambdas with horizontal lines at predicted upper-bound rates."""
    plt.figure(figsize=(7, 4))
    plt.scatter(range(len(lambdas_experimental)), lambdas_experimental,
                color='steelblue', zorder=3, label='experimental')
    for i, rate in enumerate(possible_lambdas):
        plt.axhline(rate, color='red', linestyle='--', lw=1.2,
                    label='predicted (exact)' if i == 0 else None)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.show()
