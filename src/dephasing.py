import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations


def calculate_dephasing_rates_upper_bound(gamma_list):
    """All subset-sums of single-qubit dephasing rates (upper bound for lambda_mn)."""
    nqubits = len(gamma_list)
    lambdas = []
    for size in range(1, nqubits + 1):
        for subset in combinations(range(nqubits), size):
            lambdas.append(sum(gamma_list[i] for i in subset))
    return np.unique(lambdas)


def predict_dephasing_rates_exact(egvecs, gamma_dep_list, sz_list):
    """
    Exact lambda_mn for Lindblad operators L_i = sqrt(gamma_i) * sigma_z[i]:
        lambda_mn = sum_i gamma_i / 2 * (<m|sz_i|m> - <n|sz_i|n>)^2
    """
    N = egvecs.shape[0]
    sz_exp = np.array([
        np.real(np.diag(egvecs.conj().T @ sz @ egvecs))
        for sz in sz_list
    ])

    lambda_pred = np.zeros((N, N))
    for m in range(N):
        for n in range(m + 1, N):
            lambda_pred[m, n] = sum(
                gamma / 2 * (sz_exp[i, m] - sz_exp[i, n])**2
                for i, gamma in enumerate(gamma_dep_list)
            )
    return lambda_pred


def plot_dephasing_comparison(lambdas_experimental, possible_lambdas):
    """Scatter experimental lambdas with horizontal lines at predicted upper-bound rates."""
    plt.figure(figsize=(7, 4))
    plt.scatter(range(len(lambdas_experimental)), lambdas_experimental/2,
                color='steelblue', zorder=3, label='experimental')
    for i, rate in enumerate(possible_lambdas):
        plt.axhline(rate, color='red', linestyle='--', lw=1.2,
                    label='predicted (exact)' if i == 0 else None)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.show()
