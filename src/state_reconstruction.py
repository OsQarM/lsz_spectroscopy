import numpy as np
from itertools import combinations
from collections import deque


def create_transition_matrices(experimental_energies, experimental_energy_diffs,
                               experimental_amps, experimental_phases, experimental_decays):
    n = len(experimental_energies)
    w_mn = np.zeros((n, n))
    w_predicted = np.zeros((n, n))
    a_mn = np.zeros((n, n))
    d_mn = np.zeros((n, n))
    lmd_mn = np.zeros((n, n))

    for i in range(n - 1):
        for j in range(i + 1, n):
            w_predicted[i, j] = experimental_energies[j] - experimental_energies[i]

    upper_tri_mask = np.zeros((n, n), dtype=bool)
    for i in range(n - 1):
        for j in range(i + 1, n):
            upper_tri_mask[i, j] = True

    used = set()

    for idx, diff in enumerate(experimental_energy_diffs):
        dist = np.where(upper_tri_mask, np.abs(w_predicted - diff), np.inf)

        ij_pairs = [np.unravel_index(k, (n, n)) for k in np.argsort(dist, axis=None)]
        for i, j in ij_pairs:
            i, j = int(i), int(j)
            if dist[i, j] == np.inf:
                break
            if (i, j) not in used:
                used.add((i, j))
                a_mn[i, j] = experimental_amps[idx]
                d_mn[i, j] = experimental_phases[idx]
                w_mn[i, j] = diff
                lmd_mn[i, j] = experimental_decays[idx]
                break

    return w_mn, a_mn, d_mn, lmd_mn, w_predicted


def compute_u_m(a_mn):
    """Estimate amplitudes |u_m| from the transition amplitude matrix via median of triplet estimates."""
    N = a_mn.shape[0]
    u_m = np.full(N, np.nan)

    for m in range(N):
        others = [k for k in range(N) if k != m]
        estimates = []
        for n, o in combinations(others, 2):
            a_mn_val = a_mn[min(m, n), max(m, n)]
            a_mo_val = a_mn[min(m, o), max(m, o)]
            a_no_val = a_mn[min(n, o), max(n, o)]
            if a_mn_val > 0 and a_mo_val > 0 and a_no_val > 0:
                estimates.append((a_mn_val * a_mo_val / (a_no_val)) ** 0.25)
        if estimates:
            u_m[m] = np.median(estimates)
        else:
            print(f"Warning: no valid triangle for m={m}; u_m[{m}] set to NaN.")

    return u_m


def compute_phi_m(d_mn):
    """Estimate phases phi_m mod pi via BFS over the detected-transition graph."""
    N = d_mn.shape[0]
    phi_m = np.full(N, np.nan)
    phi_m[0] = 0.0
    visited = [False] * N
    visited[0] = True
    queue = deque([0])

    while queue:
        node = queue.popleft()
        for neighbor in range(N):
            if visited[neighbor]:
                continue
            i, j = min(node, neighbor), max(node, neighbor)
            if d_mn[i, j] == 0:
                continue
            sign = -1 if node == i else +1
            phi_m[neighbor] = (phi_m[node] + sign * d_mn[i, j] / 2) % np.pi
            visited[neighbor] = True
            queue.append(neighbor)

    for m in range(N):
        if np.isnan(phi_m[m]):
            print(f"Warning: node {m} unreachable from node 0; phi_m[{m}] set to NaN.")

    return phi_m


def validate_state_vector(experiment, target_H):
    """Run a zero-wait evolution and return true |u_m| and phases mod pi relative to u_0."""
    ramp_1_sim, _, _ = experiment.time_evolution()
    final_state = ramp_1_sim.states[-1].full().flatten()

    egvals, egvecs = np.linalg.eigh(target_H)

    u_exact = egvecs.conj().T @ final_state
    u_exact_amplitudes = np.abs(u_exact)
    u_exact_phases = np.angle(u_exact)
    u_exact_phases_rel = (u_exact_phases - u_exact_phases[0]) % np.pi

    return u_exact_amplitudes, u_exact_phases_rel, egvals, egvecs
