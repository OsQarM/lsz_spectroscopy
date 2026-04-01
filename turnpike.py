"""
Turnpike Problem Solver for LZS Spectroscopy
=============================================

Given a multiset of pairwise energy differences (transition frequencies)
observed in a Landau-Zener-Stückelberg experiment, reconstruct the
energy level spectrum.

Algorithm: Backtracking approach (Skiena et al., 1990), following the
correct branching logic: the largest remaining distance must be from
a new point to either the current minimum or maximum placed energy.

Uses numpy for robust floating-point distance matching.
"""

import numpy as np
from itertools import combinations
import random


# =============================================================================
# Step 1: Helper functions
# =============================================================================

def contains(arr, val, slack=1e-5):
    """Check if `val` exists in `arr` within floating-point tolerance."""
    return (np.abs(arr - val) <= slack).any()


def is_sorted(arr):
    """Check if a numpy array is sorted in non-decreasing order."""
    return (np.diff(arr) >= 0).all()


# =============================================================================
# Step 2: Core recursive solver
# =============================================================================

def turnpike(stack, distances, energies=[], min_energy=0., max_energy=None,
             slack=0.01, verbose=False):
    """
    Recursively reconstruct energy levels from pairwise differences.

    Parameters
    ----------
    stack : np.ndarray
        Sorted array of remaining (unaccounted) pairwise distances.
    distances : np.ndarray
        The original full set of distances (kept for reference).
    energies : list[float]
        Energy levels placed so far.
    min_energy : float
        The minimum placed energy (always 0 after initialization).
    max_energy : float or None
        The maximum placed energy (the spectral width).
    slack : float
        Tolerance for floating-point matching of distances.
    verbose : bool
        Print debug information at each recursion step.

    Returns
    -------
    np.ndarray or None
        Sorted array of reconstructed energy levels, or None if
        this branch is unsolvable.
    """
    # Ensure inputs are sorted
    if not is_sorted(stack):
        stack = np.sort(stack)
    if len(energies) > 0 and not is_sorted(energies):
        energies = np.sort(energies)

    # ----- Base case: all distances accounted for -----
    if len(stack) == 0:
        return np.sort(energies)

    max_distance = stack[-1]

    # ----- Initialization: place 0 and the largest distance -----
    if len(energies) == 0:
        new_energies = [0, max_distance]
        new_stack = stack[:-1].copy()  # remove the width from the stack
        return turnpike(new_stack, distances, energies=new_energies,
                        min_energy=0., max_energy=max_distance,
                        slack=slack, verbose=verbose)

    # ----- Branch A: new level at (max_energy - max_distance) -----
    # This places a point that is max_distance below the top level.
    if verbose:
        print(f"\n--- Branch A (from top) ---")

    candidate_a = max_energy - max_distance
    dists_a = np.abs(np.array(energies) - candidate_a)

    new_stack = stack.copy()
    all_found = True

    for dist in dists_a:
        if not contains(new_stack, dist, slack=slack):
            all_found = False
            break
        else:
            idx = np.argmin(np.abs(dist - new_stack))
            new_stack = np.delete(new_stack, idx)

    if all_found:
        if verbose:
            print(f"  Candidate: {candidate_a:.4f}")
            print(f"  Placed so far: {sorted(energies)}")
            print(f"  Remaining dists: {len(new_stack)}")
        new_energies = list(energies) + [candidate_a]
        result = turnpike(new_stack, distances, energies=new_energies,
                          min_energy=min_energy, max_energy=max_energy,
                          slack=slack, verbose=verbose)
        if result is not None:
            return result

    # ----- Branch B: new level at (min_energy + max_distance) -----
    # This places a point that is max_distance above the bottom level.
    if verbose:
        print(f"\n--- Branch B (from bottom) ---")

    candidate_b = min_energy + max_distance
    dists_b = np.abs(np.array(energies) - candidate_b)

    new_stack = stack.copy()
    all_found = True

    for dist in dists_b:
        if not contains(new_stack, dist, slack=slack):
            all_found = False
            break
        else:
            idx = np.argmin(np.abs(dist - new_stack))
            new_stack = np.delete(new_stack, idx)

    if all_found:
        if verbose:
            print(f"  Candidate: {candidate_b:.4f}")
            print(f"  Placed so far: {sorted(energies)}")
            print(f"  Remaining dists: {len(new_stack)}")
        new_energies = list(energies) + [candidate_b]
        result = turnpike(new_stack, distances, energies=new_energies,
                          min_energy=min_energy, max_energy=max_energy,
                          slack=slack, verbose=verbose)
        if result is not None:
            return result

    # ----- Both branches failed: backtrack -----
    return None


# =============================================================================
# Step 3: Convenience wrapper
# =============================================================================

def solve_turnpike(differences, slack=0.01, verbose=False):
    """
    Solve the Turnpike Problem from a list of pairwise differences.

    Parameters
    ----------
    differences : list[float]
        All pairwise |E_i - E_j| values. Length must be n*(n-1)/2.
    slack : float
        Tolerance for distance matching.
    verbose : bool
        Print debug output.

    Returns
    -------
    np.ndarray
        Sorted array of reconstructed energy levels (shifted so min = 0).
    """
    m = len(differences)
    n = (1 + (1 + 8 * m) ** 0.5) / 2
    if abs(n - round(n)) > 0.01:
        raise ValueError(
            f"Got {m} differences — not n*(n-1)/2 for any integer n "
            f"(computed n ≈ {n:.2f})."
        )
    n = int(round(n))
    print(f"Reconstructing {n} energy levels from {m} pairwise differences.\n")

    stack = np.sort(np.array(differences, dtype=float))
    result = turnpike(stack, stack.copy(), energies=[], slack=slack, verbose=verbose)

    if result is None:
        raise ValueError("Turnpike problem has no solution with the given data.")
    return result


# =============================================================================
# Step 4: Generate pairwise differences (for testing)
# =============================================================================

def generate_differences(levels):
    """Compute all |E_i - E_j| for i < j."""
    diffs = []
    for i in range(len(levels)):
        for j in range(i + 1, len(levels)):
            diffs.append(abs(levels[i] - levels[j]))
    return sorted(diffs)


# =============================================================================
# Step 5: Demo
# =============================================================================

def demo():
    print("=" * 65)
    print("  TURNPIKE PROBLEM SOLVER  —  LZS Spectroscopy Demo")
    print("=" * 65)

    # --- 2-qubit system: 4 levels, 6 differences ---
    print("\n" + "─" * 65)
    print("  TEST 1:  2-qubit system  (4 levels, 6 differences)")
    print("─" * 65)

    true_levels = [0.0, 1.5, 3.7, 6.0]
    print(f"  True energy levels (GHz): {true_levels}")

    diffs = generate_differences(true_levels)
    print(f"  Pairwise differences:     {diffs}")
    print(f"  (shuffled for input)\n")

    random.shuffle(diffs)
    result = solve_turnpike(diffs)

    print(f"  Reconstructed: {result}")
    print(f"  Expected:      {true_levels}")
    match = np.allclose(result, true_levels, atol=0.01)
    print(f"  Match: {'✓ YES' if match else '✗ NO'}")

    # --- 3-qubit system: 8 levels, 28 differences ---
    print("\n" + "─" * 65)
    print("  TEST 2:  3-qubit system  (8 levels, 28 differences)")
    print("─" * 65)

    true_levels_3q = [0.0, 0.8, 2.1, 3.5, 5.0, 6.3, 7.9, 10.0]
    print(f"  True energy levels (GHz): {true_levels_3q}")

    diffs_3q = generate_differences(true_levels_3q)
    print(f"  Number of differences:    {len(diffs_3q)}\n")

    random.shuffle(diffs_3q)
    result_3q = solve_turnpike(diffs_3q)

    print(f"  Reconstructed: {list(result_3q)}")
    print(f"  Expected:      {true_levels_3q}")
    match_3q = np.allclose(result_3q, true_levels_3q, atol=0.01)
    print(f"  Match: {'✓ YES' if match_3q else '✗ NO'}")

    # --- Noisy data example ---
    print("\n" + "─" * 65)
    print("  TEST 3:  2-qubit system with measurement noise")
    print("─" * 65)

    noise_level = 0.005  # 5 MHz noise on GHz-scale transitions
    true_levels_noisy = [0.0, 1.5, 3.7, 6.0]
    diffs_noisy = generate_differences(true_levels_noisy)
    diffs_noisy = [d + np.random.normal(0, noise_level) for d in diffs_noisy]
    print(f"  True levels:   {true_levels_noisy}")
    print(f"  Noise level:   {noise_level} GHz")
    print(f"  Noisy diffs:   {[f'{d:.4f}' for d in diffs_noisy]}\n")

    result_noisy = solve_turnpike(diffs_noisy, slack=0.05)
    print(f"  Reconstructed: {list(np.round(result_noisy, 4))}")
    print(f"  Expected:      {true_levels_noisy}")
    match_noisy = np.allclose(result_noisy, true_levels_noisy, atol=0.1)
    print(f"  Match (within 0.1 GHz): {'✓ YES' if match_noisy else '✗ NO'}")

    print("\n" + "=" * 65)
    print("  All tests complete.")
    print("=" * 65)


if __name__ == "__main__":
    demo()