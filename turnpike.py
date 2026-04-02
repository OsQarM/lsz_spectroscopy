"""
Incomplete Turnpike Problem Solver for LZS Spectroscopy
========================================================

Reconstructs energy level spectra from INCOMPLETE sets of pairwise
differences — the realistic case in LZS spectroscopy where selection
rules or noise cause some transitions to be unobserved.

Two solvers are provided:
  1. solve_turnpike()          — classic exact solver (all differences known)
  2. solve_incomplete_turnpike() — robust solver for partial difference sets

Algorithm based on Skiena et al. (1990), extended with:
  - Match-threshold branching for missing differences
  - Multi-candidate exploration from the top of the stack
  - Solution scoring by number of explained observations
"""

import numpy as np
from itertools import combinations
import random


# =============================================================================
# Helpers
# =============================================================================

def contains(arr, val, slack=1e-5):
    """Check if val exists in arr within tolerance."""
    if len(arr) == 0:
        return False
    return (np.abs(arr - val) <= slack).any()


def remove_closest(arr, val, slack=1e-5):
    """Remove the element closest to val from arr. Returns new array."""
    idx = np.argmin(np.abs(arr - val))
    if np.abs(arr[idx] - val) <= slack:
        return np.delete(arr, idx), True
    return arr, False


def generate_differences(levels):
    """Compute all |E_i - E_j| for i < j."""
    diffs = []
    for i in range(len(levels)):
        for j in range(i + 1, len(levels)):
            diffs.append(abs(levels[i] - levels[j]))
    return sorted(diffs)


def score_solution(energies, observed_diffs, slack=0.01):
    """
    Score a candidate solution: what fraction of observed differences
    does this energy level set explain?

    Returns (n_explained, n_total, fraction).
    """
    predicted = []
    for i in range(len(energies)):
        for j in range(i + 1, len(energies)):
            predicted.append(abs(energies[i] - energies[j]))
    predicted = np.array(sorted(predicted))

    explained = 0
    pred_remaining = predicted.copy()
    for obs in observed_diffs:
        if len(pred_remaining) == 0:
            break
        idx = np.argmin(np.abs(pred_remaining - obs))
        if np.abs(pred_remaining[idx] - obs) <= slack:
            explained += 1
            pred_remaining = np.delete(pred_remaining, idx)

    return explained, len(observed_diffs), explained / len(observed_diffs)


# =============================================================================
# Solver 1: Classic exact Turnpike (complete difference sets)
# =============================================================================

def _turnpike_exact(stack, energies, min_e, max_e, slack=0.01):
    """Backtracking solver for the complete turnpike problem."""
    stack = np.sort(stack)

    if len(stack) == 0:
        return np.sort(energies)

    max_dist = stack[-1]

    # Branch A: place at (max_e - max_dist), i.e. max_dist below the top
    candidate_a = max_e - max_dist
    new_stack = stack.copy()
    all_found = True
    for e in energies:
        new_stack, found = remove_closest(new_stack, abs(candidate_a - e), slack)
        if not found:
            all_found = False
            break

    if all_found:
        result = _turnpike_exact(
            new_stack, np.append(energies, candidate_a), min_e, max_e, slack
        )
        if result is not None:
            return result

    # Branch B: place at (min_e + max_dist), i.e. max_dist above the bottom
    candidate_b = min_e + max_dist
    if abs(candidate_b - candidate_a) > slack:
        new_stack = stack.copy()
        all_found = True
        for e in energies:
            new_stack, found = remove_closest(new_stack, abs(candidate_b - e), slack)
            if not found:
                all_found = False
                break

        if all_found:
            result = _turnpike_exact(
                new_stack, np.append(energies, candidate_b), min_e, max_e, slack
            )
            if result is not None:
                return result

    return None


def solve_turnpike(differences, slack=0.01):
    """
    Solve the classic (complete) Turnpike Problem.

    Parameters
    ----------
    differences : list[float]
        All n*(n-1)/2 pairwise distances.
    slack : float
        Matching tolerance.

    Returns
    -------
    np.ndarray — sorted energy levels.
    """
    stack = np.sort(np.array(differences, dtype=float))
    width = stack[-1]
    energies = np.array([0.0, width])
    stack = stack[:-1]
    result = _turnpike_exact(stack, energies, 0.0, width, slack)
    if result is None:
        raise ValueError("No solution found for the complete turnpike problem.")
    return result


# =============================================================================
# Solver 2: Incomplete Turnpike (missing differences)
# =============================================================================

def _try_place(candidate, energies, stack, slack):
    """
    Try to place a candidate energy level.
    
    For each already-placed energy, check if the distance to the
    candidate exists in the stack. Remove matches found.

    Returns
    -------
    (new_stack, n_matched, n_expected)
        new_stack has matched distances removed.
        n_matched / n_expected is the match fraction.
    """
    new_stack = stack.copy()
    n_matched = 0
    n_expected = len(energies)

    for e in energies:
        dist = abs(candidate - e)
        if dist < slack:
            # Candidate is on top of an existing level — skip this distance
            # but it still "matches" trivially
            n_matched += 1
            continue
        new_stack, found = remove_closest(new_stack, dist, slack)
        if found:
            n_matched += 1

    return new_stack, n_matched, n_expected


def _turnpike_incomplete(stack, energies, min_e, max_e, n_target,
                         observed_diffs, slack, match_threshold,
                         best, depth=0, max_depth=50,
                         n_candidates_to_try=3):
    """
    Backtracking solver for the incomplete turnpike problem.

    Key difference from exact solver: we accept candidate placements
    even when some distances to existing points are missing from the
    stack, as long as the match fraction exceeds match_threshold.

    We also try multiple top-of-stack distances as potential candidates
    (not just the single largest), since the true largest difference
    may be missing from our observations.

    Parameters
    ----------
    stack : np.ndarray
        Remaining unmatched observed distances (sorted).
    energies : np.ndarray
        Levels placed so far.
    min_e, max_e : float
        Spectral bounds.
    n_target : int
        Total number of levels to reconstruct (2^N).
    observed_diffs : np.ndarray
        Full set of observed differences (for scoring).
    slack : float
        Distance matching tolerance.
    match_threshold : float
        Minimum fraction of distances that must match for a placement
        to be accepted (0.0 = accept anything, 1.0 = require all).
    best : dict
        Mutable dict tracking the best solution found so far:
        {'solution': np.ndarray, 'score': float}
    depth : int
        Current recursion depth.
    max_depth : int
        Safety limit on recursion depth.
    n_candidates_to_try : int
        How many of the largest stack values to try as distance seeds.
    """

    # ---- Success: we've placed all target levels ----
    if len(energies) >= n_target:
        sorted_e = np.sort(energies)
        expl, total, frac = score_solution(sorted_e, observed_diffs, slack)
        if frac > best['score']:
            best['score'] = frac
            best['solution'] = sorted_e.copy()
            best['explained'] = expl
            best['total'] = total
        return

    # ---- Safety: depth limit ----
    if depth >= max_depth or len(stack) == 0:
        return

    # ---- Try several of the largest remaining distances ----
    # In the incomplete case, the *true* largest difference might
    # be missing. So we try the top n_candidates_to_try values.
    n_try = min(n_candidates_to_try, len(stack))
    tried_candidates = set()

    for k in range(n_try):
        dist_seed = stack[-(k + 1)]

        for candidate in [max_e - dist_seed, min_e + dist_seed]:
            # Round to avoid floating-point duplicates
            cand_key = round(candidate / slack) * slack

            if cand_key in tried_candidates:
                continue
            tried_candidates.add(cand_key)

            # Skip if out of spectral bounds
            if candidate < min_e - slack or candidate > max_e + slack:
                continue

            # Skip if too close to an already-placed level
            if any(abs(candidate - e) < slack for e in energies):
                continue

            new_stack, n_matched, n_expected = _try_place(
                candidate, energies, stack, slack
            )

            match_frac = n_matched / n_expected if n_expected > 0 else 0

            if match_frac >= match_threshold:
                new_energies = np.append(energies, candidate)
                _turnpike_incomplete(
                    new_stack, new_energies, min_e, max_e, n_target,
                    observed_diffs, slack, match_threshold, best,
                    depth + 1, max_depth, n_candidates_to_try
                )


def solve_incomplete_turnpike(observed_diffs, n_levels, slack=0.05,
                              match_thresholds=None, n_candidates=3,
                              verbose=True):
    """
    Solve the Turnpike Problem with missing differences.

    Uses iterative relaxation: starts with a strict match threshold
    and gradually relaxes, preferring solutions that explain the most
    observed data.

    Parameters
    ----------
    observed_diffs : list[float]
        The INCOMPLETE set of observed pairwise differences.
    n_levels : int
        Number of energy levels to reconstruct (2^N for N qubits).
    slack : float
        Distance matching tolerance.
    match_thresholds : list[float] or None
        Sequence of match fractions to try, from strict to relaxed.
        Default: [1.0, 0.8, 0.6, 0.4]
    n_candidates : int
        Number of top-of-stack values to branch on at each step.
    verbose : bool
        Print progress.

    Returns
    -------
    dict with keys:
        'levels'    : np.ndarray of reconstructed energy levels
        'explained' : number of observed diffs explained
        'total'     : total observed diffs
        'score'     : fraction explained
    """
    if match_thresholds is None:
        match_thresholds = [1.0, 0.8, 0.6, 0.4]

    n_expected_total = n_levels * (n_levels - 1) // 2
    n_observed = len(observed_diffs)

    if verbose:
        print(f"Incomplete Turnpike Solver")
        print(f"  Target levels:       {n_levels}")
        print(f"  Expected differences: {n_expected_total}")
        print(f"  Observed differences: {n_observed} "
              f"({100 * n_observed / n_expected_total:.0f}%)")
        print()

    obs = np.sort(np.array(observed_diffs, dtype=float))
    width = obs[-1]

    best = {'solution': None, 'score': 0.0, 'explained': 0, 'total': n_observed}

    for threshold in match_thresholds:
        if verbose:
            print(f"  Trying match threshold = {threshold:.0%} ... ", end="")

        initial_energies = np.array([0.0, width])
        initial_stack = obs[:-1].copy()  # remove width

        _turnpike_incomplete(
            stack=initial_stack,
            energies=initial_energies,
            min_e=0.0,
            max_e=width,
            n_target=n_levels,
            observed_diffs=obs,
            slack=slack,
            match_threshold=threshold,
            best=best,
            depth=0,
            max_depth=n_levels * 3,
            n_candidates_to_try=n_candidates,
        )

        if best['solution'] is not None:
            if verbose:
                print(f"found solution explaining "
                      f"{best['explained']}/{best['total']} "
                      f"({best['score']:.0%}) of observations.")
            # If we already explain >90% of observations, stop relaxing
            if best['score'] >= 0.9:
                break
        else:
            if verbose:
                print("no solution at this threshold.")

    if best['solution'] is None:
        raise ValueError("Could not find any consistent energy level set.")

    if verbose:
        print(f"\n  Final solution: {list(np.round(best['solution'], 4))}")
        print(f"  Explains {best['explained']}/{best['total']} observed "
              f"differences ({best['score']:.0%})\n")

    return {
        'levels': best['solution'],
        'explained': best['explained'],
        'total': best['total'],
        'score': best['score'],
    }


# =============================================================================
# Demo
# =============================================================================

def demo():
    print("=" * 65)
    print("  INCOMPLETE TURNPIKE SOLVER  —  LZS Spectroscopy Demo")
    print("=" * 65)

    # ─── Test 1: Complete data (should still work perfectly) ───
    print("\n" + "─" * 65)
    print("  TEST 1:  Complete data — 2 qubits (4 levels, all 6 diffs)")
    print("─" * 65 + "\n")

    true_4 = [0.0, 1.5, 3.7, 6.0]
    diffs_4 = generate_differences(true_4)
    random.shuffle(diffs_4)

    result = solve_turnpike(diffs_4, slack=0.01)
    print(f"  True:          {true_4}")
    print(f"  Reconstructed: {list(np.round(result, 4))}")
    print(f"  Match: {'✓' if np.allclose(result, true_4, atol=0.05) else '✗'}")

    # ─── Test 2: Complete data, 3 qubits ───
    print("\n" + "─" * 65)
    print("  TEST 2:  Complete data — 3 qubits (8 levels, all 28 diffs)")
    print("─" * 65 + "\n")

    true_8 = [0.0, 0.8, 2.1, 3.5, 5.0, 6.3, 7.9, 10.0]
    diffs_8 = generate_differences(true_8)
    random.shuffle(diffs_8)

    result_8 = solve_turnpike(diffs_8, slack=0.01)
    print(f"  True:          {true_8}")
    print(f"  Reconstructed: {list(np.round(result_8, 4))}")
    print(f"  Match: {'✓' if np.allclose(result_8, true_8, atol=0.05) else '✗'}")

    # ─── Test 3: Incomplete data — 3 qubits, ~60% of differences ───
    print("\n" + "─" * 65)
    print("  TEST 3:  Incomplete data — 3 qubits, ~60% of 28 diffs")
    print("─" * 65 + "\n")

    true_8 = [0.0, 0.8, 2.1, 3.5, 5.0, 6.3, 7.9, 10.0]
    all_diffs = generate_differences(true_8)
    # Randomly drop ~40% of differences
    np.random.seed(42)
    keep_mask = np.random.random(len(all_diffs)) < 0.6
    # Always keep the largest diff (spectral width)
    keep_mask[-1] = True
    partial_diffs = [d for d, k in zip(all_diffs, keep_mask) if k]

    print(f"  True levels: {true_8}")
    print(f"  Kept {len(partial_diffs)} / {len(all_diffs)} differences\n")

    result_3 = solve_incomplete_turnpike(
        partial_diffs, n_levels=8, slack=0.05, n_candidates=4
    )
    print(f"  True:          {true_8}")
    print(f"  Reconstructed: {list(np.round(result_3['levels'], 2))}")
    match = np.allclose(result_3['levels'], true_8, atol=0.15)
    print(f"  Match (0.15 GHz tol): {'✓' if match else '✗'}")

    # ─── Test 4: Incomplete data — 4 qubits, ~50% of differences ───
    print("\n" + "─" * 65)
    print("  TEST 4:  Incomplete data — 4 qubits (16 levels), ~50% of 120 diffs")
    print("─" * 65 + "\n")

    np.random.seed(7)
    true_16 = sorted(np.round(np.cumsum(np.random.uniform(0.3, 1.5, 16)), 3))
    true_16 = list(true_16 - true_16[0])  # shift to start at 0
    all_diffs_16 = generate_differences(true_16)

    keep_mask_16 = np.random.random(len(all_diffs_16)) < 0.50
    keep_mask_16[-1] = True  # keep the width
    partial_16 = [d for d, k in zip(all_diffs_16, keep_mask_16) if k]

    print(f"  True levels (16): {[round(x, 2) for x in true_16]}")
    print(f"  Kept {len(partial_16)} / {len(all_diffs_16)} differences\n")

    result_4 = solve_incomplete_turnpike(
        partial_16, n_levels=16, slack=0.05, n_candidates=4,
        match_thresholds=[1.0, 0.7, 0.5, 0.35]
    )
    print(f"  True:          {[round(x, 2) for x in true_16]}")
    print(f"  Reconstructed: {list(np.round(result_4['levels'], 2))}")

    # ─── Test 5: Noisy + incomplete ───
    print("\n" + "─" * 65)
    print("  TEST 5:  Noisy + incomplete — 3 qubits")
    print("─" * 65 + "\n")

    true_noisy = [0.0, 0.8, 2.1, 3.5, 5.0, 6.3, 7.9, 10.0]
    all_diffs_n = generate_differences(true_noisy)
    np.random.seed(123)
    # Add Gaussian noise
    noisy_diffs = [d + np.random.normal(0, 0.02) for d in all_diffs_n]
    # Drop ~30%
    keep = np.random.random(len(noisy_diffs)) > 0.3
    keep[-1] = True
    partial_noisy = [d for d, k in zip(noisy_diffs, keep) if k]

    print(f"  True levels: {true_noisy}")
    print(f"  Noise: σ = 0.02 GHz, kept {len(partial_noisy)}/{len(noisy_diffs)}\n")

    result_5 = solve_incomplete_turnpike(
        partial_noisy, n_levels=8, slack=0.08, n_candidates=4
    )
    print(f"  True:          {true_noisy}")
    print(f"  Reconstructed: {list(np.round(result_5['levels'], 2))}")
    match_5 = np.allclose(result_5['levels'], true_noisy, atol=0.2)
    print(f"  Match (0.2 GHz tol): {'✓' if match_5 else '✗'}")

    print("\n" + "=" * 65)
    print("  All tests complete.")
    print("=" * 65)


if __name__ == "__main__":
    demo()
