"""
Incomplete Turnpike Problem Solver for LZS Spectroscopy
========================================================

Reconstructs energy level spectra from INCOMPLETE sets of pairwise
differences — the realistic case in LZS spectroscopy where selection
rules or noise cause some transitions to be unobserved.

Two solvers are provided:
  1. solve_turnpike()            — classic exact solver (all differences known)
  2. solve_incomplete_turnpike() — robust solver for partial difference sets
                                   Always returns at least one solution.

Algorithm based on Skiena et al. (1990), extended with:
  - Match-threshold branching for missing differences
  - Multi-candidate exploration from the top of the stack
  - Solution scoring by number of explained observations
  - Greedy fallback to guarantee a solution is always returned
"""

import numpy as np
from itertools import combinations
import random
import sys

# Increase recursion limit for large problems
sys.setrecursionlimit(10000)


# =============================================================================
# Helpers
# =============================================================================

def contains(arr, val, slack=1e-5):
    """Check if val exists in arr within tolerance."""
    if len(arr) == 0:
        return False
    return (np.abs(arr - val) <= slack).any()


def remove_closest(arr, val, slack=1e-5):
    """Remove the element closest to val from arr. Returns (new_array, found)."""
    if len(arr) == 0:
        return arr, False
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

    return explained, len(observed_diffs), explained / max(len(observed_diffs), 1)


# =============================================================================
# Solver 1: Classic exact Turnpike (complete difference sets)
# =============================================================================

def _turnpike_exact(stack, energies, min_e, max_e, slack=0.01):
    """Backtracking solver for the complete turnpike problem."""
    stack = np.sort(stack)

    if len(stack) == 0:
        return np.sort(energies)

    max_dist = stack[-1]

    # Branch A: place at (max_e - max_dist)
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

    # Branch B: place at (min_e + max_dist)
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
    stack = np.sort(np.array(differences, dtype=float))
    width = stack[-1]
    energies = np.array([0.0, width])

    # ✓ Use remove_closest instead of stack[:-1]
    # This correctly removes exactly one copy of `width` from the stack,
    # accounting for the {0 → width} anchor difference.
    stack, found = remove_closest(stack, width, slack)
    if not found:
        raise ValueError("Could not remove anchor difference from stack.")

    try:
        result = _turnpike_exact(stack, energies, 0.0, width, slack)
    except RecursionError:
        raise ValueError(
            "Recursion limit hit — increase sys.setrecursionlimit() "
            "or reduce problem size."
        )

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
    """
    new_stack = stack.copy()
    n_matched = 0
    n_expected = len(energies)

    for e in energies:
        dist = abs(candidate - e)
        if dist < slack:
            # Candidate overlaps an existing level
            n_matched += 1
            continue
        if len(new_stack) == 0:
            # Stack exhausted — this distance is "missing", not a failure
            continue
        new_stack, found = remove_closest(new_stack, dist, slack)
        if found:
            n_matched += 1

    return new_stack, n_matched, n_expected


def _turnpike_incomplete(stack, energies, min_e, max_e, n_target,
                         observed_diffs, slack, match_threshold,
                         best, depth=0, max_depth=200,
                         n_candidates_to_try=3):
    """
    Backtracking solver for the incomplete turnpike problem.

    Key differences from the exact solver:
      - Accepts placements even when some distances are missing
        (controlled by match_threshold).
      - Tries multiple distance seeds (not just the largest).
      - Handles empty stack gracefully (missing diffs are expected).
    """
    # ---- Success: placed all target levels ----
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
    if depth >= max_depth:
        return

    # ---- If the stack is empty, we can still try to place levels ----
    # This happens when many differences are missing. We generate
    # candidate positions by subdividing the remaining gaps.
    if len(stack) == 0:
        _fill_remaining_from_gaps(
            energies, min_e, max_e, n_target, observed_diffs, slack, best
        )
        return

    # ---- Try several of the largest remaining distances ----
    n_try = min(n_candidates_to_try, len(stack))
    tried_candidates = set()

    for k in range(n_try):
        dist_seed = stack[-(k + 1)]

        for candidate in [max_e - dist_seed, min_e + dist_seed]:
            # Round to avoid floating-point duplicate candidates
            cand_key = round(candidate / max(slack, 1e-10))

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


def _fill_remaining_from_gaps(energies, min_e, max_e, n_target,
                              observed_diffs, slack, best):
    """
    When the stack is exhausted but we still need more levels,
    place remaining levels at the midpoints of the largest gaps
    in the current energy set. Then score the result.

    This is a greedy fallback that ensures we always produce a
    complete solution, even if it's approximate.
    """
    sorted_e = np.sort(energies)
    n_remaining = n_target - len(sorted_e)

    for _ in range(n_remaining):
        # Find the largest gap
        gaps = np.diff(sorted_e)
        if len(gaps) == 0:
            break
        idx = np.argmax(gaps)
        midpoint = (sorted_e[idx] + sorted_e[idx + 1]) / 2.0
        sorted_e = np.sort(np.append(sorted_e, midpoint))

    expl, total, frac = score_solution(sorted_e, observed_diffs, slack)
    if frac > best['score']:
        best['score'] = frac
        best['solution'] = sorted_e.copy()
        best['explained'] = expl
        best['total'] = total


# =============================================================================
# Greedy fallback solver (guaranteed to produce a solution)
# =============================================================================

def _greedy_solve(observed_diffs, n_levels, e_max, slack=0.05):
    """
    Greedy algorithm that always produces a complete solution.

    Strategy: iteratively place the level that explains the most
    unmatched observed differences.

    This is used as the fallback when backtracking fails.
    """
    obs = np.sort(np.array(observed_diffs, dtype=float))

    # Start with anchors
    levels = [0.0, e_max]

    for _ in range(n_levels - 2):
        # Generate candidate positions from remaining observed diffs
        candidates = set()
        for d in obs:
            candidates.add(d)              # d from bottom
            candidates.add(e_max - d)      # d from top
            for lev in levels:
                candidates.add(lev + d)    # d above existing level
                candidates.add(lev - d)    # d below existing level

        # Filter: must be in range and not duplicate an existing level
        valid = []
        for c in candidates:
            if c < -slack or c > e_max + slack:
                continue
            if any(abs(c - lev) < slack for lev in levels):
                continue
            valid.append(c)

        if len(valid) == 0:
            # No candidates left — fill gaps
            sorted_levs = np.sort(levels)
            gaps = np.diff(sorted_levs)
            idx = np.argmax(gaps)
            levels.append((sorted_levs[idx] + sorted_levs[idx + 1]) / 2.0)
            continue

        # Score each candidate: how many observed diffs does it explain
        # with the current level set?
        best_cand = valid[0]
        best_new_explained = -1

        for c in valid:
            test_levels = sorted(levels + [c])
            expl, _, _ = score_solution(np.array(test_levels), obs, slack)
            # How many MORE diffs does this candidate explain vs. current?
            current_expl, _, _ = score_solution(np.array(sorted(levels)), obs, slack)
            new_explained = expl - current_expl
            if new_explained > best_new_explained:
                best_new_explained = new_explained
                best_cand = c

        levels.append(best_cand)

    return np.sort(np.array(levels))


# =============================================================================
# Main solver interface
# =============================================================================

def solve_incomplete_turnpike(observed_diffs, n_levels, slack=0.05,
                              match_thresholds=None, n_candidates=3,
                              verbose=True):
    """
    Solve the Turnpike Problem with missing differences.
    ALWAYS returns at least one solution.

    Uses iterative relaxation: starts with a strict match threshold
    and gradually relaxes. If backtracking finds nothing, falls back
    to a greedy solver.

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
        Default: [1.0, 0.8, 0.6, 0.4, 0.2]
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
        'method'    : 'backtracking' or 'greedy_fallback'
    """
    if match_thresholds is None:
        match_thresholds = [1.0, 0.8, 0.6, 0.4, 0.2]

    n_expected_total = n_levels * (n_levels - 1) // 2
    n_observed = len(observed_diffs)

    if verbose:
        print(f"Incomplete Turnpike Solver")
        print(f"  Target levels:        {n_levels}")
        print(f"  Expected differences: {n_expected_total}")
        print(f"  Observed differences: {n_observed} "
              f"({100 * n_observed / max(n_expected_total, 1):.0f}%)")
        print()

    obs = np.sort(np.array(observed_diffs, dtype=float))

    if len(obs) == 0:
        raise ValueError("No observed differences provided.")

    width = obs[-1]

    best = {'solution': None, 'score': 0.0, 'explained': 0, 'total': n_observed}

    # Scale max_depth with problem size
    max_depth = max(n_levels * 5, 100)

    for threshold in match_thresholds:
        if verbose:
            print(f"  Trying match threshold = {threshold:.0%} ... ", end="",
                  flush=True)

        initial_energies = np.array([0.0, width])
        initial_stack = obs[:-1].copy()  # remove width

        try:
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
                max_depth=max_depth,
                n_candidates_to_try=n_candidates,
            )
        except RecursionError:
            if verbose:
                print("hit recursion limit, moving on.")
            continue

        if best['solution'] is not None:
            if verbose:
                print(f"found solution explaining "
                      f"{best['explained']}/{best['total']} "
                      f"({best['score']:.0%}) of observations.")
            if best['score'] >= 0.9:
                break
        else:
            if verbose:
                print("no solution at this threshold.")

    # ---- Greedy fallback: ALWAYS produces a solution ----
    if verbose:
        print(f"\n  Running greedy solver for comparison...", flush=True)

    greedy_result = _greedy_solve(obs, n_levels, width, slack)
    g_expl, g_total, g_frac = score_solution(greedy_result, obs, slack)

    if verbose:
        print(f"  Greedy explains {g_expl}/{g_total} ({g_frac:.0%})")

    # Pick whichever is better
    if best['solution'] is None or g_frac > best['score']:
        best['solution'] = greedy_result
        best['score'] = g_frac
        best['explained'] = g_expl
        best['total'] = g_total
        method = 'greedy_fallback'
    else:
        method = 'backtracking'

    if verbose:
        print(f"\n  Final solution ({method}):")
        print(f"    {list(np.round(best['solution'], 4))}")
        print(f"    Explains {best['explained']}/{best['total']} observed "
              f"differences ({best['score']:.0%})\n")

    return {
        'levels': best['solution'],
        'explained': best['explained'],
        'total': best['total'],
        'score': best['score'],
        'method': method,
    }


# =============================================================================
# Demo
# =============================================================================

def demo():
    print("=" * 65)
    print("  INCOMPLETE TURNPIKE SOLVER  —  LZS Spectroscopy Demo")
    print("=" * 65)

    # ─── Test 1: Complete data ───
    print("\n" + "─" * 65)
    print("  TEST 1: Complete data — 2 qubits (4 levels, all 6 diffs)")
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
    print("  TEST 2: Complete data — 3 qubits (8 levels, all 28 diffs)")
    print("─" * 65 + "\n")

    true_8 = [0.0, 0.8, 2.1, 3.5, 5.0, 6.3, 7.9, 10.0]
    diffs_8 = generate_differences(true_8)
    random.shuffle(diffs_8)

    result_8 = solve_turnpike(diffs_8, slack=0.01)
    print(f"  True:          {true_8}")
    print(f"  Reconstructed: {list(np.round(result_8, 4))}")
    print(f"  Match: {'✓' if np.allclose(result_8, true_8, atol=0.05) else '✗'}")

    # ─── Test 3: Incomplete, 3 qubits, ~60% ───
    print("\n" + "─" * 65)
    print("  TEST 3: Incomplete — 3 qubits, ~60% of 28 diffs")
    print("─" * 65 + "\n")

    np.random.seed(42)
    all_diffs = generate_differences(true_8)
    keep_mask = np.random.random(len(all_diffs)) < 0.6
    keep_mask[-1] = True
    partial_diffs = [d for d, k in zip(all_diffs, keep_mask) if k]

    print(f"  True levels: {true_8}")
    print(f"  Kept {len(partial_diffs)} / {len(all_diffs)} differences\n")

    r3 = solve_incomplete_turnpike(partial_diffs, n_levels=8, slack=0.05,
                                   n_candidates=4)
    print(f"  True:          {true_8}")
    print(f"  Reconstructed: {list(np.round(r3['levels'], 2))}")

    # ─── Test 4: 4 qubits, ~50% ───
    print("\n" + "─" * 65)
    print("  TEST 4: Incomplete — 4 qubits (16 levels), ~50% of 120 diffs")
    print("─" * 65 + "\n")

    np.random.seed(7)
    true_16 = sorted(np.round(np.cumsum(np.random.uniform(0.3, 1.5, 16)), 3))
    true_16 = list(true_16 - true_16[0])
    all_diffs_16 = generate_differences(true_16)
    keep_mask_16 = np.random.random(len(all_diffs_16)) < 0.50
    keep_mask_16[-1] = True
    partial_16 = [d for d, k in zip(all_diffs_16, keep_mask_16) if k]

    print(f"  True levels (16): {[round(x, 2) for x in true_16]}")
    print(f"  Kept {len(partial_16)} / {len(all_diffs_16)} differences\n")

    r4 = solve_incomplete_turnpike(partial_16, n_levels=16, slack=0.05,
                                   n_candidates=4)
    print(f"  True:          {[round(x, 2) for x in true_16]}")
    print(f"  Reconstructed: {list(np.round(r4['levels'], 2))}")

    # ─── Test 5: Severely incomplete — 15 of 120 ───
    print("\n" + "─" * 65)
    print("  TEST 5: Severely incomplete — 4 qubits, ~15/120 diffs")
    print("─" * 65 + "\n")

    np.random.seed(42)
    true_16b = sorted(np.round(np.cumsum(np.random.uniform(0.5, 2.0, 16)), 3))
    true_16b = list(true_16b - true_16b[0])
    all_16b = generate_differences(true_16b)
    keep_mask_b = np.random.random(len(all_16b)) < 0.125
    keep_mask_b[-1] = True
    partial_16b = [d for d, k in zip(all_16b, keep_mask_b) if k]

    print(f"  True levels: {[round(x, 2) for x in true_16b]}")
    print(f"  Kept {len(partial_16b)} / {len(all_16b)} differences\n")

    r5 = solve_incomplete_turnpike(partial_16b, n_levels=16, slack=0.1,
                                   n_candidates=4)
    print(f"  True:          {[round(x, 2) for x in true_16b]}")
    print(f"  Reconstructed: {list(np.round(r5['levels'], 2))}")
    print(f"  Method used:   {r5['method']}")

    print("\n" + "=" * 65)
    print("  All tests complete.")
    print("=" * 65)


if __name__ == "__main__":
    demo()