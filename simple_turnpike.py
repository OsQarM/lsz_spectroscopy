
import numpy as np
from itertools import combinations


def solve_simple_turnpike(distances, n_levels, slack=1e-6):
    """
    Solve the Turnpike Problem given a complete set of pairwise distances.

    Parameters
    ----------
    distances : list or array of float
        All n*(n-1)/2 pairwise distances. Must be complete.
    n_levels  : int
        Number of points to reconstruct.
    slack     : float
        Numerical tolerance for distance matching.

    Returns
    -------
    np.ndarray
        Sorted array of reconstructed positions (anchored so first point = 0).

    Raises
    ------
    ValueError
        If the number of distances is incompatible with n_levels,
        or if no solution is found.
    """
    n_expected = n_levels * (n_levels - 1) // 2
    if len(distances) != n_expected:
        raise ValueError(
            f"Got {len(distances)} distances but n_levels={n_levels} "
            f"requires exactly {n_expected}."
        )

    # Work with a sorted numpy array as a mutable multiset
    D = np.sort(np.array(distances, dtype=float))
    width = D[-1]

    # The two anchor points are 0 and width.
    # Remove the single distance they produce (width itself).
    D = _remove_one(D, width, slack)

    placed = np.array([0.0, width])
    result = _place(D, placed, width, slack)

    if result is None:
        raise ValueError(
            "No solution found. Check that the distance set is complete "
            "and consistent."
        )
    return np.sort(result)


# ── internal helpers ──────────────────────────────────────────────────────────

def _remove_one(arr, val, slack):
    """Return arr with the element closest to val removed (in-place copy)."""
    idx = np.argmin(np.abs(arr - val))
    return np.delete(arr, idx)


def _try_remove_all(D, candidate, placed, slack):
    """
    Try to account for all distances from `candidate` to already-placed points.
    Returns the updated D if successful, or None if any distance is missing.
    """
    D = D.copy()
    for p in placed:
        dist = abs(candidate - p)
        if dist < slack:          # candidate coincides with an existing point
            return None
        idx = np.argmin(np.abs(D - dist))
        if np.abs(D[idx] - dist) > slack:
            return None           # required distance not in multiset
        D = np.delete(D, idx)
    return D


def _place(D, placed, width, slack):
    """
    Backtracking core. At each step the largest remaining distance
    must equal |x - 0| or |x - width| for the next point x.
    """
    if len(D) == 0:
        return placed

    d_max = D[-1]

    # Branch A: place at d_max from origin  →  x = d_max
    D_a = _try_remove_all(D, d_max, placed, slack)
    if D_a is not None:
        result = _place(D_a, np.append(placed, d_max), width, slack)
        if result is not None:
            return result

    # Branch B: place at d_max from width  →  x = width - d_max
    candidate_b = width - d_max
    if abs(candidate_b - d_max) > slack:   # skip if same as branch A
        D_b = _try_remove_all(D, candidate_b, placed, slack)
        if D_b is not None:
            result = _place(D_b, np.append(placed, candidate_b), width, slack)
            if result is not None:
                return result

    return None   # backtrack

