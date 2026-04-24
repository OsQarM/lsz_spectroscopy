"""
Incomplete Turnpike Problem Solver for LZS Spectroscopy
========================================================

Reconstructs energy level spectra from INCOMPLETE and/or NOISY sets of
pairwise differences — the realistic case in LZS spectroscopy where 
selection rules or noise cause some transitions to be unobserved.

Three solvers provided:
  1. solve_turnpike_exact()  — Classic backtracking (complete data only)
  2. solve_turnpike_ddm()    — Distance Distribution Matching via projected
                               gradient descent (Huang & Dokmanić, IEEE TSP 2021).
                               Handles noise + missing differences natively.
  3. solve_turnpike_combined() — Tries exact first, falls back to DDM.

The DDM approach discretizes the energy axis and represents the N energy
levels as an N-hot density vector x. It then minimizes the mismatch between
the distance distribution predicted by x and the empirical distribution
built from the observed differences. This is solved via projected gradient
descent and naturally handles missing data: the empirical distribution is
simply built from whatever differences are available, renormalized, and
the optimizer finds the best-fitting point set.

References:
  [1] S. Huang, I. Dokmanić, "Reconstructing Point Sets from Distance
      Distributions," IEEE Trans. Signal Process., 69, 2021.
  [2] C.S. Elder et al., "Approximate and Exact Optimization Algorithms
      for the Beltway and Turnpike Problems…," J. Comput. Biol., 2024.
  [3] S. Skiena, W. Smith, P. Lemke, "Reconstructing Sets from
      Interpoint Distances," SoCG 1990.
"""

import numpy as np
from itertools import combinations
import random


# =============================================================================
# Part A: Classic exact Turnpike solver (for complete data)
# =============================================================================

def _contains(arr, val, slack=1e-5):
    if len(arr) == 0:
        return False
    return (np.abs(arr - val) <= slack).any()


def _remove_closest(arr, val, slack=1e-5):
    idx = np.argmin(np.abs(arr - val))
    if np.abs(arr[idx] - val) <= slack:
        return np.delete(arr, idx), True
    return arr, False


def _backtrack_exact(stack, energies, min_e, max_e, slack=0.01):
    stack = np.sort(stack)
    if len(stack) == 0:
        return np.sort(energies)

    d_max = stack[-1]

    # Branch A: candidate at max_e - d_max
    cand_a = max_e - d_max
    s = stack.copy()
    ok = True
    for e in energies:
        s, found = _remove_closest(s, abs(cand_a - e), slack)
        if not found:
            ok = False
            break
    if ok:
        r = _backtrack_exact(s, np.append(energies, cand_a), min_e, max_e, slack)
        if r is not None:
            return r

    # Branch B: candidate at min_e + d_max
    cand_b = min_e + d_max
    if abs(cand_b - cand_a) > slack:
        s = stack.copy()
        ok = True
        for e in energies:
            s, found = _remove_closest(s, abs(cand_b - e), slack)
            if not found:
                ok = False
                break
        if ok:
            r = _backtrack_exact(s, np.append(energies, cand_b), min_e, max_e, slack)
            if r is not None:
                return r

    return None


def solve_turnpike_exact(differences, slack=0.01):
    """
    Classic backtracking Turnpike solver. Requires ALL n*(n-1)/2 differences.
    """
    stack = np.sort(np.array(differences, dtype=float))
    width = stack[-1]
    energies = np.array([0.0, width])
    stack = stack[:-1]
    result = _backtrack_exact(stack, energies, 0.0, width, slack)
    if result is None:
        raise ValueError("No solution found (exact solver).")
    return result


# =============================================================================
# Part B: Distance Distribution Matching (DDM) solver
#         Core algorithm for incomplete / noisy data
# =============================================================================

class DDMSolver:
    """
    Solves the (incomplete, noisy) Turnpike problem via Distance Distribution
    Matching with projected gradient descent.

    The domain [0, E_max] is discretized into M bins. The unknown energy
    levels are represented by a relaxed N-hot vector x in R^M, where x_m
    is the (unnormalized) probability that a level sits at bin m.

    The predicted distance distribution q(y) = (1/K) x^T A_y x is compared
    to the empirical distribution p(y) built from the observed differences.
    We minimize  sum_y (q(y) - p(y))^2  subject to  0 <= x_m <= 1, sum x_m = N.

    This formulation:
      - Handles missing differences: p(y) is built from whatever is observed.
      - Handles noise: Gaussian smoothing of p(y) absorbs measurement error.
      - Always produces a result: gradient descent converges to a local min.
    """

    def __init__(self, n_levels, observed_diffs, e_max=None,
                 grid_oversample=4, sigma=None, verbose=True):
        """
        Parameters
        ----------
        n_levels : int
            Number of energy levels to reconstruct (2^N for N qubits).
        observed_diffs : array-like
            Incomplete/noisy set of observed pairwise energy differences.
        e_max : float or None
            Upper bound of energy range. If None, uses max(observed_diffs)*1.05.
        grid_oversample : int
            Controls grid resolution: M = grid_oversample * n_levels^2.
            Higher = more precise but slower.
        sigma : float or None
            Gaussian smoothing width for empirical distribution.
            If None, auto-set to 2 * bin_width.
        verbose : bool
            Print progress.
        """
        self.N = n_levels
        self.K = n_levels * (n_levels - 1) // 2 + n_levels  # include self-dists
        self.diffs = np.array(observed_diffs, dtype=float)
        self.verbose = verbose

        # Domain setup
        if e_max is None:
            e_max = np.max(self.diffs) * 1.05
        self.e_max = e_max
        self.M = max(grid_oversample * n_levels ** 2, 50)
        self.lam = e_max / self.M  # bin width
        if self.verbose:
            print(f"  DDM grid: M={self.M} bins, bin width={self.lam:.4f}, "
                  f"range=[0, {e_max:.4f}]")

        # Smoothing
        if sigma is None:
            sigma = 2.0 * self.lam
        self.sigma = sigma

        # Build measurement matrices A_y (Toeplitz structure)
        self._build_measurement_matrices()

        # Build target distribution p(y)
        self._build_target_distribution()

    def _build_measurement_matrices(self):
        """
        For the turnpike problem, A_y is an M×M matrix where
        A_y[i,j] = 1 if j - i == y and i <= j, else 0.

        Since these are sparse Toeplitz matrices, we store them
        efficiently and compute x^T A_y x via correlation.
        """
        # We don't store full matrices; instead we compute q(y) directly
        # using the autocorrelation interpretation.
        pass

    def _compute_q(self, x):
        """
        Compute predicted distance distribution q(y) for y = 0, ..., M-1.

        q(y) = (1/K) * sum_{i: i+y <= M-1} x[i] * x[i+y]

        This is the autocorrelation of x (one-sided).
        """
        q = np.zeros(self.M)
        for y in range(self.M):
            q[y] = np.sum(x[:self.M - y] * x[y:self.M])
        q /= self.K
        return q

    def _compute_q_fast(self, x):
        """
        Compute q(y) via FFT-based autocorrelation for speed.
        """
        # Zero-pad x to avoid circular correlation
        x_pad = np.zeros(2 * self.M)
        x_pad[:self.M] = x
        X = np.fft.rfft(x_pad)
        autocorr = np.fft.irfft(X * np.conj(X))
        q = autocorr[:self.M] / self.K
        return q

    def _build_target_distribution(self):
        """
        Build empirical distance distribution p(y) from observed differences.
        Each observed difference d_k contributes a Gaussian centered at
        d_k / lambda (in bin units) with width sigma / lambda.

        Missing differences simply don't contribute — the distribution
        is renormalized so it still sums to 1.
        """
        self.p = np.zeros(self.M)
        sig_bins = self.sigma / self.lam
        y_grid = np.arange(self.M)

        # Add Gaussian for each observed diff
        for d in self.diffs:
            y_center = d / self.lam
            self.p += np.exp(-0.5 * ((y_grid - y_center) / sig_bins) ** 2)

        # Add self-distance peak at y=0
        self.p[0] += self.N * np.exp(-0.5 * (y_grid / sig_bins) ** 2)[0] * 0

        # Normalize to match q(y) normalization
        # q sums to sum_y sum_i x_i x_{i+y} / K = (sum x_i)^2 / K = N^2/K
        # but more practically, we normalize p to match q's expected total mass
        n_obs_with_self = len(self.diffs) + self.N  # observed + self distances
        if self.p.sum() > 0:
            self.p *= (n_obs_with_self / self.K) / self.p.sum()
            # Scale so total mass of p ≈ total mass of q for a good solution
            self.p *= self.N ** 2 / (self.K * self.p.sum()) if self.p.sum() > 0 else 1

    def _gradient(self, x, q):
        """
        Gradient of f(x) = sum_y (q(y) - p(y))^2  w.r.t. x.

        df/dx_m = (2/K) * sum_y (q(y) - p(y)) * d[q(y)]/dx_m

        where d[q(y)]/dx_m = (1/K) * (x[m-y] + x[m+y]) for valid indices.

        Efficiently: grad = (2/K) * correlation of residual with x.
        """
        residual = q - self.p  # shape (M,)
        grad = np.zeros(self.M)

        # For each y, the contribution to d[q(y)]/dx_m is:
        # x[m+y] (from x[i]*x[i+y] where i=m) and x[m-y] (where i+y=m)
        for y in range(self.M):
            r = residual[y]
            if abs(r) < 1e-15:
                continue
            # x_i * x_{i+y}: derivative w.r.t. x_m involves
            #   x_{m+y} (from i=m term) and x_{m-y} (from i=m-y term)
            for m in range(self.M):
                if m + y < self.M:
                    grad[m] += r * x[m + y]
                if m - y >= 0:
                    grad[m] += r * x[m - y]

        grad *= 2.0 / self.K
        return grad

    def _gradient_fast(self, x, q):
        """
        FFT-accelerated gradient computation.

        The gradient is: grad_m = (2/K) * sum_y r(y) * [x(m+y) + x(m-y)]
                       = (2/K) * [corr(r, x) + corr_flipped(r, x)]

        where r(y) = q(y) - p(y).
        """
        r = q - self.p
        M = self.M

        # Correlation of r with x: sum_y r[y] * x[m+y]
        # This is a cross-correlation, computed via FFT
        r_pad = np.zeros(2 * M)
        x_pad = np.zeros(2 * M)
        r_pad[:M] = r
        x_pad[:M] = x

        R = np.fft.rfft(r_pad)
        X = np.fft.rfft(x_pad)

        # corr1[m] = sum_y r[y] * x[m+y]  (x shifted left by m)
        corr1 = np.fft.irfft(np.conj(R) * X)[:M]

        # corr2[m] = sum_y r[y] * x[m-y]  (convolution)
        corr2 = np.fft.irfft(R * X)[:M]

        grad = (2.0 / self.K) * (corr1 + corr2)
        return grad

    def _project(self, x):
        """
        Project x onto the feasible set:
            0 <= x_m <= 1,  sum(x) = N.

        Uses iterative Dykstra-like projection:
        1. Clip to [0, 1]
        2. Rescale to sum = N
        Repeat until converged.
        """
        for _ in range(100):
            x = np.clip(x, 0.0, 1.0)
            s = x.sum()
            if s < 1e-10:
                # Degenerate: reinitialize
                x = np.ones(self.M) * self.N / self.M
            elif abs(s - self.N) > 1e-10:
                x *= self.N / s
            x = np.clip(x, 0.0, 1.0)
            if abs(x.sum() - self.N) < 1e-8:
                break
        return x

    def _spectral_initializer(self):
        """
        Spectral initialization inspired by Huang & Dokmanić (2021).

        Build a rough estimate by back-projecting the target distribution
        onto the point domain.  x_m ∝ sum_y p(y) * (row_sum of A_y at m).

        For a Toeplitz A_y, row m of A_y has a 1 at column m+y (if valid).
        So: x_m ∝ sum_y p(y)  for all y s.t. m+y < M  ≈  partial sum of p.

        This gives a rough density that we then project.
        """
        x = np.zeros(self.M)
        for y in range(self.M):
            for m in range(self.M - y):
                x[m] += self.p[y]
                x[m + y] += self.p[y]
        # Normalize and project
        if x.max() > 0:
            x = x / x.max()
        x = self._project(x)
        return x

    def _greedy_initializer(self):
        """
        Greedy initializer: place peaks at the N highest-density locations
        in a rough density estimated from the observed differences.
        """
        density = np.zeros(self.M)
        sig_bins = max(self.sigma / self.lam, 1.0)
        y_grid = np.arange(self.M)

        # Each diff suggests a level near 0+d or max-d
        for d in self.diffs:
            bin_d = d / self.lam
            density += np.exp(-0.5 * ((y_grid - bin_d) / sig_bins) ** 2)
            bin_from_top = (self.e_max - d) / self.lam
            if 0 <= bin_from_top < self.M:
                density += np.exp(-0.5 * ((y_grid - bin_from_top) / sig_bins) ** 2)

        # Always have a peak at 0 (we fix E_min = 0)
        density[0] += density.max() * 2
        # Peak at max
        max_bin = int(round(self.e_max / self.lam)) - 1
        if 0 <= max_bin < self.M:
            density[max_bin] += density.max()

        # Pick top N peaks
        x = np.zeros(self.M)
        for _ in range(self.N):
            idx = np.argmax(density)
            x[idx] = 1.0
            # Suppress nearby bins to avoid placing two levels too close
            suppress_range = max(int(self.M / (self.N * 3)), 2)
            lo = max(0, idx - suppress_range)
            hi = min(self.M, idx + suppress_range + 1)
            density[lo:hi] = 0

        return self._project(x)

    def solve(self, n_iters=2000, lr=None, n_restarts=5):
        """
        Run projected gradient descent with multiple random restarts.

        Parameters
        ----------
        n_iters : int
            Gradient descent iterations per restart.
        lr : float or None
            Learning rate. Auto-tuned if None.
        n_restarts : int
            Number of random restarts (best result kept).

        Returns
        -------
        dict with:
            'levels'    : np.ndarray of reconstructed energy levels
            'x'         : the full density vector (for inspection)
            'loss'      : final loss value
            'explained' : fraction of observed diffs explained
        """
        if lr is None:
            lr = 0.5 * self.K / (self.N ** 2 * self.M)

        best_loss = np.inf
        best_x = None

        initializers = []
        # Always include spectral and greedy initializers
        initializers.append(("spectral", self._spectral_initializer()))
        initializers.append(("greedy", self._greedy_initializer()))
        # Add random restarts
        for i in range(max(n_restarts - 2, 1)):
            x0 = np.random.dirichlet(np.ones(self.M)) * self.N
            initializers.append((f"random_{i}", self._project(x0)))

        for name, x in initializers:
            x = x.copy()
            loss_prev = np.inf
            lr_k = lr

            for it in range(n_iters):
                q = self._compute_q_fast(x)
                loss = np.sum((q - self.p) ** 2)

                # Adaptive step size
                if loss > loss_prev:
                    lr_k *= 0.5
                elif it > 0 and it % 200 == 0:
                    lr_k *= 1.2  # try accelerating

                grad = self._gradient_fast(x, q)
                x = x - lr_k * grad
                x = self._project(x)
                loss_prev = loss

            q = self._compute_q_fast(x)
            loss = np.sum((q - self.p) ** 2)

            if self.verbose:
                print(f"    Init '{name}': loss = {loss:.6e}")

            if loss < best_loss:
                best_loss = loss
                best_x = x.copy()

        # --- Extract N point locations from the density vector ---
        levels = self._extract_levels(best_x)

        # Score: how many observed diffs does this explain?
        explained = self._score(levels)

        if self.verbose:
            print(f"\n  Best loss: {best_loss:.6e}")
            print(f"  Extracted levels: {list(np.round(levels, 4))}")
            print(f"  Explains {explained}/{len(self.diffs)} observed "
                  f"differences ({100*explained/len(self.diffs):.0f}%)")

        return {
            'levels': levels,
            'x': best_x,
            'loss': best_loss,
            'explained': explained,
            'total_observed': len(self.diffs),
        }

    def _extract_levels(self, x):
        """
        Extract N energy level positions from the continuous density x.

        Strategy: find the N tallest well-separated peaks.
        """
        x_smooth = np.convolve(x, np.ones(3)/3, mode='same')
        levels = []
        remaining = x_smooth.copy()
        min_sep = max(int(self.M / (self.N * 4)), 1)

        for _ in range(self.N):
            idx = np.argmax(remaining)
            level = idx * self.lam

            # Refine: weighted average around the peak
            lo = max(0, idx - 2)
            hi = min(self.M, idx + 3)
            weights = x[lo:hi]
            if weights.sum() > 1e-10:
                bins = np.arange(lo, hi)
                level = np.average(bins, weights=weights) * self.lam

            levels.append(level)

            # Suppress neighborhood
            sup_lo = max(0, idx - min_sep)
            sup_hi = min(self.M, idx + min_sep + 1)
            remaining[sup_lo:sup_hi] = 0

        levels = np.sort(levels)
        # Shift so minimum = 0
        levels -= levels[0]
        return levels

    def _score(self, levels, slack=None):
        """Count how many observed diffs are explained by these levels."""
        if slack is None:
            slack = 2.0 * self.lam
        predicted = []
        for i in range(len(levels)):
            for j in range(i + 1, len(levels)):
                predicted.append(abs(levels[i] - levels[j]))
        predicted = np.array(sorted(predicted))

        explained = 0
        pred_avail = list(predicted)
        for d in self.diffs:
            best_idx = -1
            best_err = slack + 1
            for i, p in enumerate(pred_avail):
                err = abs(p - d)
                if err < best_err:
                    best_err = err
                    best_idx = i
            if best_idx >= 0 and best_err <= slack:
                explained += 1
                pred_avail.pop(best_idx)

        return explained


# =============================================================================
# Part C: Convenience wrappers
# =============================================================================

def solve_turnpike_ddm(observed_diffs, n_levels, e_max=None,
                       grid_oversample=4, n_iters=2000,
                       n_restarts=5, verbose=True):
    """
    Solve the (possibly incomplete) Turnpike problem using DDM.

    Parameters
    ----------
    observed_diffs : list[float]
        Observed pairwise differences (can be incomplete and noisy).
    n_levels : int
        Number of energy levels to find (2^N for N qubits).
    e_max : float or None
        Energy range upper bound. Auto-detected if None.
    grid_oversample : int
        Grid resolution factor.
    n_iters : int
        Gradient descent iterations.
    n_restarts : int
        Number of initializations to try.
    verbose : bool
        Print progress.

    Returns
    -------
    dict with 'levels', 'loss', 'explained', 'total_observed', 'x'.
    """
    solver = DDMSolver(n_levels, observed_diffs, e_max=e_max,
                       grid_oversample=grid_oversample, verbose=verbose)
    return solver.solve(n_iters=n_iters, n_restarts=n_restarts)


def solve_turnpike_combined(observed_diffs, n_levels, slack=0.01,
                            verbose=True):
    """
    Try exact solver first; fall back to DDM if it fails.
    Always returns at least one solution.
    """
    m = len(observed_diffs)
    m_expected = n_levels * (n_levels - 1) // 2

    if m == m_expected:
        if verbose:
            print("Complete difference set detected. Trying exact solver...")
        try:
            result = solve_turnpike_exact(observed_diffs, slack=slack)
            if verbose:
                print(f"  Exact solution: {list(np.round(result, 4))}")
            return {
                'levels': result,
                'loss': 0.0,
                'explained': m,
                'total_observed': m,
                'method': 'exact',
            }
        except ValueError:
            if verbose:
                print("  Exact solver failed. Falling back to DDM...")

    if verbose:
        print(f"Using DDM solver ({m}/{m_expected} differences observed)...")
    result = solve_turnpike_ddm(observed_diffs, n_levels, verbose=verbose)
    result['method'] = 'ddm'
    return result


# =============================================================================
# Part D: Utilities
# =============================================================================

def generate_differences(levels):
    """Compute all |E_i - E_j| for i < j."""
    diffs = []
    for i in range(len(levels)):
        for j in range(i + 1, len(levels)):
            diffs.append(abs(levels[i] - levels[j]))
    return sorted(diffs)


def subsample_differences(all_diffs, keep_fraction, keep_largest=True):
    """Randomly drop differences to simulate incomplete observations."""
    mask = np.random.random(len(all_diffs)) < keep_fraction
    if keep_largest:
        mask[-1] = True  # keep spectral width
    return [d for d, k in zip(all_diffs, mask) if k]


# =============================================================================
# Part E: Demo
# =============================================================================

def demo():
    print("=" * 70)
    print("  INCOMPLETE TURNPIKE SOLVER — LZS Spectroscopy")
    print("  Using Distance Distribution Matching (Huang & Dokmanić, 2021)")
    print("=" * 70)

    # --- Test 1: Complete data, exact solver ---
    print("\n" + "─" * 70)
    print("  TEST 1: Complete data — 2 qubits (4 levels, all 6 diffs)")
    print("─" * 70 + "\n")

    true_4 = [0.0, 1.5, 3.7, 6.0]
    diffs_4 = generate_differences(true_4)
    random.shuffle(diffs_4)

    r1 = solve_turnpike_combined(diffs_4, n_levels=4)
    print(f"\n  True:          {true_4}")
    print(f"  Reconstructed: {list(np.round(r1['levels'], 3))}")
    print(f"  Method: {r1['method']}")

    # --- Test 2: Complete data, 3 qubits ---
    print("\n" + "─" * 70)
    print("  TEST 2: Complete data — 3 qubits (8 levels, all 28 diffs)")
    print("─" * 70 + "\n")

    true_8 = [0.0, 0.8, 2.1, 3.5, 5.0, 6.3, 7.9, 10.0]
    diffs_8 = generate_differences(true_8)
    random.shuffle(diffs_8)

    r2 = solve_turnpike_combined(diffs_8, n_levels=8)
    print(f"\n  True:          {true_8}")
    print(f"  Reconstructed: {list(np.round(r2['levels'], 3))}")
    print(f"  Method: {r2['method']}")

    # --- Test 3: INCOMPLETE data, 3 qubits, ~60% observed ---
    print("\n" + "─" * 70)
    print("  TEST 3: Incomplete — 3 qubits, ~60% of 28 diffs observed")
    print("─" * 70 + "\n")

    np.random.seed(42)
    true_8 = [0.0, 0.8, 2.1, 3.5, 5.0, 6.3, 7.9, 10.0]
    all_diffs = generate_differences(true_8)
    partial = subsample_differences(all_diffs, keep_fraction=0.6)
    print(f"  True levels: {true_8}")
    print(f"  Observed: {len(partial)}/{len(all_diffs)} differences\n")

    r3 = solve_turnpike_ddm(partial, n_levels=8, n_restarts=8, n_iters=3000)
    print(f"\n  True:          {true_8}")
    print(f"  Reconstructed: {list(np.round(r3['levels'], 2))}")
    err = np.mean(np.abs(np.array(r3['levels']) - np.array(true_8)))
    print(f"  Mean abs error: {err:.3f} GHz")

    # --- Test 4: INCOMPLETE data, 4 qubits (16 levels), ~50% ---
    print("\n" + "─" * 70)
    print("  TEST 4: Incomplete — 4 qubits (16 levels), ~50% of 120 diffs")
    print("─" * 70 + "\n")

    np.random.seed(7)
    true_16 = np.sort(np.round(np.cumsum(np.random.uniform(0.3, 1.5, 16)), 3))
    true_16 = true_16 - true_16[0]
    all_16 = generate_differences(true_16)
    partial_16 = subsample_differences(all_16, keep_fraction=0.50)

    print(f"  True levels: {list(np.round(true_16, 2))}")
    print(f"  Observed: {len(partial_16)}/{len(all_16)} differences\n")

    r4 = solve_turnpike_ddm(partial_16, n_levels=16,
                            n_restarts=8, n_iters=4000, grid_oversample=3)
    print(f"\n  True:          {list(np.round(true_16, 2))}")
    print(f"  Reconstructed: {list(np.round(r4['levels'], 2))}")
    err4 = np.mean(np.abs(np.array(r4['levels']) - np.array(true_16)))
    print(f"  Mean abs error: {err4:.3f} GHz")

    # --- Test 5: Noisy + incomplete ---
    print("\n" + "─" * 70)
    print("  TEST 5: Noisy + incomplete — 3 qubits, σ=0.02, ~70% observed")
    print("─" * 70 + "\n")

    np.random.seed(99)
    true_n = [0.0, 0.8, 2.1, 3.5, 5.0, 6.3, 7.9, 10.0]
    all_n = generate_differences(true_n)
    noisy = [d + np.random.normal(0, 0.02) for d in all_n]
    partial_n = subsample_differences(noisy, keep_fraction=0.7)

    print(f"  True levels: {true_n}")
    print(f"  Noise σ=0.02, observed {len(partial_n)}/{len(noisy)} diffs\n")

    r5 = solve_turnpike_ddm(partial_n, n_levels=8, n_restarts=8, n_iters=3000)
    print(f"\n  True:          {true_n}")
    print(f"  Reconstructed: {list(np.round(r5['levels'], 2))}")
    err5 = np.mean(np.abs(np.array(r5['levels']) - np.array(true_n)))
    print(f"  Mean abs error: {err5:.3f} GHz")

    # --- Test 6: Severely incomplete — only 15/120 diffs for 4 qubits ---
    print("\n" + "─" * 70)
    print("  TEST 6: Severely incomplete — 4 qubits, only 15/120 diffs (~12%)")
    print("─" * 70 + "\n")

    np.random.seed(42)
    true_16b = np.sort(np.round(np.cumsum(np.random.uniform(0.5, 2.0, 16)), 3))
    true_16b = true_16b - true_16b[0]
    all_16b = generate_differences(true_16b)
    partial_16b = subsample_differences(all_16b, keep_fraction=0.125)
    # Ensure we have at least ~15 diffs
    while len(partial_16b) < 15:
        partial_16b = subsample_differences(all_16b, keep_fraction=0.15)

    print(f"  True levels: {list(np.round(true_16b, 2))}")
    print(f"  Observed: {len(partial_16b)}/{len(all_16b)} differences\n")

    r6 = solve_turnpike_ddm(partial_16b, n_levels=16,
                            n_restarts=10, n_iters=5000, grid_oversample=3)
    print(f"\n  True:          {list(np.round(true_16b, 2))}")
    print(f"  Reconstructed: {list(np.round(r6['levels'], 2))}")
    print(f"  (Severely incomplete — result is approximate)")

    print("\n" + "=" * 70)
    print("  All tests complete.")
    print("=" * 70)


if __name__ == "__main__":
    demo()
