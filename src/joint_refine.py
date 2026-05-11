"""
joint_refine.py
===============

Stage B of the LZS noisy-state reconstruction pipeline: joint nonlinear
least-squares refinement of physical parameters, warm-started from the
existing pipeline (peak picking + Turnpike + NNLS rate extraction +
population sector warm-start from Stage A).

Forward model
-------------
The time-domain signal ansatz is

    y(τ) = Σ_S  v_S * exp(-μ_S τ)                                 [population]
         + 2 Σ_{n>m}  A_mn * exp(-Γ_mn τ) * cos(ω_mn τ + δ_mn)    [coherence]

with all derived quantities expressed in terms of physical parameters:

    A_mn   = |u_m|² |u_n|²
    δ_mn   = 2 (φ_m - φ_n)
    ω_mn   = E_m - E_n
    Γ_mn   = Σ_q [ 2 κ_q^φ * d_q(m,n) + (1/2) κ_q^T1 * (e_q(m) + e_q(n)) ]
    μ_S    = Σ_{q ∈ S} κ_q^T1

where
    d_q(m,n) ∈ {0,1}  bit-disagreement indicator on qubit q
    e_q(m)   ∈ {0,1}  excitation indicator of qubit q in eigenstate m

Free parameters (count for M = 2^N_q eigenstates):
    |u_m|, m=0..M-1   with Σ|u_m|² = 1     -->  M-1 spherical angles
    φ_m,   m=1..M-1   (φ_0 ≡ 0)            -->  M-1
    E_m,   m=1..M-1   (E_0 ≡ 0)            -->  M-1
    κ_q^φ,   q=1..N_q          (≥ 0)       -->  N_q
    κ_q^T1,  q=1..N_q          (≥ 0)       -->  N_q
    v_S,     S ⊆ {1..N_q}      (≥ 0)       -->  2^N_q

The parameters are packed into a flat vector for scipy.optimize.least_squares.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

import jax
import jax.numpy as jnp
from jax import jit
from scipy.optimize import least_squares

jax.config.update("jax_enable_x64", True)


# =============================================================================
# Parameter packing / unpacking
# =============================================================================
#
# Spherical-coordinate parametrization for |u_m|:
#   |u_0| = cos(α_0)
#   |u_1| = sin(α_0) cos(α_1)
#   |u_2| = sin(α_0) sin(α_1) cos(α_2)
#   ...
#   |u_{M-1}| = sin(α_0) sin(α_1) ... sin(α_{M-2})
#
# This automatically enforces Σ|u_m|² = 1 with α_k ∈ [0, π/2] giving |u_m| ∈ [0,1].

@dataclass
class ParamLayout:
    """Index layout for the flat parameter vector."""
    M: int                          # number of eigenstates = 2^N_q
    Nq: int                         # number of qubits
    n_alpha: int = field(init=False)    # M - 1
    n_phi:   int = field(init=False)    # M - 1
    n_E:     int = field(init=False)    # M - 1
    n_kphi:  int = field(init=False)    # N_q
    n_kT1:   int = field(init=False)    # N_q
    n_v:     int = field(init=False)    # 2^N_q = M

    def __post_init__(self):
        self.n_alpha = self.M - 1
        self.n_phi   = self.M - 1
        self.n_E     = self.M - 1
        self.n_kphi  = self.Nq
        self.n_kT1   = self.Nq
        self.n_v     = self.M

    @property
    def total(self) -> int:
        return self.n_alpha + self.n_phi + self.n_E + self.n_kphi + self.n_kT1 + self.n_v

    def slices(self) -> dict[str, slice]:
        i = 0
        out = {}
        for name, n in [
            ("alpha", self.n_alpha),
            ("phi",   self.n_phi),
            ("E",     self.n_E),
            ("kphi",  self.n_kphi),
            ("kT1",   self.n_kT1),
            ("v",     self.n_v),
        ]:
            out[name] = slice(i, i + n)
            i += n
        return out


def pack(alpha, phi, E, kphi, kT1, v, layout: ParamLayout) -> np.ndarray:
    """Pack physical parameter blocks into a flat vector."""
    return np.concatenate([alpha, phi, E, kphi, kT1, v])


def unpack(theta: jnp.ndarray, layout: ParamLayout):
    """Unpack a flat parameter vector into named blocks (jax arrays)."""
    s = layout.slices()
    return (
        theta[s["alpha"]],
        theta[s["phi"]],
        theta[s["E"]],
        theta[s["kphi"]],
        theta[s["kT1"]],
        theta[s["v"]],
    )


# =============================================================================
# Derived quantities from physical parameters
# =============================================================================

def spherical_to_moduli(alpha: jnp.ndarray) -> jnp.ndarray:
    """
    Map (M-1) spherical angles to M moduli with Σ|u_m|² = 1.

    |u_0|   = cos(α_0)
    |u_k|   = (Π_{j<k} sin α_j) cos α_k        for 0 < k < M-1
    |u_M-1| = Π_{j<M-1} sin α_j

    Returns a length-M vector of non-negative entries summing in square to 1.
    """
    sin_a = jnp.sin(alpha)
    cos_a = jnp.cos(alpha)
    # cumulative product of sines, prepended with 1 so that index k gives Π_{j<k} sin α_j
    cumsin = jnp.concatenate([jnp.array([1.0]), jnp.cumprod(sin_a)])
    # u[k] = cumsin[k] * cos(α_k)  for k < M-1
    # u[M-1] = cumsin[M-1]
    u_head = cumsin[:-1] * cos_a
    u_tail = cumsin[-1:]
    return jnp.concatenate([u_head, u_tail])


def moduli_to_spherical(u_abs: np.ndarray) -> np.ndarray:
    """
    Inverse of spherical_to_moduli. Used to convert warm-start |ũ_m|
    into the (M-1) angle parametrization.

    Assumes |u_abs| has non-negative entries with Σ|u_m|² ≈ 1.
    """
    u = np.asarray(u_abs, dtype=float)
    u = u / np.linalg.norm(u)        # enforce normalization
    M = u.size
    alpha = np.zeros(M - 1)
    # remaining "tail mass" starts as 1; pull off one cos at a time
    remaining = 1.0
    for k in range(M - 1):
        # cos α_k = u[k] / sqrt(remaining)
        c = np.clip(u[k] / np.sqrt(max(remaining, 1e-30)), -1.0, 1.0)
        alpha[k] = np.arccos(c)
        # remaining *= sin² α_k
        remaining *= 1.0 - c * c
    return alpha


def build_disagreement(bitstrings: np.ndarray) -> np.ndarray:
    """
    d[q, m, n] = 1 if bitstrings[m, q] != bitstrings[n, q], else 0.

    bitstrings: (M, N_q) array of {0, 1}.
    Returns a (N_q, M, M) array.
    """
    bs = bitstrings.astype(np.int8)
    # (M, N_q) -> (1, M, N_q) - (M, 1, N_q) -> (M, M, N_q), then transpose
    diff = (bs[None, :, :] != bs[:, None, :]).astype(np.float64)  # (M, M, N_q)
    return np.transpose(diff, (2, 0, 1))                          # (N_q, M, M)


def build_excitation_sum(bitstrings: np.ndarray) -> np.ndarray:
    """
    e[q, m, n] = bitstrings[m, q] + bitstrings[n, q]  (in {0, 1, 2}).
    Returns (N_q, M, M).
    """
    bs = bitstrings.astype(np.float64)             # (M, N_q)
    e = bs[None, :, None, :] + bs[None, None, :, :]   # (1, M, M, N_q)
    e = e[0]                                       # (M, M, N_q)
    return np.transpose(e, (2, 0, 1))              # (N_q, M, M)


def build_subset_mask(Nq: int) -> np.ndarray:
    """
    For each subset index S (0 ≤ S < 2^N_q), return a row indicating which
    qubits are in S (in canonical bit order). Returns (2^N_q, N_q).
    """
    M = 1 << Nq
    rows = np.zeros((M, Nq), dtype=np.float64)
    for S in range(M):
        for q in range(Nq):
            if (S >> q) & 1:
                rows[S, q] = 1.0
    return rows


# =============================================================================
# Forward model
# =============================================================================

@dataclass
class ModelStatic:
    """Quantities that don't change during optimization."""
    tau: jnp.ndarray         # (T,) measurement times
    d_qmn: jnp.ndarray       # (N_q, M, M) disagreement indicator
    e_qmn: jnp.ndarray       # (N_q, M, M) excitation-sum indicator
    subset_mask: jnp.ndarray # (2^N_q, N_q) qubit-membership per subset
    pair_idx: tuple          # (rows, cols) for upper-triangular m < n


def make_model_static(tau: np.ndarray, bitstrings: np.ndarray) -> ModelStatic:
    M, Nq = bitstrings.shape
    d_qmn = build_disagreement(bitstrings)
    e_qmn = build_excitation_sum(bitstrings)
    subset_mask = build_subset_mask(Nq)
    rows, cols = np.triu_indices(M, k=1)
    return ModelStatic(
        tau=jnp.asarray(tau),
        d_qmn=jnp.asarray(d_qmn),
        e_qmn=jnp.asarray(e_qmn),
        subset_mask=jnp.asarray(subset_mask),
        pair_idx=(jnp.asarray(rows), jnp.asarray(cols)),
    )


def forward_model(theta: jnp.ndarray, layout: ParamLayout, static: ModelStatic) -> jnp.ndarray:
    """
    Evaluate y_model(τ; θ) on the measurement grid.

    Returns a (T,) jax array.
    """
    alpha, phi_free, E_free, kphi, kT1, v = unpack(theta, layout)

    # ---- Build full phase, energy, moduli vectors (with fixed φ_0 = 0, E_0 = 0)
    u_abs = spherical_to_moduli(alpha)                            # (M,)
    phi   = jnp.concatenate([jnp.array([0.0]), phi_free])         # (M,)
    E     = jnp.concatenate([jnp.array([0.0]), E_free])           # (M,)

    # ---- Derived pair quantities (only need m < n entries)
    rows, cols = static.pair_idx
    A_pair   = (u_abs[rows] ** 2) * (u_abs[cols] ** 2)              # (P,)
    delta    = 2.0 * (phi[rows] - phi[cols])                        # (P,)
    omega    = E[rows] - E[cols]                                    # (P,)
    # Γ_mn = Σ_q [2 κ_q^φ d_q + 0.5 κ_q^T1 e_q]   evaluated at (m,n)
    d_pair = static.d_qmn[:, rows, cols]                            # (N_q, P)
    e_pair = static.e_qmn[:, rows, cols]                            # (N_q, P)
    Gamma  = 2.0 * (kphi[:, None] * d_pair).sum(0) + 0.5 * (kT1[:, None] * e_pair).sum(0)

    # ---- Coherence (oscillatory) sector
    tau = static.tau                                                # (T,)
    # shape gymnastics: (T, 1) and (1, P) -> (T, P)
    arg = omega[None, :] * tau[:, None] + delta[None, :]            # (T, P)
    decay = jnp.exp(-Gamma[None, :] * tau[:, None])                 # (T, P)
    y_osc = 2.0 * (A_pair[None, :] * decay * jnp.cos(arg)).sum(-1)  # (T,)

    # ---- Population (zero-frequency) sector
    # μ_S = subset_mask @ kT1
    mu_S = static.subset_mask @ kT1                                 # (2^N_q,)
    y_pop = (v[None, :] * jnp.exp(-mu_S[None, :] * tau[:, None])).sum(-1)  # (T,)

    return y_pop + y_osc


# JIT-compiled residual function for speed
def make_residual_fn(layout: ParamLayout, static: ModelStatic, y_meas: jnp.ndarray,
                     reg: dict | None = None) -> tuple[Callable, Callable]:
    """
    Returns (residual_fn, jacobian_fn) suitable for scipy.optimize.least_squares.

    residual_fn(theta_np) -> 1D np.ndarray of residuals.
    jacobian_fn(theta_np) -> 2D np.ndarray Jacobian.

    Optional regularization `reg` is a dict that may contain:
        "anchor_kphi": (kphi_target, weight)
        "anchor_kT1":  (kT1_target,  weight)
        "l1_v":        weight        (sparse populations)
    Anchors add (weight * (kphi - target))^2 terms to the residual vector;
    L1 on v is implemented as Huber-smoothed L1 to keep things differentiable.
    """
    reg = reg or {}

    @jit
    def _residual(theta):
        r_data = forward_model(theta, layout, static) - y_meas
        extra = []
        if "anchor_kphi" in reg:
            target, w = reg["anchor_kphi"]
            extra.append(jnp.sqrt(w) * (theta[layout.slices()["kphi"]] - jnp.asarray(target)))
        if "anchor_kT1" in reg:
            target, w = reg["anchor_kT1"]
            extra.append(jnp.sqrt(w) * (theta[layout.slices()["kT1"]] - jnp.asarray(target)))
        if "l1_v" in reg:
            w = reg["l1_v"]
            v = theta[layout.slices()["v"]]
            # Huber-smoothed |v|: ≈ |v| for large v, quadratic near zero
            eps = 1e-6
            extra.append(jnp.sqrt(w) * jnp.sqrt(v * v + eps * eps))
        if extra:
            return jnp.concatenate([r_data] + extra)
        return r_data

    _jac = jit(jax.jacfwd(_residual))

    def residual_np(theta_np):
        return np.asarray(_residual(jnp.asarray(theta_np)))

    def jacobian_np(theta_np):
        return np.asarray(_jac(jnp.asarray(theta_np)))

    return residual_np, jacobian_np


# =============================================================================
# Warm start construction
# =============================================================================

@dataclass
class WarmStart:
    """All inputs needed to seed the joint refinement."""
    u_abs:    np.ndarray   # (M,)         from existing pipeline step 5
    phi:      np.ndarray   # (M,)         from existing pipeline step 5, with φ_0 = 0
    E:        np.ndarray   # (M,)         from Turnpike (one of two inversion candidates), with E_0 = 0
    kphi:     np.ndarray   # (N_q,)       from extract_rates.py
    kT1:      np.ndarray   # (N_q,)       from extract_rates.py
    v:        np.ndarray   # (2^N_q,)     from Stage A (population_warmstart.py)
    bitstrings: np.ndarray # (M, N_q)     from Turnpike + assignment


def warm_start_to_theta(ws: WarmStart, layout: ParamLayout) -> np.ndarray:
    """Convert a WarmStart into a flat parameter vector."""
    alpha = moduli_to_spherical(ws.u_abs)
    phi_free = ws.phi[1:].copy()
    E_free   = ws.E[1:].copy()
    return pack(alpha, phi_free, E_free, ws.kphi, ws.kT1, ws.v, layout)


def build_bounds(layout: ParamLayout) -> tuple[np.ndarray, np.ndarray]:
    """Box bounds for scipy.optimize.least_squares (TRF method)."""
    lo = np.full(layout.total, -np.inf)
    hi = np.full(layout.total,  np.inf)

    s = layout.slices()
    # spherical angles in [0, π/2]
    lo[s["alpha"]] = 0.0
    hi[s["alpha"]] = np.pi / 2.0
    # phases in [-π, π) — note refinement cannot escape mod-π sign ambiguity,
    # but a generous range lets the optimizer move freely
    lo[s["phi"]] = -np.pi
    hi[s["phi"]] =  np.pi
    # rates non-negative
    lo[s["kphi"]] = 0.0
    lo[s["kT1"]]  = 0.0
    lo[s["v"]]    = 0.0
    # energies unbounded (free shifts already removed by fixing E_0 = 0)
    return lo, hi


# =============================================================================
# Top-level entry point
# =============================================================================

@dataclass
class RefineResult:
    theta:        np.ndarray
    layout:       ParamLayout
    u_abs:        np.ndarray
    phi:          np.ndarray
    E:            np.ndarray
    kphi:         np.ndarray
    kT1:          np.ndarray
    v:            np.ndarray
    cost:         float            # 0.5 * ||r||² at solution
    cost_data:    float            # data-fit part only
    cost_reg:    float             # regularizer part
    n_fev:        int
    n_jev:        int
    success:      bool
    message:      str


def _split_cost(residual_fn, theta, n_data: int) -> tuple[float, float]:
    r = residual_fn(theta)
    rd = r[:n_data]
    rr = r[n_data:]
    return 0.5 * float(rd @ rd), 0.5 * float(rr @ rr)


def refine(
    tau:        np.ndarray,
    y_meas:     np.ndarray,
    warm:       WarmStart,
    reg:        dict | None = None,
    max_nfev:   int  = 500,
    verbose:    int  = 1,
    xtol:       float = 1e-10,
    ftol:       float = 1e-10,
    gtol:       float = 1e-10,
) -> RefineResult:
    """
    Run the joint nonlinear LS refinement.

    Parameters
    ----------
    tau, y_meas : the measurement grid and signal.
    warm        : WarmStart with all warm-start values, including bitstring
                  labels (these define the structural rate formula).
    reg         : optional dict of regularizers (see make_residual_fn).
    max_nfev    : forwarded to scipy.optimize.least_squares.
    verbose     : 0, 1, or 2 — forwarded.

    Returns RefineResult with the refined physical parameters.
    """
    M, Nq = warm.bitstrings.shape
    assert M == 1 << Nq, "warm.bitstrings must have 2^N_q rows"
    layout = ParamLayout(M=M, Nq=Nq)
    static = make_model_static(tau, warm.bitstrings)

    theta0 = warm_start_to_theta(warm, layout)
    lo, hi = build_bounds(layout)

    y_meas_j = jnp.asarray(y_meas)
    residual_np, jacobian_np = make_residual_fn(layout, static, y_meas_j, reg=reg)

    n_data = len(y_meas)

    result = least_squares(
        fun=residual_np,
        x0=theta0,
        jac=jacobian_np,
        bounds=(lo, hi),
        method="trf",
        xtol=xtol,
        ftol=ftol,
        gtol=gtol,
        max_nfev=max_nfev,
        verbose=verbose,
    )

    # Unpack refined parameters back into physical quantities
    theta_star = result.x
    alpha, phi_free, E_free, kphi, kT1, v = unpack(jnp.asarray(theta_star), layout)
    u_abs = np.asarray(spherical_to_moduli(alpha))
    phi   = np.concatenate([[0.0], np.asarray(phi_free)])
    E     = np.concatenate([[0.0], np.asarray(E_free)])

    cost_data, cost_reg = _split_cost(residual_np, theta_star, n_data)

    return RefineResult(
        theta=theta_star,
        layout=layout,
        u_abs=u_abs,
        phi=phi,
        E=E,
        kphi=np.asarray(kphi),
        kT1=np.asarray(kT1),
        v=np.asarray(v),
        cost=float(result.cost),
        cost_data=cost_data,
        cost_reg=cost_reg,
        n_fev=int(result.nfev),
        n_jev=int(result.njev) if result.njev is not None else -1,
        success=bool(result.success),
        message=str(result.message),
    )


# =============================================================================
# Smoke test
# =============================================================================

if __name__ == "__main__":
    # Tiny N_q = 2 sanity test: build synthetic signal from known params,
    # warm-start with a perturbation, confirm refinement recovers the truth.
    rng = np.random.default_rng(0)
    Nq = 2
    M = 1 << Nq
    bitstrings = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=np.int8)

    # Ground truth
    u_true   = np.array([0.6, 0.5, 0.4, np.sqrt(1 - 0.6**2 - 0.5**2 - 0.4**2)])
    phi_true = np.array([0.0, 0.3, -0.7, 1.2])
    E_true   = np.array([0.0, 0.05, 0.12, 0.18])           # in 2π·GHz·ns units
    kphi_true = np.array([0.002, 0.003])                    # ns^-1
    kT1_true  = np.array([0.001, 0.0005])                   # ns^-1
    v_true    = np.array([0.05, 0.02, 0.0, 0.01])           # populations
    tau = np.linspace(0, 1500, 4000)                        # ns

    layout = ParamLayout(M=M, Nq=Nq)
    static = make_model_static(tau, bitstrings)
    theta_true = pack(
        moduli_to_spherical(u_true),
        phi_true[1:], E_true[1:], kphi_true, kT1_true, v_true, layout
    )
    y_true = np.asarray(forward_model(jnp.asarray(theta_true), layout, static))

    # Perturb to make a warm start
    warm = WarmStart(
        u_abs=u_true + 0.03 * rng.standard_normal(M),
        phi=phi_true + 0.1 * rng.standard_normal(M),
        E=E_true + 0.005 * rng.standard_normal(M),
        kphi=kphi_true * (1 + 0.3 * rng.standard_normal(Nq)),
        kT1=kT1_true * (1 + 0.3 * rng.standard_normal(Nq)),
        v=np.clip(v_true + 0.01 * rng.standard_normal(M), 0, None),
        bitstrings=bitstrings,
    )
    warm.phi[0] = 0.0
    warm.E[0]   = 0.0
    warm.u_abs  = np.abs(warm.u_abs)
    warm.u_abs /= np.linalg.norm(warm.u_abs)

    res = refine(tau, y_true, warm, verbose=2)
    print(f"\nFinal data cost: {res.cost_data:.3e}")
    print(f"|u| recovery err: {np.linalg.norm(res.u_abs - u_true):.3e}")
    print(f"E recovery err:   {np.linalg.norm(res.E - E_true):.3e}")
    print(f"κ_φ recovery err: {np.linalg.norm(res.kphi - kphi_true):.3e}")
    print(f"κ_T1 recovery err:{np.linalg.norm(res.kT1 - kT1_true):.3e}")
