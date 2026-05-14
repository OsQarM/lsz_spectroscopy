"""
End-to-end refinement orchestrator. Given the outputs of
`run_full_diagnostics` and the measured signal, it:

  1. Extracts per-qubit kappa_q^phi, kappa_q^T1 via NNLS on the observed
     decay rates (extract_rates.nnls_extract_kappas).
  2. Builds the warm-start population amplitudes {v_S} (Stage A).
  3. Runs the joint nonlinear LS refinement (Stage B, joint_refine.refine).

Returns a dict with refined physical parameters plus diagnostics.
"""

import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations

from extract_rates import nnls_extract_kappas, bitstrings_from_egvecs
from population_warmstart import estimate_populations, mu_S_from_kT1
from joint_refine import WarmStart, refine


def run_refinement(diag_results, tau, y_meas, reg=None,
                   allow_negative_v=False, max_nfev=500, verbose=1,
                   freeze_E=True,
                   anchor_kappa_weight=0.0,
                   loss="linear", f_scale=1.0,
                   x_scale="auto",
                   keep_warm_if_worse=True,
                   population_mode="dc"):
    """
    Stabilizers:
      anchor_kappa_weight : if > 0, add L2 anchor on (kphi, kT1) toward the
                            NNLS warm-start with this weight. Prevents the
                            joint fit from drifting too far from the rate
                            estimate that the structural formula gives.
      loss        : passed to least_squares ("linear", "soft_l1", "huber",
                    "cauchy"). Robust losses dampen the influence of large
                    residuals at high noise.
      f_scale     : robust-loss scale (residual magnitude treated as 'normal').
      x_scale     : "auto" uses per-block characteristic magnitudes derived
                    from the warm-start (recommended at high noise).
      keep_warm_if_worse : if True, revert to warm-start when the LS makes
                           the data fit worse.
    """
    """
    Parameters
    ----------
    diag_results : dict returned by `run_full_diagnostics`. Must contain
                   'u_m', 'phi_m', 'experimental_energies', 'a_mn',
                   'lmd_mn', 'w_mn', 'd_mn', 'egvecs_target'.
    tau          : (T,) measurement times (same as tw_l).
    y_meas       : (T,) measured signal (same as pc_list).
    reg          : optional regularizer dict (see joint_refine.make_residual_fn).
    allow_negative_v : passed to Stage A.
    max_nfev, verbose : forwarded to least_squares.

    Returns
    -------
    dict with refined parameters and warm-start values for comparison.
    """
    egvecs_t = diag_results['egvecs_target']
    bitstrings = bitstrings_from_egvecs(egvecs_t)
    M, Nq = bitstrings.shape

    lmd_mn = diag_results['lmd_mn']
    a_mn = diag_results['a_mn']
    w_mn = diag_results['w_mn']
    d_mn = diag_results['d_mn']

    kphi_init, kT1_init, _, kappa_residual = nnls_extract_kappas(lmd_mn, bitstrings)

    v_init, mu_S, pop_residual = estimate_populations(
        np.asarray(y_meas), np.asarray(tau), kT1_init,
        a_mn, lmd_mn, w_mn, d_mn,
        allow_negative=allow_negative_v,
    )

    # E_m: anchor to E_0 = 0
    E_warm = np.asarray(diag_results['experimental_energies'], dtype=float)
    E_warm = E_warm - E_warm[0]

    u_warm = np.asarray(diag_results['u_m'], dtype=float)
    u_warm = np.nan_to_num(np.abs(u_warm), nan=0.0)
    if np.linalg.norm(u_warm) > 0:
        u_warm = u_warm / np.linalg.norm(u_warm)

    phi_warm = np.nan_to_num(np.asarray(diag_results['phi_m'], dtype=float), nan=0.0)
    phi_warm = phi_warm - phi_warm[0]

    # Warm-start DC: use the global-fit DC if available, else the mean of y_meas
    # at late times (where oscillations have decayed).
    dc_warm = float(diag_results.get('dc_fit', np.mean(np.asarray(y_meas))))

    warm = WarmStart(
        u_abs=u_warm,
        phi=phi_warm,
        E=E_warm,
        kphi=kphi_init,
        kT1=kT1_init,
        v=v_init,
        bitstrings=bitstrings,
        dc=dc_warm,
    )

    # Default anchors prevent (kphi, kT1) from drifting far from the NNLS
    # warm-start when the data is noisy. Disabled by setting weight=0.
    reg = dict(reg) if reg else {}
    if anchor_kappa_weight > 0:
        reg.setdefault("anchor_kphi", (kphi_init.copy(), anchor_kappa_weight))
        reg.setdefault("anchor_kT1",  (kT1_init.copy(),  anchor_kappa_weight))

    result = refine(np.asarray(tau), np.asarray(y_meas), warm,
                    reg=reg, max_nfev=max_nfev, verbose=verbose,
                    freeze=frozenset({"E"}) if freeze_E else frozenset(),
                    x_scale=x_scale, loss=loss, f_scale=f_scale,
                    keep_warm_if_worse=keep_warm_if_worse,
                    population_mode=population_mode)

    return {
        'warm': {
            'u_abs': u_warm,
            'phi':   phi_warm,
            'E':     E_warm,
            'kphi':  kphi_init,
            'kT1':   kT1_init,
            'v':     v_init,
            'mu_S':  mu_S,
            'kappa_residual': kappa_residual,
            'pop_residual':   pop_residual,
        },
        'refined': {
            'u_abs': result.u_abs,
            'phi':   result.phi,
            'E':     result.E,
            'kphi':  result.kphi,
            'kT1':   result.kT1,
            'v':     result.v,
            'dc':    result.dc,
            'population_mode': population_mode,
        },
        'bitstrings': bitstrings,
        'cost':       result.cost,
        'cost_data':  result.cost_data,
        'cost_reg':   result.cost_reg,
        'n_fev':      result.n_fev,
        'success':    result.success,
        'message':    result.message,
        'raw_result': result,
    }


def _fmt(arr, prec=4):
    return np.array2string(np.asarray(arr), precision=prec, suppress_small=True)


def print_refinement_summary(refine_results, truth=None):
    """
    Print a side-by-side summary of warm-start vs refined parameters.
    If `truth` is provided (dict with keys 'u_abs', 'phi', 'E', 'kphi',
    'kT1'), also show absolute errors.

    Parameters
    ----------
    refine_results : output of `run_refinement`.
    truth          : optional ground-truth dict for comparison.
    """
    warm = refine_results['warm']
    ref = refine_results['refined']
    bits = refine_results['bitstrings']

    print("=" * 72)
    print("Joint refinement summary")
    print("=" * 72)
    print(f"converged: {refine_results['success']}    "
          f"n_fev = {refine_results['n_fev']}    "
          f"cost (data) = {refine_results['cost_data']:.4e}    "
          f"cost (reg) = {refine_results['cost_reg']:.4e}")
    print(f"message: {refine_results['message']}")

    M, Nq = bits.shape
    labels = ["".join(str(b) for b in bits[m]) for m in range(M)]

    def _row(name, warm_v, ref_v, true_v=None):
        print(f"  {name:>10s}   warm: {_fmt(warm_v)}")
        print(f"  {'':>10s}   refn: {_fmt(ref_v)}")
        if true_v is not None:
            print(f"  {'':>10s}   true: {_fmt(true_v)}")
            print(f"  {'':>10s}   |refn-true|: {_fmt(np.abs(np.asarray(ref_v) - np.asarray(true_v)))}")

    print("\n-- Eigenstate labels (bitstrings) --")
    print(f"  {labels}")

    print("\n-- State-vector moduli |u_m| --")
    _row("|u_m|", warm['u_abs'], ref['u_abs'],
         truth.get('u_abs') if truth else None)

    print("\n-- Phases phi_m --")
    _row("phi_m", warm['phi'], ref['phi'],
         truth.get('phi') if truth else None)

    print("\n-- Energies E_m (relative to E_0 = 0) --")
    _row("E_m", warm['E'], ref['E'],
         truth.get('E') if truth else None)

    print("\n-- Per-qubit dephasing kappa_q^phi (= 1/T2_phi) --")
    _row("kphi_q", warm['kphi'], ref['kphi'],
         truth.get('kphi') if truth else None)

    print("\n-- Per-qubit T1 rates kappa_q^T1 (= 1/T1) --")
    _row("kT1_q", warm['kT1'], ref['kT1'],
         truth.get('kT1') if truth else None)

    print("\n-- Population amplitudes v_S (one per subset, ordered by mask) --")
    _row("v_S", warm['v'], ref['v'],
         truth.get('v') if truth else None)

    print("\n-- Stage diagnostics --")
    print(f"  NNLS rate-extraction residual : {warm['kappa_residual']:.4e}")
    print(f"  Stage-A population residual   : {warm['pop_residual']:.4e}")
    print("=" * 72)


def evaluate_refined_signal(refined, bitstrings, tau):
    """
    Evaluate y_model(tau) using the refined physical parameters (numpy-only).
    Also returns derived quantities (a_mn, w_mn, d_mn, Gamma_mn) for plots.
    """
    u_abs = np.asarray(refined['u_abs'])
    phi = np.asarray(refined['phi'])
    E = np.asarray(refined['E'])
    kphi = np.asarray(refined['kphi'])
    kT1 = np.asarray(refined['kT1'])
    v = np.asarray(refined['v'])
    dc = float(refined.get('dc', 0.0))
    pop_mode = refined.get('population_mode', 'subsets')

    M, Nq = bitstrings.shape

    a_mn = np.zeros((M, M))
    d_mn = np.zeros((M, M))
    w_mn = np.zeros((M, M))
    G_mn = np.zeros((M, M))

    y_osc = np.zeros_like(tau, dtype=float)
    for m, n in combinations(range(M), 2):
        A = (u_abs[m] ** 2) * (u_abs[n] ** 2)
        delta = 2.0 * (phi[m] - phi[n])
        omega = E[m] - E[n]
        g_phi = sum(kphi[q] for q in range(Nq) if bitstrings[m, q] != bitstrings[n, q])
        g_t1 = 0.5 * sum(kT1[q] * (bitstrings[m, q] + bitstrings[n, q]) for q in range(Nq))
        G = g_phi + g_t1

        a_mn[m, n] = A
        d_mn[m, n] = delta
        w_mn[m, n] = omega
        G_mn[m, n] = G

        y_osc += 2.0 * A * np.exp(-G * tau) * np.cos(omega * tau + delta)

    if pop_mode == "subsets" and v.size > 0:
        mu_S, _ = mu_S_from_kT1(kT1)
        y_pop = (v[None, :] * np.exp(-mu_S[None, :] * tau[:, None])).sum(-1)
    elif pop_mode == "dc":
        mu_S = np.zeros(1)
        y_pop = dc * np.ones_like(tau)
    else:
        mu_S = np.zeros(0)
        y_pop = np.zeros_like(tau)

    return {
        'y_model': y_pop + y_osc,
        'y_osc':   y_osc,
        'y_pop':   y_pop,
        'a_mn':    a_mn,
        'd_mn':    d_mn,
        'w_mn':    w_mn,
        'G_mn':    G_mn,
        'mu_S':    mu_S,
    }


def _sorted_scatter(ax, est, true, xlabel, ylabel, title=None):
    est = np.asarray(est, dtype=float)
    true = np.asarray(true, dtype=float)
    ax.scatter(np.sort(est), np.sort(true), color='steelblue', zorder=3)
    lo = min(np.min(est), np.min(true))
    hi = max(np.max(est), np.max(true))
    diag = np.linspace(lo, hi, 100)
    ax.plot(diag, diag, "--", c="red", label="true = estimate")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.legend(frameon=False)


def plot_refined_diagnostics(refine_results, results, tau, y_meas,
                              exact_lambdas=None):
    """
    Re-emit the run_full_diagnostics plots using the refined parameters.

    Plots:
      1. Time-domain signal: y_meas vs y_model(refined), and residual.
      2. Frequencies (w_mn = E_m - E_n) refined vs true differences.
      3. Energy levels E_m refined vs true (sorted).
      4. |u_m| refined vs true (sorted).
      5. phi_m refined vs true (sorted).
      6. Coherence decay rates Gamma_mn refined vs exact (if provided).
    """
    refined = refine_results['refined']
    bitstrings = refine_results['bitstrings']
    tau = np.asarray(tau)
    y_meas = np.asarray(y_meas)

    sig = evaluate_refined_signal(refined, bitstrings, tau)
    y_model = sig['y_model']
    residual = y_meas - y_model
    rmse = float(np.sqrt(np.mean(residual ** 2)))

    # 1. Time-domain fit
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True,
                              gridspec_kw={'height_ratios': [3, 1]})
    axes[0].plot(tau, y_meas, lw=1.0, label='measured', color='steelblue')
    axes[0].plot(tau, y_model, lw=1.0, label='refined model', color='crimson',
                  alpha=0.85)
    axes[0].set_ylabel('y(tau)')
    axes[0].set_title(f'Time-domain fit (RMSE = {rmse:.3e})')
    axes[0].legend(frameon=False)
    axes[1].plot(tau, residual, lw=0.8, color='gray')
    axes[1].axhline(0, color='k', lw=0.5)
    axes[1].set_ylabel('residual')
    axes[1].set_xlabel('tau')
    plt.tight_layout(); plt.show()

    # 2. Frequencies (w_mn)
    M = bitstrings.shape[0]
    pairs = list(combinations(range(M), 2))
    w_ref = np.array([sig['w_mn'][m, n] for m, n in pairs])
    true_diffs = np.asarray(results['true_differences'])
    fig, ax = plt.subplots(figsize=(6, 5))
    _sorted_scatter(ax, np.abs(w_ref), np.abs(true_diffs),
                    "refined |w_mn| = |E_m - E_n|",
                    "true |delta E_mn|", title="Energy differences (refined)")
    plt.tight_layout(); plt.show()

    # 3. Energies E_m (sorted; refinement fixes E_0=0, so apply same shift to truth)
    E_ref = np.asarray(refined['E'])
    E_true = np.asarray(results['shifted_true_energies'])
    fig, ax = plt.subplots(figsize=(6, 5))
    _sorted_scatter(ax, E_ref, E_true,
                    "refined E_m", "true E_m",
                    title="Energy levels (refined)")
    plt.tight_layout(); plt.show()

    # 4. |u_m|
    u_ref = np.asarray(refined['u_abs'])
    u_true = np.asarray(results['u_exact_amplitudes'])
    fig, ax = plt.subplots(figsize=(6, 5))
    _sorted_scatter(ax, u_ref, u_true,
                    "refined |u_m|", "true |u_m|", title="State moduli (refined)")
    plt.tight_layout(); plt.show()

    # 5. phi_m
    phi_ref = np.mod(np.asarray(refined['phi']), np.pi)
    phi_true = np.asarray(results['u_exact_phases_rel'])
    fig, ax = plt.subplots(figsize=(6, 5))
    _sorted_scatter(ax, phi_ref, phi_true,
                    "refined phi_m (mod pi)", "true phi_m (mod pi)",
                    title="State phases (refined)")
    plt.tight_layout(); plt.show()

    # 6. Decay rates Gamma_mn (refined) vs exact predicted
    G_ref = np.array([sig['G_mn'][m, n] for m, n in pairs])
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(range(len(G_ref)), G_ref, color='steelblue', zorder=3,
                label='refined Gamma_mn')
    if exact_lambdas is not None:
        for i, rate in enumerate(np.asarray(exact_lambdas)):
            ax.axhline(rate, color='red', linestyle='--', lw=1.2,
                       label='predicted (exact)' if i == 0 else None)
    ax.set_xlabel('pair index')
    ax.set_ylabel('rate')
    ax.set_title('Coherence decay rates (refined)')
    ax.legend(frameon=False)
    plt.tight_layout(); plt.show()

    return {'rmse': rmse, 'y_model': y_model, 'derived': sig}
