"""
Pre-simulation work shared by scripts 1 and 3.

Everything here runs *before* the expensive sweep: the parameter report, the
schedule / spectrum / population preview figures, the sampling-adequacy check,
and the runtime estimate obtained by timing a few representative wait times.
"""
from __future__ import annotations

import time

import numpy as np

from lsz_experiment import LSZ_experiment
from spectrum_matching import check_sampling_adequacy
from plotting.diagnostic_plots import (
    plot_eigenbasis_populations,
    plot_eigenvector_weights,
    plot_instantaneous_spectrum,
    plot_schedule,
)

from . import io_utils as io


def build_experiment(common: dict, wait_time: float, *, with_noise: bool = False,
                     up_z=None, down_z=None, up_x=None, down_x=None,
                     noise_seed=None) -> LSZ_experiment:
    """Construct one `LSZ_experiment` from a resolved `common` block.

    The per-term asymmetry lists default to the ones in `common`, so script 3
    can override just the Z (or X) lists for its sweep.
    """
    return LSZ_experiment(
        common["n_qubits"], common["epsilon"], common["H_target_dict"],
        common["ramp_time"], float(wait_time), common["dt"],
        up_z_assymetry_factors=common["up_z"] if up_z is None else up_z,
        down_z_assymetry_factors=common["down_z"] if down_z is None else down_z,
        up_x_assymetry_factors=common["up_x"] if up_x is None else up_x,
        down_x_assymetry_factors=common["down_x"] if down_x is None else down_x,
        ramp_noise=common["r_noise"] if with_noise else False,
        wait_noise=common["w_noise"] if with_noise else False,
        gamma_dec_list=common["t1_rates"] if with_noise else None,
        gamma_dep_list=common["t2_rates"] if with_noise else None,
        initial_state=common["psi0"],
        ramp_error=common["ramp_error"],
        wait_error=common["wait_error"],
        ramp_bias=common["ramp_bias"],
        noise_seed=noise_seed,
    )


# --------------------------------------------------------------------------
# Parameter report
# --------------------------------------------------------------------------
def print_parameter_report(common: dict, provenance: dict | None = None) -> None:
    """Print the full physics / protocol configuration."""
    n = common["n_qubits"]
    prov = provenance or {}

    def tag(section):
        src = prov.get(section)
        return f"   [from {src}]" if src else ""

    io.section("System")
    io.kv("Number of qubits", n)
    io.kv("Hilbert space dimension", 2 ** n)
    io.kv("Transition pairs (expected peaks)", 2 ** n * (2 ** n - 1) // 2)

    io.section(f"Hamiltonian{tag('hamiltonian')}")
    io.kv("Local Z weights", io.fmt_array(common["local_z"]))
    io.kv("Local X weights", io.fmt_array(common["local_x"]))
    io.kv("Two-body ZZ couplings", io.fmt_array(common["two_body"]))
    io.kv("Driving epsilon", common["epsilon"])

    io.section(f"Protocol{tag('protocol')}")
    io.kv("Ramp time", common["ramp_time"])
    io.kv("Max wait time", common["max_wait_time"])
    io.kv("Number of wait times (n_sims)", common["n_sims"])
    io.kv("Wait-time step", f"{common['max_wait_time'] / common['n_sims']:.6g}")
    io.kv("Integration dt", common["dt"])
    io.kv("Preview resolution (n_steps)", common["n_steps"])
    total_sim_time = float(np.sum(2 * common["ramp_time"] + common["tw_l"]))
    io.kv("Total simulated time", f"{total_sim_time:.6g}")

    io.section(f"Initial state{tag('initial_state')}")
    io.kv("Specification", common["initial_state_spec"])
    amps = np.asarray(common["psi0"].full()).flatten()
    io.kv("Amplitudes", io.fmt_array(amps))

    io.section(f"Ramp asymmetry{tag('ramps')}")
    io.kv("Up   Z coefficients", io.fmt_array(common["up_z"]))
    io.kv("Down Z coefficients", io.fmt_array(common["down_z"]))
    io.kv("Up   X coefficients", io.fmt_array(common["up_x"]))
    io.kv("Down X coefficients", io.fmt_array(common["down_x"]))
    io.kv("Average ramp error", common["ramp_error"])
    io.kv("Average wait error", common["wait_error"])
    io.kv("Average ramp bias", common["ramp_bias"])

    io.section(f"Decoherence{tag('decoherence')}")
    io.kv("Ramp noise enabled", common["r_noise"])
    io.kv("Wait noise enabled", common["w_noise"])
    io.kv("T1 rates (kappa_T1)", io.fmt_array(common["t1_rates"])
          if common["t1_rates"] is not None else "none")
    io.kv("T2 rates (kappa_phi)", io.fmt_array(common["t2_rates"])
          if common["t2_rates"] is not None else "none")
    solver = ("mesolve (density matrix)"
              if (common["r_noise"] or common["w_noise"]) else "sesolve (state vector)")
    io.kv("Solver", solver)

    io.section(f"Run{tag('run')}")
    io.kv("Noise seed", common["noise_seed"])
    io.kv("Palette", common["palette"])
    io.kv("Figure formats", ", ".join(common["figure_formats"]))


# --------------------------------------------------------------------------
# Preview figures + spectrum
# --------------------------------------------------------------------------
def run_schedule_and_spectrum(common: dict, saver, *, prefix: str = "presim"):
    """Schedule, ramp areas, instantaneous spectrum and midpoint eigensystem.

    Returns a dict carrying `egvals_midpoint` (needed by the sampling check
    and by the diagnostics) and the ramp-area breakdown.
    """
    preview = build_experiment(common, 0.0)

    io.section("Ramp schedule")
    t_list, common_sched, hz, hx, ramp_areas = preview.show_schedule(common["n_steps"])
    fig, _ = plot_schedule(t_list, common_sched, hz, hx, show=False)
    saver.save(fig, f"{prefix}_schedule")

    if ramp_areas is not None:
        print("  Signed ramp-difference areas (qubit 0 - qubit 1;")
        print("  positive => qubit 0 ramps faster):")
        print(f"    {'term':<6} {'ramp1':>12} {'wait':>12} {'ramp2':>12} {'total':>12}")
        for term in ("z", "x"):
            a = ramp_areas[term]
            print(f"    {term:<6} {a['ramp1']:>+12.6f} {a['wait']:>+12.6f} "
                  f"{a['ramp2']:>+12.6f} {a['total']:>+12.6f}")
    else:
        print("  (ramp areas are only defined for >= 2 qubits)")

    io.section("Instantaneous spectrum")
    t_list, egvals, egvecs = preview.show_spectrum(common["n_steps"])
    midpoint = len(t_list) // 2
    fig, _ = plot_instantaneous_spectrum(
        t_list[:midpoint], np.asarray(egvals)[:midpoint], show=False)
    saver.save(fig, f"{prefix}_instantaneous_spectrum")

    egvals_midpoint = np.asarray(egvals[midpoint])
    io.kv("Midpoint eigenvalues", io.fmt_array(egvals_midpoint))

    vecs_mid = np.asarray(egvecs[midpoint])
    print("\n  Midpoint eigenvectors (columns):")
    for k, vec in enumerate(vecs_mid.T):
        print(f"    v[{k}]: {io.fmt_array(vec)}")

    fig, _ = plot_eigenvector_weights(vecs_mid, egvals_midpoint,
                                      common["n_qubits"], show=False)
    saver.save(fig, f"{prefix}_eigenvector_weights")

    return {
        "egvals_midpoint": egvals_midpoint,
        "egvecs_midpoint": vecs_mid,
        "ramp_areas": ramp_areas,
        "midpoint_index": midpoint,
    }


def run_population_preview(common: dict, saver, *, sample_wait: float = 0.0,
                           prefix: str = "presim"):
    """Evolve one representative wait time and plot eigenbasis populations."""
    io.section(f"Population preview (wait time = {sample_wait:g})")
    evo = build_experiment(common, sample_wait)

    t0 = time.perf_counter()
    sim_r1, sim_w, sim_r2 = evo.time_evolution()
    elapsed = time.perf_counter() - t0

    pops = [evo.calculate_eigenbasis_populations(s) for s in (sim_r1, sim_w, sim_r2)]
    t_all = np.concatenate([sim_r1.times, sim_w.times, sim_r2.times])
    pops_all = np.concatenate(pops, axis=0)

    fig, _ = plot_eigenbasis_populations(t_all, pops_all, tr=evo.tr, tw=evo.tw,
                                         show=False)
    saver.save(fig, f"{prefix}_eigenbasis_populations")

    io.kv("Populations after first ramp", io.fmt_array(pops[0][-1]))
    io.kv("Populations at end", io.fmt_array(pops[2][-1]))
    io.kv("Wall time for this single run", io.fmt_duration(elapsed))

    return {"elapsed": elapsed, "final_populations": pops[2][-1]}


def run_sampling_check(common: dict, egvals_midpoint) -> dict:
    """Check the planned (max_wait_time, n_sims) grid against the spectrum."""
    io.section("Sampling adequacy")
    check = check_sampling_adequacy(
        egvals_midpoint, common["max_wait_time"], common["n_sims"], verbose=True)
    if not (check["resolved"] and check["nyquist_ok"]):
        print()
        print("  *** WARNING: the planned wait-time grid will give wrong or "
              "blended peaks.")
        print("  *** Adjust max_wait_time / n_sims as suggested above before "
              "trusting the sweep.")
    else:
        print("  Grid is adequate: all peaks resolved and sampled below Nyquist.")
    return check


# --------------------------------------------------------------------------
# Runtime estimate
# --------------------------------------------------------------------------
def estimate_runtime(common: dict, *, n_probe: int = 3, n_repeats: int = 1,
                     label: str = "sweep") -> dict:
    """Estimate the sweep's wall time by timing a few representative runs.

    The cost of one wait time grows essentially linearly with its simulated
    duration (2*ramp_time + tw), because the solver's step count does. So we
    time `n_probe` wait times spread across the grid, fit cost = a + b*duration
    by least squares, and integrate that over the full grid.

    `n_repeats` multiplies the result (used by the averaged sweep, which runs
    the whole grid once per noise realization).
    """
    tw_l = np.asarray(common["tw_l"], dtype=float)
    durations = 2 * common["ramp_time"] + tw_l

    io.section("Runtime estimate")
    n_probe = int(max(1, min(n_probe, len(tw_l))))
    probe_idx = np.unique(np.linspace(0, len(tw_l) - 1, n_probe).astype(int))
    print(f"  Timing {len(probe_idx)} probe run(s) spread across the grid...")

    probe_durations, probe_costs = [], []
    with_noise = common["r_noise"] or common["w_noise"]
    for idx in probe_idx:
        exp = build_experiment(common, float(tw_l[idx]), with_noise=with_noise)
        t0 = time.perf_counter()
        exp.time_evolution()
        cost = time.perf_counter() - t0
        probe_durations.append(durations[idx])
        probe_costs.append(cost)
        print(f"    tw = {tw_l[idx]:9.3f}  (simulated {durations[idx]:9.3f})  "
              f"-> {cost:7.3f} s")

    probe_durations = np.asarray(probe_durations, dtype=float)
    probe_costs = np.asarray(probe_costs, dtype=float)

    # Fit cost = a + b * duration. With one probe, or a degenerate spread,
    # fall back to a pure proportionality through that point.
    if len(probe_durations) >= 2 and np.ptp(probe_durations) > 0:
        design = np.vstack([np.ones_like(probe_durations), probe_durations]).T
        (a, b), *_ = np.linalg.lstsq(design, probe_costs, rcond=None)
    else:
        a, b = 0.0, probe_costs[0] / max(probe_durations[0], 1e-12)

    per_run = a + b * durations
    per_run = np.clip(per_run, 0.0, None)
    total = float(per_run.sum()) * int(n_repeats)

    print()
    io.kv("Fitted fixed cost per run", f"{a:.4f} s")
    io.kv("Fitted cost per unit sim-time", f"{b:.6f} s")
    io.kv("Wait times in the grid", len(tw_l))
    if n_repeats > 1:
        io.kv("Repetitions (noise realizations)", n_repeats)
        io.kv("Estimated time per realization",
              io.fmt_duration(total / n_repeats))
    io.kv(f"ESTIMATED {label.upper()} RUNTIME", io.fmt_duration(total))
    print(f"  (expected to finish around "
          f"{_eta_clock(total)}; a rough extrapolation, not a guarantee)")

    return {
        "probe_durations": probe_durations.tolist(),
        "probe_costs": probe_costs.tolist(),
        "fit_intercept": float(a),
        "fit_slope": float(b),
        "estimated_seconds": total,
        "estimated_human": io.fmt_duration(total),
        "n_repeats": int(n_repeats),
    }


def _eta_clock(seconds: float) -> str:
    import datetime as _dt

    eta = _dt.datetime.now() + _dt.timedelta(seconds=float(seconds))
    return eta.strftime("%Y-%m-%d %H:%M:%S")
