#!/usr/bin/env python
"""
Script 3 -- amplitude vs. ramp-asymmetry sweep.

For each asymmetry value `a` the target qubit's ramp coefficients are set to
`a` (the others stay at 1), the full wait-time sweep is run, a light
diagnostics pass (FFT + global amplitude fit) extracts the mode amplitudes,
and the sum of the `n_smallest` fitted amplitudes is recorded together with
the Z first-ramp area. The two are then plotted against each other.

This is an independent study, so it writes to its own run directory tree --
separate from the script-1 sweeps:

    <asymmetry.output_root>/asymmetry_<timestamp>/
        asymmetry_sweep.npz   a_values, amp_sum, ramp1_z_areas, ramp1_x_areas
        pc_lists.npy          the raw signal for every asymmetry value
        tw_l.npy              the wait-time grid
        metadata.yaml         every parameter used
        output.txt            a transcript of everything printed below
        plots/                pre-simulation and result figures

Usage:
    python 03_run_asymmetry.py [--config config.yaml] [--no-run]
"""
from __future__ import annotations

import argparse
import sys
import time
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from lzs_pipeline import config as cfgmod  # noqa: E402
from lzs_pipeline import io_utils as io  # noqa: E402
from lzs_pipeline import presim  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")

from fourier import fit_decay_rates, fourier_analysis  # noqa: E402
from lsz_experiment import run_experiment_sweep, set_global_seed  # noqa: E402
from plotting import style  # noqa: E402
from plotting.diagnostic_plots import (  # noqa: E402
    plot_amplitude_vs_area,
    plot_asymmetry_vs_a,
)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=None,
                   help="path to config.yaml (default: alongside this script)")
    p.add_argument("--run-name", default=None,
                   help="name for the run directory "
                        "(default: asymmetry_<timestamp>)")
    p.add_argument("--no-run", action="store_true",
                   help="force diagnostics only, overriding "
                        "asymmetry.run_simulation")
    p.add_argument("--run", action="store_true",
                   help="force running the sweep")
    return p.parse_args(argv)


def build_a_values(asym_cfg: dict) -> np.ndarray:
    """The asymmetry grid: geometric (default) or linear between a_min, a_max."""
    n = int(asym_cfg.get("n_asym", 15))
    a_min = float(asym_cfg.get("a_min", 1.0))
    a_max = float(asym_cfg.get("a_max", 10.0))
    spacing = str(asym_cfg.get("spacing", "geometric")).lower()

    if n < 1:
        raise ValueError("asymmetry.n_asym must be at least 1")
    if n == 1:
        return np.array([a_min], dtype=float)

    if spacing == "linear":
        return np.linspace(a_min, a_max, n)
    if spacing == "geometric":
        if a_min <= 0 or a_max <= 0:
            raise ValueError("geometric spacing needs a_min, a_max > 0")
        return 2.0 ** np.linspace(np.log2(a_min), np.log2(a_max), n)
    raise ValueError(f"unknown asymmetry.spacing: {spacing!r}")


def coefficients_for(a: float, common: dict, asym_cfg: dict):
    """The four per-qubit coefficient lists for one asymmetry value.

    The target qubit gets `a`; every other qubit keeps 1. Z and X are swept
    independently according to `sweep_z` / `sweep_x`.
    """
    n = common["n_qubits"]
    target = int(asym_cfg.get("target_qubit", 0))
    if not 0 <= target < n:
        raise ValueError(
            f"asymmetry.target_qubit = {target} is out of range for {n} qubits")

    def make(sweep_it):
        coeffs = [1.0] * n
        if sweep_it:
            coeffs[target] = float(a)
        return coeffs

    z = make(bool(asym_cfg.get("sweep_z", True)))
    x = make(bool(asym_cfg.get("sweep_x", False)))
    return z, list(z), x, list(x)


def analyse_signal(pc, tw_l, common, fourier_cfg, n_smallest, noise_on):
    """FFT + global fit -> the sum of the `n_smallest` fitted amplitudes."""
    n_levels = 2 ** common["n_qubits"]
    n_peaks = fourier_cfg.get("n_peaks")
    if n_peaks is None:
        n_peaks = n_levels * (n_levels - 1) // 2

    freqs, phases = fourier_analysis(
        pc, tw_l,
        n_peaks=int(n_peaks),
        prominence_threshold=fourier_cfg.get("prominence_threshold", 0.001),
        zero_pad_factor=fourier_cfg.get("zero_pad_factor", 32),
        window=fourier_cfg.get("window", "hann"),
        detect_prominence=fourier_cfg.get("detect_prominence", 0.001),
        f_min=fourier_cfg.get("f_min", 0.1),
        plots=False,
    )
    amps, lambdas, dc = fit_decay_rates(pc, tw_l, freqs, phases,
                                        noise=noise_on, plots=False)
    amps = np.asarray(amps, dtype=float)
    k = int(min(n_smallest, amps.size))
    smallest = np.sort(amps)[:k]
    return float(np.sum(smallest)), amps, freqs


def main(argv=None):
    args = parse_args(argv)

    script_dir = Path(__file__).resolve().parent
    config_path = Path(args.config) if args.config else script_dir / "config.yaml"
    cfg = cfgmod.load_config(config_path)

    common = cfgmod.resolve_common(cfg["common"])
    asym_cfg = cfg["asymmetry"]

    run_simulation = bool(asym_cfg.get("run_simulation", True))
    if args.no_run:
        run_simulation = False
    if args.run:
        run_simulation = True

    style.apply(common["palette"])

    output_root = Path(asym_cfg.get("output_root", "results/asymmetry"))
    if not output_root.is_absolute():
        output_root = script_dir / output_root
    run_dir = io.create_run_dir(output_root, "asymmetry",
                                args.run_name or asym_cfg.get("run_name"))
    saver = io.FigureSaver(run_dir / "plots", formats=common["figure_formats"])

    status = "completed"
    error_text = None
    runtime_est = None
    wall_seconds = None
    result_files = {}
    a_values = None

    with io.RunLogger(run_dir) as logger:
        io.banner("LZS SPECTROSCOPY -- SCRIPT 3: AMPLITUDE vs. RAMP ASYMMETRY",
                  f"run directory: {run_dir}")
        io.kv("Config file", config_path, width=34)
        io.kv("Simulation will run", "YES" if run_simulation
              else "NO (diagnostics only)", width=34)

        try:
            a_values = build_a_values(asym_cfg)
            n_asym = len(a_values)
            n_smallest = int(asym_cfg.get("n_smallest", 3))
            target = int(asym_cfg.get("target_qubit", 0))
            noise_on = common["r_noise"] or common["w_noise"]

            # ---------------- Configuration ----------------
            io.banner("CONFIGURATION")
            presim.print_parameter_report(common)

            io.section("Asymmetry sweep settings")
            io.kv("Number of asymmetry values", n_asym)
            io.kv("Spacing", asym_cfg.get("spacing", "geometric"))
            io.kv("Range", f"{asym_cfg.get('a_min', 1.0)} ... "
                           f"{asym_cfg.get('a_max', 10.0)}")
            io.kv("Values", io.fmt_array(a_values, precision=3))
            io.kv("Target qubit", target)
            io.kv("Sweep Z ramps", bool(asym_cfg.get("sweep_z", True)))
            io.kv("Sweep X ramps", bool(asym_cfg.get("sweep_x", False)))
            io.kv("Amplitudes summed (n_smallest)", n_smallest)
            io.kv("Total wait-time sweeps", n_asym)
            io.kv("Total time evolutions", n_asym * common["n_sims"])

            io.section("Environment")
            for key, value in io.environment_info().items():
                io.kv(key, value)

            # ---------------- Pre-simulation diagnostics ----------------
            io.banner("PRE-SIMULATION DIAGNOSTICS")
            spectrum_info = presim.run_schedule_and_spectrum(common, saver)
            egvals_midpoint = spectrum_info["egvals_midpoint"]
            presim.run_sampling_check(common, egvals_midpoint)

            # Ramp areas are cheap and independent of the sweep, so tabulate
            # them for every asymmetry value up front.
            io.section("Ramp areas across the asymmetry grid")
            ramp1_z_areas, ramp1_x_areas = [], []
            if common["n_qubits"] >= 2:
                print(f"    {'a':>8} {'ramp1 area (z)':>18} {'ramp1 area (x)':>18}")
                for a in a_values:
                    up_z, down_z, up_x, down_x = coefficients_for(a, common, asym_cfg)
                    exp_area = presim.build_experiment(
                        common, 0.0, up_z=up_z, down_z=down_z,
                        up_x=up_x, down_x=down_x)
                    areas = exp_area.calculate_ramp_areas()
                    ramp1_z_areas.append(areas["z"]["ramp1"])
                    ramp1_x_areas.append(areas["x"]["ramp1"])
                    print(f"    {a:>8.3f} {ramp1_z_areas[-1]:>+18.6f} "
                          f"{ramp1_x_areas[-1]:>+18.6f}")
            else:
                print("  (ramp areas need >= 2 qubits; recording NaN)")
                ramp1_z_areas = [np.nan] * n_asym
                ramp1_x_areas = [np.nan] * n_asym
            ramp1_z_areas = np.asarray(ramp1_z_areas, dtype=float)
            ramp1_x_areas = np.asarray(ramp1_x_areas, dtype=float)

            # ---------------- Runtime estimate ----------------
            io.banner("RUNTIME ESTIMATE")
            rt_cfg = asym_cfg.get("runtime_estimate", {}) or {}
            if rt_cfg.get("enabled", True):
                runtime_est = presim.estimate_runtime(
                    common,
                    n_probe=int(rt_cfg.get("n_probe", 3)),
                    n_repeats=n_asym,
                    label="asymmetry sweep",
                )
                print()
                print(f"  That is {n_asym} full wait-time sweeps, one per "
                      f"asymmetry value.")
                print("  The per-sweep FFT and global fit add a small, roughly "
                      "constant overhead.")
            else:
                print("  Runtime estimation disabled in the config.")

            # ---------------- The sweep ----------------
            if not run_simulation:
                io.banner("SIMULATION SKIPPED",
                          "asymmetry.run_simulation is false -- diagnostics only")
                print("  Review the ramp-area table and the figures above, then")
                print("  set asymmetry.run_simulation: true (or pass --run).")
                status = "diagnostics_only"
            else:
                io.banner("RUNNING THE ASYMMETRY SWEEP")
                fourier_cfg = asym_cfg.get("fourier", {}) or {}
                amp_sums, pc_lists, all_amps = [], [], []
                t0 = time.perf_counter()

                for j, a in enumerate(a_values):
                    up_z, down_z, up_x, down_x = coefficients_for(
                        a, common, asym_cfg)
                    print(f"\n  [{j + 1}/{n_asym}]  a = {a:.4f}")

                    set_global_seed(common["noise_seed"])
                    pc = run_experiment_sweep(
                        common["n_qubits"], common["epsilon"],
                        common["H_target_dict"], common["ramp_time"],
                        common["tw_l"], common["dt"],
                        up_z_assym_list=up_z, down_z_assym_list=down_z,
                        up_x_assym_list=up_x, down_x_assym_list=down_x,
                        r_noise=common["r_noise"], w_noise=common["w_noise"],
                        gamma_dec_list=common["t1_rates"],
                        gamma_dep_list=common["t2_rates"],
                        initial_state=common["psi0"],
                        ramp_error=common["ramp_error"],
                        wait_error=common["wait_error"],
                        ramp_bias=common["ramp_bias"],
                        noise_seed=None, show_progress=True,
                    )
                    pc_lists.append(pc)

                    amp_sum, amps, freqs = analyse_signal(
                        pc, common["tw_l"], common, fourier_cfg,
                        n_smallest, noise_on)
                    amp_sums.append(amp_sum)
                    all_amps.append(amps)

                    print(f"      ramp1 z-area = {ramp1_z_areas[j]:+.6f}   "
                          f"sum({n_smallest} smallest |A|) = {amp_sum:.6f}   "
                          f"peaks found = {len(freqs)}")

                wall_seconds = time.perf_counter() - t0
                amp_sums = np.asarray(amp_sums, dtype=float)
                pc_lists = np.asarray(pc_lists, dtype=float)

                io.section("Sweep complete")
                io.kv("Wall time", io.fmt_duration(wall_seconds))
                if runtime_est:
                    est = runtime_est["estimated_seconds"]
                    io.kv("Estimated beforehand", io.fmt_duration(est))
                    if est > 0:
                        io.kv("Estimate / actual", f"{est / wall_seconds:.2f}x")

                io.section("Results")
                print(f"    {'a':>8} {'ramp1 z-area':>16} "
                      f"{'sum(' + str(n_smallest) + ' smallest |A|)':>26}")
                for a, area, s in zip(a_values, ramp1_z_areas, amp_sums):
                    print(f"    {a:>8.3f} {area:>+16.6f} {s:>26.6f}")

                # ---------------- Figures ----------------
                io.section("Figures")
                fig, _ = plot_asymmetry_vs_a(a_values, amp_sums,
                                             ramp1_z_areas, show=False)
                saver.save(fig, "asymmetry_vs_a")
                fig, _ = plot_amplitude_vs_area(ramp1_z_areas, amp_sums,
                                                show=False)
                saver.save(fig, "amplitude_vs_area")
                io.kv("Figures written", len(saver.saved))

                # ---------------- Save ----------------
                io.section("Saving data")
                npz_path = io.unique_path(run_dir / "asymmetry_sweep.npz")
                np.savez(
                    npz_path,
                    a_values=a_values,
                    amp_sum_smallest=amp_sums,
                    ramp1_z_areas=ramp1_z_areas,
                    ramp1_x_areas=ramp1_x_areas,
                    all_amplitudes=np.asarray(all_amps, dtype=object),
                    allow_pickle=True,
                )
                io.kv("Sweep results", npz_path)
                result_files["asymmetry_sweep"] = npz_path.name

                pc_path = io.unique_path(run_dir / "pc_lists.npy")
                np.save(pc_path, pc_lists)
                io.kv("Raw signals", pc_path)
                result_files["pc_lists"] = pc_path.name

                tw_path = io.unique_path(run_dir / "tw_l.npy")
                np.save(tw_path, common["tw_l"])
                io.kv("Wait-time grid", tw_path)
                result_files["tw_l"] = tw_path.name

        except Exception:
            status = "failed"
            error_text = traceback.format_exc()
            print()
            io.banner("RUN FAILED")
            print(error_text)

        # ---------------- Metadata ----------------
        metadata = {
            "script": "03_run_asymmetry.py",
            "status": status,
            "run_directory": str(run_dir),
            "config_file": str(config_path),
            "environment": io.environment_info(),
            "common": cfgmod.common_to_metadata(common),
            "asymmetry": {
                **{k: v for k, v in asym_cfg.items() if k != "runtime_estimate"},
                "run_simulation": run_simulation,
                "a_values": a_values.tolist() if a_values is not None else None,
                "runtime_estimate": runtime_est,
                "wall_seconds": wall_seconds,
                "wall_human": io.fmt_duration(wall_seconds) if wall_seconds else None,
                "data_files": result_files,
            },
            "figures": [str(p.relative_to(run_dir)) for p in saver.saved],
        }
        if error_text:
            metadata["error"] = error_text

        meta_path = io.write_metadata(run_dir / "metadata.yaml", metadata)

        io.banner("DONE", f"status: {status}")
        io.kv("Run directory", run_dir)
        io.kv("Metadata", meta_path)
        io.kv("Figures", saver.summary())
        io.kv("Log", logger.path)
        print()

    return 0 if status != "failed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
