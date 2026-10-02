#!/usr/bin/env python
"""
Script 1 -- pre-simulation diagnostics and the wait-time sweep.

Prints a full report of the configuration, produces the pre-simulation
figures (ramp schedule, instantaneous spectrum, midpoint eigenvectors,
eigenbasis populations), checks that the planned wait-time grid can resolve
the spectrum, estimates how long the sweep will take, and then -- if
`sweep.run_simulation` is true in the config -- runs the sweep and stores the
resulting `pc_list`.

Everything lands in a fresh run directory that is never written over:

    <sweep.output_root>/sweep_<timestamp>/
        pc_list.npy        the swept signal P(psi0) vs wait time
        tw_l.npy           the wait-time grid
        metadata.yaml      every parameter used, plus provenance
        output.txt         a transcript of everything printed below
        plots/             the pre-simulation figures

Usage:
    python 01_run_sweep.py [--config config.yaml] [--no-run] [--run-name NAME]
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

matplotlib.use("Agg")  # figures are saved, never shown

from lsz_experiment import (  # noqa: E402
    run_averaged_sweep,
    run_experiment_sweep,
    set_global_seed,
)
from plotting import style  # noqa: E402


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=None,
                   help="path to config.yaml (default: alongside this script)")
    p.add_argument("--run-name", default=None,
                   help="name for the run directory (default: sweep_<timestamp>)")
    p.add_argument("--no-run", action="store_true",
                   help="force diagnostics only, overriding sweep.run_simulation")
    p.add_argument("--run", action="store_true",
                   help="force running the sweep, overriding sweep.run_simulation")
    return p.parse_args(argv)


def plot_signal(tw_l, pc_list, saver, name="sweep_signal", title=None):
    """The swept signal itself -- the one figure script 1 makes after the sweep."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=style.FIG_WIDE)
    ax.plot(tw_l, pc_list, color=style.DATA, lw=style.LINEWIDTH)
    style.style_axis(ax, xlabel="wait time", ylabel=r"$P(\psi_0)$", title=title)
    fig.tight_layout()
    saver.save(fig, name)


def plot_realizations(tw_l, pc_all, pc_mean, saver, name="sweep_realizations"):
    """Individual noise realizations against their average."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=style.FIG_WIDE)
    for r in range(pc_all.shape[0]):
        ax.plot(tw_l, pc_all[r], color=style.SUPPORT, lw=0.8, alpha=0.6,
                label="realizations" if r == 0 else None)
    ax.plot(tw_l, pc_mean, color=style.DATA, lw=style.LINEWIDTH,
            label=f"average of {pc_all.shape[0]}")
    style.style_axis(ax, xlabel="wait time", ylabel=r"$P(\psi_0)$", legend=True)
    fig.tight_layout()
    saver.save(fig, name)


def main(argv=None):
    args = parse_args(argv)

    script_dir = Path(__file__).resolve().parent
    config_path = Path(args.config) if args.config else script_dir / "config.yaml"
    cfg = cfgmod.load_config(config_path)

    common = cfgmod.resolve_common(cfg["common"])
    sweep_cfg = cfg["sweep"]

    run_simulation = bool(sweep_cfg.get("run_simulation", True))
    if args.no_run:
        run_simulation = False
    if args.run:
        run_simulation = True

    style.apply(common["palette"])

    output_root = Path(sweep_cfg.get("output_root", "results/sweeps"))
    if not output_root.is_absolute():
        output_root = script_dir / output_root
    run_dir = io.create_run_dir(output_root, "sweep",
                                args.run_name or sweep_cfg.get("run_name"))

    saver = io.FigureSaver(run_dir / "plots", formats=common["figure_formats"])

    status = "completed"
    error_text = None
    sweep_info = {}
    runtime_est = None
    wall_seconds = None

    with io.RunLogger(run_dir) as logger:
        io.banner("LZS SPECTROSCOPY -- SCRIPT 1: PRE-SIMULATION + SWEEP",
                  f"run directory: {run_dir}")
        io.kv("Config file", config_path, width=34)
        io.kv("Simulation will run", "YES" if run_simulation
              else "NO (diagnostics only)", width=34)

        try:
            # ---------------- Configuration report ----------------
            io.banner("CONFIGURATION")
            presim.print_parameter_report(common)

            io.section("Environment")
            for key, value in io.environment_info().items():
                io.kv(key, value)

            # ---------------- Pre-simulation diagnostics ----------------
            io.banner("PRE-SIMULATION DIAGNOSTICS")
            spectrum_info = presim.run_schedule_and_spectrum(common, saver)
            egvals_midpoint = spectrum_info["egvals_midpoint"]

            pop_info = presim.run_population_preview(common, saver)
            check = presim.run_sampling_check(common, egvals_midpoint)

            # ---------------- Runtime estimate ----------------
            averaging = sweep_cfg.get("averaging", {}) or {}
            averaging_on = bool(averaging.get("enabled", False))
            n_averages = int(averaging.get("n_averages", 10))
            rt_cfg = sweep_cfg.get("runtime_estimate", {}) or {}

            io.banner("RUNTIME ESTIMATE")
            if averaging_on:
                io.kv("Averaged sweep", f"YES ({n_averages} realizations)")
            if rt_cfg.get("enabled", True):
                runtime_est = presim.estimate_runtime(
                    common,
                    n_probe=int(rt_cfg.get("n_probe", 3)),
                    n_repeats=n_averages if averaging_on else 1,
                    label="sweep",
                )
            else:
                print("  Runtime estimation disabled in the config.")

            # ---------------- The sweep ----------------
            if not run_simulation:
                io.banner("SIMULATION SKIPPED",
                          "sweep.run_simulation is false -- diagnostics only")
                print("  Review the figures and the sampling check above, then")
                print("  set sweep.run_simulation: true (or pass --run) to "
                      "run the sweep.")
                status = "diagnostics_only"
            else:
                io.banner("RUNNING THE SWEEP")
                if not (check["resolved"] and check["nyquist_ok"]):
                    print("  NOTE: proceeding despite the sampling-adequacy "
                          "warning above.\n")

                set_global_seed(common["noise_seed"])
                t0 = time.perf_counter()

                if averaging_on:
                    print(f"  Averaging over {n_averages} independent noise "
                          f"realizations.\n")
                    pc_list, pc_all = run_averaged_sweep(
                        common["n_qubits"], common["epsilon"],
                        common["H_target_dict"], common["ramp_time"],
                        common["tw_l"], common["dt"], n_averages,
                        up_z_assym_list=common["up_z"],
                        down_z_assym_list=common["down_z"],
                        up_x_assym_list=common["up_x"],
                        down_x_assym_list=common["down_x"],
                        r_noise=common["r_noise"], w_noise=common["w_noise"],
                        gamma_dec_list=common["t1_rates"],
                        gamma_dep_list=common["t2_rates"],
                        initial_state=common["psi0"],
                        ramp_error=common["ramp_error"],
                        wait_error=common["wait_error"],
                        ramp_bias=common["ramp_bias"],
                        seed_rng=common["noise_seed"],
                        return_all=True,
                    )
                else:
                    pc_list = run_experiment_sweep(
                        common["n_qubits"], common["epsilon"],
                        common["H_target_dict"], common["ramp_time"],
                        common["tw_l"], common["dt"],
                        up_z_assym_list=common["up_z"],
                        down_z_assym_list=common["down_z"],
                        up_x_assym_list=common["up_x"],
                        down_x_assym_list=common["down_x"],
                        r_noise=common["r_noise"], w_noise=common["w_noise"],
                        gamma_dec_list=common["t1_rates"],
                        gamma_dep_list=common["t2_rates"],
                        initial_state=common["psi0"],
                        ramp_error=common["ramp_error"],
                        wait_error=common["wait_error"],
                        ramp_bias=common["ramp_bias"],
                        noise_seed=None,
                    )
                    pc_all = None

                wall_seconds = time.perf_counter() - t0

                io.section("Sweep complete")
                io.kv("Wall time", io.fmt_duration(wall_seconds))
                if runtime_est:
                    est = runtime_est["estimated_seconds"]
                    io.kv("Estimated beforehand", io.fmt_duration(est))
                    if est > 0:
                        io.kv("Estimate / actual", f"{est / wall_seconds:.2f}x")
                io.kv("Signal length", len(pc_list))
                io.kv("Signal min / max",
                      f"{np.min(pc_list):.6f} / {np.max(pc_list):.6f}")
                io.kv("Signal mean", f"{np.mean(pc_list):.6f}")

                # ---------------- Save ----------------
                io.section("Saving data")
                pc_path = io.unique_path(run_dir / "pc_list.npy")
                np.save(pc_path, pc_list)
                io.kv("pc_list", pc_path)

                tw_path = io.unique_path(run_dir / "tw_l.npy")
                np.save(tw_path, common["tw_l"])
                io.kv("tw_l", tw_path)

                sweep_info = {
                    "pc_list_file": pc_path.name,
                    "tw_l_file": tw_path.name,
                    "signal_min": float(np.min(pc_list)),
                    "signal_max": float(np.max(pc_list)),
                    "signal_mean": float(np.mean(pc_list)),
                }

                plot_signal(common["tw_l"], pc_list, saver)

                if pc_all is not None:
                    all_path = io.unique_path(run_dir / "pc_all_realizations.npy")
                    np.save(all_path, pc_all)
                    io.kv("pc_all (realizations)", all_path)
                    sweep_info["pc_all_file"] = all_path.name
                    plot_realizations(common["tw_l"], pc_all, pc_list, saver)

        except Exception:
            status = "failed"
            error_text = traceback.format_exc()
            print()
            io.banner("RUN FAILED")
            print(error_text)

        # ---------------- Metadata ----------------
        metadata = {
            "script": "01_run_sweep.py",
            "status": status,
            "run_directory": str(run_dir),
            "config_file": str(config_path),
            "environment": io.environment_info(),
            "common": cfgmod.common_to_metadata(common),
            "sweep": {
                "run_simulation": run_simulation,
                "averaging": sweep_cfg.get("averaging", {}),
                "runtime_estimate": runtime_est,
                "wall_seconds": wall_seconds,
                "wall_human": io.fmt_duration(wall_seconds) if wall_seconds else None,
                **sweep_info,
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
        if status == "completed":
            print()
            print("  Next: run script 2 on this directory --")
            print(f"    python 02_run_diagnostics.py --input-dir {run_dir}")
        print()

    return 0 if status != "failed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
