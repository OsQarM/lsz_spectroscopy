#!/usr/bin/env python
"""
Script 2 -- full diagnostics on a stored pc_list.

Takes a run directory produced by script 1, reloads its signal, and runs the
whole analysis chain: FFT peak extraction, global amplitude/decay fit,
matching against the true energy differences, turnpike spectrum
reconstruction, u_m / phi_m recovery, exact-state validation, optional
dephasing comparison, and the optional joint refinement.

Physics parameters come from the run's `metadata.yaml`; anything the metadata
does not carry falls back to the `common` block of `config.yaml`. The
diagnostics-only settings always come from the config's `diagnostics` block.

Output is written *into the source run directory*, without overwriting
anything: figures go into `plots/`, data files and a
`metadata_diagnostics.yaml` sit at the top level, and the transcript is saved
as `output_diagnostics.txt`.

Usage:
    python 02_run_diagnostics.py --input-dir results/sweeps/sweep_20260102-101530
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

from diagnostics import run_full_diagnostics  # noqa: E402
from plotting import style  # noqa: E402


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=None,
                   help="path to config.yaml (default: alongside this script)")
    p.add_argument("--input-dir", default=None,
                   help="run directory from script 1 "
                        "(default: diagnostics.input_dir in the config)")
    p.add_argument("--no-refine", action="store_true",
                   help="skip the joint refinement stage")
    return p.parse_args(argv)


def load_signal(input_dir: Path, metadata: dict):
    """Load `pc_list` (and `tw_l` if stored) from a script-1 run directory."""
    sweep_meta = metadata.get("sweep", {}) if metadata else {}

    pc_name = sweep_meta.get("pc_list_file", "pc_list.npy")
    pc_path = input_dir / pc_name
    if not pc_path.is_file():
        candidates = sorted(input_dir.glob("pc_list*.npy"))
        if not candidates:
            raise FileNotFoundError(
                f"no pc_list .npy found in {input_dir}. Did script 1 run with "
                f"sweep.run_simulation: true?"
            )
        pc_path = candidates[0]
    pc_list = np.load(pc_path)

    tw_path = input_dir / sweep_meta.get("tw_l_file", "tw_l.npy")
    tw_l = np.load(tw_path) if tw_path.is_file() else None

    return pc_list, tw_l, pc_path, (tw_path if tw_l is not None else None)


def save_diagnostics_data(run_dir: Path, results: dict, refine_results):
    """Store the numeric diagnostics output as a single compressed archive."""
    arrays = {}
    for key in ("freqs", "phases", "amplitudes", "lambdas", "energy_diffs",
                "true_differences", "matched_detected", "matched_true",
                "experimental_energies", "shifted_true_energies",
                "e_m", "u_m", "phi_m", "u_exact_amplitudes",
                "u_exact_phases_rel", "w_mn", "a_mn", "d_mn", "lmd_mn",
                "w_predicted", "possible_lambdas", "exact_lambdas"):
        value = results.get(key)
        if value is None:
            continue
        try:
            arrays[key] = np.asarray(value, dtype=float)
        except (TypeError, ValueError):
            # Non-numeric (e.g. dict-valued exact_lambdas) -- keep as object.
            arrays[key] = np.asarray(value, dtype=object)

    dc_fit = results.get("dc_fit")
    if dc_fit is not None:
        arrays["dc_fit"] = np.asarray(dc_fit, dtype=float)

    paths = {}
    diag_path = io.unique_path(run_dir / "diagnostics_results.npz")
    np.savez(diag_path, **arrays)
    paths["diagnostics_results"] = diag_path

    if refine_results is not None:
        refined = refine_results.get("refined", {})
        ref_arrays = {}
        for key, value in refined.items():
            try:
                ref_arrays[f"refined_{key}"] = np.asarray(value, dtype=float)
            except (TypeError, ValueError):
                continue
        if ref_arrays:
            ref_path = io.unique_path(run_dir / "refinement_results.npz")
            np.savez(ref_path, **ref_arrays)
            paths["refinement_results"] = ref_path

    return paths


def main(argv=None):
    args = parse_args(argv)

    script_dir = Path(__file__).resolve().parent
    config_path = Path(args.config) if args.config else script_dir / "config.yaml"
    cfg = cfgmod.load_config(config_path)
    diag_cfg = cfg["diagnostics"]

    input_dir = args.input_dir or diag_cfg.get("input_dir")
    if not input_dir:
        raise SystemExit(
            "no input directory given: pass --input-dir or set "
            "diagnostics.input_dir in the config"
        )
    input_dir = Path(input_dir)
    if not input_dir.is_absolute():
        input_dir = (script_dir / input_dir).resolve()
    if not input_dir.is_dir():
        raise SystemExit(f"input directory does not exist: {input_dir}")

    # Metadata drives the physics; the config's `common` block fills the gaps.
    meta_path = input_dir / "metadata.yaml"
    metadata = io.read_metadata(meta_path) if meta_path.is_file() else {}
    common = cfgmod.resolve_common(cfg["common"], metadata=metadata)

    style.apply(common["palette"])
    saver = io.FigureSaver(input_dir / "plots", formats=common["figure_formats"])

    status = "completed"
    error_text = None
    data_paths = {}
    wall_seconds = None

    with io.RunLogger(input_dir, filename="output_diagnostics.txt") as logger:
        io.banner("LZS SPECTROSCOPY -- SCRIPT 2: FULL DIAGNOSTICS",
                  f"source run: {input_dir}")
        io.kv("Config file", config_path, width=34)
        io.kv("Source metadata", meta_path if metadata else "none (config only)",
              width=34)

        try:
            # ---------------- Load the signal ----------------
            io.section("Input data")
            pc_list, tw_l_stored, pc_path, tw_path = load_signal(input_dir, metadata)
            io.kv("pc_list", pc_path)
            io.kv("Signal length", len(pc_list))

            if tw_l_stored is not None:
                tw_l = np.asarray(tw_l_stored, dtype=float)
                io.kv("tw_l", f"{tw_path}  (stored with the run)")
            else:
                tw_l = common["tw_l"]
                io.kv("tw_l", "reconstructed from protocol parameters")

            if len(tw_l) != len(pc_list):
                raise ValueError(
                    f"length mismatch: pc_list has {len(pc_list)} points but "
                    f"tw_l has {len(tw_l)}. The metadata's protocol block does "
                    f"not match the stored signal."
                )
            io.kv("Wait-time range", f"0 ... {tw_l[-1]:g}")
            io.kv("Signal min / max",
                  f"{np.min(pc_list):.6f} / {np.max(pc_list):.6f}")

            # ---------------- Parameters + provenance ----------------
            io.banner("PARAMETERS")
            print("  Each section is tagged with where its values came from:")
            print("  'metadata' = the source run's metadata.yaml, "
                  "'config' = config.yaml fallback.")
            presim.print_parameter_report(common, common["_provenance"])

            io.section("Diagnostics settings  [from config]")
            fourier_cfg = diag_cfg.get("fourier", {}) or {}
            n_peaks = fourier_cfg.get("n_peaks")
            if n_peaks is None:
                n_levels = 2 ** common["n_qubits"]
                n_peaks = n_levels * (n_levels - 1) // 2
                io.kv("n_peaks", f"{n_peaks}  (derived from n_qubits)")
            else:
                io.kv("n_peaks", n_peaks)
            for key in ("prominence_threshold", "zero_pad_factor", "window",
                        "detect_prominence", "f_min"):
                io.kv(key, fourier_cfg.get(key))

            refine_cfg = dict(diag_cfg.get("refinement", {}) or {})
            refine_on = bool(refine_cfg.pop("enabled", True)) and not args.no_refine
            io.kv("refinement enabled", refine_on)
            if refine_on:
                for key, value in refine_cfg.items():
                    io.kv(f"  refinement.{key}", value)

            # ---------------- Midpoint spectrum ----------------
            io.banner("MIDPOINT SPECTRUM")
            print("  Recomputed from the Hamiltonian; this supplies the true")
            print("  energy differences the detected peaks are matched against.")
            preview = presim.build_experiment(common, 0.0)
            _, egvals, _ = preview.show_spectrum(common["n_steps"])
            egvals_midpoint = np.asarray(egvals[len(egvals) // 2])
            io.kv("Midpoint eigenvalues", io.fmt_array(egvals_midpoint))

            presim.run_sampling_check(common, egvals_midpoint)

            # ---------------- Full diagnostics ----------------
            io.banner("FULL DIAGNOSTICS")
            saver.mark_existing()
            noise_on = common["w_noise"] or common["r_noise"]
            t0 = time.perf_counter()

            results = run_full_diagnostics(
                pc_list, tw_l, common["n_qubits"], common["epsilon"],
                common["H_target_dict"], common["ramp_time"], common["dt"],
                egvals_midpoint,
                w_noise=noise_on,
                t1_rates=common["t1_rates"] if noise_on else None,
                t2_rates=common["t2_rates"] if noise_on else None,
                initial_state=common["psi0"],
                up_z_assym_list=common["up_z"],
                down_z_assym_list=common["down_z"],
                up_x_assym_list=common["up_x"],
                down_x_assym_list=common["down_x"],
                plots=True,
            )

            saved = saver.save_new("diagnostics")
            io.section("Diagnostics figures")
            io.kv("Figures captured", len(saved))

            # ---------------- Joint refinement ----------------
            refine_results = None
            if refine_on:
                io.banner("JOINT REFINEMENT")
                from refinement import (
                    plot_refined_diagnostics,
                    print_refinement_summary,
                    run_refinement,
                )

                saver.mark_existing()
                refine_results = run_refinement(
                    results, tw_l, pc_list, **refine_cfg)

                truth = {
                    "u_abs": results["u_exact_amplitudes"],
                    "phi": results["u_exact_phases_rel"],
                    "kphi": np.asarray(common["t2_rates"])
                    if (noise_on and common["t2_rates"] is not None) else None,
                    "kT1": np.asarray(common["t1_rates"])
                    if (noise_on and common["t1_rates"] is not None) else None,
                }
                print_refinement_summary(refine_results, truth=truth)

                plot_refined_diagnostics(
                    refine_results, results, tw_l, pc_list,
                    exact_lambdas=results.get("exact_lambdas"),
                )
                saved_ref = saver.save_new("refinement")
                io.section("Refinement figures")
                io.kv("Figures captured", len(saved_ref))

            wall_seconds = time.perf_counter() - t0

            # ---------------- Save ----------------
            io.banner("SAVING RESULTS")
            data_paths = save_diagnostics_data(input_dir, results, refine_results)
            for label, path in data_paths.items():
                io.kv(label, path)
            io.kv("Analysis wall time", io.fmt_duration(wall_seconds))

        except Exception:
            status = "failed"
            error_text = traceback.format_exc()
            print()
            io.banner("DIAGNOSTICS FAILED")
            print(error_text)

        # ---------------- Metadata ----------------
        diag_metadata = {
            "script": "02_run_diagnostics.py",
            "status": status,
            "source_run_directory": str(input_dir),
            "source_metadata": str(meta_path) if metadata else None,
            "config_file": str(config_path),
            "environment": io.environment_info(),
            "parameter_provenance": common["_provenance"],
            "common_used": cfgmod.common_to_metadata(common),
            "diagnostics": cfg["diagnostics"],
            "wall_seconds": wall_seconds,
            "wall_human": io.fmt_duration(wall_seconds) if wall_seconds else None,
            "data_files": {k: Path(v).name for k, v in data_paths.items()},
            "figures": [str(p.relative_to(input_dir)) for p in saver.saved],
        }
        if error_text:
            diag_metadata["error"] = error_text

        meta_out = io.write_metadata(input_dir / "metadata_diagnostics.yaml",
                                     diag_metadata)

        io.banner("DONE", f"status: {status}")
        io.kv("Run directory", input_dir)
        io.kv("Diagnostics metadata", meta_out)
        io.kv("Figures", saver.summary())
        io.kv("Log", logger.path)
        print()

    return 0 if status != "failed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
