"""
Loading and assembling the diagnostic panels for the paper figures.

Everything is read from the `diagnostics_results.npz` that
`refactored_code/02_run_diagnostics.py` writes, so these figures re-draw the
analysis that was actually run -- nothing is recomputed.

Five panels are available:

    transitions   detected vs. true transition energies
    energies      reconstructed vs. true energy levels (both turnpike branches)
    amplitudes    estimated vs. exact state-vector moduli |u_m|
    phases        estimated vs. exact phases phi_m
    rates         fitted vs. predicted coherence decay rates

All five are estimate-vs-true scatters against a y = x reference, so they
share one drawing routine and one set of axis/legend options.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml


PANELS = ("transitions", "energies", "amplitudes", "phases", "rates",
          "rate_levels", "noise_params")

# `rate_levels` is not an estimate-vs-true scatter: it plots the fitted rates
# against mode index (or frequency) with a dashed horizontal line at every
# distinct predicted rate, as `plotting.diagnostic_plots.plot_decay_rates`
# does. It is drawn by its own routine.
# `noise_params` is also drawn on its own: recovered per-qubit kappa_phi /
# kappa_T1 (or T2 / T1 times) against the configured values, per qubit.
SCATTER_PANELS = tuple(p for p in PANELS
                       if p not in ("rate_levels", "noise_params"))

PANEL_DEFAULT_LABELS = {
    "transitions": {
        "xlabel": r"true $\Delta\varepsilon$",
        "ylabel": r"detected $\Delta\varepsilon$",
    },
    "energies": {
        "xlabel": r"true $E_m$",
        "ylabel": r"reconstructed $E_m$",
    },
    "amplitudes": {
        "xlabel": r"true $|u_m|$",
        "ylabel": r"estimated $|\tilde{u}_m|$",
    },
    "phases": {
        "xlabel": r"true $\phi_m$",
        "ylabel": r"estimated $\tilde{\phi}_m$",
    },
    "rates": {
        "xlabel": r"predicted $\Gamma_{mn}$",
        "ylabel": r"fitted $\Gamma_{mn}$",
    },
    "rate_levels": {
        "xlabel": r"mode index",
        "ylabel": r"decay rate $\Gamma_{mn}$",
    },
    "noise_params": {
        "xlabel": r"qubit",
        "ylabel": r"rate",
    },
}


class DiagnosticsError(RuntimeError):
    """The requested diagnostics data could not be loaded."""


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------
def load_run(path, base_dir: Path) -> dict:
    """Load one run's diagnostics archive plus its metadata."""
    p = Path(path).expanduser()
    if not p.is_absolute():
        p = (base_dir / p).resolve()

    if p.is_dir():
        run_dir = p
        candidates = sorted(run_dir.glob("diagnostics_results*.npz"))
        if not candidates:
            raise DiagnosticsError(
                f"{run_dir} has no diagnostics_results.npz -- run "
                f"refactored_code/02_run_diagnostics.py on it first")
        npz_path = candidates[0]
    elif p.is_file():
        npz_path, run_dir = p, p.parent
    else:
        raise DiagnosticsError(f"source path does not exist: {p}")

    data = {}
    with np.load(npz_path, allow_pickle=True) as archive:
        for key in archive.files:
            data[key] = archive[key]

    metadata = {}
    meta_path = run_dir / "metadata.yaml"
    if meta_path.is_file():
        with open(meta_path) as fh:
            metadata = yaml.safe_load(fh) or {}

    ham = (metadata.get("common", {}) or {}).get("hamiltonian", {}) or {}

    return {
        "run_dir": run_dir,
        "npz_path": npz_path,
        "data": data,
        "metadata": metadata,
        "hamiltonian": ham,
        "n_qubits": len(ham.get("local_z", [])) or None,
    }


def _need(data: dict, keys, panel: str):
    missing = [k for k in keys if k not in data]
    if missing:
        raise DiagnosticsError(
            f"panel '{panel}' needs {', '.join(missing)}, which this run's "
            f"diagnostics archive does not contain")
    return [np.asarray(data[k], dtype=float) for k in keys]


# --------------------------------------------------------------------------
# Turnpike branches
# --------------------------------------------------------------------------
def turnpike_branches(levels, true_levels):
    """Both reflections of a turnpike reconstruction, labelled by fit.

    A turnpike solution is only determined up to reflection: if `levels`
    solves the problem then so does `max(levels) - levels[::-1]`. The stored
    `experimental_energies` is whichever of the two matched the true spectrum
    better (see `spectrum_matching.compare_spectrum_to_turnpike`), so the
    other branch is recovered by applying the reflection -- it is an
    involution, so this returns the pair regardless of which one was stored.

    Returns `(branches, best_key)` where `branches` maps a key to a dict with
    `levels`, `label` and `rms`.
    """
    levels = np.asarray(levels, dtype=float)
    true_levels = np.asarray(true_levels, dtype=float)
    reflected = np.max(levels) - levels[::-1]

    def rms(est):
        return float(np.sqrt(np.mean((np.sort(est) - np.sort(true_levels)) ** 2)))

    rms_stored, rms_reflected = rms(levels), rms(reflected)

    # The stored one is the better fit by construction; name them so the
    # figure reads the same way the diagnostics printout does.
    if rms_stored <= rms_reflected:
        branches = {
            "forward": {"levels": levels, "rms": rms_stored},
            "reversed": {"levels": reflected, "rms": rms_reflected},
        }
        best = "forward"
    else:
        branches = {
            "forward": {"levels": reflected, "rms": rms_reflected},
            "reversed": {"levels": levels, "rms": rms_stored},
        }
        best = "reversed"
    return branches, best


# --------------------------------------------------------------------------
# Panel data
# --------------------------------------------------------------------------
def panel_series(panel: str, run: dict, cfg_panel: dict) -> list[dict]:
    """The (true, estimated) series a panel draws.

    Most panels give one series. `energies` gives two when both turnpike
    branches are requested.
    """
    data = run["data"]

    if panel == "transitions":
        true_v, est_v = _need(data, ("matched_true", "matched_detected"), panel)
        return [{"key": "transitions", "true": true_v, "est": est_v,
                 "label": cfg_panel.get("label")}]

    if panel == "energies":
        est_v, true_v = _need(
            data, ("experimental_energies", "shifted_true_energies"), panel)
        branches, best = turnpike_branches(est_v, true_v)

        which = cfg_panel.get("branches", "both")
        if which == "both":
            keys = ["forward", "reversed"]
        elif which in ("best", "forward", "reversed"):
            keys = [best if which == "best" else which]
        else:
            raise ValueError(
                f"panels.energies.branches must be 'both', 'best', 'forward' "
                f"or 'reversed', got {which!r}")

        labels = cfg_panel.get("branch_labels", {}) or {}
        default_labels = {"forward": "forward", "reversed": "reversed"}
        out = []
        for k in keys:
            label = labels.get(k, default_labels[k])
            if cfg_panel.get("mark_best", True) and k == best and len(keys) > 1:
                label = f"{label} (best)"
            out.append({"key": k, "true": true_v, "est": branches[k]["levels"],
                        "label": label, "rms": branches[k]["rms"],
                        "is_best": k == best})
        return out

    if panel == "amplitudes":
        true_v, est_v = _need(data, ("u_exact_amplitudes", "u_m"), panel)
        return [{"key": "amplitudes", "true": true_v, "est": est_v,
                 "label": cfg_panel.get("label")}]

    if panel == "phases":
        true_v, est_v = _need(data, ("u_exact_phases_rel", "phi_m"), panel)
        return [{"key": "phases", "true": true_v, "est": est_v,
                 "label": cfg_panel.get("label")}]

    if panel == "rates":
        if "exact_lambdas" not in data:
            raise DiagnosticsError(
                "panel 'rates' needs exact_lambdas, which is only stored when "
                "the run was analysed with noise enabled (w_noise/r_noise). "
                "Drop 'rates' from panels for a noiseless run.")
        est_v, true_v = _need(data, ("lambdas", "exact_lambdas"), panel)
        return [{"key": "rates", "true": true_v, "est": est_v,
                 "label": cfg_panel.get("label")}]

    raise ValueError(f"unknown panel: {panel!r} "
                     f"(expected one of {', '.join(PANELS)})")


def sorted_pairs(series: dict, sort: bool = True):
    """The (x, y) a panel plots.

    Estimated and true values come out of the analysis in different orders --
    the diagnostics compare them sorted, so by default this does too. Set
    `sort: false` on the panel to keep the stored pairing instead, which is
    the right choice for `transitions`, where the matching already paired
    them up.
    """
    true_v = np.asarray(series["true"], dtype=float)
    est_v = np.asarray(series["est"], dtype=float)
    if true_v.size != est_v.size:
        raise DiagnosticsError(
            f"panel series '{series['key']}' has {true_v.size} true values "
            f"but {est_v.size} estimated ones")
    if sort:
        return np.sort(true_v), np.sort(est_v)
    return true_v, est_v


# --------------------------------------------------------------------------
# rate_levels panel
# --------------------------------------------------------------------------
def rate_level_data(run: dict, cfg_panel: dict) -> dict:
    """Fitted decay rates plus the predicted levels to mark with dashed lines.

    `predicted` selects which prediction the lines come from:

        exact     the per-pair rates Gamma_mn from the T1/T2 structural
                  formula -- these are what the fitted rates should sit on
                  (the default);
        possible  the dephasing-only subset sums, i.e. the upper-bound set
                  `calculate_dephasing_rates_upper_bound` returns. This is
                  what the pipeline's own figure draws, but it ignores T1, so
                  the lines will not line up with the markers when amplitude
                  damping is present;
        none      no lines, just the markers.

    Only the distinct predicted values are drawn: `exact_lambdas` holds one
    entry per transition pair, and many pairs share a rate, so drawing all of
    them would stack identical lines on top of each other.
    """
    data = run["data"]
    if "lambdas" not in data:
        raise DiagnosticsError(
            "panel 'rate_levels' needs the fitted 'lambdas', which this run's "
            "diagnostics archive does not contain")
    lambdas = np.asarray(data["lambdas"], dtype=float)

    which = str(cfg_panel.get("predicted", "exact")).lower()
    predicted = None
    if which == "exact":
        if "exact_lambdas" not in data:
            raise DiagnosticsError(
                "panel 'rate_levels' with predicted: exact needs "
                "exact_lambdas, which is only stored when the run was "
                "analysed with noise enabled. Use predicted: none, or "
                "analyse a noisy run.")
        predicted = np.asarray(data["exact_lambdas"], dtype=float)
    elif which == "possible":
        if "possible_lambdas" not in data:
            raise DiagnosticsError(
                "panel 'rate_levels' with predicted: possible needs "
                "possible_lambdas, which this run does not store")
        predicted = np.asarray(data["possible_lambdas"], dtype=float)
    elif which not in ("none", "off"):
        raise ValueError(
            f"panels.rate_levels.predicted must be 'exact', 'possible' or "
            f"'none', got {which!r}")

    if predicted is not None:
        # Collapse duplicates so each distinct level is one line.
        tol = float(cfg_panel.get("level_tolerance", 1e-9))
        decimals = max(0, int(round(-np.log10(tol)))) if tol > 0 else 12
        predicted = np.unique(np.round(predicted, decimals))

    # x axis: mode index, or the mode frequency when asked for and available.
    x_mode = str(cfg_panel.get("x", "index")).lower()
    if x_mode == "frequency":
        if "freqs" not in data:
            raise DiagnosticsError(
                "panel 'rate_levels' with x: frequency needs 'freqs', which "
                "this run's archive does not contain")
        x = np.asarray(data["freqs"], dtype=float)
        default_xlabel = r"$\omega_{mn} / 2\pi$"
    elif x_mode == "index":
        x = np.arange(lambdas.size, dtype=float)
        default_xlabel = "mode index"
    else:
        raise ValueError(
            f"panels.rate_levels.x must be 'index' or 'frequency', "
            f"got {x_mode!r}")

    order = cfg_panel.get("sort_by")
    if order == "rate":
        # Sorting by fitted rate makes the grouping onto the predicted
        # levels obvious: the markers form visible plateaus.
        idx = np.argsort(lambdas)
        lambdas = lambdas[idx]
        if x_mode == "frequency":
            x = x[idx]
    elif order not in (None, "none", "x"):
        raise ValueError(
            f"panels.rate_levels.sort_by must be 'rate', 'x' or null, "
            f"got {order!r}")

    return {
        "x": x,
        "lambdas": lambdas,
        "predicted": predicted,
        "x_mode": x_mode,
        "default_xlabel": default_xlabel,
        "predicted_source": which,
    }


# --------------------------------------------------------------------------
# noise_params panel
# --------------------------------------------------------------------------
def noise_param_data(run: dict, cfg_panel: dict) -> dict:
    """Recovered per-qubit noise parameters against the configured ones.

    Prefers the `noise_extraction.npz` that script 2 writes, since that was
    solved where the eigenvectors were available. Falls back to inverting
    the stored rates here, which needs the Hamiltonian from the run's
    metadata to put the levels in the right order.
    """
    import sys
    from pathlib import Path as _Path

    run_dir = run["run_dir"]
    # Re-running script 2 never overwrites, so a corrected extraction lands
    # beside the original as _2, _3, ... Take the most recently written one.
    candidates = sorted(run_dir.glob("noise_extraction*.npz"),
                        key=lambda f: f.stat().st_mtime)

    kphi = kt1 = None
    source = None
    if candidates:
        npz = candidates[-1]
        with np.load(npz, allow_pickle=True) as archive:
            kphi = np.asarray(archive["kappa_phi"], dtype=float)
            kt1 = np.asarray(archive["kappa_T1"], dtype=float)
        source = npz.name
    else:
        # Invert here. The extraction module lives with the pipeline, so it
        # is imported by path rather than duplicated.
        pipeline = _Path(__file__).resolve().parents[2] / "refactored_code"
        if str(pipeline) not in sys.path:
            sys.path.insert(0, str(pipeline))
        try:
            from lzs_pipeline.noise_extraction import extract_noise_parameters
        except ImportError as exc:
            raise DiagnosticsError(
                f"panel 'noise_params' needs either a noise_extraction.npz in "
                f"{run_dir} (re-run script 2) or the refactored_code package "
                f"on the path: {exc}")

        data = run["data"]
        for key in ("lambdas", "freqs", "experimental_energies"):
            if key not in data:
                raise DiagnosticsError(
                    f"panel 'noise_params' needs '{key}' to invert the rates, "
                    f"which this run's archive does not contain")
        ham = run["hamiltonian"]
        if not ham.get("local_z"):
            raise DiagnosticsError(
                "panel 'noise_params' needs the Hamiltonian from the run's "
                "metadata.yaml to order the levels; none was found")
        n_qubits = len(ham["local_z"])
        extracted = extract_noise_parameters(
            data["lambdas"], data["freqs"], data["experimental_energies"],
            n_qubits,
            local_z=ham["local_z"], two_body=ham.get("two_body", []),
            local_x=ham.get("local_x"),
            lmd_mn=data.get("lmd_mn"))
        kphi = extracted["kappa_phi"]
        kt1 = extracted["kappa_T1"]
        source = "re-inverted here"

    # The configured values, if the run recorded them -- but only if the
    # run actually *applied* them. A metadata block can carry t1/t2 rates
    # while r_noise and w_noise are both false, in which case the signal is
    # unitary: showing those rates as "configured" would suggest the
    # extraction failed when there was simply no decoherence to find.
    dec = (run["metadata"].get("common", {}) or {}).get("decoherence", {}) or {}
    noise_applied = bool(dec.get("r_noise", False) or dec.get("w_noise", False))
    if dec and not noise_applied:
        raise DiagnosticsError(
            "panel 'noise_params': this run was simulated without "
            "decoherence (r_noise and w_noise are both false), so the fitted "
            "decay rates carry no per-qubit noise to invert. The t1/t2 rates "
            "in its metadata were never applied. Use a run with noise "
            "enabled.")
    true_phi = dec.get("t2_rates")
    true_t1 = dec.get("t1_rates")
    true_phi = np.asarray(true_phi, dtype=float) if true_phi is not None else None
    true_t1 = np.asarray(true_t1, dtype=float) if true_t1 is not None else None

    quantity = str(cfg_panel.get("quantity", "rates")).lower()
    if quantity == "times":
        def invert(v):
            if v is None:
                return None
            v = np.asarray(v, dtype=float)
            with np.errstate(divide="ignore"):
                return np.where(v > 0, 1.0 / np.where(v > 0, v, 1.0), np.inf)
        kphi, kt1 = invert(kphi), invert(kt1)
        true_phi, true_t1 = invert(true_phi), invert(true_t1)
        default_ylabel = "time"
    elif quantity == "rates":
        default_ylabel = "rate"
    else:
        raise ValueError(
            f"panels.noise_params.quantity must be 'rates' or 'times', "
            f"got {quantity!r}")

    channels = cfg_panel.get("channels") or ["dephasing", "damping"]
    for c in channels:
        if c not in ("dephasing", "damping"):
            raise ValueError(
                f"panels.noise_params.channels entries must be 'dephasing' "
                f"or 'damping', got {c!r}")

    return {
        "kappa_phi": kphi,
        "kappa_T1": kt1,
        "true_phi": true_phi,
        "true_T1": true_t1,
        "channels": channels,
        "quantity": quantity,
        "default_ylabel": default_ylabel,
        "source": source,
        "n_qubits": int(np.asarray(kphi).size),
    }
