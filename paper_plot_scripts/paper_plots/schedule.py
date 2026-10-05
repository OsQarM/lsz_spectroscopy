"""
Building and drawing the LZS ramp schedule for the paper figure.

The schedules come from the real `LSZ_experiment` -- the same trapezoid,
asymmetry and noise code the simulations use -- so the figure shows the
protocol as it is actually applied, not an idealised redrawing of it.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml


# --------------------------------------------------------------------------
# Protocol
# --------------------------------------------------------------------------
def _as_coeffs(value, n, name):
    if value is None:
        return [1.0] * n
    arr = [float(v) for v in value]
    if len(arr) != n:
        raise ValueError(f"{name} has {len(arr)} entries but there are "
                         f"{n} qubits")
    return arr


def _load_run_protocol(run_dir: Path) -> dict:
    """Ramp/wait/asymmetry settings recorded in a run's metadata.yaml."""
    meta_path = run_dir / "metadata.yaml"
    if not meta_path.is_file():
        raise FileNotFoundError(
            f"{run_dir} has no metadata.yaml -- give the protocol explicitly "
            f"in the config instead of using from_run")
    with open(meta_path) as fh:
        meta = yaml.safe_load(fh) or {}

    common = meta.get("common", {}) or {}
    proto = common.get("protocol", {}) or {}
    ramps = common.get("ramps", {}) or {}
    ham = common.get("hamiltonian", {}) or {}

    out = {}
    if ham.get("local_z"):
        out["n_qubits"] = len(ham["local_z"])
    if proto.get("ramp_time") is not None:
        out["ramp_time"] = float(proto["ramp_time"])
    for key in ("up_z_assymetry_coefficients", "down_z_assymetry_coefficients",
                "up_x_assymetry_coefficients", "down_x_assymetry_coefficients",
                "average_ramp_error", "average_wait_error",
                "average_ramp_bias"):
        if ramps.get(key) is not None:
            out[key] = ramps[key]
    return out


def resolve_protocol(cfg_protocol: dict, from_run, base_dir: Path):
    """Merge the config's protocol block with a run's metadata, if given.

    An explicit value in the config always wins, so a run can supply the
    asymmetries while the figure still uses a hand-picked wait time.
    """
    merged = dict(cfg_protocol)
    provenance = "config"

    if from_run:
        run_dir = Path(from_run).expanduser()
        if not run_dir.is_absolute():
            run_dir = (base_dir / run_dir).resolve()
        from_meta = _load_run_protocol(run_dir)
        # The config's own non-null entries take precedence over the run's.
        for key, value in from_meta.items():
            if cfg_protocol.get(key) is None:
                merged[key] = value
        provenance = f"run metadata ({run_dir.name}) + config overrides"

    n_qubits = merged.get("n_qubits")
    n_qubits = 2 if n_qubits is None else int(n_qubits)
    if n_qubits < 1:
        raise ValueError("protocol.n_qubits must be at least 1")

    # A run's asymmetry lists are sized by *its* qubit count. If the config
    # also narrows n_qubits, those lists no longer fit, so drop the inherited
    # ones rather than failing on a mismatch the user did not write.
    if from_run:
        for key in ("up_z_assymetry_coefficients",
                    "down_z_assymetry_coefficients",
                    "up_x_assymetry_coefficients",
                    "down_x_assymetry_coefficients"):
            value = merged.get(key)
            if (cfg_protocol.get(key) is None and value is not None
                    and len(value) != n_qubits):
                merged[key] = None

    # `or` would turn an explicit 0 into the default, hiding a bad value.
    ramp_time = merged.get("ramp_time")
    ramp_time = 1.0 if ramp_time is None else float(ramp_time)
    wait_time = merged.get("wait_time")
    wait_time = 0.0 if wait_time is None else float(wait_time)
    if ramp_time <= 0:
        raise ValueError("protocol.ramp_time must be positive")
    if wait_time < 0:
        raise ValueError("protocol.wait_time cannot be negative")

    n_steps = int(merged.get("n_steps", 1001))
    if n_steps < 2:
        raise ValueError("protocol.n_steps must be at least 2")

    return {
        "n_qubits": n_qubits,
        "ramp_time": ramp_time,
        "wait_time": wait_time,
        "n_steps": n_steps,
        "tr": ramp_time,
        "tf": 2 * ramp_time + wait_time,
        "up_z": _as_coeffs(merged.get("up_z_assymetry_coefficients"),
                           n_qubits, "up_z_assymetry_coefficients"),
        "down_z": _as_coeffs(merged.get("down_z_assymetry_coefficients"),
                             n_qubits, "down_z_assymetry_coefficients"),
        "up_x": _as_coeffs(merged.get("up_x_assymetry_coefficients"),
                           n_qubits, "up_x_assymetry_coefficients"),
        "down_x": _as_coeffs(merged.get("down_x_assymetry_coefficients"),
                             n_qubits, "down_x_assymetry_coefficients"),
        "ramp_error": float(merged.get("average_ramp_error", 0.0) or 0.0),
        "wait_error": float(merged.get("average_wait_error", 0.0) or 0.0),
        "ramp_bias": float(merged.get("average_ramp_bias", 0.0) or 0.0),
        "noise_seed": merged.get("noise_seed", 0),
    }, provenance


# --------------------------------------------------------------------------
# Schedules
# --------------------------------------------------------------------------
def _select_qubits(spec, n_qubits, family):
    """Which qubit indices to draw for the z or x family."""
    wanted = spec.get("qubits")
    if wanted is None:
        return list(range(n_qubits))
    if np.isscalar(wanted):
        wanted = [wanted]
    out = []
    for q in wanted:
        q = int(q)
        if not 0 <= q < n_qubits:
            raise ValueError(
                f"lines.{family}.qubits lists qubit {q}, which is out of "
                f"range for {n_qubits} qubits")
        out.append(q)
    return out


def build_schedules(protocol: dict, cfg_lines: dict):
    """Evaluate every requested schedule on a common time grid.

    Returns `(t, curves)`, each curve a dict with `key`, `label`, `values`
    and the per-family style options.
    """
    from lsz_experiment import LSZ_experiment, set_global_seed

    n = protocol["n_qubits"]

    # A placeholder Hamiltonian: the schedule depends only on the timing and
    # asymmetries, not on the couplings, but LSZ_experiment needs the dict.
    h_target = {
        "local_z": np.ones(n),
        "local_x": np.zeros(n),
        "two_body": np.zeros(n * (n - 1) // 2),
    }

    if protocol["noise_seed"] is not None:
        set_global_seed(protocol["noise_seed"])

    exp = LSZ_experiment(
        n, 1.0, h_target, protocol["ramp_time"], protocol["wait_time"], 1.0,
        up_z_assymetry_factors=protocol["up_z"],
        down_z_assymetry_factors=protocol["down_z"],
        up_x_assymetry_factors=protocol["up_x"],
        down_x_assymetry_factors=protocol["down_x"],
        ramp_error=protocol["ramp_error"],
        wait_error=protocol["wait_error"],
        ramp_bias=protocol["ramp_bias"],
    )

    t = np.linspace(exp.to, exp.tf, protocol["n_steps"])
    common = np.array([exp.trapezoid(float(ti)) for ti in t])

    curves = []

    spec = cfg_lines.get("common", {}) or {}
    if spec.get("show", True):
        curves.append({"key": "common", "qubit": None,
                       "label": spec.get("label", "common"),
                       "values": common, "spec": spec})

    spec = cfg_lines.get("h0", {}) or {}
    if spec.get("show", False):
        # H0 carries the complementary weight (1 - s).
        curves.append({"key": "h0", "qubit": None,
                       "label": spec.get("label", r"$H_0$"),
                       "values": 1.0 - common, "spec": spec})

    for family, sched_fn in (("z", exp.z_schedule), ("x", exp.x_schedule)):
        spec = cfg_lines.get(family, {}) or {}
        if not spec.get("show", True):
            continue
        template = spec.get("label") or f"$H_{family}[{{i}}]$"
        for q in _select_qubits(spec, n, family):
            values = np.array([sched_fn(float(ti), q) for ti in t])
            try:
                label = template.format(i=q, q=q, n=q + 1)
            except (KeyError, IndexError):
                # A label with no placeholder, or braces meant literally.
                label = template
            curves.append({"key": f"{family}{q}", "qubit": q, "family": family,
                           "label": label, "values": values, "spec": spec})

    return t, curves


# --------------------------------------------------------------------------
# Drawing
# --------------------------------------------------------------------------
def _curve_colors(curves, cfg_style, style):
    """A colour per curve: explicit per-family, then global, then the cycle."""
    explicit_global = cfg_style.get("colors")
    colors = []
    cycle = list(style.CYCLE)
    auto_i = 0

    for c in curves:
        spec = c["spec"]
        chosen = None
        if c["key"] in ("common", "h0"):
            chosen = spec.get("color")
        else:
            per_family = spec.get("colors")
            if per_family:
                # Index within this family, so z and x can be coloured apart.
                same = [d for d in curves if d.get("family") == c.get("family")]
                idx = same.index(c)
                chosen = per_family[idx % len(per_family)]
        if chosen is None and explicit_global:
            chosen = explicit_global[auto_i % len(explicit_global)]
        if chosen is None:
            chosen = cycle[auto_i % len(cycle)]
        colors.append(chosen)
        auto_i += 1
    return colors


def draw_curves(ax, t, curves, cfg_style: dict, style):
    colors = _curve_colors(curves, cfg_style, style)
    base_lw = cfg_style.get("linewidth") or style.LINEWIDTH
    base_alpha = float(cfg_style.get("alpha", 1.0))

    for c, color in zip(curves, colors):
        spec = c["spec"]
        lw = float(spec.get("linewidth") or base_lw)
        ls = spec.get("linestyle") or "-"
        alpha = float(spec.get("alpha", base_alpha))
        ax.plot(t, c["values"], color=color, lw=lw, linestyle=ls,
                alpha=alpha, label=c["label"], zorder=3)
    return colors


def draw_markers(ax, protocol: dict, cfg_markers: dict, style):
    """Vertical lines at the segment boundaries.

    Returns the positions, their names and their labels so the x ticks and
    the saved data file can use them.
    """
    info = {"positions": [], "names": [], "labels": []}
    if not cfg_markers.get("show", True):
        return info

    tr = protocol["tr"]
    tw_end = protocol["tr"] + protocol["wait_time"]
    tf = protocol["tf"]
    labels = cfg_markers.get("labels", {}) or {}

    candidates = [
        ("start", 0.0, cfg_markers.get("start", False)),
        ("end_ramp1", tr, cfg_markers.get("end_ramp1", True)),
        ("end_wait", tw_end, cfg_markers.get("end_wait", True)),
        ("end_ramp2", tf, cfg_markers.get("end_ramp2", True)),
    ]

    color = cfg_markers.get("color") or style.GUIDE

    # A boundary can carry a tick label without a vertical line. `lines`
    # lists the boundaries that get a line; by default that is every enabled
    # one, so a plain `start: true` behaves as before. Naming a boundary in
    # `no_line` (or listing only some in `lines`) gives it just the label --
    # which is what you want at t = 0, where a line would sit on the spine.
    line_only_for = cfg_markers.get("lines")
    no_line = set(cfg_markers.get("no_line") or [])

    def wants_line(name):
        if name in no_line:
            return False
        if line_only_for is None:
            return True
        return name in line_only_for

    for name, pos, enabled in candidates:
        if not enabled:
            continue
        # With no wait time the end of ramp 1 and the end of the wait
        # coincide; drawing both would stack two lines on one position.
        if info["positions"] and np.isclose(pos, info["positions"][-1]):
            continue
        if wants_line(name):
            ax.axvline(pos, color=color,
                       lw=float(cfg_markers.get("linewidth", 0.9)),
                       linestyle=cfg_markers.get("linestyle", ":"),
                       alpha=float(cfg_markers.get("alpha", 0.8)), zorder=1)
        info["positions"].append(float(pos))
        info["names"].append(name)
        info["labels"].append(labels.get(name, f"{pos:g}"))

    if cfg_markers.get("shade_wait", False) and protocol["wait_time"] > 0:
        ax.axvspan(tr, tw_end,
                   color=cfg_markers.get("shade_color") or style.SUPPORT,
                   alpha=float(cfg_markers.get("shade_alpha", 0.08)),
                   lw=0, zorder=0)

    if cfg_markers.get("annotate_segments", False):
        seg_labels = cfg_markers.get("segment_labels", {}) or {}
        spans = [("ramp1", 0.0, tr), ("wait", tr, tw_end), ("ramp2", tw_end, tf)]
        for key, lo, hi in spans:
            if hi <= lo:
                continue
            text = seg_labels.get(key)
            if not text:
                continue
            ax.annotate(text, xy=(0.5 * (lo + hi), 1.0),
                        xycoords=("data", "axes fraction"),
                        xytext=(0, 4), textcoords="offset points",
                        ha="center", va="bottom",
                        fontsize=float(cfg_markers.get("segment_fontsize", 10)))

    return info


def finish_axes(ax, t, curves, cfg_axes: dict, cfg_legend: dict,
                marker_info: dict, cfg_markers: dict, style):
    xlim = cfg_axes.get("xlim")
    if xlim:
        if len(xlim) != 2:
            raise ValueError("axes.xlim must be [min, max]")
        ax.set_xlim(float(xlim[0]), float(xlim[1]))
    else:
        # Tight against the spines by default, so the curve starts exactly
        # at t = 0 with no white gap. Set axes.x_margin > 0 only if you want
        # a vertical marker at t = 0 or t = tf to clear the spine.
        span = float(t[-1]) - float(t[0])
        pad = float(cfg_axes.get("x_margin", 0.0) or 0.0) * (span if span > 0 else 1.0)
        ax.set_xlim(float(t[0]) - pad, float(t[-1]) + pad)

    ylim = cfg_axes.get("ylim")
    if ylim:
        if len(ylim) != 2:
            raise ValueError("axes.ylim must be [min, max]")
        ax.set_ylim(float(ylim[0]), float(ylim[1]))
    else:
        all_v = np.concatenate([c["values"] for c in curves])
        lo, hi = float(all_v.min()), float(all_v.max())
        margin = float(cfg_axes.get("y_margin", 0.08)) * max(hi - lo, 1e-12)
        ax.set_ylim(lo - margin, hi + margin)

    if cfg_axes.get("grid"):
        ax.grid(True, which="major", lw=0.4, alpha=0.3)

    style.style_axis(ax,
                     xlabel=cfg_axes.get("xlabel"),
                     ylabel=cfg_axes.get("ylabel"),
                     title=cfg_axes.get("title"))

    # Replace the numeric ticks with the named boundaries -- this is what
    # makes the figure read as a protocol diagram rather than a plot. Done
    # after style_axis, which sets the tick label sizes.
    apply_marker_xticks(ax, marker_info, cfg_markers)

    if cfg_legend.get("show", True):
        ax.legend(frameon=bool(cfg_legend.get("frameon", False)),
                  fontsize=cfg_legend.get("fontsize") or style.FONT_LEGEND,
                  loc=cfg_legend.get("loc", "best"),
                  ncol=int(cfg_legend.get("ncol", 1)),
                  title=cfg_legend.get("title"))
    return ax


def apply_marker_xticks(ax, marker_info: dict, cfg_markers: dict):
    """Put the boundary names on the x axis in place of numeric ticks."""
    if not cfg_markers.get("show", True):
        return
    if not cfg_markers.get("replace_xticks", True):
        return
    if not marker_info["positions"]:
        return
    ax.set_xticks(marker_info["positions"])
    ax.set_xticklabels(marker_info["labels"])
