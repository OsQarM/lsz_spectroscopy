#!/usr/bin/env python
"""
Paper figure: the diagnostic estimate-vs-true panels.

Draws any combination of five panels from a run analysed by
`refactored_code/02_run_diagnostics.py`:

    transitions   detected vs. true transition energies
    energies      reconstructed vs. true energy levels
    amplitudes    estimated vs. exact state-vector moduli |u_m|
    phases        estimated vs. exact phases phi_m
    rates         fitted vs. predicted coherence decay rates

Each is a scatter against the y = x reference. For `energies` both solutions
of the turnpike problem are drawn together: a turnpike reconstruction is
determined only up to reflection, so the forward and reversed branches are
both valid solutions of the difference set, and showing both makes clear
which one the true spectrum selects.

Panels can go in one figure (a row, a grid) or be written as separate files.
Axes and legend are configurable globally and per panel.

Usage:
    python plot_diagnostics.py [--config diagnostics_config.yaml]
                               [--source DIR] [--panels energies,phases]
                               [--separate] [--show]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from paper_plots import diagnostics as pdiag  # noqa: E402
from paper_plots import figure as pfig  # noqa: E402


DEFAULTS = {
    "source": None,
    "panels": ["transitions", "energies", "amplitudes", "phases"],

    # Per-panel overrides, keyed by panel name. Anything omitted falls back
    # to `common_panel` and then to the panel's own sensible default.
    "panel_options": {},

    # Applied to every panel unless the panel overrides it.
    "common_panel": {
        "xlabel": None,          # null -> the panel's default label
        "ylabel": None,
        "title": None,
        "xlim": None,
        "ylim": None,
        "equal_limits": True,    # same range on both axes (an identity plot)
        "sort": True,            # compare sorted estimates against sorted truth
        "grid": False,
        "marker": None,          # null -> the style's marker cycle
        "markersize": None,
        "color": None,
        "alpha": 1.0,
        "identity": True,        # the y = x reference line
        "identity_label": None,  # e.g. "estimate = true"; null -> no entry
        "annotate_rms": False,   # print the RMS error in the corner
        "rms_fontsize": 10,

        # --- rate_levels only ---
        "predicted": "exact",      # exact | possible | none
        "x": "index",              # index | frequency
        "sort_by": "rate",         # rate | x | null
        "level_tolerance": 1.0e-9,
        "level_color": None,       # null -> the palette's reference colour
        "level_linestyle": "--",
        "level_linewidth": None,
        "level_alpha": 0.9,
        "level_label": "predicted",
        "marker_label": None,      # e.g. "fitted" to put markers in the legend
    },

    "legend": {
        "show": True,
        "loc": "best",
        "frameon": False,
        "fontsize": None,
        "ncol": 1,
        "title": None,
        "only_when_multiple": True,   # skip a legend for a lone unlabelled series
    },

    "style": {
        "palette": "sober",
        "colors": None,
        "markers": None,
        "linewidth": None,
        "alpha": 1.0,
        "font_sizes": {},
        "rcparams": {},
    },

    "layout": {
        "separate": False,       # true -> one file per panel
        "ncols": None,           # null -> all panels in a single row
        "panel_size": None,      # [w, h] per panel; null -> the style's square
        "panel_letters": False,  # label panels (a), (b), ...
        "letter_fontsize": 13,
    },

    "figure": {
        "preset": "single",
        "size": None,
    },

    "output": {
        "directory": "paper_plot_images",
        "name": "diagnostics",
        "timestamp": False,
        "formats": ["pdf", "png"],
        "dpi": 500,
        "transparent": False,
        "save_data": True,
        "save_csv": False,
    },
}


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=None,
                   help="config file (default: diagnostics_config.yaml "
                        "beside this script)")
    p.add_argument("--source", default=None,
                   help="the analysed run directory (overrides the config)")
    p.add_argument("--panels", default=None,
                   help="comma-separated panels to draw, e.g. energies,phases")
    p.add_argument("--separate", action="store_true",
                   help="write one file per panel")
    p.add_argument("--name", default=None, help="override output.name")
    p.add_argument("--no-legend", action="store_true", help="omit legends")
    p.add_argument("--show", action="store_true",
                   help="also open the figure in a window")
    return p.parse_args(argv)


def panel_config(panel: str, cfg: dict) -> dict:
    """The effective options for one panel: defaults < common < per-panel."""
    merged = dict(cfg["common_panel"])
    merged.update(
        {k: v for k, v in (cfg["panel_options"].get(panel) or {}).items()})
    for key in ("xlabel", "ylabel"):
        if merged.get(key) is None:
            merged[key] = pdiag.PANEL_DEFAULT_LABELS[panel][key]
    return merged


def draw_rate_levels(ax, run: dict, cfg: dict, style):
    """Fitted decay rates as markers, predicted rates as dashed h-lines.

    Mirrors `plotting.diagnostic_plots.plot_decay_rates`, which is what the
    pipeline's own dephasing figure uses.
    """
    opts = panel_config("rate_levels", cfg)
    cfg_style = cfg["style"]
    info = pdiag.rate_level_data(run, opts)

    x, lambdas = info["x"], info["lambdas"]
    predicted = info["predicted"]

    colors = cfg_style.get("colors")
    markers = cfg_style.get("markers") or list(style.MARKERS)
    color = opts.get("color") or (colors[0] if colors else style.DATA)
    marker = opts.get("marker") or markers[0]
    ms = float(opts.get("markersize") or style.MARKERSIZE)

    ax.plot(x, lambdas, marker, color=color, linestyle="none",
            markersize=ms, alpha=float(opts.get("alpha", 1.0)),
            label=opts.get("marker_label"), zorder=3)

    if predicted is not None and len(predicted):
        lvl_color = opts.get("level_color") or style.REFERENCE
        lvl_lw = float(opts.get("level_linewidth") or style.REF_LINEWIDTH)
        for i, rate in enumerate(np.atleast_1d(predicted)):
            ax.axhline(rate, color=lvl_color,
                       linestyle=opts.get("level_linestyle", "--"),
                       lw=lvl_lw, alpha=float(opts.get("level_alpha", 0.9)),
                       zorder=1,
                       label=opts.get("level_label") if i == 0 else None)

    xlim, ylim = opts.get("xlim"), opts.get("ylim")
    if xlim:
        ax.set_xlim(float(xlim[0]), float(xlim[1]))
    if ylim:
        ax.set_ylim(float(ylim[0]), float(ylim[1]))

    if opts.get("grid"):
        ax.grid(True, which="major", lw=0.4, alpha=0.3)

    xlabel = opts.get("xlabel")
    # The default label depends on what the x axis actually is.
    if xlabel == pdiag.PANEL_DEFAULT_LABELS["rate_levels"]["xlabel"]:
        xlabel = info["default_xlabel"]
    style.style_axis(ax, xlabel=xlabel, ylabel=opts.get("ylabel"),
                     title=opts.get("title"))

    cfg_legend = cfg["legend"]
    if cfg_legend.get("show", True):
        handles, labels = ax.get_legend_handles_labels()
        if labels:
            ax.legend(frameon=bool(cfg_legend.get("frameon", False)),
                      fontsize=cfg_legend.get("fontsize") or style.FONT_LEGEND,
                      loc=cfg_legend.get("loc", "best"),
                      ncol=int(cfg_legend.get("ncol", 1)),
                      title=cfg_legend.get("title"))

    # Reported like the scatter panels so the printout and the saved data
    # stay uniform: how far each fitted rate sits from its nearest level.
    if predicted is not None and len(predicted):
        nearest = predicted[np.abs(lambdas[:, None] - predicted[None, :]).argmin(axis=1)]
        rms = float(np.sqrt(np.mean((lambdas - nearest) ** 2)))
    else:
        nearest, rms = np.full_like(lambdas, np.nan), float("nan")

    return [{
        "key": "rate_levels",
        "label": opts.get("marker_label") or "fitted rates",
        "x": x,
        "y": lambdas,
        "predicted": predicted if predicted is not None else np.array([]),
        "nearest": nearest,
        "rms_plotted": rms,
        "color": color,
        "marker": marker,
        "predicted_source": info["predicted_source"],
        "x_mode": info["x_mode"],
    }]


def draw_panel(ax, panel: str, run: dict, cfg: dict, style):
    """Draw one estimate-vs-true panel; returns the series it plotted."""
    if panel == "rate_levels":
        return draw_rate_levels(ax, run, cfg, style)

    opts = panel_config(panel, cfg)
    cfg_style = cfg["style"]
    series = pdiag.panel_series(panel, run, opts)

    explicit_colors = cfg_style.get("colors")
    markers = cfg_style.get("markers") or list(style.MARKERS)

    drawn = []
    all_values = []

    for i, s in enumerate(series):
        x, y = pdiag.sorted_pairs(s, sort=bool(opts.get("sort", True)))
        all_values.append(np.concatenate([x, y]))

        if opts.get("color"):
            color = opts["color"]
        elif explicit_colors:
            color = explicit_colors[i % len(explicit_colors)]
        elif len(series) == 1:
            color = style.DATA
        else:
            # Several branches in one panel: the best fit stays the primary
            # colour so the eye lands on it first.
            color = style.DATA if s.get("is_best") else style.SECONDARY

        marker = opts.get("marker") or markers[i % len(markers)]
        ms = float(opts.get("markersize") or style.MARKERSIZE)

        ax.scatter(x, y, color=color, marker=marker, s=ms ** 2,
                   alpha=float(opts.get("alpha", 1.0)),
                   label=s.get("label"), zorder=3)

        rms = float(np.sqrt(np.mean((y - x) ** 2)))
        drawn.append({**s, "x": x, "y": y, "rms_plotted": rms,
                      "color": color, "marker": marker})

    combined = np.concatenate(all_values)

    if opts.get("identity", True):
        style.identity_line(ax, combined,
                            label=opts.get("identity_label"))

    xlim, ylim = opts.get("xlim"), opts.get("ylim")
    if xlim:
        ax.set_xlim(float(xlim[0]), float(xlim[1]))
    if ylim:
        ax.set_ylim(float(ylim[0]), float(ylim[1]))
    if opts.get("equal_limits", True) and not (xlim or ylim):
        lo, hi = float(combined.min()), float(combined.max())
        pad = 0.06 * (hi - lo if hi > lo else max(abs(hi), 1.0))
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_ylim(lo - pad, hi + pad)

    if opts.get("grid"):
        ax.grid(True, which="major", lw=0.4, alpha=0.3)

    style.style_axis(ax, xlabel=opts.get("xlabel"), ylabel=opts.get("ylabel"),
                     title=opts.get("title"))

    if opts.get("annotate_rms", False):
        text = "\n".join(
            (f"{d['label']}: RMS {d['rms_plotted']:.3g}" if d.get("label")
             else f"RMS {d['rms_plotted']:.3g}")
            for d in drawn)
        ax.annotate(text, xy=(0.04, 0.96), xycoords="axes fraction",
                    va="top", ha="left",
                    fontsize=float(opts.get("rms_fontsize", 10)))

    cfg_legend = cfg["legend"]
    if cfg_legend.get("show", True):
        handles, labels = ax.get_legend_handles_labels()
        enough = (len(labels) > 1
                  or not cfg_legend.get("only_when_multiple", True))
        if labels and enough:
            ax.legend(frameon=bool(cfg_legend.get("frameon", False)),
                      fontsize=cfg_legend.get("fontsize") or style.FONT_LEGEND,
                      loc=cfg_legend.get("loc", "best"),
                      ncol=int(cfg_legend.get("ncol", 1)),
                      title=cfg_legend.get("title"))

    return drawn


def main(argv=None):
    args = parse_args(argv)

    script_dir = Path(__file__).resolve().parent
    config_path = (Path(args.config) if args.config
                   else script_dir / "diagnostics_config.yaml")
    cfg = pfig.load_config(config_path, DEFAULTS)

    if args.source:
        cfg["source"] = args.source
    if args.panels:
        cfg["panels"] = [p.strip() for p in args.panels.split(",") if p.strip()]
    if args.separate:
        cfg["layout"]["separate"] = True
    if args.name:
        cfg["output"]["name"] = args.name
    if args.no_legend:
        cfg["legend"]["show"] = False

    if not cfg.get("source"):
        raise SystemExit(
            "no source given: set `source:` in the config or pass --source")

    panels = list(cfg["panels"])
    if not panels:
        raise SystemExit("the config lists no panels to draw")
    unknown = [p for p in panels if p not in pdiag.PANELS]
    if unknown:
        raise SystemExit(
            f"unknown panel(s): {', '.join(unknown)}. "
            f"Available: {', '.join(pdiag.PANELS)}")

    import matplotlib
    if not args.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    style = pfig.apply_style(cfg["style"])

    print(f"Config: {config_path}")
    run = pdiag.load_run(cfg["source"], script_dir)
    print(f"Source: {run['npz_path']}")
    if run["n_qubits"]:
        print(f"Qubits: {run['n_qubits']}")
    print(f"Panels: {', '.join(panels)}\n")

    cfg_layout = cfg["layout"]
    panel_size = cfg_layout.get("panel_size")
    if panel_size:
        pw, ph = float(panel_size[0]), float(panel_size[1])
    else:
        pw, ph = pfig.figure_size(cfg["figure"], style)

    out_dir = pfig.output_dir(cfg["output"], script_dir)
    base_name = cfg["output"].get("name", "diagnostics")
    if cfg["output"].get("timestamp", False):
        base_name = pfig.timestamped(base_name)

    written, data_written = [], []
    all_drawn = {}

    if cfg_layout.get("separate", False):
        # One file per panel, each named after the panel.
        for panel in panels:
            fig, ax = plt.subplots(figsize=(pw, ph))
            drawn = draw_panel(ax, panel, run, cfg, style)
            all_drawn[panel] = drawn
            fig.tight_layout()

            name = f"{base_name}_{panel}"
            written += pfig.save_figure(fig, out_dir, name, cfg["output"])
            rms_text = ", ".join(f"{d['rms_plotted']:.4g}" for d in drawn)
            print(f"  {panel}: {len(drawn)} series, RMS {rms_text}")
            plt.close(fig)
    else:
        ncols = cfg_layout.get("ncols") or len(panels)
        ncols = max(1, min(int(ncols), len(panels)))
        nrows = int(np.ceil(len(panels) / ncols))

        fig, axes = plt.subplots(nrows, ncols,
                                 figsize=(pw * ncols, ph * nrows),
                                 squeeze=False)
        flat = axes.ravel()

        for i, panel in enumerate(panels):
            drawn = draw_panel(flat[i], panel, run, cfg, style)
            all_drawn[panel] = drawn
            rms_text = ", ".join(f"{d['rms_plotted']:.4g}" for d in drawn)
            print(f"  {panel}: {len(drawn)} series, RMS {rms_text}")
            if cfg_layout.get("panel_letters", False):
                flat[i].annotate(
                    f"({chr(ord('a') + i)})", xy=(0.0, 1.0),
                    xycoords="axes fraction", xytext=(2, 6),
                    textcoords="offset points", va="bottom", ha="left",
                    fontsize=float(cfg_layout.get("letter_fontsize", 13)))

        # Hide any unused cell in the grid.
        for j in range(len(panels), flat.size):
            flat[j].set_visible(False)

        fig.tight_layout()
        written += pfig.save_figure(fig, out_dir, base_name, cfg["output"])
        plt.close(fig)

    print("\nFigure written:")
    for p in written:
        print(f"  {p}")

    # ---------------- Data ----------------
    if cfg["output"].get("save_data", True):
        import datetime as _dt

        series_out = []
        for panel in panels:
            for d in all_drawn[panel]:
                label = d.get("label") or panel
                series_out.append({
                    "label": f"{panel}: {label}",
                    "freqs": d["x"],        # x: true values (or mode index)
                    "magnitude": d["y"],    # y: estimated values
                    # For rate_levels this carries the predicted levels the
                    # dashed lines sit at, as the other figures' 'peaks' do.
                    "peaks": np.asarray(d.get("predicted", []), dtype=float),
                })

        meta = {
            "figure": base_name,
            "config_file": str(config_path),
            "created": _dt.datetime.now().isoformat(timespec="seconds"),
            "note": ("x values ('freqs_i') are the true/reference quantity and "
                     "y values ('magnitude_i') the estimated one, so the "
                     "archive matches the other paper figures."),
            "source_run": str(run["run_dir"]),
            "source_npz": str(run["npz_path"]),
            "n_qubits": run["n_qubits"],
            "hamiltonian": pfig.to_plain(run["hamiltonian"]),
            "panels": panels,
            "layout": pfig.to_plain(cfg_layout),
            "series": [
                {
                    "index": i,
                    "panel": panel,
                    "key": d["key"],
                    "label": d.get("label"),
                    "n_points": int(len(d["x"])),
                    "rms": float(d["rms_plotted"]),
                    **({"predicted_source": d["predicted_source"],
                        "x_axis": d["x_mode"],
                        "n_predicted_levels": int(len(d["predicted"]))}
                       if d["key"] == "rate_levels" else {}),
                    **({"is_best_turnpike_branch": bool(d["is_best"])}
                       if "is_best" in d else {}),
                }
                for i, (panel, d) in enumerate(
                    (p, d) for p in panels for d in all_drawn[p])
            ],
        }
        data_written = pfig.save_plot_data(out_dir, base_name, series_out, meta)
        if cfg["output"].get("save_csv", False):
            data_written += pfig.save_series_csv(out_dir, base_name, series_out)
        print("Plot data written:")
        for p in data_written:
            print(f"  {p}")

    if args.show:
        plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
