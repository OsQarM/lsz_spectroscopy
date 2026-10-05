#!/usr/bin/env python
"""
Paper figure: the ramp schedule of an example run.

Draws the scheduling functions s(t) of the LZS protocol -- the common
(H0 + Hzz) envelope, the per-qubit local-Z schedules and the per-qubit
local-X schedules -- over the ramp-wait-ramp cycle.

Everything is configurable from `schedule_config.yaml`:

  * which families of lines are drawn (common / H0 / Z / X), and which
    qubits within each -- so a figure can show only H_z when that is all the
    point needs;
  * the ramp time, wait time and per-qubit ramp asymmetries, either taken
    from an existing run's metadata or given directly;
  * optional markers at the end of the first ramp, the end of the wait and
    the end of the second ramp, with configurable tick labels on the x axis.

The figure and a data file holding exactly what was drawn both go into the
configured output directory, the same way `plot_fft.py` does it.

Usage:
    python plot_schedule.py [--config schedule_config.yaml] [--name FIG]
                            [--from-run DIR] [--show]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from paper_plots import figure as pfig  # noqa: E402
from paper_plots import schedule as psched  # noqa: E402


DEFAULTS = {
    # Take ramp/wait/asymmetries from a run directory's metadata. Anything
    # set explicitly under `protocol` still wins.
    "from_run": None,

    "protocol": {
        # null so that `from_run` can supply them; falls back to the
        # documented defaults (2 qubits, ramp 1.0) when neither is given.
        "n_qubits": None,
        "ramp_time": None,
        "wait_time": 2.0,
        "n_steps": 1001,
        "up_z_assymetry_coefficients": None,
        "down_z_assymetry_coefficients": None,
        "up_x_assymetry_coefficients": None,
        "down_x_assymetry_coefficients": None,
        # Schedule errors. Zero keeps the figure clean and idealised.
        "average_ramp_error": 0.0,
        "average_wait_error": 0.0,
        "average_ramp_bias": 0.0,
        "noise_seed": 0,
    },

    # Which families of lines to draw, and which qubits within each.
    # `qubits: null` means every qubit; give a list to pick.
    "lines": {
        "common": {"show": True, "label": r"common ($H_0 + H_{zz}$)",
                   "color": None, "linestyle": "-", "linewidth": None,
                   "alpha": 1.0},
        "h0": {"show": False, "label": r"$H_0$  $(1-s)$",
               "color": None, "linestyle": ":", "linewidth": None,
               "alpha": 1.0},
        "z": {"show": True, "qubits": None, "label": r"$H_z[{i}]$",
              "colors": None, "linestyle": "-", "linewidth": None,
              "alpha": 1.0},
        "x": {"show": True, "qubits": None, "label": r"$H_x[{i}]$",
              "colors": None, "linestyle": "--", "linewidth": None,
              "alpha": 1.0},
    },

    # Vertical markers at the segment boundaries, with x tick labels.
    "markers": {
        "show": True,
        "end_ramp1": True,
        "end_wait": True,
        "end_ramp2": True,
        "start": False,
        "labels": {
            "start": r"$0$",
            "end_ramp1": r"$t_r$",
            "end_wait": r"$t_r + t_w$",
            "end_ramp2": r"$t_f$",
        },
        "replace_xticks": True,   # put the labels on the x axis
        "lines": None,            # which boundaries get a vertical line
        "no_line": [],            # boundaries that get only a tick label
        "color": None,            # null -> the palette's guide colour
        "linestyle": ":",
        "linewidth": 0.9,
        "alpha": 0.8,
        "annotate_segments": False,   # name the ramp/wait spans across the top
        "segment_labels": {
            "ramp1": "ramp up",
            "wait": "wait",
            "ramp2": "ramp down",
        },
        "segment_fontsize": 10,
        "shade_wait": False,      # light band over the wait window
        "shade_color": None,
        "shade_alpha": 0.08,
    },

    "axes": {
        "xlabel": r"$t$",
        "ylabel": r"$s(t)$",
        "title": None,
        "xlim": None,
        "ylim": None,
        "x_margin": 0.0,
        "y_margin": 0.08,
        "grid": False,
    },

    "legend": {
        "show": True,
        "loc": "best",
        "frameon": False,
        "fontsize": None,
        "ncol": 1,
        "title": None,
    },

    "style": {
        "palette": "sober",
        "colors": None,
        "linewidth": None,
        "alpha": 1.0,
        "font_sizes": {},
        "rcparams": {},
    },

    "figure": {
        "preset": "wide",
        "size": None,
    },

    "output": {
        "directory": "paper_plot_images",
        "name": "schedule",
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
                   help="config file (default: schedule_config.yaml "
                        "beside this script)")
    p.add_argument("--from-run", default=None,
                   help="take the protocol from this run directory's metadata")
    p.add_argument("--name", default=None, help="override output.name")
    p.add_argument("--ramp-time", type=float, default=None)
    p.add_argument("--wait-time", type=float, default=None)
    p.add_argument("--no-markers", action="store_true",
                   help="omit the segment boundary markers")
    p.add_argument("--no-legend", action="store_true", help="omit the legend")
    p.add_argument("--show", action="store_true",
                   help="also open the figure in a window")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    script_dir = Path(__file__).resolve().parent
    config_path = (Path(args.config) if args.config
                   else script_dir / "schedule_config.yaml")
    cfg = pfig.load_config(config_path, DEFAULTS)

    if args.from_run:
        cfg["from_run"] = args.from_run
    if args.name:
        cfg["output"]["name"] = args.name
    if args.ramp_time is not None:
        cfg["protocol"]["ramp_time"] = args.ramp_time
    if args.wait_time is not None:
        cfg["protocol"]["wait_time"] = args.wait_time
    if args.no_markers:
        cfg["markers"]["show"] = False
    if args.no_legend:
        cfg["legend"]["show"] = False

    import matplotlib
    if not args.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    style = pfig.apply_style(cfg["style"])

    print(f"Config: {config_path}")

    # ---------------- Protocol ----------------
    protocol, provenance = psched.resolve_protocol(
        cfg["protocol"], cfg.get("from_run"), script_dir)

    print(f"Protocol [{provenance}]:")
    print(f"  qubits        : {protocol['n_qubits']}")
    print(f"  ramp time     : {protocol['ramp_time']:g}")
    print(f"  wait time     : {protocol['wait_time']:g}")
    print(f"  total time    : {protocol['tf']:g}")
    print(f"  boundaries    : ramp1 ends {protocol['tr']:g}, "
          f"wait ends {protocol['tr'] + protocol['wait_time']:g}, "
          f"ramp2 ends {protocol['tf']:g}")
    for term in ("z", "x"):
        up = protocol[f"up_{term}"]
        down = protocol[f"down_{term}"]
        if not (np.allclose(up, 1.0) and np.allclose(down, 1.0)):
            print(f"  {term} asymmetry : up {np.round(up, 4).tolist()}  "
                  f"down {np.round(down, 4).tolist()}")
    print()

    # ---------------- Schedules ----------------
    t, curves = psched.build_schedules(protocol, cfg["lines"])
    if not curves:
        raise SystemExit(
            "no lines selected: enable at least one of lines.common, "
            "lines.h0, lines.z or lines.x")

    print(f"Drawing {len(curves)} line(s):")
    for c in curves:
        print(f"  {c['key']:<12} {c['label']}")
    print()

    # ---------------- Draw ----------------
    width, height = pfig.figure_size(cfg["figure"], style)
    fig, ax = plt.subplots(figsize=(width, height))

    psched.draw_curves(ax, t, curves, cfg["style"], style)

    marker_info = psched.draw_markers(ax, protocol, cfg["markers"], style)
    if marker_info["positions"]:
        shown = ", ".join(f"{n}={v:g}" for n, v in
                          zip(marker_info["names"], marker_info["positions"]))
        print(f"Markers: {shown}")

    psched.finish_axes(ax, t, curves, cfg["axes"], cfg["legend"],
                       marker_info, cfg["markers"], style)
    fig.tight_layout()

    # ---------------- Save ----------------
    out_dir = pfig.output_dir(cfg["output"], script_dir)
    name = cfg["output"].get("name", "schedule")
    if cfg["output"].get("timestamp", False):
        name = pfig.timestamped(name)

    written = pfig.save_figure(fig, out_dir, name, cfg["output"])
    print("\nFigure written:")
    for p in written:
        print(f"  {p}")

    if cfg["output"].get("save_data", True):
        import datetime as _dt

        series = [{"label": c["label"],
                   "freqs": t,                # x: time
                   "magnitude": c["values"],  # y: s(t)
                   "peaks": np.asarray(marker_info["positions"], dtype=float)}
                  for c in curves]
        meta = {
            "figure": name,
            "config_file": str(config_path),
            "created": _dt.datetime.now().isoformat(timespec="seconds"),
            "note": ("x values are stored as 'freqs_i' and y values as "
                     "'magnitude_i' so the archive matches the other paper "
                     "figures; here x is time t and y is the schedule s(t). "
                     "'peaks_i' holds the segment-boundary times."),
            "protocol": pfig.to_plain({
                k: protocol[k] for k in
                ("n_qubits", "ramp_time", "wait_time", "tr", "tf", "n_steps",
                 "up_z", "down_z", "up_x", "down_x",
                 "ramp_error", "wait_error", "ramp_bias", "noise_seed")
            }),
            "protocol_source": provenance,
            "lines": pfig.to_plain(cfg["lines"]),
            "markers": {
                "names": marker_info["names"],
                "positions": pfig.to_plain(marker_info["positions"]),
                "labels": marker_info["labels"],
            },
            "series": [
                {"index": i, "key": c["key"], "label": c["label"],
                 "qubit": c.get("qubit"), "n_points": int(t.size)}
                for i, c in enumerate(curves)
            ],
        }
        data_paths = pfig.save_plot_data(out_dir, name, series, meta)
        if cfg["output"].get("save_csv", False):
            data_paths += pfig.save_series_csv(out_dir, name, series)
        print("Plot data written:")
        for p in data_paths:
            print(f"  {p}")

    if args.show:
        plt.show()
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
