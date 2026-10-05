#!/usr/bin/env python
"""
Paper figure: the Fourier transform of one or more swept signals.

Reads sweeps produced by `refactored_code/01_run_sweep.py`, computes their
spectra exactly as the analysis pipeline does, and draws them in one panel.
Everything the paper needs to control is in `fft_config.yaml`: axis labels and
limits, whether the dashed peak markers are drawn, whether there is a legend
and where it sits, colours, line widths, figure size and output formats.

With more than one source the spectra are overlaid for comparison (or offset /
stacked, if the config asks for that).

The figure and a data file holding exactly what was drawn both go into
`paper_plot_images/`.

Usage:
    python plot_fft.py [--config fft_config.yaml] [--name FIGURE_NAME]
                       [--source DIR ...] [--show]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from paper_plots import data as pdata  # noqa: E402
from paper_plots import figure as pfig  # noqa: E402
from paper_plots import spectrum as pspec  # noqa: E402


DEFAULTS = {
    "sources": [],
    "fft": {
        "window": "hann",
        "zero_pad_factor": 32,
        "normalise": None,
    },
    "peaks": {
        "show": True,
        "source": "auto",          # auto | detected | true_differences | detect
        "n_peaks": None,
        "detect_prominence": 0.001,
        "f_min": None,
        "zero_pad_factor": 32,
        "color": None,             # None -> the style's REFERENCE colour
        "linestyle": "--",
        "linewidth": 0.8,
        "alpha": 0.75,
        "per_series_color": False,
        "label": None,             # a legend entry for the markers
        "annotate": False,         # write the frequency above each marker
        "annotate_precision": 3,
        "annotate_fontsize": 8,
    },
    "axes": {
        "xlabel": "frequency",
        "ylabel": r"$|\mathrm{FFT}|$",
        "title": None,
        "xlim": None,
        "ylim": None,
        "x_max_from_peaks": True,
        "x_max_margin": 1.2,
        "yscale": "linear",        # linear | log
        "log_decades": 4.0,
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
        "linestyles": None,
        "alpha": 1.0,
        "font_sizes": {},
        "rcparams": {},
    },
    "layout": {
        "mode": "overlay",         # overlay | offset | stacked
        "offset": 0.0,
        "sharex": True,
    },
    "figure": {
        "preset": "wide",
        "size": None,
    },
    "output": {
        "directory": "paper_plot_images",
        "name": "fft_spectrum",
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
                   help="config file (default: fft_config.yaml beside this script)")
    p.add_argument("--source", action="append", default=None,
                   help="a run directory or .npy to plot; repeatable, "
                        "overrides the config's sources")
    p.add_argument("--name", default=None, help="override output.name")
    p.add_argument("--no-peaks", action="store_true",
                   help="do not draw the dashed peak markers")
    p.add_argument("--no-legend", action="store_true", help="omit the legend")
    p.add_argument("--show", action="store_true",
                   help="also open the figure in a window")
    return p.parse_args(argv)


def resolve_xlim(cfg_axes: dict, series: list[dict]):
    """The x limits: explicit if given, else framed around the marked peaks."""
    xlim = cfg_axes.get("xlim")
    if xlim:
        if len(xlim) != 2:
            raise ValueError("axes.xlim must be [min, max]")
        return float(xlim[0]), float(xlim[1])

    if cfg_axes.get("x_max_from_peaks", True):
        all_peaks = np.concatenate(
            [np.asarray(s["peaks"], dtype=float) for s in series
             if len(s.get("peaks", []))]) if any(
                 len(s.get("peaks", [])) for s in series) else np.array([])
        if all_peaks.size:
            margin = float(cfg_axes.get("x_max_margin", 1.2))
            return 0.0, float(all_peaks.max()) * margin
    return None


def apply_yscale(ax, cfg_axes: dict, mags):
    """Linear or log y, with the same decade floor the pipeline figures use."""
    combined = np.concatenate([np.asarray(m, dtype=float) for m in mags])
    yscale = str(cfg_axes.get("yscale", "linear")).lower()

    ylim = cfg_axes.get("ylim")
    if ylim:
        if len(ylim) != 2:
            raise ValueError("axes.ylim must be [min, max]")
        if yscale == "log":
            ax.set_yscale("log")
        ax.set_ylim(float(ylim[0]), float(ylim[1]))
        return

    if yscale == "log":
        positive = combined[combined > 0]
        if positive.size:
            ax.set_yscale("log")
            top = combined.max() * 2.0
            decades = float(cfg_axes.get("log_decades", 4.0))
            floor = max(positive.min() * 0.5, top / 10 ** decades)
            ax.set_ylim(floor, top)
    else:
        top = combined.max()
        if top > 0:
            ax.set_ylim(0, top * 1.08)


def main(argv=None):
    args = parse_args(argv)

    script_dir = Path(__file__).resolve().parent
    config_path = (Path(args.config) if args.config
                   else script_dir / "fft_config.yaml")
    cfg = pfig.load_config(config_path, DEFAULTS)

    if args.source:
        cfg["sources"] = list(args.source)
    if args.name:
        cfg["output"]["name"] = args.name
    if args.no_peaks:
        cfg["peaks"]["show"] = False
    if args.no_legend:
        cfg["legend"]["show"] = False

    cfg_fft = cfg["fft"]
    cfg_peaks = cfg["peaks"]
    cfg_axes = cfg["axes"]
    cfg_legend = cfg["legend"]
    cfg_style = cfg["style"]
    cfg_layout = cfg["layout"]
    cfg_out = cfg["output"]

    import matplotlib
    if not args.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    style = pfig.apply_style(cfg_style)

    print(f"Config: {config_path}")
    sources = pdata.load_sources(cfg["sources"], script_dir)
    print(f"Loaded {len(sources)} dataset(s).\n")

    # ---------------- Compute every spectrum ----------------
    series = []
    for src in sources:
        freqs, mag = pspec.compute_spectrum(
            src["signal"], src["tw"],
            window=cfg_fft.get("window", "hann"),
            zero_pad_factor=cfg_fft.get("zero_pad_factor", 32),
            normalise=cfg_fft.get("normalise"),
        )

        peaks, peak_origin = (np.array([]), "none")
        if cfg_peaks.get("show", True):
            peaks, peak_origin = pspec.peaks_for_source(
                src, cfg_peaks, freqs, mag)

        series.append({
            "label": src["label"],
            "freqs": freqs,
            "magnitude": mag,
            "peaks": peaks,
            "peak_origin": peak_origin,
            "source": src,
        })

        n_pts = src["signal"].size
        print(f"  {src['label']}")
        print(f"    signal     : {src['signal_file']}")
        print(f"    points     : {n_pts}   wait-time grid: {src['tw_origin']}")
        print(f"    tw range   : 0 ... {src['tw'][-1]:g}")
        if src["n_qubits"]:
            print(f"    qubits     : {src['n_qubits']}")
        print(f"    resolution : {1.0 / (src['tw'][-1] - src['tw'][0]):.6f}"
              f"   (1 / max_wait_time)")
        if cfg_peaks.get("show", True):
            print(f"    peaks      : {len(peaks)} ({peak_origin})")
        print()

    # ---------------- Draw ----------------
    mode = str(cfg_layout.get("mode", "overlay")).lower()
    if mode not in ("overlay", "offset", "stacked"):
        raise ValueError(f"unknown layout.mode: {mode!r} "
                         f"(expected overlay, offset or stacked)")

    width, height = pfig.figure_size(cfg["figure"], style)
    colors = pfig.series_colors(len(series), cfg_style, style)
    base_lw = cfg_style.get("linewidth") or style.LINEWIDTH
    linestyles = cfg_style.get("linestyles")

    if mode == "stacked":
        fig, axes = plt.subplots(
            len(series), 1,
            figsize=(width, height * len(series)),
            sharex=bool(cfg_layout.get("sharex", True)))
        axes = np.atleast_1d(axes)
    else:
        fig, ax = plt.subplots(figsize=(width, height))
        axes = np.array([ax] * len(series))

    offset_step = float(cfg_layout.get("offset", 0.0))
    peak_color_cfg = cfg_peaks.get("color")
    legend_handles_needed = False

    for i, s in enumerate(series):
        ax = axes[i]
        spec = s["source"]["spec"]
        color = pfig.series_option(spec, {}, "color") or colors[i]
        lw = float(pfig.series_option(spec, {}, "linewidth") or base_lw)
        ls = pfig.series_option(spec, {}, "linestyle")
        if ls is None and linestyles:
            ls = linestyles[i % len(linestyles)]
        ls = ls or "-"
        alpha = float(pfig.series_option(spec, cfg_style, "alpha", 1.0))

        mag = np.asarray(s["magnitude"], dtype=float)
        if mode == "offset" and offset_step:
            mag = mag + i * offset_step
            s["magnitude_drawn"] = mag

        ax.plot(s["freqs"], mag, color=color, lw=lw, linestyle=ls,
                alpha=alpha, label=s["label"], zorder=2 + i)
        legend_handles_needed = True

        # Dashed peak markers -- optional, and optionally per-series coloured
        # so overlaid runs can be told apart.
        if cfg_peaks.get("show", True) and len(s["peaks"]):
            if peak_color_cfg:
                pcolor = peak_color_cfg
            elif cfg_peaks.get("per_series_color", False):
                pcolor = color
            else:
                pcolor = style.REFERENCE
            plabel = cfg_peaks.get("label") if i == 0 else None
            for j, f in enumerate(np.asarray(s["peaks"], dtype=float)):
                ax.axvline(
                    f, color=pcolor,
                    lw=float(cfg_peaks.get("linewidth", 0.8)),
                    linestyle=cfg_peaks.get("linestyle", "--"),
                    alpha=float(cfg_peaks.get("alpha", 0.75)),
                    zorder=1,
                    label=plabel if j == 0 else None)
            if cfg_peaks.get("annotate", False):
                prec = int(cfg_peaks.get("annotate_precision", 3))
                for f in np.asarray(s["peaks"], dtype=float):
                    ax.annotate(
                        f"{f:.{prec}f}", xy=(f, 1.0),
                        xycoords=("data", "axes fraction"),
                        xytext=(2, -4), textcoords="offset points",
                        rotation=90, va="top", ha="left",
                        fontsize=float(cfg_peaks.get("annotate_fontsize", 8)),
                        color=pcolor)

    # ---------------- Axes ----------------
    xlim = resolve_xlim(cfg_axes, series)
    mags_for_scale = [s.get("magnitude_drawn", s["magnitude"]) for s in series]

    # In overlay/offset mode every entry of `axes` is the same Axes object, so
    # deduplicate by identity (Axes are not orderable, so np.unique cannot).
    seen_ids: set[int] = set()
    unique_axes = []
    for ax in axes:
        if id(ax) not in seen_ids:
            seen_ids.add(id(ax))
            unique_axes.append(ax)

    for ax in unique_axes:
        if xlim:
            ax.set_xlim(*xlim)
        if cfg_axes.get("grid"):
            ax.grid(True, which="major", lw=0.4, alpha=0.3)

    if mode == "stacked":
        for i, ax in enumerate(axes):
            apply_yscale(ax, cfg_axes, [mags_for_scale[i]])
            last = (i == len(axes) - 1)
            style.style_axis(
                ax,
                xlabel=cfg_axes.get("xlabel") if (last or not cfg_layout.get("sharex", True)) else None,
                ylabel=cfg_axes.get("ylabel"),
                title=cfg_axes.get("title") if i == 0 else None)
            if cfg_legend.get("show", True):
                ax.legend(frameon=bool(cfg_legend.get("frameon", False)),
                          fontsize=cfg_legend.get("fontsize") or style.FONT_LEGEND,
                          loc=cfg_legend.get("loc", "best"),
                          ncol=int(cfg_legend.get("ncol", 1)),
                          title=cfg_legend.get("title"))
    else:
        ax = axes[0]
        apply_yscale(ax, cfg_axes, mags_for_scale)
        style.style_axis(ax,
                         xlabel=cfg_axes.get("xlabel"),
                         ylabel=cfg_axes.get("ylabel"),
                         title=cfg_axes.get("title"))
        # A legend for a single unlabelled curve is noise, so it is drawn only
        # when there is more than one series or the peaks carry a label.
        want_legend = bool(cfg_legend.get("show", True)) and legend_handles_needed
        if want_legend and (len(series) > 1 or cfg_peaks.get("label")):
            ax.legend(frameon=bool(cfg_legend.get("frameon", False)),
                      fontsize=cfg_legend.get("fontsize") or style.FONT_LEGEND,
                      loc=cfg_legend.get("loc", "best"),
                      ncol=int(cfg_legend.get("ncol", 1)),
                      title=cfg_legend.get("title"))

    fig.tight_layout()

    # ---------------- Save ----------------
    out_dir = pfig.output_dir(cfg_out, script_dir)
    name = cfg_out.get("name", "fft_spectrum")
    if cfg_out.get("timestamp", False):
        name = pfig.timestamped(name)

    written = pfig.save_figure(fig, out_dir, name, cfg_out)
    print("Figure written:")
    for p in written:
        print(f"  {p}")

    if cfg_out.get("save_data", True):
        meta = {
            "figure": name,
            "config_file": str(config_path),
            "created": __import__("datetime").datetime.now().isoformat(
                timespec="seconds"),
            "fft": pfig.to_plain(cfg_fft),
            "peaks": pfig.to_plain(cfg_peaks),
            "layout": pfig.to_plain(cfg_layout),
            "series": [
                {
                    "index": i,
                    "label": s["label"],
                    "source_path": str(s["source"]["path"]),
                    "signal_file": str(s["source"]["signal_file"]),
                    "n_points": int(s["source"]["signal"].size),
                    "max_wait_time": float(s["source"]["tw"][-1]),
                    "wait_grid_origin": s["source"]["tw_origin"],
                    "n_qubits": s["source"]["n_qubits"],
                    "hamiltonian": pfig.to_plain(s["source"]["hamiltonian"]),
                    "n_peaks_marked": int(len(s["peaks"])),
                    "peak_origin": s["peak_origin"],
                    "peaks": pfig.to_plain(np.asarray(s["peaks"], dtype=float)),
                }
                for i, s in enumerate(series)
            ],
        }
        data_paths = pfig.save_plot_data(out_dir, name, series, meta)
        if cfg_out.get("save_csv", False):
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
