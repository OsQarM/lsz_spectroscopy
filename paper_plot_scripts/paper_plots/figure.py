"""
Figure styling, output directories and data export for the paper figures.

Appearance defaults come from `src/plotting/style.py` -- the same source the
rest of the pipeline uses -- and every one of them can be overridden from the
YAML config. Nothing is ever overwritten: an existing output file gets a
numeric suffix.
"""
from __future__ import annotations

import copy
import datetime as _dt
from pathlib import Path

import numpy as np
import yaml


# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
def deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def load_config(path, defaults: dict) -> dict:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"config file not found: {path}")
    with open(path) as fh:
        user = yaml.safe_load(fh) or {}
    if not isinstance(user, dict):
        raise ValueError(f"{path}: top level of the config must be a mapping")
    return deep_merge(defaults, user)


# --------------------------------------------------------------------------
# Output paths
# --------------------------------------------------------------------------
def unique_path(path) -> Path:
    """`path` if free, else the first `name_2`, `name_3`, ... that is."""
    path = Path(path)
    if not path.exists():
        return path
    stem, suffix, parent = path.stem, path.suffix, path.parent
    n = 2
    while True:
        candidate = parent / f"{stem}_{n}{suffix}"
        if not candidate.exists():
            return candidate
        n += 1


def output_dir(cfg_output: dict, base_dir: Path) -> Path:
    """Create and return the directory the figure and its data go into."""
    root = Path(cfg_output.get("directory", "paper_plot_images")).expanduser()
    if not root.is_absolute():
        root = (base_dir / root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    return root


def timestamped(name: str) -> str:
    return f"{name}_{_dt.datetime.now().strftime('%Y%m%d-%H%M%S')}"


# --------------------------------------------------------------------------
# Style
# --------------------------------------------------------------------------
def apply_style(cfg_style: dict):
    """Apply the shared paper style, then the config's overrides."""
    import matplotlib as mpl
    from plotting import style

    style.apply(cfg_style.get("palette", "sober"))

    fonts = cfg_style.get("font_sizes", {}) or {}
    mapping = {
        "label": "FONT_LABEL",
        "tick": "FONT_TICK",
        "legend": "FONT_LEGEND",
        "title": "FONT_TITLE",
    }
    for key, attr in mapping.items():
        if fonts.get(key) is not None:
            setattr(style, attr, float(fonts[key]))

    if cfg_style.get("rcparams"):
        mpl.rcParams.update(cfg_style["rcparams"])

    return style


def figure_size(cfg_fig: dict, style) -> tuple[float, float]:
    """The figure size, from an explicit [w, h] or one of the named presets."""
    size = cfg_fig.get("size")
    if size:
        if len(size) != 2:
            raise ValueError("figure.size must be [width, height] in inches")
        return float(size[0]), float(size[1])
    preset = str(cfg_fig.get("preset", "wide")).lower()
    presets = {
        "single": style.FIG_SINGLE,
        "wide": style.FIG_WIDE,
        "double": style.FIG_DOUBLE,
        "triple": style.FIG_TRIPLE,
    }
    if preset not in presets:
        raise ValueError(f"unknown figure.preset: {preset!r} "
                         f"(expected one of {', '.join(presets)})")
    return presets[preset]


def series_colors(n: int, cfg_style: dict, style) -> list:
    """One colour per dataset: the configured list, else the palette cycle."""
    explicit = cfg_style.get("colors")
    if explicit:
        return [explicit[i % len(explicit)] for i in range(n)]
    if n == 1:
        return [style.DATA]
    cycle = list(style.CYCLE)
    return [cycle[i % len(cycle)] for i in range(n)]


def series_option(spec: dict, cfg: dict, key: str, default=None):
    """A per-series override from the source block, else the global value."""
    if spec and spec.get(key) is not None:
        return spec[key]
    value = cfg.get(key)
    return default if value is None else value


# --------------------------------------------------------------------------
# Saving
# --------------------------------------------------------------------------
def save_figure(fig, out_dir: Path, name: str, cfg_output: dict) -> list[Path]:
    formats = cfg_output.get("formats") or ["pdf", "png"]
    dpi = int(cfg_output.get("dpi", 500))
    transparent = bool(cfg_output.get("transparent", False))

    written = []
    for ext in formats:
        path = unique_path(out_dir / f"{name}.{str(ext).lstrip('.')}")
        fig.savefig(path, dpi=dpi, bbox_inches="tight", transparent=transparent)
        written.append(path)
    return written


def save_plot_data(out_dir: Path, name: str, series: list[dict],
                   meta: dict) -> list[Path]:
    """Store exactly what was drawn, so the figure can be rebuilt or re-styled.

    Writes an .npz with every curve and its marked peaks, plus a small .yaml
    describing the series and where they came from. Per-series CSVs are
    written too when `formats` asks for them.
    """
    written = []

    arrays = {}
    for i, s in enumerate(series):
        arrays[f"freqs_{i}"] = np.asarray(s["freqs"], dtype=float)
        arrays[f"magnitude_{i}"] = np.asarray(s["magnitude"], dtype=float)
        peaks = np.asarray(s.get("peaks", []), dtype=float)
        arrays[f"peaks_{i}"] = peaks
    arrays["labels"] = np.array([s["label"] for s in series], dtype=object)

    npz_path = unique_path(out_dir / f"{name}_data.npz")
    np.savez_compressed(npz_path, **arrays)
    written.append(npz_path)

    info_path = unique_path(out_dir / f"{name}_data.yaml")
    with open(info_path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(meta, fh, sort_keys=False, default_flow_style=False)
    written.append(info_path)

    return written


def save_series_csv(out_dir: Path, name: str, series: list[dict]) -> list[Path]:
    """One CSV per curve, for plotting software that wants plain text."""
    written = []
    for i, s in enumerate(series):
        safe = "".join(c if (c.isalnum() or c in "-_") else "_"
                       for c in s["label"])[:60] or f"series{i}"
        path = unique_path(out_dir / f"{name}_{i:02d}_{safe}.csv")
        data = np.column_stack([np.asarray(s["freqs"], dtype=float),
                                np.asarray(s["magnitude"], dtype=float)])
        np.savetxt(path, data, delimiter=",", header="frequency,magnitude",
                   comments="")
        written.append(path)
    return written


def to_plain(obj):
    """numpy / Path -> YAML-safe plain Python."""
    if isinstance(obj, dict):
        return {str(k): to_plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_plain(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return to_plain(obj.tolist())
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, Path):
        return str(obj)
    return obj
