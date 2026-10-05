"""
Loading swept signals for the paper figures.

A "source" in the config is one dataset to draw. It can be:

  * a run directory produced by `refactored_code/01_run_sweep.py`
    (pc_list.npy + tw_l.npy + metadata.yaml), or
  * a bare .npy file holding the signal, with the wait-time grid taken from
    a sibling tw_l.npy, from an explicit `tw_file`, or reconstructed from
    `max_wait_time` given in the source block.

Everything a figure might want to label or annotate with -- the Hamiltonian,
the protocol settings, the detected peaks from a diagnostics run -- is
collected here so the plotting code only deals with arrays.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml


class SourceError(RuntimeError):
    """A dataset named in the config could not be loaded."""


def _read_yaml(path: Path):
    if not path.is_file():
        return None
    with open(path) as fh:
        return yaml.safe_load(fh) or {}


def _load_metadata(run_dir: Path) -> dict:
    """A run's metadata, if script 1 wrote one. Missing is not an error:
    diagnostics-only runs and hand-made .npy files have none."""
    return _read_yaml(run_dir / "metadata.yaml") or {}


def _load_diagnostics(run_dir: Path) -> dict:
    """Detected peaks and true differences, if script 2 has been run here.

    Used to mark peaks without recomputing the peak-finding: the positions
    the pipeline actually detected are the ones worth drawing.
    """
    out = {}
    npz = run_dir / "diagnostics_results.npz"
    if not npz.is_file():
        return out
    try:
        with np.load(npz, allow_pickle=True) as data:
            for key in ("freqs", "true_differences", "amplitudes"):
                if key in data.files:
                    out[key] = np.asarray(data[key], dtype=float)
    except Exception:
        # A malformed archive should not stop the figure from being drawn.
        return {}
    return out


def _resolve_signal_path(path: Path) -> tuple[Path, Path]:
    """Return (signal_file, run_dir) for a source path."""
    if path.is_dir():
        meta = _load_metadata(path)
        name = (meta.get("sweep", {}) or {}).get("pc_list_file", "pc_list.npy")
        candidate = path / name
        if candidate.is_file():
            return candidate, path
        found = sorted(path.glob("pc_list*.npy"))
        if not found:
            raise SourceError(
                f"{path} contains no pc_list .npy -- was the sweep actually "
                f"run (sweep.run_simulation: true)?")
        return found[0], path
    if path.is_file():
        return path, path.parent
    raise SourceError(f"source path does not exist: {path}")


def load_source(spec, base_dir: Path, index: int) -> dict:
    """Load one dataset described by a config `sources` entry.

    `spec` is either a path string or a mapping with a `path` key plus
    optional `label`, `tw_file`, `max_wait_time`, and any per-series style
    overrides, which are passed through untouched for the plotting code.
    """
    if isinstance(spec, (str, Path)):
        spec = {"path": str(spec)}
    if not isinstance(spec, dict):
        raise SourceError(f"sources[{index}] must be a path or a mapping, "
                          f"got {type(spec).__name__}")

    raw_path = spec.get("path")
    if not raw_path:
        raise SourceError(f"sources[{index}] has no 'path'")

    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = (base_dir / path).resolve()

    signal_file, run_dir = _resolve_signal_path(path)
    signal = np.asarray(np.load(signal_file), dtype=float).ravel()
    if signal.size == 0:
        raise SourceError(f"{signal_file} holds an empty array")

    metadata = _load_metadata(run_dir)
    protocol = (metadata.get("common", {}) or {}).get("protocol", {}) or {}

    # The wait-time grid, in order of preference: an explicit file, the grid
    # stored beside the signal, the run's protocol parameters, or an explicit
    # max_wait_time in the source block.
    tw = None
    origin = None

    if spec.get("tw_file"):
        tw_path = Path(spec["tw_file"]).expanduser()
        if not tw_path.is_absolute():
            tw_path = (base_dir / tw_path).resolve()
        if not tw_path.is_file():
            raise SourceError(f"sources[{index}].tw_file not found: {tw_path}")
        tw = np.asarray(np.load(tw_path), dtype=float).ravel()
        origin = f"tw_file ({tw_path.name})"
    else:
        sibling = run_dir / (metadata.get("sweep", {}) or {}).get("tw_l_file", "tw_l.npy")
        if sibling.is_file():
            tw = np.asarray(np.load(sibling), dtype=float).ravel()
            origin = f"stored grid ({sibling.name})"

    if tw is None:
        max_wait = spec.get("max_wait_time", protocol.get("max_wait_time"))
        if max_wait is None:
            raise SourceError(
                f"sources[{index}] ({path}): no wait-time grid found. Provide "
                f"a tw_file, a max_wait_time, or point at a run directory "
                f"that stores tw_l.npy.")
        tw = np.linspace(0.0, float(max_wait), signal.size)
        origin = f"reconstructed from max_wait_time = {float(max_wait):g}"

    if tw.size != signal.size:
        raise SourceError(
            f"sources[{index}] ({path}): signal has {signal.size} points but "
            f"the wait-time grid has {tw.size}")
    if tw.size < 2:
        raise SourceError(f"sources[{index}] ({path}): need at least 2 points")

    ham = (metadata.get("common", {}) or {}).get("hamiltonian", {}) or {}
    n_qubits = len(ham.get("local_z", [])) or None

    label = spec.get("label")
    if label is None:
        label = run_dir.name if path.is_dir() else signal_file.stem

    return {
        "label": str(label),
        "path": path,
        "signal_file": signal_file,
        "run_dir": run_dir,
        "signal": signal,
        "tw": tw,
        "tw_origin": origin,
        "metadata": metadata,
        "hamiltonian": ham,
        "protocol": protocol,
        "n_qubits": n_qubits,
        "diagnostics": _load_diagnostics(run_dir),
        "spec": spec,
    }


def load_sources(specs, base_dir: Path) -> list[dict]:
    """Load every dataset in the config's `sources` list."""
    if not specs:
        raise SourceError("the config lists no sources to plot")
    if isinstance(specs, (str, Path, dict)):
        specs = [specs]
    return [load_source(spec, base_dir, i) for i, spec in enumerate(specs)]
