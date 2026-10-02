"""
Run directories, tee'd logging, figure capture and metadata serialisation.

Two guarantees hold throughout:

  * nothing is ever overwritten -- new run directories get a timestamp and, if
    that still collides, a numeric suffix; files written into an existing
    directory (script 2 writing into a script-1 run) go through
    `unique_path`, which appends `_2`, `_3`, ... as needed;
  * every directory is created on demand.
"""
from __future__ import annotations

import datetime as _dt
import os
import platform
import socket
import sys
from pathlib import Path

import numpy as np
import yaml


# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
def timestamp(fmt: str = "%Y%m%d-%H%M%S") -> str:
    return _dt.datetime.now().strftime(fmt)


def unique_path(path) -> Path:
    """Return `path` if free, else the first `name_2`, `name_3`, ... that is.

    Used for every file written into a directory that may already hold output
    from an earlier run, so no result is ever silently replaced.
    """
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


def create_run_dir(root, prefix: str, run_name: str | None = None) -> Path:
    """Create and return a fresh run directory under `root`.

    The name is `<prefix>_<timestamp>` unless `run_name` is given, in which
    case it is used verbatim. Either way a numeric suffix is appended if the
    directory already exists, so an existing run is never written into.
    """
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    base = run_name if run_name else f"{prefix}_{timestamp()}"
    candidate = root / base
    n = 2
    while candidate.exists():
        candidate = root / f"{base}_{n}"
        n += 1
    candidate.mkdir(parents=True)
    (candidate / "plots").mkdir()
    return candidate


# --------------------------------------------------------------------------
# Logging: everything printed goes to the terminal and to the run's log file
# --------------------------------------------------------------------------
class Tee:
    """A write-through stream splitter for `sys.stdout` / `sys.stderr`.

    Carriage-return progress bars are kept readable in the log file: text
    written after a '\\r' overwrites the current unterminated line rather than
    accumulating thousands of partial lines.
    """

    def __init__(self, stream, fh):
        self._stream = stream
        self._fh = fh
        self._line_open = False

    def write(self, data):
        self._stream.write(data)
        if "\r" in data:
            # Keep only the last refresh of a progress line.
            tail = data.rsplit("\r", 1)[-1]
            if self._line_open:
                self._fh.write("\n")
            self._fh.write(tail)
            self._line_open = not tail.endswith("\n")
        else:
            self._fh.write(data)
            if data:
                self._line_open = not data.endswith("\n")
        return len(data)

    def flush(self):
        self._stream.flush()
        self._fh.flush()

    def isatty(self):
        return self._stream.isatty()

    @property
    def encoding(self):
        return getattr(self._stream, "encoding", "utf-8")


class RunLogger:
    """Context manager that tees stdout/stderr into `<run_dir>/output.txt`."""

    def __init__(self, run_dir, filename: str = "output.txt"):
        self.path = unique_path(Path(run_dir) / filename)
        self._fh = None
        self._stdout = None
        self._stderr = None

    def __enter__(self):
        self._fh = open(self.path, "w", encoding="utf-8")
        self._stdout, self._stderr = sys.stdout, sys.stderr
        sys.stdout = Tee(self._stdout, self._fh)
        sys.stderr = Tee(self._stderr, self._fh)
        return self

    def __exit__(self, exc_type, exc, tb):
        sys.stdout, sys.stderr = self._stdout, self._stderr
        if self._fh:
            self._fh.flush()
            self._fh.close()
        return False


# --------------------------------------------------------------------------
# Formatted printing
# --------------------------------------------------------------------------
WIDTH = 78


def rule(char: str = "-") -> None:
    print(char * WIDTH)


def banner(title: str, subtitle: str | None = None) -> None:
    """A heavy, boxed heading for the top of a script."""
    print()
    print("=" * WIDTH)
    print(f"  {title}")
    if subtitle:
        print(f"  {subtitle}")
    print("=" * WIDTH)


def section(title: str) -> None:
    """A light section heading."""
    print()
    print(f"-- {title} " + "-" * max(0, WIDTH - len(title) - 4))


def kv(key: str, value, indent: int = 2, width: int = 34) -> None:
    """Print one aligned `key : value` line."""
    print(f"{' ' * indent}{key:<{width}} {value}")


def fmt_duration(seconds: float) -> str:
    """Human-readable duration, e.g. '2 h 14 min 3 s'."""
    if seconds is None or not np.isfinite(seconds):
        return "unknown"
    seconds = float(seconds)
    if seconds < 1:
        return f"{seconds * 1000:.0f} ms"
    if seconds < 60:
        return f"{seconds:.1f} s"
    minutes, sec = divmod(seconds, 60)
    if minutes < 60:
        return f"{int(minutes)} min {sec:.0f} s"
    hours, minutes = divmod(minutes, 60)
    if hours < 24:
        return f"{int(hours)} h {int(minutes)} min {sec:.0f} s"
    days, hours = divmod(hours, 24)
    return f"{int(days)} d {int(hours)} h {int(minutes)} min"


def fmt_array(arr, precision: int = 4) -> str:
    return np.array2string(np.asarray(arr), precision=precision,
                           suppress_small=True, max_line_width=WIDTH)


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------
class FigureSaver:
    """Saves matplotlib figures into a run's `plots/` directory.

    `save(fig, name)` writes the named figure. `save_new(prefix)` sweeps up
    every figure that has been created since the last call and saves them --
    which is how the figures produced inside `run_full_diagnostics` (it calls
    `plt.show()` itself and returns nothing) are captured.
    """

    def __init__(self, plots_dir, formats=("png",), dpi: int = 300):
        self.dir = Path(plots_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.formats = tuple(formats) or ("png",)
        self.dpi = dpi
        self.saved: list[Path] = []
        self._seen: set[int] = set()

    def _write(self, fig, name: str) -> list[Path]:
        written = []
        for ext in self.formats:
            path = unique_path(self.dir / f"{name}.{ext.lstrip('.')}")
            fig.savefig(path, dpi=self.dpi, bbox_inches="tight")
            written.append(path)
            self.saved.append(path)
        return written

    def save(self, fig, name: str, close: bool = True) -> list[Path]:
        """Save one figure under `name` (no extension)."""
        import matplotlib.pyplot as plt

        if fig is None:
            return []
        written = self._write(fig, name)
        self._seen.add(id(fig))
        if close:
            plt.close(fig)
            self._seen.discard(id(fig))
        return written

    def mark_existing(self) -> None:
        """Ignore the figures currently open -- only save ones made later."""
        import matplotlib.pyplot as plt

        for num in plt.get_fignums():
            self._seen.add(id(plt.figure(num)))

    def save_new(self, prefix: str, close: bool = True) -> list[Path]:
        """Save every still-open figure not seen before, as `prefix_01`, ..."""
        import matplotlib.pyplot as plt

        written = []
        new_figs = [plt.figure(n) for n in plt.get_fignums()
                    if id(plt.figure(n)) not in self._seen]
        for i, fig in enumerate(new_figs, start=1):
            written.extend(self._write(fig, f"{prefix}_{i:02d}"))
            self._seen.add(id(fig))
        if close:
            for fig in new_figs:
                self._seen.discard(id(fig))
                plt.close(fig)
        return written

    def summary(self) -> str:
        return f"{len(self.saved)} figure file(s) in {self.dir}"


# --------------------------------------------------------------------------
# Metadata
# --------------------------------------------------------------------------
def to_plain(obj):
    """Recursively convert numpy / Path objects into YAML-safe plain Python."""
    if isinstance(obj, dict):
        return {str(k): to_plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_plain(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return to_plain(obj.tolist())
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, complex):
        return {"real": obj.real, "imag": obj.imag}
    if isinstance(obj, Path):
        return str(obj)
    return obj


def environment_info() -> dict:
    """Provenance: who ran this, where, with which library versions."""
    import matplotlib
    import qutip
    import scipy

    return {
        "timestamp": _dt.datetime.now().isoformat(timespec="seconds"),
        "host": socket.gethostname(),
        "user": os.environ.get("USER", "unknown"),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "python_executable": sys.executable,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "qutip": qutip.__version__,
        "matplotlib": matplotlib.__version__,
    }


def write_metadata(path, payload: dict) -> Path:
    """Write `payload` as YAML to `path`, never overwriting an existing file."""
    path = unique_path(Path(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(to_plain(payload), fh, sort_keys=False,
                       default_flow_style=False)
    return path


def read_metadata(path) -> dict:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"metadata file not found: {path}")
    with open(path) as fh:
        return yaml.safe_load(fh) or {}
