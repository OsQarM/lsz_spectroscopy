#!/usr/bin/env python
"""
Script 4 -- random-Hamiltonian search for the largest minimum peak separation.

Samples Hamiltonians with random parameters inside configured bounds,
diagonalises each one's midpoint (target) Hamiltonian, forms all
2^n*(2^n-1)/2 transition frequencies, and records the smallest gap between
any two of them. It keeps only the best Hamiltonians found, and reports the
largest minimum separation together with the parameters that produced it.

That gap is exactly the "Closest true freq. spacing" of the pipeline's
sampling-adequacy check: resolving the two closest peaks needs
`max_wait_time >= 1 / gap`, so a bigger gap buys a shorter, cheaper sweep.
The script prints the implied `max_wait_time` (and the `n_sims` that keeps
the fastest transition below Nyquist) for each reported candidate, ready to
paste into `config.yaml`.

Usage:
    python 04_search_hamiltonians.py [--config search_config.yaml]
                                     [--n-samples N] [--n-qubits Q] [--seed S]
"""
from __future__ import annotations

import argparse
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

from lzs_pipeline import io_utils as io  # noqa: E402


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
DEFAULTS = {
    "search": {
        "n_qubits": 3,
        "n_samples": 200000,
        "seed": 0,
        "batch_size": 20000,
        "units": "frequency",
        "max_frequency": None,
        "top_k": 10,
        "output_root": "results/hamiltonian_search",
        "run_name": None,
    },
    "terms": {
        "local_z": {"sample": True, "bounds": [-4.0, 4.0], "discrete": None},
        "local_x": {"sample": False, "value": 0.0},
        "two_body": {"sample": True, "bounds": [-1.5, 1.5], "discrete": None},
    },
}

TERM_NAMES = ("local_z", "local_x", "two_body")


def _deep_merge(base: dict, override: dict) -> dict:
    import copy

    out = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def load_config(path) -> dict:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"config file not found: {path}")
    with open(path) as fh:
        user = yaml.safe_load(fh) or {}
    if not isinstance(user, dict):
        raise ValueError(f"{path}: top level of the config must be a mapping")

    cfg = _deep_merge(DEFAULTS, user)

    # A term group the user spelled out replaces its default wholesale rather
    # than merging into it. Otherwise writing `sample: true` with no bounds
    # would silently inherit the default bounds, and the search would run with
    # limits that appear nowhere in the user's file.
    for name in TERM_NAMES:
        user_spec = (user.get("terms") or {}).get(name)
        if isinstance(user_spec, dict):
            cfg["terms"][name] = dict(user_spec)

    return cfg


def n_terms_for(name: str, n_qubits: int) -> int:
    """How many entries a term group has for `n_qubits` qubits."""
    if name == "two_body":
        return n_qubits * (n_qubits - 1) // 2
    return n_qubits


def resolve_term(name: str, spec: dict, n_qubits: int) -> dict:
    """Validate one term group's config into a sampling plan."""
    size = n_terms_for(name, n_qubits)
    spec = spec or {}
    sample = bool(spec.get("sample", False))

    if sample:
        bounds = spec.get("bounds")
        if bounds is None or len(bounds) != 2:
            raise ValueError(
                f"terms.{name}: sample is true, so 'bounds: [lo, hi]' is required")
        lo, hi = float(bounds[0]), float(bounds[1])
        if not hi > lo:
            raise ValueError(
                f"terms.{name}.bounds must have hi > lo, got [{lo}, {hi}]")
        step = spec.get("discrete")
        step = float(step) if step else None
        if step is not None and step <= 0:
            raise ValueError(f"terms.{name}.discrete must be positive")
        return {"name": name, "size": size, "sample": True,
                "lo": lo, "hi": hi, "discrete": step}

    value = spec.get("value", 0.0)
    if np.isscalar(value):
        fixed = np.full(size, float(value))
    else:
        fixed = np.asarray([float(v) for v in value], dtype=float)
        if fixed.size != size:
            raise ValueError(
                f"terms.{name}.value has {fixed.size} entries but "
                f"{n_qubits} qubits need {size}")
    return {"name": name, "size": size, "sample": False, "fixed": fixed}


# --------------------------------------------------------------------------
# Spectrum machinery
# --------------------------------------------------------------------------
def basis_tables(n_qubits: int):
    """Sign tables for the diagonal (local_x = 0) evaluation.

    Returns `Z` of shape (2^n, n) with entries +-1 for each basis state, and
    `ZZ` of shape (2^n, n_pairs) holding z_i * z_j for each coupled pair, in
    the (0,1), (0,2), ..., (1,2), ... order the rest of the codebase uses.
    """
    m = 2 ** n_qubits
    bits = np.array([[(s >> (n_qubits - 1 - q)) & 1 for q in range(n_qubits)]
                     for s in range(m)], dtype=int)
    z = 1 - 2 * bits
    pairs = [(i, j) for i in range(n_qubits - 1)
             for j in range(i + 1, n_qubits)]
    zz = np.array([z[:, i] * z[:, j] for i, j in pairs], dtype=float).T
    if zz.size == 0:
        zz = np.zeros((m, 0))
    return z.astype(float), zz, pairs


def pauli_operators(n_qubits: int):
    """Per-qubit sigma_z, sigma_x and the ZZ products, as dense matrices.

    Only needed when local_x is nonzero, which makes the target Hamiltonian
    non-diagonal and forces an explicit diagonalisation.
    """
    sz = np.array([[1.0, 0.0], [0.0, -1.0]])
    sx = np.array([[0.0, 1.0], [1.0, 0.0]])
    eye = np.eye(2)

    def embed(op, q):
        out = np.array([[1.0]])
        for k in range(n_qubits):
            out = np.kron(out, op if k == q else eye)
        return out

    sz_list = [embed(sz, q) for q in range(n_qubits)]
    sx_list = [embed(sx, q) for q in range(n_qubits)]
    pairs = [(i, j) for i in range(n_qubits - 1)
             for j in range(i + 1, n_qubits)]
    zz_list = [sz_list[i] @ sz_list[j] for i, j in pairs]
    return sz_list, sx_list, zz_list, pairs


def eigenvalues_diagonal(lz, lx, tb, z_tab, zz_tab):
    """Batched spectra when local_x is identically zero.

    The target Hamiltonian is then diagonal in the computational basis, so a
    matrix product gives every eigenvalue at once -- no diagonalisation.
    `lz` is (B, n), `tb` is (B, n_pairs); returns (B, 2^n) sorted ascending.
    """
    energies = lz @ z_tab.T
    if zz_tab.shape[1]:
        energies = energies + tb @ zz_tab.T
    return np.sort(energies, axis=1)


def eigenvalues_dense(lz, lx, tb, sz_list, sx_list, zz_list):
    """Batched spectra for the general (local_x nonzero) case."""
    b = lz.shape[0]
    dim = sz_list[0].shape[0]
    h = np.zeros((b, dim, dim))
    for q, op in enumerate(sz_list):
        h += lz[:, q, None, None] * op
    for q, op in enumerate(sx_list):
        h += lx[:, q, None, None] * op
    for k, op in enumerate(zz_list):
        h += tb[:, k, None, None] * op
    return np.linalg.eigvalsh(h)


def min_transition_gap(energies, triu_i, triu_j):
    """Smallest spacing between any two transition frequencies, per sample.

    `energies` is (B, M) sorted ascending. Every pair (i<j) gives a positive
    transition E_j - E_i; sorting those and taking the smallest consecutive
    difference gives the closest spacing -- the quantity the pipeline's
    sampling check calls "Closest true freq. spacing".

    Returns (gaps, max_transition) both of shape (B,), in angular units.
    """
    diffs = energies[:, triu_j] - energies[:, triu_i]
    diffs = np.sort(diffs, axis=1)
    gaps = np.diff(diffs, axis=1).min(axis=1)
    return gaps, diffs[:, -1]


# --------------------------------------------------------------------------
# Sampling
# --------------------------------------------------------------------------
def draw(plan: dict, n: int, rng) -> np.ndarray:
    """Draw `n` samples of one term group, shape (n, plan['size'])."""
    if not plan["sample"]:
        return np.broadcast_to(plan["fixed"], (n, plan["size"]))
    if plan["size"] == 0:
        return np.zeros((n, 0))
    vals = rng.uniform(plan["lo"], plan["hi"], size=(n, plan["size"]))
    step = plan["discrete"]
    if step is not None:
        vals = np.round(vals / step) * step
        vals = np.clip(vals, plan["lo"], plan["hi"])
    return vals


class TopK:
    """Keeps the `k` best (largest-gap) candidates seen so far.

    Only the winners are retained -- the whole point is to scan a large
    number of Hamiltonians without storing them all.
    """

    def __init__(self, k: int):
        self.k = max(1, int(k))
        self.items: list[dict] = []

    def offer_batch(self, gaps, max_freqs, lz, lx, tb):
        """Merge a batch's best candidates into the running top-k."""
        if gaps.size == 0:
            return
        take = min(self.k, gaps.size)
        # argpartition picks the `take` largest without a full sort.
        idx = np.argpartition(gaps, -take)[-take:]
        for i in idx:
            self.items.append({
                "gap": float(gaps[i]),
                "max_transition": float(max_freqs[i]),
                "local_z": np.array(lz[i], dtype=float),
                "local_x": np.array(lx[i], dtype=float),
                "two_body": np.array(tb[i], dtype=float),
            })
        self.items.sort(key=lambda d: d["gap"], reverse=True)
        del self.items[self.k:]

    @property
    def best(self):
        return self.items[0] if self.items else None


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------
def describe_candidate(cand: dict, rank: int, scale: float, pairs, units: str):
    """Print one candidate and what it implies for the sweep parameters."""
    gap = cand["gap"] / scale
    max_tr = cand["max_transition"] / scale

    print()
    print(f"  --- rank {rank} " + "-" * 56)
    io.kv("Closest true freq. spacing", f"{gap:.6f}   [{units}]")
    io.kv("Highest transition", f"{max_tr:.6f}   [{units}]")
    io.kv("local_z", io.fmt_array(cand["local_z"], precision=4))
    io.kv("local_x", io.fmt_array(cand["local_x"], precision=4))
    io.kv("two_body", io.fmt_array(cand["two_body"], precision=4))
    io.kv("  (pair order)", ", ".join(f"({i},{j})" for i, j in pairs)
          if pairs else "none")

    if units == "frequency" and gap > 0:
        # The sampling check compares 1/max_wait_time against the gap, and
        # the Nyquist limit 1/(2*dt_wait) against the highest frequency.
        needed_wait = 1.0 / gap
        print()
        io.kv("-> min max_wait_time to resolve", f"{needed_wait:.2f}")
        if max_tr > 0:
            needed_nsims = int(np.ceil(needed_wait * 2 * max_tr)) + 1
            io.kv("-> min n_sims at that wait time", needed_nsims)


def yaml_block(cand: dict) -> str:
    """The candidate as a `hamiltonian:` block ready to paste into config.yaml."""
    def fmt(arr):
        return "[" + ", ".join(f"{v:.6g}" for v in arr) + "]"

    return (
        "  hamiltonian:\n"
        f"    local_z:  {fmt(cand['local_z'])}\n"
        f"    local_x:  {fmt(cand['local_x'])}\n"
        f"    two_body: {fmt(cand['two_body'])}\n"
    )


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=None,
                   help="path to the search config "
                        "(default: search_config.yaml beside this script)")
    p.add_argument("--n-samples", type=int, default=None,
                   help="override search.n_samples")
    p.add_argument("--n-qubits", type=int, default=None,
                   help="override search.n_qubits")
    p.add_argument("--seed", type=int, default=None, help="override search.seed")
    p.add_argument("--run-name", default=None,
                   help="name for the run directory")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    script_dir = Path(__file__).resolve().parent
    config_path = (Path(args.config) if args.config
                   else script_dir / "search_config.yaml")
    cfg = load_config(config_path)
    scfg = cfg["search"]

    if args.n_samples is not None:
        scfg["n_samples"] = args.n_samples
    if args.n_qubits is not None:
        scfg["n_qubits"] = args.n_qubits
    if args.seed is not None:
        scfg["seed"] = args.seed

    n_qubits = int(scfg["n_qubits"])
    if n_qubits < 1:
        raise SystemExit("search.n_qubits must be at least 1")
    n_samples = int(scfg["n_samples"])
    if n_samples < 1:
        raise SystemExit("search.n_samples must be at least 1")

    units = str(scfg.get("units", "frequency")).lower()
    if units not in ("frequency", "angular"):
        raise SystemExit("search.units must be 'frequency' or 'angular'")
    # Transition energies are angular (hbar = 1); the sampling check works in
    # ordinary frequency, so divide by 2*pi when asked for that.
    scale = 2 * np.pi if units == "frequency" else 1.0

    plans = {name: resolve_term(name, cfg["terms"].get(name, {}), n_qubits)
             for name in TERM_NAMES}

    output_root = Path(scfg.get("output_root", "results/hamiltonian_search"))
    if not output_root.is_absolute():
        output_root = script_dir / output_root
    run_dir = io.create_run_dir(output_root, "search",
                                args.run_name or scfg.get("run_name"))

    status = "completed"
    error_text = None
    best_payload = None
    elapsed = None
    n_evaluated = 0
    n_rejected = 0

    with io.RunLogger(run_dir) as logger:
        io.banner("LZS SPECTROSCOPY -- SCRIPT 4: RANDOM-HAMILTONIAN SEARCH",
                  f"run directory: {run_dir}")
        io.kv("Config file", config_path, width=34)

        try:
            m = 2 ** n_qubits
            n_transitions = m * (m - 1) // 2
            z_tab, zz_tab, pairs = basis_tables(n_qubits)

            io.section("Search settings")
            io.kv("Number of qubits", n_qubits)
            io.kv("Hilbert space dimension", m)
            io.kv("Transition frequencies per H", n_transitions)
            io.kv("Pairs of transitions compared", n_transitions * (n_transitions - 1) // 2)
            io.kv("Samples requested", f"{n_samples:,}")
            io.kv("Batch size", f"{int(scfg['batch_size']):,}")
            io.kv("Seed", scfg["seed"])
            io.kv("Reported units", units)
            io.kv("Top-k kept", scfg["top_k"])

            io.section("Term sampling plan")
            for name in TERM_NAMES:
                plan = plans[name]
                if plan["size"] == 0:
                    io.kv(name, "no entries at this qubit count")
                elif plan["sample"]:
                    disc = (f", snapped to {plan['discrete']}"
                            if plan["discrete"] else "")
                    io.kv(name, f"SAMPLED  {plan['size']} value(s) in "
                                f"[{plan['lo']}, {plan['hi']}]{disc}")
                else:
                    io.kv(name, f"FIXED    {io.fmt_array(plan['fixed'])}")
            if pairs:
                io.kv("two_body pair order",
                      ", ".join(f"({i},{j})" for i, j in pairs))

            max_frequency = scfg.get("max_frequency")
            if max_frequency is not None:
                io.kv("Rejecting highest transition >",
                      f"{float(max_frequency)}   [{units}]")

            # local_x all-zero (and fixed) keeps the Hamiltonian diagonal, so
            # the spectrum is a matrix product rather than a diagonalisation.
            lx_plan = plans["local_x"]
            diagonal_ok = (not lx_plan["sample"]) and np.all(lx_plan["fixed"] == 0)
            io.section("Method")
            if diagonal_ok:
                print("  local_x is fixed at zero, so the target Hamiltonian is")
                print("  diagonal in the computational basis: eigenvalues come")
                print("  from a matrix product, no diagonalisation needed.")
                sz_list = sx_list = zz_list = None
            else:
                print("  local_x is nonzero, so each Hamiltonian is built and")
                print("  diagonalised explicitly (batched numpy.linalg.eigvalsh).")
                sz_list, sx_list, zz_list, _ = pauli_operators(n_qubits)

            if n_transitions < 2:
                raise ValueError(
                    f"{n_qubits} qubit(s) give only {n_transitions} transition "
                    f"frequency; at least 2 are needed for a separation")

            triu_i, triu_j = np.triu_indices(m, 1)
            rng = np.random.default_rng(scfg["seed"])
            top = TopK(int(scfg["top_k"]))

            io.banner("SEARCHING")
            batch_size = max(1, int(scfg["batch_size"]))
            n_batches = int(np.ceil(n_samples / batch_size))
            t0 = time.perf_counter()

            for b in range(n_batches):
                size = min(batch_size, n_samples - b * batch_size)
                lz = draw(plans["local_z"], size, rng)
                lx = draw(plans["local_x"], size, rng)
                tb = draw(plans["two_body"], size, rng)

                if diagonal_ok:
                    energies = eigenvalues_diagonal(lz, lx, tb, z_tab, zz_tab)
                else:
                    energies = eigenvalues_dense(lz, lx, tb,
                                                 sz_list, sx_list, zz_list)

                gaps, max_tr = min_transition_gap(energies, triu_i, triu_j)
                n_evaluated += size

                if max_frequency is not None:
                    keep = max_tr <= float(max_frequency) * scale
                    n_rejected += int((~keep).sum())
                    gaps, max_tr = gaps[keep], max_tr[keep]
                    lz, lx, tb = lz[keep], lx[keep], tb[keep]

                top.offer_batch(gaps, max_tr, lz, lx, tb)

                done = n_evaluated / n_samples
                best_so_far = top.best["gap"] / scale if top.best else float("nan")
                bar_len = 30
                filled = int(bar_len * done)
                sys.stdout.write(
                    f"\r  [{'#' * filled}{'-' * (bar_len - filled)}] "
                    f"{100 * done:6.2f}%  {n_evaluated:,}/{n_samples:,}  "
                    f"best gap {best_so_far:.6f}")
                sys.stdout.flush()

            sys.stdout.write("\n")
            elapsed = time.perf_counter() - t0

            io.section("Search complete")
            io.kv("Hamiltonians evaluated", f"{n_evaluated:,}")
            if max_frequency is not None:
                io.kv("Rejected by max_frequency", f"{n_rejected:,}")
                io.kv("Accepted", f"{n_evaluated - n_rejected:,}")
            io.kv("Wall time", io.fmt_duration(elapsed))
            if elapsed > 0:
                io.kv("Throughput", f"{n_evaluated / elapsed:,.0f} Hamiltonians/s")

            if top.best is None:
                raise ValueError(
                    "no Hamiltonian satisfied the constraints -- loosen "
                    "search.max_frequency or widen the bounds")

            io.banner("BEST HAMILTONIAN FOUND")
            best = top.best
            io.kv("MAX closest-freq. spacing",
                  f"{best['gap'] / scale:.6f}   [{units}]")
            describe_candidate(best, 1, scale, pairs, units)

            print()
            print("  Paste into config.yaml under `common:` --")
            print()
            print(yaml_block(best), end="")

            if len(top.items) > 1:
                io.banner(f"NEXT {len(top.items) - 1} CANDIDATES")
                for rank, cand in enumerate(top.items[1:], start=2):
                    describe_candidate(cand, rank, scale, pairs, units)

            # ---------------- Save ----------------
            io.banner("SAVING RESULTS")
            npz_path = io.unique_path(run_dir / "top_candidates.npz")
            np.savez(
                npz_path,
                gap=np.array([c["gap"] for c in top.items]),
                gap_reported_units=np.array([c["gap"] / scale
                                             for c in top.items]),
                max_transition=np.array([c["max_transition"] for c in top.items]),
                local_z=np.array([c["local_z"] for c in top.items]),
                local_x=np.array([c["local_x"] for c in top.items]),
                two_body=np.array([c["two_body"] for c in top.items]),
            )
            io.kv("Top candidates", npz_path)

            best_payload = {
                "gap_reported_units": float(best["gap"] / scale),
                "gap_angular": float(best["gap"]),
                "max_transition_reported_units": float(best["max_transition"] / scale),
                "units": units,
                "local_z": best["local_z"].tolist(),
                "local_x": best["local_x"].tolist(),
                "two_body": best["two_body"].tolist(),
            }
            if units == "frequency" and best["gap"] > 0:
                needed_wait = scale / best["gap"]
                best_payload["implied_min_max_wait_time"] = float(needed_wait)
                if best["max_transition"] > 0:
                    best_payload["implied_min_n_sims"] = int(
                        np.ceil(needed_wait * 2 * best["max_transition"] / scale)) + 1

            best_path = io.write_metadata(run_dir / "best_hamiltonian.yaml",
                                          best_payload)
            io.kv("Best Hamiltonian", best_path)

        except Exception:
            status = "failed"
            error_text = traceback.format_exc()
            print()
            io.banner("SEARCH FAILED")
            print(error_text)

        metadata = {
            "script": "04_search_hamiltonians.py",
            "status": status,
            "run_directory": str(run_dir),
            "config_file": str(config_path),
            "environment": io.environment_info(),
            "search": scfg,
            "terms": cfg["terms"],
            "n_evaluated": n_evaluated,
            "n_rejected": n_rejected,
            "wall_seconds": elapsed,
            "wall_human": io.fmt_duration(elapsed) if elapsed else None,
            "best": best_payload,
        }
        if error_text:
            metadata["error"] = error_text
        meta_path = io.write_metadata(run_dir / "metadata.yaml", metadata)

        io.banner("DONE", f"status: {status}")
        io.kv("Run directory", run_dir)
        io.kv("Metadata", meta_path)
        io.kv("Log", logger.path)
        print()

    return 0 if status != "failed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
