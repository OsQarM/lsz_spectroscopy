"""
Configuration loading for the three LZS pipeline scripts.

A single `config.yaml` drives all three scripts. It is organised into a shared
block plus one block per script:

    common:      physics + protocol parameters shared by every script
    sweep:       script 1 (pre-simulation diagnostics + wait-time sweep)
    diagnostics: script 2 (full diagnostics on a stored pc_list)
    asymmetry:   script 3 (amplitude vs. ramp-asymmetry sweep)

Script 2 may be pointed at a run directory produced by script 1. In that case
the physics parameters are read from that run's `metadata.yaml`, and the
`common` block of the config is used only as a fallback for anything the
metadata does not carry (see `resolve_common`).
"""
from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import qutip as qt
import yaml


# --------------------------------------------------------------------------
# Defaults. Every key the scripts read has a default here, so a user's
# config.yaml only needs to carry what it wants to change.
# --------------------------------------------------------------------------
DEFAULTS = {
    "common": {
        "protocol": {
            "epsilon": 1.0,
            "max_wait_time": 315.0,
            "ramp_time": 1.0,
            "dt": 1.0,
            "n_sims": 1400,
            "n_steps": 201,
        },
        "hamiltonian": {
            "local_z": [2.1, 3.05, 1.78],
            "local_x": [0.0, 0.0, 0.0],
            "two_body": [1.22, 0.54, 0.3],
        },
        "decoherence": {
            "t1_rates": [0.005, 0.0051, 0.0048],
            "t2_rates": [0.01, 0.012, 0.0098],
            "r_noise": False,
            "w_noise": False,
        },
        "ramps": {
            "up_z_assymetry_coefficients": None,
            "down_z_assymetry_coefficients": None,
            "up_x_assymetry_coefficients": None,
            "down_x_assymetry_coefficients": None,
            "average_ramp_error": 0.0,
            "average_wait_error": 0.0,
            "average_ramp_bias": 0.0,
        },
        "initial_state": {"preset": "minus"},
        "run": {"noise_seed": 0},
        "plotting": {"palette": "modern", "formats": ["png"]},
    },
    "sweep": {
        "run_simulation": True,
        "output_root": "results/sweeps",
        "run_name": None,
        "averaging": {"enabled": False, "n_averages": 10},
        "runtime_estimate": {"enabled": True, "n_probe": 3},
    },
    "diagnostics": {
        "input_dir": None,
        "fourier": {
            "n_peaks": None,
            "prominence_threshold": 0.001,
            "zero_pad_factor": 32,
            "window": "hann",
            "detect_prominence": 0.001,
            "f_min": 0.001,
        },
        "refinement": {
            "enabled": True,
            "reg": None,
            "max_nfev": 500,
            "verbose": 1,
            "freeze_E": True,
            "population_mode": "subsets",
            "x_scale": "auto",
            "loss": "linear",
            "f_scale": 1.0,
            "anchor_kappa_weight": 0.0,
            "keep_warm_if_worse": True,
        },
    },
    "asymmetry": {
        "run_simulation": True,
        "output_root": "results/asymmetry",
        "run_name": None,
        "n_asym": 15,
        "a_min": 1.0,
        "a_max": 10.0,
        "spacing": "geometric",
        "n_smallest": 3,
        "target_qubit": 0,
        "sweep_z": True,
        "sweep_x": False,
        "runtime_estimate": {"enabled": True, "n_probe": 3},
        "fourier": {
            "n_peaks": None,
            "prominence_threshold": 0.001,
            "zero_pad_factor": 32,
            "window": "hann",
            "detect_prominence": 0.001,
            "f_min": 0.1,
        },
    },
}


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge `override` into a copy of `base`.

    Mappings are merged key by key; every other value (including lists and
    explicit `None`s) replaces the base value outright.
    """
    out = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def load_config(path) -> dict:
    """Load `config.yaml` and merge it onto `DEFAULTS`."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"config file not found: {path}")
    with open(path) as fh:
        user = yaml.safe_load(fh) or {}
    if not isinstance(user, dict):
        raise ValueError(f"{path}: top level of the config must be a mapping")
    return _deep_merge(DEFAULTS, user)


# --------------------------------------------------------------------------
# Initial state
# --------------------------------------------------------------------------
KET0 = qt.basis(2, 0)
KET1 = qt.basis(2, 1)

_MINUS = (KET0 - KET1).unit()
_PLUS = (KET0 + KET1).unit()


def build_initial_state(spec, n_qubits: int) -> qt.Qobj:
    """Build psi0 from an `initial_state` config block.

    Accepted forms:
        {'preset': 'minus'}            -> |--...-> on every qubit
        {'preset': 'plus'}             -> |++...+>
        {'preset': 'zero'}             -> |00...0>
        {'preset': 'one'}              -> |11...1>
        {'preset': 'custom',
         'per_qubit': ['minus','plus', ...]}   -> one label per qubit
        {'preset': 'custom',
         'amplitudes': [...]}          -> 2**n_qubits complex amplitudes
    """
    if spec is None:
        spec = {"preset": "minus"}
    if isinstance(spec, str):
        spec = {"preset": spec}

    preset = str(spec.get("preset", "minus")).lower()
    single = {"minus": _MINUS, "plus": _PLUS, "zero": KET0, "one": KET1}

    if preset in single:
        return qt.tensor([single[preset]] * n_qubits)

    if preset == "custom":
        per_qubit = spec.get("per_qubit")
        if per_qubit is not None:
            if len(per_qubit) != n_qubits:
                raise ValueError(
                    f"initial_state.per_qubit has {len(per_qubit)} entries "
                    f"but there are {n_qubits} qubits"
                )
            kets = []
            for label in per_qubit:
                key = str(label).lower()
                if key not in single:
                    raise ValueError(f"unknown per-qubit state label: {label!r}")
                kets.append(single[key])
            return qt.tensor(kets)

        amplitudes = spec.get("amplitudes")
        if amplitudes is not None:
            vec = np.asarray([complex(a) for a in amplitudes], dtype=complex)
            if vec.size != 2 ** n_qubits:
                raise ValueError(
                    f"initial_state.amplitudes has {vec.size} entries but "
                    f"{2 ** n_qubits} are needed for {n_qubits} qubits"
                )
            norm = np.linalg.norm(vec)
            if norm == 0:
                raise ValueError("initial_state.amplitudes is the zero vector")
            dims = [[2] * n_qubits, [1] * n_qubits]
            return qt.Qobj((vec / norm).reshape(-1, 1), dims=dims)

        raise ValueError(
            "initial_state.preset is 'custom' but neither 'per_qubit' nor "
            "'amplitudes' was given"
        )

    raise ValueError(f"unknown initial-state preset: {preset!r}")


# --------------------------------------------------------------------------
# Derived physics quantities
# --------------------------------------------------------------------------
def _as_list(value, n, name, default_fill=1.0):
    """Normalise a per-qubit coefficient list, defaulting to all-`default_fill`."""
    if value is None:
        return [float(default_fill)] * n
    arr = [float(v) for v in value]
    if len(arr) != n:
        raise ValueError(f"{name} has {len(arr)} entries but there are {n} qubits")
    return arr


def resolve_common(cfg_common: dict, metadata: dict | None = None) -> dict:
    """Turn a `common` config block into the concrete objects the scripts use.

    When `metadata` (a run's `metadata.yaml`, as written by script 1) is
    supplied, its `common` block takes precedence and the config's `common`
    block fills in only what the metadata lacks. That is what lets script 2
    reproduce a run's physics exactly while still picking up new
    diagnostics-only settings from the config file.

    Returns a dict with ready-to-use values: numpy arrays for the Hamiltonian
    terms, a qutip ket for `psi0`, the wait-time grid `tw_l`, and the fully
    expanded per-qubit asymmetry lists.
    """
    merged = cfg_common
    provenance = {}
    if metadata:
        meta_common = metadata.get("common", {})
        merged = _deep_merge(cfg_common, meta_common)
        # Record, per top-level section, where the values came from.
        for section in cfg_common:
            provenance[section] = "metadata" if section in meta_common else "config"
    else:
        for section in cfg_common:
            provenance[section] = "config"

    proto = merged["protocol"]
    ham = merged["hamiltonian"]
    dec = merged["decoherence"]
    ramps = merged["ramps"]

    local_z = np.asarray(ham["local_z"], dtype=float)
    local_x = np.asarray(ham["local_x"], dtype=float)
    two_body = np.asarray(ham["two_body"], dtype=float)
    n_qubits = int(local_z.size)

    if local_x.size != n_qubits:
        raise ValueError(
            f"hamiltonian.local_x has {local_x.size} entries but local_z "
            f"implies {n_qubits} qubits"
        )
    n_pairs = n_qubits * (n_qubits - 1) // 2
    if two_body.size != n_pairs:
        raise ValueError(
            f"hamiltonian.two_body has {two_body.size} entries but "
            f"{n_qubits} qubits need {n_pairs}"
        )

    t1_rates = dec.get("t1_rates")
    t2_rates = dec.get("t2_rates")
    if t1_rates is not None:
        t1_rates = _as_list(t1_rates, n_qubits, "decoherence.t1_rates")
    if t2_rates is not None:
        t2_rates = _as_list(t2_rates, n_qubits, "decoherence.t2_rates")

    max_wait_time = float(proto["max_wait_time"])
    n_sims = int(proto["n_sims"])
    if n_sims < 2:
        raise ValueError("protocol.n_sims must be at least 2")

    psi0 = build_initial_state(merged.get("initial_state"), n_qubits)

    return {
        "n_qubits": n_qubits,
        "epsilon": float(proto["epsilon"]),
        "max_wait_time": max_wait_time,
        "ramp_time": float(proto["ramp_time"]),
        "dt": float(proto["dt"]),
        "n_sims": n_sims,
        "n_steps": int(proto["n_steps"]),
        "tw_l": np.linspace(0.0, max_wait_time, n_sims),
        "local_z": local_z,
        "local_x": local_x,
        "two_body": two_body,
        "H_target_dict": {
            "local_z": local_z,
            "local_x": local_x,
            "two_body": two_body,
        },
        "t1_rates": t1_rates,
        "t2_rates": t2_rates,
        "r_noise": bool(dec.get("r_noise", False)),
        "w_noise": bool(dec.get("w_noise", False)),
        "up_z": _as_list(ramps.get("up_z_assymetry_coefficients"), n_qubits,
                         "ramps.up_z_assymetry_coefficients"),
        "down_z": _as_list(ramps.get("down_z_assymetry_coefficients"), n_qubits,
                           "ramps.down_z_assymetry_coefficients"),
        "up_x": _as_list(ramps.get("up_x_assymetry_coefficients"), n_qubits,
                         "ramps.up_x_assymetry_coefficients"),
        "down_x": _as_list(ramps.get("down_x_assymetry_coefficients"), n_qubits,
                           "ramps.down_x_assymetry_coefficients"),
        "ramp_error": float(ramps.get("average_ramp_error", 0.0)),
        "wait_error": float(ramps.get("average_wait_error", 0.0)),
        "ramp_bias": float(ramps.get("average_ramp_bias", 0.0)),
        "initial_state_spec": merged.get("initial_state"),
        "psi0": psi0,
        "noise_seed": merged["run"]["noise_seed"],
        "palette": merged["plotting"]["palette"],
        "figure_formats": list(merged["plotting"]["formats"]),
        "_merged": merged,
        "_provenance": provenance,
    }


def common_to_metadata(common: dict) -> dict:
    """The serialisable `common` block to store in a run's `metadata.yaml`.

    This is what makes a run self-describing: script 2 reads it back through
    `resolve_common(..., metadata=...)` and reproduces the exact physics.
    """
    return {
        "protocol": {
            "epsilon": common["epsilon"],
            "max_wait_time": common["max_wait_time"],
            "ramp_time": common["ramp_time"],
            "dt": common["dt"],
            "n_sims": common["n_sims"],
            "n_steps": common["n_steps"],
        },
        "hamiltonian": {
            "local_z": [float(v) for v in common["local_z"]],
            "local_x": [float(v) for v in common["local_x"]],
            "two_body": [float(v) for v in common["two_body"]],
        },
        "decoherence": {
            "t1_rates": common["t1_rates"],
            "t2_rates": common["t2_rates"],
            "r_noise": common["r_noise"],
            "w_noise": common["w_noise"],
        },
        "ramps": {
            "up_z_assymetry_coefficients": common["up_z"],
            "down_z_assymetry_coefficients": common["down_z"],
            "up_x_assymetry_coefficients": common["up_x"],
            "down_x_assymetry_coefficients": common["down_x"],
            "average_ramp_error": common["ramp_error"],
            "average_wait_error": common["wait_error"],
            "average_ramp_bias": common["ramp_bias"],
        },
        "initial_state": common["initial_state_spec"],
        "run": {"noise_seed": common["noise_seed"]},
        "plotting": {
            "palette": common["palette"],
            "formats": common["figure_formats"],
        },
    }
