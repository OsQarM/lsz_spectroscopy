# LZS spectroscopy pipeline

The work that used to live in `notebooks/LSZ_main.ipynb`, split into three
scripts driven by a single `config.yaml`. The physics is unchanged: everything
still comes from the repository's `src/` package, which these scripts import
but never modify.

```
01_run_sweep.py        pre-simulation diagnostics + the wait-time sweep
02_run_diagnostics.py  full diagnostics on a stored pc_list
03_run_asymmetry.py    amplitude vs. ramp-asymmetry sweep
config.yaml            every parameter for all three scripts
lzs_pipeline/          shared infrastructure (config, I/O, pre-simulation)
```

## Quick start

```bash
# 1. Look before you leap: report, plots and a runtime estimate, no sweep.
python 01_run_sweep.py --no-run

# 2. Happy with the sampling check? Run it.
python 01_run_sweep.py --run

# 3. Analyse what it produced.
python 02_run_diagnostics.py --input-dir results/sweeps/sweep_<timestamp>

# 4. The separate asymmetry study.
python 03_run_asymmetry.py
```

Use the `spectroscopy` pyenv interpreter:
`~/.pyenv/versions/spectroscopy/bin/python`.

## Script 1 — `01_run_sweep.py`

Prints the full configuration, produces the pre-simulation figures (ramp
schedule, instantaneous spectrum, midpoint eigenvectors, eigenbasis
populations), checks that the planned wait-time grid can resolve the spectrum,
estimates the runtime, and then runs the sweep.

The runtime estimate times a few wait times spread across the grid, fits
`cost = a + b * simulated_duration`, and integrates that over the whole grid.
The probes use the same solver the sweep will (`sesolve` or `mesolve`), so the
estimate tracks the real cost — in testing it landed within 5–15% of actual.

`sweep.run_simulation: false` (or `--no-run`) stops after the estimate. This is
the intended first step: the sampling-adequacy check tells you whether
`max_wait_time` and `n_sims` can resolve your spectrum *before* you spend the
time.

Output, in a fresh directory per run:

```
results/sweeps/sweep_<timestamp>/
    pc_list.npy                 the swept signal P(psi0) vs wait time
    tw_l.npy                    the wait-time grid
    pc_all_realizations.npy     only when averaging is enabled
    metadata.yaml               every parameter used, plus provenance
    output.txt                  a transcript of everything printed
    plots/                      the pre-simulation figures
```

## Script 2 — `02_run_diagnostics.py`

Takes a script-1 run directory and runs the full chain: FFT peak extraction,
global amplitude/decay fit, matching against the true energy differences,
turnpike spectrum reconstruction, `u_m` / `phi_m` recovery, exact-state
validation, optional dephasing comparison, and the optional joint refinement.

**Where the parameters come from.** The physics is read from the run's
`metadata.yaml`, so the analysis always matches the data. Anything the metadata
does not carry falls back to the `common` block of `config.yaml`. The
diagnostics-only settings (`fourier`, `refinement`) always come from the
config. Each section of the printed report is tagged `[from metadata]` or
`[from config]` so the provenance is visible, and the same mapping is recorded
in `metadata_diagnostics.yaml`.

Results are written **into the source run directory**: figures into `plots/`,
data and metadata at the top level.

```
results/sweeps/sweep_<timestamp>/
    diagnostics_results.npz     freqs, amplitudes, energies, u_m, phi_m, ...
    refinement_results.npz      the refined parameters
    metadata_diagnostics.yaml   the diagnostics settings and provenance
    output_diagnostics.txt      a transcript
    plots/diagnostics_*.png     figures from run_full_diagnostics
    plots/refinement_*.png      figures from the refinement stage
```

## Script 3 — `03_run_asymmetry.py`

For each asymmetry value `a`, the target qubit's ramp coefficients are set to
`a` while the others stay at 1; the full wait-time sweep runs, a light FFT +
global fit extracts the mode amplitudes, and the sum of the `n_smallest`
amplitudes is recorded with the Z first-ramp area. It prints the same kind of
pre-simulation report and runtime estimate as script 1 (scaled by the number of
asymmetry values), and tabulates the ramp areas up front — they are cheap and
independent of the sweep.

This is a separate study, so it writes to its own tree:

```
results/asymmetry/asymmetry_<timestamp>/
    asymmetry_sweep.npz   a_values, amp_sum_smallest, ramp1_z/x_areas, all_amplitudes
    pc_lists.npy          the raw signal for every asymmetry value
    tw_l.npy              the wait-time grid
    metadata.yaml         every parameter used
    output.txt            a transcript
    plots/                pre-simulation + asymmetry_vs_a + amplitude_vs_area
```

## Nothing is ever overwritten

- New run directories are named with a timestamp; if that still collides, a
  numeric suffix is appended.
- Every file written into an existing directory (script 2 writing into a
  script-1 run) goes through `unique_path`, which appends `_2`, `_3`, … So
  re-running script 2 on the same directory leaves the first results intact and
  writes `diagnostics_results_2.npz`, `output_diagnostics_2.txt`, and so on.
- All directories are created on demand.

## Configuration

`config.yaml` is commented throughout. Its shape:

| Block | Used by | Holds |
|---|---|---|
| `common` | all three | protocol, Hamiltonian, decoherence, ramps, initial state, seed, plotting |
| `sweep` | script 1 | `run_simulation`, output root, averaging, runtime estimate |
| `diagnostics` | script 2 | input directory, Fourier settings, refinement settings |
| `asymmetry` | script 3 | the asymmetry grid, target qubit, `n_smallest`, Fourier settings |

The number of qubits is inferred from the length of `hamiltonian.local_z`;
`local_x` must match it and `two_body` needs `n(n-1)/2` entries. Mismatches are
caught with a clear message before any run directory is created.

Initial states: the presets `minus`, `plus`, `zero`, `one` apply one
single-qubit state to every qubit. For anything else use
`preset: custom` with either `per_qubit: [minus, plus, ...]` or a full list of
`2**n_qubits` `amplitudes` (normalised for you) — that is how you get a Bell or
GHZ state.

Set `plotting.formats: [png, pdf]` to also write vector figures for the paper.

## Notes

- Figures are saved, never shown: the scripts force matplotlib's `Agg` backend.
  The figures `run_full_diagnostics` draws internally (it calls `plt.show()`
  itself) are captured by sweeping up every figure created during the call.
- Everything printed goes to both the terminal and the run's log file.
  Progress bars are collapsed to one line per refresh in the log.
- A failing run still writes its metadata, with `status: failed` and the
  traceback, so a crashed run is still self-describing.
- `results/` is gitignored.
