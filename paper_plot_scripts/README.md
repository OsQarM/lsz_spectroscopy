# Paper figure scripts

Final-quality figures for the paper, built from the sweeps that
`refactored_code/` produces. Each script reads a YAML config, so the same
data can be re-drawn with different labels, limits and styling without
touching any code.

```
plot_fft.py            FFT of the swept signal (one or several runs)
fft_config.yaml        its configuration
plot_schedule.py       the ramp schedule s(t) of an example run
schedule_config.yaml   its configuration
plot_diagnostics.py    estimate-vs-true diagnostic panels
diagnostics_config.yaml  its configuration
paper_plots/           shared helpers (loading, FFT, schedule, styling, saving)
paper_plot_images/     output: figures + the data behind them
```

Figure appearance comes from `src/plotting/style.py`, the same source the
pipeline's own figures use, so paper figures and diagnostic figures stay
consistent. Run with the `spectroscopy` pyenv interpreter.

## `plot_fft.py`

```bash
python plot_fft.py                                  # uses fft_config.yaml
python plot_fft.py --config my_figure.yaml
python plot_fft.py --source ../../new_results/sweeps/sweep_X --name fig3
python plot_fft.py --no-peaks --no-legend           # quick variants
python plot_fft.py --show                           # also open a window
```

The spectrum is computed exactly as `src/fourier.py` does — mean removed,
windowed, zero-padded, DC bin dropped — so the figure shows the same
transform the peak detection ran on.

### Inputs

Each entry under `sources:` is one curve. It can be a run directory from
script 1 (`pc_list.npy` + `tw_l.npy` + `metadata.yaml`) or a bare `.npy`
holding the signal. For a bare file, give `tw_file:` or `max_wait_time:` so
the frequency axis can be built.

```yaml
sources:
  - path: ../../new_results/sweeps/sweep_20261002-192058
    label: 4 qubits
  - path: ../../new_results/sweeps/sweep_20261002-174736
    label: shorter sweep
    linestyle: "--"
```

Per-source `label`, `color`, `linestyle`, `linewidth` override the globals.
Labels default to the directory name.

### What is configurable

| Block | Controls |
|---|---|
| `fft` | window, zero-padding, and `normalise` (`max`/`area`/`coherent`) so runs of different length share a panel |
| `peaks` | **whether the dashed markers are drawn at all**, where they come from, colour, style, width, alpha, per-series colouring, a legend entry, and optional frequency annotations |
| `axes` | x/y labels, title, limits, `linear`/`log` y, decades shown, grid, auto-framing on the peaks |
| `legend` | **on/off**, location, frame, font size, columns, title |
| `style` | palette, explicit colours, line widths and dash patterns, font sizes, arbitrary rcParams |
| `layout` | `overlay`, `offset` (curves shifted apart) or `stacked` (one panel each) |
| `figure` | a named preset or an explicit size in inches |
| `output` | directory, name, timestamping, formats, dpi, transparency, data export |

### Peak markers

`peaks.show: false` removes them entirely. When shown, `peaks.source` picks
what to mark:

- `auto` *(default)* — the peaks script 2 detected, falling back to
  re-detecting them if that run has no diagnostics;
- `detected` — only the stored peaks, erroring if absent;
- `true_differences` — the exact transition frequencies from the
  Hamiltonian (converted from angular units for you);
- `detect` — always re-detect from the spectrum being drawn.

Prefer the stored peaks: script 2 refines each to sub-bin precision.
Re-detected positions agree to well under one bin (~1e-4 in testing), so
they look identical, but they are not the exact values the analysis reported.
Re-detection with no `n_peaks` cap and no `f_min` keeps every prominent
maximum, so it can mark more peaks than the analysis did.

### Output

Everything lands in `paper_plot_images/`:

```
fft_spectrum.pdf        the figure (plus .png, or whatever formats you list)
fft_spectrum_data.npz   every curve drawn: freqs_i, magnitude_i, peaks_i, labels
fft_spectrum_data.yaml  what each series is, where it came from, its
                        Hamiltonian, and the settings used
```

Set `output.save_csv: true` to also get one plain `frequency,magnitude` CSV
per curve. The `.npz` holds exactly what was plotted, so a figure can be
rebuilt or restyled later without re-reading the sweeps.

Nothing is ever overwritten: re-running appends `_2`, `_3`, … to the names.
Set `output.timestamp: true` to name files by run time instead.

## `plot_schedule.py`

```bash
python plot_schedule.py                                   # schedule_config.yaml
python plot_schedule.py --ramp-time 1 --wait-time 3
python plot_schedule.py --from-run ../../new_results/sweeps/3_qubits_good_1ns
python plot_schedule.py --no-markers --no-legend
```

Draws the scheduling functions s(t) over the ramp-wait-ramp cycle. The
curves are evaluated with the real `LSZ_experiment`, so ramp asymmetries and
schedule noise appear exactly as the simulations apply them.

### Which lines are drawn

Four families, each switched on or off independently under `lines:`

| key | what it is |
|---|---|
| `common` | the `(H0 + Hzz)` envelope `s(t)` |
| `h0` | the complementary weight `1 - s(t)` |
| `z` | the per-qubit local-Z schedules |
| `x` | the per-qubit local-X schedules |

So a figure showing only `H_z` is `common`, `h0` and `x` set to
`show: false`. Within `z` and `x`, `qubits:` picks which qubits to draw
(`null` for all, or e.g. `[0]`). Labels take `{i}` for the qubit index.
Each family has its own `color`/`colors`, `linestyle`, `linewidth`, `alpha`.

### Protocol

Set `ramp_time` and `wait_time` directly under `protocol:`, along with the
qubit count and the per-qubit asymmetries. A figure usually wants a *short*
wait time — it only has to show that a plateau exists, and a realistic
300-unit wait squashes the ramps into two vertical lines.

`from_run:` (or `--from-run`) borrows the qubit count, ramp time,
asymmetries and errors from an existing run's `metadata.yaml`. Anything set
explicitly in the config still wins, so you can take a run's asymmetries but
choose your own wait time.

### Segment markers

`markers:` draws a vertical line at the end of the first ramp, the end of
the wait and the end of the second ramp — each toggled separately, plus an
optional one at `t = 0`. Their names go on the x axis in place of the
numeric ticks (`replace_xticks: true`), and the names themselves are
configurable:

```yaml
markers:
  end_ramp1: true
  end_wait: true
  end_ramp2: true
  labels:
    end_ramp1: $t_r$
    end_wait: $t_r + t_w$
    end_ramp2: $t_f$
```

Also available: `shade_wait` for a light band over the wait window, and
`annotate_segments` to name the ramp/wait spans across the top.

With `wait_time: 0` the end of ramp 1 and the end of the wait coincide; the
duplicate marker is dropped automatically.

### Output

Same convention as `plot_fft.py` — figure plus an `.npz` and a `.yaml`
describing it, nothing overwritten. In the schedule's archive the x values
(time) are stored as `freqs_i` and the y values (`s(t)`) as `magnitude_i`,
so the file has the same shape as the other figures; `peaks_i` holds the
segment-boundary times. The `.yaml` records which lines were drawn, the full
protocol, and the marker positions and labels.

## `plot_diagnostics.py`

```bash
python plot_diagnostics.py                                  # diagnostics_config.yaml
python plot_diagnostics.py --panels energies,phases
python plot_diagnostics.py --source ../../new_results/sweeps/3_qubits_T2
python plot_diagnostics.py --separate                       # one file per panel
```

Five estimate-vs-true panels, each a scatter against the red dashed `y = x`
reference, read straight from the `diagnostics_results.npz` that script 2
wrote — nothing is recomputed.

| panel | shows |
|---|---|
| `transitions` | detected vs. true transition energies |
| `energies` | reconstructed vs. true energy levels |
| `amplitudes` | estimated vs. exact `|u_m|` |
| `phases` | estimated vs. exact `phi_m` |
| `rates` | fitted vs. predicted coherence decay rates `Gamma_mn` |
| `rate_levels` | fitted rates as markers, one dashed line per predicted rate |

`rates` and `rate_levels` need a run analysed with noise enabled — a
noiseless run stores no predicted rates, and the script says so instead of
drawing an empty panel.

### The `rate_levels` panel

This is the pipeline's own decay-rate figure (`plot_decay_rates`) as a paper
figure: fitted rates plotted against mode index, with a red dashed horizontal
line at every *distinct* predicted rate. `sort_by: rate` is the default and
worth keeping — it turns the 28 fitted rates into visible plateaus sitting on
the 7 predicted levels.

```yaml
panel_options:
  rate_levels:
    predicted: exact     # exact | possible | none
    x: index             # index | frequency
    sort_by: rate
    level_label: predicted
    marker_label: fitted
```

**On `predicted`:** the pipeline's figure draws `possible_lambdas`, the
dephasing-only subset sums, which omit the T1 contribution — on a run with
amplitude damping those lines sit well below the markers (RMS 2.7e-3 vs
1.4e-4 on the 3-qubit T2 run). The default here is `exact`, the per-pair
`Gamma_mn` from the full T1 + T2 formula, which is what the fitted rates
actually land on. Use `possible` only to reproduce the pipeline's version.

### Both turnpike solutions

A turnpike reconstruction is determined only up to reflection: if `E` solves
the difference set then so does `max(E) - E[::-1]`. The diagnostics keep only
the branch that fit better, but the other is recovered exactly from it (the
reflection is an involution), so the `energies` panel draws **both**:

```yaml
panel_options:
  energies:
    branches: both         # both | best | forward | reversed
    branch_labels: {forward: forward, reversed: reversed}
    mark_best: true        # append "(best)" to the better-fitting branch
```

The better branch takes the primary colour, the other a secondary one. They
necessarily coincide at the two endpoints — those are the reflection's fixed
points — which is what makes the degeneracy and its resolution visible.

### Axes and legend

`common_panel:` sets labels, limits, `equal_limits`, markers, colours, the
identity line and an optional in-panel RMS annotation for every panel;
`panel_options.<panel>:` overrides any of those for one panel. `legend:` takes
the usual on/off, location, frame, font size, columns and title.

One subtlety: estimated and true values come out of the analysis in different
orders, so panels compare them **sorted** by default. `transitions` is the
exception — those were already matched one-to-one — so its config sets
`sort: false` to keep the real pairing.

### Layout

`layout.ncols` arranges the panels in a grid (null = one row),
`layout.separate: true` writes one file per panel instead, and
`panel_letters: true` adds (a), (b), (c)…

### Output

Same convention as the other figures. In the archive, `freqs_i` holds the
true/reference values and `magnitude_i` the estimated ones; the `.yaml`
records each series' panel, RMS and, for `energies`, which turnpike branch
it is.

## Notes

- The figure is written with a non-interactive backend unless `--show` is
  passed, so the scripts run fine over SSH or in a batch job.
- If LaTeX is not installed, the style falls back to matplotlib's Computer
  Modern mathtext and prints a note. Figures still render correctly.
- Math in labels needs quoting in YAML when written in flow style, e.g.
  `ylabel: "$|\\mathrm{FFT}|$"`.
