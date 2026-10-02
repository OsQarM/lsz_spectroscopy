# Paper figure style

All figures for the paper are produced through this package, so their
appearance is set in one place rather than per plot.

## Usage

At the top of a notebook:

```python
from plotting import style
style.apply("sober")     # or "modern"
```

`style.apply()` installs the typography (Computer Modern via LaTeX when a TeX
toolchain is available, otherwise matplotlib's `cm` mathtext, which looks
nearly identical) together with the shared font sizes, palette and axis
treatment.

## Files

| file | contents |
| --- | --- |
| `paper_typography.py` | Computer Modern typography only; `apply()` probes for a working LaTeX and falls back to mathtext. |
| `style.py` | The single source of truth: active palette, font sizes, figure sizes, rcParams, plus `style_axis()`, `identity_line()` and `save()`. |
| `palettes.py` | The selectable colour schemes (`sober`, `modern`). |
| `spectrum_plots.py` | FFT spectrum (linear-y and log-y variants), mode amplitudes, global time-domain fit. |
| `diagnostic_plots.py` | Estimate-vs-true scatters, energy-difference matching, decay rates, schedules, populations, eigenvector weights, asymmetry sweep. |

## Palettes

`style.apply(name)` picks the scheme; every helper reads colours through
`style`, so switching needs no other edit.

- **`sober`** (default) — black data, red dashed references, greys. Maximum
  restraint, perfect in greyscale.
- **`modern`** — near-black base with muted Okabe–Ito accents (desaturated
  blue, vermillion, teal). Colour-blind-safe and still restrained: colour marks
  the series, never decorates. Worth it for the multi-series panels
  (populations, schedules, twin-axis sweep) where four grey dashed curves are
  hard to tell apart.

Single-series panels (spectra, scatters) look near-identical either way, so the
choice really only affects the multi-series figures.

## Conventions

- Data takes the palette's primary colour, **dashed references** (identity
  line, predicted rate) the reference colour, **greys/support** the background.
  Colour never decorates.
- Series are distinguished by **marker and dash pattern** first, so every
  figure still reads in greyscale even under `modern`.
- No grids, no panel titles in the final figures (put that in the caption);
  ticks point inward on all four spines.
- Font sizes (`FONT_LABEL` 15, `FONT_TICK` 13, `FONT_LEGEND` 13) are chosen to
  stay readable when a figure is reduced to a single column.

To retune the whole figure set, edit `style.py` alone.

## Spectrum: linear vs log

`plot_spectrum_both_scales()` returns the spectrum twice, as two separate
figures, so either can be chosen for the paper. The log variant floors its axis
`log_decades` (default 4) below the tallest peak, which keeps the peaks
dominant instead of letting the axis stretch down to the leakage sidelobes.
Mode amplitudes are a standalone figure via `plot_mode_amplitudes()`.

## Asymmetry sweep

`plot_asymmetry_vs_a()` and `plot_amplitude_vs_area()` are separate figures so
each can be placed independently. `plot_asymmetry_sweep()` calls both and
returns the two figures.

## Suppressing figures

`fourier_analysis()`, `fourier_analysis_iterative()`, `fit_decay_rates()`,
`run_full_diagnostics()` and the matching/dephasing helpers all take
`plots=True/False`. Pass `plots=False` inside sweeps instead of monkey-patching
`plt.show`.
